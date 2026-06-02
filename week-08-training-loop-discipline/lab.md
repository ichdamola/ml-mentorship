# Week 08: Lab — Training-Loop Discipline

You'll turn your week-07 MLP training script into a senior-engineering one: overfit-one-batch sanity, LR finder, scheduler, gradient clipping, TensorBoard logging, reproducibility, best-val checkpointing.

By the end, you'll have a training script you'd actually run on a real project.

## Setup

```bash
uv add tensorboard
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import random
import os
```

---

## Exercise 8.1 — Reproducibility scaffold

Wrap the seed-dance in a function and **call it first thing every time**.

```python
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}")
```

---

## Exercise 8.2 — Data

Same MNIST setup as week 07 — but now we'll carve out a proper validation split.

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

full_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

# Carve out 5000 val examples from train
g = torch.Generator().manual_seed(42)
train_set, val_set = torch.utils.data.random_split(full_train, [55000, 5000], generator=g)
print(f"train {len(train_set)}, val {len(val_set)}, test {len(test_set)}")

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=512, shuffle=False)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False)
```

**Notice:** test is for the *final* report only. Train and validate while developing. Touching test in a tuning loop is data leakage (week 04).

---

## Exercise 8.3 — The model and a generic train/eval

```python
class MLP(nn.Module):
    def __init__(self, hidden=128, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc2(self.drop(F.relu(self.fc1(x))))

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0; correct = 0; total = 0
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    with torch.inference_mode():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += loss_fn(logits, y).item()
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total
```

---

## Exercise 8.4 — Overfit one batch

The single most important sanity check.

```python
set_seed(42)
model = MLP().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Take ONE batch
X, y = next(iter(train_loader))
X, y = X.to(device), y.to(device)

losses = []
for step in range(300):
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model(X), y)
    loss.backward()
    opt.step()
    losses.append(loss.item())

print(f"start loss:  {losses[0]:.6f}")
print(f"end loss:    {losses[-1]:.6f}")

plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel("step"); plt.ylabel("loss"); plt.yscale("log")
plt.title("Overfit-one-batch sanity check")
plt.grid(alpha=0.3); plt.show()
```

You should see `end loss` approach `1e-3` or smaller. If your model **can't** drive a single batch's loss to zero, something is broken structurally. Stop and debug.

---

## Exercise 8.5 — LR finder

Smith's LR finder. Train one epoch with exponentially-rising LR, plot loss vs LR.

```python
def lr_finder(model, loader, opt, loss_fn, min_lr=1e-7, max_lr=10.0, n_steps=100):
    lrs, losses = [], []
    it = iter(loader)
    for i in range(n_steps):
        try:
            X, y = next(it)
        except StopIteration:
            it = iter(loader)
            X, y = next(it)
        X, y = X.to(device), y.to(device)
        lr = min_lr * (max_lr / min_lr) ** (i / (n_steps - 1))
        for g in opt.param_groups:
            g['lr'] = lr

        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()

        lrs.append(lr); losses.append(loss.item())
        if loss.item() > losses[0] * 4:    # diverged
            break
    return lrs, losses

set_seed(42)
model = MLP().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-7)
lrs, losses = lr_finder(model, train_loader, opt, loss_fn)

plt.figure(figsize=(10, 4))
plt.semilogx(lrs, losses)
plt.xlabel("learning rate (log)"); plt.ylabel("loss")
plt.title("LR Finder")
plt.grid(alpha=0.3); plt.show()
```

**Reading the plot:** the steepest negative slope of the loss-vs-LR curve is the LR where the model is "learning the most per step." Pick **~10× lower** than the minimum-loss LR to leave headroom. For MNIST with this model you'll typically see a sweet spot around `1e-3` to `3e-3` for Adam.

---

## Exercise 8.6 — Full training with discipline

Now the senior loop. Includes:
- Cosine LR schedule with warmup
- Gradient clipping
- Best-val checkpointing
- Early stopping
- TensorBoard logging

```python
def train_disciplined(
    lr=1e-3, n_epochs=15, weight_decay=0.0,
    grad_clip=1.0, dropout=0.2, warmup_steps=200,
    patience=3, exp_name="exp_001",
):
    set_seed(42)
    model = MLP(dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # Cosine schedule with linear warmup
    total_steps = n_epochs * len(train_loader)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    writer = SummaryWriter(f"runs/{exp_name}")
    best_val = float('inf'); bad_epochs = 0
    step = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0; n_batches = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()

            # Gradient clipping — print the actual norm so you can see if it's biting
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            scheduler.step()

            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", opt.param_groups[0]['lr'], step)
            writer.add_scalar("train/grad_norm", grad_norm.item(), step)

            epoch_loss += loss.item(); n_batches += 1; step += 1

        train_loss = epoch_loss / n_batches
        val_loss, val_acc = evaluate(model, val_loader, device)
        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)
        writer.add_scalar("epoch/val_acc", val_acc, epoch)
        print(f"epoch {epoch:2d}  train {train_loss:.4f}  val {val_loss:.4f}  val_acc {val_acc:.4f}")

        # Best-val checkpoint
        if val_loss < best_val:
            best_val = val_loss; bad_epochs = 0
            torch.save({
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_val': best_val,
            }, f"runs/{exp_name}/best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stop at epoch {epoch}")
                break

    writer.close()
    # Reload best and evaluate on test
    ckpt = torch.load(f"runs/{exp_name}/best.pt")
    model.load_state_dict(ckpt['model'])
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\n=== final (best-val ckpt) ===")
    print(f"test loss {test_loss:.4f}  test acc {test_acc:.4f}")
    return test_acc

test_acc = train_disciplined()
```

Then run TensorBoard:

```bash
tensorboard --logdir runs/
```

Open the URL it prints. You should see:
- `train/loss` falling
- `train/lr` rising then cosine-decaying
- `train/grad_norm` mostly below your clip threshold (occasional spikes get clipped)
- `epoch/val_acc` rising
- Early stopping kicking in if val plateaus

---

## Exercise 8.7 — Diagnose three "bad" runs

For each of the configurations below, predict what will go wrong, then run it and check.

```python
# (a) LR way too high
print("=== bad run (a): lr too high ===")
train_disciplined(lr=1.0, n_epochs=3, exp_name="bad_high_lr")

# (b) No dropout, no weight decay, train forever — should overfit
print("\n=== bad run (b): expected to overfit ===")
train_disciplined(lr=1e-3, n_epochs=30, weight_decay=0.0, dropout=0.0,
                  patience=100, exp_name="bad_overfit")

# (c) LR way too low
print("\n=== bad run (c): lr too low ===")
train_disciplined(lr=1e-6, n_epochs=5, exp_name="bad_low_lr")
```

In TensorBoard, compare all four runs side by side. You'll see:
- (a) Loss explodes / NaNs / grad_norm spikes
- (b) Train loss continues falling but val loss bottoms out then rises
- (c) Both losses fall painfully slowly; never reach high accuracy

**This is how you'd diagnose a real failed run.** The shapes of the curves tell you what to fix.

---

## Exercise 8.8 — Repro check

Run the same disciplined training twice with the same seed. They should agree to many decimals.

```python
acc_a = train_disciplined(exp_name="repro_a", n_epochs=3)
acc_b = train_disciplined(exp_name="repro_b", n_epochs=3)
print(f"\nrun A: {acc_a:.4f}\nrun B: {acc_b:.4f}\nagreement: {abs(acc_a - acc_b) < 0.0005}")
```

If they don't match, something in your pipeline isn't seeded. Common culprits:
- `num_workers > 0` without `worker_init_fn` (DataLoader workers have separate RNG)
- Forgot `torch.cuda.manual_seed_all`
- Using a CUDA op that's nondeterministic (atomic-add scatter, etc.)

---

## Exercise 8.9 — Resume from checkpoint

Test that you can stop and restart training without losing progress:

```python
# Suppose 'runs/exp_001/best.pt' exists from earlier
ckpt = torch.load("runs/exp_001/best.pt")

model = MLP(dropout=0.2).to(device)
model.load_state_dict(ckpt['model'])

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
opt.load_state_dict(ckpt['opt'])

print(f"resumed from epoch {ckpt['epoch']}, best val {ckpt['best_val']:.4f}")

# Continue training for 3 more epochs (skipping setup)
# ... your train loop here ...
```

The optimizer's `state_dict` includes Adam's `m_t` and `v_t` momentum estimates. **Resuming without loading the optimizer means losing those — your first few resumed steps will be wildly off.** Always save and load both.

---

## Exercise 8.10 (stretch) — W&B integration

If you have a W&B account (free), replace TensorBoard with `wandb`:

```python
import wandb
wandb.init(project="ml-mentorship-week08", name="disciplined_v1", config={
    "lr": 1e-3, "batch_size": 64, "weight_decay": 0.01, "dropout": 0.2,
})

# In the training loop:
wandb.log({
    "train/loss": loss.item(),
    "train/lr": opt.param_groups[0]['lr'],
    "train/grad_norm": grad_norm.item(),
}, step=step)

# At the end:
wandb.finish()
```

W&B's UI is significantly nicer for comparing runs. You can also use it to save model checkpoints (`wandb.save("runs/exp_001/best.pt")`) and tag runs with notes.

---

## Submission checklist

- [ ] `set_seed` works; two runs with same seed agree to 4+ decimals on val accuracy
- [ ] Overfit-one-batch sanity check brings a single batch's loss to < 1e-3
- [ ] LR finder shows a clear sweet spot; you pick an LR from the plot
- [ ] Full disciplined training script with cosine+warmup, grad clip, best-val ckpt, early stopping
- [ ] TensorBoard (or W&B) shows train/loss, val/loss, LR, grad_norm, val_acc
- [ ] Three "bad runs" visualized and diagnosed
- [ ] Checkpoint resume works — optimizer state included
- [ ] Test accuracy reported only from the best-val checkpoint, never from "last epoch"

---

## What you just did

You upgraded "I can train a model" to "I can train a model and trust the result." From here on, every week's project should follow this discipline:

1. Set seed
2. Overfit one batch (sanity)
3. LR finder (set the LR)
4. Train with schedule + clipping + logging
5. Best-val checkpoint + early stop
6. Test set is touched **once** at the end

If a senior engineer reviews your training code from this week forward, they should nod. That's the bar.

---

**Next**: [Week 09: CNNs →](../week-09-cnns/readme.md)
