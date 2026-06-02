# Week 08: Theory — Training-Loop Discipline

You can write a training loop. So can most people. The difference between an ML engineer who ships and one who doesn't is **discipline around the loop**: pre-flight checks before training, sane defaults, diagnostic plots, reproducibility, and knowing what a "bad" loss curve looks like before you've wasted a GPU-day.

This week is the senior playbook. Most of it comes from [Karpathy's recipe](https://karpathy.github.io/2019/04/25/recipe/) — read it once a month for the rest of your career.

---

## Part 1: The single most important rule — *overfit one batch first*

Before you train on the full dataset, before you set up logging, before anything: **make sure your model can drive loss to zero on a single batch of ~32 examples.**

```python
X, y = next(iter(train_loader))
X, y = X.to(device), y.to(device)
model = MyModel().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(500):
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model(X), y)
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(f"step {step}: loss {loss.item():.6f}")
```

If after 500 steps the loss isn't approaching zero (or very close), **stop**. There's a bug. Common culprits:

| Symptom | Likely bug |
|---|---|
| Loss doesn't go down at all | Forgot `opt.zero_grad()`; gradient flow blocked (`detach`, `no_grad`) |
| Loss decreases then plateaus high | Output activation incompatible with loss (e.g., softmax + CrossEntropyLoss — double softmax) |
| Loss goes to NaN immediately | LR too high; dtype mismatch; division by zero in loss |
| Loss decreases for one class only | Label encoding wrong (off-by-one, wrong dtype) |
| Loss decreases for the *wrong* answers | Data shuffling broke the X-y alignment |

Catching a bug after 500 steps on a single batch = 30 seconds. Catching it after 6 hours on the full dataset = lost day. **Overfit one batch.**

---

## Part 2: The Karpathy recipe (compressed)

In order:

1. **Become one with the data.** Look at it. Plot histograms. Eyeball labels. Datasets lie.
2. **Build the dumb baseline.** Mean predictor, logistic regression — whatever sets the floor. If your fancy model isn't beating this, the model is broken.
3. **Overfit one batch.** (Part 1.)
4. **Find an LR.** (Part 5.)
5. **Train.** Watch curves religiously.
6. **Regularize.** Only after you've overfit.
7. **Squeeze.** Augmentation, more data, ensembling.

The reverse-engineered version: most "my model isn't learning" issues are 1, 2, or 3. Almost never 5 or 6.

---

## Part 3: Sane defaults — what to set first time

```python
# Hyperparameters that work most of the time for medium models
lr = 3e-4              # Adam's "Karpathy constant"
batch_size = 64        # bigger if GPU permits
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
weight_decay = 0.01    # L2 regularization (recall MAP from week 02)
n_epochs = 10          # start here; extend if loss is still decreasing
```

- **3e-4** for Adam was Karpathy's joke ("the best LR for Adam is 3e-4") that turned out to be roughly true for many image and small-NLP models. Not magic — just a sensible default.
- **AdamW** over **Adam** by default. The decoupling matters more for transformers; for MLPs both are fine.
- **`weight_decay=0.01`** is the modern default for transformer training. For small MLPs you can use 0.

Then tune the LR. Always.

---

## Part 4: Learning rate — the single most important hyperparameter

LR matters more than batch size, more than weight decay, more than your choice of optimizer. Get it wrong and nothing else matters.

### The shape of the loss curve tells you what's wrong

```
LR too high:                  LR too low:
  loss                           loss
   │   /\  /\                     │
   │  /  \/  \   /\               │  \
   │ /        \_/  \              │   \
   │/                              │    \___________
   └────────── time                └─────────── time
   (oscillates / diverges)         (decreasing too slowly)

LR ~right:
  loss
   │ \
   │  \
   │   \_____
   │         \____________
   └──────────────── time
   (smooth descent, then plateau)
```

### The LR finder (Smith, 2017)

Don't guess; sweep. Train one epoch with LR ramping exponentially from 1e-7 to 1.0; plot loss vs LR; pick the LR where loss is decreasing the fastest (steepest negative slope, before the curve explodes).

```python
def lr_finder(model, loader, opt, loss_fn, min_lr=1e-7, max_lr=1.0, n_steps=100):
    lrs, losses = [], []
    for i, (X, y) in enumerate(loader):
        if i >= n_steps:
            break
        lr = min_lr * (max_lr / min_lr) ** (i / (n_steps - 1))
        for g in opt.param_groups:
            g['lr'] = lr
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
        lrs.append(lr); losses.append(loss.item())
    return lrs, losses
```

Plot `losses` vs `lrs` on log-x scale. The sweet-spot LR is roughly **10× smaller** than the LR where the curve hits the minimum.

---

## Part 5: Learning rate schedulers

You'll often want LR to **change** during training. Three patterns dominate:

### Step decay

```python
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
```

Classic. LR drops by 10× every 30 epochs. ResNet's original training used this.

### Cosine annealing

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
```

LR smoothly decreases from max to ~0 following a cosine curve. **Modern default for most non-LLM training.**

### Cosine with warmup (transformer-style)

```python
from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(
    opt,
    num_warmup_steps=1000,
    num_training_steps=total_steps,
)
```

LR rises linearly for the first N steps (warmup), then cosine-decays. **The standard for transformers.** Warmup prevents early instability when gradients are huge.

Call `scheduler.step()` at the end of each training step (or epoch, depending on the scheduler).

---

## Part 6: Gradient clipping

Sometimes gradients explode — especially in transformers, RNNs, and any model with sharp loss landscapes. The cure: **clip the global gradient norm before the optimizer step**.

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
opt.step()
```

Common values: `1.0` for transformers, `5.0` or `10.0` for CNNs, `0.5` for RNNs/LSTMs. You'll know clipping is doing real work when training NaNs without it. Cheap insurance; always include.

---

## Part 7: Logging — TensorBoard and W&B

A training run with no logging is a black box. Two tools to know:

### TensorBoard (built into PyTorch)

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/exp_001")

for step, (X, y) in enumerate(train_loader):
    ...
    writer.add_scalar("train/loss", loss.item(), step)
    writer.add_scalar("train/lr", opt.param_groups[0]['lr'], step)

writer.close()
```

Launch with `tensorboard --logdir runs/`. Local, free, no account.

### Weights & Biases (W&B)

```python
import wandb
wandb.init(project="ml-mentorship", config={"lr": 1e-3, "batch_size": 64})

# Same idea
wandb.log({"train/loss": loss.item(), "train/lr": opt.param_groups[0]['lr']})
```

Cloud-hosted, free for personal use, much better collaboration features than TensorBoard. The de-facto standard at modern ML labs. Worth setting up an account.

### What to log

| Always | Sometimes |
|---|---|
| Train loss (per step) | Per-class accuracy |
| Validation loss + metrics (per epoch) | Gradient norm |
| Learning rate (per step) | Sample predictions (image grids, generated text) |
| Steps / sec or examples / sec | Model parameter histograms |
| Hyperparameter config | GPU memory usage |

Log **everything you'd want to look back at**. Don't log so much it slows training or buries signal.

---

## Part 8: Diagnostic plots — reading loss curves

### Healthy

```
loss
 │\
 │ \____
 │      \________________
 │
 └────────────── epoch
```

Train loss drops, plateaus. Val loss tracks closely. Test accuracy increases. Boring is good.

### Overfit

```
loss
 │\
 │ \___ train
 │      \____________
 │
 │      ╱╱  val
 │ ___╱╱
 │╱
 └────────────── epoch
```

Train loss keeps dropping; val loss U-shapes (decreases, then rises). The gap = the overfit. Cure: more data, augmentation, dropout, weight decay, early stopping.

### Underfit

```
loss   train + val (close together, but high)
 │ ──────────────────────
 │
 └────────────── epoch
```

Loss plateaus at a high value. Model is too small or LR too low. Cure: bigger model, more capacity, longer training.

### Dead training (LR too high)

```
loss
 │  /\  /\
 │ /  \/  \  /\
 │/        \/  \
 └────────────── step
```

Loss oscillates wildly or NaNs. Cure: lower LR, smaller initial weights, gradient clipping.

### Plateau / no learning

```
loss
 │ ────────────
 │
 └────────────── step
```

Loss flat from step 0. Cure: check (1) is there a zero_grad missing? (2) are gradients actually flowing? (3) is your loss function correct? (4) is one batch overfitting (Part 1)?

---

## Part 9: Reproducibility — the seed dance, harder version

Beyond week 07's seed-setting, true reproducibility requires:

```python
import os, random, numpy as np, torch
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False     # disables auto-tuning
    # PyTorch 1.8+ — error on any nondeterministic op
    torch.use_deterministic_algorithms(True, warn_only=True)
```

Even then, atomic-add operations on CUDA (e.g., scatter) are nondeterministic. For truly bit-exact runs you'd disable those — at a perf cost.

A pragmatic stance: **same-seed runs should reproduce to ~5 sig figs of test accuracy**. If they don't, find out why before trusting the numbers.

### Reproducible data loading

```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=g,
)
```

Without `worker_init_fn`, each DataLoader worker has its own (unseeded) random state — your augmentation order varies between runs.

---

## Part 10: Checkpointing strategy

**Save often.** Specifically:

- **Best-val checkpoint** — overwrites only when validation metric improves. This is the model you'll use for the final report.
- **Last checkpoint** — overwritten every N steps. The recovery point if training crashes.
- **Periodic checkpoints** — every M epochs, to a separate file. For comparing checkpoints later (e.g., did the model "know" something at epoch 10 that it forgot by epoch 50?).

```python
best_val = float('inf')
for epoch in range(n_epochs):
    train_one_epoch(...)
    val_loss = evaluate(...)

    # Save last
    torch.save({...}, "ckpt/last.pt")

    # Save best
    if val_loss < best_val:
        best_val = val_loss
        torch.save({...}, "ckpt/best.pt")

    # Periodic
    if epoch % 10 == 0:
        torch.save({...}, f"ckpt/epoch_{epoch}.pt")
```

**Always save the optimizer + scheduler state too.** A naked model checkpoint can't resume training cleanly; it'll lose Adam's moment estimates and the scheduler's position.

---

## Part 11: Early stopping

When val loss stops improving for `patience` epochs, stop training. Saves time, prevents overfitting.

```python
patience = 5
best = float('inf'); bad_epochs = 0
for epoch in range(n_epochs):
    val_loss = evaluate(...)
    if val_loss < best:
        best = val_loss
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print(f"early stop at epoch {epoch}")
            break
```

Pair with the "best-val" checkpoint above. You stop training but use the best snapshot, not the last one.

---

## Part 12: Common silent killers

A non-exhaustive list of bugs that don't crash but ruin training:

| Bug | Symptom | Cure |
|---|---|---|
| Forgot `model.eval()` before validation | Val loss higher than train loss with no actual overfit (dropout still on) | `model.eval()`; `with torch.no_grad()` |
| Forgot `model.train()` after validation | Subsequent training is slower, worse | `model.train()` at the top of the train loop |
| `softmax + CrossEntropyLoss` | Loss decreases but slowly, plateaus high | Remove the softmax; CE wants raw logits |
| Wrong loss reduction | Loss scales wildly with batch size | Pick `reduction="mean"` consistently |
| Off-by-one in label encoding | One class never predicted | Check `y.unique()` and class names align |
| LR schedule out of sync with optimizer | LR doesn't actually change | Call `scheduler.step()` at the right place |
| Validation set leaking into train | Val accuracy implausibly high (~99% on hard task) | Re-check `train_test_split` and any augmentation |

Read this list once per project. The bugs you avoid are the days you save.

---

## What's next

In [lab.md](lab.md) you'll:
- Run the "overfit one batch" sanity check
- Build an LR finder for your week-07 MNIST MLP
- Add cosine-with-warmup scheduling and gradient clipping
- Set up TensorBoard or W&B logging
- Read three "bad" loss curves and identify the bug in each
- Implement reproducible-seed + best-val checkpointing

By week's end your training loop will be a senior-engineering one. You'll spend the rest of the curriculum applying it to bigger and bigger models.
