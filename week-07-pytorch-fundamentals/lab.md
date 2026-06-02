# Week 07: Lab — PyTorch Fundamentals

You'll port your week-05 numpy MLP to PyTorch, train it on MNIST, and confirm the same >97% test accuracy in ~30 lines of training code. Then push to GPU.

## Setup

```bash
# For NVIDIA GPU on Linux/Windows (CUDA 12.4):
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CPU only / macOS:
uv add torch torchvision
```

Verify:

```python
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())   # Apple Silicon
```

In a new notebook `07_pytorch.ipynb`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# Reproducibility (week 08 has the full story)
import random, numpy as np
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
```

---

## Exercise 7.1 — Tensors warm-up

```python
# Create a few tensors
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.randn(2, 3, dtype=torch.float32)
c = torch.zeros(5, 5).long()         # int64
print(a.dtype, b.shape, c.dtype)

# numpy interop
np_arr = np.random.randn(4, 4)
t1 = torch.from_numpy(np_arr)         # WARNING: float64 by default
t2 = torch.tensor(np_arr).float()      # explicit float32
print(t1.dtype, t2.dtype)

# Memory sharing (with from_numpy)
np_arr[0, 0] = 999
print(t1[0, 0])     # also 999 — same memory!

# Reshape, view, permute
x = torch.randn(2, 3, 4)
print(x.view(6, 4).shape)
print(x.reshape(-1, 4).shape)
print(x.permute(2, 0, 1).shape)
```

**Bug to internalize:** `from_numpy` defaults to whatever dtype the numpy array has, usually `float64`. PyTorch's models are `float32`. Mismatching them gives subtle errors. **Always `.float()` your inputs explicitly.**

---

## Exercise 7.2 — Autograd

Verify PyTorch's autograd against a hand-computed gradient:

```python
# f(x) = x^3 + 2x^2 - 4x + 1
# f'(x) = 3x^2 + 4x - 4

x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x ** 2 - 4 * x + 1
y.backward()
print(f"y = {y.item():.4f}")              # y = 12 + 8 - 8 + 1 = 13
print(f"dy/dx = {x.grad.item():.4f}")     # 3(4) + 8 - 4 = 16

# Accumulation gotcha
y2 = x ** 2
y2.backward()
print(f"x.grad after 2nd backward (accumulated): {x.grad.item()}")
# 16 + 4 = 20 — see, it accumulated. ALWAYS zero_grad in training loops.
```

---

## Exercise 7.3 — Load MNIST through torchvision

```python
transform = transforms.Compose([
    transforms.ToTensor(),                 # PIL → (1, 28, 28) tensor in [0, 1]
    transforms.Normalize((0.1307,), (0.3081,)),   # MNIST mean and std
])

train_ds = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_ds = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
print(f"train: {len(train_ds)}, test: {len(test_ds)}")
print(f"sample image: {train_ds[0][0].shape}, label: {train_ds[0][1]}")

# DataLoaders
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)
```

(Use `num_workers=0` in Jupyter to avoid subprocess complications. Bump up to 4 for `.py` scripts.)

Quick visualize:

```python
X_batch, y_batch = next(iter(train_loader))
fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i, ax in enumerate(axes):
    ax.imshow(X_batch[i].squeeze(), cmap='gray')
    ax.set_title(str(y_batch[i].item()))
    ax.axis('off')
plt.show()
```

---

## Exercise 7.4 — Build the MLP

Two ways. Make sure both produce the same architecture.

```python
# (A) Subclass nn.Module
class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=128, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)          # flatten (B, 1, 28, 28) → (B, 784)
        x = F.relu(self.fc1(x))
        return self.fc2(x)                 # raw logits — loss applies log_softmax

model_a = MLP()

# (B) nn.Sequential — same model, less code
model_b = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 10),
)

# Both have the same parameter count
print("MLP params:       ", sum(p.numel() for p in model_a.parameters()))
print("Sequential params:", sum(p.numel() for p in model_b.parameters()))
```

Both should print `(784*128 + 128) + (128*10 + 10) = 101,770` parameters.

---

## Exercise 7.5 — Training loop

```python
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"using device: {device}")

model = MLP().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

n_epochs = 5
train_losses, test_accs = [], []

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        epoch_loss += loss.item()
        n_batches += 1

    train_losses.append(epoch_loss / n_batches)
    acc = evaluate(model, test_loader, device)
    test_accs.append(acc)
    print(f"epoch {epoch+1}: train loss {train_losses[-1]:.4f}  test acc {acc:.4f}")
```

You should see test accuracy reach **>97%** in 5 epochs — same as week 05's numpy version. **Now you've done it with autograd handling the gradients.**

---

## Exercise 7.6 — Confirm against week 05

You manually computed gradients in week 05 and got the same answer. Now compare to PyTorch's gradient on a single batch:

```python
# Reset model to known state
torch.manual_seed(123)
model = MLP().to('cpu')
opt = torch.optim.SGD(model.parameters(), lr=0.0)   # zero-LR so weights don't move

X, y = next(iter(train_loader))
X, y = X.to('cpu'), y.to('cpu')

opt.zero_grad()
loss = loss_fn(model(X), y)
loss.backward()

print(f"fc1.weight.grad norm: {model.fc1.weight.grad.norm().item():.6f}")
print(f"fc2.weight.grad norm: {model.fc2.weight.grad.norm().item():.6f}")
```

These should be reproducibly the same number every time you run with seed 123. **The gradients PyTorch computes are bit-exact the formulas you derived in week 05** (modulo float32 rounding).

---

## Exercise 7.7 — Save and load

```python
# Save
torch.save({
    'model': model.state_dict(),
    'optimizer': opt.state_dict(),
    'epoch': n_epochs,
}, 'mnist_mlp.pt')

# Load (later, maybe in a different session)
ckpt = torch.load('mnist_mlp.pt', map_location='cpu')
new_model = MLP()
new_model.load_state_dict(ckpt['model'])
new_model.eval()

# Verify it gives the same predictions
with torch.inference_mode():
    X_test, _ = next(iter(test_loader))
    preds_a = model.to('cpu')(X_test).argmax(dim=-1)
    preds_b = new_model(X_test).argmax(dim=-1)
    print(f"agreement: {(preds_a == preds_b).float().mean().item():.4f}")    # 1.0
```

A loaded checkpoint that doesn't produce the same predictions = something is wrong (random init was used, or the architecture changed, or `eval()` wasn't called).

---

## Exercise 7.8 — CPU vs GPU benchmark

If you have a GPU, measure the speedup:

```python
import time

def benchmark(device, n_steps=100):
    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X = torch.randn(256, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (256,), device=device)

    # Warm up (especially for GPU; first kernels are slower)
    for _ in range(5):
        opt.zero_grad(); loss = loss_fn(model(X), y); loss.backward(); opt.step()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        opt.zero_grad(); loss = loss_fn(model(X), y); loss.backward(); opt.step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return time.perf_counter() - t0

cpu_time = benchmark(torch.device("cpu"))
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    gpu_time = benchmark(torch.device("cuda"))
    print(f"GPU: {gpu_time:.3f}s")
    print(f"speedup: {cpu_time/gpu_time:.1f}×")
```

For this tiny MLP the speedup will be modest (maybe 2-10×) — most of the time is overhead, not compute. By week 09 you'll have CNNs where the speedup is 50-100×.

---

## Exercise 7.9 — Write a custom Dataset for a CSV

The skill that comes up every week in real ML jobs. Make a synthetic CSV and write a `Dataset` for it:

```python
# Make a CSV
df = __import__('pandas').DataFrame({
    'feature_a': np.random.randn(500),
    'feature_b': np.random.randn(500),
    'feature_c': np.random.randn(500),
    'label': np.random.randint(0, 3, 500),
})
df.to_csv('synthetic.csv', index=False)

# Custom Dataset
from torch.utils.data import Dataset
import pandas as pd

class CSVDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df[['feature_a','feature_b','feature_c']].to_numpy(),
                              dtype=torch.float32)
        self.y = torch.tensor(df['label'].to_numpy(), dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

ds = CSVDataset('synthetic.csv')
print(f"dataset size: {len(ds)}")
print(f"sample: {ds[0]}")

# Use with a DataLoader
loader = DataLoader(ds, batch_size=32, shuffle=True)
for Xb, yb in loader:
    print(Xb.shape, yb.shape)
    break
```

That pattern — load CSV in `__init__`, fetch one row in `__getitem__` — generalizes to JSONL, parquet, images-on-disk, anything. Plus image augmentation (week 09) and tokenization (week 12) typically lives in `__getitem__`.

---

## Exercise 7.10 (stretch) — Compare `nn.Module` vs `nn.Sequential` training

Train both `model_a` (subclass) and `model_b` (Sequential) with the same seed, same data, same optimizer. Confirm they reach the same final accuracy. **They should** — the architecture is identical; the API is just different.

---

## Submission checklist

- [ ] All tensor/dtype/reshape examples ran
- [ ] PyTorch autograd gradient matched the analytic answer
- [ ] MNIST trained to **>97% test accuracy** in ≤ 5 epochs
- [ ] Same training run also done as `nn.Sequential` — produced same accuracy
- [ ] Checkpoint saved, reloaded, predictions verified equal
- [ ] (If you have a GPU) speedup measured and ≥ 2×
- [ ] Custom `CSVDataset` ran with a real `DataLoader`

---

## What you just did

You replaced your hand-rolled numpy backward (week 05) and your hand-built autograd (week 06) with the industry-standard one. The training loop is now:

```python
for X, y in loader:
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model(X), y)
    loss.backward()
    opt.step()
```

Four lines. That's the same four lines you'll write for transformers (week 10), CNN training (week 09), and LLM fine-tuning (week 12). Everything else this curriculum touches is built around variations of this loop. **Memorize the order.**

---

**Next**: [Week 08: Training-Loop Discipline →](../week-08-training-loop-discipline/readme.md)
