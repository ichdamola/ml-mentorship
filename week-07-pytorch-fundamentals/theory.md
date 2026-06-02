# Week 07: Theory — PyTorch Fundamentals

You've built backprop by hand (week 05) and built autograd from scratch (week 06). Now switch to the professional version. PyTorch is the industry-standard deep-learning framework — it's what 95% of research papers, 70% of production teams, and 100% of this curriculum use from here on.

The goal this week: stop fighting the tool. By the end you'll write PyTorch code that reads exactly like the official tutorials, no head-scratching.

---

## Part 1: Tensors — numpy with a few extras

A `torch.Tensor` is essentially a numpy array with three superpowers:

1. **Autograd** — it can track operations and compute gradients
2. **GPU support** — it can live on a CUDA device
3. **Type-strict** — `float32` by default (not `float64` like numpy)

```python
import torch

# Creation — same patterns as numpy
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.zeros(3, 4)
c = torch.randn(2, 3, 4)
d = torch.arange(10)
e = torch.linspace(0, 1, 100)

# From numpy
import numpy as np
np_arr = np.random.randn(5)
t = torch.from_numpy(np_arr)        # zero-copy, shares memory
back = t.numpy()                     # zero-copy, shares memory

# Type and shape
print(a.dtype, a.shape, a.device)   # torch.float32, torch.Size([3]), cpu
```

### Dtype rules — the ones that bite

| Default | Reason |
|---|---|
| `torch.float32` | Faster than 64; "enough" precision for ML |
| `torch.int64` | "Long" ints — used for indices, class labels |
| `torch.bool` | Masks |

**Common gotchas:**

```python
# numpy → torch defaults to float64 (numpy's default)
t = torch.from_numpy(np.array([1.0, 2.0]))
print(t.dtype)    # float64

# A model in float32 then receives float64 input → silent slow path
model = torch.nn.Linear(2, 2)         # float32
out = model(t)                          # error or implicit upcast

# Fix: explicit
t = torch.from_numpy(np.array([1.0, 2.0])).float()      # float32
```

### Reshape, view, permute, contiguous

```python
x = torch.randn(2, 3, 4)
x.view(6, 4)                # like reshape, but requires contiguous memory
x.reshape(6, 4)             # like view, but copies if needed
x.permute(2, 0, 1)          # swap axes → shape (4, 2, 3)
x.transpose(0, 1)           # swap two axes → (3, 2, 4)

# After permute/transpose, memory is non-contiguous
# .contiguous() makes a fresh copy with contiguous layout
y = x.permute(2, 0, 1).contiguous()
```

**Rule:** after `permute` or `transpose`, call `.contiguous()` before passing to operations that require contiguous layouts (like `.view()`). Or just use `.reshape()` which handles it for you.

---

## Part 2: Autograd in PyTorch

PyTorch builds the computation graph you wrote in week 06 — but automatically, on every operation.

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)    # tensor([7.0])   (∂y/∂x = 2x + 3 at x=2)
```

The key trio:

- `requires_grad=True` — mark the tensor as "track operations on me"
- `loss.backward()` — start backward from this scalar, populate `.grad` on all `requires_grad` leaf tensors
- `x.grad` — read the accumulated gradient

### Gradient accumulation (the same gotcha as week 06)

PyTorch's `.grad` **accumulates** — exactly like your micrograd. If you call `backward()` twice without zeroing, gradients add up:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)    # 4.0
y = x ** 2
y.backward()
print(x.grad)    # 8.0  ← accumulated!
```

That's why every training loop has this line:

```python
optimizer.zero_grad()
```

(Or, more efficient, `optimizer.zero_grad(set_to_none=True)` — sets `.grad` to `None` instead of zero, saving a memset.)

### `with torch.no_grad():` and `torch.inference_mode()`

For inference, you don't need gradients. Disabling tracking saves memory:

```python
# Older idiom, still common
with torch.no_grad():
    predictions = model(inputs)

# Newer idiom (PyTorch 1.9+) — slightly faster
with torch.inference_mode():
    predictions = model(inputs)
```

`inference_mode` is stricter — tensors created inside it can't be used in autograd later. For production inference, that's exactly what you want.

### `detach()`

Sometimes you want to use a tensor's value without it being part of the graph. `detach()` returns a new tensor sharing data but cut off from autograd:

```python
features = encoder(x)
detached = features.detach()     # value is the same; gradient won't flow through
```

This is how you implement teacher-student training, knowledge distillation, and many RL tricks.

---

## Part 3: `nn.Module` — the API

Every PyTorch model inherits from `nn.Module`. The contract:

1. **In `__init__`**: declare your sub-modules (layers) as attributes. PyTorch automatically registers them.
2. **In `forward`**: define the computation. **Don't call `forward()` directly** — call the module instance, which routes through `__call__` (which adds hooks etc.).

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x      # raw logits; loss applies softmax internally

model = MLP()
print(model)
# MLP(
#   (fc1): Linear(in_features=784, out_features=128, bias=True)
#   (fc2): Linear(in_features=128, out_features=10, bias=True)
# )
```

### `nn.Sequential` — the boilerplate shortcut

For pure feedforward stacks:

```python
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
```

Same model, zero subclassing. Use `Sequential` when you don't need control flow; subclass `nn.Module` when you do.

### Parameters and state

```python
# All learnable parameters
for name, p in model.named_parameters():
    print(name, p.shape, p.requires_grad)
# fc1.weight torch.Size([128, 784]) True
# fc1.bias   torch.Size([128])      True
# fc2.weight torch.Size([10, 128])  True
# fc2.bias   torch.Size([10])       True

# All sub-modules
for name, m in model.named_modules():
    print(name, type(m).__name__)
```

`model.parameters()` is what you'll pass to an optimizer.

### `.train()` vs `.eval()`

Some layers behave differently during training vs evaluation:

| Layer | `.train()` | `.eval()` |
|---|---|---|
| `Dropout` | Drops random units | No dropout, scales accordingly |
| `BatchNorm` | Updates running mean/var; uses batch stats | Uses stored running stats |
| `LayerNorm` | Same in both | Same in both |

**Rule:** before evaluating, `model.eval()`. After eval, `model.train()`. Mixing these up is bug #1 in every PyTorch tutorial.

---

## Part 4: Layers you'll use this week and next

| Layer | What it does |
|---|---|
| `nn.Linear(in, out)` | `y = xW^T + b` |
| `nn.ReLU()` | Elementwise `max(0, x)` |
| `nn.Sigmoid()` | `1/(1+e^-x)` |
| `nn.Softmax(dim=-1)` | Probability vector along given dim. **Don't use** if your loss is `CrossEntropyLoss` (it fuses softmax) |
| `nn.Dropout(p)` | Randomly zeros `p` fraction of activations during training |
| `nn.BatchNorm1d(features)` | Normalizes per-feature across batch |
| `nn.LayerNorm(features)` | Normalizes per-feature, per-example |
| `nn.Conv2d(in_ch, out_ch, k)` | 2D convolution (week 09) |
| `nn.Embedding(num, dim)` | Lookup table — for token embeddings (week 10) |
| `nn.Identity()` | No-op; placeholder for ablations |

### Losses

| Loss | Use |
|---|---|
| `nn.CrossEntropyLoss` | K-class classification. **Fuses log-softmax + NLL.** Input is *raw logits*, target is integer class. |
| `nn.BCEWithLogitsLoss` | Binary classification. **Fuses sigmoid + BCE.** Input is *raw logit*, target is `0.`/`1.`. |
| `nn.MSELoss` | Regression |
| `nn.L1Loss` | Robust regression |
| `nn.HuberLoss` | Outlier-robust regression |
| `nn.NLLLoss` | If you've already applied `log_softmax` |

**Critical:** `CrossEntropyLoss` expects raw logits, *not* probabilities. If you `.softmax()` before passing to it, you'll double-apply softmax and your training will be terrible.

---

## Part 5: Optimizers

PyTorch's optimizers wrap parameter lists and implement the update rule.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

### When each wins

| Optimizer | When |
|---|---|
| **SGD** with momentum | Classical training; sometimes still best for very large models when tuned |
| **Adam** | Sane default; adapts learning rate per parameter |
| **AdamW** | **Best default for transformers.** Decouples weight decay from gradient — fixes a subtle bug in Adam's L2 handling |
| **Lion** | Newer (2023); good for fine-tuning at smaller LRs |
| **Adafactor** | Memory-efficient for huge models |

For this curriculum: **Adam or AdamW for everything until week 11 explicitly says otherwise.**

### The standard loop

```python
for X, y in dataloader:
    optimizer.zero_grad(set_to_none=True)
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
```

This four-line pattern is the heart of every PyTorch training script you'll ever read. Memorize the order:

1. **Zero grads** (clear previous step)
2. **Forward** — compute predictions and loss
3. **Backward** — compute gradients
4. **Step** — apply the gradients

---

## Part 6: Datasets and DataLoaders

### `Dataset` — knows how to fetch one sample

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

That's it. Two methods.

### `DataLoader` — batches + shuffles + parallel-loads

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    MyDataset(X_train, y_train),
    batch_size=64,
    shuffle=True,
    num_workers=4,       # parallel workers for I/O — set 0 on Windows / debugger
    pin_memory=True,     # for GPU; copies to pinned host memory for faster H→D
    drop_last=False,     # drop last partial batch
)

for X_batch, y_batch in loader:
    ...
```

### When to write a custom `Dataset` vs use `torchvision.datasets`

- For canonical benchmarks (MNIST, CIFAR-10, ImageNet) — use `torchvision.datasets`. They handle download, caching, transforms.
- For your own CSV/parquet/JSONL — write a `Dataset`. Three methods. Done.
- For images on disk in `train/dog/`, `train/cat/` — use `torchvision.datasets.ImageFolder`.

---

## Part 7: GPUs — `.to(device)`

PyTorch can put any tensor or module on a GPU:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Apple Silicon: device = torch.device("mps")

model = MLP().to(device)
for X, y in loader:
    X, y = X.to(device), y.to(device)
    ...
```

**Rules:**

- Everything that interacts in a computation must be on the **same device**. Mixing CPU + CUDA → `RuntimeError`.
- `optimizer` is constructed from `model.parameters()` **after** `model.to(device)` — so it picks up the GPU pointers.
- For inference, also call `model.eval()` so dropout is off.

### Multi-GPU and beyond

Single GPU: just call `.to(device)`. Two-or-more GPUs:

- **`nn.DataParallel`** — old, simple, suboptimal. Skip.
- **`nn.parallel.DistributedDataParallel` (DDP)** — proper data parallelism. Week 11.
- **FSDP / DeepSpeed** — for models too big for one GPU. Also week 11.

---

## Part 8: Saving and loading

The right thing to save is the **state dict** (parameter weights), not the model object itself:

```python
torch.save(model.state_dict(), "model.pt")

# Later:
model = MLP()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()
```

A "checkpoint" usually includes more — optimizer state, epoch number, scheduler state:

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "loss": best_loss,
}
torch.save(checkpoint, "ckpt.pt")
```

Don't `torch.save(model)` (the whole object) unless you have a really good reason — it serializes class hierarchy, breaks across PyTorch versions, and is a security risk if loading untrusted files.

---

## Part 9: Reproducibility — the seed dance

```python
import random, numpy as np, torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

This makes most operations reproducible. **It doesn't guarantee bit-exact reproduction across:**
- Different PyTorch versions
- Different CUDA versions
- Different GPU models
- Atomic-add operations on CUDA (e.g., scatter, certain pooling) — these have nondeterministic floating-point order

For a deeper guarantee, see [`torch.use_deterministic_algorithms(True)`](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html), which will throw on nondeterministic ops so you can replace them.

Reproducibility properly is a [Week 08 topic](../week-08-training-loop-discipline/).

---

## Part 10: What's coming

Week 08 adds the *discipline* around this — the loop you have now is the skeleton; next week we add learning-rate schedules, gradient clipping, early stopping, logging, and the "overfit one batch first" sanity check. Week 09 swaps the MLP for a CNN.

Don't overload this week with optimization tricks. **Get fluent at the basics.** Train MNIST again, but in PyTorch this time. Confirm you can build, train, save, load, and reload to GPU without consulting docs every five seconds.

---

## What's next

In [lab.md](lab.md) you'll:
- Convert your week-05 numpy MLP to PyTorch
- Train MNIST and confirm the same >97% accuracy
- Move it to GPU (if available) and measure the speedup
- Save and reload a checkpoint
- Build a custom `Dataset` for your own CSV
- (Stretch) Use `nn.Sequential` and confirm it matches your subclass

By end of week 07 the gap between "I want to train a model" and "I have a model training" should be measured in minutes.
