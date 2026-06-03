# Week 09: Lab — CNNs

You'll implement convolution by hand, build a small CNN, then a ResNet that hits >90% on CIFAR-10, and finally transfer-learn from a pretrained ResNet-50. All under the discipline you learned in week 08.

## Setup

```bash
uv add torchvision pillow
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# Week 08 reproducibility scaffold
import random, os
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}")
```

If you don't have a GPU, this lab is doable on CPU but week 09's training will take 30+ minutes. Colab T4 is free and runs all exercises in under 15 minutes.

---

## Exercise 9.1 — Convolution by hand (numpy)

Implement a single-channel 2D convolution from scratch, then verify against `nn.Conv2d`.

```python
def conv2d_naive(X, K):
    """X: (H, W) input. K: (kH, kW) kernel. No padding, stride 1.
    Returns Y: (H - kH + 1, W - kW + 1)."""
    H, W = X.shape
    kH, kW = K.shape
    Y = np.zeros((H - kH + 1, W - kW + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+kH, j:j+kW] * K).sum()
    return Y

# Test
X = np.array([[1, 2, 3, 0],
              [0, 1, 2, 3],
              [3, 0, 1, 2],
              [2, 3, 0, 1]], dtype=np.float32)
K = np.array([[1, 0],
              [0, -1]], dtype=np.float32)
Y = conv2d_naive(X, K)
print("Naive conv output:")
print(Y)

# Verify against PyTorch
X_t = torch.tensor(X).unsqueeze(0).unsqueeze(0)         # (1, 1, 4, 4)
K_t = torch.tensor(K).unsqueeze(0).unsqueeze(0)         # (1, 1, 2, 2)
Y_t = F.conv2d(X_t, K_t).squeeze().numpy()
print("\nPyTorch conv output:")
print(Y_t)

print(f"\nmatch: {np.allclose(Y, Y_t)}")
```

**Bug to internalize:** Python-textbook convolution flips the kernel; deep-learning libraries don't (they do cross-correlation). The output above matches `F.conv2d` directly. If you tried to match `scipy.signal.convolve2d`, the kernel would be flipped.

---

## Exercise 9.2 — Visualize a learned edge detector

Pick a few hand-crafted edge kernels and apply them to a real image to see what conv learns.

```python
from torchvision.datasets import CIFAR10
from torchvision import transforms as T

# Load one CIFAR image as numpy
cifar = CIFAR10('./data', train=True, download=True, transform=T.ToTensor())
img_t, label = cifar[7]                                 # (3, 32, 32)
img = img_t.mean(dim=0).numpy()                         # gray, (32, 32)

# Edge detectors
K_sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
K_sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
K_blur   = np.ones((3, 3), dtype=np.float32) / 9

img_t_gray = torch.tensor(img).unsqueeze(0).unsqueeze(0)   # (1, 1, 32, 32)
edge_x = F.conv2d(img_t_gray, torch.tensor(K_sobel_x).unsqueeze(0).unsqueeze(0), padding=1).squeeze()
edge_y = F.conv2d(img_t_gray, torch.tensor(K_sobel_y).unsqueeze(0).unsqueeze(0), padding=1).squeeze()
blurred = F.conv2d(img_t_gray, torch.tensor(K_blur).unsqueeze(0).unsqueeze(0), padding=1).squeeze()

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
axes[0].imshow(img, cmap='gray'); axes[0].set_title('original')
axes[1].imshow(edge_x.numpy(), cmap='gray'); axes[1].set_title('Sobel x (vert edges)')
axes[2].imshow(edge_y.numpy(), cmap='gray'); axes[2].set_title('Sobel y (horiz edges)')
axes[3].imshow(blurred.numpy(), cmap='gray'); axes[3].set_title('box blur')
for ax in axes: ax.axis('off')
plt.tight_layout(); plt.show()
```

The Sobel kernels detect edges in specific directions. **A learned CNN's first layer mostly contains kernels that look like these** — oriented edge detectors. You'll visualize that explicitly in 9.6.

---

## Exercise 9.3 — Load CIFAR-10 with augmentation

```python
# ImageNet normalization stats are close enough for CIFAR transfer learning
NORMALIZE = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2470, 0.2435, 0.2616]    # CIFAR-10 actual stats
)

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),         # weak crop augmentation
    transforms.RandomHorizontalFlip(),            # flip
    transforms.ToTensor(),
    NORMALIZE,
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    NORMALIZE,
])

train_set = CIFAR10('./data', train=True, download=True, transform=train_tf)
test_set = CIFAR10('./data', train=False, download=True, transform=test_tf)

# Carve val from train (standard CIFAR practice: hold out 5000 from training)
g = torch.Generator().manual_seed(42)
train_set, val_set = torch.utils.data.random_split(train_set, [45000, 5000], generator=g)
# Note: val_set inherits train_tf — for proper eval you'd swap to test_tf;
# Sub-exercise: write a wrapper Dataset that swaps transforms post-split.

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

# Visualize a batch
Xb, yb = next(iter(train_loader))
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def denorm(x):
    mean = torch.tensor([0.4914, 0.4822, 0.4465])[:, None, None]
    std  = torch.tensor([0.2470, 0.2435, 0.2616])[:, None, None]
    return (x * std + mean).clamp(0, 1)

fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i, ax in enumerate(axes):
    ax.imshow(denorm(Xb[i]).permute(1, 2, 0))
    ax.set_title(classes[yb[i].item()])
    ax.axis('off')
plt.tight_layout(); plt.show()
```

---

## Exercise 9.4 — A small CNN to baseline

Three conv blocks → flatten → linear. Use this as a baseline before the ResNet.

```python
class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # block 1: 32×32 → 16×16
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            # block 2: 16×16 → 8×8
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            # block 3: 8×8 → 4×4
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

print(SmallCNN())
n_params = sum(p.numel() for p in SmallCNN().parameters())
print(f"#parameters: {n_params:,}")
```

A few hundred thousand parameters. Train it with the week-08 discipline:

```python
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0.0
    with torch.inference_mode():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += loss_fn(logits, y).item()
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total

def train_model(model, train_loader, val_loader, n_epochs=20, lr=1e-3, wd=5e-4, name="small_cnn"):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_val = float('inf')
    for epoch in range(n_epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch:2d}  val loss {val_loss:.4f}  val acc {val_acc:.4f}  lr {opt.param_groups[0]['lr']:.5f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{name}_best.pt")

    model.load_state_dict(torch.load(f"{name}_best.pt"))
    return model

small = train_model(SmallCNN(), train_loader, val_loader, n_epochs=20, name="small_cnn")
test_loss, test_acc = evaluate(small, test_loader, device)
print(f"\nSmallCNN test acc: {test_acc:.4f}")
```

Expect ~82-85% test accuracy. Better than your MLP would manage by a huge margin, even with way fewer parameters.

---

## Exercise 9.5 — Build a ResNet block, then a small ResNet

Now the iconic residual block. Implement it from scratch.

```python
class BasicBlock(nn.Module):
    """A 2-conv residual block."""
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # Skip connection: if dims don't match, project with a 1×1 conv
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)           # the residual connection
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, n_classes=10, blocks_per_stage=(2, 2, 2)):
        super().__init__()
        # CIFAR-style stem: 3×3 conv (instead of ImageNet's 7×7)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.stage1 = self._make_stage(64, 64, blocks_per_stage[0], stride=1)
        self.stage2 = self._make_stage(64, 128, blocks_per_stage[1], stride=2)
        self.stage3 = self._make_stage(128, 256, blocks_per_stage[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, n_classes)

    def _make_stage(self, in_ch, out_ch, n_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(n_blocks - 1):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

resnet = ResNet()
n_params = sum(p.numel() for p in resnet.parameters())
print(f"ResNet #parameters: {n_params:,}")
```

This is roughly ResNet-14 in size. Train it longer than the small CNN:

```python
resnet = train_model(ResNet(), train_loader, val_loader, n_epochs=40, lr=1e-3, name="resnet")
test_loss, test_acc = evaluate(resnet, test_loader, device)
print(f"\nResNet test acc: {test_acc:.4f}")
```

**Target: >90% test accuracy.** It'll take 5-15 minutes on a modern GPU. If you're not getting >90%, increase epochs to 80 or check that augmentation is on (`RandomCrop` + `RandomHorizontalFlip`).

---

## Exercise 9.6 — Visualize learned filters

The first conv layer's weights — reshape to `(64, 3, 3, 3)` — should look like color edge detectors after training.

```python
first_conv = resnet.stem[0]
W = first_conv.weight.detach().cpu()           # (64, 3, 3, 3)
print(W.shape)

# Normalize each filter for display
W_vis = (W - W.amin(dim=(1,2,3), keepdim=True)) / (W.amax(dim=(1,2,3), keepdim=True) - W.amin(dim=(1,2,3), keepdim=True))

fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(W_vis[i].permute(1, 2, 0))      # (3, 3, 3) → (3, 3, 3) HWC
    ax.axis('off')
plt.suptitle("First conv layer filters — look for color/edge detectors")
plt.show()
```

You should see oriented edges, color blobs, and texture detectors — analogous to what Hubel & Wiesel found in cat visual cortex in the 1960s. **Networks rediscover edge detection from random initialization because that's what works for vision.**

---

## Exercise 9.7 — Transfer learning from ImageNet-pretrained ResNet-50

For a "real" task with limited data, transfer learning beats from-scratch by a wide margin.

```python
import torchvision.models as models

# Load pretrained
pretrained = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
print(f"pretrained #params: {sum(p.numel() for p in pretrained.parameters()):,}")

# Adapt to CIFAR-10: replace classifier
pretrained.fc = nn.Linear(pretrained.fc.in_features, 10).to(device)

# Linear probe: freeze everything except classifier
for name, p in pretrained.named_parameters():
    p.requires_grad = (name.startswith('fc'))

# Use ImageNet's expected input size (224) — upsample CIFAR
train_tf_224 = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
test_tf_224 = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
train_set_224 = CIFAR10('./data', train=True, download=False, transform=train_tf_224)
test_set_224 = CIFAR10('./data', train=False, download=False, transform=test_tf_224)
train_loader_224 = DataLoader(train_set_224, batch_size=64, shuffle=True, num_workers=0)
test_loader_224 = DataLoader(test_set_224, batch_size=128, shuffle=False, num_workers=0)

# Linear-probe training (only fc layer learns)
opt = torch.optim.AdamW(pretrained.fc.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

# Run a few epochs — for production you'd want more
for epoch in range(3):
    pretrained.train()
    for X, y in train_loader_224:
        X, y = X.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(pretrained(X), y)
        loss.backward()
        opt.step()
    test_loss, test_acc = evaluate(pretrained, test_loader_224, device)
    print(f"linear-probe epoch {epoch}: test acc {test_acc:.4f}")
```

**You should hit >92% test accuracy in 3 epochs**, beating your from-scratch ResNet — and you only trained the final 20,490 parameters. The Big-Tech-funded pretraining did the heavy lifting.

For comparison, try `pretrained = models.resnet50(weights=None)` (random init) and run the same loop — you'll get ~30-40% in 3 epochs. **The pretrained features are the single biggest lever in production vision ML.**

---

## Exercise 9.8 — Receptive field of a deep neuron

Measure the receptive field of a neuron in `stage3` of your ResNet by gradient analysis.

```python
resnet.eval()
# Pick one neuron in the final conv output
X_dummy = torch.randn(1, 3, 32, 32, device=device, requires_grad=True)

# Forward through stem + stage1 + stage2 + stage3
x = resnet.stem(X_dummy)
x = resnet.stage1(x)
x = resnet.stage2(x)
x = resnet.stage3(x)   # (1, 256, 8, 8) — pick a middle neuron

target = x[0, 0, 4, 4]   # one channel, center spatial
target.backward()

# Gradient at the input shows which input pixels affect that one output neuron
grad_at_input = X_dummy.grad.abs().sum(dim=1).squeeze().cpu().numpy()  # (32, 32)

plt.imshow(grad_at_input, cmap='hot')
plt.title("Receptive field of a neuron at stage3[0, 0, 4, 4]")
plt.colorbar()
plt.show()
```

The bright region shows the receptive field — should cover much of the 32×32 input, confirming that deep neurons see most of the image.

---

## Exercise 9.9 (stretch) — Profile speed: small CNN vs ResNet vs ResNet-50

```python
import time

def benchmark_inference(model, batch_size=64, n_iter=50):
    model.eval()
    X = torch.randn(batch_size, 3, 32, 32, device=device)
    for _ in range(5):                          # warmup
        with torch.inference_mode():
            model(X)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        with torch.inference_mode():
            model(X)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000      # ms/iter

print(f"SmallCNN: {benchmark_inference(small.to(device)):.2f} ms")
print(f"Your ResNet: {benchmark_inference(resnet.to(device)):.2f} ms")

# For ResNet-50, need 224×224 inputs (its conv strides assume bigger input)
pretrained_50 = models.resnet50(weights=None).to(device).eval()
def benchmark_resnet50():
    X = torch.randn(64, 3, 224, 224, device=device)
    for _ in range(5):
        with torch.inference_mode():
            pretrained_50(X)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        with torch.inference_mode():
            pretrained_50(X)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 50 * 1000

print(f"ResNet-50 (224×224 input): {benchmark_resnet50():.2f} ms")
```

You'll see the model-size vs latency trade-off in numbers. **This is the kind of measurement that drives architecture choices in production.** Week 15 turns this into a real profiling discipline.

---

## Submission checklist

- [ ] Hand-rolled `conv2d_naive` matches `F.conv2d` on a worked example
- [ ] Sobel/blur kernels applied to a CIFAR image — visible filtering effects
- [ ] CIFAR-10 loaded with augmentation; train/val/test split correctly
- [ ] `SmallCNN` trains to ~83-85% test accuracy
- [ ] `ResNet` with `BasicBlock` trains to **>90% test accuracy** in 40 epochs
- [ ] First-conv filters visualized — look like oriented edges
- [ ] Linear-probe transfer learning from pretrained ResNet-50 hits >92% in 3 epochs
- [ ] Receptive field of a deep neuron visualized
- [ ] Inference benchmarks reported for at least 3 models

---

## What you just did

You implemented convolution from scratch, built a CNN that beats your MLP on CIFAR-10 by 30+ accuracy points, built a ResNet block with skip connections that lets you train deeper without degradation, and demonstrated the single biggest lever in production vision ML: transfer learning from a pretrained backbone.

Every modern vision system — autonomous driving perception stacks, medical imaging diagnosis, satellite imagery analysis — sits on the foundation you just built. The next week (transformers) replaces the conv layers but keeps the residual connections, BatchNorm's cousin LayerNorm, and the training-loop discipline you've now spent 4 weeks internalizing.

---

**Next**: [Week 10: Transformers →](../week-10-transformers/readme.md)
