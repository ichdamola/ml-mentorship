# Week 05: Lab — MLP from Numpy

By the end of this lab you'll have an MLP that classifies MNIST digits at >97% accuracy. **Type every line.** Don't paste. Your fingers learn what your eyes can't.

## Setup

```bash
uv add scikit-learn matplotlib
```

```python
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(seed=42)
np.set_printoptions(precision=4, suppress=True)
```

---

## Exercise 5.1 — Load MNIST

We'll use sklearn's `fetch_openml` so you don't need torchvision yet:

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_full = mnist.data.astype(np.float32) / 255.0       # (70000, 784), scaled to [0,1]
y_full = mnist.target.astype(int)                      # (70000,)

# Train/test split (the canonical MNIST split: first 60k train, last 10k test)
X_train, X_test = X_full[:60000], X_full[60000:]
y_train, y_test = y_full[:60000], y_full[60000:]

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"first label: {y_train[0]}")
```

Visualize a few digits to confirm:

```python
fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i, ax in enumerate(axes):
    ax.imshow(X_train[i].reshape(28, 28), cmap='gray')
    ax.set_title(str(y_train[i]))
    ax.axis('off')
plt.tight_layout()
plt.show()
```

---

## Exercise 5.2 — Stable softmax and cross-entropy

Implement the numerically stable versions from theory.md Part 7:

```python
def softmax(z):
    """z: (B, K). Returns probabilities."""
    z = z - z.max(axis=-1, keepdims=True)        # the stability trick
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def log_softmax(z):
    """Numerically stable log-softmax: log_softmax(z) = z - logsumexp(z)."""
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

def cross_entropy_loss(z, y):
    """z: (B, K) logits. y: (B,) integer labels. Returns scalar mean loss."""
    log_p = log_softmax(z)
    B = z.shape[0]
    return -log_p[np.arange(B), y].mean()
```

Quick sanity check:

```python
z = np.array([[1.0, 2.0, 3.0],
              [10.0, 0.0, -5.0]])
print(softmax(z).sum(axis=-1))                  # [1.0, 1.0]
print(cross_entropy_loss(z, np.array([2, 0])))  # ~0.40

# Stability check: this would overflow naive softmax
z_big = np.array([[1000.0, 999.0, 998.0]])
print(softmax(z_big))                            # finite, sane
```

---

## Exercise 5.3 — Initialize the network

He init for both layers, biases at zero:

```python
def init_params(rng):
    W1 = rng.normal(0, np.sqrt(2 / 784), size=(784, 128)).astype(np.float32)
    b1 = np.zeros(128, dtype=np.float32)
    W2 = rng.normal(0, np.sqrt(2 / 128), size=(128, 10)).astype(np.float32)
    b2 = np.zeros(10, dtype=np.float32)
    return W1, b1, W2, b2

W1, b1, W2, b2 = init_params(rng)
print(f"W1: {W1.shape}, range [{W1.min():.3f}, {W1.max():.3f}], std {W1.std():.3f}")
print(f"W2: {W2.shape}, range [{W2.min():.3f}, {W2.max():.3f}], std {W2.std():.3f}")
```

`W1.std()` should be around `0.05` (= `√(2/784) ≈ 0.0505`).

---

## Exercise 5.4 — Forward and backward pass

This is the heart of the lab. Type slowly. Check shapes against the mnemonic table in [theory.md Part 3](theory.md).

```python
def forward(X, W1, b1, W2, b2):
    """Returns the loss components needed for backward."""
    z1 = X @ W1 + b1                  # (B, 128)
    h1 = np.maximum(0, z1)            # ReLU
    z2 = h1 @ W2 + b2                 # (B, 10)
    return z1, h1, z2

def backward(X, y, z1, h1, z2, W2):
    """Returns gradients (dW1, db1, dW2, db2)."""
    B = X.shape[0]

    # Softmax + cross-entropy gradient (the fused identity from theory Part 3)
    y_one_hot = np.zeros_like(z2)
    y_one_hot[np.arange(B), y] = 1
    dz2 = (softmax(z2) - y_one_hot) / B    # (B, 10)

    # Layer 2 grads
    dW2 = h1.T @ dz2                       # (128, 10)
    db2 = dz2.sum(axis=0)                  # (10,)

    # Flow back through layer 2
    dh1 = dz2 @ W2.T                       # (B, 128)

    # Through ReLU
    dz1 = dh1 * (z1 > 0)                   # (B, 128)

    # Layer 1 grads
    dW1 = X.T @ dz1                         # (784, 128)
    db1 = dz1.sum(axis=0)                  # (128,)

    return dW1, db1, dW2, db2
```

---

## Exercise 5.5 — Gradient check the backward pass

This is non-optional. **Do this before training.** It catches the bug you don't know you have.

```python
def loss_of(W1, b1, W2, b2, X, y):
    z1, h1, z2 = forward(X, W1, b1, W2, b2)
    return cross_entropy_loss(z2, y)

def numerical_gradient(W, idx, X, y, W1, b1, W2, b2, h=1e-4):
    """Centered numerical gradient at the single entry idx of W."""
    orig = W.flat[idx]
    W.flat[idx] = orig + h
    L_plus = loss_of(W1, b1, W2, b2, X, y)
    W.flat[idx] = orig - h
    L_minus = loss_of(W1, b1, W2, b2, X, y)
    W.flat[idx] = orig
    return (L_plus - L_minus) / (2 * h)

# Use a small subset so this finishes fast
X_small = X_train[:64]
y_small = y_train[:64]

W1, b1, W2, b2 = init_params(rng)
z1, h1, z2 = forward(X_small, W1, b1, W2, b2)
dW1, db1, dW2, db2 = backward(X_small, y_small, z1, h1, z2, W2)

# Spot-check 5 random weights of W2
print(f"{'i':>4} {'analytic':>12} {'numeric':>12} {'rel_err':>12}")
for _ in range(5):
    idx = rng.integers(0, W2.size)
    num = numerical_gradient(W2, idx, X_small, y_small, W1, b1, W2, b2)
    ana = dW2.flat[idx]
    rel_err = abs(num - ana) / max(abs(num), abs(ana), 1e-12)
    print(f"{idx:>4} {ana:>12.6g} {num:>12.6g} {rel_err:>12.2e}")
```

**Expected:** `rel_err` ≤ `1e-4` for every entry. If any are off by more than that, your backward has a bug — most commonly a missing `/B` or a transpose error. Fix before going further.

---

## Exercise 5.6 — Mini-batch training loop

Now train. Track train loss + test accuracy each epoch.

```python
def evaluate(X, y, W1, b1, W2, b2):
    z1, h1, z2 = forward(X, W1, b1, W2, b2)
    preds = z2.argmax(axis=-1)
    return (preds == y).mean()

# Reset params, train for real
W1, b1, W2, b2 = init_params(rng)

lr = 0.1
batch_size = 64
n_epochs = 10
N_train = len(X_train)

train_losses = []
test_accs = []

for epoch in range(n_epochs):
    perm = rng.permutation(N_train)
    epoch_losses = []
    for i in range(0, N_train, batch_size):
        idx = perm[i:i+batch_size]
        Xb, yb = X_train[idx], y_train[idx]

        z1, h1, z2 = forward(Xb, W1, b1, W2, b2)
        loss = cross_entropy_loss(z2, yb)
        dW1, db1, dW2, db2 = backward(Xb, yb, z1, h1, z2, W2)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        epoch_losses.append(loss)

    train_losses.append(np.mean(epoch_losses))
    test_acc = evaluate(X_test, y_test, W1, b1, W2, b2)
    test_accs.append(test_acc)
    print(f"epoch {epoch:2d}: loss {train_losses[-1]:.4f}  test acc {test_acc:.4f}")
```

You should see test accuracy reach **>97%** by epoch 10.

---

## Exercise 5.7 — Plot learning curves

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_losses)
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss")
axes[0].set_title("Training loss")
axes[0].grid(alpha=0.3)

axes[1].plot(test_accs)
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("test accuracy")
axes[1].set_title("Test accuracy")
axes[1].set_ylim(0.9, 1.0)
axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
```

A healthy training curve: train loss decreases smoothly; test accuracy rises and plateaus.

---

## Exercise 5.8 — Visualize the first-layer filters

Each column of `W1` is the weight vector of one hidden neuron — reshaped to 28×28, it shows what input pattern activates that neuron. For a trained MLP these look like fragments of digits.

```python
fig, axes = plt.subplots(8, 16, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    if i >= W1.shape[1]:
        ax.axis('off'); continue
    filt = W1[:, i].reshape(28, 28)
    ax.imshow(filt, cmap='gray')
    ax.axis('off')
plt.suptitle("Learned first-layer filters")
plt.show()
```

You should see digit fragments — pen-stroke detectors, curves, edges. Compare to the random-init filters from before training; the difference is the model "understanding" digits.

---

## Exercise 5.9 — Confusion matrix and worst mistakes

Look at where your model is wrong:

```python
from sklearn.metrics import confusion_matrix, classification_report

z1, h1, z2 = forward(X_test, W1, b1, W2, b2)
preds = z2.argmax(axis=-1)
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))

# Find the most confidently wrong predictions
probs = softmax(z2)
wrong = np.where(preds != y_test)[0]
confidences = probs[wrong, preds[wrong]]
worst = wrong[np.argsort(confidences)[::-1][:10]]    # top-10 most confident wrong

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    idx = worst[i]
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f"true:{y_test[idx]}  pred:{preds[idx]}  ({confidences[i]:.2f})")
    ax.axis('off')
plt.tight_layout(); plt.show()
```

Several will be ambiguous digits even to a human (a `4` shaped like a `9`, etc.). That's the irreducible-noise floor — no model fits beyond it.

---

## Exercise 5.10 (stretch) — Add a third layer

Add a hidden layer of 64 units: 784 → 128 → 64 → 10. Re-derive `dz1_5` for the new middle layer and update the backward function. Train and check that test accuracy is similar (≥97%) — going deeper doesn't help much for MNIST, but the discipline of deriving the gradients is the point.

<details>
<summary>Solution sketch</summary>

```python
# Forward: extra layer
z1 = X @ W1 + b1; h1 = relu(z1)
z2 = h1 @ W2 + b2; h2 = relu(z2)
z3 = h2 @ W3 + b3

# Backward (right to left):
dz3 = (softmax(z3) - one_hot) / B
dW3 = h2.T @ dz3;  db3 = dz3.sum(0)
dh2 = dz3 @ W3.T
dz2 = dh2 * (z2 > 0)
dW2 = h1.T @ dz2;  db2 = dz2.sum(0)
dh1 = dz2 @ W2.T
dz1 = dh1 * (z1 > 0)
dW1 = X.T @ dz1;   db1 = dz1.sum(0)
```

Same pattern, one extra rung in the chain. Gradient-check it.
</details>

---

## Submission checklist

- [ ] MNIST loaded and visualized
- [ ] Stable softmax / cross-entropy implemented
- [ ] Forward + backward pass implemented for a 784→128→10 MLP
- [ ] Gradient check passes (rel_err ≤ 1e-4) on at least 5 random weights
- [ ] Training reaches **>97% test accuracy** in 10 epochs
- [ ] Learning curves plotted; the loss curve is monotone-ish
- [ ] First-layer filters visualized; they look like pen-stroke detectors, not noise
- [ ] Confusion matrix + worst-mistakes visualization done
- [ ] (Stretch) 3-layer variant gradient-checks and trains

---

## What you just did

You implemented backpropagation. Not "ran an autograd library that does it" — derived it by hand, coded it in pure numpy, verified the gradients against finite differences, and trained a classifier that works.

Everything in weeks 06-16 — autograd engines, PyTorch, transformers, CUDA kernels — is automating what you just did manually. You'll never read "loss.backward()" the same way again.

---

**Next**: [Week 06: Autograd Engine →](../week-06-autograd-engine/readme.md)
