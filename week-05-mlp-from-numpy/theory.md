# Week 05: Theory — MLP from Numpy

This is the week the abstract math becomes a working classifier. You'll derive the backward pass of every layer by hand, then implement an MLP that reaches >97% on MNIST in ~150 lines of pure numpy. After this week, "deep learning" is no longer magic — it's bookkeeping, applied carefully.

The single most important sentence in this curriculum lives here: **backpropagation is just chain rule, propagated through a computation graph layer by layer.** Everything you'll do in weeks 06-16 is automation of this.

---

## Part 1: The architecture

A multi-layer perceptron is a stack of (linear, activation) pairs:

```
x      → Linear₁(W₁, b₁) → z₁ → ReLU → h₁
       → Linear₂(W₂, b₂) → z₂ → ReLU → h₂
       ...
       → Linear_L(W_L, b_L) → z_L → softmax → ŷ
```

Two conventions you'll see for shapes — pick one and stick with it:

**Row-vector convention** (what we'll use): `x` is `(batch, in_dim)`, `W` is `(in_dim, out_dim)`, layer is `z = x @ W + b`. Matches sklearn and PyTorch.

**Column-vector convention** (textbooks): `x` is `(in_dim, batch)`, `W` is `(out_dim, in_dim)`, layer is `z = W @ x + b`. Matches week 01 math.

The math is identical. Throughout this week we use the row-vector convention because it's what your code will use.

---

## Part 2: The forward pass

Layer L:

```
zₗ = hₗ₋₁ @ Wₗ + bₗ        # linear
hₗ = ReLU(zₗ)              # activation
```

Loss for K-class classification with one-hot targets `y`:

```
ŷ = softmax(z_L)                       # the final layer's pre-activation, then softmax
L = -mean(log ŷ_{true_class}) over batch     # categorical NLL (cross-entropy)
```

For a two-layer MLP on MNIST (784 → 128 → 10):

```python
def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1               # (B, 128)
    h1 = np.maximum(0, z1)         # ReLU
    z2 = h1 @ W2 + b2              # (B, 10)
    # softmax + cross-entropy, fused below
    return z2, h1, z1
```

That's the entire forward. Eight lines. The hard part is everything below.

---

## Part 3: The backward pass — chain rule per layer

We want `∂L/∂Wₗ` and `∂L/∂bₗ` for every layer, so we can take a gradient-descent step.

Recall the three rules from [week 01](../week-01-math-foundations/theory.md):

```
y = x @ W + b
∂L/∂W = xᵀ @ (∂L/∂y)        # gradient at weights
∂L/∂x = (∂L/∂y) @ Wᵀ        # gradient flowing backwards
∂L/∂b = (∂L/∂y).sum(axis=0)  # bias gradient (sum over batch)
```

And for an elementwise activation:

```
h = ReLU(z)
∂L/∂z = (∂L/∂h) ⊙ ReLU'(z)    # ⊙ is elementwise; ReLU'(z) = 1 if z>0 else 0
```

These two patterns repeated through every layer = backpropagation.

### Softmax + cross-entropy gradient (the clean one)

For the last layer, instead of computing softmax and then cross-entropy separately (which is numerically unstable), we use the fused identity:

```
ŷ = softmax(z_L)
L = -log ŷ_{true}
∂L/∂z_L = ŷ - y_one_hot
```

That's it. One line. The `(ŷ - y)` cancellation is one of the prettiest results in ML — and the reason every deep-learning library fuses softmax with cross-entropy.

**Quick derivation** (worth doing once, by hand):

Let `s_k = exp(z_k) / Σⱼ exp(zⱼ)` (softmax). Cross-entropy:

```
L = -log s_{c}    where c is the true class
  = -z_c + log Σⱼ exp(zⱼ)
```

Take `∂L/∂z_k`:

```
∂L/∂z_k = -𝟙[k = c] + exp(z_k) / Σⱼ exp(zⱼ)
        = -𝟙[k = c] + s_k
        = s_k - 𝟙[k = c]
```

In vector form: `∂L/∂z_L = ŷ - y_one_hot`. Beautiful. Done.

### The full backward, layer by layer

For our two-layer MLP, working backwards from the loss:

```
# Start: ∂L/∂z2 = (ŷ - y_one_hot) / batch_size       (categorical NLL gradient)
dz2 = (softmax(z2) - y_one_hot) / B

# Layer 2 gradients
dW2 = h1.T @ dz2                    # shape (128, 10)
db2 = dz2.sum(axis=0)               # shape (10,)

# Flow gradient back through layer 2
dh1 = dz2 @ W2.T                    # shape (B, 128)

# Through ReLU
dz1 = dh1 * (z1 > 0)                # shape (B, 128) — ReLU' is the mask

# Layer 1 gradients
dW1 = X.T @ dz1                      # shape (784, 128)
db1 = dz1.sum(axis=0)               # shape (128,)
```

Twelve lines of arithmetic. Every neural net you'll ever train, whether 2 layers or 200, is this pattern repeated.

### The shape mnemonic

You can spot most bugs by looking at shapes:

| Quantity | Shape |
|---|---|
| `X` | `(B, in_dim)` |
| `Wₗ` | `(in_l, out_l)` |
| `bₗ` | `(out_l,)` |
| `zₗ`, `hₗ` | `(B, out_l)` |
| `dzₗ` | `(B, out_l)` — same as `zₗ` |
| `dWₗ` | `(in_l, out_l)` — same as `Wₗ` |
| `dbₗ` | `(out_l,)` — same as `bₗ` |
| `dhₗ₋₁` | `(B, in_l)` — same as `hₗ₋₁` |

**Every gradient has the same shape as the thing it's the gradient of.** Forget this and shape-mismatch errors will eat days.

---

## Part 4: The SGD update

After backward you have all the gradients. SGD updates:

```
W ← W - α · dW
b ← b - α · db
```

`α` is the learning rate. Pick it too high → loss diverges (NaN). Too low → loss decreases too slowly. Week 08 covers the disciplined way to find a good `α`; for week 05 we'll use `α = 0.1` and move on.

### Mini-batch

Instead of computing the gradient over the entire training set (slow, smooth) or one example at a time (fast, noisy), we use a **mini-batch**:

```
for epoch in range(N_epochs):
    for batch in iter_batches(X_train, y_train, batch_size=64):
        forward()
        backward()
        update()
```

Batch sizes between 32 and 512 are typical for MLPs. The optimal batch size is a function of GPU memory, learning rate, and convergence — week 11 has the full story.

---

## Part 5: Activations — ReLU and why it's the default

### Sigmoid (the old default)

```
σ(z) = 1 / (1 + e^-z)
σ'(z) = σ(z) · (1 - σ(z))
```

The maximum value of `σ'` is `0.25` (at `z=0`). For deep networks, gradients shrink layer by layer: `0.25 × 0.25 × 0.25 × ... → 0`. This is the **vanishing-gradient problem**. Deeper than ~3 layers and the lower layers don't learn.

### ReLU (the modern default)

```
ReLU(z) = max(0, z)
ReLU'(z) = 1 if z > 0 else 0
```

For positive `z`, the gradient is exactly `1` — no shrinkage. This unlocks training deep nets. Two trade-offs:

1. **Dead ReLU**: if a neuron's `z` is always negative, its gradient is always zero — it stops learning forever. Mitigated by good initialization and modest learning rates.
2. **Not differentiable at `z = 0`**: in practice we pick a convention (gradient = 0 at z = 0); it doesn't matter.

### Variants you'll see

| Activation | Formula | When |
|---|---|---|
| ReLU | `max(0, z)` | Default; almost always fine |
| Leaky ReLU | `max(0.01z, z)` | When dead ReLUs become a problem |
| GELU | `z · Φ(z)` | Transformers (week 10) |
| SiLU / Swish | `z · σ(z)` | Many modern LLMs |
| Tanh | `(e^z - e^-z) / (e^z + e^-z)` | Older models, gated RNNs |
| Sigmoid | `1/(1+e^-z)` | Output layer for binary classification only |

For week 05: ReLU everywhere except the final layer, which is followed by softmax via the cross-entropy loss.

---

## Part 6: Weight initialization — why it makes or breaks training

If `W` starts at zeros, every neuron in a layer is identical. They all receive the same gradient. They never differentiate. Training fails. **Initialization breaks the symmetry.**

If `W` starts too small, activations shrink to zero layer by layer; gradients vanish.
If `W` starts too large, activations explode; gradients explode; NaN.

The Goldilocks initialization preserves the magnitude of activations and gradients across layers.

### Xavier / Glorot initialization (tanh, sigmoid)

```
W ~ N(0, 2 / (fan_in + fan_out))
```

Designed so the variance of activations stays roughly constant moving forward, and the variance of gradients stays roughly constant moving backward. From [Glorot & Bengio (2010)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).

### He initialization (ReLU)

```
W ~ N(0, 2 / fan_in)
```

Because ReLU kills half the activations on average, we compensate by doubling the variance. From [He et al. (2015)](https://arxiv.org/abs/1502.01852). Use this for any ReLU-based network.

In code:

```python
W1 = rng.normal(0, np.sqrt(2 / 784), size=(784, 128))    # He init for 784→128
W2 = rng.normal(0, np.sqrt(2 / 128), size=(128, 10))     # He init for 128→10
b1 = np.zeros(128)                                        # biases start at zero
b2 = np.zeros(10)
```

**Get this wrong and your MLP will not train.** Test it: try `W = np.zeros(...)` and watch the loss curve do nothing.

---

## Part 7: Numerical stability — softmax and log-sum-exp

The naive softmax:

```python
def softmax_naive(z):
    return np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
```

For `z` with large entries (say `z = [1000, 0]`), `exp(1000)` overflows to `inf`. The fix is the **logsumexp trick** — subtract the max:

```python
def softmax_stable(z):
    z_shifted = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z_shifted)
    return e / e.sum(axis=-1, keepdims=True)
```

`softmax(z) = softmax(z - c)` for any constant `c`. Choosing `c = z.max()` makes the largest exponent equal to `exp(0) = 1`, avoiding overflow. Every production softmax does this. Build the habit now.

Same trick for the cross-entropy: `log(Σ exp(z)) = max(z) + log(Σ exp(z - max(z)))` — the famous "logsumexp" function, which is `scipy.special.logsumexp`.

---

## Part 8: One full training step, narrated

Putting it all together for a (784 → 128 → 10) MLP on MNIST:

```
Input batch:        X    shape (B, 784)
                    y    shape (B,) of integers in 0..9

Forward:
  z1 = X @ W1 + b1                                 (B, 128)
  h1 = relu(z1)                                    (B, 128)
  z2 = h1 @ W2 + b2                                 (B, 10)
  log_probs = log_softmax(z2)                       (B, 10)  ← numerically stable
  loss = -log_probs[arange(B), y].mean()           scalar

Backward (work right-to-left):
  dz2  = (softmax(z2) - one_hot(y, K=10)) / B      (B, 10)
  dW2  = h1.T @ dz2                                 (128, 10)
  db2  = dz2.sum(axis=0)                            (10,)
  dh1  = dz2 @ W2.T                                 (B, 128)
  dz1  = dh1 * (z1 > 0)                             (B, 128)  ← ReLU mask
  dW1  = X.T @ dz1                                  (784, 128)
  db1  = dz1.sum(axis=0)                            (128,)

Update:
  W1 -= lr * dW1
  b1 -= lr * db1
  W2 -= lr * dW2
  b2 -= lr * db2
```

Type this whole pattern in your sleep. The rest of the curriculum is about layers more interesting than `Linear + ReLU` — but the loop structure stays exactly this.

---

## Part 9: Debugging the backward pass

When your backward gives the wrong gradient, **gradient check it.** From week 01:

```python
def numerical_gradient(f, theta, h=1e-5):
    grad = np.zeros_like(theta)
    flat = theta.ravel()
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + h;  fp = f(theta)
        flat[i] = orig - h;  fm = f(theta)
        flat[i] = orig
        grad.flat[i] = (fp - fm) / (2 * h)
    return grad
```

For each weight `W`, compare `analytic_dW` (from your backward) with `numerical_dW = numerical_gradient(loss_fn, W)`. They should match to `~1e-6`. If they don't, your backward has a bug — usually a transpose or a missed `/B` or a missing axis sum.

**Do this. Every. Time.** Even Karpathy gradient-checks. Your future self will thank you.

---

## Part 10: What an MLP can and can't do

After this week you'll have a working MLP. Worth knowing its limits:

| Can | Can't |
|---|---|
| Classify MNIST → 97-98% | Classify CIFAR-10 well — needs spatial structure (CNN, week 09) |
| Tabular classification + regression | Variable-length sequences |
| Function approximation (universal approximator theorem) | Generalize to translations/rotations of inputs |

MLPs are universal approximators in theory but woefully sample-inefficient for images, sequences, and graphs. That's why weeks 09-12 introduce architectures with **inductive biases** (translation invariance for CNNs, sequence structure for transformers).

---

## What's next

In [lab.md](lab.md) you'll:
- Implement the forward and backward pass for a 2-layer MLP in pure numpy
- Gradient-check both layers to ~1e-6
- Train to >97% MNIST test accuracy in ~150 lines
- Visualize the learned first-layer filters (they look like digit fragments)
- (Stretch) Add a third layer and re-derive its gradient

If any step above feels fuzzy, **re-read Part 3** before opening the lab. The 12 lines of backward are the conceptual heart of the entire field.
