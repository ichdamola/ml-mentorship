# Week 01: Lab — Math Foundations

Type every line. Don't copy-paste — your fingers learn the math. By the end, every code block here should run cold in a fresh Jupyter notebook.

## Setup

```bash
mkdir ml-mentorship-work && cd ml-mentorship-work
uv init
uv add numpy matplotlib jupyter
uv run jupyter notebook
```

Create a new notebook called `01_math_foundations.ipynb`.

```python
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)
```

---

## Exercise 1.1 — Vectors and shapes

```python
v = np.array([1.0, 2.0, 3.0])
print(v.shape)        # (3,)
print(v.dtype)        # float64

# Reshape to a column vector (3, 1) and a row vector (1, 3)
col = v.reshape(3, 1)
row = v.reshape(1, 3)
print(col.shape, row.shape)

# Stack into a matrix
A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape)        # (2, 3)
```

**Check yourself:** what's the shape of `A.T`? What about `A.flatten()`? Write your answer in a comment, then check.

---

## Exercise 1.2 — Matrix multiplication by hand

Compute the following by hand on paper, then verify with numpy:

```
A = [[1, 2],            B = [[5],
     [3, 4]]                  [6]]

A @ B = ?
```

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5], [6]])
print(A @ B)
# Expected: [[17], [39]]
#   row 0: 1*5 + 2*6 = 17
#   row 1: 3*5 + 4*6 = 39
```

Now the same with the @ in the other order — note the shape mismatch:

```python
try:
    print(B @ A)
except ValueError as e:
    print("expected failure:", e)
```

**Question:** what's the shape of `B.T @ A`? Predict before running.

---

## Exercise 1.3 — Broadcasting

Numpy's killer feature: when shapes don't match exactly but one dimension is `1`, it auto-stretches.

```python
X = np.array([[1, 2, 3],
              [4, 5, 6]])              # (2, 3)
b = np.array([10, 20, 30])             # (3,)

print(X + b)                           # b broadcasts to (2, 3)
# [[11, 22, 33],
#  [14, 25, 36]]
```

This is how `Wx + b` works in a real layer — the bias is `(out_dim,)` and broadcasts across the batch dimension.

```python
# Batch multiply
W = np.random.randn(5, 3)              # (out=5, in=3)
X_batch = np.random.randn(32, 3)        # batch of 32 inputs

# We want output (32, 5). Try:
Y = X_batch @ W.T + np.zeros(5)         # shape: (32, 5)
print(Y.shape)
```

**Question:** what's the order of magnitude of `Y.mean()`? Predict, then check. (It should be around 0 — why?)

---

## Exercise 1.4 — Derivatives by hand

Compute the derivatives by hand, then verify numerically:

| Function | Your answer | 
|---|---|
| `f(x) = 3x² + 2x − 1` | `f'(x) = ` |
| `g(x) = e^(−x²)` | `g'(x) = ` |
| `h(x) = sin(x²)` | `h'(x) = ` |
| `σ(x) = 1/(1+e^(−x))` | `σ'(x) = ` |

For `σ`, the answer is `σ(x)·(1−σ(x))` — a beautiful identity worth memorizing. Prove it by writing `σ(x) = (1 + e^(−x))^(−1)` and applying chain rule.

Now check numerically:

```python
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x): return 3*x**2 + 2*x - 1
def g(x): return np.exp(-x**2)
def sigma(x): return 1 / (1 + np.exp(-x))

x0 = 2.0
print(f"f'(2) numerical: {numerical_derivative(f, x0):.6f}, analytic: {6*x0 + 2}")
print(f"g'(2) numerical: {numerical_derivative(g, x0):.6f}, analytic: {-2*x0*np.exp(-x0**2):.6f}")
print(f"σ'(2) numerical: {numerical_derivative(sigma, x0):.6f}, analytic: {sigma(x0)*(1-sigma(x0)):.6f}")
```

All three should match to ~1e-7. **If yours don't match, redo the derivation.**

---

## Exercise 1.5 — Gradients of multivariate functions

Compute the gradient of `f(x, y) = x²y + 3y²` by hand:

- `∂f/∂x = ?`
- `∂f/∂y = ?`

Verify:

```python
def f(p):
    x, y = p
    return x**2 * y + 3 * y**2

def numerical_gradient(f, p, h=1e-5):
    grad = np.zeros_like(p)
    for i in range(len(p)):
        plus, minus = p.copy(), p.copy()
        plus[i] += h
        minus[i] -= h
        grad[i] = (f(plus) - f(minus)) / (2*h)
    return grad

p = np.array([1.5, 2.0])
print("numerical:", numerical_gradient(f, p))
# Analytic: [2xy, x² + 6y] = [6.0, 14.25]
```

---

## Exercise 1.6 — Gradient descent in 2D

The "look it falls down a hill" demo. Use gradient descent to find the minimum of `f(x, y) = x² + 3y²`:

```python
def f(p): return p[0]**2 + 3*p[1]**2
def grad_f(p): return np.array([2*p[0], 6*p[1]])

p = np.array([4.0, 4.0])    # start point
lr = 0.1
trajectory = [p.copy()]

for step in range(50):
    p = p - lr * grad_f(p)
    trajectory.append(p.copy())

trajectory = np.array(trajectory)
print(f"Final point: {p}")
print(f"Final loss:  {f(p):.6f}")

# Plot the descent
xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
zz = xx**2 + 3*yy**2
plt.contour(xx, yy, zz, levels=20)
plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-o', markersize=3)
plt.title("Gradient descent on f(x, y) = x² + 3y²")
plt.xlabel("x"); plt.ylabel("y")
plt.axis("equal")
plt.show()
```

**Observe:** the trajectory takes "stairsteps" — it zigzags down the steeper y-axis. This is exactly the kind of pathology that motivates momentum-based optimizers like Adam (week 07-08).

**Question:** what happens if you set `lr = 0.5`? `lr = 1.0`? Try both. Learning-rate sensitivity is a curse you'll spend the whole curriculum learning to manage.

---

## Exercise 1.7 — A single neuron (the building block of week 05)

A single neuron is just `y = σ(w·x + b)`, where `σ` is sigmoid. Implement it:

```python
def neuron_forward(x, w, b):
    """x: (n,), w: (n,), b: scalar. Returns scalar."""
    z = np.dot(w, x) + b
    return 1 / (1 + np.exp(-z))     # sigmoid

# Test: a 3-input neuron
x = np.array([1.0, 2.0, 3.0])
w = np.array([0.5, -0.3, 0.1])
b = -0.2

y = neuron_forward(x, w, b)
print(f"y = {y:.4f}")    # ~0.5499
```

Now derive `∂y/∂w` by hand. (Hint: chain rule. `y = σ(z)`, `z = w·x + b`. So `∂y/∂w = σ'(z) · ∂z/∂w = σ(z)(1 - σ(z)) · x`.) Verify:

```python
def numerical_grad_w(x, w, b, h=1e-5):
    grad = np.zeros_like(w)
    for i in range(len(w)):
        wp, wm = w.copy(), w.copy()
        wp[i] += h; wm[i] -= h
        grad[i] = (neuron_forward(x, wp, b) - neuron_forward(x, wm, b)) / (2*h)
    return grad

analytic = y * (1 - y) * x
numeric = numerical_grad_w(x, w, b)
print("analytic:", analytic)
print("numeric: ", numeric)
print("max abs diff:", np.max(np.abs(analytic - numeric)))   # should be ~1e-8
```

**This is the gradient of one neuron's output with respect to its weights** — the atom of backpropagation. In week 05 you'll do this same exercise but for an entire MLP, layer by layer.

---

## Exercise 1.8 — Verifying the MLP forward pass

Put it together. Build a 2-layer MLP with random weights and verify the shapes flow:

```python
def relu(z): return np.maximum(0, z)

# Architecture: 4 inputs → 8 hidden → 2 outputs
W1 = np.random.randn(8, 4) * 0.1
b1 = np.zeros(8)
W2 = np.random.randn(2, 8) * 0.1
b2 = np.zeros(2)

def mlp_forward(x):
    h = relu(W1 @ x + b1)
    y = W2 @ h + b2
    return y

x = np.random.randn(4)
y = mlp_forward(x)
print(f"input shape: {x.shape}, output shape: {y.shape}")
# (4,) → (2,)
```

Now batch it: pass 32 inputs at once.

```python
X = np.random.randn(32, 4)        # batch of 32 inputs

def mlp_forward_batch(X):
    # Shapes: X is (B, in_dim), W1 is (out_dim, in_dim).
    # We want H of shape (B, hidden_dim).
    # So: H = ReLU(X @ W1.T + b1)
    H = relu(X @ W1.T + b1)         # broadcasting b1 across batch dim
    Y = H @ W2.T + b2
    return Y

Y = mlp_forward_batch(X)
print(f"Y shape: {Y.shape}")        # (32, 2)
```

**Predict the shapes at every step** before running. If your prediction is wrong, that's the bug to fix.

---

## Submission checklist

- [ ] Exercises 1.1-1.8 completed in a notebook
- [ ] All `numerical vs analytic` comparisons match to ≤ 1e-6
- [ ] The gradient-descent plot from 1.6 reproduces the zig-zag trajectory
- [ ] You can articulate, without notes, why `∂L/∂W` has the same shape as `W`
- [ ] You can articulate, without notes, why ReLU's derivative is a step function

---

## What you just did

You built — by hand and by numpy — the forward pass and a single neuron's gradient. In week 02 you'll add probability (the language losses are written in). In week 05 you'll generalize this to a full MLP with backprop. By week 07 you'll be writing this in PyTorch in 10 lines, but you'll *understand* every one of those lines because of what you did this week.

---

**Next**: [Week 02: Probability & Statistics →](../week-02-probability-stats/readme.md)
