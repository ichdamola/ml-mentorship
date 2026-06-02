# Week 06: Lab — Build a Tiny Autograd Engine

By the end of this lab you'll have a working scalar autograd library — your own micrograd. Then you'll train an MLP on it. **Resist the temptation to peek at Karpathy's micrograd source until you've finished.** The point is building it yourself.

## Setup

Just numpy + matplotlib:

```python
import math
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(seed=42)
```

Open a new notebook `06_autograd.ipynb` (and a Python file `micrograd.py` for the engine code — easier to import into your training script).

---

## Exercise 6.1 — The `Value` class (forward only)

Start with a `Value` that holds a number, tracks parents, and supports addition and multiplication. No backward yet.

```python
# In micrograd.py
class Value:
    def __init__(self, data, _parents=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._parents = _parents
        self._backward = lambda: None
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), '*')

    # Operator support so a + b and b + a, a + 1.0, etc. all work
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __neg__(self):  return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return (-self) + other
```

Sanity-check:

```python
from micrograd import Value
a = Value(2.0)
b = Value(3.0)
print(a + b)       # data=5.0
print(a * b + 1)   # data=7.0
```

---

## Exercise 6.2 — Add `_backward` to operations

Now each op needs to know how to push gradients to its parents. Modify `__add__` and `__mul__`:

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    def _backward():
        self.grad  += out.grad        # ∂(a+b)/∂a = 1
        other.grad += out.grad        # ∂(a+b)/∂b = 1
    out._backward = _backward
    return out

def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
    def _backward():
        self.grad  += other.data * out.grad   # ∂(a*b)/∂a = b
        other.grad += self.data * out.grad    # ∂(a*b)/∂b = a
    out._backward = _backward
    return out
```

---

## Exercise 6.3 — Topological sort + `backward()`

Implement `backward` on `Value` — topo-sort the graph, then walk it in reverse calling each node's `_backward`:

```python
def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
        if id(v) in visited:
            return
        visited.add(id(v))
        for p in v._parents:
            build_topo(p)
        topo.append(v)

    build_topo(self)
    self.grad = 1.0
    for v in reversed(topo):
        v._backward()
```

(Use `id(v)` for the visited check because `Value`s aren't hashable by default.)

Test:

```python
a = Value(2.0)
b = Value(3.0)
c = a * b + b
c.backward()
print(f"a.grad = {a.grad}  (expected 3.0, = ∂(ab+b)/∂a = b)")
print(f"b.grad = {b.grad}  (expected 3.0, = ∂(ab+b)/∂b = a + 1)")
```

If those don't match, your topological sort or your operator gradients are off.

---

## Exercise 6.4 — Add more operations

Add support for `**` (power), `relu`, `exp`, and `log`:

```python
def __pow__(self, n):
    assert isinstance(n, (int, float))
    out = Value(self.data ** n, (self,), f'**{n}')
    def _backward():
        self.grad += n * (self.data ** (n - 1)) * out.grad
    out._backward = _backward
    return out

def relu(self):
    out = Value(max(0, self.data), (self,), 'relu')
    def _backward():
        self.grad += (self.data > 0) * out.grad
    out._backward = _backward
    return out

def exp(self):
    out = Value(math.exp(self.data), (self,), 'exp')
    def _backward():
        self.grad += out.data * out.grad      # d(e^x)/dx = e^x
    out._backward = _backward
    return out

def log(self):
    assert self.data > 0
    out = Value(math.log(self.data), (self,), 'log')
    def _backward():
        self.grad += (1.0 / self.data) * out.grad
    out._backward = _backward
    return out
```

Sanity-check with `relu`:

```python
x = Value(-1.0)
y = x.relu()
y.backward()
print(f"y = {y.data}, x.grad = {x.grad}")   # y=0, x.grad=0

x = Value(2.0)
y = x.relu()
y.backward()
print(f"y = {y.data}, x.grad = {x.grad}")   # y=2, x.grad=1
```

---

## Exercise 6.5 — Verify against numerical gradients

Comprehensive test: pick some expression, compute its analytic gradient via `backward()`, compute its numerical gradient via finite differences, compare.

```python
def numerical_grad(f, x, h=1e-5):
    """f: Value→Value function. x: float."""
    return (f(x + h) - f(x - h)) / (2 * h)

# Test a few combinations
def test_fn_1(x):
    return (x ** 2 + 3 * x + 1)        # ∂/∂x = 2x + 3

def test_fn_value(x):
    a = Value(x)
    out = a * a + a * 3 + 1
    return out

xs = [-2.0, -0.5, 0.7, 3.2]
for x in xs:
    out = test_fn_value(x)
    out.backward()
    a = out._parents[0]._parents[0]._parents[0]  # navigate back to `a` — or just rewrite using a known reference
    # Simpler: re-extract the original Value `a` from the graph by storing it:
    a = Value(x); out = a * a + a * 3 + 1
    a.grad = 0.0; out.backward()
    numeric = numerical_grad(test_fn_1, x)
    print(f"x = {x:>5.2f}   analytic = {a.grad:.6f}   numeric = {numeric:.6f}")
```

All should match to ~1e-6. If any are off, your op gradient is wrong.

---

## Exercise 6.6 — Build `Neuron`, `Layer`, `MLP` on top

Now use `Value` to build the higher-level abstractions. Same as PyTorch's `nn.Linear` + `nn.Sequential` — just rolled by hand.

```python
import random
random.seed(42)

class Neuron:
    def __init__(self, nin, nonlin=True):
        # He init: std = sqrt(2 / nin) for ReLU
        std = math.sqrt(2 / nin)
        self.w = [Value(random.gauss(0, std)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        # nouts is a list, e.g. [16, 16, 1] for input → 16 → 16 → 1
        sizes = [nin] + nouts
        self.layers = [
            Layer(sizes[i], sizes[i+1], nonlin=(i != len(nouts) - 1))
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

Notice that this whole API is ~40 lines and looks suspiciously similar to PyTorch — because PyTorch's `nn.Module` is the same idea.

---

## Exercise 6.7 — Train your MLP on a 2D classification toy

```python
# Make a "moons" dataset
from sklearn.datasets import make_moons
X_raw, y_raw = make_moons(n_samples=200, noise=0.1, random_state=42)
y_raw = y_raw * 2 - 1   # turn labels into {-1, +1} for hinge-style loss

# Build the model: 2 features → 16 → 16 → 1
model = MLP(2, [16, 16, 1])
print(f"#parameters = {len(model.parameters())}")

# Training loop
n_steps = 100
for step in range(n_steps):
    # Forward — compute MSE over the dataset (one big batch)
    total_loss = Value(0.0)
    for x_row, y_row in zip(X_raw, y_raw):
        x_vals = [Value(float(x)) for x in x_row]
        pred = model(x_vals)
        # Squared hinge loss: max(0, 1 - y * pred)**2
        margin = (Value(y_row) * pred * -1) + 1.0
        hinge = margin.relu()
        total_loss = total_loss + hinge * hinge

    # Zero grads
    for p in model.parameters():
        p.grad = 0.0

    # Backward
    total_loss.backward()

    # SGD step
    lr = 0.05
    for p in model.parameters():
        p.data -= lr * p.grad

    # Accuracy
    if step % 10 == 0:
        correct = 0
        for x_row, y_row in zip(X_raw, y_raw):
            x_vals = [Value(float(x)) for x in x_row]
            pred = model(x_vals)
            if (pred.data > 0) == (y_row > 0):
                correct += 1
        print(f"step {step:3d}  loss {total_loss.data:.4f}  acc {correct/len(X_raw):.4f}")
```

You should see accuracy climb to >95%. **You just trained a neural network using an autograd library you wrote yourself.**

---

## Exercise 6.8 — Visualize the decision boundary

```python
xx, yy = np.meshgrid(
    np.linspace(X_raw[:, 0].min() - 0.5, X_raw[:, 0].max() + 0.5, 100),
    np.linspace(X_raw[:, 1].min() - 0.5, X_raw[:, 1].max() + 0.5, 100),
)
grid = np.c_[xx.ravel(), yy.ravel()]
preds = np.array([
    1 if model([Value(float(g[0])), Value(float(g[1]))]).data > 0 else -1
    for g in grid
]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, preds, levels=20, alpha=0.4, cmap='RdBu')
plt.scatter(X_raw[y_raw==1, 0], X_raw[y_raw==1, 1], c='r', edgecolor='k')
plt.scatter(X_raw[y_raw==-1, 0], X_raw[y_raw==-1, 1], c='b', edgecolor='k')
plt.title("Decision boundary learned by your MLP")
plt.axis('equal')
plt.show()
```

The boundary should curve to follow the two moons. **A linear model can't do this (week 04); your MLP can — because it has non-linear activations.**

---

## Exercise 6.9 — Compare against PyTorch (sanity check)

```python
import torch

# Same model + data in PyTorch
torch.manual_seed(42)
X_t = torch.tensor(X_raw, dtype=torch.float32)
y_t = torch.tensor(y_raw, dtype=torch.float32).unsqueeze(1)

torch_model = torch.nn.Sequential(
    torch.nn.Linear(2, 16), torch.nn.ReLU(),
    torch.nn.Linear(16, 16), torch.nn.ReLU(),
    torch.nn.Linear(16, 1),
)

opt = torch.optim.SGD(torch_model.parameters(), lr=0.05)
for step in range(100):
    preds = torch_model(X_t)
    margins = 1 - y_t * preds
    loss = torch.clamp(margins, min=0).pow(2).sum()
    opt.zero_grad(); loss.backward(); opt.step()

# Final accuracy
with torch.no_grad():
    preds_t = torch_model(X_t).squeeze()
    acc = ((preds_t > 0) == (y_t.squeeze() > 0)).float().mean()
print(f"PyTorch accuracy: {acc:.4f}")
```

Both should reach similar accuracies. The difference is implementation speed: your version is slow because each `Value` is a Python object. PyTorch is fast because each tensor is a numpy array with C-level autograd.

---

## Exercise 6.10 (stretch) — Tensor-valued autograd

Generalize `Value` to `Tensor`: each node holds a numpy array, not a scalar. Implement `__matmul__` (with the matrix-calculus gradient rules from week 05), broadcasting-aware `+`, and `sum()`.

The tricky part is broadcasting: when a `(K,)` bias broadcasts to `(B, K)`, its `_backward` must sum over the batch axis to return a `(K,)` gradient. Pattern:

```python
def _backward():
    g = out.grad
    # Sum over axes that were broadcast for `other`
    if other.data.shape != g.shape:
        # ... reduce g to other's shape ...
    other.grad += g
```

Build this and you can train the week-05 MNIST MLP using your own tensor autograd. ~300 lines of code. A real achievement.

---

## Submission checklist

- [ ] `Value` class with `+`, `*`, `**`, `relu`, `exp`, `log` and accumulating `_backward`
- [ ] `backward()` does a correct topological sort and reverse traversal
- [ ] Gradient check passes for at least 3 different test expressions
- [ ] `Neuron` / `Layer` / `MLP` API built on top of `Value`
- [ ] MLP trains on the moons dataset to >95% accuracy
- [ ] Decision boundary visualization shows non-linear separation
- [ ] PyTorch comparison gives matching final accuracy
- [ ] (Stretch) `Tensor` autograd with broadcasting works on MNIST

---

## What you just did

You wrote autograd. **From scratch.** You'll never write `loss.backward()` again without knowing exactly what it does. Every operation tracks parents, every gradient is `+=`'d, topo-sort guarantees the order, and the chain rule is applied mechanically per node.

By next week (week 07) you'll switch to PyTorch — but every time `autograd` does something surprising, you'll be able to picture the graph that's being built behind the scenes. That mental model is what this week unlocked.

---

**Next**: [Week 07: PyTorch Fundamentals →](../week-07-pytorch-fundamentals/readme.md)
