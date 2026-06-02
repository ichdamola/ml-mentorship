# Week 06: Theory — Build a Tiny Autograd Engine

In week 05 you wrote backward by hand for each layer. That's fine for a 2-layer MLP. It's hopeless for a 100-layer transformer with 4 attention heads, layer norms, residuals, and a dropout schedule. You'd be deriving gradients for the rest of your career.

The solution: **build the computation graph once during forward, then traverse it backwards mechanically**. This is autograd. Every deep-learning framework — PyTorch, TensorFlow, JAX — does this. By the end of this week you'll have written a 200-line autograd library yourself, and `loss.backward()` will never be opaque again.

---

## Part 1: The computation graph

Every mathematical expression is a tree (or, more generally, a directed acyclic graph). Consider:

```
e = (a + b) * (c + d)
```

As a graph:

```
        a   b     c   d
         \ /       \ /
          +         +
           \       /
            \     /
             \   /
              \ /
               *
               |
               e
```

Each **node** is a value. Each node tracks:

1. Its **value** (the number)
2. Its **parents** (the nodes it was computed from)
3. The **operation** that produced it (`+`, `*`, etc.)
4. A function for computing **local gradients** to each parent

The graph is **built during forward**. Every time you do `e = a + b`, you create a new node `e` whose parents are `a` and `b`.

---

## Part 2: Reverse-mode automatic differentiation

Two ways to compute derivatives through a graph:

### Forward mode

Start at the inputs, push derivatives forward. For each node, you maintain `∂node/∂x` for one specific input `x`. To get gradients for `d` inputs, you do `d` forward passes.

Good when **outputs > inputs**. Used in physics, optimization, classical ML with few parameters.

### Reverse mode

Start at the output, push derivatives backwards. For each node, you maintain `∂L/∂node`. One backward pass gives you the gradient with respect to **every** input simultaneously.

Good when **outputs < inputs**. In deep learning, output is a scalar loss and input is millions of weights — so reverse mode wins by a factor of `# parameters`. **This is what every ML autograd library does.**

### Chain rule on a graph

For node `e = f(parents)`, by chain rule:

```
∂L/∂parent = ∂L/∂e · ∂e/∂parent
```

So if you know `∂L/∂e` (the **upstream gradient**) and you can compute `∂e/∂parent` (the **local gradient**), you can pass gradient down to each parent.

The backward traversal:

1. Start at output node, set its gradient to `1.0` (i.e., `∂L/∂L = 1`)
2. Visit nodes in **reverse topological order**
3. For each node, compute and add to each parent's gradient using chain rule
4. Done — every input has its accumulated gradient in `.grad`

---

## Part 3: The `Value` class — autograd in 60 lines

Here's the entire idea, in scalar form (from [Karpathy's micrograd](https://github.com/karpathy/micrograd)):

```python
class Value:
    def __init__(self, data, _parents=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._parents = _parents     # tuple of Values this was computed from
        self._backward = lambda: None   # function that pushes grad to parents
        self._op = _op               # label for debugging

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # ∂(a+b)/∂a = 1,  ∂(a+b)/∂b = 1
            self.grad += out.grad           # += because grads accumulate over uses
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # ∂(a*b)/∂a = b,  ∂(a*b)/∂b = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # Topological sort
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for p in v._parents:
                    build_topo(p)
                topo.append(v)
        build_topo(self)

        # Initialize gradient at the root
        self.grad = 1.0

        # Visit in reverse topological order, calling each node's _backward
        for v in reversed(topo):
            v._backward()
```

That's enough to do:

```python
a = Value(2.0)
b = Value(3.0)
c = a * b + b   # = 9.0
c.backward()
print(a.grad)   # 3.0  (∂c/∂a = b = 3.0)
print(b.grad)   # 3.0  (∂c/∂b = a + 1 = 3.0)
```

**60 lines and you have working autograd.** Everything else is adding more operations.

### Why `+=` not `=` for gradients

A variable can appear multiple times in a computation:

```python
a = Value(2.0)
c = a * a   # ∂c/∂a = 2a = 4.0
```

When backward runs, `__mul__'s` `_backward` runs twice (once per parent — and both are `a`). Each accumulates `out.grad * other.data = 1 * 2 = 2` into `a.grad`. Final `a.grad = 4`. ✓

If we wrote `self.grad = ...` instead of `+=`, the second call would overwrite the first. Bug.

**Rule:** gradients always accumulate.

---

## Part 4: Adding more operations

Every operation needs a forward (returning a new `Value`) and a `_backward` (pushing gradients to parents).

Pattern:

```python
def __pow__(self, n):
    out = Value(self.data ** n, (self,), f'**{n}')

    def _backward():
        # ∂(a^n)/∂a = n * a^(n-1)
        self.grad += n * (self.data ** (n - 1)) * out.grad
    out._backward = _backward
    return out
```

Same pattern for `exp`, `log`, `relu`, `tanh`, `sigmoid`, divisions, etc. Each is one line of forward, one line of derivative.

Subtraction becomes `a + (-b)`; division becomes `a * b**(-1)`. Build everything else on top of `+`, `*`, `**`, and a unary `-`.

For a `ReLU`:

```python
def relu(self):
    out = Value(max(0, self.data), (self,), 'relu')

    def _backward():
        # ReLU'(x) = 1 if x > 0 else 0
        self.grad += (self.data > 0) * out.grad
    out._backward = _backward
    return out
```

---

## Part 5: Topological sort — why it's required

When backward runs, we must process **each node only after all its consumers** have already passed gradient to it. Otherwise some consumers run later and their contributions are lost.

A topological sort orders the graph so every node comes after all its parents. We then iterate in **reverse**, which means every node comes before all its parents.

The recursive DFS in `backward()` does this with the `visited` set guaranteeing each node is added to `topo` exactly once. The whole sort is `O(V + E)`.

---

## Part 6: From scalars to tensors

Scalar autograd is illustrative but unusably slow — every weight in your MLP would be a separate Python object. PyTorch is fast because it operates on **tensor-valued** nodes: each node holds a numpy-style array, and gradients flow through entire arrays at once.

The math is the same, just vectorized. For `Z = X @ W`:

- Local gradient at `W`: `∂L/∂W = Xᵀ @ (∂L/∂Z)`  ← week 05's matrix-calculus rule
- Local gradient at `X`: `∂L/∂X = (∂L/∂Z) @ Wᵀ`

You implement these as the `_backward` for matmul.

Broadcasting introduces a wrinkle. If forward did `Z = X + b` where `X` is `(B, K)` and `b` is `(K,)` (broadcast), the `_backward` for `b` must **sum over the broadcasted axes** to match `b`'s original shape:

```python
def _backward():
    self.grad += out.grad
    other.grad += out.grad.sum(axis=0) if out.grad.shape != other.data.shape else out.grad
```

This bookkeeping is annoying but mechanical. PyTorch handles it for you because every op encodes its broadcast pattern. In your toy implementation, you'll handle the common case (sum reduction back to original shape) and call it good.

---

## Part 7: How PyTorch's autograd actually works

PyTorch builds a graph of `Function` objects (not the user-facing `Tensor`s). Each `Function` records the inputs needed for backward in a saved-tensors list. When you call `.backward()`:

1. PyTorch traverses the graph in reverse topological order
2. For each `Function`, it calls `backward(ctx, grad_output)` to compute parent gradients
3. Parent `Tensor.grad` attributes are accumulated (`grad += ...`)
4. Graph nodes are freed unless you set `retain_graph=True`

The graph is **dynamic** — it's rebuilt every forward pass. This is "define by run." TensorFlow 1.x was "define then run," which was faster but more painful to debug. PyTorch + TF 2 + JAX all use dynamic graphs now.

A few PyTorch-specific quirks worth knowing:

| Quirk | What it does |
|---|---|
| `requires_grad=True` | Marks a tensor as a "leaf" the graph should track |
| `with torch.no_grad():` | Disables graph building inside the block — used for inference |
| `tensor.detach()` | Returns a copy outside the graph |
| `tensor.retain_grad()` | Keeps grad on a non-leaf tensor (usually freed) |
| `loss.backward(retain_graph=True)` | Don't free the graph — needed when calling backward multiple times |

You'll meet all of these in week 07.

---

## Part 8: Optimizations PyTorch makes that you won't

Your toy autograd is correct but slow. PyTorch optimizes:

- **In-place operations** (`x.add_(y)`) reuse memory
- **Fused kernels** (e.g. `addmm`) combine matmul + bias add into one CUDA call
- **Gradient checkpointing** trades compute for memory by re-running forward on backward
- **Mixed precision** (week 11) — forward in fp16, backward in fp32
- **`torch.compile`** generates a static computation graph from your dynamic code

You don't need any of this for your toy autograd. But knowing it exists clarifies what you're giving up when you don't reach for PyTorch.

---

## Part 9: What about JAX, Mojo, etc.?

Three modern alternatives worth knowing:

- **JAX** uses **traced functional autograd** — you write pure functions, `jax.grad` transforms them. No graph-building per-call; everything is JIT-compiled by XLA. Different mental model; same chain rule underneath.
- **Mojo** is a Python-like language with manual memory control. Very early; interesting if you do GPU work in the long term.
- **Triton** (week 14) compiles Python-flavored kernels to PTX. Not an autograd library, but a kernel-writing alternative.

For this curriculum we'll stay on PyTorch because it's still the dominant tool. The lessons transfer.

---

## Part 10: Why this week matters

By the end of this week you'll have:

- Written autograd from scratch — you understand what `loss.backward()` does
- Built a tiny `Neuron`, `Layer`, `MLP` on top of your `Value` class
- Trained an MLP using your own autograd — verifying it agrees with week 05's manual gradients

After this week, **the gap between you and the PyTorch source code closes by a few feet**. When you eventually read `torch/autograd/function.py`, you'll recognize every line. That's the point.

---

## What's next

In [lab.md](lab.md) you'll:
- Build the `Value` class with `__add__`, `__mul__`, `__pow__`, `relu`, `exp`, `log`
- Add the topological-sort backward pass
- Test every operation against numerical gradients
- Build a tiny `Neuron`, `Layer`, `MLP` API on top
- Train your own MLP on a 2D toy classification problem
- (Stretch) Extend to tensor-valued nodes with broadcasting

By the end you'll have a working autograd library you understand all the way down.
