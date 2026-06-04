# Week 01: Theory — Math Foundations

This is the survival kit. Everything below is the *minimum* you need to read a deep-learning paper, understand `torch.nn.Linear`, and not get lost when someone writes `∂L/∂W` on a whiteboard. We're not aiming for a math degree — we're aiming for fluency.

---

## Part 1: Vectors and matrices as data

### What a vector actually is

A vector is just an ordered list of numbers. In ML, we'll usually treat it as:

- A **data point** — e.g. a 784-dimensional vector is one flattened 28×28 MNIST image
- A **feature** — e.g. the embedding of a word from a vocabulary
- A **parameter** — e.g. the weights of one neuron

Notation: a column vector `x` in n-dimensional real space, `x ∈ ℝⁿ`.

In numpy: `x = np.array([1.0, 2.0, 3.0])`. Shape: `(3,)`.

### What a matrix actually is

A matrix is a stack of vectors arranged in a grid. In ML we'll usually treat it as:

- A **batch of data points** — `X` of shape `(batch_size, num_features)`
- A **linear transformation** — `W` of shape `(out_features, in_features)`
- A **layer's parameters** — the same `W` rotated/reshaped depending on the layer type

In numpy: `W = np.array([[1, 2, 3], [4, 5, 6]])`. Shape: `(2, 3)`.

### Tensors

A tensor is just "an array with arbitrary number of axes." A scalar is a 0-D tensor; a vector is 1-D; a matrix is 2-D; a 3-channel image of size 224×224 is a 3-D tensor of shape `(3, 224, 224)`; a batch of 32 such images is a 4-D tensor of shape `(32, 3, 224, 224)`. PyTorch and numpy both use this vocabulary. Don't be intimidated by "tensor" — it's a generalized array.

---

## Part 2: Matrix multiplication (the most important operation in ML)

### The definition

Given `A` of shape `(m, k)` and `B` of shape `(k, n)`, the product `C = AB` is `(m, n)`, where entry `(i, j)` of `C` is the dot product of row `i` of `A` with column `j` of `B`:

```
C[i, j] = sum over l of A[i, l] * B[l, j]
```

### The shape rule (memorize this)

The inner dimensions must match:

```
(m, k) @ (k, n) → (m, n)
```

If they don't match, the multiply is undefined. This is the most common shape bug in deep learning.

```python
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C = A @ B           # (3, 5) ✓
D = np.random.randn(6, 4)
E = A @ D           # ValueError: shapes (3,4) and (6,4) not aligned
```

### Three useful ways to think about it

1. **As dot products** — every output entry is one dot product (the definition above).
2. **As linear combinations** — `Ax` (where `x` is a vector) is a linear combination of the columns of `A`, weighted by entries of `x`. This is the "data flows through a layer" view.
3. **As a function** — `A : ℝⁿ → ℝᵐ` is a linear map from one space to another. This is the "geometric transformation" view that makes 3Blue1Brown click.

All three views describe the same operation; you'll use different ones for different problems.

### Useful identities

| Identity | Form |
|---|---|
| Distributivity | `A(B + C) = AB + AC` |
| Associativity | `(AB)C = A(BC)` |
| NOT commutative | `AB ≠ BA` in general |
| Transpose of product | `(AB)ᵀ = BᵀAᵀ` |
| Identity matrix | `IA = A = AI` |
| Inverse | `A⁻¹A = I` (when it exists) |

The transpose rule and non-commutativity are the two you'll cite weekly.

---

## Part 3: A neural network is just matrices

A single linear layer:

```
y = W @ x + b
```

That's it. `W` is the matrix of weights, `b` is the bias vector, `x` is the input. PyTorch's `nn.Linear(in_features, out_features)` is literally these two parameters.

A two-layer MLP with ReLU activation:

```
h = ReLU(W1 @ x + b1)
y = W2 @ h + b2
```

Where `ReLU(z) = max(0, z)` applied elementwise. That's a complete two-layer MLP — you have everything to build it. (Which you will, in week 05.)

In code:

```python
import numpy as np

def mlp_forward(x, W1, b1, W2, b2):
    h = np.maximum(0, W1 @ x + b1)    # ReLU
    y = W2 @ h + b2
    return y
```

Eight lines. The rest of the curriculum is "what to put in the W's, how to find them, and how to do this fast on a GPU."

---

## Part 4: Derivatives — what changes when

### Single variable

The derivative of a function `f(x)` at a point `x` is the slope of `f` at that point. You only need to memorize a handful for ML:

| Function | Derivative |
|---|---|
| `c` (constant) | `0` |
| `x` | `1` |
| `xⁿ` | `n·xⁿ⁻¹` |
| `eˣ` | `eˣ` |
| `ln(x)` | `1/x` |
| `sin(x)` | `cos(x)` |
| `σ(x) = 1/(1+e⁻ˣ)` (sigmoid) | `σ(x)·(1−σ(x))` |
| `tanh(x)` | `1 − tanh²(x)` |
| `max(0, x)` (ReLU) | `1 if x > 0 else 0` |

The sigmoid and ReLU rows are the ones you'll cite the most. Note how clean ReLU's derivative is — that's a huge part of why deep nets work.

### The chain rule (the single most important page in this curriculum)

If `f(x) = g(h(x))`, then:

```
f'(x) = g'(h(x)) * h'(x)
```

In words: "outer derivative times inner derivative."

Why this matters: **a neural network is a composition of functions**. To get the gradient of the loss with respect to a weight in layer 1, we apply chain rule through every layer between layer 1 and the loss. That's backpropagation, fully derived. The rest of week 05 is just bookkeeping.

Example:

```
f(x) = (x² + 1)³
```

Let `h(x) = x² + 1`, `g(u) = u³`. Then `h'(x) = 2x`, `g'(u) = 3u²`, so:

```
f'(x) = 3·(x² + 1)² · 2x = 6x·(x² + 1)²
```

We'll repeat this pattern thousands of times this curriculum. Get comfortable.

### Multivariate derivatives — the gradient

When `f` takes a vector and returns a scalar (which is exactly the case for a loss function), the **gradient** is the vector of partial derivatives. Think of `∇f` as "the direction of steepest ascent" — moving `x` in the direction `+∇f` increases `f` fastest; moving in `-∇f` decreases it fastest. **That's gradient descent in one sentence.**

For `f(x, y) = x² + 3y`:

```
∂f/∂x = 2x
∂f/∂y = 3
∇f(x, y) = [2x, 3]
```

### The Jacobian — when output is also a vector

If `f : ℝⁿ → ℝᵐ` (takes a vector, returns a vector), the **Jacobian** is the matrix of all partials:

```
J[i, j] = ∂f_i / ∂x_j
```

Shape `(m, n)`. You'll see this when a layer's output flows into the next layer's loss — chain rule on vector-valued functions becomes a Jacobian-vector product (JVP) or vector-Jacobian product (VJP). PyTorch's autograd is, in essence, a fast way to compute VJPs.

---

## Part 5: Matrix calculus — the rules you actually need

When `L` is a scalar loss and `W` is a matrix of weights, the gradient `∂L/∂W` has the same shape as `W`. (Always check this. It's the easiest correctness check in matrix calculus.)

Three rules cover most of what you'll see in deep learning:

### Rule 1: Gradient of `Wx + b` with respect to `W`

If `y = Wx + b` and `L = ½‖y − t‖²` (the conventional ½ factor makes the 2 from differentiation cancel), then:

```
∂L/∂W = (y - t) @ xᵀ     # shape (out_dim, in_dim), same as W ✓
```

This is "the gradient of the weights equals the upstream error times the input."

> ⚠️ If your loss is `L = ‖y − t‖²` (no ½) instead — common in some textbooks — the gradient is `2(y - t) @ xᵀ`. A factor-of-2 mismatch between analytic and numerical-gradient checks usually comes from this. Pick a convention up front and stay with it.

### Rule 2: Gradient of `Wx + b` with respect to `x`

```
∂L/∂x = Wᵀ @ (∂L/∂y)
```

This is the one that "flows the gradient backwards through a layer."

### Rule 3: Gradient through an elementwise function

If `y = σ(z)` (elementwise sigmoid), then:

```
∂L/∂z = (∂L/∂y) ⊙ σ'(z)    # ⊙ is elementwise product
```

The same pattern holds for ReLU, tanh, GELU, etc.

These three rules + chain rule = full backpropagation. We'll derive each layer this way in week 05.

---

## Part 6: Numerical gradient checking (the bug-detector)

When you derive a gradient by hand, you can verify it numerically. For a tiny `h`:

```
∂f/∂x_i ≈ (f(x + h·eᵢ) - f(x - h·eᵢ)) / (2h)
```

(`eᵢ` is the unit vector in dimension `i`. The two-sided form has much lower numerical error than the one-sided form.)

In code:

```python
def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(x.size):
        plus, minus = x.copy(), x.copy()
        plus.flat[i] += h
        minus.flat[i] -= h
        grad.flat[i] = (f(plus) - f(minus)) / (2 * h)
    return grad
```

Your analytic gradient and numerical gradient should match to about `1e-6` for reasonable functions. If they don't, your hand derivation has a bug. **Use this. It will save you a week of debugging in week 05.**

---

## Part 7: Things you can safely defer

There are math topics you might think you need but really only matter for specific subareas. Don't let lack of background here stop you from moving forward:

| Topic | When you'll need it |
|---|---|
| Eigenvalues / SVD | PCA, recommender systems, some optimization theory — not until much later |
| Lagrange multipliers | Constrained optimization, SVMs — week 04 only briefly |
| Convex analysis | Optimization theory, not used directly when training NNs |
| Information theory (entropy, KL) | Week 02 covers what you need |
| Functional analysis | Theory of NTKs and similar — not in this curriculum |
| Real analysis (ε-δ proofs) | Never. Engineers don't need it. |

If a paper uses something you don't know, look it up *then*. Don't pre-load.

---

## Part 8: How to read papers with math

Three tactics:

1. **Read the figures first.** A good paper's figures are a tour of the idea. If you can't follow the figure caption, the body won't help.
2. **Match shapes.** When the paper writes `Q Kᵀ / √d_k`, write down the shape of `Q` (e.g. `(B, h, n, d_k)`), the shape of `Kᵀ`, the shape of the product. If the shapes work, you understand the operation.
3. **Skip the proofs on first read.** Most ML papers' "Theorem 1" is bounding something asymptotically — useful for theory work, not for understanding the method. Come back to proofs after you understand the method.

---

## What's next

In [lab.md](lab.md) you'll:
- Compute a matrix multiply by hand and check with numpy
- Derive a gradient by hand and verify numerically
- Build the forward pass of a single neuron from scratch — the building block for week 05

If any single section above felt opaque, **stop and rewatch the relevant 3Blue1Brown video** before moving on. The math compounds. Week 05 will be much harder if you skip this one.
