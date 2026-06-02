# Week 03: Theory — The Python ML Stack

You'll spend every remaining week of this curriculum inside numpy, pandas, matplotlib, and Jupyter. The difference between a fluent ML engineer and a slow one is roughly 80% mastery of these four tools. This week is about making the tools disappear so you can focus on the math.

---

## Part 1: Why numpy exists (and why pure Python is too slow)

A Python `list` is a list of pointers to PyObject instances. Each element has type info, ref counts, padding. Iterating over a list of 1M floats and adding them costs millions of pointer dereferences and type checks.

A numpy `ndarray` is one contiguous block of memory, holding raw bytes of a uniform `dtype`. Iterating internally is a tight C loop with vector instructions. The same 1M-element sum is 100-1000× faster.

```python
import numpy as np
import time

n = 10_000_000
py_list = list(range(n))
np_arr = np.arange(n)

t0 = time.perf_counter()
s1 = sum(py_list)
t1 = time.perf_counter()
s2 = np_arr.sum()
t2 = time.perf_counter()

print(f"python sum: {t1-t0:.3f}s")
print(f"numpy sum:  {t2-t1:.3f}s")
print(f"speedup:    {(t1-t0)/(t2-t1):.0f}×")
```

You'll usually see 50-200× speedup. That's the gap you're trying to live inside for the rest of the curriculum.

The rule: **anywhere you have a Python `for` loop over numbers, you have a bug or a missed vectorization.** Express it as array math instead.

---

## Part 2: Shape, dtype, and views (the three things that matter)

Every numpy array has:

```python
a = np.zeros((3, 4), dtype=np.float32)
print(a.shape)    # (3, 4) — tuple of axis sizes
print(a.dtype)    # float32 — element type
print(a.ndim)     # 2 — number of axes
print(a.size)     # 12 — total elements
print(a.strides)  # (16, 4) — bytes to step in each axis
```

### `dtype` — what the bits mean

The most common dtypes for ML:

| dtype | bytes | range |
|---|---|---|
| `float64` | 8 | ~1e-308 to 1e308 |
| `float32` | 4 | ~1e-38 to 1e38 — the default for deep learning |
| `float16` / `bfloat16` | 2 | smaller range; matters for mixed-precision (week 11) |
| `int64` | 8 | -9e18 to 9e18 |
| `int32` | 4 | -2e9 to 2e9 |
| `int8` | 1 | -128 to 127 — quantization (week 16) |
| `bool` | 1 | True / False |

The default for `np.array([1.0, 2.0])` is `float64`. The default for `torch.tensor([1.0, 2.0])` is `float32`. Mismatching dtypes between numpy and torch is a top-3 source of bugs.

### Views vs copies — when does numpy duplicate data?

Slicing returns a **view**, not a copy. The view shares memory with the original.

```python
a = np.array([1, 2, 3, 4, 5])
b = a[1:4]
b[0] = 99
print(a)   # [1, 99, 3, 4, 5] — a changed too!
```

The following create **views**: basic slicing (`a[1:5]`), reshape (`a.reshape(2,3)`), transpose (`a.T`).

The following create **copies**: fancy indexing (`a[[0, 2, 4]]`), boolean indexing (`a[a > 0]`), `np.copy(a)`, `a.copy()`.

**Bug pattern:** modifying a view when you thought you had a copy. **Fix pattern:** explicit `.copy()` when in doubt, and use `np.shares_memory(a, b)` to check.

---

## Part 3: Broadcasting — numpy's killer feature

Broadcasting is automatic alignment of arrays of different shapes when doing elementwise ops.

### The rules

Two shapes are compatible when, aligned **right to left**:

1. They are equal, OR
2. One of them is `1`

If any pair fails both, broadcasting fails.

```
(3, 4)
   (4,)         → (3, 4) ✓  (the (4,) broadcasts to (1, 4) then to (3, 4))

(3, 1, 4)
   (5, 4)       → (3, 5, 4) ✓

(3, 4)
   (5,)         → ERROR (4 ≠ 5)
```

### Why it matters for ML

The single most common ML pattern:

```python
# Batched linear layer: X @ W.T + b
X = np.random.randn(32, 784)    # 32 inputs, 784 features
W = np.random.randn(128, 784)   # 128 output units, 784 input
b = np.random.randn(128)        # one bias per unit

Y = X @ W.T + b                 # X@W.T is (32, 128); b broadcasts to (32, 128)
print(Y.shape)                  # (32, 128)
```

Without broadcasting, you'd have to manually tile `b` to `(32, 128)`. With broadcasting, it just works. Multiply that convenience by every layer in the curriculum.

### Inserting axes — `None` / `np.newaxis`

When you want to broadcast in a specific direction:

```python
v = np.array([1, 2, 3])
v[:, None]      # shape (3, 1) — column vector
v[None, :]      # shape (1, 3) — row vector

# Outer product via broadcasting
a = np.array([1, 2, 3])      # (3,)
b = np.array([10, 20, 30, 40])  # (4,)
outer = a[:, None] * b[None, :]   # (3, 1) × (1, 4) → (3, 4)
```

That trick comes up in every attention computation in week 10.

---

## Part 4: Axis-aware reductions

When you `.sum()` or `.mean()`, you usually want to reduce over a specific axis.

```python
X = np.array([[1, 2, 3],
              [4, 5, 6]])         # shape (2, 3)

X.sum()                 # 21 — sum over everything
X.sum(axis=0)           # [5, 7, 9] — sum down each column (drops axis 0)
X.sum(axis=1)           # [6, 15] — sum across each row (drops axis 1)
X.sum(axis=0, keepdims=True)   # shape (1, 3) — preserves axis for broadcasting
```

**Mnemonic:** `axis=k` "collapses axis k." Output has `ndim - 1` dimensions unless `keepdims=True`.

This matters in week 05 when you'll sum a batch's losses over the batch dimension to get a scalar loss.

---

## Part 5: Indexing — basic, fancy, boolean

```python
a = np.arange(20).reshape(4, 5)
# array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14],
#        [15, 16, 17, 18, 19]])

# Basic (view)
a[1, 2]              # 7 (scalar)
a[1:3, 0:2]          # 2x2 view
a[::2]               # every other row

# Fancy / integer (copy)
a[[0, 2, 3]]         # rows 0, 2, 3 stacked
a[[0, 2], [1, 4]]    # elements (0,1) and (2,4) — 1D output [1, 14]

# Boolean (copy)
mask = a > 10
a[mask]              # 1D array of all elements > 10

# Modify via boolean
a[a > 10] = 0        # in-place modification
```

Fancy + boolean indexing **always copy**, never view. Worth remembering when you wonder why a Python loop crept back into your perf profile.

---

## Part 6: pandas in one section

pandas is "numpy for tabular data." The two main types:

### `Series` — 1D array with labels

```python
import pandas as pd

s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
s['a']         # 10 — label-based
s.iloc[0]      # 10 — position-based
s.values       # array([10, 20, 30]) — underlying numpy
```

### `DataFrame` — 2D table of Series

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Carol'],
    'age': [30, 25, 35],
    'score': [88.5, 92.1, 75.0],
})

df['name']         # column → Series
df[['name', 'age']]   # subset → DataFrame
df.iloc[0]         # first row → Series
df.loc[0, 'age']   # 30 — label/position lookup
```

### The two access patterns to memorize

| You want | Use |
|---|---|
| Column by name | `df['col']` |
| Row by position | `df.iloc[i]` |
| Row by label | `df.loc[label]` |
| Element by row/col | `df.loc[row_label, col_label]` or `df.iloc[i, j]` |

### groupby — the workhorse

```python
# Average score per age group
df.groupby('age')['score'].mean()

# Multiple aggregations
df.groupby('age').agg({'score': ['mean', 'std', 'count']})

# Add the aggregate back to the original rows
df['mean_score_for_age'] = df.groupby('age')['score'].transform('mean')
```

**Mental model:** `groupby` splits the data into groups, applies an aggregation to each, combines the results. "Split-apply-combine." You'll use this for almost every dataset analysis.

### When to drop to numpy

pandas is built on numpy. `df.values` or `df.to_numpy()` gives you the underlying array. For heavy numerical work — training a model on tabular features — convert to numpy or torch once, then iterate.

---

## Part 7: matplotlib — figures, axes, and the rest

matplotlib has two APIs: a quick-script one (`plt.plot(...)`) and an object-oriented one (`fig, ax = plt.subplots()`). Use the object-oriented one for anything you'd want to revisit.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='sin')
ax.plot(x, np.cos(x), label='cos')

ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Sine and cosine")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("plot.png", dpi=150)
plt.show()
```

The hierarchy:

```
Figure  (the whole window)
└── Axes  (one subplot — coordinate system + axes labels + title)
    ├── Lines, Patches, Text, Image, etc.
    └── Spines, Ticks
```

The five lines you'll use 90% of the time: `subplots`, `plot`, `set_xlabel/ylabel/title`, `legend`, `savefig`.

For multi-panel:

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes[0, 0].plot(...)
axes[0, 1].imshow(...)
# etc.
fig.tight_layout()
```

For papers / portfolios, tweak the rcParams once at the top of the notebook:

```python
plt.rcParams.update({
    'figure.dpi': 100,
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
```

---

## Part 8: Reproducibility — uv, lockfiles, virtualenvs

The bane of every ML repo: "it works on my machine." The cure is a frozen environment.

### uv — the modern Python toolchain

```bash
# Install once
curl -LsSf https://astral.sh/uv/install.sh | sh

# Per-project workflow
uv init                          # creates pyproject.toml
uv add numpy pandas matplotlib   # adds + installs + locks
uv run jupyter notebook           # runs in the project env
uv sync                          # restore env from lockfile
```

`uv` is ~10-100× faster than `pip` and ~equivalent to Poetry in features. Adopt it now; you'll be glad in week 11 when you need to spin up an env on a rented GPU box and have 5 minutes.

The two files that define your env:

- `pyproject.toml` — declared dependencies (`numpy >= 2.0`)
- `uv.lock` — fully resolved versions (`numpy == 2.0.1`)

Commit both. The `uv.lock` makes installs reproducible across machines.

### Random seeds — the other half of reproducibility

```python
# numpy
rng = np.random.default_rng(seed=42)
rng.normal(0, 1, 100)            # use rng explicitly

# Python stdlib
import random
random.seed(42)

# PyTorch (week 07+)
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

This makes "I ran it again and got different numbers" go away. It does NOT make GPU CuDNN nondeterminism go away — that's a week 08 topic.

---

## Part 9: Jupyter discipline

Notebooks are great for exploration. They're a disaster for shipping. A few rules to keep them sane:

### Restart-and-run-all is the only valid notebook

A notebook that only works if you run cells in a specific weird order is broken. Once a week, do `Kernel → Restart → Run All` to catch hidden state.

### Use `%load_ext autoreload` for imports under development

```python
%load_ext autoreload
%autoreload 2
```

Now edits to your `.py` files are picked up on save without restarting the kernel.

### `%timeit` and `%%timeit` for benchmarks

```python
%timeit np.sum(np.arange(1_000_000))
# 100 loops, best of 5: 1.42 ms per loop

%%timeit
# multi-line block
x = np.random.randn(1000, 1000)
y = x @ x.T
```

### `%debug` after an exception

When you get a traceback, run `%debug` in the next cell to drop into a `pdb` prompt at the exact frame where the exception was raised. This is the single most valuable Jupyter feature for fixing bugs.

### Move stable code out of the notebook

Once a function is "done," move it into a `utils.py` in your project. The notebook becomes a thin orchestrator. This is how a notebook stops being a junk drawer.

---

## Part 10: What to skip (for now)

Topics that look adjacent but aren't on the path to weeks 05-16:

| Topic | When to revisit |
|---|---|
| Cython / Numba | Once you've tried vectorization and Triton and need C-speed Python |
| Dask, Ray Data | When data > RAM. Most curriculum projects fit in RAM. |
| Polars | Pandas alternative. Worth a look after the curriculum (mentioned in [week 16 Rust sidebar](../week-16-production-inference/readme.md)) |
| Plotly / seaborn | When you want interactive or fancy default styles. matplotlib first. |
| Conda | uv replaces it for most cases. Use conda only for non-Python deps like CUDA toolkit (week 13). |

---

## What's next

In [lab.md](lab.md) you'll:
- Set up an `uv` project from scratch with a lockfile
- Solve a battery of broadcasting puzzles (predict the output before running)
- Vectorize a slow Python loop and measure the speedup
- Do real pandas analysis on a small dataset
- Produce one publication-quality plot

If "broadcasting" feels mysterious by the end, **re-read Part 3**. The next 13 weeks lean on it.
