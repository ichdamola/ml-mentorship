# Week 03: Lab — The Python ML Stack

Drills. The point isn't to write clever code; it's to get fast at the boring operations so they stop slowing you down. By the end of this lab the patterns should feel automatic.

## Setup

If you haven't yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Initialize a project:

```bash
mkdir ml-mentorship-work && cd ml-mentorship-work
uv init
uv add numpy pandas matplotlib jupyter ipykernel scikit-learn
uv run jupyter notebook
```

Verify the lockfile and pyproject:

```bash
cat pyproject.toml | head -10
ls -la uv.lock
```

Both should exist. Commit both to your repo.

In a new notebook `03_python_ml_stack.ipynb`:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%load_ext autoreload
%autoreload 2

rng = np.random.default_rng(seed=42)
np.set_printoptions(precision=4, suppress=True)
```

---

## Exercise 3.1 — Broadcasting puzzles

**Predict the output shape before running each cell.** If you're wrong, re-read the broadcasting rules in [theory.md](theory.md) Part 3 before moving on.

```python
a = np.zeros((3, 4))
b = np.zeros((4,))
print("a + b:", (a + b).shape)            # predict?

c = np.zeros((3, 4))
d = np.zeros((3, 1))
print("c + d:", (c + d).shape)            # predict?

e = np.zeros((3, 1, 5))
f = np.zeros((7, 5))
print("e + f:", (e + f).shape)            # predict?

g = np.zeros((3, 4))
h = np.zeros((4, 3))
try:
    print("g + h:", (g + h).shape)
except ValueError as err:
    print("ValueError:", err)             # explain why
```

**Bonus puzzle.** Write a single line that turns vectors `a = np.array([1,2,3])` and `b = np.array([10,20,30,40])` into an outer-product matrix of shape `(3, 4)`.

<details>
<summary>Solution</summary>

```python
outer = a[:, None] * b[None, :]
```
</details>

---

## Exercise 3.2 — Vectorize a slow loop

Here's a deliberately bad implementation of "compute pairwise Euclidean distances between two sets of points":

```python
def slow_pairwise(A, B):
    """A: (n, d), B: (m, d). Returns D: (n, m) with D[i,j] = ||A[i] - B[j]||."""
    n = A.shape[0]; m = B.shape[0]
    D = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            diff = A[i] - B[j]
            D[i, j] = np.sqrt((diff ** 2).sum())
    return D
```

Write `fast_pairwise(A, B)` that does the same thing in pure numpy with no Python loops. Hint: `A[:, None, :] - B[None, :, :]` is the (n, m, d) tensor of all pairwise differences.

<details>
<summary>Solution</summary>

```python
def fast_pairwise(A, B):
    diffs = A[:, None, :] - B[None, :, :]    # (n, m, d)
    return np.sqrt((diffs ** 2).sum(axis=-1))
```
</details>

Verify correctness and measure speedup:

```python
A = rng.standard_normal((500, 16))
B = rng.standard_normal((500, 16))

D_slow = slow_pairwise(A, B)
D_fast = fast_pairwise(A, B)
print(f"max abs diff: {np.abs(D_slow - D_fast).max():.2e}")    # ~1e-14

%timeit slow_pairwise(A, B)
%timeit fast_pairwise(A, B)
```

You should see ~100-1000× speedup. **This is the gap you live in for the rest of the curriculum.** Every time you find yourself writing a `for` over numbers, ask: "can I broadcast this?"

---

## Exercise 3.3 — Axis-aware reductions

For a batch of 32 RGB images of shape `(32, 3, 64, 64)`:

```python
imgs = rng.standard_normal((32, 3, 64, 64))

# Per-image mean (one number per image)
# Predict output shape, then verify.
per_image_mean = imgs.mean(axis=(1, 2, 3))
print("per_image_mean:", per_image_mean.shape)           # predict?

# Per-channel mean across the whole batch (one number per RGB channel)
per_channel_mean = imgs.mean(axis=(0, 2, 3))
print("per_channel_mean:", per_channel_mean.shape)       # predict?

# Per-pixel mean across the batch and channels — one (64, 64) "average image"
avg_image = imgs.mean(axis=(0, 1))
print("avg_image:", avg_image.shape)                     # predict?
```

The "per-channel mean" pattern is exactly what you do to normalize images by ImageNet stats:

```python
imagenet_mean = np.array([0.485, 0.456, 0.406])    # (3,)
imagenet_std = np.array([0.229, 0.224, 0.225])     # (3,)

# Normalize a single (3, 224, 224) image
normalized = (imgs[0] - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
print("normalized.shape:", normalized.shape)        # (3, 64, 64)
```

You'll do this in week 09 with `torchvision.transforms.Normalize`. Knowing what's happening underneath is the point of week 03.

---

## Exercise 3.4 — Views and copies

Predict which of these modifies the original.

```python
a = np.arange(20).reshape(4, 5)
print("a before:")
print(a)

# (a) slicing
v1 = a[1:3]
v1[:] = 99
print("\nafter slicing modification:")
print(a)        # which rows changed?

# Reset
a = np.arange(20).reshape(4, 5)

# (b) fancy indexing
v2 = a[[0, 2]]
v2[:] = 99
print("\nafter fancy-index modification:")
print(a)        # which rows changed?

# Reset
a = np.arange(20).reshape(4, 5)

# (c) boolean indexing assignment
a[a % 2 == 0] = -1
print("\nafter boolean assignment:")
print(a)
```

**Rule of thumb:**
- Basic slicing → view (modifies original)
- Fancy/boolean indexing on the right of `=` → copy
- Boolean indexing on the left of `=` (assignment) → modifies original

---

## Exercise 3.5 — pandas with a real dataset

Download or generate a small dataset. We'll generate one so the exercise is self-contained:

```python
# Synthetic "users" table
N = 1000
users = pd.DataFrame({
    'user_id': np.arange(N),
    'age': rng.integers(18, 80, N),
    'country': rng.choice(['US', 'UK', 'NG', 'IN', 'BR'], N, p=[0.3, 0.15, 0.2, 0.25, 0.1]),
    'signup_year': rng.integers(2018, 2027, N),
    'lifetime_value': rng.exponential(100, N).round(2),
})
users.head()
```

### Q1 — Average lifetime value per country, sorted desc

```python
avg_ltv = users.groupby('country')['lifetime_value'].mean().sort_values(ascending=False)
print(avg_ltv)
```

### Q2 — How many users per country × signup_year?

```python
crosstab = pd.crosstab(users['country'], users['signup_year'])
print(crosstab)
```

### Q3 — Add a "tier" column based on lifetime_value bucket

```python
users['tier'] = pd.cut(
    users['lifetime_value'],
    bins=[0, 50, 200, np.inf],
    labels=['low', 'mid', 'high'],
)
print(users[['user_id', 'lifetime_value', 'tier']].head())
print("\ntier counts:")
print(users['tier'].value_counts())
```

### Q4 — Pull a numpy array of features for ML

```python
# Numeric features only
feature_cols = ['age', 'signup_year', 'lifetime_value']
X = users[feature_cols].to_numpy()           # (1000, 3)
print(X.shape, X.dtype)

# One-hot encode country
country_dummies = pd.get_dummies(users['country'], prefix='country')
print(country_dummies.head())

# Concatenate to a feature matrix
features = pd.concat([users[feature_cols], country_dummies], axis=1).to_numpy()
print("final feature matrix:", features.shape)
```

This `features` array is in the exact form you'd pass to a sklearn or PyTorch model in week 04+.

---

## Exercise 3.6 — One publication-quality plot

Make a single plot that:
1. Shows two quantities on the same axes (different colors)
2. Has axis labels with units
3. Has a title
4. Has a legend
5. Has grid lines at low alpha
6. Saves to a 300dpi PNG

```python
# Synthetic training curves
epochs = np.arange(1, 51)
train_loss = 2.5 * np.exp(-epochs / 10) + 0.3 + rng.normal(0, 0.05, len(epochs))
val_loss = 2.5 * np.exp(-epochs / 12) + 0.4 + rng.normal(0, 0.08, len(epochs))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, train_loss, label='train', linewidth=2)
ax.plot(epochs, val_loss, label='validation', linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (NLL)")
ax.set_title("Training curves — toy model")
ax.legend()
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig("training_curves.png", dpi=300)
plt.show()
```

Open the saved PNG. Could you drop this into a slide or paper? If not, what's missing?

---

## Exercise 3.7 — Save and load `.npz`

`.npz` is the standard format for "I have several numpy arrays I want to ship together." Use it for caching preprocessed datasets.

```python
np.savez('cache.npz',
         X=features,
         y=users['tier'].cat.codes.to_numpy(),
         columns=np.array(list(country_dummies.columns)))

# Later, in a different cell or notebook:
data = np.load('cache.npz', allow_pickle=False)
print(data.files)              # ['X', 'y', 'columns']
print(data['X'].shape)          # (1000, 8) or so
```

Use this between weeks. Don't recompute when you can load.

---

## Exercise 3.8 — `%timeit` discipline

Train yourself to measure, not guess. Pick three implementations of the same operation and benchmark them:

```python
# Three ways to sum the squares of an array
x = rng.standard_normal(100_000)

%timeit sum(xi**2 for xi in x)            # pure python
%timeit (x ** 2).sum()                    # vectorized
%timeit np.einsum('i,i->', x, x)          # einsum
%timeit x @ x                             # dot product
```

What did you predict? What did you measure? The top three should all be in the same ballpark; the python generator should be 100-1000× slower.

**Lesson:** even among "vectorized" options, the choice between `(x**2).sum()` and `x @ x` matters slightly because the former allocates an intermediate array. `%timeit` is how you find out.

---

## Submission checklist

- [ ] An `uv`-managed project with `pyproject.toml` + `uv.lock` committed
- [ ] All broadcasting puzzles solved without rerunning to check
- [ ] `fast_pairwise` matches `slow_pairwise` to ~1e-14 with a 100×+ speedup
- [ ] Pandas exercises produce the right output without help
- [ ] One publication-quality plot saved to a 300dpi PNG
- [ ] `cache.npz` saved and loaded round-trip
- [ ] You ran `%timeit` on at least three implementations of the same operation

---

## What you just did

You now have the daily-driver fluency to keep up with the rest of the curriculum without fighting the tools. From here on, every week assumes you can:

- Vectorize a Python loop without thinking about it
- Read/write a DataFrame and groupby
- Plot anything in 5 lines
- Install a deterministic env on a new machine

If any of those still feel slow, **redo this lab**. The tools don't get easier in week 11 just because you've moved on.

---

**Next**: [Week 04: Classical ML →](../week-04-classical-ml/readme.md)
