# Week 02: Lab — Probability & Statistics for ML

Hands-on. Every cell should run; every numerical claim is verifiable. Open a notebook called `02_probability.ipynb` from the same venv as week 01.

## Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
rng = np.random.default_rng(seed=42)   # reproducible
np.set_printoptions(precision=4, suppress=True)
```

If you don't have `scipy`:

```bash
uv add scipy
```

---

## Exercise 2.1 — Sampling and the law of large numbers

The law of large numbers (LLN) says: the sample average of i.i.d. draws converges to the true mean. Let's see it.

```python
# True mean of a Bernoulli(0.3) is 0.3
true_p = 0.3
sample_sizes = [10, 100, 1_000, 10_000, 100_000]
for n in sample_sizes:
    samples = rng.binomial(1, true_p, size=n)
    print(f"n={n:>7d}, mean={samples.mean():.4f}, error from true={abs(samples.mean()-true_p):.4f}")
```

**Watch:** the error shrinks as `n` grows. Specifically, the standard error of the sample mean is `σ/√n` — so to halve the error, you need 4× the data.

### Visualize convergence

```python
# Running average of 10,000 coin flips
N = 10_000
samples = rng.binomial(1, true_p, size=N)
running_mean = np.cumsum(samples) / np.arange(1, N+1)

plt.figure(figsize=(10, 4))
plt.plot(running_mean)
plt.axhline(true_p, color='red', linestyle='--', label=f'true p = {true_p}')
plt.xlabel("number of samples")
plt.ylabel("running mean")
plt.legend()
plt.title("Law of Large Numbers — running mean → true mean")
plt.show()
```

---

## Exercise 2.2 — Distributions, side by side

Sample 10,000 points from each of the canonical distributions and plot histograms.

```python
N = 10_000
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Gaussian
axes[0,0].hist(rng.normal(0, 1, N), bins=50)
axes[0,0].set_title("Standard Gaussian N(0,1)")

# Uniform
axes[0,1].hist(rng.uniform(-1, 1, N), bins=50)
axes[0,1].set_title("Uniform(-1, 1)")

# Exponential
axes[0,2].hist(rng.exponential(1.0, N), bins=50)
axes[0,2].set_title("Exponential(λ=1)")

# Bernoulli
axes[1,0].hist(rng.binomial(1, 0.3, N), bins=2)
axes[1,0].set_title("Bernoulli(p=0.3)")

# Poisson
axes[1,1].hist(rng.poisson(3, N), bins=15)
axes[1,1].set_title("Poisson(λ=3)")

# Categorical (5 classes, unequal probs)
probs = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
axes[1,2].hist(rng.choice(5, N, p=probs), bins=5)
axes[1,2].set_title(f"Categorical, probs={probs}")

plt.tight_layout()
plt.show()
```

**Observe:** the shapes match the math. The Gaussian is bell-shaped. The exponential decays. The categorical is a histogram of integers.

---

## Exercise 2.3 — Central Limit Theorem in action

CLT says: averages of i.i.d. random variables become Gaussian, regardless of source.

```python
# Average 30 uniform(0,1) draws, repeated 10,000 times.
# Each "averaged-30" value should look approximately Gaussian.
N_repeats = 10_000
n_per_avg = 30
averages = rng.uniform(0, 1, size=(N_repeats, n_per_avg)).mean(axis=1)

plt.figure(figsize=(10, 4))
plt.hist(averages, bins=50, density=True, alpha=0.7, label="empirical")

# Overlay the theoretical Gaussian — mean 0.5, std σ/√n where σ² = 1/12 (uniform variance)
mu, sigma = 0.5, np.sqrt(1/12) / np.sqrt(n_per_avg)
xx = np.linspace(averages.min(), averages.max(), 200)
plt.plot(xx, stats.norm.pdf(xx, mu, sigma), 'r-', linewidth=2, label="N(μ, σ²/n)")
plt.legend()
plt.title(f"CLT: average of {n_per_avg} Uniform(0,1) draws → Gaussian")
plt.show()
```

The empirical histogram and the theoretical Gaussian overlap. This is why "noise on minibatch gradients is Gaussian" — minibatch gradients are averages.

---

## Exercise 2.4 — Bayes' rule on a medical test

The classic example. A disease has prevalence 1% in the population. A test for it has:
- **Sensitivity:** `P(positive | disease) = 0.99` (99% true positive rate)
- **Specificity:** `P(negative | no disease) = 0.95` (5% false positive rate)

**Question:** If you test positive, what's the probability you actually have the disease?

Most people guess ~99% (matching sensitivity). Let's compute:

```
P(disease | positive) = P(positive | disease) · P(disease) / P(positive)
                       = 0.99 · 0.01 / [0.99 · 0.01 + 0.05 · 0.99]
                       = 0.0099 / 0.0594
                       ≈ 0.167
```

About **17%**. Verify:

```python
sensitivity = 0.99   # P(pos | disease)
specificity = 0.95   # P(neg | no disease)
prevalence = 0.01    # P(disease)

p_pos_given_disease = sensitivity
p_pos_given_no_disease = 1 - specificity
p_disease = prevalence
p_no_disease = 1 - prevalence

p_pos = p_pos_given_disease * p_disease + p_pos_given_no_disease * p_no_disease
p_disease_given_pos = p_pos_given_disease * p_disease / p_pos

print(f"P(disease | positive) = {p_disease_given_pos:.4f}")     # 0.1667
```

**Lesson:** when the base rate is low, a positive test from an imperfect test is often *not* strong evidence. This is the math behind "follow-up tests for rare conditions." It's also why you should be suspicious of "the model said yes" without knowing the model's precision on your distribution.

---

## Exercise 2.5 — Expected value and variance from samples

```python
# Sample 100,000 from N(5, 2²)
samples = rng.normal(loc=5.0, scale=2.0, size=100_000)

print(f"sample mean:      {samples.mean():.4f}        (true: 5.0)")
print(f"sample variance:  {samples.var():.4f}        (true: 4.0)")
print(f"sample std:       {samples.std():.4f}        (true: 2.0)")

# Biased vs unbiased variance — note the divisor
biased = samples.var(ddof=0)        # divide by N
unbiased = samples.var(ddof=1)      # divide by N-1 (Bessel)
print(f"biased (÷N):      {biased:.6f}")
print(f"unbiased (÷N-1):  {unbiased:.6f}")
```

For N = 100,000 the difference is tiny. For small N it's significant — that's why tests like `t-tests` use the N-1 divisor.

---

## Exercise 2.6 — Covariance and correlation

```python
# Generate correlated 2D data
N = 1000
true_corr = 0.7
X = rng.normal(0, 1, N)
Y = true_corr * X + np.sqrt(1 - true_corr**2) * rng.normal(0, 1, N)

# Empirical
cov_matrix = np.cov(X, Y)
corr_matrix = np.corrcoef(X, Y)

print(f"Covariance matrix:\n{cov_matrix}\n")
print(f"Correlation matrix:\n{corr_matrix}")
print(f"\nEmpirical correlation: {corr_matrix[0,1]:.4f} (true: {true_corr})")

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(X, Y, alpha=0.3, s=10)
plt.xlabel("X"); plt.ylabel("Y")
plt.title(f"Correlated data (ρ ≈ {corr_matrix[0,1]:.2f})")
plt.axis("equal")
plt.show()
```

**Question:** what would `Y = -X + noise` look like in the scatter plot? Predict, then re-run with `true_corr = -0.7`.

---

## Exercise 2.7 — MLE for a Gaussian by hand, verified

Derive the MLE for `μ` and `σ²` of a Gaussian (you did `μ` in theory.md; do `σ²` here).

For Gaussian `N(μ, σ²)`:

```
NLL(μ, σ²) = (N/2)·log(2πσ²) + (1/(2σ²)) Σᵢ (xᵢ - μ)²

∂NLL/∂σ² = N/(2σ²) - (1/(2σ⁴)) Σᵢ (xᵢ - μ)²
         = 0  →  σ̂² = (1/N) Σᵢ (xᵢ - μ̂)²
```

That's the biased estimator. Now verify with code:

```python
true_mu, true_sigma = 3.0, 1.5
N = 1000
samples = rng.normal(true_mu, true_sigma, N)

# MLE estimators
mu_hat = samples.mean()
sigma2_hat = ((samples - mu_hat) ** 2).mean()   # divisor N (MLE / biased)
sigma_hat = np.sqrt(sigma2_hat)

print(f"MLE μ̂ = {mu_hat:.4f}  (true: {true_mu})")
print(f"MLE σ̂² = {sigma2_hat:.4f}  (true σ² = {true_sigma**2})  ← the biased estimator")
print(f"MLE σ̂  = {sigma_hat:.4f}  (true σ  = {true_sigma})")
```

---

## Exercise 2.8 — MSE = Gaussian NLL (numerical proof)

This is the punchline of the week. Show that minimizing MSE on a regression problem is identical to minimizing the negative log-likelihood under a Gaussian noise model.

```python
# Synthetic 1D regression: y = 2x + 1 + ε, ε ~ N(0, 0.5²)
N = 200
true_w, true_b = 2.0, 1.0
true_sigma = 0.5
x = rng.uniform(-3, 3, N)
y = true_w * x + true_b + rng.normal(0, true_sigma, N)

# Fit by minimizing two different objectives:
# (a) MSE
# (b) NLL under N(μ=wx+b, σ²)
# They should give the same w, b.

def mse_loss(w, b):
    return np.mean((y - (w*x + b)) ** 2)

def gaussian_nll(w, b, sigma=true_sigma):
    residuals = y - (w*x + b)
    # NLL of N(0, σ²) for each residual:
    return 0.5 * np.log(2 * np.pi * sigma**2) * N + np.sum(residuals**2) / (2 * sigma**2)

# Grid search both
ws = np.linspace(0, 4, 100)
bs = np.linspace(-1, 3, 100)
W, B = np.meshgrid(ws, bs)

mse_grid = np.vectorize(mse_loss)(W, B)
nll_grid = np.vectorize(gaussian_nll)(W, B)

# The argmin of MSE and the argmin of NLL should be the same point
mse_min_idx = np.unravel_index(mse_grid.argmin(), mse_grid.shape)
nll_min_idx = np.unravel_index(nll_grid.argmin(), nll_grid.shape)

print(f"MSE argmin: w={ws[mse_min_idx[1]]:.4f}, b={bs[mse_min_idx[0]]:.4f}")
print(f"NLL argmin: w={ws[nll_min_idx[1]]:.4f}, b={bs[nll_min_idx[0]]:.4f}")
print(f"\nTrue:       w={true_w}, b={true_b}")
```

The two `argmin`s match exactly. **MSE loss == Gaussian NLL up to a constant.** That's why both methods recover the same `w, b`.

Now plot the two loss landscapes side by side:

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].contour(W, B, mse_grid, levels=20)
axes[0].set_title("MSE landscape")
axes[0].set_xlabel("w"); axes[0].set_ylabel("b")
axes[0].plot(true_w, true_b, 'r*', markersize=15, label="true")
axes[0].legend()

axes[1].contour(W, B, nll_grid, levels=20)
axes[1].set_title("Gaussian NLL landscape")
axes[1].set_xlabel("w"); axes[1].set_ylabel("b")
axes[1].plot(true_w, true_b, 'r*', markersize=15)

plt.show()
```

**Same shape, different vertical scale.** They are the same optimization problem.

---

## Exercise 2.9 — Cross-entropy / KL divergence by hand

Implement cross-entropy and KL divergence; verify on a categorical example.

```python
def cross_entropy(p, q, eps=1e-12):
    """H(p, q) = -Σ p(x) log q(x). p is the truth, q is the prediction."""
    q = np.clip(q, eps, 1 - eps)
    return -np.sum(p * np.log(q))

def kl_divergence(p, q, eps=1e-12):
    """KL(p || q) = Σ p(x) log(p(x) / q(x))."""
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return np.sum(p * np.log(p / q))

# Truth distribution (one-hot on class 2 out of 5)
p_true = np.array([0.0, 0.0, 1.0, 0.0, 0.0])

# A confident prediction (good)
q_good = np.array([0.05, 0.05, 0.8, 0.05, 0.05])

# A wrong-but-confident prediction (bad)
q_bad = np.array([0.8, 0.05, 0.05, 0.05, 0.05])

print(f"Cross-entropy (good): {cross_entropy(p_true, q_good):.4f}")    # ~0.223 (small)
print(f"Cross-entropy (bad):  {cross_entropy(p_true, q_bad):.4f}")     # ~2.996 (big)
print(f"\nKL(p_true || q_good): {kl_divergence(p_true, q_good):.4f}")
print(f"KL(p_true || q_bad):  {kl_divergence(p_true, q_bad):.4f}")
```

**Observation:** when `p_true` is one-hot, `H(p_true, q) = -log q(true class)`. That's exactly the `nn.CrossEntropyLoss` formula. Cross-entropy = NLL of the categorical = "how unconfident is the model about the right answer."

---

## Submission checklist

- [ ] Exercises 2.1-2.9 completed in a notebook
- [ ] You can articulate, without notes, why MSE corresponds to a Gaussian noise model
- [ ] You can articulate, without notes, why cross-entropy is just `-log p(correct class)`
- [ ] You ran the Bayes' rule calculation and understand why a 99%-sensitive test can produce ~17% posteriors at low base rates
- [ ] You verified empirically that the CLT makes averages Gaussian

---

## What you just did

You unified "loss functions in deep learning" under one idea: every loss is the negative log-likelihood of some distribution. That's the conceptual move that makes the rest of the curriculum coherent. By week 10 (transformers), the loss for next-token prediction will just be categorical NLL over the vocabulary — and you'll recognize it as the same operation you just coded by hand.

---

**Next**: [Week 03: Python ML Stack →](../week-03-python-ml-stack/readme.md)
