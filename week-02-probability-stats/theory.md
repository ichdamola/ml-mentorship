# Week 02: Theory — Probability & Statistics for ML

If week 01 gave you the calculus to compute gradients, this week gives you the language to write down what you're computing the gradient *of*. Every loss function in deep learning is a probability statement in disguise. Once you see that, the choice of "MSE for regression, cross-entropy for classification" stops being a rule to memorize and starts being a consequence of what distribution you assumed.

---

## Part 1: Probability — the minimum vocabulary

### Sample space, events, probability

- A **sample space** `Ω` is the set of all possible outcomes of an experiment. (`{H, T}` for a coin flip; `{1, 2, 3, 4, 5, 6}` for a die.)
- An **event** is a subset of `Ω`. ("Even number on the die" = `{2, 4, 6}`.)
- A **probability** is a function that assigns numbers in `[0, 1]` to events, with `P(Ω) = 1` and additivity for disjoint events.

For ML purposes you almost never reason about Ω directly. What you do reason about is **random variables**.

### Random variables

A random variable `X` is a function from `Ω` to a numeric value. "What number came up on the die" is a random variable.

Two flavors:

| Type | Values | Described by |
|---|---|---|
| **Discrete** | Countable (`{0, 1}`, `{1, 2, ..., 6}`, the integers) | Probability mass function (PMF), `p(x) = P(X = x)` |
| **Continuous** | Uncountable (real line, an interval) | Probability density function (PDF), `p(x)`, with `P(a ≤ X ≤ b) = ∫p(x)dx` |

**Critical distinction:** for continuous variables, `p(x)` is a *density*, not a probability. `p(x) > 1` is fine. `P(X = exact value) = 0` for any specific point. This trips everyone up the first time.

### The distributions you'll see all week

| Distribution | Type | Use |
|---|---|---|
| **Bernoulli(p)** | Discrete | One coin flip — `P(X=1)=p`, `P(X=0)=1-p` |
| **Categorical(p₁, ..., p_K)** | Discrete | One die roll over K classes — what classification predicts |
| **Binomial(n, p)** | Discrete | Sum of n Bernoullis |
| **Gaussian / Normal N(μ, σ²)** | Continuous | The default for continuous things; regression's noise model |
| **Uniform(a, b)** | Continuous | Equal probability over `[a, b]` |
| **Exponential(λ)** | Continuous | Waiting time until next event |
| **Poisson(λ)** | Discrete | Number of events in fixed interval |

**Memorize these PDFs/PMFs:**

| Distribution | Formula |
|---|---|
| Bernoulli(p) | `p^x · (1-p)^(1-x)` for x ∈ {0,1} |
| Categorical(p₁,...,p_K) | `p_x` for x ∈ {1,...,K} (a "one-hot pick") |
| Gaussian N(μ, σ²) | `(1/√(2πσ²)) · exp(-(x-μ)²/(2σ²))` |

That's enough to derive every ML loss function this curriculum will hit.

### Joint, marginal, conditional

For two random variables `X` and `Y`:

- **Joint:** `p(x, y)` — probability of both being a particular value
- **Marginal:** `p(x) = Σ_y p(x, y)` (discrete) or `∫p(x,y)dy` (continuous) — sum out the other variable
- **Conditional:** `p(x | y) = p(x, y) / p(y)` — probability of X given that Y took a value

Independence: `X` and `Y` are independent iff `p(x, y) = p(x) · p(y)`. Equivalently, `p(x | y) = p(x)` — knowing Y tells you nothing about X.

### Bayes' theorem — the formula every Bayesian wears as a tattoo

From the definition of conditional probability:

```
p(x | y) = p(y | x) · p(x) / p(y)
```

In ML phrasing:

```
posterior = (likelihood × prior) / evidence
```

Where:
- `p(x | y)` is the **posterior** — what you believe about `x` after seeing `y`
- `p(y | x)` is the **likelihood** — how probable `y` would be if `x` were the truth
- `p(x)` is the **prior** — what you believed about `x` before
- `p(y) = ∫p(y|x)p(x)dx` is the **evidence** (just a normalizer)

Bayes' rule is the engine of:
- Bayesian classification (week 04)
- MAP estimation (this week, below)
- Variational autoencoders (much later)
- Most of probabilistic ML

---

## Part 2: Expectations, variance, covariance

### Expected value

The mean of `X` under distribution `p`:

```
E[X] = Σ_x x · p(x)        (discrete)
E[X] = ∫ x · p(x) dx        (continuous)
```

Properties you'll use weekly:
- `E[a·X + b] = a·E[X] + b` (linearity)
- `E[X + Y] = E[X] + E[Y]` (always, even when X, Y are not independent)
- `E[X·Y] = E[X]·E[Y]` only if X and Y are independent

### Variance and standard deviation

```
Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²
σ(X) = √Var(X)
```

`Var(X)` has the unit of `X²`; `σ(X)` has the unit of `X`. You'll always quote σ in human-facing reports.

### Covariance — when two variables co-vary

```
Cov(X, Y) = E[(X - E[X])(Y - E[Y])]
```

Positive when X and Y move together, negative when they move opposite, zero when uncorrelated. The **correlation coefficient** is just the normalized version:

```
ρ(X, Y) = Cov(X, Y) / (σ(X)·σ(Y))   ∈ [-1, 1]
```

### Why this matters for ML

- **Loss decomposition** ("bias-variance" — week 04) is variance arithmetic
- **The Adam optimizer** tracks running means and variances of gradients
- **Batch normalization** uses per-batch means and variances
- **Whitening** of features uses the covariance matrix

You'll be calling `.mean()` and `.var()` constantly. Know what they compute.

---

## Part 3: Maximum Likelihood Estimation (MLE)

### The setup

You have data `D = {x₁, x₂, ..., x_N}`, and you assume each `xᵢ` was drawn from a distribution `p(x | θ)` parameterized by some unknown `θ`. (For a Gaussian, `θ = (μ, σ²)`. For a neural network, `θ` is all the weights.)

**Question:** which `θ` "best explains" the data?

**MLE answer:** pick `θ` that maximizes the probability of having observed exactly the data you observed.

Mathematically, the **likelihood** is the joint probability of all data under that parameter:

```
L(θ) = p(D | θ) = Π_i p(xᵢ | θ)        (assuming i.i.d. data)
```

We maximize `L(θ)`. In practice we maximize the **log-likelihood** (which has the same argmax but is easier numerically — multiplying many tiny numbers underflows; summing logs doesn't):

```
ℓ(θ) = log L(θ) = Σ_i log p(xᵢ | θ)
```

And by negation we **minimize the negative log-likelihood (NLL)**:

```
NLL(θ) = -Σ_i log p(xᵢ | θ)
```

**This is what every loss function in deep learning is.** Cross-entropy is NLL for categorical likelihood. MSE is NLL for Gaussian likelihood. Once you see that, the rest is bookkeeping.

### Worked example: MLE for the mean of a Gaussian

Data `{x₁, ..., x_N}` assumed drawn from `N(μ, σ²)` with σ known. Find the MLE for μ.

```
NLL(μ) = -Σᵢ log [(1/√(2πσ²)) · exp(-(xᵢ-μ)²/(2σ²))]
       = (1/(2σ²)) · Σᵢ (xᵢ - μ)²  +  const

Set d(NLL)/dμ = 0:
       Σᵢ 2(xᵢ - μ)·(-1) / (2σ²) = 0
       Σᵢ (xᵢ - μ) = 0
       μ̂ = (1/N) Σᵢ xᵢ
```

**The MLE for the mean of a Gaussian is the sample mean.** This is why "fit a Gaussian to data" works the way it does.

### The big derivation: MSE = Gaussian NLL

Now the punchline. In regression we model:

```
y = f(x; θ) + ε,  where ε ~ N(0, σ²)
```

I.e., the true value is a deterministic function plus Gaussian noise. The likelihood of seeing `yᵢ` given `xᵢ` is:

```
p(yᵢ | xᵢ, θ) = N(yᵢ; μ=f(xᵢ; θ), σ²) = (1/√(2πσ²)) · exp(-(yᵢ - f(xᵢ; θ))² / (2σ²))
```

NLL:

```
NLL(θ) = -Σᵢ log p(yᵢ | xᵢ, θ)
       = (1/(2σ²)) · Σᵢ (yᵢ - f(xᵢ; θ))²  +  const
       ∝ Σᵢ (yᵢ - f(xᵢ; θ))²
```

That's MSE. **Minimizing MSE is exactly maximum-likelihood under a Gaussian noise model.** Every regression problem you ever fit with `mse_loss` was, secretly, fitting a Gaussian.

### The big derivation: Cross-entropy = Categorical NLL

For K-class classification, we model `y ∈ {1,...,K}` with `p(y = k | x, θ) = f_k(x; θ)` (the model outputs a probability distribution over classes, usually via softmax).

The likelihood of one example:

```
p(yᵢ | xᵢ, θ) = Π_k f_k(xᵢ; θ)^[yᵢ = k]
```

(That's just "the predicted probability of the true class.") NLL:

```
NLL(θ) = -Σᵢ log f_{yᵢ}(xᵢ; θ)
```

That's cross-entropy. So `nn.CrossEntropyLoss` is the categorical likelihood. Again — not a magic loss, just NLL.

This single connection — "every loss is NLL of some distribution" — explains every loss function you'll meet:

| Loss | Distribution assumption |
|---|---|
| MSE | Gaussian noise |
| Cross-entropy | Categorical |
| Binary cross-entropy | Bernoulli |
| Huber loss | Laplace-like with quadratic-in-the-middle |
| Poisson loss | Poisson count data |

---

## Part 4: Maximum A Posteriori (MAP)

MLE asks "which `θ` makes the data most likely?" MAP asks "which `θ` is most likely *given* the data?":

```
θ_MAP = argmax_θ p(θ | D) = argmax_θ p(D | θ) · p(θ)
```

The only difference from MLE is the **prior** `p(θ)`. Take the log:

```
log p(θ|D) = log p(D|θ) + log p(θ) + const
            = -NLL(θ) + log p(θ) + const
```

So MAP = MLE + a regularizer that comes from the prior:

- **Gaussian prior on weights → L2 regularization (weight decay)** — the most-used prior in deep learning, usually implicit
- **Laplace prior on weights → L1 regularization** — encourages sparsity
- **Uniform improper prior → MLE** (no regularization)

This is why "weight decay" and "L2 regularization" are the same thing. They're both the log-probability of a Gaussian prior on the weights.

---

## Part 5: KL divergence and cross-entropy

### KL divergence — distance between distributions

For two distributions `p` and `q` over the same space:

```
KL(p || q) = Σ_x p(x) · log(p(x) / q(x))         (discrete)
           = ∫ p(x) · log(p(x) / q(x)) dx        (continuous)
```

Properties:
- `KL(p || q) ≥ 0`, with equality iff `p = q`
- `KL(p || q) ≠ KL(q || p)` in general (not a true metric)
- `KL(p || q) = E_p[log p] - E_p[log q] = -H(p) + H(p, q)`

Where `H(p) = -Σp(x)log p(x)` is the **entropy** of p, and `H(p, q) = -Σp(x) log q(x)` is the **cross-entropy** between p and q.

### Why this matters

Many ML objectives are KL divergences in disguise:

- **Classification loss** is `H(p_data, p_model)` where `p_data` is one-hot — that's just `-log p_model(true_class)`, the negative log-likelihood you've seen.
- **VAEs** minimize KL(approximate posterior || prior) + reconstruction loss.
- **GANs** (in theory) minimize Jensen-Shannon divergence, a symmetrized KL.
- **Distillation** is KL from a teacher to a student.

You'll meet KL again in week 12 (LoRA training objectives) and week 16 (model alignment / DPO).

### Entropy intuition

`H(p)` is the average "surprise" of samples from `p`. A uniform distribution over K outcomes has `H = log K` (most surprise). A point mass has `H = 0` (no surprise). Entropy is in bits if you use `log₂`, in nats if you use `log_e`. ML uses nats by default.

---

## Part 6: Bias, variance, and sample efficiency

When you estimate a quantity from data (`μ̂` from N samples), the estimator has:

- **Bias:** `E[μ̂] - μ_true` — does it systematically over/underestimate?
- **Variance:** `Var(μ̂)` — how noisy is your estimate?
- **Mean squared error:** `MSE = Bias² + Variance`

The MLE estimator for the Gaussian mean is **unbiased**. For the Gaussian variance, the MLE estimator `(1/N)Σ(xᵢ - μ̂)²` is **biased** — that's why the "unbiased sample variance" uses `(1/(N-1))` instead. Bessel's correction.

This decomposition (bias² + variance + irreducible noise) reappears in week 04 as the **bias-variance tradeoff** for entire models: simple models have high bias / low variance; complex models have low bias / high variance; you want the sweet spot.

---

## Part 7: The Central Limit Theorem (one sentence)

The CLT says: **sums (or averages) of many i.i.d. random variables, with finite variance, are approximately Gaussian**, regardless of the underlying distribution.

This is why Gaussian shows up everywhere — most aggregate quantities are sums of many small things. Stochastic gradient descent averages many gradient samples; "the noise on minibatch gradients is approximately Gaussian" is CLT in action.

---

## Part 8: What to defer

| Topic | When you'll need it |
|---|---|
| Measure theory | Never for ML engineering. Useful only if you want to do theoretical probability. |
| Markov chains, ergodicity | RL course; small role in MCMC; mostly skippable |
| Hypothesis testing (p-values, t-tests) | A/B testing role; not core ML |
| Gaussian processes | A niche of supervised learning; beautiful but not on the path |
| Information geometry | Theoretical foundations work; not engineering |
| Concentration inequalities | Theory papers; rarely engineering |

---

## What's next

In [lab.md](lab.md) you'll:
- Sample from distributions and verify the law of large numbers empirically
- Compute means, variances, and covariances of real data
- Walk Bayes' rule on a medical-test example (the classic that surprises everyone)
- Implement KL divergence and verify it's positive
- Derive MLE for a Gaussian by hand, then verify with sklearn
- Code a tiny demo that proves MSE-loss = Gaussian-NLL

If the MSE/cross-entropy derivations felt fast, **re-read Part 3 before moving on**. The "every loss is NLL" view is the single most important idea this week.
