# Week 04: Lab — Classical Machine Learning

You'll build linear regression and logistic regression by hand in numpy, verify against sklearn, then use the real sklearn (and XGBoost) on a realistic tabular problem.

## Setup

```bash
uv add scikit-learn xgboost
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics, model_selection, preprocessing, pipeline
rng = np.random.default_rng(seed=42)
np.set_printoptions(precision=4, suppress=True)
```

---

## Exercise 4.1 — Linear regression from scratch (closed form)

Generate synthetic data `y = 3x₁ - 2x₂ + 1 + noise` and fit it two ways: the normal equation, and `np.linalg.lstsq`. Both should match.

```python
N = 500
X = rng.normal(0, 1, size=(N, 2))
true_w = np.array([3.0, -2.0])
true_b = 1.0
y = X @ true_w + true_b + rng.normal(0, 0.5, N)

# --- Solve via normal equation (after augmenting X with a column of 1s for the bias) ---
X_aug = np.hstack([X, np.ones((N, 1))])    # (N, 3)
w_full = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)
print("normal eqn:", w_full)               # [~3.0, ~-2.0, ~1.0]

# --- Solve via lstsq (more numerically stable) ---
w_lstsq, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
print("lstsq:    ", w_lstsq)

# --- Verify against sklearn ---
lr = linear_model.LinearRegression().fit(X, y)
print("sklearn:   coef={}, intercept={:.4f}".format(lr.coef_, lr.intercept_))
```

All three should agree to ~1e-12. **You just implemented `sklearn.LinearRegression`.**

---

## Exercise 4.2 — Linear regression from scratch (gradient descent)

Now solve the same problem with iterative gradient descent. This is the iteration you'll use *everywhere*:

```python
# Initialize parameters
w = np.zeros(2)
b = 0.0
lr = 0.01
losses = []

for epoch in range(200):
    y_pred = X @ w + b
    loss = np.mean((y_pred - y) ** 2)

    # Gradients
    grad_w = (2 / N) * X.T @ (y_pred - y)
    grad_b = (2 / N) * (y_pred - y).sum()

    # Update
    w -= lr * grad_w
    b -= lr * grad_b
    losses.append(loss)

print(f"learned: w={w}, b={b:.4f}")        # close to [3.0, -2.0], 1.0
print(f"final loss: {losses[-1]:.6f}")
```

Plot the loss curve:

```python
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel("epoch"); plt.ylabel("MSE")
plt.title("Gradient descent on linear regression")
plt.grid(alpha=0.3)
plt.show()
```

**Question:** what happens if `lr = 1.0`? `lr = 0.001`? Re-run and observe. This is your first taste of the "learning rate sensitivity" you'll spend weeks 08+ managing.

---

## Exercise 4.3 — Logistic regression from scratch on a 2D toy

Generate two well-separated blobs and fit a logistic classifier with gradient descent.

```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=300, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, class_sep=2.0, random_state=42,
)

# Visualize
plt.figure(figsize=(6, 6))
plt.scatter(X[y==0, 0], X[y==0, 1], label="class 0", alpha=0.6)
plt.scatter(X[y==1, 0], X[y==1, 1], label="class 1", alpha=0.6)
plt.legend()
plt.title("Toy binary classification")
plt.axis("equal")
plt.show()
```

Implement logistic regression manually:

```python
def sigmoid(z):
    # numerically stable variant
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

def predict_proba(X, w, b):
    return sigmoid(X @ w + b)

def bce_loss(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Initialize
w = np.zeros(X.shape[1])
b = 0.0
lr = 0.1
losses = []

for epoch in range(500):
    p = predict_proba(X, w, b)
    loss = bce_loss(y, p)

    # Gradients — these are clean because the sigmoid derivative cancels.
    # See theory.md Part 4.
    grad_w = X.T @ (p - y) / len(y)
    grad_b = (p - y).mean()

    w -= lr * grad_w
    b -= lr * grad_b
    losses.append(loss)

# Final accuracy
preds = (predict_proba(X, w, b) > 0.5).astype(int)
acc = (preds == y).mean()
print(f"learned w = {w}, b = {b:.4f}")
print(f"accuracy = {acc:.4f}")
```

Plot the decision boundary:

```python
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200),
)
grid = np.c_[xx.ravel(), yy.ravel()]
probs = predict_proba(grid, w, b).reshape(xx.shape)

plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, probs, levels=20, alpha=0.4, cmap="RdBu")
plt.scatter(X[y==0, 0], X[y==0, 1], c="b", edgecolor="k", label="class 0")
plt.scatter(X[y==1, 0], X[y==1, 1], c="r", edgecolor="k", label="class 1")
plt.title("Logistic regression decision boundary")
plt.legend()
plt.axis("equal")
plt.show()
```

The boundary is a straight line — that's why it's a *linear* classifier. For data that requires a curved boundary, you'll need polynomial features, kernels, or a deeper model.

Compare with sklearn:

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(X, y)
print("sklearn coef:", clf.coef_, "intercept:", clf.intercept_)
print("yours:        ", w, "                   ", b)
```

They should agree to ~3 decimal places (modulo the L2 default in sklearn, which we'll add next).

---

## Exercise 4.4 — Add L2 regularization

Add a `λ‖w‖²` term to the loss and re-derive the gradient. Then verify by varying λ and watching the weights shrink.

```python
def fit_logistic_l2(X, y, lam=0.0, lr=0.1, epochs=500):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        p = predict_proba(X, w, b)
        # Gradient of BCE + λ‖w‖²:
        grad_w = X.T @ (p - y) / len(y) + 2 * lam * w
        grad_b = (p - y).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

for lam in [0.0, 0.1, 1.0, 10.0]:
    w_l, b_l = fit_logistic_l2(X, y, lam=lam)
    print(f"λ={lam:>5.2f}   ‖w‖={np.linalg.norm(w_l):.4f}   acc={(predict_proba(X, w_l, b_l) > 0.5).mean():.4f}")
```

**Observe:** as λ rises, `‖w‖` shrinks. At very large λ, the model loses too much capacity and accuracy drops. **You just sketched the regularization sweep that produces the bias-variance U-curve.**

---

## Exercise 4.5 — Bias-variance experiment

Visualize bias vs variance by fitting models of varying complexity to the same noisy data.

```python
# True function: a sine wave, observed noisily
true_f = lambda x: np.sin(2 * np.pi * x)
N_train = 30

x_full = np.linspace(0, 1, 200)
xs_train = rng.uniform(0, 1, N_train)
ys_train = true_f(xs_train) + rng.normal(0, 0.25, N_train)

# Fit polynomials of degree 1, 3, 9, 15
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, deg in zip(axes, [1, 3, 9, 15]):
    coeffs = np.polyfit(xs_train, ys_train, deg=deg)
    y_pred = np.polyval(coeffs, x_full)
    ax.plot(x_full, true_f(x_full), 'g--', label='truth')
    ax.scatter(xs_train, ys_train, c='black', s=20, label='train')
    ax.plot(x_full, y_pred, 'r-', label=f'degree {deg}')
    ax.set_ylim(-2, 2)
    ax.set_title(f"degree {deg}")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
```

You should see:
- Degree 1 — **high bias, low variance** (underfits, doesn't capture the curve)
- Degree 3 — sweet spot
- Degree 9 — starting to wiggle
- Degree 15 — **low bias, high variance** (memorizes every training point, wild between them)

This is the bias-variance tradeoff, made visible.

---

## Exercise 4.6 — A real classification problem with sklearn

We'll use the `iris` dataset because it's reliable, and write the pipeline properly.

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

iris = datasets.load_iris(as_frame=True)
X, y = iris.data, iris.target

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Pipeline: scaling → classifier
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, C=1.0)),
])
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
print("Logistic regression test report:")
print(classification_report(y_test, preds, target_names=iris.target_names))

# Cross-validation
scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
print(f"5-fold CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

**Key idea:** wrapping `StandardScaler` and `LogisticRegression` in a `Pipeline` means the scaler is re-fit on each cross-validation fold's *training* data only. Without that, you'd leak test statistics into training. This is the #1 leakage bug in ML pipelines — and `Pipeline` is the fix.

---

## Exercise 4.7 — Compare three models on the same data

```python
from xgboost import XGBClassifier

models = {
    "Logistic Regression":
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]),
    "Random Forest":
        RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost":
        XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42),
}

for name, m in models.items():
    scores = cross_val_score(m, X_train, y_train, cv=5, scoring="accuracy")
    print(f"{name:<22}  CV acc = {scores.mean():.4f} ± {scores.std():.4f}")
```

Iris is small, so all three should look similar. The point is the *pattern*: define your models, run the same CV protocol, compare. Don't grade them on a single number — average across folds.

For a harder dataset (try `sklearn.datasets.fetch_openml('credit-g')` or any Kaggle binary classification): you'll typically see XGBoost ahead of random forest by ~1-3 pp, both significantly ahead of logistic regression.

---

## Exercise 4.8 — Read a confusion matrix

```python
xgb = models["XGBoost"]
xgb.fit(X_train, y_train)
preds = xgb.predict(X_test)
cm = confusion_matrix(y_test, preds)
print("Confusion matrix (rows=true, cols=pred):")
print(cm)
print("\nClassification report:")
print(classification_report(y_test, preds, target_names=iris.target_names))
```

Numbers on the diagonal are correct predictions; off-diagonal are errors. For iris you'll mostly see diagonal entries.

**Practice:** if the model misclassifies setosa as virginica, in which cell of the matrix does that count? (Row 0, column 2.) This kind of question becomes second nature with practice.

---

## Submission checklist

- [ ] Linear regression from scratch (normal eqn + gradient descent) matches sklearn to ≤ 1e-3
- [ ] Logistic regression from scratch on a 2D toy gives accuracy comparable to sklearn (≥ 95%)
- [ ] L2 regularization sweep shows `‖w‖` shrinking as λ increases
- [ ] Bias-variance polynomial-fitting figure rendered with the expected behavior at degrees 1, 3, 9, 15
- [ ] sklearn `Pipeline` used correctly (scaler inside the pipeline, not outside)
- [ ] CV scores reported for at least three model families on iris
- [ ] One confusion matrix read and interpreted in words

---

## What you just did

You can now look at a tabular dataset, write a sklearn pipeline, evaluate it with cross-validation, and explain to someone *why* you picked XGBoost over logistic regression — without hand-waving. That's the level expected of a junior ML engineer working on production tabular models, which is most of the field by volume.

From next week (week 05) we leave classical ML and never come back. But the ideas — train/val/test, bias-variance, regularization, cross-validation, calibration — stay relevant for every deep learning model from here to inference.

---

**Next**: [Week 05: MLP from Numpy →](../week-05-mlp-from-numpy/readme.md)
