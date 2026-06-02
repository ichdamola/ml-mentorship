# Week 04: Classical Machine Learning

## 🎯 What you'll learn

Before deep learning there was classical ML — and most production "ML" today is still linear regression, random forests, or gradient boosting. You'll implement linear and logistic regression from scratch, then use sklearn, then learn when to reach for each.

By the end of this week you'll be able to:

- Derive the closed-form solution for linear regression (least squares)
- Implement gradient descent for linear and logistic regression from numpy alone
- Use sklearn's `Pipeline`, `train_test_split`, `cross_val_score` fluently
- Diagnose bias vs variance — read learning curves
- Explain L1 vs L2 regularization and what each does geometrically
- Pick the right model for tabular data (linear / trees / boosting / NN — and when each wins)

## 🧰 Lab setup

```bash
uv add scikit-learn xgboost
```

## ✅ Your job

1. Read [theory.md](theory.md) — bias/variance + the regularization geometry
2. Work through [lab.md](lab.md) — implement linear regression from scratch in numpy, then verify against sklearn
3. Use sklearn to fit a model on a real Kaggle tabular dataset of your choice
4. (Stretch) implement gradient-boosted trees from scratch — XGBoost is "just" a clever loss + clever splits

## 📚 Required reading

| Resource | Why | Time |
|---|---|---|
| [ISLR — Chapters 3, 4, 6](https://www.statlearning.com/) | The textbook of classical ML, free PDF | 3 hours |
| [Andrew Ng — Lecture notes on linear regression](https://cs229.stanford.edu/main_notes.pdf) | Crisp derivation | 60 min |
| [scikit-learn user guide — Pipelines](https://scikit-learn.org/stable/modules/compose.html) | The right way to use sklearn | 30 min |

## 💡 What you should already know

- Weeks 01-03

---

> 🚧 **Scaffolded.** Full theory + lab content lands in the next pass.

**Next**: [Week 05: MLP from Numpy →](../week-05-mlp-from-numpy/readme.md)
