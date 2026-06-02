# Week 01: Math Foundations

## 🎯 What you'll learn

The minimum linear algebra and calculus you need to read a deep-learning paper and recognize what's happening in a `torch.nn` layer. Not a math course — a survival kit.

By the end of this week you'll be able to:

- Reason about vectors, matrices, and tensors as data containers and as transformations
- Multiply matrices by hand on a small example and predict the shape of the result
- Compute gradients of multivariate functions and apply the chain rule
- Recognize the gradient pattern of a loss function written in PyTorch
- Explain why "deep learning is matrix multiplications and gradients" without it being a lie

## 🧰 Lab setup

```bash
mkdir ml-mentorship-work && cd ml-mentorship-work
uv init
uv add numpy matplotlib jupyter
uv run jupyter notebook
```

That's everything you need this week. Real ML libraries arrive in week 03.

## ✅ Your job

1. Read [theory.md](theory.md). Don't skim the chain-rule derivation — it's the single most important page of math in the curriculum.
2. Work through [lab.md](lab.md). Every code cell should run cold. The lab ends with a hand-derived gradient that matches a numerical check to ~1e-6.
3. The stretch exercise (lab.md, Exercise 1.7) implements the perceptron forward pass. You'll be glad you did when week 05 asks you to do it from memory.

## 📚 Required reading

| Resource | Why | Time |
|---|---|---|
| [3Blue1Brown — Essence of linear algebra (Ch. 1-7)](https://www.3blue1brown.com/topics/linear-algebra) | The visual intuition that makes the algebra stick | 90 min |
| [3Blue1Brown — Essence of calculus (Ch. 1-4)](https://www.3blue1brown.com/topics/calculus) | Derivatives the way you'll actually use them | 60 min |
| [Deep Learning book — Chapter 2 (Linear Algebra)](https://www.deeplearningbook.org/contents/linear_algebra.html) | The reference summary; skim, then come back when stuck | 45 min |
| [The Matrix Cookbook (sections 1, 2, 4)](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) | Bookmark; you'll cite it for years | 20 min skim |

## 💡 What you should already know

- Algebra (solving for x)
- Basic single-variable derivatives (`d/dx (x²) = 2x`)
- Python at the level of writing a loop and a function. If `def f(x): return x**2` is comfortable, you're set.

---

**Next**: [Week 02: Probability & Statistics →](../week-02-probability-stats/readme.md)
