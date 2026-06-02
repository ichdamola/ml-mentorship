# Week 02: Probability & Statistics for ML

## 🎯 What you'll learn

Probability is the math of uncertainty. Every loss function, every regularizer, every Bayesian prior in modern ML traces back to a probability distribution. This week makes that visible.

By the end of this week you'll be able to:

- Read and compute basic probabilities (joint, marginal, conditional, Bayes)
- Recognize Gaussian, Bernoulli, categorical, and Poisson distributions in code
- Explain Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP)
- Derive why MSE loss is "Gaussian likelihood" and cross-entropy is "categorical likelihood"
- Reason about bias, variance, and sample efficiency

## 🧰 Lab setup

Same environment as week 01 (numpy + matplotlib + jupyter).

## ✅ Your job

1. Read [theory.md](theory.md) — focus on the MLE-derives-MSE walkthrough
2. Work through [lab.md](lab.md) — sampling, KL divergence, a tiny Bayesian update
3. Solve the assigned exercises; verify the MSE-as-Gaussian-NLL derivation by hand

## 📚 Required reading

| Resource | Why | Time |
|---|---|---|
| [Deep Learning book — Chapter 3 (Probability)](https://www.deeplearningbook.org/contents/prob.html) | The compressed reference | 60 min |
| [Distill — Why does cross-entropy work?](https://distill.pub) | Loss-function intuition | 30 min |
| [Bishop — Pattern Recognition Ch. 1.2](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/) | The classic | 45 min |

## 💡 What you should already know

- Week 01 (vectors, gradients)
- Counting (combinations / permutations)

---

> 🚧 **Scaffolded.** Full theory + lab content lands in the next pass.

**Next**: [Week 03: Python ML Stack →](../week-03-python-ml-stack/readme.md)
