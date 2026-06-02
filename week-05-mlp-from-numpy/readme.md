# Week 05: MLP from Numpy

## 🎯 What you'll learn

Build a multi-layer perceptron — forward pass, loss, backward pass, weight updates — using nothing but numpy. No PyTorch. No autograd. The point: when you reach for `nn.Linear` next week, you'll know exactly what's inside.

By the end of this week you'll be able to:

- Implement a 2-layer MLP forward pass on MNIST
- Derive the backward pass by applying chain rule to each layer
- Implement SGD with mini-batches and a learning rate
- Train to >97% MNIST test accuracy with ~100 lines of numpy
- Explain why ReLU + Glorot/He init unlocks deeper nets

## 🧰 Lab setup

Same as week 03 (numpy + matplotlib).

## ✅ Your job

1. Read [theory.md](theory.md) — the matrix-calculus derivation for each layer's gradient
2. Work through [lab.md](lab.md) — type every line; resist copy-paste
3. Achieve >97% MNIST test accuracy with your numpy MLP
4. (Stretch) Add a third layer and re-derive the gradients

## 📚 Required reading

| Resource | Why | Time |
|---|---|---|
| [Karpathy — Yes, you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b) | Why hand-deriving matters | 15 min |
| [CS231n — Backprop notes](https://cs231n.github.io/optimization-2/) | The canonical walkthrough | 60 min |
| [Glorot & Bengio (2010)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) | Why initialization matters | 20 min skim |

## 💡 What you should already know

- Weeks 01-04 — especially chain rule from week 01

---

> 🚧 **Scaffolded.** Full theory + lab content lands in the next pass.

**Next**: [Week 06: Autograd Engine →](../week-06-autograd-engine/readme.md)
