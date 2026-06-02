# Week 06: Build a Tiny Autograd Engine

## 🎯 What you'll learn

PyTorch's autograd feels like magic until you build one yourself. Following the [micrograd](https://github.com/karpathy/micrograd) approach, you'll build a 200-line scalar autograd library, then extend it to tensors. After this week, "autograd" stops being a black box.

By the end of this week you'll be able to:

- Explain how a computation graph is built and traversed
- Implement `Value` (scalar autograd) with `__add__`, `__mul__`, `__pow__`, `relu`, `backward`
- Extend to a tensor autograd that handles broadcasting
- Build a tiny `nn.Module` analog on top of your autograd
- Explain why `loss.backward()` works "for free" once the graph exists

## 🧰 Lab setup

Just numpy. We're rebuilding what PyTorch does.

## ✅ Your job

1. Read [theory.md](theory.md) — computational graphs and reverse-mode autodiff
2. Build through [lab.md](lab.md) — `Value` class then `Tensor` class
3. Train an MLP on a toy dataset using *your own* autograd engine
4. (Stretch) Add a `Conv2d` op — this is what week 14's CUDA kernels will compute on GPU

## 📚 Required reading

| Resource | Why | Time |
|---|---|---|
| [Karpathy — micrograd repo + video](https://github.com/karpathy/micrograd) | The reference build | 90 min (video) |
| [PyTorch autograd internals](https://pytorch.org/docs/stable/notes/autograd.html) | What you're approximating | 30 min |
| [Reverse-mode autodiff explained](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation) | The math | 30 min |

## 💡 What you should already know

- Week 05 (you've already done backprop by hand; now you'll generalize it)

---

**Next**: [Week 07: PyTorch Fundamentals →](../week-07-pytorch-fundamentals/readme.md)
