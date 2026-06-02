# Week 07: PyTorch Fundamentals

## 🎯 What you'll learn

Now that you've built a toy autograd, switch to the professional one. This week is about reading and writing idiomatic PyTorch — tensors, modules, optimizers, devices — without the magic.

By the end of this week you'll be able to:

- Move tensors between CPU and GPU, manage `dtype` and `device` correctly
- Write a custom `nn.Module` with `__init__` and `forward`
- Use `DataLoader` with custom `Dataset`s and collate functions
- Configure optimizers (SGD, Adam, AdamW) and schedulers
- Save and load checkpoints reliably
- Explain `.eval()` vs `.train()`, `with torch.no_grad()`, `torch.inference_mode()`

## 🧰 Lab setup

```bash
# For NVIDIA GPU (CUDA 12.4):
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CPU only / macOS:
uv add torch torchvision
```

## ✅ Your job

1. Read [theory.md](theory.md) — tensor semantics, view vs copy, autograd vs nograd
2. Work through [lab.md](lab.md) — port your numpy MLP from week 05 into PyTorch
3. Train MNIST on GPU if you have one; measure the speedup vs CPU
4. (Stretch) Implement a custom `Dataset` for a CSV file you care about

## 📚 Required reading

| Resource | Why | Time |
|---|---|---|
| [PyTorch — 60-minute blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) | The official intro | 60 min |
| [PyTorch internals — Edward Yang](http://blog.ezyang.com/2019/05/pytorch-internals/) | What's happening under the hood | 45 min |
| [PyTorch — `nn.Module` source](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py) | The real one | 30 min skim |

## 💡 What you should already know

- Week 06 (you understand what autograd is doing)

---

**Next**: [Week 08: Training-Loop Discipline →](../week-08-training-loop-discipline/readme.md)
