# Week 08: Training-Loop Discipline

## 🎯 What you'll learn

The professional moves that turn "I ran train.py" into "I trained a model I trust." Most training failures aren't subtle bugs — they're missing seeds, missing LR finders, untracked metrics, and "let's just train it longer" hopes.

By the end of this week you'll be able to:

- Apply the "overfit one batch" sanity check before any real training
- Use Karpathy's recipe: simple baseline → fix one thing at a time → measure
- Implement gradient clipping, LR scheduling, warmup
- Set deterministic seeds; reproduce a run
- Log with TensorBoard or W&B; read learning curves
- Detect overfitting, vanishing gradients, NaNs, and dead ReLUs from the logs
- Checkpoint, resume, and version-control your training runs

## 🧰 Lab setup

```bash
uv add tensorboard wandb torchmetrics
```

## ✅ Your job

1. Read [theory.md](theory.md) — the Karpathy training recipe in full
2. Work through [lab.md](lab.md) — overfit one batch, then scale to full MNIST with proper logging
3. Reproduce one of your runs by setting a seed and re-running. Verify byte-for-byte.
4. (Stretch) Implement an LR finder per Smith (2017)

## 📚 Required reading

| Resource | Why | Time |
|---|---|---|
| [Karpathy — A recipe for training neural networks](https://karpathy.github.io/2019/04/25/recipe/) | The canonical reference | 45 min |
| [Smith — Cyclical learning rates](https://arxiv.org/abs/1506.01186) | The LR-finder paper | 30 min |
| [PyTorch — reproducibility docs](https://pytorch.org/docs/stable/notes/randomness.html) | All the things that aren't deterministic by default | 15 min |

## 💡 What you should already know

- Week 07 (PyTorch fluency)

---

> 🚧 **Scaffolded.** Full theory + lab content lands in the next pass.

**Next**: [Week 09: CNNs →](../week-09-cnns/readme.md)
