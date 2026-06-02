# Week 15: Profiling and Performance

## 🎯 What you'll learn

Where your training run actually spends its time and memory — and what to do about it. The senior move: never optimize without profiling first.

By the end of this week you'll be able to:

- Use `torch.profiler` to see CPU + CUDA time per op for a PyTorch model
- Use Nsight Systems (`nsys`) for system-level timeline traces
- Use Nsight Compute (`ncu`) for per-kernel performance counters
- Read a roofline diagram and place your kernel on it
- Identify memory-bound vs compute-bound kernels
- Apply the standard fixes: kernel fusion, memory coalescing, occupancy tuning, AMP, gradient checkpointing
- Reason about activation memory vs parameter memory vs KV cache (sets up week 16)

## 🧰 Lab setup

```bash
# Already installed: PyTorch, CUDA toolkit
# Add Nsight Systems CLI — usually bundled with CUDA toolkit
nsys --version
ncu --version
```

```bash
# PyTorch profiler (built in)
python -c "from torch.profiler import profile; print('ok')"
```

## ✅ Your job

1. Read [theory.md](theory.md) — the roofline model, compute vs memory bound, occupancy
2. Work through [lab.md](lab.md) — profile your week-10 transformer training; identify the 3 slowest ops; fuse the top one
3. Run `nsys` on a real training step; read the resulting timeline in the GUI
4. Run `ncu` on one kernel; read achieved-vs-peak occupancy
5. (Stretch) Use `torch.compile` and compare before/after

## 📚 Required reading

| Resource | Why | Time |
|---|---|---|
| [Horace He — Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) | The mental model | 30 min |
| [PyTorch Profiler tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) | The official walkthrough | 45 min |
| [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | The reference | 60 min skim |
| [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691) | What kernel-level optimization looks like in practice | 45 min |
| [Roofline model — Williams et al., 2009](https://dl.acm.org/doi/10.1145/1498765.1498785) | The classic | 30 min |

## 💡 What you should already know

- Weeks 13-14 (you can read kernel-level metrics)

---

> 🚧 **Scaffolded.** Full theory + lab content lands in the next pass.

**Next**: [Week 16: Production Inference →](../week-16-production-inference/readme.md)
