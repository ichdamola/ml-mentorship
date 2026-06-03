# Week 13: GPU Architecture — The Mental Model

## 🎯 What you'll learn

Up to here you've been treating GPUs as "fast PyTorch boxes." This week we open the lid. You'll build the mental model — Streaming Multiprocessors (SMs), warps, memory hierarchy — that the next three weeks build on.

By the end of this week you'll be able to:

- Explain SIMT execution and the 32-thread warp
- Read a GPU spec sheet and pull out the numbers that matter (SM count, shared memory per SM, memory bandwidth, FP16 / BF16 / INT8 / FP8 throughput)
- Distinguish global memory, shared memory, registers, L1/L2 caches — and the latency/bandwidth gap between them
- Explain the NVIDIA generation lineup: Volta → Turing → Ampere → Ada → Hopper → Blackwell (and what changed each time — Tensor Cores, sparsity, FP8, transformer engine, NVLink)
- Reason about why memory bandwidth dominates ML perf, not FLOPs
- Read `nvidia-smi`, `nvcc --version`, `nvidia-smi topo -m`

## 🧰 Lab setup

You need an actual NVIDIA GPU for the next four weeks. Options:

| Option | Setup |
|---|---|
| Local NVIDIA GPU | Verify with `nvidia-smi`; install CUDA 12.4+ |
| Colab (free T4 or paid V100/A100) | Already configured; `!nvidia-smi` |
| vast.ai / Lambda / RunPod (rented A100/H100) | Pick the cheapest spot instance |
| AWS / GCP / Azure | More expensive; only if your employer pays |

Verify your setup:

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

## ✅ Your job

1. Read [theory.md](theory.md) — the SM/warp/memory model from first principles
2. Work through [lab.md](lab.md) — read the spec sheet of your actual GPU; calculate its theoretical roofline; measure how close you get with a simple benchmark
3. Identify which generation your GPU is and what its peak BF16 / FP16 / INT8 / FP8 TFLOPS are
4. (Stretch) Read the H100 / B200 whitepapers and identify three new features per generation

## 📚 Required reading

| Resource | Why | Time |
|---|---|---|
| [Programming Massively Parallel Processors (Kirk + Hwu, 4th ed) — Ch. 1-4](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) | The textbook for this whole tier of the curriculum | 4 hours |
| [NVIDIA — CUDA Programming Guide (Ch. 1-5)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | The official reference; bookmark forever | 2 hours |
| [NVIDIA Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core) | What modern enterprise hardware looks like | 60 min skim |
| [NVIDIA Blackwell Architecture Whitepaper](https://resources.nvidia.com/en-us-blackwell-architecture) | Current generation (2025) | 60 min skim |
| [Horace He — Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) | Why the roofline matters | 30 min |

## 💡 What you should already know

- A laptop's CPU mental model (registers → L1 → L2 → L3 → RAM)
- Why parallelism is fundamentally different from concurrency

---

**Next**: [Week 14: CUDA Programming →](../week-14-cuda-programming/readme.md)
