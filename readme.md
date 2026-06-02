# ML Mentorship — From First Principles to GPU/NVIDIA Depth

A 16-week curriculum for building a working mental model of machine learning **and** the hardware/runtime stack underneath it. Most ML curricula stop at "use PyTorch"; this one keeps going down to the SM and warp.

## 🎯 Why this exists

You can ship an ML feature today by importing `transformers` and calling `.from_pretrained()`. That's fine for prototypes. It's not enough when:

- The model is too slow and you need to know which kernel is the bottleneck
- The model OOMs and you need to know what's actually living in HBM
- Inference cost is the budget line item and you need to make architectural calls about batching, KV cache, quantization
- An interviewer asks "walk me through what happens when you call `model(x)` on a GPU"

This curriculum builds upward — math → autograd → PyTorch → transformers → training-at-scale → CUDA → inference engines — so you have a working answer at every layer.

## 🧭 The arc

```mermaid
---
config:
  look: handDrawn
  theme: neutral
---
flowchart LR
    subgraph foundations["Foundations (01-04)"]
        direction TB
        W1["01 · Math"]
        W2["02 · Probability"]
        W3["03 · Python ML stack"]
        W4["04 · Classical ML"]
    end

    subgraph dl["Core deep learning (05-08)"]
        direction TB
        W5["05 · MLP from numpy"]
        W6["06 · Autograd engine"]
        W7["07 · PyTorch fundamentals"]
        W8["08 · Training-loop discipline"]
    end

    subgraph modern["Modern architectures (09-12)"]
        direction TB
        W9["09 · CNNs"]
        W10["10 · Transformers"]
        W11["11 · Training at scale"]
        W12["12 · LLM fine-tuning"]
    end

    subgraph gpu["GPU + NVIDIA depth (13-16)"]
        direction TB
        W13["13 · GPU architecture"]
        W14["14 · CUDA programming"]
        W15["15 · Profiling + perf"]
        W16["16 · Production inference"]
    end

    foundations --> dl --> modern --> gpu
```

## 📁 Curriculum

| ☐ | Week | Topic | What you'll do |
|---|---|---|---|
| ☐ | 01 | [Math foundations](week-01-math-foundations) | Linear algebra + calculus you actually need — vectors, matrix multiplication, gradients, chain rule |
| ☐ | 02 | [Probability & statistics](week-02-probability-stats) | Distributions, Bayes, MLE/MAP, the math behind every loss function |
| ☐ | 03 | [Python ML stack](week-03-python-ml-stack) | numpy, pandas, matplotlib, jupyter, uv — fluency in the tools you'll use every day |
| ☐ | 04 | [Classical ML](week-04-classical-ml) | Linear/logistic regression from scratch, then sklearn — bias/variance, regularization, train/val/test |
| ☐ | 05 | [MLP from numpy](week-05-mlp-from-numpy) | Build a multi-layer perceptron with manual forward + backward in pure numpy |
| ☐ | 06 | [Autograd engine](week-06-autograd-engine) | Build a tiny autograd library (micrograd-style); understand what PyTorch is doing under the hood |
| ☐ | 07 | [PyTorch fundamentals](week-07-pytorch-fundamentals) | Tensors, `nn.Module`, `DataLoader`, optimizers, device management |
| ☐ | 08 | [Training-loop discipline](week-08-training-loop-discipline) | The senior moves: overfit one batch first, LR finder, gradient clipping, checkpointing, reproducibility |
| ☐ | 09 | [CNNs](week-09-cnns) | Convolution, pooling, batch norm, residuals — train a ResNet on CIFAR-10 |
| ☐ | 10 | [Transformers](week-10-transformers) | Self-attention, multi-head, position encoding (RoPE), KV cache — build a GPT from scratch |
| ☐ | 11 | [Training at scale](week-11-training-at-scale) | Mixed precision (AMP), gradient accumulation, schedulers, distributed (DDP/FSDP) basics |
| ☐ | 12 | [LLM fine-tuning](week-12-llm-finetuning) | LoRA, QLoRA, PEFT, eval harnesses, prompt + response tuning |
| ☐ | 13 | [GPU architecture](week-13-gpu-architecture) | SMs, warps, memory hierarchy, NVIDIA's Hopper/Blackwell generations — the mental model |
| ☐ | 14 | [CUDA programming](week-14-cuda-programming) | Write your first kernels — vector add, matmul, fused softmax. Understand grids, blocks, threads, shared memory |
| ☐ | 15 | [Profiling + perf](week-15-profiling-and-perf) | Nsight Systems, Nsight Compute, memory coalescing, kernel fusion, occupancy |
| ☐ | 16 | [Production inference](week-16-production-inference) | TensorRT, vLLM, continuous batching, KV cache management, paged attention, deployment |

## 📋 How each week works

Three files per week:

| File | Role |
|---|---|
| `readme.md` | What you'll learn this week, the lab setup, your assigned exercises, required reading |
| `theory.md` | The "why" — derivations, conceptual walkthroughs, the math when it matters |
| `lab.md` | Guided code-along with verifiable output — you should be able to run every snippet |

Plus a `solutions/` directory in weeks where the exercises have canonical answers.

## 🌐 Languages in the ML stack — what to learn, what to skip

The curriculum is Python-first because the day-to-day ML world is Python-first. But optimization work pulls you into other languages, and the honest ranking is:

| Language | Where it shows up | This curriculum |
|---|---|---|
| **Python** | Day-to-day ML, training scripts, research code | Used in every week |
| **C++ (with CUDA)** | Kernels, PyTorch internals, cuBLAS/cuDNN/TensorRT APIs, Triton-generated PTX. The single most important "second language" for ML perf. | **Week 14** has you write CUDA C++ kernels by hand |
| **Rust** | `tokenizers`, `safetensors`, `polars`, HuggingFace's `candle`. Where modern hot-path ML infra is migrating. Rewriting a Python-bound hot loop in Rust is a real production pattern. | Sidebar in **Week 16**; optional follow-on after the capstone |
| **Triton** (Python-flavored CUDA) | Modern alternative to raw CUDA for fused kernels. Same mental model as CUDA, ~10× shorter code. | Covered alongside CUDA in **Week 14** |
| **Go** | Kubernetes, KServe, Triton Inference Server's gRPC layer, model-serving infra glue. Zero ML kernel work happens in Go. | **Out of scope.** Belongs in [system-design-mentorship](https://github.com/ichdamola/system-design-mentorship). |
| Julia / Mojo | Niche; interesting; not yet broadly adopted | Skipped |

The short version: if a curriculum claims "GPU + NVIDIA depth" and never makes you write C++ for CUDA, it's lying. Rust is the right second mention because it's where modern ML *infra* libraries actually live. Go is a great language — just not for the optimization layer.

## 🚀 Getting started

### Hardware

You can do weeks 01-12 entirely on a laptop. Weeks 13-16 want a real GPU:

| Tier | What works |
|---|---|
| Laptop CPU | 01-12 (small models) |
| Free Colab T4 | 01-14 (everything but the heaviest profiling) |
| Paid Colab Pro / Lambda Cloud / vast.ai (A100, H100) | 01-16 fully |
| Local NVIDIA GPU (RTX 3090/4090, A6000) | 01-16 fully |

If you don't own a GPU and aren't ready to rent one, plan to do weeks 13-16 in Colab or on a short rented session.

### Software

```bash
# Modern Python toolchain
curl -LsSf https://astral.sh/uv/install.sh | sh

# Project setup (week 03 covers this in full)
uv init
uv add numpy pandas matplotlib jupyter scikit-learn
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu124  # for NVIDIA
```

CUDA toolkit (weeks 14+):

```bash
# Install matching your driver version
# https://developer.nvidia.com/cuda-downloads
nvcc --version    # should report >= 12.4
```

## 🎓 Learning philosophy

1. **Build it before you call it.** Every layer of abstraction you'll use professionally — autograd, PyTorch modules, transformer blocks, CUDA kernels — you'll first build a tiny version of yourself. The professional version then makes sense, instead of being magic.
2. **Numbers, not adjectives.** "Fast" is meaningless; "0.4 ms per token at batch 32" is a budget. Every week ends with a measurement.
3. **Overfit one batch first.** Before any real training, prove your model can drive loss to zero on one batch. If it can't, the bug is in your code, not your hyperparameters.
4. **Read the paper, then the code.** The paper tells you why; the code tells you how. Both matter.
5. **Profile before optimizing.** Premature optimization is exactly as bad in ML as anywhere else, and the profiler will surprise you every time.

## 📚 Reference shelf

Books to keep open across the curriculum:

- **Deep Learning** — Goodfellow, Bengio, Courville (the math reference)
- **The Little Book of Deep Learning** — François Fleuret (the concise companion, free PDF)
- **Programming Massively Parallel Processors** — Kirk + Hwu (the CUDA textbook)
- **CUDA C++ Best Practices Guide** — NVIDIA (the official one, free)
- **The PyTorch source code** — `torch/nn/modules/transformer.py` reads beautifully

Papers referenced across weeks:
- **Attention Is All You Need** — Vaswani et al., 2017 (week 10)
- **GPT-3 / Scaling Laws** — Kaplan et al., 2020; Hoffmann et al. (Chinchilla), 2022 (week 11)
- **LoRA** — Hu et al., 2021 (week 12)
- **FlashAttention** — Dao et al., 2022, 2023 (weeks 14, 15)
- **Efficient Memory Management for LLM Serving with PagedAttention** — Kwon et al., 2023 (week 16)

## 🔗 Contributing

This curriculum is built in the open. If you find an error in a derivation, a code snippet that doesn't run on current hardware, or a better way to explain something, open a PR.

---

**Related curricula by the same author:**
- [django-mentorship](https://github.com/ichdamola/django-mentorship)
- [system-design-mentorship](https://github.com/ichdamola/system-design-mentorship)
- [appsec-mentorship](https://github.com/ichdamola/appsec-mentorship)
