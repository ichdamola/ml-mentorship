# Week 13: Theory — GPU Architecture (the mental model)

For 12 weeks you treated the GPU as a black box that runs PyTorch fast. From here we open the box. The next 4 weeks build downward: this week is the mental model — SMs, warps, memory hierarchy, NVIDIA generations. Week 14 you'll write CUDA kernels. Week 15 you'll profile them. Week 16 you'll serve a real LLM with all of it.

By the end of this week you'll be able to:
- Read a GPU spec sheet and pick out the numbers that matter
- Calculate the theoretical peak FLOPS and memory bandwidth of your GPU
- Reason about why memory bandwidth — not FLOPs — usually bounds ML perf
- Tell apart NVIDIA generations and what new feature each one shipped

---

## Part 1: CPU vs GPU — the right mental model

A CPU is a small number of very smart cores. An H100 GPU is **132 cores ("SMs"), each running 2048 simple threads at once**. Total: ~270,000 active threads.

| | CPU (modern x86 server) | GPU (H100) |
|---|---|---|
| Cores | 64-128 | 132 SMs |
| Hardware threads per core | 2 (SMT) | 2048 |
| Total concurrent threads | ~256 | ~270,000 |
| Per-core ALU width | ~512 bits AVX-512 | 128 FMA units × 32-thread warp |
| L1 cache | 32-64 KB private per core | 256 KB shared per SM |
| Branch prediction | Excellent | None |
| Out-of-order execution | Yes | No |
| Memory bandwidth | ~300 GB/s | 3 TB/s (H100 HBM3) |

**Rule of thumb:** if your problem can be expressed as "do roughly the same operation on a million pieces of data," GPU wins by 10-100×. If it's "do many different things sequentially, each dependent on the last," CPU wins.

ML training and inference are dense matmuls — the GPU's killer use case.

---

## Part 2: SIMT and the 32-thread warp

NVIDIA GPUs execute in **SIMT** (Single Instruction, Multiple Thread) mode. Threads are grouped into **warps** of 32, and all 32 threads in a warp execute the **same instruction at the same time** on different data.

This is the most important fact about GPU programming:

```
A warp = 32 threads marching in lockstep.

If threads in a warp take different code paths (warp divergence), the hardware
serializes them, masking off threads that aren't on the current path. You pay
for every divergent branch.
```

**Don't write GPU kernels with `if (threadIdx.x % 2 == 0) { ... }` patterns** unless you must. The 16 odd threads wait while the 16 even threads execute.

The warp is **the fundamental scheduling unit**. NVIDIA's hardware doesn't think in threads; it thinks in warps.

---

## Part 3: The SM (Streaming Multiprocessor)

An SM is the "core" of a GPU. Inside one H100 SM:

```
┌──────────────────────────────────────────────────────────┐
│                    Streaming Multiprocessor              │
│                                                          │
│  ┌────────────────────┐  ┌────────────────────────┐      │
│  │  4 warp schedulers │  │  Tensor Cores (4)     │      │
│  │  (4 warp-instr.    │  │  fp16/bf16/fp8 dense  │      │
│  │   issued / cycle)  │  │  matrix multiply       │      │
│  └────────────────────┘  └────────────────────────┘      │
│                                                          │
│  ┌────────────────────────────────────────────┐          │
│  │  CUDA cores: 128 FP32, 64 FP64,             │          │
│  │  64 INT, 128 special-function (SFU)         │          │
│  └────────────────────────────────────────────┘          │
│                                                          │
│  ┌────────────────────────────────────────────┐          │
│  │  Register file: 256 KB                      │          │
│  │  Shared memory / L1: 228 KB (configurable) │          │
│  └────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────┘
```

Per-SM key numbers (H100):
- **Max threads:** 2048 (= 64 warps)
- **Max warps in flight:** 64
- **Max threadblocks:** 32
- **Registers:** 256 KB (65,536 × 32-bit)
- **Shared memory + L1 (configurable):** 256 KB total
- **FP32 ops per cycle:** 128 (from CUDA cores)
- **Tensor core ops per cycle:** ~512 fp16 MACs

The H100 has **132 SMs**. So whole-GPU: ~17,000 fp32 cores, ~67,000 tensor MAC units, 33 MB total shared memory.

---

## Part 4: The memory hierarchy

This is the single most important concept for GPU perf. Latency and bandwidth scale wildly across levels.

```
┌──────────────────────────────────────────────────────────────┐
│  Registers per thread          (256 KB / SM = 32 KB/warp)    │
│  Latency: 1 cycle              Bandwidth: ~10 TB/s effective  │
├──────────────────────────────────────────────────────────────┤
│  Shared memory / L1 (configurable, ~228 KB/SM)               │
│  Latency: ~20-30 cycles        Bandwidth: ~10 TB/s            │
├──────────────────────────────────────────────────────────────┤
│  L2 cache (50 MB on H100, 60 MB on B200)                      │
│  Latency: ~200 cycles          Bandwidth: ~6 TB/s             │
├──────────────────────────────────────────────────────────────┤
│  HBM (global memory, 80 GB on H100, 192 GB on B200)           │
│  Latency: ~500 cycles          Bandwidth: 3 TB/s              │
├──────────────────────────────────────────────────────────────┤
│  PCIe (host RAM)               Bandwidth: ~64 GB/s             │
└──────────────────────────────────────────────────────────────┘
```

**Bandwidth to HBM is the bottleneck for most ML workloads.** Every byte you can keep in registers or shared memory instead of HBM is a 100× speedup on that operation.

This is what kernel optimization is mostly about — restructuring the math so each byte loaded from HBM is reused many times. (FlashAttention is the canonical example: keep attention scores in SRAM, recompute instead of fetching from HBM.)

### Memory coalescing

When 32 threads in a warp all load 32 consecutive 4-byte floats, NVIDIA's memory controller serves it as **one 128-byte transaction**. That's coalesced — fast.

When threads load 32 scattered floats from random addresses, it's **32 separate transactions**. 32× slower.

```cuda
// Coalesced — adjacent threads load adjacent floats
float val = X[threadIdx.x];

// Uncoalesced — stride-32 access
float val = X[threadIdx.x * 32];
```

For 2D tensors, the **inner stride** (usually the last dim) is the one that needs to be contiguous. This is why PyTorch's `.contiguous()` matters before some operations.

---

## Part 5: Tensor Cores — the ML accelerators

Starting with Volta (2017), NVIDIA added **Tensor Cores** — specialized hardware that does one operation: **a small matrix multiply-accumulate**.

A single tensor core operation on H100 computes:

```
D = A · B + C
```

Where `A` is `(M, K)`, `B` is `(K, N)`, `C, D` are `(M, N)` and (typical sizes) M = N = K = 16 or 8.

The whole matmul fires in **a few clock cycles** for tens of fp16 multiply-adds. Compared to doing the same math one element at a time on CUDA cores, the speedup is 10-30×.

Tensor cores accept progressively lower-precision inputs each generation:

| Generation | Precisions |
|---|---|
| Volta (V100) | fp16 |
| Turing (T4, RTX 20xx) | fp16, int8, int4 |
| Ampere (A100, RTX 30xx) | bf16, TF32, int4, int1, **sparsity 2:4** |
| Hopper (H100) | bf16, TF32, **fp8 (E4M3, E5M2)** |
| Blackwell (B200) | bf16, **fp4** + transformer engine v2 |

**Why this matters for ML:** the published "TFLOPS" of a GPU is almost always the *tensor core* number for some precision. An H100's fp16 tensor-core throughput is ~1000 TFLOPS; its fp32 CUDA-core throughput is only ~60 TFLOPS. **17× faster — that's why we train in bf16/fp8.**

---

## Part 6: NVIDIA generations — what changed when

Reading a spec sheet means knowing what generation you're on. A quick reference:

| Year | Arch | Chips | What was new |
|---|---|---|---|
| 2017 | **Volta** | V100, Titan V | First tensor cores |
| 2018 | **Turing** | T4, RTX 2080 | RT cores; int8 tensor cores |
| 2020 | **Ampere** | A100, RTX 3090/A6000 | bf16 + TF32 + 2:4 sparsity; 3rd-gen tensor cores; MIG (multi-instance GPU) |
| 2022 | **Ada Lovelace** | RTX 4090, L40 | Consumer-focused; same compute capability 8.9 as Hopper compute cap 9.0 |
| 2022 | **Hopper** | H100, H200 | fp8 + Transformer Engine; thread block clusters; tensor memory accelerator |
| 2024+ | **Blackwell** | B100, B200, GB200 | fp4; 2nd-gen Transformer Engine; massive HBM3e |

**As of 2026:**
- **H100** is the production workhorse for training and high-volume inference
- **A100** still abundant on rental clouds; ~3× slower than H100 for fp16 matmul
- **B200** rolling out as the new flagship; ~3× faster than H100
- **L40** / RTX **6000 Ada** for inference and small-fleet training
- **RTX 3090/4090** are the "fits under your desk" choice — 24 GB VRAM, decent fp16/bf16 throughput, no nvlink

For learners: **a T4 (free Colab) handles weeks 13-16 fully.** An RTX 3090 / 4090 if you have one is great for the lab work.

---

## Part 7: The roofline model

The single most useful tool for reasoning about kernel performance.

The roofline says: a kernel's max performance is bounded by **either compute throughput or memory bandwidth**, whichever runs out first.

The crossover point is determined by **arithmetic intensity** — the number of FLOPs done per byte of memory loaded.

```
Y axis: GFLOPS achieved
X axis: arithmetic intensity (FLOPs / byte)

   peak compute ─────────────────────────
                  ╱
                 ╱   roof (memory bandwidth × intensity)
                ╱
               ╱
              ╱
              └───────────────────────────────
              compute-bound region →
              ← memory-bound region
```

For an H100:

- Peak compute (bf16 tensor): ~1000 TFLOPS = 10^15 ops/s
- Peak bandwidth: 3 TB/s = 3×10^12 bytes/s
- Crossover intensity: 1000 / 3 = **~333 FLOPs/byte**

**What this means for ML kernels:**

| Kernel | Intensity | Region |
|---|---|---|
| Elementwise add (`Y = X + b`) | 1 FLOP / 12 bytes | Severely memory-bound (×30 below crossover) |
| Softmax | ~3 FLOPs / 8 bytes | Memory-bound |
| LayerNorm | ~10 FLOPs / 8 bytes | Memory-bound |
| Attention (QK^T) at typical seq | ~10-50 FLOPs / byte | Memory-bound |
| Large matmul (M=N=K=4096) | ~700 FLOPs / byte | Compute-bound (above crossover) |

**Most transformer ops are memory-bound.** Big matmuls are compute-bound. This is why kernel-fusion (turning many small memory-bound ops into one) and FlashAttention (restructuring attention to be more compute-bound) are the dominant optimizations.

---

## Part 8: Why memory bandwidth dominates ML perf

When you train a transformer:

1. **Big matmuls** (attention QKV+O, FFN) — compute-bound. Tensor cores do their job. Pull in batches of weights and activations from HBM, do M×N×K ops on them.
2. **Everything else** (LayerNorm, softmax, GELU, elementwise residuals, dropout) — **memory-bound**. Almost no math; just shuffling bytes.

The matmuls hit ~50-70% of peak tensor-core throughput on a well-tuned kernel. The memory-bound ops hit ~70-90% of peak memory bandwidth.

If you sum the **time** spent in each, the memory-bound ops are typically 20-40% of total runtime. **They're tiny in FLOPs but expensive in time.**

The optimization play: **fuse adjacent memory-bound ops into one kernel** so you read inputs once and write outputs once instead of N times. PyTorch's `torch.compile` does this automatically; the FlashAttention paper does it by hand for the attention specifically.

You'll see this fusion math come alive in week 15.

---

## Part 9: GPU communication — NVLink and InfiniBand

A single GPU isn't enough for serious training. Multi-GPU adds another layer:

| Connection | Bandwidth |
|---|---|
| PCIe 5.0 (GPU ↔ CPU) | 128 GB/s per direction |
| NVLink 4 (intra-node H100) | 900 GB/s per direction |
| InfiniBand HDR (across nodes) | 200 Gb/s = 25 GB/s |
| InfiniBand NDR (modern) | 400 Gb/s = 50 GB/s |

A DGX H100 has **8× H100 GPUs all connected to each other at 900 GB/s** via NVLink. That's why intra-node training (DDP, FSDP within a box) is fast — gradients all-reduce at HBM bandwidth, not network bandwidth.

Across nodes you drop to InfiniBand speeds — ~10-30× slower. **This is why large clusters use 3D parallelism**: tensor parallelism for intra-node (high bandwidth needed), data parallelism for cross-node (lower bandwidth fine for batched gradients), pipeline parallelism for very deep models.

You won't write code that calls NCCL directly in this curriculum, but knowing the topology is what makes deployment decisions intelligent. `nvidia-smi topo -m` shows you your physical topology.

---

## Part 10: nvidia-smi and friends

Four commands you'll run constantly:

```bash
# What GPUs do I have, what's their utilization?
nvidia-smi

# Watch GPU utilization live (every second)
watch -n 1 nvidia-smi

# Show inter-GPU connections (NVLink? PCIe?)
nvidia-smi topo -m

# Per-process memory usage on the GPU
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# CUDA toolkit version
nvcc --version

# Driver version
nvidia-smi --query-gpu=driver_version --format=csv
```

For programmatic access (in Python):

```python
import torch
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.get_device_name(0)
torch.cuda.get_device_capability(0)        # e.g. (9, 0) for H100
torch.cuda.get_device_properties(0)         # full prop struct
torch.cuda.mem_get_info()                   # (free, total) in bytes
torch.cuda.max_memory_allocated()           # peak allocator usage
```

Get fluent at reading these. **Most "why is my training slow?" diagnoses start with `nvidia-smi`.**

---

## Part 11: A reading of an H100 spec sheet

Putting it all together — what each number on the H100 datasheet means and why you should care:

| Spec | H100 SXM5 | What it tells you |
|---|---|---|
| SMs | 132 | How much work can run in parallel |
| FP32 cores | 16,896 | (132 × 128) — peak fp32 CUDA throughput |
| Tensor cores | 528 | (132 × 4) — peak ML throughput |
| Peak BF16 tensor TFLOPS | 989 | Most relevant single number for ML training |
| Peak FP8 tensor TFLOPS | 1,979 | Modern training and inference |
| Peak FP32 (CUDA core) TFLOPS | 67 | Almost never the relevant number for ML |
| HBM3 memory | 80 GB | How big a model can fit |
| HBM bandwidth | 3.35 TB/s | The bottleneck for most ML kernels |
| L2 cache | 50 MB | Reuse window for tiled algorithms |
| NVLink 4 | 900 GB/s | Speed of multi-GPU within a node |
| TDP | 700 W | Power per GPU; matters for cluster cooling |

You'll see "989 TFLOPS bf16" in marketing. That's only achievable with a sustained large dense matmul where bandwidth doesn't bottleneck. **Real workloads usually run at 30-60% of that.** Getting closer is the job of weeks 14-15.

---

## What's next

In [lab.md](lab.md) you'll:

- Identify what GPU you have and read its spec sheet for real numbers
- Calculate the theoretical roofline of your GPU
- Run a series of matmul benchmarks to see what fraction of peak you actually hit
- Place common ML ops on the roofline diagram
- Compare GPUs (yours vs T4 vs A100 vs H100) in TFLOPS, bandwidth, and your $/hour cost

By end of week 13 you'll have the mental model. Week 14 is when you actually write CUDA C++ that goes through it.
