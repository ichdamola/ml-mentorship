# Week 13: Lab — Read Your GPU

This lab is mostly measurement, not coding. By the end you'll have your GPU's spec sheet in your head and a roofline diagram showing where common ML ops sit. That mental model is what makes weeks 14-15's optimization work intelligible.

## Setup

You need a CUDA GPU. If you don't have one locally:

| Option | Tier |
|---|---|
| Google Colab (free T4) | Good for everything in this lab |
| Colab Pro (A100/V100 at times) | Better |
| vast.ai / runpod / lambda labs (H100, B200) | Best, costs ~$2-4/hr |

```python
import torch
import math, time
import numpy as np
import matplotlib.pyplot as plt

assert torch.cuda.is_available(), "This lab requires a CUDA GPU."
device = torch.device("cuda")
```

---

## Exercise 13.1 — Identify your GPU

```python
print(f"GPU name:               {torch.cuda.get_device_name(0)}")
props = torch.cuda.get_device_properties(0)
print(f"Compute capability:     {props.major}.{props.minor}")
print(f"SMs (multiProcessors):  {props.multi_processor_count}")
print(f"VRAM:                    {props.total_memory / 1e9:.1f} GB")
print(f"Max threads per SM:      {props.max_threads_per_multi_processor}")
print(f"Max threads per block:   {props.max_threads_per_block}")
print(f"Shared mem per block:    {props.shared_memory_per_block / 1024:.0f} KB")
print(f"Warp size:               {props.warp_size}")
print(f"L2 cache:                {props.L2_cache_size / 1e6:.1f} MB")
print(f"Memory bus width:        {props.memory_bus_width} bits")
print(f"Memory clock (max):      {props.memory_clock_rate / 1e6:.2f} GHz")
print(f"SM clock (max):          {props.clock_rate / 1e6:.2f} GHz")
print(f"bf16 supported:          {torch.cuda.is_bf16_supported()}")

# Compute theoretical HBM bandwidth from clock × bus width
peak_bw = 2 * props.memory_clock_rate * 1e3 * props.memory_bus_width / 8 / 1e9
print(f"\nTheoretical HBM BW:      {peak_bw:.0f} GB/s")
```

Compare with the manufacturer's spec sheet (NVIDIA H100 product brief, RTX 4090 / 3090 datasheets, etc.). The numbers above are what *your* card reports; the published numbers may differ slightly (boost clocks, etc.).

---

## Exercise 13.2 — Lookup peak TFLOPS for your GPU

PyTorch doesn't expose tensor-core peaks directly. Quick reference (peak, theoretical, vendor-stated):

| GPU | BF16 tensor TFLOPS | FP8 tensor TFLOPS | HBM bandwidth |
|---|---|---|---|
| T4 (Turing) | 65 (fp16) | — | 320 GB/s |
| V100 (Volta) | 125 (fp16) | — | 900 GB/s |
| A100 80GB | 312 (bf16) | — | 2 TB/s |
| RTX 3090 | 142 (fp16) | — | 936 GB/s |
| RTX 4090 | 165 (fp16/bf16) | — | 1 TB/s |
| A6000 | 155 (bf16) | — | 768 GB/s |
| L40 | 181 (bf16) | 362 | 864 GB/s |
| H100 SXM5 | 989 (bf16) | 1979 | 3.35 TB/s |
| B200 | 2200 (bf16) | 4500 | 8 TB/s |

Note your card's numbers. We'll use them as the "roof" of the roofline.

```python
# Set these for your GPU
PEAK_BF16_TFLOPS = 165       # ← change to match your card
PEAK_HBM_GB_PER_S = 1000      # ← change to match your card
```

---

## Exercise 13.3 — Measure actual matmul throughput

Run a big matmul, time it, see what fraction of peak you hit.

```python
def matmul_tflops(M, N, K, dtype=torch.bfloat16, n_iter=50):
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)

    # Warmup
    for _ in range(5):
        C = A @ B
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        C = A @ B
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_iter

    flops = 2 * M * N * K   # 2 ops (mul + add) per output element
    return flops / dt / 1e12     # TFLOPS

sizes = [256, 512, 1024, 2048, 4096, 8192]
print(f"{'size':>6s}  {'TFLOPS':>8s}  {'% peak':>8s}")
for s in sizes:
    tf = matmul_tflops(s, s, s)
    print(f"{s:>6d}  {tf:>8.1f}  {100*tf/PEAK_BF16_TFLOPS:>7.1f}%")
```

You should see:

- Small matmuls (256-512) hit only ~10-30% of peak — overhead dominates
- Medium matmuls (1024-2048) hit ~40-60%
- Large matmuls (4096-8192) hit **70-85%** — this is what production cuBLAS hits

**Real PyTorch transformer training is bound by these large-matmul speeds**, not by the marketing-spec peak. The 989 TFLOPS on the H100 spec sheet is achievable only on enormous matrices in just the right precision.

---

## Exercise 13.4 — Measure actual HBM bandwidth

```python
def bandwidth_gb_s(N, dtype=torch.bfloat16, n_iter=50):
    """Measure bandwidth via a copy."""
    src = torch.randn(N, device=device, dtype=dtype)
    dst = torch.empty_like(src)

    for _ in range(5):
        dst.copy_(src)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        dst.copy_(src)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_iter

    bytes_moved = 2 * N * src.element_size()    # read + write
    return bytes_moved / dt / 1e9     # GB/s

for n_mb in [1, 16, 64, 256, 1024]:
    N = n_mb * 1024 * 1024 // 2    # 2 bytes per bf16 element
    bw = bandwidth_gb_s(N)
    print(f"{n_mb:>5d} MB:  {bw:>7.1f} GB/s  ({100*bw/PEAK_HBM_GB_PER_S:>5.1f}% of peak)")
```

You should see bandwidth approach peak as transfer size grows. Tiny transfers don't amortize launch overhead; large ones approach the theoretical max.

---

## Exercise 13.5 — Place common ops on the roofline

Calculate arithmetic intensity (FLOPs / byte) for several ML ops and plot them.

```python
# Crossover intensity for your GPU
crossover = PEAK_BF16_TFLOPS * 1e12 / (PEAK_HBM_GB_PER_S * 1e9)
print(f"Crossover (compute = bandwidth × intensity) at: {crossover:.1f} FLOP/byte")

# Common ML ops — (name, intensity)
ops = [
    ("elementwise add", 1 / 12),                       # 1 FLOP per 12 bytes (3 fp32 = 12)
    ("LayerNorm fwd", 10 / 8),                          # ~10 FLOP per 8 bytes (2 bf16)
    ("softmax fwd", 3 / 8),
    ("Attention QK^T (seq=512)", 1024 / 4),            # rough — bottleneck depends
    ("Matmul 256×256×256",      2 * 256 / (4 * 3)),
    ("Matmul 1024×1024×1024",   2 * 1024 / (4 * 3)),
    ("Matmul 4096×4096×4096",   2 * 4096 / (4 * 3)),
]

# Plot the roofline
intensities = np.logspace(-2, 3, 200)
mem_bound = PEAK_HBM_GB_PER_S * intensities / 1000        # GB/s × FLOPs/byte → TFLOPS
roof = np.minimum(mem_bound, PEAK_BF16_TFLOPS)

plt.figure(figsize=(10, 6))
plt.loglog(intensities, roof, 'b-', linewidth=2, label='roofline')
plt.axhline(PEAK_BF16_TFLOPS, ls='--', color='gray', alpha=0.5, label=f'compute peak ({PEAK_BF16_TFLOPS} TFLOPS)')
plt.axvline(crossover, ls=':', color='gray', alpha=0.5, label=f'crossover @ {crossover:.0f} FLOP/byte')
for name, ai in ops:
    achievable = min(ai * PEAK_HBM_GB_PER_S / 1000, PEAK_BF16_TFLOPS)
    plt.scatter(ai, achievable, s=80, zorder=5)
    plt.annotate(name, (ai, achievable), fontsize=8, xytext=(5, 5), textcoords='offset points')
plt.xlabel("Arithmetic Intensity (FLOPs / byte)")
plt.ylabel("Achievable Performance (TFLOPS)")
plt.title(f"Roofline diagram — {torch.cuda.get_device_name(0)}")
plt.legend()
plt.grid(alpha=0.3, which='both')
plt.show()
```

**What this tells you:**

- Memory-bound ops (LayerNorm, softmax, elementwise) sit far below peak. **No matter how clever your kernel, you can't beat the memory wall.**
- Small matmuls are also memory-bound (the matrices fit in cache; per-byte FLOPs are low).
- Large matmuls cross over and hit compute peak.
- **The way to make memory-bound code faster is to do less of it — fuse adjacent ops together.** That's the FlashAttention trick.

---

## Exercise 13.6 — Memory coalescing demo

Show the cost of bad memory access patterns.

```python
def benchmark_kernel_pattern(stride, n_iter=100):
    """Sum a large array with a given stride."""
    N = 1 << 24   # 16M
    X = torch.ones(N, device=device, dtype=torch.float32)

    # Build an "index" tensor with the given stride
    idx = torch.arange(0, N, stride, device=device, dtype=torch.int64)
    if len(idx) > N // stride:
        idx = idx[:N // stride]

    for _ in range(5):
        s = X[idx].sum()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        s = X[idx].sum()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter

for stride in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    t = benchmark_kernel_pattern(stride) * 1000
    print(f"stride={stride:>4d}: {t:.3f} ms")
```

You should see **strided access being substantially slower than contiguous**, even when reading the same number of elements. This is the cost of memory non-coalescing.

In real ML code: this is why `tensor.contiguous()` matters before some ops, and why FlashAttention works hard to keep memory accesses aligned.

---

## Exercise 13.7 — Compare GPUs

Build a quick comparison table for the GPU you have vs the most common cloud options.

```python
def compute_to_bandwidth_ratio(peak_tflops, peak_gb_s):
    """FLOPs / byte — the crossover point."""
    return peak_tflops * 1e12 / (peak_gb_s * 1e9)

gpus = [
    ("T4",        65,    320,  0.35),    # GB/s and rough $/hr
    ("V100",      125,   900,  1.0),
    ("A100 80GB", 312,   2000, 1.5),
    ("RTX 4090",  165,   1000, None),     # not rented hourly
    ("L40",       181,   864,  None),
    ("H100",      989,   3350, 4.0),
    ("B200",      2200,  8000, None),     # too new to price cleanly
]

print(f"{'GPU':<12s}  {'BF16 TFLOPS':>12s}  {'HBM GB/s':>10s}  {'crossover':>11s}  {'~$/hr':>7s}")
for name, tf, bw, price in gpus:
    print(f"{name:<12s}  {tf:>12.0f}  {bw:>10.0f}  {compute_to_bandwidth_ratio(tf, bw):>11.0f}  {price if price else '-':>7}")
```

What you'll see:
- The H100 and B200 have similar crossover points to older cards (~300-350 FLOP/byte) — they scaled compute AND bandwidth together
- The crossover stays surprisingly stable. **The "memory-bound" regime is intrinsic to ML, not a fixable architectural choice.**
- $/hr does not scale 1:1 with raw TFLOPS. An H100 is ~3× faster than an A100 but costs ~2.5× more. **At-cost ML training is reasonably priced; consumer-rent paid markups dominate.**

---

## Exercise 13.8 — Read `nvidia-smi` topology

Run in your shell or via `os.system`:

```python
import os
print("\n=== nvidia-smi ===")
os.system("nvidia-smi")
print("\n=== topology ===")
os.system("nvidia-smi topo -m")
```

On a single-GPU box, the topology is uninteresting. On a multi-GPU node, you'll see something like:

```
        GPU0    GPU1    GPU2    GPU3
GPU0     X      NV12    NV12    NV12
GPU1    NV12     X      NV12    NV12
GPU2    NV12    NV12     X      NV12
GPU3    NV12    NV12    NV12     X
```

`NV12` means 12 NVLink lanes between two GPUs — the fastest possible inter-GPU connection. `PIX` would mean PCIe-only — much slower. **If you're training multi-GPU, you want NVLink, not PCIe-only — gradient all-reduce time depends on the slowest link.**

---

## Exercise 13.9 — Predict and verify

For a `(4096, 4096) @ (4096, 4096)` bf16 matmul on your GPU:

1. Compute FLOPs: `2 * 4096^3 ≈ 137 GFLOPs`
2. Compute bytes to/from HBM: `2 input matrices + 1 output = 3 * 4096^2 * 2 bytes ≈ 100 MB`
3. Arithmetic intensity: `137e9 / 100e6 = 1370 FLOPs/byte`
4. Predict: compute-bound, achievable ≈ peak
5. Measured time = FLOPs / achieved TFLOPS

```python
M = N = K = 4096
flops_total = 2 * M * N * K                            # 137 GFLOPs
bytes_hbm = (M*K + K*N + M*N) * 2                       # ~100 MB
intensity = flops_total / bytes_hbm
print(f"Intensity: {intensity:.0f} FLOPs/byte")
print(f"Crossover for your GPU: {compute_to_bandwidth_ratio(PEAK_BF16_TFLOPS, PEAK_HBM_GB_PER_S):.0f}")

# Compute-bound or memory-bound?
if intensity > compute_to_bandwidth_ratio(PEAK_BF16_TFLOPS, PEAK_HBM_GB_PER_S):
    print(f"→ COMPUTE-BOUND. Achievable: ~{PEAK_BF16_TFLOPS} TFLOPS")
else:
    achievable_tflops = intensity * PEAK_HBM_GB_PER_S / 1000
    print(f"→ MEMORY-BOUND. Achievable: ~{achievable_tflops:.0f} TFLOPS")

# Measure
A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
B = torch.randn(K, N, device=device, dtype=torch.bfloat16)
for _ in range(5): A @ B
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50): A @ B
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 50
measured = flops_total / dt / 1e12
print(f"\nMeasured: {measured:.1f} TFLOPS ({100*measured/PEAK_BF16_TFLOPS:.0f}% of peak)")
```

If your prediction was "compute-bound, ~70% of peak achievable", you should see actually ~60-80% of peak. **Now you can reason about kernel performance before you run it.**

---

## Submission checklist

- [ ] GPU identified; spec numbers noted (SMs, VRAM, theoretical bandwidth)
- [ ] Peak BF16 TFLOPS and HBM bandwidth set for your specific GPU
- [ ] Matmul throughput measured across multiple sizes; large sizes hit ≥ 50% of peak
- [ ] HBM bandwidth measured; large transfers approach peak
- [ ] Roofline diagram plotted with at least 4 ML ops placed correctly
- [ ] Strided memory access shown to be slower than contiguous
- [ ] Multi-GPU comparison table including crossover points
- [ ] `nvidia-smi` topology examined
- [ ] At least one kernel time predicted and verified within 30% accuracy

---

## What you just did

You took the GPU from black-box to glass-box. You can read its spec sheet, calculate the theoretical roofline, measure where real ops sit on it, and predict whether a given kernel will be compute-bound or memory-bound.

**That mental model — roofline + memory hierarchy + tensor cores + warps — is the bedrock for weeks 14-15.** Week 14 you'll write CUDA C++ that respects this model. Week 15 you'll profile real models and find the slow ops to fuse.

---

**Next**: [Week 14: CUDA Programming →](../week-14-cuda-programming/readme.md)
