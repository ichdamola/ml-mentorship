# Week 15: Theory — Profiling and Performance

You wrote kernels in week 14. This week you learn what to **stop** writing because they aren't your bottleneck. **Never optimize without profiling first.** The single most reliable senior-engineering move in ML perf work.

The two tools you'll master: PyTorch's built-in profiler for the application layer, and NVIDIA's Nsight Systems/Compute for the kernel layer. Plus `torch.compile` — the automation that subsumes 80% of the manual fusion you might otherwise write.

By the end you'll be able to look at a training run, identify the three slowest kernels, decide whether each is compute-bound or memory-bound, and pick the right optimization (fusion / `torch.compile` / quantization / write a kernel).

---

## Part 1: The hierarchy of profiling tools

Different scales need different tools:

| Tool | What it shows | When |
|---|---|---|
| `time.perf_counter()` | One measurement | A quick sanity check; don't conclude from this |
| `torch.profiler` | Op-level CPU + CUDA time, memory, FLOPs | First pass — where in PyTorch is time going? |
| `nsys` (Nsight Systems) | System-wide timeline of CUDA kernels, NVTX ranges, NCCL | Identify what kernel is slow + when it happens |
| `ncu` (Nsight Compute) | Per-kernel HW counters: occupancy, mem bandwidth, roofline | Diagnose **why** a specific kernel is slow |
| `torch.utils.bottleneck` | Wrapper around cProfile + autograd | Python-level overhead |

The standard workflow:

```
1. torch.profiler  →  "the slow part is the FFN"
2. nsys            →  "specifically, the bias-add kernel after the matmul"
3. ncu             →  "it's memory-bound at 80% of peak bandwidth — won't get faster"
                       OR
                      "it's compute-bound at 20% of peak — try fusion"
```

Don't jump straight to `ncu`. Find the slow op first, then drill in.

---

## Part 2: PyTorch profiler

Built in. Free. Should be your first move.

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
) as prof:
    with record_function("training_step"):
        for X, y in loader:
            opt.zero_grad(set_to_none=True)
            with record_function("forward"):
                logits = model(X)
                loss = loss_fn(logits, y)
            with record_function("backward"):
                loss.backward()
            opt.step()
            break  # one step is enough

# Console summary, sorted by self CUDA time
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

# Save trace for Chrome
prof.export_chrome_trace("trace.json")
```

Open `trace.json` in Chrome via `chrome://tracing` (or [perfetto.dev](https://ui.perfetto.dev/)). You'll see a flame-chart timeline of every op.

### Reading the table

The key columns:

| Column | Meaning |
|---|---|
| Self CUDA | Time **just this op** spent on GPU (excludes children) |
| CUDA total | Time including children — usually bigger than Self CUDA |
| Self CPU | Time on CPU — Python overhead, kernel launches |
| # of Calls | How many times the op ran |

**Sort by Self CUDA time.** The top 5-10 rows are where to focus. Anything below the top 10 isn't your problem.

### Top patterns to recognize

- **`aten::addmm` / `aten::linear` dominating** → matmul-bound. Probably fine; maybe upgrade dtype or batch size.
- **`aten::softmax` + `aten::contiguous` appearing together** → an unfused attention pattern.
- **Lots of small `aten::cat` / `aten::reshape`** → maybe a Python loop. Try `torch.compile` or refactor.
- **Many tiny kernel launches with `cuLaunchKernel` overhead** → launch latency dominates. Fuse or use bigger batches.

### CPU-side overhead

`Self CPU` matters when:
- You're doing too much Python work between GPU calls (use bigger batches)
- DataLoader is slow (increase `num_workers`)
- You're synchronizing too often (avoid `.item()`, `.cpu()` in hot loops)

The pattern: CPU launches kernels much faster than GPU executes them, so launches queue up. **Until they don't** — and then GPU goes idle. The profiler shows the gaps.

---

## Part 3: Nsight Systems (`nsys`)

System-wide timeline. Gives you a Gantt chart of CUDA kernels, memory transfers, CPU calls, and NCCL communication.

```bash
nsys profile \
  --output=my_run \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  python train.py
```

Then open `my_run.nsys-rep` in **Nsight Systems GUI** (Windows / macOS / Linux). You'll see:

```
CPU thread 0:    ████████████ Python code ████████ ████ ████
CUDA stream 0:        ████ kernel_A ████  ┃ ████ kernel_B ████  ┃ ████
NCCL:                                     ┃ all-reduce               ┃
HBM activity:    ░░ read ░░ write    ░░░░░░░░░░░░░░░░░
```

Looking for:
- **Gaps in CUDA stream** — GPU idle. Why?
- **NCCL bars covering training** — communication-bound (in DDP)
- **Memory bars filling the timeline** — bandwidth saturated
- **Tiny kernel bars far apart** — launch overhead, fuse them

Pair with **NVTX ranges** to annotate your code:

```python
import torch.cuda.nvtx as nvtx

nvtx.range_push("forward")
loss = model(X)
nvtx.range_pop()

nvtx.range_push("backward")
loss.backward()
nvtx.range_pop()
```

These ranges appear as named bars in the timeline. You can find your `forward` block instantly even in a noisy profile.

---

## Part 4: Nsight Compute (`ncu`) — per-kernel deep dive

Slowest tool, deepest answers. Picks **one kernel** and reports hardware counters.

```bash
ncu --set full --target-processes all -o my_kernel python my_script.py
```

Or attach to a Python script and profile only specific kernels:

```bash
ncu --kernel-name "matmul" \
    --launch-skip 5 \
    --launch-count 1 \
    -o my_matmul \
    python my_script.py
```

Open `my_matmul.ncu-rep` in **Nsight Compute GUI**.

### Key metrics to read

| Section | What it tells you |
|---|---|
| **GPU Speed of Light** | Compute throughput % and memory throughput % — which is your bottleneck |
| **Memory Workload Analysis** | Coalescing, L1/L2 hit rates, HBM bandwidth |
| **Compute Workload Analysis** | Tensor core utilization, fp32 utilization, occupancy |
| **Warp State Statistics** | Why warps stalled — memory, sync, math |
| **Source Counters** | Line-by-line bottleneck for hot loops (where in your kernel time is spent) |

The **Speed of Light** view is the first thing you should read for any kernel:

```
Compute (SM)  Throughput   : 8.4 %       ← only 8% of compute used
Memory        Throughput   : 89.3 %      ← almost saturated bandwidth
```

This kernel is **memory-bound**. No amount of better math will help. Only data-movement reduction (fusion, dtype shrink) will.

The opposite case:

```
Compute (SM)  Throughput   : 88 %
Memory        Throughput   : 12 %
```

This is **compute-bound**. You're hitting the FLOPs ceiling. Quantizing or moving to fp8 / int8 can help; restructuring the algorithm probably won't.

---

## Part 5: The roofline model — applied

Recall week 13: every kernel sits on a 2-axis plot with arithmetic intensity (FLOPs/byte) on x and achievable TFLOPS on y.

`ncu` will literally **plot the kernel on the roofline** for you in the "Roofline Analysis" section. Three regions:

```
                                  ╱─── compute ceiling
                                 ╱
                                ╱
              ╱── memory roof ╱   ← compute-bound region (kernels here)
             ╱               ╱
            ╱               ╱
           ╱        ──────  ← memory-bound region (kernels here)
          ╱       ╱
         ╱       ╱
        ╱       ╱
       ╱       ╱
      └────────────────────────────────  intensity
```

If your kernel is well below either roof, look at occupancy and stalls (Part 6). It's likely under-using the SM.

If it's near the memory roof, accept the memory bound and reduce traffic. If it's near the compute roof, accept that and use lower-precision arithmetic if applicable.

---

## Part 6: Occupancy — "how many warps can run on this SM?"

The SM has resources (registers, shared memory, threads) shared by all blocks running on it. If your block uses 64 registers × 256 threads, but the SM has 65,536 registers, only 4 blocks fit at most. With 4 blocks × 8 warps = 32 active warps. SM max is 64 warps. **Occupancy: 50%.**

Occupancy `< 50%` means the SM is under-utilized — when one warp stalls (memory, sync), there aren't enough other warps to hide the stall.

`ncu` reports:
- **Theoretical occupancy** — best case given your block's resource usage
- **Achieved occupancy** — what actually happened (usually lower)

If achieved is much less than theoretical, you have warp-divergence or sync-stall issues to debug.

### Knobs

- **Reduce registers per thread** (compile with `--maxrregcount=N`) → more blocks fit
- **Reduce shared memory per block** → more blocks fit
- **Reduce thread count per block** → more blocks fit, more total warps

**Tradeoff:** more occupancy isn't always better. If higher occupancy spills registers to local memory (slow), it's worse. Modern kernels (cuBLAS, FlashAttention) often run at 25-40% occupancy because they're tuned to use registers maximally.

---

## Part 7: Memory bandwidth bound — the dominant case in ML

Most non-matmul kernels (LayerNorm, softmax, GeLU, residual add, embedding lookup) are memory-bound. They run at 80-95% of HBM bandwidth and there's nothing you can do about a single kernel.

**The wins come from fusing them.** If you have `Y = ReLU(LN(X + residual))`, doing it as three separate kernels reads X+residual five times and writes intermediate Y twice. **Fused** it's one read and one write — 3-5× faster.

### Fusion patterns

| Pattern | Example |
|---|---|
| Activation + linear | `linear → relu` fused |
| Residual + norm | `x + residual → LayerNorm` fused |
| Bias + activation | `linear → bias → gelu` |
| Attention end-to-end | FlashAttention fuses softmax + matmul + dropout |

These are **the** optimization in ML inference. `torch.compile` and `xformers` and `vllm` do this automatically; FlashAttention and FasterTransformer do it by hand.

---

## Part 8: `torch.compile` — automated fusion

[Added in PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/), `torch.compile` traces your model graph, optimizes it via Inductor + Triton, and produces fused kernels.

```python
model = MyModel().cuda()
model = torch.compile(model)        # adds JIT compilation

# First batch is slow (compiles); subsequent are fast
out = model(X)
```

What it does:
- Fuses adjacent pointwise ops into a single kernel
- Replaces small CPU-side ops with CUDA equivalents
- Uses Inductor's library of optimized matmul + attention kernels
- Generates Triton kernels for fused pointwise sequences

Speedup is typically:
- **1.3-2× for training** on transformer models
- **2-4× for inference**
- **Sometimes slower** on small models where JIT overhead dominates

`torch.compile` should be **the first optimization you try** after a profile shows you're op-launch-overhead-bound or have fusable patterns. It's free and reversible.

### Modes

```python
model = torch.compile(model, mode="reduce-overhead")     # caches CUDA graphs
model = torch.compile(model, mode="max-autotune")        # tries hard to find best kernels
model = torch.compile(model, fullgraph=True)             # errors instead of falling back
```

`max-autotune` can be much faster but compiles slowly. `reduce-overhead` is good for inference servers. `fullgraph=True` forces compilation of the whole graph (errors on unsupported constructs); use during development to find non-traceable code.

---

## Part 9: Activation memory and gradient checkpointing in production

Week 11 covered the theory. Here's the application:

When `ncu` shows your forward pass is held up because activations don't fit, **trade compute for memory**:

```python
import torch.utils.checkpoint as ckpt
for block in self.blocks:
    x = ckpt.checkpoint(block, x, use_reentrant=False)
```

Trade: ~30% more compute time, 50-80% less activation memory. Used when:
- Training a model that won't fit otherwise
- Training with very long context

Don't use when you have memory to spare — pure waste.

---

## Part 10: Quantization for inference

For **inference** (not training), you can run in lower precision than you trained:

| Precision | Speedup vs fp32 | Quality cost |
|---|---|---|
| bf16 | 2-3× | ~0 |
| fp16 | 2-3× | ~0 |
| INT8 weight + bf16 act | 1.5-2× more vs bf16 | Small (<1%) for most tasks |
| INT4 weight only (GPTQ/AWQ) | 2-3× more | Small to medium |
| FP8 (H100+) | ~2× more than bf16 | ~0 with calibration |

For training, fp8 is on the frontier; for **inference**, INT8/INT4 is mainstream.

Quantization recipes are in week 16. Mention here because **performance profiling reveals where quantization will pay off**: linear layers and matmuls dominate compute → quantizing them is the highest-ROI inference optimization.

---

## Part 11: The performance diagnosis playbook

When someone says "training is slow," here's the order:

1. **`nvidia-smi -l 1`** — Is GPU even busy? (Often it's not.)
2. **PyTorch profiler** — Where in your code is time going?
3. **Look at top 5 ops by Self CUDA time.**
   - Dominated by matmul → it's working as designed; consider AMP / larger batch
   - Dominated by elementwise → enable `torch.compile`
   - Dominated by attention → use SDPA / FlashAttention
   - Tiny ops with big launch overhead → fuse or use bigger batches
4. **If still slow, `nsys`** — Are there big gaps in the timeline? (data loading? CPU bottleneck? all-reduce?)
5. **If a specific kernel is slow, `ncu`** — Is it bandwidth-bound or compute-bound?

90% of "slow training" complaints are solved at step 3 with `torch.compile` + AMP + larger batch size. The other 10% need real kernel work.

---

## Part 12: FlashAttention — the canonical fused kernel

The paradigmatic example of how fusion at scale beats any single kernel. Standard attention:

```python
S = Q @ K.T / sqrt(d)      # (N, N) — fits in HBM but is huge
P = softmax(S, dim=-1)      # (N, N)
O = P @ V                  # (N, d)
```

For seq length N=8192, `S` is 256 MB per head. Cannot fit in SMEM. Must round-trip to HBM at least 4 times (S out, P in, then P out, V in, O in).

FlashAttention's insight: **don't materialize S or P at all**. Process Q, K, V in tiles that fit in SMEM. For each tile of Q:

1. Initialize running `O = 0`, running `m = -inf`, running `l = 0` (softmax denominator)
2. For each tile of K, V:
   - Compute partial `S_tile = Q_tile @ K_tile.T / sqrt(d)`
   - Update running `m_new = max(m, max(S_tile))`
   - Update running `l_new` and `O_new` using `S_tile` and `V_tile`

When done, `O` is correct. Never wrote the full `(N, N)` S or P to HBM.

Memory traffic drops from `O(N²)` to `O(N×d)`. For seq=8192, d=128 head: from 256 MB to 4 MB. **64× less HBM traffic.**

The trade: **forward FLOPs are unchanged** — FlashAttention reorders the loops, it doesn't add work to the forward pass. The backward does **recompute** the attention scores (~2× backward arithmetic) so that S/P never have to be stashed for autograd. The headline win is **memory traffic** (and therefore wall-clock on memory-bound kernels), not raw FLOPs. Net result: 2-4× faster + dramatically less memory + enables long contexts that were impossible before.

[FlashAttention-2](https://arxiv.org/abs/2307.08691) further reorders the loops to improve work distribution across SMs. FlashAttention-3 (2024) adds Hopper-specific async copies and fp8 paths.

This is the case study of every concept this week: **profile to find the bottleneck, recognize the memory-bound pattern, fuse mercilessly.** The result is a kernel that defines an entire class of modern ML systems.

---

## What's next

In [lab.md](lab.md) you'll:

- Profile a forward + backward pass of your week-10 transformer with `torch.profiler`
- Read the Chrome trace; identify the 3 slowest ops
- Annotate code with NVTX ranges
- Run `nsys` (CLI; no GUI required for measurement)
- Apply `torch.compile` and measure the speedup
- Compare regular attention vs `F.scaled_dot_product_attention` (which dispatches to FlashAttention)
- Apply gradient checkpointing; verify memory savings
- (Stretch) Read the Nsight Compute output for one kernel and report its memory-bound vs compute-bound classification

By end of week 15 you have the playbook for "is this model slow, and what should I do?" Week 16 takes everything and serves it in production.
