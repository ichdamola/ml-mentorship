# Week 11: Theory — Training at Scale

The 1.5M-parameter GPT from week 10 trains on a laptop. Llama-2-7B doesn't. Llama-3-405B doesn't fit on any single GPU in the world. The gap between toy and frontier is the *training systems* stack — mixed precision, gradient accumulation, sharded optimizers, distributed data parallel, FSDP.

This week is dense. You won't run frontier training, but you'll understand every line of a real training script.

---

## Part 1: The memory math

Training a model consumes memory for FOUR things, each with a different size:

| Bucket | Size for an N-param model in fp32 | Notes |
|---|---|---|
| **Parameters** | 4N bytes | Just the weights |
| **Gradients** | 4N bytes | Same shape as params |
| **Optimizer state** | 8N bytes (Adam) | First + second moment per param |
| **Activations** | depends on architecture | Stored for backward |

For an N-param model trained in fp32 with Adam:

```
Total memory ≈ 16 N bytes (params + grads + optim) + activations
```

A **1B-param model** at fp32 needs:
- 16 GB just for params/grads/optim
- Plus activations (typically another 5-50 GB depending on batch size and sequence length)

**An A100 (80 GB)** can train a 1-5B model comfortably. A 70B model can't fit on one A100 at all — even *one copy* of params + grads + optim is 1.1 TB.

This is why everything that follows exists.

---

## Part 2: fp32 → fp16 → bf16 → fp8 — the precision ladder

Lower precision = less memory + faster compute. The trade-off is numerical range and precision.

| Dtype | Bytes | Exponent / Mantissa | Range | Used for |
|---|---|---|---|---|
| fp32 | 4 | 8 / 23 | ±3e38 | Default training, "master weights" |
| fp16 | 2 | 5 / 10 | ±65k | Activations + forward (with care) |
| bf16 | 2 | 8 / 7 | ±3e38 (same as fp32) | Modern default for training |
| fp8 (E4M3) | 1 | 4 / 3 | ±448 | Forward (H100+) |
| fp8 (E5M2) | 1 | 5 / 2 | ±57k | Backward gradients (H100+) |

**bf16 vs fp16:** they're both 2 bytes. bf16 has the same dynamic *range* as fp32 (8-bit exponent) but less *precision* (7-bit mantissa). fp16 has the opposite. For training, bf16's wider range matters more — gradients can be tiny, and fp16 underflows them. **Default to bf16 if your hardware supports it (Ampere+, A100/A6000/H100/B200).**

fp8 is the modern frontier — H100 and B200 ship "transformer engine" hardware that automatically picks E4M3 (forward) vs E5M2 (backward) per tensor.

---

## Part 3: Mixed precision training (AMP)

The trick: do the heavy math in fp16/bf16 (fast, low memory), but keep a fp32 master copy of the weights to maintain training stability.

```python
import torch
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()         # only needed for fp16; bf16 doesn't need scaling

for X, y in loader:
    opt.zero_grad(set_to_none=True)

    with autocast(dtype=torch.bfloat16):    # forward + loss in bf16
        logits = model(X)
        loss = loss_fn(logits, y)

    # backward outside autocast — gradients accumulate in fp32 for AdamW
    if dtype == torch.float16:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    else:  # bf16 or fp32
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
```

### Loss scaling — only for fp16

fp16's gradient underflow problem: many gradients are smaller than fp16's smallest representable nonzero (~6e-5) and round to zero, killing learning.

**Fix:** multiply the loss by a large scalar (`scale = 2^16` typical) before backward. Gradients are now in fp16's expressible range. After backward, divide gradients by `scale` before the optimizer step.

`GradScaler` automates this, dynamically growing the scale and skipping steps where gradients overflow. **bf16 doesn't need this** — its 8-bit exponent already handles tiny gradients.

### When to choose fp16 vs bf16

- **Old GPUs (V100, T4)** → fp16 with `GradScaler`. They don't support bf16.
- **Ampere+ (A100, RTX 3090/4090, A6000)** → bf16, no scaler needed. Simpler, safer.
- **Hopper/Blackwell (H100, B200)** → fp8 via the transformer engine for forward, bf16 for backward.

**Speedup**: 2-4× over fp32, depending on model size and architecture. For most modern training: free win — always on.

---

## Part 4: Gradient accumulation

You want effective batch size 256 but GPU memory only fits 32. Solution: accumulate gradients over 8 micro-batches before updating.

```python
accum_steps = 8

opt.zero_grad(set_to_none=True)
for i, (X, y) in enumerate(loader):
    with autocast(dtype=torch.bfloat16):
        logits = model(X)
        loss = loss_fn(logits, y) / accum_steps    # scale by accum to keep gradient scale right
    loss.backward()

    if (i + 1) % accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
```

**Why divide by `accum_steps`?** Gradients accumulate across the 8 backward calls. Without the division, the effective gradient magnitude is 8× what you wanted, and your LR is implicitly 8× too high.

Trade-offs:

- **Pros:** larger effective batch size, more stable gradients (better generalization in transformer-land), no extra memory beyond one micro-batch
- **Cons:** slower wall-clock per "effective step" (you do 8 forward+backward for 1 optimizer step). Most of the time, that's a fair price.

---

## Part 5: Gradient checkpointing — trading compute for memory

Activations are stored from forward so backward can use them. For a deep transformer, this can dominate memory.

Gradient checkpointing: **drop activations after forward, recompute them during backward**. Trades ~30% extra compute for typically 50-80% memory savings.

```python
import torch.utils.checkpoint as ckpt

def forward(self, x):
    x = ckpt.checkpoint(self.block1, x, use_reentrant=False)
    x = ckpt.checkpoint(self.block2, x, use_reentrant=False)
    return x
```

Or apply at the block level for transformers:

```python
for block in self.blocks:
    x = ckpt.checkpoint(block, x, use_reentrant=False)
```

**When to use**: when activations are your bottleneck. Look at `torch.cuda.memory_allocated()` after forward (week 11 lab). If activations >> parameters, you'll gain a lot. If they're comparable, less so.

---

## Part 6: Distributed Data Parallel (DDP)

The simplest form of multi-GPU training: each GPU has a complete copy of the model, processes a different chunk of data, and after backward they **all-reduce gradients** to keep weights synchronized.

```
GPU 0: forward+backward on batch chunk 0  →  grads_0
GPU 1: forward+backward on batch chunk 1  →  grads_1
GPU 2: forward+backward on batch chunk 2  →  grads_2
GPU 3: forward+backward on batch chunk 3  →  grads_3

All-reduce: each GPU ends up with (grads_0 + grads_1 + grads_2 + grads_3) / 4

Each GPU runs optimizer step independently — weights stay in sync.
```

PyTorch wraps this in `DistributedDataParallel`:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Init the process group (usually via torchrun)
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

Launch with:

```bash
torchrun --nproc_per_node=8 train.py
```

### When DDP works

- Model fits on one GPU (params + grads + optim < GPU memory)
- You want **linear speedup with more GPUs**

Most production training of <10B-param models uses DDP.

### When DDP isn't enough

For 7B+ models, the params + grads + optim no longer fit on one GPU. DDP would OOM before forward. Time for FSDP.

---

## Part 7: FSDP — sharding params + grads + optimizer

**Fully Sharded Data Parallel** (FSDP) extends DDP by **sharding** the model state across GPUs. Each GPU only holds 1/N of:

- Parameters
- Gradients
- Optimizer state

For an N-param model on G GPUs in bf16+fp32-master+Adam, each GPU holds ~14N/G bytes (vs 14N for DDP). **Suddenly a 70B model fits.**

How it works (simplified):

1. **Before forward of layer L**: GPUs all-gather layer L's params (each holds a shard, briefly holds the full layer).
2. **Forward of layer L**: every GPU has the full params for this one layer.
3. **After forward**: free the full layer params; only the local shard remains.
4. **Same for backward**: gather, compute, free.

Memory per GPU stays low because only one layer at a time is "full." Communication is higher than DDP (every layer's params are gathered), but the math works out for large models.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
```

### DeepSpeed ZeRO — the same idea, packaged

[ZeRO](https://arxiv.org/abs/1910.02054) is Microsoft's library that implements the same sharding strategies (stages 1, 2, 3). FSDP is PyTorch's native version. ZeRO stage 3 ≈ FSDP `FULL_SHARD`. Either works; FSDP is more native to PyTorch, DeepSpeed has more features (ZeRO-offload to CPU/NVMe).

---

## Part 8: Tensor parallelism, pipeline parallelism (briefly)

For truly enormous models, you can also:

- **Tensor parallelism** — split a single layer's matmul across GPUs (e.g., Megatron-LM). Each GPU does part of the matrix multiplication; results all-reduced.
- **Pipeline parallelism** — different layers on different GPUs; activations pass through like an assembly line.

These are for 100B+ models being trained on hundreds of GPUs. For this curriculum, you'll know they exist; production work would have you using libraries (NVIDIA NeMo, Megatron-LM, DeepSpeed) that handle this.

**3D parallelism** combines all three (data + tensor + pipeline) to train >1T-param models on 10000+ GPUs. This is the Llama / GPT-4 / Gemini training stack.

---

## Part 9: Learning rate schedules at scale

For transformer training, the standard recipe:

```
warmup_steps = 2000             # or 1% of total steps
total_steps = 100000

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    # cosine decay to 10% of peak LR
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
```

- **Linear warmup** prevents early instability (gradients are huge with fresh weights).
- **Cosine decay to 10% of peak** is the most-used post-warmup schedule.

Typical peak LR for transformer pretraining:
- Small models (~100M params): 6e-4
- Medium (~1B): 3e-4
- Large (~7B+): 1.5e-4
- Frontier (>70B): 8e-5

These are not magic — they're empirically what works. Smaller models tolerate larger LR. Note these are 2-3× larger than what "Adam's Karpathy constant" (3e-4) suggests for arbitrary models — transformers are forgiving.

### Chinchilla scaling laws

[Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) showed: for a fixed compute budget `C`, the optimal allocation between model size `N` and training tokens `D` is roughly:

```
N* ∝ √C,  D* ∝ √C
```

Concretely: **a 7B-param model should be trained on ~140B tokens** (Chinchilla ratio: 20 tokens per parameter). Larger models with proportionally less data ("undertrained") or smaller models with more data ("overtrained") are sub-optimal under this framework.

Llama-3 famously **violates** Chinchilla — Llama-3-8B was trained on 15T tokens (1875 tokens/param). For *inference*-time cost, undertraining a tiny model on a lot of data is a great deal. Chinchilla optimality is about *training* compute, not lifetime cost.

---

## Part 10: A peek at the full training stack

Here's what a real LLM pretraining script orchestrates:

```
┌───────────────────────────────────────────────────────────────────┐
│ torchrun --nnodes=128 --nproc_per_node=8 ...                      │
│                                                                   │
│   ┌──────────────────────────────────────────────────────────┐    │
│   │  Per-rank training loop                                   │    │
│   │                                                           │    │
│   │  1. Load tokenized shard (HDF5/Arrow/parquet)             │    │
│   │  2. Build batch, transfer to GPU                          │    │
│   │  3. autocast(bf16): forward                                │    │
│   │  4. backward (FSDP all-gathers + reduces)                 │    │
│   │  5. Gradient accumulation (every N micro-batches)         │    │
│   │  6. Grad clip + optimizer step                            │    │
│   │  7. Scheduler step                                        │    │
│   │  8. Log to W&B from rank 0; checkpoint every N hours      │    │
│   └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│   Communication: NCCL over NVLink (intra-node), InfiniBand (across) │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

You'll see this exact shape in the [nanoGPT](https://github.com/karpathy/nanoGPT) repo, the [llm.c](https://github.com/karpathy/llm.c) repo, and any production LLM codebase. The pieces are interchangeable; the structure is universal.

---

## What's next

In [lab.md](lab.md) you'll:

- Calculate memory consumption for your week-10 GPT analytically, then verify with `torch.cuda.memory_allocated`
- Train your GPT with mixed precision (bf16) and measure the speedup
- Add gradient accumulation; show the equivalence to a larger batch size
- Apply gradient checkpointing and measure the memory savings
- (If multi-GPU available) launch a DDP training run
- Read a published Chinchilla-style scaling-law graph and interpret it

By week 11's end you'll understand how Llama-3 was trained. Week 12 fine-tunes a pretrained one; weeks 13-15 dig into the hardware that makes any of this possible.
