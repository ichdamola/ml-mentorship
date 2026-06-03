# Week 15: Lab — Profile and Optimize a Real Model

You'll profile a forward + backward pass of your week-10 transformer, find the three slowest ops, apply `torch.compile`, switch to `F.scaled_dot_product_attention` (FlashAttention under the hood), add gradient checkpointing, and measure each optimization's contribution.

## Setup

```bash
# Already installed in earlier weeks: torch + cuda
# Nsight Systems CLI usually ships with the CUDA toolkit
nsys --version
ncu --version    # optional — gates Exercise 15.10
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from torch.profiler import profile, record_function, ProfilerActivity
import torch.cuda.nvtx as nvtx

assert torch.cuda.is_available()
device = torch.device("cuda")
torch.manual_seed(42)
```

Recreate (or import) your week-10 GPT model. For this lab we'll use a slightly larger one so optimization wins are visible:

```python
import math

class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None, use_sdpa=False):
        B, N, _ = x.shape
        qkv = self.W_qkv(x).reshape(B, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        if use_sdpa:
            out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        else:
            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            weights = F.softmax(scores, dim=-1)
            out = weights @ V

        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        return self.W_o(out)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, use_sdpa=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.use_sdpa = use_sdpa

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask=mask, use_sdpa=self.use_sdpa)
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size=256, d_model=512, n_layers=8, n_heads=8,
                 d_ff=2048, max_seq_len=512, use_sdpa=False):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, use_sdpa=use_sdpa) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, mask=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.lm_head(x)

# Build it
model = GPT().to(device).to(torch.bfloat16)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

vocab = 256
B, T = 8, 512
X = torch.randint(0, vocab, (B, T), device=device)
y = torch.randint(0, vocab, (B, T), device=device)
mask = torch.tril(torch.ones(T, T, device=device))

# Sanity check
out = model(X, mask)
loss = F.cross_entropy(out.reshape(-1, vocab), y.reshape(-1))
print(f"baseline forward OK; loss = {loss.item():.4f}, output shape = {out.shape}")
```

---

## Exercise 15.1 — Plain benchmark

Establish a baseline.

```python
def bench_train_step(model, X, y, mask, opt, n_iter=50):
    # Warmup
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        out = model(X, mask)
        loss = F.cross_entropy(out.reshape(-1, out.size(-1)), y.reshape(-1))
        loss.backward()
        opt.step()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        opt.zero_grad(set_to_none=True)
        out = model(X, mask)
        loss = F.cross_entropy(out.reshape(-1, out.size(-1)), y.reshape(-1))
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000

baseline_ms = bench_train_step(model, X, y, mask, opt)
print(f"baseline: {baseline_ms:.2f} ms/step")
```

Note this number. Every optimization below is measured against it.

---

## Exercise 15.2 — PyTorch profiler

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
    profile_memory=True,
) as prof:
    for step in range(10):
        with record_function("train_step"):
            opt.zero_grad(set_to_none=True)
            with record_function("forward"):
                out = model(X, mask)
                loss = F.cross_entropy(out.reshape(-1, out.size(-1)), y.reshape(-1))
            with record_function("backward"):
                loss.backward()
            with record_function("step"):
                opt.step()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
```

You'll see something like:

```
Name                               Self CUDA   CUDA total   # of Calls
aten::addmm                            8.5ms        8.5ms          80
aten::softmax                          3.2ms        3.2ms          10
aten::native_layer_norm                2.1ms        2.1ms          20
aten::index                            1.8ms        1.8ms          10
...
```

**The top 3 are your bottleneck.** Usually:
- `addmm` (matmul) — dominates by total but is hard to optimize further
- `softmax` — memory-bound; fusion-eligible
- `LayerNorm` — memory-bound; fusion-eligible

The latter two are exactly what FlashAttention and `torch.compile` fuse away.

Save the Chrome trace:

```python
prof.export_chrome_trace("trace.json")
```

Open `chrome://tracing` or [perfetto.dev](https://ui.perfetto.dev/) → load `trace.json`. You'll see a flame-chart timeline. **Look for gaps in the CUDA stream — those are GPU idle periods.**

---

## Exercise 15.3 — Add NVTX annotations

Make the timeline readable by labeling blocks.

```python
def step_with_nvtx(model, X, y, mask, opt):
    with torch.cuda.nvtx.range("zero_grad"):
        opt.zero_grad(set_to_none=True)
    with torch.cuda.nvtx.range("forward"):
        out = model(X, mask)
    with torch.cuda.nvtx.range("loss"):
        loss = F.cross_entropy(out.reshape(-1, out.size(-1)), y.reshape(-1))
    with torch.cuda.nvtx.range("backward"):
        loss.backward()
    with torch.cuda.nvtx.range("step"):
        opt.step()

# Warmup
for _ in range(5): step_with_nvtx(model, X, y, mask, opt)

# Profile a short window
torch.cuda.cudart().cudaProfilerStart()
for _ in range(3): step_with_nvtx(model, X, y, mask, opt)
torch.cuda.cudart().cudaProfilerStop()
```

If you run this script under `nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop python script.py`, your NVTX ranges appear as labeled bars in the timeline. **Critical for finding what part of your loop is slow.**

---

## Exercise 15.4 — torch.compile

```python
torch.cuda.empty_cache()
torch.manual_seed(42)
model_compiled = GPT(use_sdpa=False).to(device).to(torch.bfloat16)
opt_compiled = torch.optim.AdamW(model_compiled.parameters(), lr=3e-4)

# Warmup includes first-time JIT compile (slow)
model_compiled = torch.compile(model_compiled, mode="reduce-overhead")

# First call triggers compile
print("first call triggers JIT compile, expect 10-30s pause...")
t0 = time.perf_counter()
out = model_compiled(X, mask)
loss = F.cross_entropy(out.reshape(-1, vocab), y.reshape(-1))
loss.backward()
print(f"first call: {time.perf_counter() - t0:.1f}s (includes compile)")

# Now benchmark
compiled_ms = bench_train_step(model_compiled, X, y, mask, opt_compiled)
print(f"\nbaseline:        {baseline_ms:.2f} ms/step")
print(f"torch.compile:   {compiled_ms:.2f} ms/step  ({baseline_ms/compiled_ms:.2f}× faster)")
```

You should see **1.3-2× speedup** for free. On bigger models the gains grow; on tiny ones the JIT overhead dominates.

If you get errors, `torch.compile` is falling back to eager for some op. Try `mode="default"` (more conservative).

---

## Exercise 15.5 — FlashAttention via `scaled_dot_product_attention`

PyTorch 2.0+ ships `F.scaled_dot_product_attention` which dispatches to FlashAttention on Ampere+, math-equivalent attention on older hardware. **Almost always faster than hand-rolled.**

```python
torch.cuda.empty_cache()
torch.manual_seed(42)
model_sdpa = GPT(use_sdpa=True).to(device).to(torch.bfloat16)
opt_sdpa = torch.optim.AdamW(model_sdpa.parameters(), lr=3e-4)

sdpa_ms = bench_train_step(model_sdpa, X, y, mask, opt_sdpa)
print(f"baseline:        {baseline_ms:.2f} ms/step")
print(f"SDPA only:       {sdpa_ms:.2f} ms/step  ({baseline_ms/sdpa_ms:.2f}× faster)")

# Combined with torch.compile
model_sdpa_compiled = torch.compile(model_sdpa, mode="reduce-overhead")
# Warmup compile
out = model_sdpa_compiled(X, mask)
loss = F.cross_entropy(out.reshape(-1, vocab), y.reshape(-1))
loss.backward()
opt_sdpa.step()

both_ms = bench_train_step(model_sdpa_compiled, X, y, mask, opt_sdpa)
print(f"SDPA + compile:  {both_ms:.2f} ms/step  ({baseline_ms/both_ms:.2f}× faster than baseline)")
```

SDPA alone gives 1.5-3× on long seqs. Stacked with `torch.compile` you typically hit 2-4× total.

---

## Exercise 15.6 — Memory profile

Where is memory going?

```python
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

model = GPT(use_sdpa=False).to(device).to(torch.bfloat16)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

print(f"after model build:    {torch.cuda.memory_allocated() / 1e6:.1f} MB")
print(f"  params alone:       {sum(p.numel()*p.element_size() for p in model.parameters()) / 1e6:.1f} MB")

out = model(X, mask)
loss = F.cross_entropy(out.reshape(-1, vocab), y.reshape(-1))
print(f"after forward:        {torch.cuda.memory_allocated() / 1e6:.1f} MB")

loss.backward()
print(f"after backward:       {torch.cuda.memory_allocated() / 1e6:.1f} MB")
print(f"peak so far:          {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")

opt.step()
print(f"after first opt step: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
print(f"peak overall:         {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
```

Activations show up as the delta from "params" to "after forward." For this model they're typically 5-15× larger than params. **That's why long-context training OOMs without checkpointing.**

---

## Exercise 15.7 — Gradient checkpointing

```python
import torch.utils.checkpoint as ckpt_util

class GPTCheckpointed(GPT):
    def forward(self, idx, mask=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = ckpt_util.checkpoint(block, x, mask, use_reentrant=False)
        x = self.ln_f(x)
        return self.lm_head(x)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
model_ckpt = GPTCheckpointed(use_sdpa=True).to(device).to(torch.bfloat16)
opt_ckpt = torch.optim.AdamW(model_ckpt.parameters(), lr=3e-4)

# Single forward+backward+step
out = model_ckpt(X, mask)
loss = F.cross_entropy(out.reshape(-1, vocab), y.reshape(-1))
loss.backward()
opt_ckpt.step()
ckpt_peak_mb = torch.cuda.max_memory_allocated() / 1e6

# Compare to non-checkpointed
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
model_no = GPT(use_sdpa=True).to(device).to(torch.bfloat16)
opt_no = torch.optim.AdamW(model_no.parameters(), lr=3e-4)
out = model_no(X, mask)
loss = F.cross_entropy(out.reshape(-1, vocab), y.reshape(-1))
loss.backward()
opt_no.step()
no_peak_mb = torch.cuda.max_memory_allocated() / 1e6

# Speed too
ckpt_ms = bench_train_step(model_ckpt, X, y, mask, opt_ckpt, n_iter=20)
no_ms = bench_train_step(model_no, X, y, mask, opt_no, n_iter=20)

print(f"no checkpoint:  peak {no_peak_mb:>7.1f} MB  step {no_ms:.2f} ms")
print(f"checkpointed:   peak {ckpt_peak_mb:>7.1f} MB  step {ckpt_ms:.2f} ms")
print(f"memory savings: {100*(1 - ckpt_peak_mb/no_peak_mb):.1f}%")
print(f"time penalty:    {100*(ckpt_ms/no_ms - 1):.1f}%")
```

Typical result: ~40-60% memory savings, ~30% time penalty. **You'd take this trade any time you need to fit a bigger model or longer context.**

---

## Exercise 15.8 — Summary table

```python
results = [
    ("baseline",              baseline_ms, None),
    ("+ SDPA",                sdpa_ms,     None),
    ("+ torch.compile",       compiled_ms, None),
    ("+ both",                both_ms,     None),
]
print(f"{'config':>25s}  {'ms/step':>8s}  {'speedup':>9s}")
for name, ms, _ in results:
    print(f"{name:>25s}  {ms:>8.2f}  {baseline_ms/ms:>8.2f}x")
```

Real production wins for an 8-layer 512-dim transformer at batch 8, seq 512 typically look like:

| Config | ms/step | speedup |
|---|---|---|
| baseline (bf16) | 30-40 | 1.0× |
| + SDPA | 18-25 | 1.5× |
| + torch.compile | 22-30 | 1.4× |
| + both | 12-18 | 2.0-2.5× |

**Two-shot of "free" 2-3× from APIs that didn't exist 4 years ago.** This is why senior ML engineers always reach for them first.

---

## Exercise 15.9 — Nsight Systems CLI

Run this as a separate Python script `nsys_run.py`:

```python
# nsys_run.py
import torch, torch.nn as nn, torch.nn.functional as F
import torch.cuda.nvtx as nvtx
# ... (import your GPT class) ...

device = torch.device("cuda")
model = GPT(use_sdpa=True).to(device).to(torch.bfloat16)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
X = torch.randint(0, 256, (8, 512), device=device)
y = torch.randint(0, 256, (8, 512), device=device)
mask = torch.tril(torch.ones(512, 512, device=device))

# Warmup outside profile
for _ in range(5):
    opt.zero_grad(set_to_none=True)
    out = model(X, mask)
    F.cross_entropy(out.reshape(-1, 256), y.reshape(-1)).backward()
    opt.step()

torch.cuda.cudart().cudaProfilerStart()
for step in range(3):
    with torch.cuda.nvtx.range(f"step_{step}"):
        opt.zero_grad(set_to_none=True)
        with torch.cuda.nvtx.range("forward"):
            out = model(X, mask)
        loss = F.cross_entropy(out.reshape(-1, 256), y.reshape(-1))
        with torch.cuda.nvtx.range("backward"):
            loss.backward()
        with torch.cuda.nvtx.range("opt_step"):
            opt.step()
torch.cuda.cudart().cudaProfilerStop()
```

Run:

```bash
nsys profile \
  --output=gpt_profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  python nsys_run.py
```

Open `gpt_profile.nsys-rep` in **Nsight Systems GUI** (download from NVIDIA). The timeline shows:

- Three `step_N` bars
- Inside each, `forward`, `backward`, `opt_step` ranges
- Below: the actual CUDA kernels (matmul, softmax, GELU, etc.)

Look for:
- **Gaps in the CUDA row** = GPU idle. Why?
- **`opt_step` taking longer than expected** = Adam's per-param updates are launch-overhead-heavy
- **Kernel-launch bubbles** = too many small kernels; fuse them

For most users this is the most valuable profiling artifact you'll ever produce.

---

## Exercise 15.10 (stretch) — Nsight Compute on one kernel

```bash
ncu --kernel-name "regex:.*softmax.*" \
    --launch-skip 5 \
    --launch-count 1 \
    --set full \
    -o softmax_profile \
    python nsys_run.py
```

Open `softmax_profile.ncu-rep`. Look at:

- **Speed of Light** section: compute throughput % vs memory throughput %
- **Memory Workload Analysis**: HBM bandwidth utilization
- **Roofline Analysis**: where the kernel sits on the roofline

For a softmax in PyTorch's standard implementation, you should see:

```
Compute  : 8-15%       ← low
Memory   : 70-90%      ← high
```

**Memory-bound.** This is why fusing softmax into FlashAttention is a 2-4× win — same compute, half the memory traffic.

---

## Submission checklist

- [ ] Baseline `ms/step` measured
- [ ] PyTorch profiler shows top-5 ops by Self CUDA time
- [ ] Chrome trace exported and inspected
- [ ] NVTX annotations applied; ranges visible in profile
- [ ] `torch.compile` gives ≥ 1.3× speedup
- [ ] `F.scaled_dot_product_attention` gives ≥ 1.4× speedup
- [ ] Combined SDPA + `torch.compile` gives ≥ 2× speedup
- [ ] Memory profile shows activations dominate; gradient checkpointing saves ≥ 30%
- [ ] Summary table comparing baseline → SDPA → compile → both
- [ ] (Stretch) Nsight Systems trace produced and read
- [ ] (Stretch) Nsight Compute analysis of one kernel — classified as memory-bound or compute-bound

---

## What you just did

You profiled a real transformer, identified the dominant kernels, applied `torch.compile` and SDPA for ~2× speedup with no manual kernel work, added gradient checkpointing for memory savings, and used Nsight tools to see what the hardware is actually doing.

**This is the senior playbook.** When someone says "training is slow," you now have a 10-minute diagnosis flow: profile → identify → apply `torch.compile` + SDPA → measure → decide if more work is justified.

Week 16 is the capstone — take your trained model, quantize it, serve it with vLLM, measure throughput, and deploy. End-to-end.

---

**Next**: [Week 16: Production Inference →](../week-16-production-inference/readme.md)
