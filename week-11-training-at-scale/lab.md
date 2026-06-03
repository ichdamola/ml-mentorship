# Week 11: Lab — Training at Scale

You'll measure your week-10 GPT's memory profile, then add the scaling techniques one at a time — mixed precision, gradient accumulation, gradient checkpointing — and watch each one earn its keep.

## Setup

Same env as week 10, plus optionally `accelerate` for the DDP exercise:

```bash
uv add accelerate
```

```python
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}, name: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'}")
print(f"bf16 supported: {torch.cuda.is_bf16_supported() if device.type == 'cuda' else False}")
```

Import or recreate your week-10 GPT here (`GPT`, `TransformerBlock`, `MultiHeadAttention`).

---

## Exercise 11.1 — Memory math by hand

For a GPT with these dimensions:

- `vocab_size = 50,257` (GPT-2 vocab)
- `d_model = 768` (GPT-2 small)
- `n_layers = 12`
- `n_heads = 12`
- `d_ff = 4 × 768 = 3072`
- `max_seq_len = 1024`

Calculate **parameter count by hand**.

```python
# Embedding (tied with lm_head — count once)
embed_params = 50257 * 768                              # ≈ 38.6M
pos_emb_params = 1024 * 768                             # ≈ 0.8M

# Per layer:
#   QKV projection: 3 × d_model²            = 3 × 768²    = 1.77M
#   Out projection: d_model²                = 768²        = 0.59M
#   FFN: 2 × d_model × d_ff                 = 2×768×3072 = 4.72M
#   LayerNorms: 4 × d_model (2 LNs, γ + β) = 4 × 768    = 0.003M
per_layer_params = 3 * 768**2 + 768**2 + 2 * 768 * 3072 + 4 * 768

total_params = embed_params + pos_emb_params + 12 * per_layer_params
print(f"calculated: {total_params / 1e6:.2f}M")     # ~124M (GPT-2 small)
```

This matches the actual GPT-2 small param count (124M). The memory in fp32:

```python
def calculate_memory_gb(n_params, dtype_bytes=4, optim_state_multiplier=2, activation_per_token_gb=None):
    """Return (params, grads, optim, total) in GB"""
    params_gb = n_params * dtype_bytes / 1e9
    grads_gb = n_params * dtype_bytes / 1e9
    optim_gb = n_params * dtype_bytes * optim_state_multiplier / 1e9   # AdamW: 2x for m, v
    return params_gb, grads_gb, optim_gb, params_gb + grads_gb + optim_gb

for name, n_params, dtype_bytes in [
    ('GPT-2 small (124M) fp32', 124e6, 4),
    ('GPT-2 small (124M) bf16', 124e6, 2),
    ('Llama-2 7B fp32', 7e9, 4),
    ('Llama-2 7B bf16 (master fp32)', 7e9, 4),  # weights in bf16 but Adam state in fp32
    ('Llama-2 70B bf16', 70e9, 2),
]:
    p, g, o, t = calculate_memory_gb(n_params, dtype_bytes)
    print(f"{name:>32s}  params {p:>7.2f}  grads {g:>7.2f}  optim {o:>7.2f}  total {t:>7.2f} GB")
```

You'll see why **Llama-2 7B in fp32 is 112 GB** — needs FSDP across multiple GPUs even though the model itself is only 28 GB of bf16 weights.

---

## Exercise 11.2 — Verify with `torch.cuda.memory_allocated`

If you have a CUDA GPU, measure your actual model's memory.

```python
if device.type == 'cuda':
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    # Build your week-10 model
    model = GPT(vocab_size=65, d_model=192, n_layers=6, n_heads=6,
                d_ff=768, max_seq_len=128).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"#params: {n_params/1e6:.2f}M")
    print(f"after model build:   {torch.cuda.memory_allocated() / 1e6:.2f} MB")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # Adam's state isn't allocated until first step
    print(f"after opt build:     {torch.cuda.memory_allocated() / 1e6:.2f} MB")

    # Run one step to allocate Adam state
    X = torch.randint(0, 65, (32, 128), device=device)
    y = torch.randint(0, 65, (32, 128), device=device)
    mask = torch.tril(torch.ones(128, 128, device=device))
    loss = F.cross_entropy(model(X, mask=mask).view(-1, 65), y.view(-1))
    loss.backward()
    opt.step()
    print(f"after one step:      {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    print(f"peak during step:    {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")
```

You'll see the numbers line up with the theory: each param contributes ~12-16 bytes (params + grads + Adam state) before activations.

---

## Exercise 11.3 — Mixed precision training

Add `autocast`. Measure speedup.

```python
def benchmark_train_step(model, X, y, mask, opt, dtype, n_iters=50):
    """Run n_iters training steps; return mean step time in ms."""
    # Warmup
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=(dtype != torch.float32)):
            loss = F.cross_entropy(model(X, mask=mask).view(-1, 65), y.view(-1))
        loss.backward()
        opt.step()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=(dtype != torch.float32)):
            loss = F.cross_entropy(model(X, mask=mask).view(-1, 65), y.view(-1))
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1000

if device.type == 'cuda':
    # Use a bigger model so the speedup is visible
    model = GPT(vocab_size=65, d_model=384, n_layers=8, n_heads=6,
                d_ff=1536, max_seq_len=256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    X = torch.randint(0, 65, (32, 256), device=device)
    y = torch.randint(0, 65, (32, 256), device=device)
    mask = torch.tril(torch.ones(256, 256, device=device))

    for dtype in [torch.float32, torch.bfloat16]:
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            continue
        t = benchmark_train_step(model, X, y, mask, opt, dtype)
        print(f"{dtype}: {t:.2f} ms/step")
```

**Expected:** bf16 is ~2× faster than fp32 on Ampere+ GPUs, ~3-4× faster on Hopper. **This is a free win in real training.** Combined with the memory halving on activations, you can roughly double your batch size while training faster.

---

## Exercise 11.4 — Gradient accumulation

Show that micro-batch 4 × 8 accumulation == one big-batch 32.

```python
torch.manual_seed(42)
model_a = GPT(vocab_size=65, d_model=192, n_layers=4, n_heads=6,
              d_ff=768, max_seq_len=64).to(device)
opt_a = torch.optim.SGD(model_a.parameters(), lr=1e-2)   # SGD so this is deterministic

# Make a deterministic batch of 32
X = torch.randint(0, 65, (32, 64), device=device)
y = torch.randint(0, 65, (32, 64), device=device)
mask = torch.tril(torch.ones(64, 64, device=device))

# (a) one big batch of 32
opt_a.zero_grad(set_to_none=True)
loss = F.cross_entropy(model_a(X, mask=mask).view(-1, 65), y.view(-1))
loss.backward()
big_grad = model_a.lm_head.weight.grad.clone()
opt_a.zero_grad(set_to_none=True)

# (b) 8 micro-batches of 4 with accumulation
torch.manual_seed(42)
model_b = GPT(vocab_size=65, d_model=192, n_layers=4, n_heads=6,
              d_ff=768, max_seq_len=64).to(device)
opt_b = torch.optim.SGD(model_b.parameters(), lr=1e-2)

opt_b.zero_grad(set_to_none=True)
for i in range(8):
    X_mb = X[i*4:(i+1)*4]
    y_mb = y[i*4:(i+1)*4]
    loss = F.cross_entropy(model_b(X_mb, mask=mask).view(-1, 65), y_mb.view(-1)) / 8
    loss.backward()
accum_grad = model_b.lm_head.weight.grad.clone()

# These should match closely
diff = (big_grad - accum_grad).abs().max().item()
print(f"max abs diff between big-batch and accum gradients: {diff:.2e}")
print("✓ they're equivalent" if diff < 1e-4 else "❌ off — check your /accum_steps scaling")
```

The two gradient tensors should match to ~1e-6. **That's why you divide by `accum_steps`.** Without it, the accumulated gradient is 8× too large.

---

## Exercise 11.5 — Gradient checkpointing

Save activation memory by recomputing during backward.

```python
import torch.utils.checkpoint as ckpt_util

class GPTCheckpointed(GPT):
    """Override forward to checkpoint each block."""
    def forward(self, idx, mask=None, kv_cache=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for i, block in enumerate(self.blocks):
            # Note: checkpointing breaks the kv_cache path; only use during training
            x = ckpt_util.checkpoint(
                block, x, mask, None, i,
                use_reentrant=False,
            )
        x = self.ln_f(x)
        return self.lm_head(x)

if device.type == 'cuda':
    # Compare memory: same model with and without checkpointing
    for cls in [GPT, GPTCheckpointed]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        m = cls(vocab_size=65, d_model=384, n_layers=8, n_heads=6,
                d_ff=1536, max_seq_len=256).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
        X = torch.randint(0, 65, (16, 256), device=device)
        y = torch.randint(0, 65, (16, 256), device=device)
        mask = torch.tril(torch.ones(256, 256, device=device))

        loss = F.cross_entropy(m(X, mask=mask).view(-1, 65), y.view(-1))
        loss.backward()
        opt.step()

        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"{cls.__name__:25s}  peak {peak_mb:>7.1f} MB")
```

Checkpointed version should use **30-60% less peak memory** at the cost of ~30% extra compute time. When activations dominate, this is exactly how you fit a bigger model.

---

## Exercise 11.6 — DDP (if you have 2+ GPUs)

This requires multiple GPUs. If you only have one, skip to the next exercise.

Save the following to `ddp_train.py`:

```python
# ddp_train.py
import os, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)

    model = nn.Linear(1000, 1000).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(20):
        X = torch.randn(64, 1000).cuda(local_rank)
        y = torch.randn(64, 1000).cuda(local_rank)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(X), y)
        loss.backward()
        opt.step()
        if rank == 0:
            print(f"step {step} loss {loss.item():.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

Run with:

```bash
torchrun --standalone --nproc_per_node=2 ddp_train.py
```

You'll see two processes producing one combined log. Each holds the same model; gradients are all-reduced under the hood. **Scale this up by changing `nproc_per_node` and you get linear speedup.**

---

## Exercise 11.7 — Chinchilla scaling plot

Read the Chinchilla paper (or its summary) and plot the optimal model size vs training data:

```python
# Chinchilla optimal: ~20 tokens per parameter
N = np.logspace(7, 12, 100)        # 10M to 1T params
D_chinchilla = 20 * N

plt.figure(figsize=(10, 5))
plt.loglog(N, D_chinchilla, label='Chinchilla optimal (20 tok/param)')
# Annotate some real models
markers = {
    'GPT-2 (1.5B, 40B tok)': (1.5e9, 40e9),
    'GPT-3 (175B, 300B tok)': (175e9, 300e9),
    'Chinchilla (70B, 1.4T tok)': (70e9, 1.4e12),
    'Llama-2-7B (7B, 2T tok)': (7e9, 2e12),
    'Llama-3-8B (8B, 15T tok)': (8e9, 15e12),
    'Llama-3-70B (70B, 15T tok)': (70e9, 15e12),
}
for name, (n, d) in markers.items():
    plt.scatter(n, d, s=80, zorder=5)
    plt.annotate(name, (n, d), fontsize=9, xytext=(5, 5), textcoords='offset points')
plt.xlabel("Model size (params)"); plt.ylabel("Training tokens")
plt.legend()
plt.title("Real models vs Chinchilla-optimal compute allocation")
plt.grid(alpha=0.3); plt.show()
```

What you should see:
- **GPT-3 was massively undertrained** — way above the Chinchilla line per param. GPT-4-class models corrected this.
- **Llama-3-8B is hugely overtrained** — way below the line. Meta deliberately chose this for *inference* economics: a smaller model trained longer is cheaper to serve.
- **Chinchilla is about *training* compute optimality, not inference cost.**

This shapes a real production decision: if you'll serve 100B tokens of inference, undertraining a small model is great; if you only need 1M, train a big model on less.

---

## Exercise 11.8 — Build the full disciplined training script

Combine the techniques: AMP + grad accumulation + grad clipping + cosine-with-warmup + best-val ckpt + reproducibility. Same data as week 10 (TinyShakespeare).

```python
def train_disciplined_at_scale(
    model, train_data, val_data, n_steps=3000,
    micro_batch=16, accum_steps=4,
    block_size=256, lr=3e-4, weight_decay=0.01,
    warmup_steps=200, grad_clip=1.0,
):
    """A trainable subset of what a real LLM training script does."""
    use_bf16 = (device.type == 'cuda' and torch.cuda.is_bf16_supported())
    autocast_kwargs = {'device_type': device.type, 'dtype': torch.bfloat16, 'enabled': use_bf16}

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    mask = torch.tril(torch.ones(block_size, block_size, device=device))

    def get_batch(split):
        src = train_data if split == 'train' else val_data
        ix = torch.randint(0, len(src) - block_size - 1, (micro_batch,))
        X = torch.stack([src[i:i+block_size] for i in ix])
        y = torch.stack([src[i+1:i+block_size+1] for i in ix])
        return X.to(device), y.to(device)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val = float('inf')
    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for accum in range(accum_steps):
            X, y = get_batch('train')
            with torch.amp.autocast(**autocast_kwargs):
                logits = model(X, mask=mask)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) / accum_steps
            loss.backward()
            accumulated_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        scheduler.step()

        if step % 200 == 0 or step == n_steps - 1:
            model.eval()
            with torch.no_grad():
                v_losses = []
                for _ in range(10):
                    Xv, yv = get_batch('val')
                    with torch.amp.autocast(**autocast_kwargs):
                        vl = F.cross_entropy(
                            model(Xv, mask=mask).view(-1, logits.size(-1)),
                            yv.view(-1)
                        ).item()
                    v_losses.append(vl)
            val_loss = sum(v_losses) / len(v_losses)
            print(f"step {step:4d}  train {accumulated_loss:.4f}  val {val_loss:.4f}  lr {opt.param_groups[0]['lr']:.5f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), "best.pt")
            model.train()
    return best_val
```

Use this for any real training run from here forward. **This is the shape of a senior training script.**

---

## Submission checklist

- [ ] GPT-2 small param count calculated by hand ≈ 124M
- [ ] Llama-2-7B fp32 memory calculated ≈ 112 GB
- [ ] `torch.cuda.memory_allocated` matches your theoretical calculation
- [ ] bf16 AMP gives ≥ 1.5× speedup over fp32
- [ ] Gradient accumulation reproduces big-batch gradients to ≤ 1e-4
- [ ] Gradient checkpointing reduces peak memory by ≥ 30% on a comparable model
- [ ] (If multi-GPU) DDP runs and logs from rank 0 only
- [ ] Chinchilla scaling plot includes ≥ 4 real models
- [ ] Disciplined training script uses AMP + accum + clip + cosine + best-val ckpt

---

## What you just did

You now understand every line of a production LLM training script. From here on, when you read "trained on 128 H100s for 10 days with bf16 + FSDP + gradient checkpointing", you know exactly what's happening at each layer of the stack.

Week 12 uses these foundations to fine-tune a pretrained model — most production "LLM work" today is adaptation via LoRA, not pretraining from scratch.

---

**Next**: [Week 12: LLM Fine-tuning →](../week-12-llm-finetuning/readme.md)
