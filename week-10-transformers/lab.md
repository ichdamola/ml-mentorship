# Week 10: Lab — Build a Tiny GPT

You'll build a transformer from `nn.Linear` primitives, train it on TinyShakespeare to ~~ generate plausible Shakespeare, then implement a KV cache and watch decoding speed jump 10×.

## Setup

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os, time

torch.manual_seed(42); np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}")
```

Get TinyShakespeare (~1 MB):

```python
import urllib.request
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("shakespeare.txt"):
    urllib.request.urlretrieve(url, "shakespeare.txt")

with open("shakespeare.txt") as f:
    text = f.read()
print(f"text length: {len(text):,} chars")
print(text[:500])
```

---

## Exercise 10.1 — Character-level tokenization

For a 1 MB corpus, characters are fine — no need for BPE or sentencepiece yet (week 12).

```python
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

def encode(s: str) -> list[int]: return [stoi[c] for c in s]
def decode(ids: list[int]) -> str: return "".join(itos[i] for i in ids)

print(f"vocab size: {vocab_size}")
print(f"encode('hello') = {encode('hello')}")
print(f"decode back = {decode(encode('hello'))}")

# Encode the whole corpus
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"train: {len(train_data):,}  val: {len(val_data):,}")
```

---

## Exercise 10.2 — Scaled dot-product attention by hand

Implement it from the formula in [theory.md Part 2](theory.md).

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (B, n_heads, N, d_head) or (B, N, d_head)
    mask: (N, N) lower-triangular for causal, or None
    """
    d_k = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights

# Test against PyTorch's built-in
torch.manual_seed(0)
B, N, d_k = 2, 5, 8
Q = torch.randn(B, N, d_k)
K = torch.randn(B, N, d_k)
V = torch.randn(B, N, d_k)
mask = torch.tril(torch.ones(N, N))

mine, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
torch_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask.bool(), is_causal=False)
print(f"max abs diff vs torch built-in: {(mine - torch_out).abs().max():.2e}")

# Visualize attention weights (causal pattern should be lower triangular)
plt.figure(figsize=(4, 4))
plt.imshow(weights[0].detach().numpy(), cmap='Blues', vmin=0)
plt.title("Causal attention weights — lower triangular")
plt.xlabel("attend to position"); plt.ylabel("query position")
plt.colorbar(); plt.show()
```

You should see a lower-triangular pattern — each row only weights the diagonal and to the left.

---

## Exercise 10.3 — Multi-head attention with fused QKV

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o   = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, kv_cache=None, layer_idx=0):
        B, N, _ = x.shape

        qkv = self.W_qkv(x)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)                  # (3, B, n_heads, N, d_head)
        Q, K, V = qkv[0], qkv[1], qkv[2]                  # each (B, n_heads, N, d_head)

        # KV cache integration (used in 10.7)
        if kv_cache is not None:
            K, V = kv_cache.update(layer_idx, K, V)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = weights @ V                                  # (B, n_heads, Tq, d_head)
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        return self.W_o(out)

# Sanity check
mha = MultiHeadAttention(d_model=64, n_heads=4)
x = torch.randn(2, 10, 64)
mask = torch.tril(torch.ones(10, 10))
out = mha(x, mask=mask)
print(f"out shape: {out.shape}")     # (2, 10, 64)
```

---

## Exercise 10.4 — Sinusoidal position encoding

```python
def sinusoidal_pe(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(seq_len).float().unsqueeze(1)        # (N, 1)
    i = torch.arange(0, d_model, 2).float()                  # (d_model/2,)
    div_term = 10000 ** (i / d_model)
    pe[:, 0::2] = torch.sin(pos / div_term)
    pe[:, 1::2] = torch.cos(pos / div_term)
    return pe

# Visualize
pe = sinusoidal_pe(seq_len=200, d_model=128)
plt.figure(figsize=(8, 5))
plt.imshow(pe.T, aspect='auto', cmap='coolwarm')
plt.xlabel("position"); plt.ylabel("embedding dim")
plt.title("Sinusoidal position encoding (low dim = high freq)")
plt.colorbar(); plt.show()
```

For the lab we'll use learned position embeddings (`nn.Embedding`) because they're a single line of code and equally effective at this scale. Sinusoidal matters more when you want to extrapolate past training length.

---

## Exercise 10.5 — Transformer block + full GPT

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None, kv_cache=None, layer_idx=0):
        x = x + self.attn(self.ln1(x), mask, kv_cache=kv_cache, layer_idx=layer_idx)
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=192, n_layers=6, n_heads=6,
                 d_ff=4*192, max_seq_len=256, dropout=0.2):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout=dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight   # weight tying

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, mask=None, kv_cache=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for i, block in enumerate(self.blocks):
            x = block(x, mask=mask, kv_cache=kv_cache, layer_idx=i)
        x = self.ln_f(x)
        return self.lm_head(x)

# Build model
model = GPT(vocab_size).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"#params: {n_params/1e6:.2f}M")
```

~1.5M parameters — micro by modern standards, but enough to learn Shakespeare style.

---

## Exercise 10.6 — Train it

Standard week-08 discipline.

```python
block_size = 128                                # context length
batch_size = 64
def get_batch(split):
    src = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(src) - block_size - 1, (batch_size,))
    X = torch.stack([src[i:i+block_size] for i in ix])
    y = torch.stack([src[i+1:i+block_size+1] for i in ix])
    return X.to(device), y.to(device)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
mask = torch.tril(torch.ones(block_size, block_size, device=device))

n_steps = 3000
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        L = torch.zeros(20)
        for k in range(20):
            X, y = get_batch(split)
            logits = model(X, mask=mask)
            L[k] = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
        losses[split] = L.mean().item()
    model.train()
    return losses

train_losses = []
for step in range(n_steps):
    X, y = get_batch('train')
    logits = model(X, mask=mask)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    train_losses.append(loss.item())

    if step % 300 == 0 or step == n_steps - 1:
        losses = estimate_loss()
        print(f"step {step:4d}  train {losses['train']:.4f}  val {losses['val']:.4f}")
```

For a ~1.5M param model on TinyShakespeare you should reach **val loss ≈ 1.5** in 3000 steps (~5-10 minutes on a modern GPU, ~30 minutes on CPU). Lower is better; under 2.0 means it's learned recognizable Shakespeare.

```python
plt.figure(figsize=(8, 4))
plt.plot(train_losses, alpha=0.5)
# Smoothed
window = 50
plt.plot(np.convolve(train_losses, np.ones(window)/window, mode='valid'))
plt.xlabel("step"); plt.ylabel("train loss")
plt.title("Training loss")
plt.show()
```

---

## Exercise 10.7 — Generate (without KV cache)

The naive way — recompute attention over the entire prefix every step.

```python
@torch.no_grad()
def generate_naive(model, prompt, max_new_tokens=300, temperature=0.8, top_k=40):
    model.eval()
    idx = torch.tensor([encode(prompt)], device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_seq_len:]
        mask = torch.tril(torch.ones(idx_cond.size(1), idx_cond.size(1), device=device))
        logits = model(idx_cond, mask=mask)             # (B, T, vocab)
        logits = logits[:, -1, :] / temperature        # (B, vocab)
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_tok], dim=1)
    return decode(idx[0].tolist())

print(generate_naive(model, "ROMEO:", max_new_tokens=300))
```

Expect output that *looks* like Shakespeare — character names, iambic-ish rhythm, archaic vocabulary — without making total sense. **The model has captured the surface statistics but not the meaning.** That's still impressive for 1.5M params.

Time it:

```python
t0 = time.perf_counter()
out = generate_naive(model, "ROMEO:", max_new_tokens=500)
if device.type == 'cuda': torch.cuda.synchronize()
print(f"naive: {time.perf_counter() - t0:.3f}s for 500 tokens")
```

---

## Exercise 10.8 — Implement KV cache

Now the speed-up. The trick: cache `K` and `V` so each new token only computes its own row of attention.

```python
class KVCache:
    """Per-layer K, V cache; supports incremental appends."""
    def __init__(self, n_layers, max_len, n_heads, d_head, dtype, device):
        self.K = [torch.zeros(1, n_heads, max_len, d_head, dtype=dtype, device=device)
                  for _ in range(n_layers)]
        self.V = [torch.zeros(1, n_heads, max_len, d_head, dtype=dtype, device=device)
                  for _ in range(n_layers)]
        self.pos = 0
        self.max_len = max_len

    def update(self, layer_idx, k_new, v_new):
        # k_new, v_new: (1, n_heads, n_new, d_head)
        n = k_new.size(2)
        self.K[layer_idx][:, :, self.pos:self.pos+n] = k_new
        self.V[layer_idx][:, :, self.pos:self.pos+n] = v_new
        # only advance once per layer per step
        return self.K[layer_idx][:, :, :self.pos+n], self.V[layer_idx][:, :, :self.pos+n]

    def advance(self, n):
        self.pos += n
```

Now a cache-aware generate. The first call ingests the prompt (prefill); each subsequent call processes a single new token:

```python
@torch.no_grad()
def generate_cached(model, prompt, max_new_tokens=300, temperature=0.8, top_k=40):
    model.eval()
    cache = KVCache(
        n_layers=len(model.blocks),
        max_len=model.max_seq_len,
        n_heads=model.blocks[0].attn.n_heads,
        d_head=model.blocks[0].attn.d_head,
        dtype=torch.float32,
        device=device,
    )

    # Prefill — process the prompt once
    prompt_ids = torch.tensor([encode(prompt)], device=device)
    T = prompt_ids.size(1)
    mask = torch.tril(torch.ones(T, T, device=device))
    logits = model(prompt_ids, mask=mask, kv_cache=cache)
    cache.advance(T)

    # Decode one token at a time
    idx = prompt_ids
    for _ in range(max_new_tokens):
        last = idx[:, -1:]                        # (1, 1)
        # During decode, query attends to ALL cached K/V — no extra mask needed
        # (the cache only holds positions we've already seen)
        logits = model(last, mask=None, kv_cache=cache)
        cache.advance(1)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_tok], dim=1)
    return decode(idx[0].tolist())

t0 = time.perf_counter()
out = generate_cached(model, "ROMEO:", max_new_tokens=500)
if device.type == 'cuda': torch.cuda.synchronize()
print(f"with KV cache: {time.perf_counter() - t0:.3f}s for 500 tokens")
```

You should see **5-20× speedup**, larger as `max_new_tokens` grows. This is the single most important inference optimization in LLMs. Week 16 makes it production-grade with PagedAttention.

---

## Exercise 10.9 — Compare generations

```python
print("=== greedy ===")
print(generate_cached(model, "JULIET:", max_new_tokens=200, temperature=1e-6))

print("\n=== temperature 0.5 ===")
print(generate_cached(model, "JULIET:", max_new_tokens=200, temperature=0.5))

print("\n=== temperature 1.0 ===")
print(generate_cached(model, "JULIET:", max_new_tokens=200, temperature=1.0))

print("\n=== temperature 1.5, no top-k ===")
print(generate_cached(model, "JULIET:", max_new_tokens=200, temperature=1.5, top_k=None))
```

You'll see the spectrum: greedy is repetitive (loops on common phrases); `T=0.5` is coherent but bland; `T=1.0` is the sweet spot for creativity; `T=1.5` without `top_k` produces gibberish (the long tail of unlikely tokens gets sampled).

---

## Exercise 10.10 (stretch) — RoPE

Replace the learned position embeddings with RoPE. Approximate sketch:

```python
def precompute_rope(seq_len, d_head, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)              # (N, d_head/2)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    # x: (..., N, d_head); cos, sin: (N, d_head/2)
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    out = torch.zeros_like(x)
    out[..., ::2] = rotated_x1
    out[..., 1::2] = rotated_x2
    return out
```

Apply to `Q` and `K` inside `MultiHeadAttention.forward` before computing scores. Drop the `pos_emb` from `GPT`. Train — should converge to similar val loss with better extrapolation behavior.

---

## Submission checklist

- [ ] `scaled_dot_product_attention` matches `F.scaled_dot_product_attention` to ≤ 1e-5
- [ ] Causal attention weights visualized as lower-triangular
- [ ] Multi-head attention with fused QKV runs and produces correct shape
- [ ] Full GPT trains to val loss ≤ 1.7 on TinyShakespeare
- [ ] Generated text looks like Shakespeare-style English (recognizable structure)
- [ ] KV cache implementation works (no shape errors, generates same tokens given same seed)
- [ ] KV cache gives ≥ 5× speedup on 500-token generation
- [ ] Temperature sweep produces predictable quality vs randomness trade-off
- [ ] (Stretch) RoPE variant trains to similar val loss

---

## What you just did

You wrote a working GPT from `nn.Linear` primitives. You understand multi-head attention, position encoding, the residual block structure, causal masking, next-token prediction, and the KV cache that makes inference tractable. Every modern LLM — Llama, Mistral, GPT-4, Claude — is fundamentally what you just built, scaled up by 4-5 orders of magnitude and trained on internet-scale data.

The remaining six weeks of the curriculum (11-16) are about getting from this 1.5M-param toy to a real production system: more data + bigger model (week 11), domain adaptation via fine-tuning (week 12), the hardware that makes any of this possible (weeks 13-15), and the inference engines that serve it (week 16).

---

**Next**: [Week 11: Training at Scale →](../week-11-training-at-scale/readme.md)
