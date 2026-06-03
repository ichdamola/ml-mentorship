# Week 10: Theory — Transformers from Scratch

The architecture behind every modern LLM, every image-generation model, every speech model. Once you've built one from `nn.Linear` primitives, the gap between "uses HuggingFace" and "understands HuggingFace" closes for good.

By the end of this week you'll have:
- Derived self-attention from "I need a content-addressable lookup"
- Implemented multi-head attention, causal masking, position encoding, KV cache by hand
- Trained a nanoGPT-sized model on Shakespeare and generated text

Read the original [Attention Is All You Need](https://arxiv.org/abs/1706.03762) before this week — it's denser than its 8 pages suggest. Then this file walks every figure.

---

## Part 1: Why not RNNs?

For sequence modeling, the pre-transformer era ran on RNNs/LSTMs. They had three problems:

1. **No parallelism within a sequence.** To compute token N's hidden state, you need token N-1's. Training is sequential along the time axis, even on a GPU with thousands of cores.
2. **Long-range dependencies vanish.** Information from token 1 must survive 100 multiply-and-tanhs to reach token 100. Gradient signal degrades exponentially.
3. **Fixed hidden-state bottleneck.** Everything from the past must squeeze through one hidden vector.

The 2017 transformer fixed all three:

1. **Fully parallel within a sequence** — every token attends to every other token in one matmul.
2. **Direct path between any two tokens** — attention is one hop, regardless of distance.
3. **No fixed bottleneck** — the model decides at each step which past tokens matter.

The cost: attention is `O(N²)` in sequence length. For long contexts you pay. Workarounds (FlashAttention, sliding window, sparse patterns) are week 14+ topics.

---

## Part 2: Self-attention — content-addressable lookup

The simplest framing: every token in a sequence wants to "look up" information from other tokens. Three vectors per token:

- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what information do I carry?"
- **Value (V)** — "if you decide I'm relevant, this is what you get"

Token `i` computes its output by comparing **its** Q with **every** K (dot products) to get attention weights, then taking a weighted sum of all **V**s using those weights.

The formula:

```
attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V
```

Shapes for a batch:
- `Q`, `K`: `(B, N, d_k)` — batch size, sequence length, key dim
- `V`: `(B, N, d_v)` (usually `d_v = d_k`)
- `Q Kᵀ`: `(B, N, N)` — pairwise dot products
- After softmax: `(B, N, N)` — each row sums to 1 (attention weights)
- Final output: `(B, N, d_v)`

In code:

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q, K, V: (B, N, d_k)
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)    # (B, N, N)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)                   # (B, N, N)
    return weights @ V                                     # (B, N, d_v)
```

That's three lines. **Self-attention is just "scaled cosine-similarity lookup, then weighted sum."**

### Why divide by √d_k?

Without scaling, dot products of `d_k`-dim vectors with unit variance entries grow as `O(d_k)` in magnitude. Big scores → softmax saturates → almost all weight on one token → gradient through softmax dies.

Dividing by `√d_k` keeps dot-product variance ≈ 1 regardless of `d_k`. **One of the cheapest, most consequential design choices in modern ML.**

---

## Part 3: Where do Q, K, V come from?

Each from a linear projection of the input `X` (shape `(B, N, d_model)`):

```python
W_Q = nn.Linear(d_model, d_k, bias=False)
W_K = nn.Linear(d_model, d_k, bias=False)
W_V = nn.Linear(d_model, d_v, bias=False)

Q = W_Q(X)     # (B, N, d_k)
K = W_K(X)     # (B, N, d_k)
V = W_V(X)     # (B, N, d_v)
```

These three projection matrices are the **only learnable parameters** in attention. The matmuls / softmax are deterministic.

In **self**-attention, Q, K, V all come from the same X. In **cross**-attention (used in encoder-decoder transformers like the original 2017 paper, T5, and image-text models), Q comes from one sequence and K, V from another.

---

## Part 4: Multi-head attention

One attention head learns one kind of relationship — say, "verbs attending to their subjects." But sentences need many: syntactic dependencies, coreference, position, semantic similarity.

**Multi-head attention runs H independent attention computations in parallel**, each with its own Q/K/V projections, then concatenates and projects:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_qkv = nn.Linear(d_model, 3 * d_model)   # fused
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, N, _ = x.shape

        # One big projection, split into Q, K, V, then reshape to multi-head
        qkv = self.W_qkv(x)                              # (B, N, 3*d_model)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)                  # (3, B, n_heads, N, d_head)
        Q, K, V = qkv[0], qkv[1], qkv[2]                  # each (B, n_heads, N, d_head)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        out = weights @ V                                  # (B, n_heads, N, d_head)

        # Concatenate heads
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        return self.W_o(out)
```

A few notes:

- Total parameters: `3 × d_model² + d_model² = 4 × d_model²`. The number of heads doesn't change the parameter count — it only changes how the `d_model` budget is *organized*.
- **The fused `W_qkv` (one big projection) is a real production trick.** One matmul instead of three is faster on GPU. PyTorch's `nn.MultiheadAttention` does this.
- `mask` is broadcast across heads — usually shape `(N, N)` or `(B, 1, N, N)`.

---

## Part 5: Causal masking — the GPT trick

For an **autoregressive** model (like GPT), token `i` must not see tokens at positions `j > i`. Otherwise training is cheating — the model peeks at the answer.

The fix: a lower-triangular mask that zeros out the upper triangle of attention scores:

```python
# (N, N) lower triangular mask
mask = torch.tril(torch.ones(N, N))
# scores.masked_fill(mask == 0, -1e9)  → upper triangle becomes very negative
# softmax → upper triangle is ~0
```

After softmax, each row only attends to current + earlier positions. This is the difference between a decoder (GPT) and encoder (BERT) attention: BERT sees everything (no mask), GPT only sees the past.

**Pre-fill vs decode (sets up week 16):**
- During **training** or **prefill** at inference, you process the whole sequence at once and use a triangular mask.
- During **decode** (generating one token at a time), you don't need the mask — each new query only sees existing K, V. This is where the KV cache enters (Part 8).

---

## Part 6: Position encoding — putting "where" back in

Attention is **permutation-invariant**: shuffling the input tokens shuffles the output identically. The model has no inherent notion of order.

To inject position, we add a position-dependent vector to each token's embedding.

### Sinusoidal (original 2017)

```python
def sinusoidal_pe(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(1)
    i = torch.arange(0, d_model, 2)
    div_term = 10000 ** (i / d_model)
    pe[:, 0::2] = torch.sin(pos / div_term)
    pe[:, 1::2] = torch.cos(pos / div_term)
    return pe
```

Even dims get sine, odd dims get cosine, with frequencies on a geometric progression. The clever property: relative position can be computed as a linear function of `PE(pos+k) − PE(pos)`.

### Learned absolute (GPT-2/3)

Just `nn.Embedding(max_seq_len, d_model)`. Simple, works, doesn't extrapolate past `max_seq_len`.

### RoPE — Rotary Position Embedding (2021, now standard for LLMs)

Instead of adding to the embedding, **rotate Q and K in pairs of dimensions** by an angle proportional to position:

```
For each pair of dims (2i, 2i+1):
    [q_2i, q_2i+1]^T ← R(θ_i * pos) [q_2i, q_2i+1]^T
where R(α) is the standard 2D rotation matrix.
```

Why it works: the inner product `Q^T K` after RoPE depends only on the **relative** position `pos_q - pos_k`, not absolute positions. The model gets relative-position awareness "for free" — and RoPE extrapolates to longer contexts than seen at training, with some tricks ([NTK-aware RoPE](https://arxiv.org/abs/2306.15595), YaRN).

**Used by:** LLaMA, GPT-NeoX, Falcon, Mistral, basically every open LLM from 2023 onwards.

### ALiBi — Attention with Linear Biases

Add a position-dependent bias directly to attention scores: `score(i, j) += -m · |i - j|`. Zero parameters; extrapolates extremely well. Used by BLOOM, MosaicML.

For this week's lab you'll implement sinusoidal (it's the simplest); the readings cover RoPE for the curious.

---

## Part 7: The transformer block

The full block, in modern (pre-norm) form:

```
x → LayerNorm → MultiHeadAttention → +x (residual) →
  → LayerNorm → FFN → +(prev) (residual)
```

In code:

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x
```

Two sub-layers, each wrapped in residual + layer norm. **The recipe hasn't changed since 2017.** Whole models are just N copies of this block stacked.

### FFN: why `4× d_model`?

The standard ratio is `d_ff = 4 × d_model`. No deep theory — it's empirically what works. GeGLU variants (LLaMA, Mistral) use `d_ff = 8/3 × d_model` with a different activation; the parameter budget is similar.

### Pre-norm vs post-norm

The original 2017 paper put LayerNorm **after** the residual add ("post-norm"). It's harder to train at depth — gradients destabilize. Modern transformers (GPT-2 onwards) put LayerNorm **before** ("pre-norm"). The lab uses pre-norm.

### RMSNorm

LLaMA replaces LayerNorm with **RMSNorm**, which drops the centering and learnable shift:

```
y = x / RMS(x) * γ
```

Faster, fewer parameters, works equivalently. By 2024 every new LLM uses RMSNorm.

---

## Part 8: KV cache — making inference fast

When generating token N+1, the model recomputes attention for the entire prefix `1..N`. The Q, K, V for tokens `1..N` haven't changed since the last step. You're paying `O(N²)` work per token, leading to `O(N³)` total generation time for an N-token output.

**The fix: cache K and V across decoding steps.** Each new token only computes:
- Its own Q, K, V
- One attention row: `softmax(q_new · K_cached^T / √d) · V_cached`

This is `O(N)` per token, `O(N²)` total. **The single most important inference optimization in LLMs.** Without it, GPT-4-class models would be unusably slow.

```python
class KVCache:
    """Per-layer K and V cache, one tensor per layer."""
    def __init__(self, max_len, n_heads, d_head, dtype, device):
        self.K = torch.zeros(1, n_heads, max_len, d_head, dtype=dtype, device=device)
        self.V = torch.zeros(1, n_heads, max_len, d_head, dtype=dtype, device=device)
        self.pos = 0

    def update(self, k_new, v_new):
        n = k_new.size(2)
        self.K[:, :, self.pos:self.pos+n] = k_new
        self.V[:, :, self.pos:self.pos+n] = v_new
        self.pos += n
        return self.K[:, :, :self.pos], self.V[:, :, :self.pos]
```

You'll implement this in the lab. Week 16 makes it production-grade with **PagedAttention** (the vLLM trick) — paging the cache like an OS pages memory.

### KV cache memory math

For a model with `L` layers, `H` heads, `D_head` per head, sequence length `N`, and dtype size `B` bytes:

```
KV memory = 2 (K and V) × L × H × N × D_head × B × batch
          = 2 L H N D_head B (per request)
```

For Llama-2-7B (L=32, H=32, D_head=128, fp16 → B=2):
```
KV/token = 2 × 32 × 32 × 128 × 2 = 524,288 bytes ≈ 0.5 MB
At N=4096 tokens: 2 GB per request just for KV cache.
```

This dominates inference memory at long sequences. Quantization (week 16) and grouped-query attention (Part 9) attack this directly.

---

## Part 9: MHA, MQA, GQA — the inference-memory triangle

| Variant | K, V per layer | When used |
|---|---|---|
| **MHA** (Multi-Head Attention) | H copies (one per Q head) | The 2017 default; GPT-2/3 |
| **MQA** (Multi-Query Attention) | 1 copy shared across heads | PaLM, Falcon — saves KV memory but loses quality |
| **GQA** (Grouped-Query Attention) | G copies (groups of heads share K,V), `1 ≤ G ≤ H` | LLaMA-2/3, Mistral, Mixtral — sweet spot |

GQA with `G = 8` (8 KV heads, 32 Q heads) is the modern default. KV cache shrinks 4×, quality essentially unchanged. **Always GQA in 2026 unless you're recreating a 2020-era model.**

---

## Part 10: The full GPT model

Stacking blocks with an embedding layer up front and an LM head at the end:

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_layers=6, n_heads=6,
                 d_ff=4*384, max_seq_len=256, dropout=0.0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying — share token_emb and lm_head weights
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx, mask=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.lm_head(x)
```

**Parameter count**, roughly:

```
≈ vocab × d_model + n_layers × 12 × d_model²
```

The dominant term is `n_layers × 12 × d_model²` (the 4 d_model² for attention QKV+O plus 8 d_model² for FFN with d_ff = 4 × d_model). For Llama-2-7B with d_model=4096, n_layers=32: 32 × 12 × 4096² ≈ 6.4B parameters, which matches the "7B" tag.

---

## Part 11: Training a language model

The loss is **next-token prediction** — for each position, predict the next token.

```python
def lm_loss(logits, targets):
    # logits: (B, T, vocab), targets: (B, T)
    # Shift: predict token[t] from token[<t]
    # Logits at position t predicts targets at position t
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
```

This is **the categorical NLL from week 02** — a softmax over the vocabulary, log of the true token's predicted probability.

### Why this generalizes to "intelligence"

A model that's *really* good at next-token prediction has implicitly learned grammar, facts, reasoning patterns, code structure. **The training loss is simple; the emergent capabilities aren't.** This insight (scaling next-token prediction) is what got us GPT-3 and beyond.

### Sampling at inference time

After training, the model outputs a distribution over the next token. You sample from it:

| Strategy | What it does |
|---|---|
| **Greedy** | argmax — deterministic, often repetitive |
| **Temperature `T`** | Divide logits by T before softmax. Lower → more deterministic, higher → more random |
| **Top-k** | Sample only from the top-k tokens |
| **Top-p (nucleus)** | Sample from the smallest set whose cumulative prob ≥ p |
| **Min-p** | Newer; threshold by probability ratio to max |

Combining temperature + top-k + top-p is the standard inference recipe. `temperature=0.7, top_p=0.9` is a sensible default for creative text.

---

## Part 12: What's next

You'll spend week 11 making this trainable on real-sized data (AMP, gradient accumulation, multi-GPU), week 12 fine-tuning a pretrained one (LoRA / QLoRA), week 14 writing CUDA kernels for it (FlashAttention), and week 16 serving it (vLLM, KV cache management, paged attention).

In [lab.md](lab.md) you'll:
- Implement scaled dot-product attention by hand and verify
- Build multi-head attention with the fused QKV trick
- Implement sinusoidal position encoding
- Build the transformer block (pre-norm) and the full GPT
- Train nanoGPT on TinyShakespeare; generate plausible Shakespeare
- Implement KV cache and measure the speedup vs no-cache decoding

By end of week 10 you'll have a working GPT you can read end-to-end — and the rest of the LLM stack will feel like obvious refinements.
