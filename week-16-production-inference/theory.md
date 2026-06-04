# Week 16: Theory — Production Inference (Capstone)

The end of the curriculum. Fifteen weeks of "make a model that works." This week: make it serve real users at real throughput at real cost.

Production LLM serving has its own physics, its own bottlenecks, and its own dominant solutions — PagedAttention, continuous batching, quantization, speculative decoding. The serving stacks (vLLM, TensorRT-LLM, TGI, Triton Inference Server) all converge on the same handful of ideas.

By the end of this week you'll be able to:
- Reason about prefill vs decode latency
- Explain why naive batching wastes 90% of GPU time
- Use vLLM to serve a 7B model with proper throughput
- Quantize a model to INT8/INT4 and measure the trade-off
- Build a tiny FastAPI wrapper with streaming
- Cost-estimate a real deployment

---

## Part 1: Inference is not training

A pre-training engineer cares about how many tokens/second they push through gradient updates. An inference engineer cares about three different numbers:

| Metric | What it measures |
|---|---|
| **Time to first token (TTFT)** | How long until the user sees anything. Dominated by **prefill**. |
| **Inter-token latency (ITL)** | Time between consecutive tokens after the first. Dominated by **decode**. |
| **Throughput (tokens/sec)** | Total system throughput across all concurrent requests. |

A chat UI prioritizes low TTFT and smooth ITL. A batch job (summarize a million documents overnight) prioritizes raw throughput. The optimal serving config depends on which you're optimizing.

The bottlenecks differ from training:

- **Training:** dominated by big matmuls. Compute-bound. Goal: max MFU (Model FLOPs Utilization).
- **Decode:** dominated by reading the KV cache. **Memory-bandwidth bound** at typical batch sizes. Goal: hide the bandwidth wall via batching.
- **Prefill:** matmul-heavy. Compute-bound, like training.

---

## Part 2: Prefill vs Decode — the two-phase architecture

When a request arrives ("Tell me a joke"), the LLM does two distinct things:

### Prefill (one shot)

Process all prompt tokens at once:

```
Input:  "Tell me a joke"   (5 tokens)
Output: KV cache populated for positions 0-4; logits for position 5
```

All matmuls are big. Tensor cores happy. **Compute-bound.** Latency scales with prompt length but is amortized over the matrix size.

### Decode (one token at a time, autoregressive)

```
Position 5: read full KV cache (length 5), compute Q for new token, attention, FFN, output token
Position 6: read full KV cache (length 6), ...
Position 7: ...
```

Each decode step does **one row** of attention against the full cache. The matmul has K = 1 on the new-Q side. Tensor cores idle. **Memory-bound** — bandwidth-limited by how fast we can read the KV cache from HBM.

For a long generation (say 1000 tokens), decode dominates total time. **The single biggest perf lever in LLM serving is batching many users' decode steps together** — so that one HBM read of the KV cache serves many concurrent users.

---

## Part 3: KV cache memory — the real bottleneck

Recall from week 10:

```
KV memory per token = 2 (K and V) × n_layers × n_heads × d_head × dtype_bytes
```

For Llama-2-7B (32 layers, 32 heads, d_head=128, bf16):

```
0.5 MB / token × 4096 tokens × 1 request = 2 GB
```

**2 GB of KV cache per concurrent request at 4k context.** On an 80 GB A100, after subtracting the model itself (~14 GB), you have ~66 GB for KV — enough for ~33 concurrent requests at 4k.

Long context makes it worse. Llama-3-8B at 128k context: **32 GB of KV per request**. Two concurrent users fills an H100. **This is the memory wall of LLM serving in 2026.**

Three families of solutions:

1. **GQA / MQA** — fewer KV heads → smaller cache (week 10)
2. **Quantize the KV cache** — int8 or fp8 instead of fp16 → 2-4× more capacity
3. **PagedAttention** — better memory management → 2-4× more useable capacity

---

## Part 4: PagedAttention — vLLM's central trick

Traditional KV cache: allocate a contiguous `(max_context_length × per_token_size)` block per request. **Wastes everything beyond what the request actually uses.**

The vLLM insight ([Kwon et al., 2023](https://arxiv.org/abs/2309.06180)): **treat KV cache like virtual memory in an OS.**

- Divide the cache into fixed-size **blocks** (e.g., 16 tokens each)
- Maintain a page table mapping `(request_id, logical block) → physical block in HBM`
- Allocate blocks on demand as the request grows
- Free blocks immediately when the request finishes
- No fragmentation — small blocks fit cleanly

The result: KV memory utilization goes from ~20-40% (with contiguous allocation) to >90%. **2-4× more concurrent users on the same hardware.**

Critically, paged blocks can also be **shared** across requests:
- Multiple users with the same system prompt → share the prompt's KV cache blocks
- Beam search over the same prefix → share the prefix blocks
- Caching popular conversation starts

This is why vLLM dominated open-source serving in 2023-2024 — it converted a memory engineering insight into 2-4× throughput improvements.

---

## Part 5: Continuous batching

Static batching: gather N requests, run them together as one batch, return results, gather the next N. Simple. Slow.

**Problem:** different requests generate different numbers of tokens. A short response (10 tokens) holds the batch up waiting for a long response (1000 tokens). The 10-token user gets billed for time they didn't use.

**Continuous batching** (also called **iteration-level scheduling**, [Yu et al. 2022](https://www.usenix.org/conference/osdi22/presentation/yu)): manage requests at the token level, not the request level.

```
At each decode step:
  - Each "slot" in the GPU runs one token of attention
  - When a slot's request finishes, it's freed
  - A waiting request fills the slot immediately
```

The GPU is always doing useful work for someone. Median throughput goes up 2-4× over static batching.

vLLM, TGI, TensorRT-LLM, all major serving stacks now do this. **For generative serving with variable-length outputs, replace static batching with continuous batching.** (Caveat: for short, uniformly-sized request types — embeddings, classification, RAG retrievers — static batching is still competitive and much simpler.)

---

## Part 6: Quantization for inference

For training you've seen bf16. For inference you can go lower:

| Precision | When | Tools |
|---|---|---|
| **bf16 / fp16** | Default; ~no quality cost | Native PyTorch |
| **INT8 (dynamic act)** | 1.5-2× faster, small quality cost | bitsandbytes, TensorRT |
| **INT4 weight-only (GPTQ, AWQ)** | 2-3× faster than bf16 | autogptq, autoawq, llama.cpp |
| **FP8 (H100+)** | ~Same as bf16 quality, ~2× faster | TensorRT-LLM, Hopper Transformer Engine |
| **INT4 (matmul incl. activations)** | 3-4× faster but quality drops more | Custom; less common |

### GPTQ vs AWQ — the modern INT4 leaders

**GPTQ** ([Frantar et al. 2022](https://arxiv.org/abs/2210.17323)): post-training quantization that uses a small calibration dataset to find quantization rounding that minimizes layer-output error. Per-channel, per-group.

**AWQ** ([Lin et al. 2023](https://arxiv.org/abs/2306.00978)): observes that not all weights are equally important — keeps "salient" weights at higher precision based on activation magnitudes. Better quality than GPTQ at int4.

Both can compress a 7B model from 14 GB (bf16) to ~3-4 GB (int4). **Llama-3-8B in int4 fits on a 6 GB consumer GPU.**

The trade-offs:

- ~5-10% perplexity increase typically — often invisible in real use
- Some tasks (math, coding) degrade more than others
- Evaluation drift can be subtle; always re-eval after quantizing
- Inference speed gain ~2-3× because **the bottleneck was memory bandwidth** — reading 4 bytes/weight instead of 16 helps a lot

### FP8 on Hopper / Blackwell

H100's transformer engine quantizes activations to FP8 (E4M3) and gradients to FP8 (E5M2) automatically. **~2× faster than bf16 at essentially identical quality.** Future default.

---

## Part 7: Speculative decoding

The cleverest decode optimization: **use a tiny model to draft N tokens, then verify with the big model in parallel.**

```
1. Draft model (e.g., 1B) generates next 5 tokens speculatively
2. Big model (e.g., 70B) processes all 5 candidates in parallel
3. Accept tokens where big model agrees, reject the rest, take big model's correction
4. Net: 2-4× faster than pure big-model decode, identical quality
```

The math: if the draft is right 80% of the time, you generate ~4 tokens per big-model forward instead of 1. Verification is "almost free" not because the forward becomes compute-bound (at batch=1 the big-model decode is still memory-bound by KV-cache reads) but because **the KV-cache read amortizes across all 5 candidate positions in one forward pass** — the bulk of the bandwidth cost is paid once per 5 tokens instead of once per 1.

Implementations: vLLM, TensorRT-LLM, llama.cpp all support speculative decoding. Pick a draft model that's small but similar to the target (same family, smaller size).

---

## Part 8: The serving stack zoo

The major options in 2026:

| Stack | Strengths | Weaknesses |
|---|---|---|
| **vLLM** | Best PagedAttention, easy to deploy, broad model support | Less production-tested than TensorRT-LLM for stability extremes |
| **TensorRT-LLM** | NVIDIA's official, peak performance on NVIDIA hardware | Hardest to set up; NVIDIA-only |
| **TGI (HuggingFace Text Generation Inference)** | Easy with HF models | Smaller perf gains than vLLM |
| **Triton Inference Server** | NVIDIA's generic server (not LLM-specific) | Wraps another runtime; more layers |
| **llama.cpp / ollama** | CPU + small GPU; great for laptops | Single-user throughput; not production scale |
| **MLC-LLM** | Cross-platform (mobile, Apple, Vulkan) | Smaller ecosystem |

**For most production teams in 2026**: vLLM is the default. TensorRT-LLM if peak NVIDIA performance is the priority and you can stomach the deployment complexity.

For **inference at the edge** (laptops, phones): llama.cpp + a quantized GGUF model. Surprisingly capable.

---

## Part 9: Cost math

To deploy is to pay. Some napkin numbers (2026 rental pricing, varies wildly):

| GPU | $/hr | bf16 TFLOPS | VRAM |
|---|---|---|---|
| T4 (16 GB) | ~$0.30-0.50 | 65 | 16 GB |
| A10G (24 GB) | ~$1 | 125 | 24 GB |
| L40S (48 GB) | ~$2 | 362 (fp8) | 48 GB |
| A100 (40 GB) | ~$1.50 | 312 | 40 GB |
| A100 (80 GB) | ~$2 | 312 | 80 GB |
| H100 (80 GB) | ~$3-4 | 989 / 1979 (fp8) | 80 GB |
| H200 (141 GB) | ~$5-6 | 989 | 141 GB |

To estimate `$ per million tokens`:

```
tokens_per_hour = batch_size × output_tokens_per_request × requests_per_second_at_batch × 3600
$/M_tokens = hourly_rate / (tokens_per_hour / 1e6)
```

For Llama-3-8B in fp8 on an H100 at vLLM throughput (~5000 tokens/sec sustained):

```
tokens_per_hour = 5000 × 3600 = 18M
$/M_tokens = $3.50 / 18 = $0.19 per million tokens
```

Compare to commercial APIs (~$0.30-1.50 per million for similar-quality models). At-cost serving on rented hardware is competitive — most of the markup in APIs is reliability, fine-tuning, and human support, not raw compute.

For **inference-heavy applications** (chatbot with millions of users) this math matters enormously. For **research-internal use** (10k tokens/day) just use someone else's API.

---

## Part 10: The deployment pattern

A real production LLM endpoint architecture:

```
┌──────────────────────────────────────────────────────────────┐
│                     Load Balancer                              │
│                  (rate limiting, auth)                         │
└──────────────────────┬───────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ vLLM    │   │ vLLM    │   │ vLLM    │
   │ replica │   │ replica │   │ replica │   ← autoscale based on queue depth
   │ (H100)  │   │ (H100)  │   │ (H100)  │
   └─────────┘   └─────────┘   └─────────┘
        │              │              │
        └──────┬───────┴──────────────┘
               ▼
        ┌──────────────┐
        │ Monitoring   │   ← latency, throughput, GPU util, error rate
        │ (W&B / DD)   │
        └──────────────┘
```

For streaming generation:
- WebSocket or **Server-Sent Events** (SSE) — tokens stream back as they generate
- Client receives tokens with ~50-200ms gaps between them
- Cancel-on-disconnect — if the client closes the connection, abort the generation (frees the GPU slot)

Observability that matters:
- **TTFT p50/p95/p99** — user-facing latency
- **ITL p50/p95** — smoothness
- **Tokens/second** — throughput
- **GPU utilization** — capacity
- **KV cache utilization** — when this nears 100%, you'll start rejecting requests
- **Queue depth** — how long requests wait before processing

`nvidia-smi` doesn't tell you most of this. vLLM exposes Prometheus metrics; scrape them.

---

## Part 11: When to fine-tune for inference

A specific anti-pattern: deploying a 70B base model with a complex prompt vs. a fine-tuned 7B for the same task.

A **fine-tuned 7B** can match a **prompted 70B** for many specific tasks at ~10× lower inference cost and ~5× lower latency. **The serving math for fine-tuning is usually overwhelmingly in favor.**

Three rules of thumb:
- High volume + narrow task → fine-tune a small model
- Low volume + diverse tasks → big model with prompting
- Anything regulated (medical, financial, legal) → own the model end-to-end

---

## Part 12: The end of the curriculum — what you have now

Sixteen weeks. Start: linear algebra and Python loops. End: deploying production LLMs with custom kernels and a 10-cents-per-million-token cost model.

You can:

- **Read** the FlashAttention CUDA code and understand every line
- **Build** a transformer from `nn.Linear` primitives
- **Fine-tune** a 7B model on a consumer GPU
- **Profile** a training run and identify the bottleneck
- **Deploy** a model with vLLM and quote production economics

This is the bar a modern ML engineer should clear. Most don't — most stay at the "use PyTorch" layer. The curriculum was designed so that by week 16 you've gone three layers below.

### What to do after

For most learners, the right move now is **specialization**:

| Direction | What to add |
|---|---|
| **ML research** | Reproduce one paper per month; pick a subfield (alignment, multimodal, agents); read [Lilian Weng's blog](https://lilianweng.github.io/) start to finish |
| **ML engineering / MLOps** | Kubernetes for ML, KServe, Ray Serve, model registries; [Full Stack Deep Learning](https://fullstackdeeplearning.com/) course |
| **LLM applications** | RAG (LangChain, LlamaIndex, Haystack); agents; eval harnesses for production |
| **Kernel / systems work** | Contribute to vLLM, FlashAttention, xFormers; read more CUTLASS; apply to NVIDIA |
| **Robotics + RL** | OpenAI Spinning Up; Isaac Sim; D'Erica's RL blog |
| **Multimodal** | Diffusion models, ControlNet, vision-language models |

Pick **one** for the next 6-12 months. Depth in one beats breadth across six.

### Where to keep learning from

- **Papers**: arxiv.org → ML/cs.CL categories. Skim daily; deep-read weekly.
- **Blogs**: Sebastian Raschka, Lilian Weng, Chip Huyen, Eugene Yan, Simon Willison, Hugging Face papers blog
- **Code**: nanoGPT (Karpathy), llama.cpp, vLLM, TensorRT-LLM. **Read source.**
- **Talks**: NeurIPS, ICLR, ICML, MLSys. YouTube recordings are free.
- **Community**: Hugging Face Discord, EleutherAI Discord, your local AI meetup, Twitter/X (yes, still)

### Career notes

This curriculum doesn't replace a CS degree, but at the senior individual-contributor level, what matters is **what you can ship**. Carry a portfolio: the GPT you built (week 10), the fine-tuned pirate model (week 12), the CUDA kernels (week 14), the deployed endpoint (this week). Public GitHub. Crisp READMEs. Senior engineers and hiring managers look at code, not credentials.

---

## What's next

In [lab.md](lab.md) you'll:
- Set up vLLM and serve a 7B model
- Measure throughput at different batch sizes
- Quantize a model with AWQ
- Build a tiny FastAPI streaming endpoint
- (Capstone) Deploy your fine-tuned model on Modal / Replicate / runpod and report production-shape numbers (tokens/sec, $/M tokens, p99 latency)

Then the curriculum closes. Sixteen weeks of work, sixteen working artifacts, the full stack of modern ML in your head.
