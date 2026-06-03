# Week 16: Lab — Serve It (Capstone)

The final lab. Stand up vLLM, serve a 7B model, measure throughput and latency, quantize for cost, build a streaming API, deploy.

## Setup

You'll want a GPU with **≥ 16 GB VRAM** for the 7B-bf16 work, **≥ 8 GB** for AWQ-quantized. Free Colab T4 (16 GB) handles everything.

```bash
uv add vllm
uv add autoawq               # for AWQ quantization
uv add fastapi uvicorn sse-starlette
```

For deployment-target exercises later: an account on **Modal**, **Replicate**, or **RunPod** (all have free tiers / cheap hourly).

---

## Exercise 16.1 — Stand up vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-3B-Instruct",     # a small instruction model; under 8 GB in bf16
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.85,
)

prompts = [
    "Tell me about black holes.",
    "Explain the Pythagorean theorem.",
    "What is the capital of France?",
]

sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)
outputs = llm.generate(prompts, sampling)

for out in outputs:
    print("=" * 40)
    print(f"Prompt: {out.prompt}")
    print(f"Output: {out.outputs[0].text}")
```

That's it. **vLLM is intentionally that simple to use.** Under the hood it just loaded the model, set up PagedAttention, and ran continuous batching.

If you don't have enough VRAM for 3B, try `Qwen/Qwen2.5-0.5B-Instruct` — runs on anything.

---

## Exercise 16.2 — Throughput benchmark

How many tokens/sec does your setup actually produce?

```python
import time

# Generate a lot of prompts to keep the engine busy
N = 50
prompts = [f"Write a short story about {animal}." for animal in
           ["cats", "elephants", "robots", "dragons", "mice", "tigers", "lions", "wolves", "bears", "foxes"] * 5]

sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=200)

t0 = time.perf_counter()
outputs = llm.generate(prompts, sampling)
elapsed = time.perf_counter() - t0

total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
print(f"Generated {total_tokens} tokens in {elapsed:.2f}s")
print(f"Throughput: {total_tokens/elapsed:.0f} tokens/sec")
print(f"Per-request avg: {total_tokens/N:.0f} tokens")
```

Typical results for a small model (3B) on a T4: **600-1500 tokens/sec sustained**. On an A100/H100 with a 7B model: **2000-6000 tokens/sec**. **That's the throughput math behind "cents per million tokens."**

---

## Exercise 16.3 — TTFT vs ITL — the two latencies

For a chat experience, **TTFT** (time-to-first-token) matters more than total time. Measure both:

```python
from vllm import LLM, SamplingParams

# Single prompt, watch streaming
prompt = "Explain how a neural network learns, step by step."
sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=200)

# vLLM's offline API doesn't give per-token timings; use the AsyncLLMEngine for that
# Or measure batches at different sizes:

import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

async def measure_streaming(engine, prompt, sampling):
    request_id = random_uuid()
    results = engine.generate(prompt, sampling, request_id)
    first_token_time = None
    last_time = None
    n_tokens = 0
    start = time.perf_counter()
    async for request_output in results:
        now = time.perf_counter()
        if first_token_time is None and len(request_output.outputs[0].token_ids) > 0:
            first_token_time = now - start
        n_tokens = len(request_output.outputs[0].token_ids)
        last_time = now
    return first_token_time, n_tokens, last_time - start

# Optional: skip if AsyncLLMEngine is heavyweight; benchmark with the offline API:
def estimate_per_request_metrics(llm, prompts, sampling):
    """Run requests one at a time to measure prefill + decode separately."""
    short_prompt = prompts[0]
    sampling_one_token = SamplingParams(max_tokens=1, temperature=0.0)
    sampling_many = SamplingParams(max_tokens=128, temperature=0.0)

    # 1) Time to produce first token (prefill + 1 decode)
    t0 = time.perf_counter()
    llm.generate([short_prompt], sampling_one_token)
    t_ttft = (time.perf_counter() - t0) * 1000

    # 2) Time to produce 128 tokens
    t0 = time.perf_counter()
    out = llm.generate([short_prompt], sampling_many)
    elapsed = time.perf_counter() - t0
    n_tokens = len(out[0].outputs[0].token_ids)
    avg_per_token = (elapsed - t_ttft/1000) / max(1, n_tokens - 1) * 1000

    print(f"TTFT (prefill + 1):     {t_ttft:.1f} ms")
    print(f"Inter-token latency:     {avg_per_token:.1f} ms")
    print(f"  → throughput single: {1000/avg_per_token:.0f} tokens/sec")

estimate_per_request_metrics(llm, prompts, sampling)
```

You should see:
- **TTFT**: 200-1000 ms (depends on prompt length and model)
- **ITL**: 20-100 ms per token (single user)

**ITL × output_length = total generation time.** A 200-token response at 50ms ITL = 10s of generation. That's why streaming matters — users see progress immediately.

---

## Exercise 16.4 — Batch size affects everything

Try the same workload at different concurrency:

```python
# 30 short prompts vs 30 long prompts
short_prompts = ["What is 2 + 2?"] * 30
long_prompts = ["Explain the theory of relativity in detail with examples."] * 30

for name, prompts in [("short prompts", short_prompts), ("long prompts", long_prompts)]:
    for max_tokens in [16, 64, 256]:
        sampling = SamplingParams(max_tokens=max_tokens, temperature=0.7)
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling)
        elapsed = time.perf_counter() - t0
        total = sum(len(out.outputs[0].token_ids) for out in outputs)
        print(f"{name:<14s} max_tokens={max_tokens:>4d}: {total/elapsed:>6.0f} tok/s")
```

You'll see throughput **grow with batch size** until KV cache fills. After that point, vLLM starts rejecting requests or queueing them.

This is the optimization knob: **bigger batches = higher throughput per dollar; smaller batches = better TTFT for individual users.** Production systems tune this against their workload mix.

---

## Exercise 16.5 — Quantize a model with AWQ

For models that don't fit your VRAM, or for cost reduction:

```python
# This needs to run separately — AWQ quantization is heavy
# (For lab purposes, you can download a pre-quantized model instead)

# Use a pre-quantized AWQ model from HuggingFace (much faster than running AWQ yourself):
from vllm import LLM

llm_awq = LLM(
    model="Qwen/Qwen2.5-7B-Instruct-AWQ",   # 7B model, INT4 weight-only, ~5 GB
    quantization="awq",
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.85,
)

# Quick sanity check
sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)
out = llm_awq.generate(["Explain quantum entanglement."], sampling)
print(out[0].outputs[0].text)
```

Compare:

```python
# 7B model at bf16 needs ~16 GB
# 7B model at AWQ INT4 needs ~5 GB
# 7B AWQ is roughly 1.5-2× faster per token decode (memory-bound case)
print("AWQ INT4 7B model loaded. VRAM use:")
import subprocess
print(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv"]).decode())
```

**For a budget-constrained deployment, AWQ INT4 is the lever.** ~3× less VRAM, ~1.5-2× faster decode, with quality loss often invisible in real use.

---

## Exercise 16.6 — Build a FastAPI streaming endpoint

Wrap vLLM in a tiny web API.

```python
# server.py
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
import json

app = FastAPI()

engine_args = AsyncEngineArgs(
    model="Qwen/Qwen2.5-3B-Instruct",
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.85,
)
engine = AsyncLLMEngine.from_engine_args(engine_args)


@app.post("/generate")
async def generate(request: dict):
    prompt = request["prompt"]
    max_tokens = request.get("max_tokens", 256)
    temperature = request.get("temperature", 0.7)

    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.9,
    )

    request_id = random_uuid()
    results = engine.generate(prompt, sampling, request_id)

    async def event_stream():
        async for request_output in results:
            text = request_output.outputs[0].text
            yield json.dumps({"text": text})
        yield json.dumps({"done": True})

    return EventSourceResponse(event_stream())


@app.get("/health")
async def health():
    return {"status": "ok"}
```

Run:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Test from another terminal:

```bash
curl -N -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me about Mars", "max_tokens": 100}'
```

You should see tokens stream back as Server-Sent Events. **This is the shape of every production LLM API endpoint.** OpenAI's, Anthropic's, Cohere's — same SSE pattern under the hood.

---

## Exercise 16.7 — Concurrent request handling

Hit your server with many concurrent requests:

```python
import asyncio, httpx, time

async def fire_one(client, idx):
    t0 = time.perf_counter()
    async with client.stream("POST", "http://localhost:8000/generate",
                              json={"prompt": f"Tell me fact #{idx} about space", "max_tokens": 80}) as r:
        first_byte = None
        async for line in r.aiter_lines():
            if line.strip():
                if first_byte is None:
                    first_byte = time.perf_counter() - t0
                last_time = time.perf_counter() - t0
    return first_byte, last_time

async def main():
    async with httpx.AsyncClient(timeout=60) as client:
        N = 20
        t0 = time.perf_counter()
        results = await asyncio.gather(*[fire_one(client, i) for i in range(N)])
        elapsed = time.perf_counter() - t0
        ttfts = [r[0] for r in results if r[0]]
        totals = [r[1] for r in results if r[1]]
        print(f"{N} concurrent requests in {elapsed:.2f}s")
        print(f"TTFT p50={sorted(ttfts)[len(ttfts)//2]:.2f}s p99={sorted(ttfts)[-1]:.2f}s")
        print(f"Total p50={sorted(totals)[len(totals)//2]:.2f}s p99={sorted(totals)[-1]:.2f}s")

asyncio.run(main())
```

vLLM's continuous batching does this efficiently — 20 concurrent users don't take 20× the time of one user. **Typically 2-4× the time, depending on KV cache headroom.**

---

## Exercise 16.8 — Cost / throughput math

Pick numbers from Exercises 16.2 and 16.5 and compute your $/M tokens.

```python
def cost_per_million_tokens(throughput_tok_per_sec, gpu_hourly_cost):
    tokens_per_hour = throughput_tok_per_sec * 3600
    cost = gpu_hourly_cost / (tokens_per_hour / 1e6)
    return cost

# Plug in your numbers!
for gpu, hourly, throughput in [
    ("T4 (3B bf16)",          0.35,  600),
    ("A10G (7B bf16)",        1.00,  1500),
    ("L40S (7B AWQ)",         2.00,  4000),
    ("A100 (7B bf16)",        1.50,  2500),
    ("H100 (7B bf16)",        3.50,  5000),
    ("H100 (7B FP8)",         3.50,  9000),
    ("Closed-source API",     None,  None),
]:
    if hourly:
        cost = cost_per_million_tokens(throughput, hourly)
        print(f"{gpu:<25s}  ${cost:.3f} / M tokens (1M tokens takes {1e6/throughput:.0f}s)")
```

For comparison, commercial API pricing (early 2026):
- **GPT-4o**: ~$2.50 input / $10 output per M tokens
- **Claude Sonnet**: ~$3 input / $15 output per M tokens
- **Gemini Pro**: ~$0.50 input / $1.50 output per M tokens

For **at-cost serving of 7B-class fine-tunes**, you're typically 5-30× cheaper than commercial APIs. That's the business case for self-hosting.

---

## Exercise 16.9 — Deploy somewhere real (capstone deliverable)

Pick one and deploy your model:

### Option A — Modal (simplest)

```python
# modal_deploy.py
import modal

stub = modal.Stub("ml-mentorship-capstone")
image = (modal.Image.debian_slim()
         .pip_install("vllm==0.6.3.post1", "fastapi", "uvicorn", "sse-starlette"))

@stub.function(
    image=image,
    gpu="L4",                   # 24 GB; or "A100" or "H100" for bigger
    timeout=600,
    container_idle_timeout=300,
)
@modal.web_endpoint(method="POST")
def generate(request: dict):
    from vllm import LLM, SamplingParams
    llm = LLM("Qwen/Qwen2.5-3B-Instruct", dtype="bfloat16")
    sampling = SamplingParams(
        temperature=request.get("temperature", 0.7),
        max_tokens=request.get("max_tokens", 256),
    )
    outputs = llm.generate(request["prompt"], sampling)
    return {"text": outputs[0].outputs[0].text}
```

```bash
modal deploy modal_deploy.py
# Returns a public URL
```

### Option B — Replicate

Sign up, follow [cog deployment guide](https://github.com/replicate/cog). Slightly more setup, gets you a public model card and API.

### Option C — RunPod / Lambda Cloud serverless

Spin up a GPU pod, run your FastAPI server, expose port 8000. More DIY; cheaper for sustained traffic.

### Option D — Your own GPU + Cloudflare Tunnel

If you have a local GPU, expose it to the internet via `cloudflared tunnel`. Free, totally yours.

---

## Capstone Deliverable

Build a public-facing endpoint serving **your fine-tuned model from week 12** (or a 3B base model if that's where you stopped).

Write a `CAPSTONE.md` in your project repo containing:

1. **Architecture diagram** (ASCII art or Mermaid)
2. **Live URL** that returns generations
3. **Throughput numbers**: tokens/sec sustained, p50/p99 TTFT, peak concurrent users
4. **Cost analysis**: $/M tokens at your config
5. **A demo curl command** anyone can run
6. **What you'd do next** if this was a real product

Push to GitHub. Send the link. **This is your portfolio piece.**

---

## Submission checklist

- [ ] vLLM serves a real model and returns sensible generations
- [ ] Throughput benchmarked at multiple batch sizes
- [ ] TTFT and ITL measured separately
- [ ] AWQ-quantized 7B model loaded; VRAM compared to bf16
- [ ] FastAPI server with SSE streaming runs and tokens stream over the wire
- [ ] Concurrent request handling demonstrated; latency holds up
- [ ] Cost math computed for at least 3 GPU configurations
- [ ] Live deployment URL produced (Modal / Replicate / RunPod / your own)
- [ ] `CAPSTONE.md` written with arch diagram, numbers, demo, next-step

---

## What you just did

You took a model — pretrained, fine-tuned, profiled, your own — and put it in front of real users with production-grade economics. **The full ML stack, end to end.** Every concept from weeks 01-15 just paid its dividend in this one week.

---

# 🎓 You finished the curriculum.

Sixteen weeks. Math foundations to production LLM inference. **Look at what you've built**:

- A hand-derived MLP backward pass that trains MNIST to 97% in pure numpy (week 5)
- Your own autograd engine, ~200 lines (week 6)
- A nanoGPT-sized transformer with a working KV cache (week 10)
- A fine-tuned LLM speaking a custom voice (week 12)
- CUDA kernels that approach cuBLAS performance (week 14)
- A live, throughput-benchmarked, cost-accounted serving endpoint (week 16)

That portfolio is, no exaggeration, what most ML engineers in 2026 *should* be able to do but few actually can. **You're in rare company.**

## Where to go next

Pick one direction for the next 6-12 months and go deep:

| If you want to | Add |
|---|---|
| **Do ML research** | Reproduce papers monthly; pick a subfield (alignment, multimodal, agents); read [Lilian Weng's blog](https://lilianweng.github.io/) start to finish; submit to NeurIPS workshops |
| **Build ML systems / MLOps** | Kubernetes for ML, Ray Serve, KServe, model registries, [Full Stack Deep Learning](https://fullstackdeeplearning.com/) |
| **Build LLM products** | RAG pipelines, agent frameworks, eval harnesses, structured-output systems |
| **Write GPU kernels** | Contribute to vLLM / FlashAttention / xFormers; read CUTLASS; apply to NVIDIA |
| **Do robotics / RL** | OpenAI Spinning Up, Isaac Sim, Sutton & Barto, robotics arms |
| **Work on multimodal** | Diffusion models, ControlNet, VLMs, audio models |

## Read these next

- [Lilian Weng's blog](https://lilianweng.github.io/) — front to back
- [Sebastian Raschka's substack](https://magazine.sebastianraschka.com/) — practical & current
- [Chip Huyen's blog](https://huyenchip.com/blog/) — ML systems & ML in industry
- [Simon Willison's blog](https://simonwillison.net/) — daily LLM ecosystem updates
- [Eugene Yan's blog](https://eugeneyan.com/) — production ML lessons
- [HuggingFace papers digest](https://huggingface.co/papers) — daily, with discussion

## A final note

Hand-rolling everything you've hand-rolled this curriculum is **not** what production looks like. In production you'll mostly use libraries — PyTorch, transformers, vLLM, the rest. **The point was never to make you implement these forever.** The point was to make you a person who reads their source and understands what they're doing.

After 16 weeks at this depth, you're hard to surprise. New models, new frameworks, new optimizations — they'll fit into your mental model instead of feeling like magic. Keep that bar. Keep reading source code. Keep measuring. Keep shipping.

Welcome to the inside of the field.

— end of curriculum —
