# Week 16: Production Inference — Capstone

## 🎯 What you'll learn

The final mile: turn the model you've been training into a production-serving system. Throughput, latency, cost-per-token, deployment.

By the end of this week you'll be able to:

- Explain continuous batching, KV cache management, and PagedAttention
- Stand up a vLLM server and serve a 7B model with reasonable throughput
- Quantize a model to INT8 / INT4 with `bitsandbytes` or `AutoGPTQ` or `AWQ`
- Export an ONNX model and run it through TensorRT for max NVIDIA perf
- Reason about prefill vs decode latency, time-to-first-token vs tokens/sec
- Pick a serving stack — vLLM, TGI, TensorRT-LLM, Triton Inference Server — for a use case
- Build a tiny LLM serving API in front of your inference engine
- Cost-estimate a production deployment (GPU-hours per million tokens)

## 🧰 Lab setup

```bash
# vLLM
uv add vllm
# Or TensorRT-LLM (NVIDIA hardware required, more setup)
# https://github.com/NVIDIA/TensorRT-LLM
```

You'll want a GPU with ≥24GB VRAM for the 7B serving lab. T4 (16GB) works for quantized 7B.

## ✅ Your job

1. Read [theory.md](theory.md) — prefill vs decode, KV cache memory math, PagedAttention
2. Work through [lab.md](lab.md) — stand up vLLM, benchmark throughput, quantize, redeploy
3. Build a small FastAPI wrapper that serves your model with token streaming
4. Measure: requests/sec, tokens/sec, p50/p99 latency, GPU memory headroom
5. **Capstone deliverable:** a deployed (even if on Modal / Replicate / RunPod) LLM endpoint you can demo, with a README showing the throughput numbers and a one-page architecture write-up

## 📚 Required reading

| Resource | Why | Time |
|---|---|---|
| [vLLM paper — Efficient memory management with PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180) | The most influential serving paper of 2023 | 45 min |
| [vLLM docs](https://docs.vllm.ai/) | The most-used open serving stack | 60 min |
| [TensorRT-LLM docs](https://nvidia.github.io/TensorRT-LLM/) | NVIDIA's official LLM serving | 45 min skim |
| [HuggingFace TGI docs](https://huggingface.co/docs/text-generation-inference/index) | The other major open serving option | 30 min |
| [Anyscale — Continuous batching](https://www.anyscale.com/blog/continuous-batching-llm-inference) | The canonical explainer | 30 min |
| [INT4 quantization — GPTQ (Frantar et al.)](https://arxiv.org/abs/2210.17323) and [AWQ (Lin et al.)](https://arxiv.org/abs/2306.00978) | Modern weight-only quantization | 60 min |

## 💡 What you should already know

- All prior weeks. This is the synthesis.

## 🦀 Sidebar: Rust in modern ML infra

By the time you finish this week, you'll have noticed Rust quietly running underneath a lot of what you used:

| Library | What it does | Why Rust |
|---|---|---|
| **`tokenizers`** (HuggingFace) | The tokenizer behind every `transformers` model | Per-call overhead matters at training and serving scale; Python loops were the bottleneck |
| **`safetensors`** | The modern checkpoint format replacing `.pt` / `.bin` | Memory-mapped, zero-copy, no `pickle` RCE risk ([appsec week 11](https://github.com/ichdamola/appsec-mentorship/blob/main/week-11-insecure-deserialization/attack.md)) |
| **`candle`** (HuggingFace) | A Rust deep-learning framework | Lean inference runtime; embeds in Rust services without Python |
| **`polars`** | Pandas alternative | Multi-threaded by default; eats pandas for breakfast on large frames |
| **`burn`** | Another Rust DL framework | Multi-backend (CUDA, Metal, WGPU) |

When does it pay off to learn Rust for ML?

- You're rewriting a Python-bound hot loop (a custom tokenizer, a sampling strategy, an environment in RL) and Python's overhead is the bottleneck
- You're shipping inference inside a non-Python service (a Rust web backend, an edge binary)
- You want safetensors / candle / polars to "just work" and not feel like a black box

**When to skip it:** if your bottleneck is GPU kernels (covered in [Week 14](../week-14-cuda-programming/)) or model architecture choices, Rust doesn't help — those are C++/CUDA problems. Rust supplements CUDA C++, it doesn't replace it.

After this curriculum, the lightest "second language" addition is: spend a weekend with the Rust book ch. 1-13, then port one of your `tokenizers`-adjacent scripts to Rust + PyO3.

## 🎓 Where to go next

You've finished a 16-week ML curriculum that started at vectors and ended at production inference. What comes next depends on your direction:

| Direction | What to add |
|---|---|
| **ML research** | Read [Lilian Weng's blog](https://lilianweng.github.io/), the latest NeurIPS/ICLR proceedings; reproduce one new paper per month |
| **ML engineering / MLOps** | Kubernetes + Ray + KServe; FullStackDeepLearning course |
| **LLM applications** | Build retrieval-augmented (RAG) and agentic systems; LangChain / LlamaIndex caveats |
| **CUDA / kernel work** | More Triton; contribute to FlashAttention / xFormers; read the CUTLASS docs |
| **NVIDIA / hardware** | Apply to NVIDIA's residency programs; read their CUDA samples repo end-to-end |
| **Robotics / vision** | Take CS231n, then a robotics course; Isaac Sim |

---

**Curriculum complete** — you've gone from vectors to vLLM.
