# Week 11: Training at Scale

## 🎯 What you'll learn

The techniques that turn "I have a model" into "I can train it on as much data as I can afford." Mixed precision, gradient accumulation, distributed training — the senior toolbox.

By the end of this week you'll be able to:

- Use `torch.cuda.amp` for mixed-precision training (fp16/bf16), understand loss scaling
- Use gradient accumulation to train with effective batch sizes larger than memory allows
- Pick optimizers and schedulers for your scale (AdamW, Lion, cosine + warmup)
- Run `torch.compile` and read the compilation errors
- Understand DDP (Distributed Data Parallel) — what each rank does
- Understand FSDP / DeepSpeed — sharding optimizer state and gradients
- Read the Chinchilla scaling laws and pick a model/data budget

## 🧰 Lab setup

Two-GPU machine helpful (a single 24GB GPU works for the AMP + accumulation parts).

```bash
uv add accelerate deepspeed
```

## ✅ Your job

1. Read [theory.md](theory.md) — the memory math for fp16 vs bf16, AdamW state, KV cache vs activations
2. Work through [lab.md](lab.md) — train your transformer from week 10 with AMP + gradient accumulation; measure speedup and memory
3. Run a 2-GPU DDP training if hardware permits
4. (Stretch) Use FSDP to train a model that doesn't fit on one GPU

## 📚 Required reading

| Resource | Why | Time |
|---|---|---|
| [Chinchilla scaling laws (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556) | The model/data trade-off | 30 min |
| [PyTorch DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) | The official walkthrough | 45 min |
| [HuggingFace — accelerate docs](https://huggingface.co/docs/accelerate/index) | Boilerplate-free distributed training | 30 min |
| [ZeRO paper (Rajbhandari et al., 2019)](https://arxiv.org/abs/1910.02054) | The memory-sharding idea behind FSDP/DeepSpeed | 30 min |

## 💡 What you should already know

- Week 10

---

> 🚧 **Scaffolded.** Full theory + lab content lands in the next pass.

**Next**: [Week 12: LLM Fine-tuning →](../week-12-llm-finetuning/readme.md)
