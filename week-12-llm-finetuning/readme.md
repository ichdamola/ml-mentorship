# Week 12: LLM Fine-tuning

## 🎯 What you'll learn

Most real LLM work isn't pre-training — it's adapting an open-source base model to a specific task. LoRA, QLoRA, and PEFT make this affordable on consumer hardware.

By the end of this week you'll be able to:

- Explain why full-finetuning is expensive and how LoRA cuts it ~100×
- Apply LoRA + QLoRA to a HuggingFace model
- Build an instruction dataset in the right format
- Evaluate with a held-out test set + at least one rubric beyond loss
- Use `peft`, `transformers.Trainer`, and `bitsandbytes` correctly
- Reason about what kinds of behavior you can and can't change via fine-tuning

## 🧰 Lab setup

```bash
uv add transformers datasets peft bitsandbytes accelerate
```

GPU with ≥16 GB VRAM strongly recommended (T4 in Colab works for LoRA on 7B models).

## ✅ Your job

1. Read [theory.md](theory.md) — the LoRA decomposition, what QLoRA actually quantizes
2. Work through [lab.md](lab.md) — LoRA fine-tune a 7B base model on a small instruction dataset
3. Build a simple eval harness — compare base vs fine-tuned on held-out examples
4. (Stretch) Try DPO (Direct Preference Optimization) instead of supervised fine-tuning

## 📚 Required reading

| Resource | Why | Time |
|---|---|---|
| [LoRA paper (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) | The original | 30 min |
| [QLoRA paper (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314) | 4-bit base model + LoRA = consumer-GPU fine-tuning | 30 min |
| [HuggingFace PEFT docs](https://huggingface.co/docs/peft/index) | Modern interface | 30 min |
| [Eleuther — lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) | The standard eval framework | 20 min skim |

## 💡 What you should already know

- Week 10 (you understand what a transformer is); ideally week 11 too

---

**Next**: [Week 13: GPU Architecture →](../week-13-gpu-architecture/readme.md)
