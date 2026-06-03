# Week 10: Transformers from Scratch

## 🎯 What you'll learn

The architecture behind every modern LLM, every image-generation model, every speech model. You'll build a decoder-only transformer (GPT-style) from `nn.Linear` primitives — no `nn.MultiheadAttention` shortcuts.

By the end of this week you'll be able to:

- Derive scaled dot-product attention from "we need a way to look up by content"
- Implement multi-head attention, causal masking, position encoding (sinusoidal + RoPE)
- Build a full transformer block (LayerNorm → attention → residual → FFN → residual)
- Train a tiny GPT (~10M params) on Shakespeare and generate text
- Implement a KV cache for inference (this becomes critical in week 16)
- Read the original "Attention Is All You Need" paper and understand every figure

## 🧰 Lab setup

Same as week 07.

## ✅ Your job

1. Read [theory.md](theory.md) — the geometry of attention, why softmax/√d, RoPE intuition
2. Work through [lab.md](lab.md) — build nanoGPT-style from scratch, train on TinyShakespeare
3. Generate text from your trained model; vary temperature and top-k
4. (Stretch) Implement multi-query attention (MQA) for faster inference

## 📚 Required reading

| Resource | Why | Time |
|---|---|---|
| [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) | The paper | 60 min |
| [Karpathy — Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) | The canonical build-along | 2 hours |
| [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | The visual companion | 30 min |
| [RoPE paper (Su et al., 2021)](https://arxiv.org/abs/2104.09864) | Modern position encoding | 30 min |

## 💡 What you should already know

- Weeks 07-08

---

**Next**: [Week 11: Training at Scale →](../week-11-training-at-scale/readme.md)
