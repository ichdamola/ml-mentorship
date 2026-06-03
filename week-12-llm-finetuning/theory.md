# Week 12: Theory — LLM Fine-tuning

99% of real-world "LLM work" is not pretraining a model from scratch — it's adapting an existing open-source base model to a specific task or style. This week covers the modern techniques that make this affordable: **LoRA, QLoRA, PEFT**, the right instruction format, and evaluation that doesn't lie to you.

By the end of this week you'll have fine-tuned a 7B parameter model on a consumer GPU. That capability — present-day routine — was world-class research three years ago.

---

## Part 1: Pretrain vs fine-tune vs prompt

Three ways to make an LLM do what you want:

| Approach | Cost | When |
|---|---|---|
| **Prompt engineering** | $0 in training; ongoing inference cost | The behavior is achievable with the right prompt. Always try first. |
| **Fine-tuning** | One-time training cost | The base model has the capability but needs domain-specific style/format/safety tuning. |
| **Pretraining from scratch** | $1M-$100M+ | You need a fundamentally different model (different language, modality, domain). |

The honest rule of thumb (2026): **try prompting first; fine-tune only if you can't reach quality with the right prompt + few-shot examples.** Modern base models are surprisingly competent at obeying detailed prompts.

When fine-tuning helps:
- **Format control** — "always output JSON matching this schema" reliably
- **Style/voice** — sound like your brand, your customer service tone, etc.
- **Domain knowledge** — medical, legal terminology + reasoning patterns
- **Safety** — refuse certain categories; respond a specific way to certain inputs
- **Cost reduction** — a fine-tuned 7B model can match a prompted 70B at 10× lower inference cost

---

## Part 2: Why full fine-tuning is expensive

Fine-tuning Llama-2-7B fully (every parameter updates) needs:

- **Params (bf16):** 14 GB
- **Gradients (bf16):** 14 GB
- **Optimizer state (fp32 master + Adam m, v):** 84 GB
- **Activations:** 10-40 GB depending on seq length and batch

Total: ~130-170 GB. **You need a multi-A100 setup or H100.**

Plus you produce a 14 GB checkpoint per task. If you want to fine-tune for 100 tasks, that's 1.4 TB of model storage.

This is why **parameter-efficient fine-tuning (PEFT)** is the universal practice for sub-frontier labs.

---

## Part 3: LoRA — the core insight

**Hypothesis** (Hu et al., 2021): when you fine-tune a model, the actual *change* to each weight matrix is **low rank**.

That is, if `W₀` is the pretrained weight matrix and `W_finetuned = W₀ + ΔW`, the update `ΔW` (despite being the same shape as W) can be approximated by the product of two much smaller matrices:

```
ΔW = B A    where A ∈ ℝ^(r × d_in), B ∈ ℝ^(d_out × r)
```

`r` is the **rank** — typically 8, 16, 32, or 64. For a 4096×4096 weight matrix and `r=16`:

- Full ΔW: 16.7M parameters
- LoRA (B, A): 4096×16 + 16×4096 = 131K parameters

**127× fewer parameters to train, store, and load.**

### How it's wired into the model

Each target Linear layer is replaced by:

```python
class LoRALinear(nn.Module):
    def __init__(self, W_pretrained, r=16, alpha=32):
        super().__init__()
        self.W = W_pretrained                 # frozen
        for p in self.W.parameters():
            p.requires_grad = False

        self.A = nn.Linear(self.W.in_features, r, bias=False)
        self.B = nn.Linear(r, self.W.out_features, bias=False)
        self.scale = alpha / r

        # Initialize A from Gaussian, B from zeros — so initial ΔW = 0
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.W(x) + self.scale * self.B(self.A(x))
```

Critical detail: **`B` initialized to zeros** means the LoRA contribution starts at exactly zero. The model behaves identically to the base at step 0, and gradually learns to deviate.

### What `alpha` and `scale` do

`alpha` is a learning-rate-like scaling: bigger alpha = bigger effective LR for LoRA params. The convention is `alpha = 2 × r` and effective LR is what you set normally. Don't overthink it — keep `alpha = 32` for `r = 16` and tune LR if needed.

### Which layers to LoRA-fy

Common targets in a transformer:

| Target | What it does |
|---|---|
| `q_proj, k_proj, v_proj, o_proj` | Attention QKV+output projections. **The minimum-viable target.** |
| `gate_proj, up_proj, down_proj` | FFN layers. Adds more params but more capacity. |
| All Linear layers | Maximum adaptation, ~3× the LoRA params of attention-only |

The original LoRA paper (Hu et al.) only used Q and V. Modern practice (QLoRA paper, [Dettmers et al. 2023](https://arxiv.org/abs/2305.14314)) targets **all linear layers** as the default and gets noticeably better results for ~3× the LoRA params.

---

## Part 4: QLoRA — fine-tuning a 7B on a consumer GPU

**LoRA** trains low-rank adapters on top of frozen base weights.
**QLoRA** also **quantizes the base weights to 4-bit** during training, so even the base model fits in much less memory.

For Llama-2-7B fine-tuning:

| Method | Base weights | Adapter | Total VRAM |
|---|---|---|---|
| Full fine-tune | 14 GB (bf16) | n/a | ~140 GB |
| LoRA | 14 GB (bf16) | 0.06 GB | ~30 GB |
| **QLoRA** | **3.5 GB (4-bit)** | 0.06 GB | **~10 GB** ✓ T4 / RTX 3090 fit! |

The base model is loaded in 4-bit NF4 (NormalFloat4, optimized for normally-distributed weights), kept frozen, and used in forward as if it were bf16 (dequantized on-the-fly). The LoRA adapters stay in bf16 because they update.

This is what makes consumer-hardware fine-tuning of 7B-70B models possible. Almost every open-source fine-tune you've heard of (Vicuna, WizardLM, OpenChat, the Hermes series) was QLoRA.

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4 — designed for normally-distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# trainable params: ~0.1% of all 7B params
```

That last line is the magic. You're training 0.1% of the parameters and capturing most of the adaptation benefit.

---

## Part 5: Instruction datasets — formats that work

A fine-tuning dataset is a collection of `(instruction, response)` pairs. Format matters.

### Alpaca format (Stanford, 2023)

```json
{
  "instruction": "Translate this to French.",
  "input": "Hello, how are you?",
  "output": "Bonjour, comment allez-vous?"
}
```

Becomes during training:

```
Below is an instruction that describes a task, paired with an input.
Write a response that appropriately completes the request.

### Instruction:
Translate this to French.

### Input:
Hello, how are you?

### Response:
Bonjour, comment allez-vous?
```

### ChatML (OpenAI, used by Mistral / Qwen / many)

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Translate this to French: Hello, how are you?<|im_end|>
<|im_start|>assistant
Bonjour, comment allez-vous?<|im_end|>
```

### Llama-2/3 chat format

```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

Translate this to French: Hello, how are you? [/INST] Bonjour, comment allez-vous? </s>
```

**Whatever format the base model was originally trained on, use that for fine-tuning.** Mixing chat formats is the #1 cause of "my fine-tuned model produces garbage." Each `transformers` tokenizer ships with a `chat_template` you can apply:

```python
tokenizer.apply_chat_template(messages, tokenize=False)
```

### Loss masking — only train on the response

You only want the model to learn the **response**, not to regenerate the instruction template. During training, mask the loss on input tokens:

```python
# After tokenizing the full conversation
input_ids = tokenizer.encode(full_conversation)
labels = input_ids.copy()

# Find where the assistant response starts; mask everything before
assistant_start = find_assistant_response_start(full_conversation)
labels[:assistant_start] = -100         # CrossEntropyLoss ignores -100
```

The `trl.SFTTrainer` and `transformers.DataCollatorForCompletionOnlyLM` handle this automatically.

---

## Part 6: SFT, DPO, and beyond

**SFT (Supervised Fine-Tuning)** — what we've been describing. Show examples of (input, desired output); minimize cross-entropy.

**DPO (Direct Preference Optimization)** — show pairs of (good response, bad response) and train the model to prefer the good one. Simpler than RLHF, often nearly as effective.

```
Loss_DPO = -log σ(β · (log π_θ(good|x) - log π_ref(good|x)
                       - log π_θ(bad|x) + log π_ref(bad|x)))
```

Where `π_ref` is a frozen reference model (often the SFT model from the previous stage) and `β` is a temperature hyperparameter.

The full alignment recipe used by frontier labs:

```
base model → SFT → DPO → safety fine-tuning → final model
```

For most production fine-tuning, **SFT alone is enough**. DPO is worth adding when you have preference data (which is harder to collect than instruction data) and quality matters intensely.

---

## Part 7: Evaluation — the part everyone gets wrong

The temptation: "train loss went down, deploy!"

The reality:

| Eval | What it tells you |
|---|---|
| Training loss | Almost nothing about quality |
| Validation loss | Slightly more, but still a proxy |
| Held-out task accuracy | Useful if you have a quantitative task |
| Human eval | The ground truth, but expensive |
| LLM-as-judge | A cheap proxy that often agrees with humans |
| Public benchmarks (MMLU, HumanEval) | Bad for fine-tunes — they measure general capabilities, not your task |

### Recommended eval setup for a fine-tune

1. **Held-out instruction set** — 100-500 examples your model didn't train on
2. **Side-by-side comparison** — base model vs fine-tuned, same input
3. **LLM-as-judge** — GPT-4 or Claude scoring each output on a rubric you define
4. **Manual eyeballing** — read 20-50 examples yourself. You'll notice things automated eval misses (style drift, formatting bugs, hallucinations).

### lm-eval-harness

EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) standardizes benchmark evaluation. You'll use it in week 12's lab to compare base vs fine-tuned on a quantitative task.

**Important caveat:** standard benchmarks (MMLU, BoolQ, HellaSwag) measure *general capabilities*. They're for comparing base models. **Your fine-tune is judged on your task, not on benchmark deltas.**

---

## Part 8: Catastrophic forgetting

Fine-tuning teaches new behavior. It can also **un-teach** existing capabilities — the model gets better at your domain and worse at math, coding, common-sense reasoning.

Mitigations:

| Mitigation | What it does |
|---|---|
| Fewer training steps | Train less, forget less |
| Lower learning rate | Smaller updates per step → less drift |
| Mix instruction data with general data | Include some "keep being a smart assistant" examples |
| LoRA (vs full fine-tune) | Frozen base means less forgetting |
| Lower LoRA rank | Less expressive adapter, less can be changed |

The QLoRA paper shows: with `r=16` and a moderate-size domain dataset (10k-100k examples), forgetting is mild. Going to `r=128` and 1M examples will erase general capabilities meaningfully.

---

## Part 9: When fine-tuning isn't the right move

Fine-tuning is a hammer; not everything is a nail.

| Problem | Better tool |
|---|---|
| Need factual recall (recent events, internal docs) | **RAG** (Retrieval-Augmented Generation) |
| Need different reasoning style | Stronger model OR chain-of-thought prompting |
| Need a JSON-strict output | Constrained decoding (`outlines`, `lm-format-enforcer`) |
| Need a tool-using agent | Function calling + structured tool definitions |
| Need access to private data only | RAG > fine-tune for most cases |
| Need the model to know your codebase | RAG with code embeddings; only fine-tune if the *style* needs changing |

**RAG vs fine-tuning** is the most-asked question. The short answer:

- **Fine-tune** for *behavior* changes (style, format, refusals)
- **RAG** for *knowledge* changes (private data, recent events, large reference text)

Often you want both: a fine-tuned model that knows how to use a retrieval system to answer from your data.

---

## Part 10: A peek at the full fine-tuning script

```python
# train.py — what a real QLoRA SFT script looks like
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config, device_map="auto",
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("yahma/alpaca-cleaned", split="train")

def format(ex):
    return f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}"

dataset = dataset.map(lambda ex: {"text": format(ex)})

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,    # effective batch 16
        learning_rate=2e-4,
        num_train_epochs=1,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
        bf16=True,
        gradient_checkpointing=True,
        output_dir="./out",
    ),
)
trainer.train()
trainer.model.save_pretrained("./out/lora_final")
```

That's ~50 lines and fine-tunes a 7B model. **Six years ago this would have been a research paper.** Now it's the standard recipe.

---

## What's next

In [lab.md](lab.md) you'll:
- Build a tiny LoRA `Linear` layer from scratch — see exactly what `peft` does
- Fine-tune a small base model (e.g. `Qwen/Qwen2.5-0.5B`) with LoRA on a custom instruction dataset
- Compare base vs fine-tuned outputs side by side
- Build a tiny LLM-as-judge eval script
- (If you have ≥16 GB VRAM) full QLoRA on a 7B model with `peft` + `bitsandbytes`

By end of week 12 you'll be able to take any open base model and adapt it to your task — the single most common production LLM skill in 2026.
