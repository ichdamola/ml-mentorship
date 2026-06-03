# Week 12: Lab — LLM Fine-tuning

You'll build a LoRA layer by hand, then use the production tooling (`peft` + `transformers`) to fine-tune a real open-source LLM. The hand-built version teaches the mechanics; the library version is what you'll actually use.

## Setup

```bash
uv add transformers datasets peft bitsandbytes accelerate trl
```

You'll want a GPU with ≥ 8 GB VRAM for the small-model exercises (1.1) and ≥ 16 GB for the full QLoRA exercise (1.6+). A free Colab T4 (16 GB) handles everything.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}, name: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'}")
print(f"free VRAM: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB" if device.type == 'cuda' else "")
```

---

## Exercise 12.1 — Build a LoRA Linear from scratch

```python
class LoRALinear(nn.Module):
    """A frozen Linear with a learnable low-rank update on top."""
    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / r

        # Critical init: A from Gaussian, B from zeros so ΔW = 0 at step 0
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(self.dropout(x)))

# Sanity check: at init, output equals base layer's output exactly
base = nn.Linear(64, 128)
lora = LoRALinear(base, r=16, alpha=32)
x = torch.randn(4, 64)
base_out = base(x)
lora_out = lora(x)
print(f"max abs diff at init: {(base_out - lora_out).abs().max():.2e}")  # ~0

# Count trainable params
total = sum(p.numel() for p in lora.parameters())
trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
print(f"total params: {total:,}")
print(f"trainable params: {trainable:,}   ({100*trainable/total:.1f}%)")
# For base 64→128, full = 8320; LoRA r=16 trainable = 64*16 + 16*128 = 3072
```

You should see **~37% trainable params** for this toy example. For real Llama-2-7B with all-attention LoRA at `r=16`, the ratio is **~0.1%** — that's the dramatic version.

---

## Exercise 12.2 — Inject LoRA into a small model

Take a small model and replace its attention `Linear` layers with LoRA-wrapped versions.

```python
class TinyMLP(nn.Module):
    """A toy "model" for demonstration."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 32)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


def lora_wrap(module: nn.Module, target_names=('fc1', 'fc2'), r=8):
    """In-place: replace named Linear children with LoRALinear."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in target_names:
            setattr(module, name, LoRALinear(child, r=r))
        else:
            lora_wrap(child, target_names, r)

model = TinyMLP()
print("Before LoRA:")
for name, p in model.named_parameters():
    print(f"  {name:30s} requires_grad={p.requires_grad}")

lora_wrap(model)
print("\nAfter LoRA:")
for name, p in model.named_parameters():
    if p.requires_grad:
        print(f"  TRAINABLE: {name:50s} {tuple(p.shape)}")
```

The `fc1` and `fc2` weights are now frozen; only the LoRA `A` and `B` matrices train.

---

## Exercise 12.3 — Train it on a toy regression task

Show that LoRA can recover behavior that the frozen base can't.

```python
torch.manual_seed(42)

# Synthetic task: y = sin(x) + 0.3 (the +0.3 is the "fine-tune target")
def make_data(n=1000):
    x = torch.linspace(-5, 5, n).unsqueeze(-1)
    x32 = x.repeat(1, 32) + torch.randn(n, 32) * 0.01    # 32-dim noise
    y = (torch.sin(x).repeat(1, 32) + 0.3)
    return x32, y

X, y = make_data()
X_train, y_train = X[:800], y[:800]
X_test, y_test = X[800:], y[800:]

# Train base (no LoRA) on sin(x) only
torch.manual_seed(42)
base_model = TinyMLP()
opt = torch.optim.Adam(base_model.parameters(), lr=1e-2)
for step in range(200):
    pred = base_model(X_train)
    loss = F.mse_loss(pred, torch.sin(X_train[:, :1]).repeat(1, 32))
    opt.zero_grad(); loss.backward(); opt.step()
print(f"\nBase model trained on sin(x). Test loss against sin(x) target: "
      f"{F.mse_loss(base_model(X_test), torch.sin(X_test[:, :1]).repeat(1, 32)).item():.4f}")

# Now wrap with LoRA and fine-tune to learn sin(x) + 0.3
lora_wrap(base_model, target_names=('fc1', 'fc2'), r=8)
opt = torch.optim.Adam([p for p in base_model.parameters() if p.requires_grad], lr=1e-2)
for step in range(200):
    pred = base_model(X_train)
    loss = F.mse_loss(pred, y_train)
    opt.zero_grad(); loss.backward(); opt.step()

# Test on shifted target
final_loss = F.mse_loss(base_model(X_test), y_test).item()
print(f"After LoRA fine-tune. Test loss against shifted target: {final_loss:.4f}")

trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in base_model.parameters())
print(f"Trainable: {trainable:,} of {total:,} ({100*trainable/total:.1f}%)")
```

Loss should drop substantially with the LoRA-only training. **You taught the model a new behavior while updating only a fraction of its weights.**

---

## Exercise 12.4 — Hello, real LLMs (a tiny model)

Now switch to actual HuggingFace models. We'll use `Qwen/Qwen2.5-0.5B` because it's small enough to fine-tune on a single GPU without quantization.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
print(f"loaded {model_name}: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

# Quick test: generate from base model
def generate(model, prompt, max_new=50, temp=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_new,
            do_sample=True, temperature=temp, top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

print("\nBase model:")
print(generate(model, "Explain what a neural network is in one sentence: "))
```

---

## Exercise 12.5 — A custom instruction dataset

We'll create a tiny "be a pirate" dataset to teach the model a style.

```python
pirate_instructions = [
    ("What's the weather today?",
     "Arrr! The skies be calm, the seas be steady, and the wind be at our backs!"),
    ("Explain photosynthesis.",
     "Avast! Plants be takin' in the sun's rays and the air's CO2, churnin' 'em into sugar — a fine alchemy, matey!"),
    ("How do I bake bread?",
     "Aye! Mix flour, water, salt, and yeast. Knead it like ye'd haul a sail, let it rise like a king tide, and bake until golden!"),
    ("Tell me about Mars.",
     "Ho there! Mars be the red planet, fourth from the sun, with two tiny moons and a great big mountain called Olympus!"),
    ("What is 2 plus 2?",
     "Arrr, that be 4, plain as the nose on yer face!"),
    ("Recommend a book.",
     "Aye, read 'Treasure Island' by Stevenson — there be no finer tale of adventure on the high seas!"),
    ("How do I learn to code?",
     "Avast! Start with the basics — pick a language like Python, build small projects, and don't ye fear the bugs! They be teachers in disguise."),
    ("What's the meaning of life?",
     "Arrr, that be a question for the philosophers, matey! But I say — find yer treasure and share it with yer crew!"),
] * 6   # repeat for more training signal — in real life you'd have 1000s of unique examples

# Format with the model's chat template (Qwen uses ChatML)
def format_example(user, assistant):
    messages = [
        {"role": "system", "content": "You are a friendly pirate assistant."},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

formatted = [format_example(u, a) for u, a in pirate_instructions]
print(formatted[0])
```

---

## Exercise 12.6 — Fine-tune with `peft` LoRA

```python
from peft import LoraConfig, get_peft_model

# Reload model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

lora_config = LoraConfig(
    r=8,                                    # small model + small dataset, low rank is fine
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# ~0.5% trainable

# Tokenize the dataset
encoded = tokenizer(formatted, padding=True, truncation=True, max_length=256, return_tensors="pt")

# Simple training loop (in production you'd use Trainer / SFTTrainer)
opt = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4, weight_decay=0.0,
)
model.train()
batch_size = 8
input_ids = encoded["input_ids"].to(device)
attention_mask = encoded["attention_mask"].to(device)

losses = []
for epoch in range(5):
    epoch_losses = []
    perm = torch.randperm(len(input_ids))
    for i in range(0, len(perm), batch_size):
        idx = perm[i:i+batch_size]
        batch_ids = input_ids[idx]
        batch_mask = attention_mask[idx]

        outputs = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_ids)
        loss = outputs.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        epoch_losses.append(loss.item())
    avg = sum(epoch_losses) / len(epoch_losses)
    losses.append(avg)
    print(f"epoch {epoch+1}: avg loss {avg:.4f}")

# Plot
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel("epoch"); plt.ylabel("loss")
plt.title("LoRA fine-tuning loss")
plt.show()
```

For this toy dataset 5 epochs is enough to see strong pirate behavior emerge.

---

## Exercise 12.7 — Compare base vs fine-tuned

```python
# Save fine-tuned LoRA weights
model.save_pretrained("./pirate_lora")

# Generate from fine-tuned
test_prompts = [
    "What's the best way to learn a language?",
    "Tell me about black holes.",
    "What did you do today?",
]

print("=== FINE-TUNED OUTPUTS ===\n")
for prompt in test_prompts:
    messages = [
        {"role": "system", "content": "You are a friendly pirate assistant."},
        {"role": "user", "content": prompt},
    ]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = generate(model, chat, max_new=60)
    print(f"Q: {prompt}")
    print(f"A: {out.split('assistant')[-1][:300]}")
    print()

# Load base model for comparison
base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
base.eval()

print("=== BASE OUTPUTS ===\n")
for prompt in test_prompts:
    messages = [
        {"role": "system", "content": "You are a friendly pirate assistant."},
        {"role": "user", "content": prompt},
    ]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = generate(base, chat, max_new=60)
    print(f"Q: {prompt}")
    print(f"A: {out.split('assistant')[-1][:300]}")
    print()
```

The fine-tuned outputs should be noticeably pirate-y. The base (with same system prompt) tries but won't be as consistent. **You've taught the model a style with 48 examples and 0.5% of its parameters.**

---

## Exercise 12.8 — Tiny eval harness

```python
def lora_judge(prompt, response):
    """Score 0-5 for 'how pirate-y is this response?'.
    In real life you'd use GPT-4 or Claude as the judge — here we use keywords."""
    pirate_words = ["arrr", "ahoy", "avast", "matey", "ye", "aye", "arr", "aye-aye"]
    response_lower = response.lower()
    score = sum(2 for w in pirate_words if w in response_lower)
    score = min(5, score)   # cap
    return score

# Run judge on held-out prompts
held_out = [
    "Describe a sunset.",
    "What's your favorite food?",
    "How does WiFi work?",
    "Tell me a joke.",
    "What is the capital of France?",
]

print(f"{'prompt':50s}  {'base':>4s}  {'lora':>4s}")
for prompt in held_out:
    messages = [
        {"role": "system", "content": "You are a friendly pirate assistant."},
        {"role": "user", "content": prompt},
    ]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    base_out = generate(base, chat, max_new=60).split("assistant")[-1]
    lora_out = generate(model, chat, max_new=60).split("assistant")[-1]
    print(f"{prompt[:50]:50s}  {lora_judge(prompt, base_out):>4d}  {lora_judge(prompt, lora_out):>4d}")
```

The fine-tuned model should score higher on average. **This is a tiny version of the LLM-as-judge eval pattern used in production** — replace the keyword scorer with a real LLM call to GPT-4 / Claude / Gemini.

---

## Exercise 12.9 (stretch) — QLoRA on a 7B model

Requires ≥ 12-16 GB VRAM. Skip if you only have CPU or small GPU.

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Use any 7B-class base. "Qwen/Qwen2.5-7B" is good. Llama-2-7b requires gated access.
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    quantization_config=bnb_config,
    device_map="auto",
)
base = prepare_model_for_kbit_training(base)

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
qlora_model = get_peft_model(base, lora_config)
qlora_model.print_trainable_parameters()
# trainable: ~0.1% of 7B = ~10M params

print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
# A 7B model in NF4 + bf16 compute typically uses ~5-7 GB
```

You can now train this with the same SFTTrainer / Trainer loop. Single-GPU 7B fine-tuning, with ~10M trainable params. **This is what every open-source community fine-tune (Vicuna, WizardLM, Hermes, OpenChat) is built on.**

---

## Exercise 12.10 — Catastrophic forgetting check

Test whether your pirate fine-tune forgot how to do math.

```python
math_prompts = [
    "What is 7 times 8?",
    "Compute 100 minus 47.",
    "What is the square root of 144?",
]

for prompt in math_prompts:
    messages = [
        {"role": "system", "content": "You are a friendly pirate assistant."},
        {"role": "user", "content": prompt},
    ]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Q: {prompt}")
    base_out = generate(base, chat, max_new=40).split("assistant")[-1][:200]
    print(f"  base:   {base_out}")
    lora_out = generate(model, chat, max_new=40).split("assistant")[-1][:200]
    print(f"  pirate: {lora_out}")
    print()
```

If your fine-tuned model is still mostly correct (just with arrr-flavor), forgetting is mild. If it stops being able to do math, you've over-fit — try fewer epochs or lower LR.

---

## Submission checklist

- [ ] `LoRALinear` matches base output at init (max abs diff < 1e-5)
- [ ] Trainable parameter fraction calculated correctly
- [ ] Toy fine-tune teaches a new behavior with frozen base
- [ ] Loaded a real HuggingFace model (Qwen-0.5B or similar)
- [ ] Fine-tuned with `peft` LoRA on a small custom dataset
- [ ] Side-by-side base vs fine-tuned comparison shows the desired behavior emerged
- [ ] Tiny LLM-as-judge eval (keyword version is fine) shows score improvement
- [ ] (Stretch) Loaded a 7B model in 4-bit and ran `prepare_model_for_kbit_training`
- [ ] Catastrophic forgetting check — model still does math reasonably

---

## What you just did

You built LoRA from scratch, then used the production tooling to fine-tune a real LLM on a custom instruction dataset. You compared outputs, scored them with an eval rubric, and checked for forgetting. **That's the full modern fine-tuning workflow.**

The remaining four weeks (13-16) get under the hood: GPU architecture, CUDA kernels, profiling, and the production inference engines (vLLM, TensorRT-LLM) that serve all the fine-tuned models people build with what you just learned.

---

**Next**: [Week 13: GPU Architecture →](../week-13-gpu-architecture/readme.md)
