# Week 09: Theory — Convolutional Neural Networks

You spent weeks 05-08 on MLPs. They classify MNIST at 98% and CIFAR-10 at about 50%. The 48-point gap is the architecture's fault — MLPs throw away every piece of structure that makes an image an image.

CNNs are the architecture that **builds image structure into the model**. By the end of this week you'll have built ResNets that hit >90% on CIFAR-10, and you'll understand why every modern vision system — even those topped with a transformer — still has convolution somewhere in the pipeline.

---

## Part 1: Why MLPs fail on images

A 32×32 RGB image is `(3, 32, 32)` = 3,072 numbers. An MLP's first layer must be `nn.Linear(3072, hidden)`, which:

1. **Destroys spatial information.** Pixel (5, 7) and pixel (5, 8) are next to each other. The MLP doesn't know that — they're just inputs 167 and 168.
2. **Is translation-naïve.** A digit shifted three pixels right is, to an MLP, an entirely new image — different pixels are active. You'd need a training example for every shift, every scale, every rotation.
3. **Has too many parameters.** A `Linear(3072, 1024)` is **3.1M parameters**. CIFAR-10 has 50k training images. You'll overfit before you learn anything.

CNNs encode three **inductive biases** to fix all three:

1. **Locality** — early features depend only on a small spatial neighborhood
2. **Translation equivariance** — the same filter slides everywhere, so "edge detector" works at any position
3. **Hierarchical composition** — low-level features (edges) combine into mid-level (textures) into high-level (objects)

These aren't optional design choices — they're priors about the world baked into the architecture. CNNs *can't* learn certain wrong things, which is why they generalize better.

---

## Part 2: The convolution operation

A 2D convolution slides a small kernel across an input. For input `X` (one channel) and kernel `K` of size `k × k`:

```
Y[i, j] = sum over a, b of X[i+a, j+b] * K[a, b]
```

Or, more honestly, **cross-correlation** — true math-textbook convolution flips the kernel, but every DL library calls it convolution and skips the flip. Don't fight the vocabulary.

### Worked example

Input (4×4):

```
1  2  3  0
0  1  2  3
3  0  1  2
2  3  0  1
```

Kernel (2×2):

```
1  0
0 -1
```

Output `Y[0,0]` is the sum of `X[0:2, 0:2] ⊙ K`:

```
(1)(1) + (2)(0) + (0)(0) + (1)(-1) = 0
```

Slide to the right, repeat. Output shape is `(input_h - k + 1, input_w - k + 1) = (3, 3)`. **This is what `nn.Conv2d` does.**

### Multi-channel convolution

Real images have C channels (3 for RGB). Each conv filter has shape `(C, k, k)` — it weights each input channel:

```
Y[i, j] = sum over c, a, b of X[c, i+a, j+b] * K[c, a, b]
```

You typically apply `out_channels` independent filters, producing an output of shape `(out_channels, H_out, W_out)`.

```python
import torch.nn as nn
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
print(conv.weight.shape)   # (64, 3, 3, 3) — out_ch × in_ch × kH × kW
print(conv.bias.shape)     # (64,)
```

`Conv2d(3, 64, kernel_size=3)` has `64 × 3 × 9 = 1,728` weights plus 64 biases = **1,792 parameters total**. Compare to MLP's 3M for a similar transformation. **That's the parameter-sharing efficiency: one filter is reused at every spatial location.**

### Padding and stride

| Knob | Effect |
|---|---|
| **Padding** `p` | Adds `p` rows/columns of zeros around the input. `padding=1, kernel=3` preserves spatial size. |
| **Stride** `s` | Step the kernel by `s` pixels at a time. `stride=2` halves the output size. |
| **Dilation** `d` | Skip `d-1` pixels inside the kernel — increases receptive field without more params. |

The output spatial size for a 2D conv:

```
H_out = floor((H_in + 2p - d(k-1) - 1) / s) + 1
```

Memorize the special cases:
- `kernel=3, padding=1, stride=1` → output same size as input (the workhorse)
- `kernel=1, padding=0, stride=1` → output same size (pointwise / channel mixing)
- `kernel=3, padding=1, stride=2` → output **halved** in each spatial dim (downsampling)

---

## Part 3: Pooling

Pooling reduces spatial size without learning. Two flavors:

| Op | What it does |
|---|---|
| **Max pool** `k×k` | Output `(i,j)` is the max of input `[i:i+k, j:j+k]`. Keeps the strongest activation. |
| **Average pool** `k×k` | Mean of the same window. Smoother. |

Modern CNNs increasingly skip pooling in favor of **strided convolutions** (a conv with stride=2 does the same downsampling but learns the right way to do it). ResNet uses both; recent architectures (EfficientNet, ConvNeXt) prefer strided convs throughout.

**Global average pooling** at the very end of a CNN — `nn.AdaptiveAvgPool2d(1)` — collapses each channel to a single number, regardless of input size. That's why ResNet works at any input resolution.

---

## Part 4: Receptive field — what each output sees

A neuron in layer L is influenced by some spatial region of the input. That region is the **receptive field**.

For a stack of `3×3` convolutions with stride 1 and no pooling:

| Layer | Kernel | Receptive field on input |
|---|---|---|
| 1 | 3×3 | 3×3 |
| 2 | 3×3 | 5×5 |
| 3 | 3×3 | 7×7 |
| 4 | 3×3 | 9×9 |
| ... | ... | ... |
| N | 3×3 | (2N+1)×(2N+1) |

Each successive layer adds `(k-1) × stride` to the receptive field — pooling and strided convs multiply by their stride.

**Rule of thumb:** the receptive field of the last conv layer should cover roughly the size of objects in your input. For 224×224 ImageNet images, ResNet-50's final conv has a receptive field of ~250×250 — bigger than the image, which is fine.

If you stack `3×3` convs `N` times, receptive field grows linearly: `2N+1`. If you replace each with a `5×5`, you'd get `4N+1` — but a `5×5` has `25/9 ≈ 2.8×` more parameters per filter. **VGG showed two 3×3 convs > one 5×5: same receptive field, fewer parameters, more nonlinearities.** This is why nobody uses 5×5 or 7×7 kernels much anymore.

---

## Part 5: Batch normalization

Activations in deep networks drift in distribution as you train. Early in training, layer 5 sees inputs centered around `μ=0, σ=1`. After a few epochs, the input distribution has shifted — layer 5 must adapt. This **internal covariate shift** slows training.

Batch norm fixes it by **normalizing each channel's activations across the batch + spatial dimensions**:

```
For each channel c:
    μ_c = mean of X[:, c, :, :] over batch and spatial dims
    σ_c² = var of X[:, c, :, :] over batch and spatial dims
    X_normalized[:, c, :, :] = (X[:, c, :, :] - μ_c) / sqrt(σ_c² + ε)
    Y[:, c, :, :] = γ_c * X_normalized[:, c, :, :] + β_c
```

`γ_c` and `β_c` are learnable scale and shift parameters. With them, the network can "un-normalize" if it wants — but in practice, the normalized representation is what trains best.

### Train vs eval — the BN gotcha

During training, BN uses **batch statistics** (μ from the current minibatch). During eval, that's unstable for small batches and meaningless for batch-of-1 inference. So BN tracks a **running average** of μ and σ² across training, and uses those at eval time.

```python
# Forget this and your eval accuracy will be mysteriously broken
model.eval()    # switches BN to running statistics
```

### Alternatives

| Layer | When |
|---|---|
| `BatchNorm2d` | CNNs with reasonable batch sizes (≥16) |
| `LayerNorm` | Transformers (week 10) — normalizes per-example |
| `GroupNorm` | When batch size is tiny (1-4); divides channels into groups |
| `InstanceNorm` | Style transfer; per-example per-channel |

For modern CNN work BN is still the default. ConvNeXt (2022) uses LayerNorm to be more transformer-like.

---

## Part 6: Why deeper used to be impossible — and how ResNets fixed it

In 2014 VGG showed 19 layers. People tried 30, 50, 100 layers. **They didn't train better — they trained worse.** Training loss got *higher* with more layers. This wasn't overfitting (test loss tracked train loss) — the optimizer just couldn't find good minima.

The diagnosis: at depth, gradient signal degrades. Even with He init and ReLU, very deep nets suffer from the **degradation problem** — the deeper layers can't learn an identity function from random init.

### The residual block (ResNet, He et al. 2015)

```
out = F(x) + x       ← x flows through "skip"
```

The block computes `F(x) + x` instead of `F(x)`. If learning the identity is best, the block can drive `F → 0` and the skip carries the input through unchanged.

```
   x ─────────────────────┐
   │                       │  (identity / skip)
   ▼                       │
  Conv 3×3                 │
  BatchNorm                │
  ReLU                     │
   │                       │
   ▼                       │
  Conv 3×3                 │
  BatchNorm                │
   │                       ▼
   └─────────── + ─────────
                │
                ▼
              ReLU
```

The skip connection gives gradients an **expressway** back through the network. Empirically:
- 50-layer ResNet trains fine (where 50-layer plain net diverges)
- 152-layer ResNet still trains
- Hundreds-layer ResNets work; the gain plateaus around 50-100 layers for ImageNet

ResNet is the single most influential architecture in modern deep learning. Skip connections appear in almost every modern model — transformers, diffusion models, you name it.

### Pre-activation vs post-activation

The original ResNet did `ReLU(BN(Conv(x))) + x` (BN inside the block). [He et al. 2016](https://arxiv.org/abs/1603.05027) showed `Conv(ReLU(BN(x))) + x` (pre-activation) trains slightly better. Modern implementations use either; the difference matters at extreme depths.

---

## Part 7: Architectures over time (the speed-run)

| Year | Model | Key idea | ImageNet top-1 |
|---|---|---|---|
| 1998 | LeNet-5 | First successful CNN | (MNIST 99%) |
| 2012 | **AlexNet** | First big-data CNN; ReLU; dropout; GPUs | 63% |
| 2014 | VGG-16/19 | Just stack 3×3 convs, go deep | 71% |
| 2014 | GoogLeNet / Inception | Multi-scale within each block | 70% |
| 2015 | **ResNet-152** | Residual connections | 78% |
| 2016 | DenseNet | Dense skip connections | 77% |
| 2019 | EfficientNet | Compound depth/width/resolution scaling | 84% |
| 2020 | **Vision Transformer (ViT)** | Drop conv entirely, use attention | 88% (with big data) |
| 2022 | ConvNeXt | Re-engineer ResNet with ViT lessons; competitive again | 87% |
| 2024+ | EfficientFormer / MobileViT / etc. | Hybrid conv + attention | varies |

**The takeaway**: vision transformers (week 10) beat CNNs when trained on huge datasets (JFT-300M, IG-3.5B). For "normal" data sizes (ImageNet-1k, your custom dataset), CNNs are competitive — and ConvNeXt shows a properly-tuned ResNet still beats early ViTs. **Both are alive in production today.**

---

## Part 8: Transfer learning — the workhorse

You almost never train a CNN from scratch in production. Instead:

1. Take a model pretrained on ImageNet (e.g. ResNet-50)
2. Replace the final classifier layer with one matching your classes
3. Fine-tune either the whole model or just the classifier on your data

```python
import torchvision.models as models
import torch.nn as nn

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
# Freeze everything
for p in model.parameters():
    p.requires_grad = False
# Replace classifier for your 10-class problem
model.fc = nn.Linear(model.fc.in_features, 10)   # only this trains
```

For "linear probe" (only classifier learns) vs "fine-tune all" (everything updates with a lower LR):

| Use | When |
|---|---|
| Linear probe | Small dataset (<1k images); pretrained features already cover your domain |
| Fine-tune last few blocks | Medium data; domain similar to ImageNet |
| Fine-tune everything (low LR) | Lots of data; domain shift from ImageNet |
| Train from scratch | Very different domain (medical, satellite) AND lots of data |

The single most actionable rule in vision ML: **always start with a pretrained backbone.** Trying to train ResNet-50 from random init on a custom 10k-image dataset is throwing away the months of GPU time someone already spent at NVIDIA / Meta / Google for you.

---

## Part 9: Data augmentation

A CNN's inductive biases handle translation, but not scale, rotation, color shifts, etc. **You augment training data to teach those invariances.**

`torchvision.transforms` provides:

```python
from torchvision import transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

Modern strong augmentations:

| Augmentation | What it does |
|---|---|
| **RandAugment** | Random subset of standard ops, automatically tuned |
| **MixUp** | Blend two images and their labels |
| **CutMix** | Paste a patch from one image into another |
| **AutoAugment** | Learned augmentation policy |

For CIFAR-10 you can reach 95%+ accuracy with the right augmentation alone. For ImageNet, MixUp/CutMix add ~1% routinely. **Augmentation is rarely sexy but consistently effective.**

---

## Part 10: What's next

Week 10 swaps the inductive bias from "locality + translation equivariance" to "attention over all tokens." Transformers don't have spatial structure baked in — they learn it from data, which is why they need more data to match a CNN. But once they have it, they win.

In [lab.md](lab.md) you'll:
- Build a `Conv2d` forward pass from scratch in numpy (educational)
- Train a small CNN on CIFAR-10 from random init to ~80% accuracy
- Build a ResNet block and a small ResNet, train to >90% on CIFAR-10
- Fine-tune a pretrained ResNet-50 on a custom small dataset
- Visualize the receptive field of a deep neuron

By week's end you'll know convolution by hand, residual blocks by hand, and how to leverage Big-Tech pretraining for your own projects.
