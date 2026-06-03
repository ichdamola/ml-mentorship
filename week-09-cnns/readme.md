# Week 09: Convolutional Neural Networks

## 🎯 What you'll learn

CNNs invented modern computer vision. Even with transformers eating their lunch, convolutions still dominate edge inference and many industrial pipelines. Understand convolution as a learned filter, then build through LeNet → VGG → ResNet.

By the end of this week you'll be able to:

- Explain convolution arithmetic — kernel size, stride, padding, dilation, output shape
- Implement a `Conv2d` forward pass by hand (we'll do the backward in the lab)
- Understand pooling, batch norm, residual connections
- Train ResNet-18 on CIFAR-10 to >90% accuracy
- Use `torchvision` transforms and `ImageFolder` datasets
- Apply transfer learning from an ImageNet-pretrained backbone

## 🧰 Lab setup

```bash
uv add torchvision pillow
```

## ✅ Your job

1. Read [theory.md](theory.md) — convolution arithmetic, the receptive field, residuals
2. Work through [lab.md](lab.md) — train ResNet-18 on CIFAR-10 with the discipline from week 08
3. Visualize the learned filters of the first conv layer
4. (Stretch) Fine-tune a pretrained ResNet-50 on a custom dataset

## 📚 Required reading

| Resource | Why | Time |
|---|---|---|
| [CS231n — Convolutional networks](https://cs231n.github.io/convolutional-networks/) | The canonical introduction | 90 min |
| [Deep Residual Learning (He et al., 2015)](https://arxiv.org/abs/1512.03385) | The ResNet paper | 30 min |
| [Batch Normalization (Ioffe & Szegedy, 2015)](https://arxiv.org/abs/1502.03167) | Why models train at all without it | 30 min |

## 💡 What you should already know

- Weeks 05-08

---

**Next**: [Week 10: Transformers →](../week-10-transformers/readme.md)
