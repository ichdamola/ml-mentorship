# Week 14: CUDA Programming

> **This is the week C++ stops being optional.** Every kernel you write here is C++ with CUDA extensions. Python wrappers come at the end, but the work happens in C++. If you've avoided C-family languages, this is your bridge — and you only need the subset that fits on a page (pointers, structs, function declarations). No templates, no STL, no exceptions.

## 🎯 What you'll learn

Write actual CUDA C++ kernels. By the end of this week, you'll have authored vector addition, matrix multiplication, and a fused softmax — three of the most important shapes in ML — and run them on real hardware.

By the end of this week you'll be able to:

- Write a CUDA kernel: `__global__ void`, threads/blocks/grids, `blockIdx` / `threadIdx`
- Launch a kernel with `<<<grid, block>>>` syntax
- Allocate device memory with `cudaMalloc` / `cudaMemcpy`
- Use shared memory (`__shared__`) to avoid global memory bottlenecks
- Implement a tiled GEMM (matrix multiply) that approaches cuBLAS performance
- Write a CUDA kernel callable from PyTorch via `torch.utils.cpp_extension`
- Understand Triton as the modern Python-flavored alternative to raw CUDA

## 🧰 Lab setup

```bash
# CUDA toolkit must be installed (see week 13)
nvcc hello.cu -o hello && ./hello   # should compile and run
```

```bash
# For PyTorch CUDA extensions:
uv add ninja
```

```bash
# Triton — the modern alternative
uv add triton
```

## ✅ Your job

1. Read [theory.md](theory.md) — kernel launch model, memory hierarchy reuse, occupancy
2. Work through [lab.md](lab.md) — vector add → naive matmul → tiled matmul → fused softmax
3. Benchmark your tiled matmul against `cublasSgemm` and reach within 50% of peak
4. Port the matmul to Triton and compare line counts
5. (Stretch) Implement FlashAttention v1 in Triton

## 📚 Required reading

| Resource | Why | Time |
|---|---|---|
| [Programming Massively Parallel Processors — Ch. 5-7](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) | Tiled GEMM is the canonical exercise | 3 hours |
| [NVIDIA — CUDA C++ Programming Guide (full)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | The reference | bookmark |
| [Simon Boehm — How to optimize a CUDA matmul kernel](https://siboehm.com/articles/22/CUDA-MMM) | The most-cited modern walkthrough | 90 min |
| [Triton documentation](https://triton-lang.org/main/index.html) | Python-flavored CUDA | 60 min |
| [FlashAttention paper (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) | The poster-child of fused kernels in ML | 45 min |

## 💡 What you should already know

- Week 13 (the mental model)
- C basics — pointers, malloc, structs. If you've written one piece of C, you're set.

---

**Next**: [Week 15: Profiling + Performance →](../week-15-profiling-and-perf/readme.md)
