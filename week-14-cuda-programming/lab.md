# Week 14: Lab — Write Your First CUDA Kernels

You'll write four kernels — vector add, naive matmul, tiled matmul, fused softmax — call them from PyTorch, benchmark vs cuBLAS, and then rewrite the matmul in Triton in 1/3 the lines.

## Setup

You need:
- An NVIDIA GPU (T4, RTX 3090/4090, A100, H100 — anything works)
- CUDA toolkit ≥ 12.0 (Colab has it; otherwise install from [NVIDIA's CUDA downloads](https://developer.nvidia.com/cuda-downloads))
- PyTorch with CUDA support
- `ninja` for `cpp_extension` builds
- `triton` for the Triton portion

```bash
uv add ninja triton
nvcc --version    # should be ≥ 12.0
```

```python
import torch
import time
import math
from torch.utils.cpp_extension import load_inline

assert torch.cuda.is_available(), "This lab requires a CUDA GPU."
device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## Exercise 14.1 — Hello, CUDA: vector_add

```python
cuda_src = r"""
__global__ void vector_add_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) C[tid] = A[tid] + B[tid];
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Float32 only");
    TORCH_CHECK(a.sizes() == b.sizes(), "Shapes must match");

    auto c = torch::empty_like(a);
    int N = a.numel();
    int block = 256;
    int grid = (N + block - 1) / block;
    vector_add_kernel<<<grid, block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(),
        c.data_ptr<float>(), N
    );
    return c;
}
"""
cpp_src = "torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);"

ext = load_inline(
    name="vector_add_ext",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["vector_add"],
    verbose=True,
)

# Test
N = 1 << 24   # 16M
a = torch.randn(N, device=device)
b = torch.randn(N, device=device)
c_mine = ext.vector_add(a, b)
c_ref = a + b
print(f"max abs diff: {(c_mine - c_ref).abs().max():.2e}")
```

The first compile takes 10-30s. Subsequent runs are cached. **If you got `max abs diff < 1e-6`, congratulations — you just wrote and ran your first CUDA kernel.**

---

## Exercise 14.2 — Benchmark vector_add

```python
def bench(fn, *args, n_iter=200):
    for _ in range(5): fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter): fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000   # ms

t_mine = bench(ext.vector_add, a, b)
t_py = bench(lambda x, y: x + y, a, b)
bytes_moved = 3 * N * 4   # read A, B; write C — fp32 = 4 bytes
bw_mine = bytes_moved / (t_mine / 1000) / 1e9
print(f"my kernel:   {t_mine:.3f} ms ({bw_mine:.0f} GB/s)")
print(f"PyTorch +:   {t_py:.3f} ms ({bytes_moved/(t_py/1000)/1e9:.0f} GB/s)")
```

PyTorch's `+` and your kernel should be in the same ballpark (within 2× of each other). Both should be **memory-bound** — close to your card's peak HBM bandwidth from week 13.

**Why isn't your kernel faster?** Elementwise add is fully memory-bound; there's no algorithmic room to win. The only way to speed it up is to fuse with a neighboring op so you load each input once instead of twice (Exercise 14.8).

---

## Exercise 14.3 — Naive matmul

```python
naive_matmul_src = r"""
__global__ void matmul_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "CUDA only");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Float32 only");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dims must match");

    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_naive_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(),
        C.data_ptr<float>(), M, N, K
    );
    return C;
}
"""

naive_ext = load_inline(
    name="matmul_naive_ext",
    cpp_sources=["torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B);"],
    cuda_sources=[naive_matmul_src],
    functions=["matmul_naive"],
    verbose=True,
)

# Correctness
M, N, K = 1024, 1024, 1024
A = torch.randn(M, K, device=device)
B = torch.randn(K, N, device=device)
C_mine = naive_ext.matmul_naive(A, B)
C_ref = A @ B
err = (C_mine - C_ref).abs().max() / C_ref.abs().max()
print(f"relative error: {err:.2e}")

# Benchmark vs cuBLAS
def matmul_tflops(fn, M, N, K, n_iter=50):
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)
    for _ in range(5): fn(A, B)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter): fn(A, B)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_iter
    return 2 * M * N * K / dt / 1e12   # TFLOPS

tf_naive = matmul_tflops(naive_ext.matmul_naive, 1024, 1024, 1024)
tf_cublas = matmul_tflops(lambda A, B: A @ B, 1024, 1024, 1024)
print(f"naive matmul:    {tf_naive:>7.2f} TFLOPS")
print(f"cuBLAS (A @ B):  {tf_cublas:>7.2f} TFLOPS  ({tf_cublas/tf_naive:.1f}× faster)")
```

You should see naive at ~0.5-2 TFLOPS, cuBLAS at 10-50 TFLOPS depending on your GPU. **20-50× gap.** That's the cost of not using shared memory.

---

## Exercise 14.4 — Tiled matmul with shared memory

```python
tiled_matmul_src = r"""
#define TILE 16

__global__ void matmul_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Cooperative load of A and B tiles
        if (row < M && t * TILE + tx < K)
            As[ty][tx] = A[row * K + t * TILE + tx];
        else
            As[ty][tx] = 0.0f;

        if (t * TILE + ty < K && col < N)
            Bs[ty][tx] = B[(t * TILE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "CUDA only");
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(),
        C.data_ptr<float>(), M, N, K
    );
    return C;
}
"""

tiled_ext = load_inline(
    name="matmul_tiled_ext",
    cpp_sources=["torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B);"],
    cuda_sources=[tiled_matmul_src],
    functions=["matmul_tiled"],
    verbose=True,
)

# Correctness
C_mine = tiled_ext.matmul_tiled(A, B)
err = (C_mine - C_ref).abs().max() / C_ref.abs().max()
print(f"relative error: {err:.2e}")

# Benchmark
tf_tiled = matmul_tflops(tiled_ext.matmul_tiled, 1024, 1024, 1024)
print(f"\nnaive:    {tf_naive:>7.2f} TFLOPS  (1×)")
print(f"tiled:    {tf_tiled:>7.2f} TFLOPS  ({tf_tiled/tf_naive:.1f}× faster than naive)")
print(f"cuBLAS:   {tf_cublas:>7.2f} TFLOPS  ({tf_cublas/tf_naive:.1f}× faster than naive, {tf_cublas/tf_tiled:.1f}× faster than tiled)")
```

You should see tiled at **5-10× faster than naive**, reaching maybe ~30-50% of cuBLAS. The remaining gap is closed by register tiling, tensor cores, async copy, and vectorized loads — see [Simon Boehm's CUDA matmul walkthrough](https://siboehm.com/articles/22/CUDA-MMM) for the full optimization path.

---

## Exercise 14.5 — Vary the tile size

Memory access patterns and SMEM bank conflicts make this a nuanced choice.

```python
def make_tiled_with_size(TILE):
    src = naive_matmul_src.replace("naive", "tiled_T" + str(TILE))   # ignore the naive ref
    # ... easier to rebuild for each TILE size in real code
    return None

# Simpler: hard-code a few tile sizes (8, 16, 32) and rebuild
for TILE in [8, 16, 32]:
    src = tiled_matmul_src.replace("#define TILE 16", f"#define TILE {TILE}").replace(
        "matmul_tiled_kernel", f"matmul_tiled_T{TILE}_kernel"
    ).replace("matmul_tiled(", f"matmul_tiled_T{TILE}(")

    ext = load_inline(
        name=f"matmul_tiled_T{TILE}_ext",
        cpp_sources=[f"torch::Tensor matmul_tiled_T{TILE}(torch::Tensor A, torch::Tensor B);"],
        cuda_sources=[src],
        functions=[f"matmul_tiled_T{TILE}"],
        verbose=False,
    )
    tf = matmul_tflops(getattr(ext, f"matmul_tiled_T{TILE}"), 1024, 1024, 1024)
    print(f"TILE={TILE}: {tf:.2f} TFLOPS")
```

You'll find TILE=16 or TILE=32 typically wins. TILE=8 isn't using enough threads per block; very large tiles run out of SMEM.

---

## Exercise 14.6 — Fused softmax

```python
softmax_src = r"""
#define BLOCK_DIM 256

__global__ void softmax_fused_kernel(
    const float* __restrict__ X, float* __restrict__ Y,
    int N, int D
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* x_row = X + row * D;
    float* y_row = Y + row * D;

    __shared__ float buf[BLOCK_DIM];
    int tid = threadIdx.x;

    // 1) Find row max
    float thread_max = -INFINITY;
    for (int i = tid; i < D; i += BLOCK_DIM) {
        thread_max = fmaxf(thread_max, x_row[i]);
    }
    buf[tid] = thread_max;
    __syncthreads();
    for (int s = BLOCK_DIM / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] = fmaxf(buf[tid], buf[tid + s]);
        __syncthreads();
    }
    float row_max = buf[0];
    __syncthreads();

    // 2) Sum of exp(x - max)
    float thread_sum = 0.0f;
    for (int i = tid; i < D; i += BLOCK_DIM) {
        thread_sum += expf(x_row[i] - row_max);
    }
    buf[tid] = thread_sum;
    __syncthreads();
    for (int s = BLOCK_DIM / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }
    float row_sum = buf[0];

    // 3) Write outputs
    for (int i = tid; i < D; i += BLOCK_DIM) {
        y_row[i] = expf(x_row[i] - row_max) / row_sum;
    }
}

torch::Tensor softmax_fused(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kFloat32, "Float32 CUDA only");
    TORCH_CHECK(X.dim() == 2, "Expect (N, D) input");
    int N = X.size(0), D = X.size(1);
    auto Y = torch::empty_like(X);
    softmax_fused_kernel<<<N, BLOCK_DIM>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), N, D
    );
    return Y;
}
"""

softmax_ext = load_inline(
    name="softmax_fused_ext",
    cpp_sources=["torch::Tensor softmax_fused(torch::Tensor X);"],
    cuda_sources=[softmax_src],
    functions=["softmax_fused"],
    verbose=True,
)

# Correctness
X = torch.randn(1024, 2048, device=device)
Y_mine = softmax_ext.softmax_fused(X)
Y_ref = torch.softmax(X, dim=-1)
print(f"max abs diff: {(Y_mine - Y_ref).abs().max():.2e}")

# Benchmark
t_mine = bench(softmax_ext.softmax_fused, X)
t_torch = bench(lambda x: torch.softmax(x, dim=-1), X)
print(f"my fused:   {t_mine:.3f} ms")
print(f"PyTorch:    {t_torch:.3f} ms")
```

PyTorch's softmax is already heavily optimized (and may use Triton internally), so beating it by a lot is hard. You should be **within 2×**. For large `D` your version is competitive.

The bandwidth math: you read N×D fp32 and write N×D fp32 = `2 × 1024 × 2048 × 4 = 16 MB`. At HBM peak (say 1 TB/s), the lower bound is **16 µs**. Your kernel should be 30-100 µs — comparable.

---

## Exercise 14.7 — Triton matmul

Now the same algorithm in Triton.

```python
import triton
import triton.language as tl

@triton.jit
def matmul_triton_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        # Mask edges so we don't load garbage outside the matrices
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul_triton(A, B):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    matmul_triton_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )
    return C

# Correctness
C_triton = matmul_triton(A, B)
err = (C_triton - C_ref).abs().max() / C_ref.abs().max()
print(f"Triton relative error: {err:.2e}")

# Benchmark
tf_triton = matmul_tflops(matmul_triton, 1024, 1024, 1024)
print(f"\nnaive CUDA C++:  {tf_naive:>7.2f} TFLOPS")
print(f"tiled CUDA C++:  {tf_tiled:>7.2f} TFLOPS")
print(f"Triton:          {tf_triton:>7.2f} TFLOPS")
print(f"cuBLAS:          {tf_cublas:>7.2f} TFLOPS")
```

Triton in ~30 lines is typically 80-90% of cuBLAS, beating your tiled CUDA-C++ version. **The productivity gain is real.** This is why FlashAttention, fused MoE, and most modern research kernels are written in Triton.

---

## Exercise 14.8 — Kernel fusion

Two ops you can fuse: bias-add + ReLU. The naive version reads + writes twice; fused reads + writes once.

```python
fused_src = r"""
__global__ void bias_relu_fused_kernel(
    const float* X, const float* bias, float* Y,
    int B, int D
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * D;
    if (tid >= total) return;
    int d = tid % D;
    float val = X[tid] + bias[d];
    Y[tid] = val > 0 ? val : 0.0f;
}

torch::Tensor bias_relu_fused(torch::Tensor X, torch::Tensor bias) {
    int B = X.size(0), D = X.size(1);
    auto Y = torch::empty_like(X);
    int total = B * D;
    int block = 256;
    int grid = (total + block - 1) / block;
    bias_relu_fused_kernel<<<grid, block>>>(
        X.data_ptr<float>(), bias.data_ptr<float>(),
        Y.data_ptr<float>(), B, D
    );
    return Y;
}
"""

fused_ext = load_inline(
    name="bias_relu_fused_ext",
    cpp_sources=["torch::Tensor bias_relu_fused(torch::Tensor X, torch::Tensor bias);"],
    cuda_sources=[fused_src],
    functions=["bias_relu_fused"],
    verbose=True,
)

X = torch.randn(4096, 4096, device=device)
bias = torch.randn(4096, device=device)

def unfused(X, bias):
    return torch.relu(X + bias)

# Correctness
y_fused = fused_ext.bias_relu_fused(X, bias)
y_ref = unfused(X, bias)
print(f"max abs diff: {(y_fused - y_ref).abs().max():.2e}")

# Benchmark
t_fused = bench(fused_ext.bias_relu_fused, X, bias)
t_unfused = bench(unfused, X, bias)
print(f"fused:    {t_fused:.3f} ms")
print(f"unfused:  {t_unfused:.3f} ms ({t_unfused/t_fused:.1f}× slower)")
```

You should see fused beating unfused by **1.3-1.8×**. That's the magnitude of the fusion gain for adjacent memory-bound ops — and exactly what `torch.compile` and FlashAttention do at scale.

---

## Exercise 14.9 (stretch) — FlashAttention v1 in Triton

The published [FlashAttention](https://github.com/Dao-AILab/flash-attention) kernel is in CUDA. A simplified Triton version is in the [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html). Walk through it; the core idea is the same as the fused softmax in 14.6 — keep intermediate results in SRAM rather than rematerializing to HBM.

The full implementation is ~300 lines of Triton. **Read it. You can.** It implements the trick that made long-context LLMs tractable.

---

## Submission checklist

- [ ] `vector_add` runs; matches PyTorch's `+` to ~1e-6
- [ ] `vector_add` bandwidth is within 2× of PyTorch's `+`
- [ ] Naive matmul matches `A @ B` to relative error ≤ 1e-4
- [ ] Tiled matmul is ≥ 5× faster than naive
- [ ] Tiled matmul reaches ≥ 30% of cuBLAS TFLOPS
- [ ] Tile-size sweep shows TILE=16 or 32 wins
- [ ] Fused softmax matches `torch.softmax` to ≤ 1e-5 and is within 2× of PyTorch
- [ ] Triton matmul reaches ≥ 70% of cuBLAS TFLOPS in fewer lines than tiled CUDA
- [ ] `bias_relu_fused` beats unfused PyTorch by ≥ 1.3×
- [ ] (Stretch) Read and understand the FlashAttention Triton kernel end-to-end

---

## What you just did

You wrote kernels in CUDA C++ that go all the way down to PTX. You saw the difference between naive and tiled memory access, measured the ~10× gain from SMEM, fused two memory-bound ops into one. You used Triton to write the same matmul in 1/3 the lines at 80-90% of cuBLAS performance.

**Most ML engineers never write a CUDA kernel.** The ones who do are the ones contributing to FlashAttention, xFormers, vLLM, and the open-source kernel libraries that everyone else builds on. After this week, you have the bar to read those kernels with comprehension — and the foundation to write them when needed.

Week 15 profiles real models to find which kernels actually matter. Week 16 takes everything you've built and serves it.

---

**Next**: [Week 15: Profiling and Performance →](../week-15-profiling-and-perf/readme.md)
