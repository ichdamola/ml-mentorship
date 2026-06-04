# Week 14: Theory — CUDA Programming

This is the week C++ stops being optional. You'll write actual CUDA kernels — vector addition, naive matrix multiply, tiled matrix multiply, fused softmax — and run them on real hardware.

Week 13 gave you the mental model (SMs, warps, memory hierarchy, roofline). Now you're going to write code that respects it. By the end you'll understand every line of a real CUDA kernel, you'll have built a tiled GEMM that approaches cuBLAS performance, and you'll have written the same kernel in Triton (Python-flavored CUDA) for comparison.

---

## Part 1: The CUDA programming model in one diagram

When you launch a CUDA kernel, the GPU creates a **3D grid of threadblocks**, each of which is a **3D block of threads**:

```
┌─────────────────────────────────────────────────────────────────┐
│                          GRID                                   │
│                                                                 │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│   │ Block   │  │ Block   │  │ Block   │  │ Block   │            │
│   │ (0,0)   │  │ (1,0)   │  │ (2,0)   │  │ (3,0)   │            │
│   │         │  │         │  │         │  │         │            │
│   │ thread  │  │         │  │         │  │         │            │
│   │ (0,0,0) │  │         │  │         │  │         │            │
│   │ thread  │  │         │  │         │  │         │            │
│   │ (1,0,0) │  │         │  │         │  │         │            │
│   │  ...    │  │         │  │         │  │         │            │
│   └─────────┘  └─────────┘  └─────────┘  └─────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

Inside a kernel, each thread can query its identity:

```cuda
int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
int idx_z = blockIdx.z * blockDim.z + threadIdx.z;
```

- `blockIdx` — which block am I in (in the grid)?
- `threadIdx` — which thread am I (within my block)?
- `blockDim` — how many threads per block?
- `gridDim` — how many blocks per grid?

The 1D form covers 99% of ML kernels:

```cuda
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

**Memorize this line.** It's how every CUDA thread figures out which piece of data it owns.

---

## Part 2: The simplest possible kernel — vector add

The CUDA "hello world":

```cuda
__global__ void vector_add(float* A, float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}
```

Three things to notice:

1. **`__global__`** — marks this as a function callable from host (CPU) that runs on device (GPU)
2. **No `for` loop** — each of the N threads handles **one** element. Parallelism comes from launching many threads.
3. **The `if (tid < N)` guard** — you usually launch slightly more threads than data points; the guard prevents out-of-bounds writes

The launch from host code:

```cuda
int N = 1024 * 1024;
int block_size = 256;
int grid_size = (N + block_size - 1) / block_size;
vector_add<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
cudaDeviceSynchronize();
```

The triple-bracket syntax `<<<grid, block>>>` is **the** CUDA syntax for launching a kernel. `grid` and `block` are `dim3` (effectively 3-tuples); scalars mean "1D".

Memory has to live on the device:

```cuda
float *d_A, *d_B, *d_C;
cudaMalloc(&d_A, N * sizeof(float));
cudaMalloc(&d_B, N * sizeof(float));
cudaMalloc(&d_C, N * sizeof(float));

cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
// ... launch kernel ...
cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
```

`cudaMalloc` is to GPU memory what `malloc` is to CPU memory. `cudaMemcpy` is the bus crossing — slow. **Minimize crossings.**

---

## Part 3: Naive matmul

The C-style version:

```cuda
__global__ void matmul_naive(float* A, float* B, float* C, int M, int N, int K) {
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
```

Launch with a 2D grid:

```cuda
dim3 block(16, 16);
dim3 grid((N + 15) / 16, (M + 15) / 16);
matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
```

This **works**. It's also **slow** — about 5-10% of cuBLAS performance. Why?

### Why naive is slow: memory access pattern

Each output `C[row, col]` requires `K` reads from row of `A` and `K` reads from column of `B`. For a single 16×16 block computing 256 outputs:

- A reads: 256 × K = 4096 K reads
- B reads: 256 × K = 4096 K reads

Most of these reads **hit HBM** (the slowest memory). With each thread reading independently, there's no reuse — the row of A is read 16 times (once per column in the block), the column of B 16 times.

The fix is **tiling**: load a tile of A and a tile of B into **shared memory** (SMEM) once, then have every thread in the block reuse the same tile from SMEM (100× faster).

---

## Part 4: Tiled matmul with shared memory

The canonical CUDA exercise. By the end of this section you'll have an answer to "how do you write fast matrix multiplication on a GPU."

The idea:

```
For each tile of the output C (e.g. 16×16):
    For each tile-step along the K dimension:
        1. Load a 16×16 tile of A from HBM into shared memory
        2. Load a 16×16 tile of B from HBM into shared memory
        3. Synchronize all threads in the block
        4. Each thread accumulates 16 multiply-adds (using the shared tiles)
        5. Synchronize again before loading next tile
```

The kernel:

```cuda
#define TILE 32   // 32 makes each warp's load coalesce into a single transaction.
                  // TILE=16 also works but a warp then spans two rows of the
                  // block, issuing two transactions per load — half-coalesced.
                  // See Part 5 below for why this matters.

__global__ void matmul_tiled(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // Iterate over tiles along K
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Cooperative load of A tile
        if (row < M && t * TILE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Cooperative load of B tile
        if (t * TILE + threadIdx.y < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();    // wait for all threads in block to finish loading

        // Accumulate this tile's contribution
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();    // wait before overwriting As, Bs
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

Critical lines explained:

- **`__shared__`** — declares memory that lives in the SM's shared memory pool. Visible to all threads in the same block, **not** across blocks.
- **Cooperative load** — each of the 256 threads in a 16×16 block loads ONE element of A and ONE of B. Together they fill the 256-element tiles.
- **`__syncthreads()`** — a barrier. All threads in the block must reach this point before any can continue. Critical because thread (0,0) needs the data thread (15,15) loaded.
- **The inner k-loop** — now reads from SMEM (50× faster than HBM). Each tile element is reused TILE times.

**Performance gain:** from ~5% of cuBLAS to ~50%. Same algorithm, just better data movement.

To get closer to cuBLAS you need register tiling (each thread does multiple outputs), vectorized loads (`float4` instead of `float`), warp-level matrix multiply with tensor cores, and double buffering. The CUTLASS library and FlashAttention paper go deeper.

---

## Part 5: Memory coalescing in your kernel

From week 13: when 32 threads in a warp access **32 consecutive 4-byte floats**, the memory controller issues **one** 128-byte transaction. When they access scattered memory, **32 separate transactions**. 32× slower.

In `matmul_tiled` above, when loading the A tile:

```cuda
As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
```

Threads in the same warp have the same `threadIdx.y` and varying `threadIdx.x` (0 to 31 if TILE=32). So they access:

```
A[row * K + t * TILE + 0]
A[row * K + t * TILE + 1]
A[row * K + t * TILE + 2]
...
A[row * K + t * TILE + 31]
```

These are 32 consecutive floats. **Coalesced.** Fast.

When loading the B tile:

```cuda
Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
```

Here `threadIdx.y` is the same within a warp (mostly), and `col = blockIdx.x * TILE + threadIdx.x` varies by 1. So this is also coalesced.

If you swapped the indices wrong (`B[col * N + ...]`), you'd be reading scattered addresses — same number of bytes, 30× slower.

**Always think about which direction memory grows when you write kernels.**

---

## Part 6: Reduction — the second canonical kernel

Reducing an array of N numbers to one sum looks easy but is non-trivial in CUDA. Each thread can't just `atomicAdd(&sum, x[i])` — atomics serialize.

The classic pattern: **tree reduction within shared memory**.

```cuda
__global__ void reduce(float* X, float* result, int N) {
    __shared__ float partial[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    partial[tid] = (gid < N) ? X[gid] : 0.0f;
    __syncthreads();

    // INVARIANT: blockDim.x must be a power of two for this tree pattern.
    // Otherwise the halving leaves stragglers that never get summed.
    // Halve at each step
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, partial[0]);   // one atomic per block, not per thread
    }
}
```

The tree: at step 1, half the threads add their partner's value. At step 2, a quarter do. After `log₂(BLOCK_SIZE)` steps, thread 0 holds the block's sum. Then one atomic per block.

For 1M elements with BLOCK_SIZE=256: 4096 blocks each do `log₂(256) = 8` parallel add steps and `1` atomic. The atomic contention is reasonable.

Modern variants: **warp-shuffle** (`__shfl_down_sync`) replaces SMEM reductions within a warp, faster. **Cooperative groups** APIs make this cleaner.

---

## Part 7: Fused softmax — fitting the formula in one kernel

Softmax along the last axis of an `(N, D)` matrix:

```
softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x[i] - max(x)))
```

You need: **the max**, **the sum after subtracting max**, then divide.

A naive implementation does **3 passes** over the data:

```
Pass 1: m = max(x)
Pass 2: s = sum(exp(x - m))
Pass 3: y = exp(x - m) / s
```

Each pass reads + writes from HBM. **3× the memory traffic.**

A **fused kernel** does it in one pass per row, keeping the row in SMEM:

```cuda
__global__ void softmax_fused(float* X, float* Y, int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;

    float* x_row = X + row * D;
    float* y_row = Y + row * D;

    __shared__ float buf[BLOCK_DIM];
    int tid = threadIdx.x;

    // Step 1: each thread finds max of its portion
    float thread_max = -INFINITY;
    for (int i = tid; i < D; i += blockDim.x) {
        thread_max = fmaxf(thread_max, x_row[i]);
    }
    buf[tid] = thread_max;
    __syncthreads();
    // Reduce to find the row max (tree reduction)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] = fmaxf(buf[tid], buf[tid + s]);
        __syncthreads();
    }
    float row_max = buf[0];
    __syncthreads();

    // Step 2: compute sum of exp(x - max) using same reduction pattern
    float thread_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        thread_sum += expf(x_row[i] - row_max);
    }
    buf[tid] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }
    float row_sum = buf[0];

    // Step 3: write outputs
    for (int i = tid; i < D; i += blockDim.x) {
        y_row[i] = expf(x_row[i] - row_max) / row_sum;
    }
}
```

This kernel still **logically** does three passes over `x_row` (max, sum-of-exp, divide), but **all three reuse the same row data** — if the row fits in L1/L2 (true for D up to a few thousand floats), the second and third passes hit cache, not HBM. So the effective HBM traffic is ~1 read of `x_row` + 1 write of `y_row` per row vs the naive version's 3+1 reads. For very large D (LLM vocabularies, last-dim softmax over tens of thousands), passes 2 and 3 fall out of L1; **the true single-pass form** is *online softmax* (running max + running sum with rescale), which FlashAttention uses and which this kernel is a stepping stone toward.

**This is the same pattern FlashAttention uses to fuse all of attention into one kernel:** keep intermediate results in SMEM/registers across reduction passes, write final output once.

---

## Part 8: Calling CUDA from PyTorch

You write CUDA, compile via PyTorch's just-in-time inline build:

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
__global__ void vector_add_kernel(
    const float* A, const float* B, float* C, int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) C[tid] = A[tid] + B[tid];
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
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

mod = load_inline(
    name="vector_add_ext",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["vector_add"],
    verbose=True,
)

a = torch.randn(1024 * 1024, device="cuda")
b = torch.randn(1024 * 1024, device="cuda")
c = mod.vector_add(a, b)
```

The first run compiles via nvcc (slow, 10-30s); subsequent runs use the cached binary.

For real projects you'd use `torch.utils.cpp_extension.CUDAExtension` in `setup.py`, but `load_inline` is perfect for the lab.

---

## Part 9: Triton — Python-flavored CUDA

OpenAI's [Triton](https://github.com/openai/triton) lets you write GPU kernels in Python that compile to PTX. The same matmul:

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
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

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.store(c_ptrs, acc)
```

**~30 lines vs ~80 lines of CUDA C++**, and within ~10% of CUTLASS performance. Triton handles tiling, vectorization, double-buffering automatically.

Modern ML kernels (FlashAttention, fused MoE, custom activations) are increasingly written in Triton because of this productivity gain. NVIDIA's CUTLASS (C++ templates) is still faster for the absolute frontier; Triton is the practical default for most research and production work.

---

## Part 10: When *not* to write a kernel

You usually shouldn't. Here's when to consider it:

| Situation | Use |
|---|---|
| You have a slow op identified in the profiler (week 15) | Try `torch.compile` first |
| `torch.compile` not enough | Write a custom op in Triton |
| Triton not enough | CUDA C++ |
| CUDA C++ not enough | CUTLASS / cuBLAS / cuDNN |

You're rarely writing CUDA from scratch in production. **More commonly, you read existing kernels** (FlashAttention, xFormers, PyTorch's `aten/src/ATen/native/cuda/`) and understand them. This week's purpose is to give you that reading ability.

---

## What's next

In [lab.md](lab.md) you'll:

- Write `vector_add` in CUDA C++; call from PyTorch
- Write naive matmul; benchmark vs cuBLAS
- Write tiled matmul with SMEM; reach ≥50% of cuBLAS performance
- Write fused softmax; reach ≥2× the naive PyTorch softmax in bandwidth-bound regime
- Port the matmul to Triton; compare line counts and performance
- (Stretch) Implement FlashAttention v1 in Triton

By end of week 14 you'll have written kernels that go all the way down to PTX. Week 15 you'll profile real models and see what to write next.
