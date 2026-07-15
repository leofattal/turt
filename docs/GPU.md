# GPU Backend (WebGPU)

Turt's math engine is pure TypeScript on the CPU. The WebGPU backend adds a GPU compute path for the one operation that dominates transformer cost — matrix multiplication — as the first step toward training models larger than a CPU can handle.

## Why WebGPU rather than CUDA

WebGPU runs the *same* WGSL kernels on every GPU: Apple Metal locally, Vulkan on Linux/NVIDIA in the cloud, and browsers. That means the backend is **built and validated here, on this Mac's own GPU**, before any cloud hardware is rented — and the identical code then runs on a rented NVIDIA box. CUDA would have been untestable on Apple silicon and non-portable. Renting a GPU to develop against would bill hours for code we can prove out for free on the M4.

## What's implemented

| File | Role |
| --- | --- |
| `src/infer/webgpu/kernels.ts` | Tiled WGSL matmul, serving both 2-D matmul and batched `bmm` |
| `src/infer/webgpu/gpu.ts` | `GpuCompute` — device, pipeline cache, buffer management, dispatch. Framework-agnostic (operates on `Float32Array`), so it runs in any WebGPU host |
| `src/infer/webgpu/gpu-backend.ts` | `GpuBackend` — adapts `GpuCompute` to the `Backend` interface and `Tensor` |
| `scripts/gpu-bench.ts` | Deno validation + benchmark (doubles as the correctness test) |

The kernel is a standard 16×16 shared-memory tiled multiply: each workgroup cooperatively loads tiles of A and B into workgroup memory and reuses each value TILE times, so global-memory traffic drops by a factor of TILE. The dispatch's z-axis indexes the batch, so one kernel covers both `matmul` (batch = 1) and `bmm`.

## Running it

WebGPU needs a runtime that provides `navigator.gpu`. Node does not (yet), so validation runs under [Deno](https://deno.com), which ships native WebGPU:

```bash
pnpm gpu-bench        # deno run --unstable-webgpu --unstable-sloppy-imports scripts/gpu-bench.ts
```

The script asserts GPU output matches a CPU reference to f32 tolerance and **exits non-zero on any mismatch**, so it is a real test, not just a demo.

## Results (Apple M4, Metal)

Correctness: **bit-identical to the CPU engine** across every shape tested, including non-tile-multiple sizes (17×33@33×9) and batched matmul.

Throughput vs the 2.3 GFLOP/s CPU baseline:

| Matmul | GPU | Speedup |
| --- | --- | --- |
| 128×128 @ 128×128 | 0.3 GFLOP/s | 0.1× — **GPU loses** |
| 4096×160 @ 160×160 (GPT qkv) | 9.6 | 4× |
| 4096×160 @ 160×640 (GPT mlp) | 26.5 | 12× |
| 4096×160 @ 160×1024 (GPT head) | 41.1 | 18× |
| 1024×1024 @ 1024×1024 | 67.2 | 29× |
| 2048×2048 @ 2048×2048 | 118.9 | **52×** |

### Reading the numbers honestly

Two things matter here:

1. **The speedup scales with matrix size.** Tiny matmuls run *slower* on the GPU — the fixed cost of a dispatch (buffer allocation, upload, and reading the result back) swamps the arithmetic. The GPU only pays off once matrices are large, and the win grows with size. This is exactly why a bigger model benefits more: its matmuls live in the range where the GPU is 20–50× ahead.

2. **119 GFLOP/s is a floor, not the ceiling.** This benchmark reallocates buffers and reads the result back to the CPU *on every call* — a full round trip per matmul. The M4 GPU's actual f32 peak is several TFLOP/s. The gap between them is overhead this naive per-op design leaves on the table.

## GPU-resident training

Accelerating a single `matmul` is enough for inference but not for a training speedup: a training step is dozens of small ops, and offloading them one at a time pays the round-trip cost over and over. The fix is **GPU-resident tensors** — keep weights and activations in GPU buffers across the whole forward *and* backward pass, so data crosses the boundary once per step (token ids in, scalar loss out), not once per op.

`src/infer/webgpu/autograd.ts` implements exactly that: a `GpuTensor` with a reverse-mode autograd tape where every op records a backward closure that dispatches kernels *accumulating* into parent gradient buffers. Accumulation is race-free because each kernel writes one output element per thread — no atomics. The op set covers a full transformer: matmul/`bmm`/`bmmBT` (one tiled kernel with transpose+batch+accumulate flags serves forward and both backward passes), broadcasted add/sub/mul, GELU, softmax, causal masking, LayerNorm (fused forward + backward), embedding gather (scatter backward), 4-D permute, the weight-tied head, fused softmax cross-entropy, and AdamW — all with resident optimizer state.

`src/models/gpu-gpt.ts` assembles these into the same GPT architecture as the CPU model.

### Validation

Every GPU op is checked against the trusted CPU autograd, forward and backward:

```bash
pnpm gpu-validate   # 14 transformer ops vs CPU, gradients to f32 tolerance
pnpm gpu-gpt        # full GPT: load identical weights, compare loss + every gradient, then benchmark
```

Loading identical weights into both models and running one training step:

- **loss** matches to 6e-8
- **all 36 parameter gradients** match, worst case 1.3e-5

So the resident training step is numerically the same computation as the CPU model — just far faster.

### Result (Apple M4, 2.04M-param GPT, batch 16 × 128)

| Trainer | tokens/sec |
| --- | --- |
| CPU, single thread | 114 |
| CPU, 8-worker data-parallel | ~620 |
| **GPU-resident (this)** | **5916** |

**52× the single-thread CPU trainer, ~9.5× the 8-core data-parallel trainer** — on the M4's integrated GPU, with a naive (un-fused, un-vectorized) kernel set. A discrete cloud GPU has far more headroom.

## Where cloud GPUs come in

Now that resident training exists and is validated, the cloud offer finally pays off. The same WGSL runs on a cloud NVIDIA GPU via Vulkan, so the path to a genuinely capable model is:

1. ✅ GPU matmul kernel, validated on Metal.
2. ✅ GPU-resident training loop, validated against the CPU engine (loss + every gradient), 52× CPU on the M4.
3. ⬜ Rent a cloud NVIDIA GPU (Paperspace/AWS) and scale the model ~100× — the code is ready; it is now a hardware question, not a code question.

The deliberate ordering held: nothing was rented until there was validated code that could use it. Optimizations that would widen the GPU lead further — kernel fusion (attention in one pass), `vec4` loads, half precision, persistent buffers instead of per-op allocation — are follow-ons, not prerequisites.
