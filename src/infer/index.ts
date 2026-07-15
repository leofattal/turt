export { type Backend, CpuBackend } from "./backend.js";
export { GpuBackend, GpuCompute, requestGpuDevice, MATMUL_WGSL, TILE } from "./webgpu/index.js";

// Planned (see PRD "Inference Engine"): generation loop with streaming
// responses (AsyncIterable<string>), batch inference, KV/context caching,
// and quantization hooks. The WebGPU backend (above) accelerates matmul;
// GPU-resident tensors for training are the next step (see docs/GPU.md).
