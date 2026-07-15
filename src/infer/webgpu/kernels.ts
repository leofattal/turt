/**
 * WGSL compute kernels for the WebGPU backend.
 *
 * These are plain strings with no project dependencies, so they can be loaded
 * from any WebGPU host — Deno (used for validation on Apple Metal), a Node
 * WebGPU binding, a browser, or a cloud NVIDIA GPU via Vulkan. The same source
 * runs unchanged across all of them; only the host that uploads buffers and
 * dispatches differs.
 */

/** Side length of the square tile each workgroup cooperatively loads. */
export const TILE = 16;

/**
 * Tiled matrix multiply C = A @ B with shared-memory reuse.
 *
 * One kernel serves both the 2-D case and batched matmul: the z dimension of
 * the dispatch indexes the batch, and `dims.batch = 1` reduces to plain matmul.
 * Each workgroup computes a TILE x TILE block of C, streaming K in TILE-wide
 * strips through workgroup-shared memory so each loaded value is reused TILE
 * times instead of being re-fetched from global memory.
 *
 * Layouts are row-major and contiguous, matching Turt's CPU `Tensor`:
 *   A: [batch, M, K]   B: [batch, K, N]   C: [batch, M, N]
 */
export const MATMUL_WGSL = /* wgsl */ `
struct Dims { M: u32, K: u32, N: u32, batch: u32 };

@group(0) @binding(0) var<storage, read>       A: array<f32>;
@group(0) @binding(1) var<storage, read>       B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

const TILE = ${TILE}u;

var<workgroup> tileA: array<f32, ${TILE * TILE}>;
var<workgroup> tileB: array<f32, ${TILE * TILE}>;

@compute @workgroup_size(${TILE}, ${TILE}, 1)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id)        wid: vec3<u32>,
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let batch = gid.z;
  let row = wid.y * TILE + lid.y;
  let col = wid.x * TILE + lid.x;

  let aBase = batch * dims.M * dims.K;
  let bBase = batch * dims.K * dims.N;
  let cBase = batch * dims.M * dims.N;

  var acc = 0.0;
  let numTiles = (dims.K + TILE - 1u) / TILE;
  for (var t = 0u; t < numTiles; t = t + 1u) {
    let kOff = t * TILE;

    let aCol = kOff + lid.x;
    if (row < dims.M && aCol < dims.K) {
      tileA[lid.y * TILE + lid.x] = A[aBase + row * dims.K + aCol];
    } else {
      tileA[lid.y * TILE + lid.x] = 0.0;
    }

    let bRow = kOff + lid.y;
    if (bRow < dims.K && col < dims.N) {
      tileB[lid.y * TILE + lid.x] = B[bBase + bRow * dims.N + col];
    } else {
      tileB[lid.y * TILE + lid.x] = 0.0;
    }

    workgroupBarrier();
    for (var k = 0u; k < TILE; k = k + 1u) {
      acc = acc + tileA[lid.y * TILE + k] * tileB[k * TILE + lid.x];
    }
    workgroupBarrier();
  }

  if (row < dims.M && col < dims.N) {
    C[cBase + row * dims.N + col] = acc;
  }
}
`;
