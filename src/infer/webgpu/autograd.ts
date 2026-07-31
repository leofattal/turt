/**
 * GPU-resident reverse-mode autodiff.
 *
 * The CPU `Tensor` (src/math) allocates a fresh Float32Array per op and runs
 * backward on the CPU. That is correct but means offloading a single op pays a
 * CPU<->GPU round trip every call — fatal for training, where one step is dozens
 * of small ops. `GpuTensor` keeps every value in a GPU buffer across the whole
 * forward and backward pass: data crosses the boundary once per step (inputs in,
 * scalar loss out), not once per op.
 *
 * The design mirrors the CPU engine so the two validate against each other:
 * every differentiable op records parents and a backward closure that reads the
 * output's gradient buffer and *accumulates* into the parents' gradient buffers.
 * Accumulation is race-free because every kernel writes one output element per
 * thread — no atomics.
 *
 * Depends only on the standard WebGPU API, so the same source runs under Deno on
 * Apple Metal (how it is validated), Node with a WebGPU binding, or cloud NVIDIA
 * via Vulkan.
 */

const TILE = 16;

/**
 * Tiled matmul with transpose + batch + accumulate flags. One kernel is the
 * forward pass and both backward passes:
 *   forward:  C  = A[M,K] @ B[K,N]
 *   dA += dC @ B^T  (logical M,N,K, transB=1, accum=1)
 *   dB += A^T @ dC  (logical K,M,N, transA=1, accum=1)
 * and batched (batch > 1) for attention. Transpose flags change only indexing,
 * so no transpose is materialized.
 */
const MATMUL = /* wgsl */ `
struct Dims { M:u32, K:u32, N:u32, batch:u32, transA:u32, transB:u32, accum:u32, _pad:u32 };
@group(0) @binding(0) var<storage, read>       A: array<f32>;
@group(0) @binding(1) var<storage, read>       B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;
const TILE = ${TILE}u;
var<workgroup> As: array<f32, ${TILE * TILE}>;
var<workgroup> Bs: array<f32, ${TILE * TILE}>;
@compute @workgroup_size(${TILE}, ${TILE}, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = gid.z;
  let row = wid.y * TILE + lid.y;
  let col = wid.x * TILE + lid.x;
  let aBase = batch * d.M * d.K;
  let bBase = batch * d.K * d.N;
  let cBase = batch * d.M * d.N;
  var acc = 0.0;
  let numTiles = (d.K + TILE - 1u) / TILE;
  for (var t = 0u; t < numTiles; t = t + 1u) {
    let kOff = t * TILE;
    let aK = kOff + lid.x;
    if (row < d.M && aK < d.K) {
      if (d.transA == 0u) { As[lid.y*TILE+lid.x] = A[aBase + row*d.K + aK]; }
      else                { As[lid.y*TILE+lid.x] = A[aBase + aK*d.M + row]; }
    } else { As[lid.y*TILE+lid.x] = 0.0; }
    let bK = kOff + lid.y;
    if (bK < d.K && col < d.N) {
      if (d.transB == 0u) { Bs[lid.y*TILE+lid.x] = B[bBase + bK*d.N + col]; }
      else                { Bs[lid.y*TILE+lid.x] = B[bBase + col*d.K + bK]; }
    } else { Bs[lid.y*TILE+lid.x] = 0.0; }
    workgroupBarrier();
    for (var k = 0u; k < TILE; k = k + 1u) { acc = acc + As[lid.y*TILE+k] * Bs[k*TILE+lid.x]; }
    workgroupBarrier();
  }
  if (row < d.M && col < d.N) {
    let idx = cBase + row*d.N + col;
    if (d.accum == 0u) { C[idx] = acc; } else { C[idx] = C[idx] + acc; }
  }
}
`;

/** Elementwise binary with right-broadcast: out[i] = a[i] OP b[i%bLen]. op 0 add,1 sub,2 mul. */
const EWISE = /* wgsl */ `
struct P { n:u32, bLen:u32, op:u32, accum:u32 };
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y*8388608u; if (i >= p.n) { return; }
  let bv = b[i % p.bLen];
  var r = a[i] + bv;
  if (p.op == 1u) { r = a[i] - bv; } else if (p.op == 2u) { r = a[i] * bv; }
  if (p.accum == 0u) { out[i] = r; } else { out[i] = out[i] + r; }
}
`;

/** out[i] (+)= a[i] * s. */
const SCALE = /* wgsl */ `
struct P { n:u32, accum:u32, _a:u32, _b:u32, s:f32, _c:f32, _d:f32, _e:f32 };
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y*8388608u; if (i >= p.n) { return; }
  let r = a[i] * p.s;
  if (p.accum == 0u) { out[i] = r; } else { out[i] = out[i] + r; }
}
`;

/** out[i] += src[i % bLen] — broadcast accumulate (for sum/broadcast backward). */
const BCAST_ACC = /* wgsl */ `
struct P { n:u32, bLen:u32, _a:u32, _b:u32, s:f32, _c:f32, _d:f32, _e:f32 };
@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y*8388608u; if (i >= p.n) { return; }
  out[i] = out[i] + src[i % p.bLen] * p.s;
}
`;

/** Segment sum: out[j] (+)= s * sum_{i%bLen==j} g[i]. */
const SEGSUM = /* wgsl */ `
struct P { n:u32, bLen:u32, accum:u32, _a:u32, s:f32, _b:f32, _c:f32, _d:f32 };
@group(0) @binding(0) var<storage, read>       g: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x; if (j >= p.bLen) { return; }
  var s = 0.0; var i = j;
  loop { if (i >= p.n) { break; } s = s + g[i]; i = i + p.bLen; }
  let r = s * p.s;
  if (p.accum == 0u) { out[j] = r; } else { out[j] = out[j] + r; }
}
`;

/** Total sum -> out[0]. Single thread; used only for the scalar loss. */
const TOTAL = /* wgsl */ `
struct P { n:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;
@compute @workgroup_size(1)
fn main() { var s = 0.0; for (var i=0u;i<p.n;i=i+1u){ s = s + a[i]; } out[0] = s; }
`;

const RELU = /* wgsl */ `
struct P { n:u32,_a:u32,_b:u32,_c:u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let i=g.x+g.y*8388608u; if(i>=p.n){return;} out[i]=max(x[i],0.0); }
`;
const RELU_BWD = /* wgsl */ `
struct P { n:u32,_a:u32,_b:u32,_c:u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gOut: array<f32>;
@group(0) @binding(2) var<storage, read_write> gIn: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let i=g.x+g.y*8388608u; if(i>=p.n){return;} if(x[i]>0.0){ gIn[i]=gIn[i]+gOut[i]; } }
`;

/** SGD update: param[i] -= lr * grad[i]. */
const SGD = /* wgsl */ `
struct P { n:u32,_a:u32,_b:u32,_c:u32, lr:f32,_d:f32,_e:f32,_f:f32 };
@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> param: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let i=g.x+g.y*8388608u; if(i>=p.n){return;} param[i]=param[i]-p.lr*grad[i]; }
`;

/** GELU (tanh approx, matching the CPU engine) forward and backward. */
const GELU = /* wgsl */ `
struct P { n:u32,_a:u32,_b:u32,_c:u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
const C = 0.7978845608028654;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let i=g.x+g.y*8388608u; if(i>=p.n){return;}
  let v = x[i]; out[i] = 0.5*v*(1.0+tanh(C*(v+0.044715*v*v*v))); }
`;
const GELU_BWD = /* wgsl */ `
struct P { n:u32,_a:u32,_b:u32,_c:u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gOut: array<f32>;
@group(0) @binding(2) var<storage, read_write> gIn: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
const C = 0.7978845608028654;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let i=g.x+g.y*8388608u; if(i>=p.n){return;}
  let v = x[i];
  let inner = C*(v+0.044715*v*v*v);
  let t = tanh(inner);
  let dinner = C*(1.0+3.0*0.044715*v*v);
  gIn[i] = gIn[i] + gOut[i]*(0.5*(1.0+t)+0.5*v*(1.0-t*t)*dinner); }
`;

/** SiLU / swish (x * sigmoid(x)) forward and backward — the SwiGLU gate. */
const SILU = /* wgsl */ `
struct P { n:u32,_a:u32,_b:u32,_c:u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let i=g.x+g.y*8388608u; if(i>=p.n){return;}
  let v = x[i]; out[i] = v/(1.0+exp(-v)); }
`;
const SILU_BWD = /* wgsl */ `
struct P { n:u32,_a:u32,_b:u32,_c:u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gOut: array<f32>;
@group(0) @binding(2) var<storage, read_write> gIn: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let i=g.x+g.y*8388608u; if(i>=p.n){return;}
  let v = x[i]; let s = 1.0/(1.0+exp(-v)); let y = v*s;
  gIn[i] = gIn[i] + gOut[i]*(s + y*(1.0-s)); }
`;

/** Softmax over the last dimension (one thread per row). */
const SOFTMAX = /* wgsl */ `
struct P { rows:u32, d:u32,_a:u32,_b:u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let r=g.x; if(r>=p.rows){return;}
  let off = r*p.d;
  var m = x[off];
  for (var j=1u;j<p.d;j=j+1u){ m = max(m, x[off+j]); }
  var s = 0.0;
  for (var j=0u;j<p.d;j=j+1u){ let e = exp(x[off+j]-m); out[off+j]=e; s=s+e; }
  for (var j=0u;j<p.d;j=j+1u){ out[off+j]=out[off+j]/s; }
}
`;
const SOFTMAX_BWD = /* wgsl */ `
struct P { rows:u32, d:u32,_a:u32,_b:u32 };
@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read> gOut: array<f32>;
@group(0) @binding(2) var<storage, read_write> gIn: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let r=g.x; if(r>=p.rows){return;}
  let off = r*p.d;
  var dot = 0.0;
  for (var j=0u;j<p.d;j=j+1u){ dot = dot + gOut[off+j]*y[off+j]; }
  for (var j=0u;j<p.d;j=j+1u){ gIn[off+j] = gIn[off+j] + y[off+j]*(gOut[off+j]-dot); }
}
`;

/** LayerNorm over the last dim: forward writes out, plus per-row mean and rstd. */
const LN_FWD = /* wgsl */ `
struct P { rows:u32, d:u32, eps:f32,_b:u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read_write> mean: array<f32>;
@group(0) @binding(5) var<storage, read_write> rstd: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let r=g.x; if(r>=p.rows){return;}
  let off=r*p.d; let nf=f32(p.d);
  var mu=0.0; for(var j=0u;j<p.d;j=j+1u){ mu=mu+x[off+j]; } mu=mu/nf;
  var v=0.0; for(var j=0u;j<p.d;j=j+1u){ let c=x[off+j]-mu; v=v+c*c; } v=v/nf;
  let rs = 1.0/sqrt(v+p.eps);
  mean[r]=mu; rstd[r]=rs;
  for(var j=0u;j<p.d;j=j+1u){ out[off+j]=((x[off+j]-mu)*rs)*gamma[j]+beta[j]; }
}
`;
const LN_DX = /* wgsl */ `
struct P { rows:u32, d:u32, eps:f32,_b:u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gOut: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> mean: array<f32>;
@group(0) @binding(4) var<storage, read> rstd: array<f32>;
@group(0) @binding(5) var<storage, read_write> gIn: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let r=g.x; if(r>=p.rows){return;}
  let off=r*p.d; let nf=f32(p.d); let mu=mean[r]; let rs=rstd[r];
  var sumDy=0.0; var sumDyXhat=0.0;
  for(var j=0u;j<p.d;j=j+1u){ let dy=gOut[off+j]*gamma[j]; let xh=(x[off+j]-mu)*rs; sumDy=sumDy+dy; sumDyXhat=sumDyXhat+dy*xh; }
  for(var j=0u;j<p.d;j=j+1u){ let dy=gOut[off+j]*gamma[j]; let xh=(x[off+j]-mu)*rs;
    gIn[off+j] = gIn[off+j] + rs*(dy - sumDy/nf - xh*sumDyXhat/nf); }
}
`;
const LN_DGB = /* wgsl */ `
struct P { rows:u32, d:u32, eps:f32,_b:u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gOut: array<f32>;
@group(0) @binding(2) var<storage, read> mean: array<f32>;
@group(0) @binding(3) var<storage, read> rstd: array<f32>;
@group(0) @binding(4) var<storage, read_write> dgamma: array<f32>;
@group(0) @binding(5) var<storage, read_write> dbeta: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let j=g.x; if(j>=p.d){return;}
  var dg=0.0; var db=0.0;
  for(var r=0u;r<p.rows;r=r+1u){ let off=r*p.d; let xh=(x[off+j]-mean[r])*rstd[r]; dg=dg+gOut[off+j]*xh; db=db+gOut[off+j]; }
  dgamma[j]=dgamma[j]+dg; dbeta[j]=dbeta[j]+db;
}
`;

/**
 * RMSNorm over the last dim. Like LN_FWD but with no centering and no beta, so
 * it saves a reduction here and a whole parameter vector in the model.
 * Forward writes out plus the per-row reciprocal RMS, which backward reuses.
 */
const RMS_FWD = /* wgsl */ `
struct P { rows:u32, d:u32, eps:f32,_b:u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read_write> rstd: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let r=g.x; if(r>=p.rows){return;}
  let off=r*p.d; let nf=f32(p.d);
  var ms=0.0; for(var j=0u;j<p.d;j=j+1u){ let v=x[off+j]; ms=ms+v*v; } ms=ms/nf;
  let rs = 1.0/sqrt(ms+p.eps);
  rstd[r]=rs;
  for(var j=0u;j<p.d;j=j+1u){ out[off+j]=x[off+j]*rs*gamma[j]; }
}
`;
// y_j = x_j * rs * g_j with rs = (mean(x^2)+eps)^-1/2, so
//   drs/dx_j = -rs^3 * x_j / d
//   dx_j = rs*g_j*dy_j - (rs^3 * x_j / d) * sum_k dy_k*g_k*x_k
const RMS_DX = /* wgsl */ `
struct P { rows:u32, d:u32, eps:f32,_b:u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gOut: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> rstd: array<f32>;
@group(0) @binding(4) var<storage, read_write> gIn: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let r=g.x; if(r>=p.rows){return;}
  let off=r*p.d; let nf=f32(p.d); let rs=rstd[r];
  var s=0.0;
  for(var j=0u;j<p.d;j=j+1u){ s = s + gOut[off+j]*gamma[j]*x[off+j]; }
  let k = rs*rs*rs*s/nf;
  for(var j=0u;j<p.d;j=j+1u){
    gIn[off+j] = gIn[off+j] + rs*gOut[off+j]*gamma[j] - k*x[off+j]; }
}
`;
const RMS_DG = /* wgsl */ `
struct P { rows:u32, d:u32,_a:u32,_b:u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gOut: array<f32>;
@group(0) @binding(2) var<storage, read> rstd: array<f32>;
@group(0) @binding(3) var<storage, read_write> dgamma: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let j=g.x; if(j>=p.d){return;}
  var dg=0.0;
  for(var r=0u;r<p.rows;r=r+1u){ let off=r*p.d; dg = dg + gOut[off+j]*x[off+j]*rstd[r]; }
  dgamma[j]=dgamma[j]+dg;
}
`;

/**
 * Rotary position embedding over [groups, time, headDim]: rotates each adjacent
 * dimension pair by an angle set by its position. One thread per pair. The
 * angles are recomputed from the index rather than read from a table — cheaper
 * than a buffer round trip, and it keeps the op stateless.
 */
const ROPE = /* wgsl */ `
struct P { groups:u32, t:u32, hd:u32, base:f32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>){
  let halfd = p.hd/2u;
  let i = g.x + g.y*8388608u; if(i >= p.groups*p.t*halfd){return;}
  let pair = i % halfd;
  let pos = (i/halfd) % p.t;
  let grp = i/(halfd*p.t);
  let off = (grp*p.t + pos)*p.hd + pair*2u;
  let theta = f32(pos) / pow(p.base, f32(2u*pair)/f32(p.hd));
  let c = cos(theta); let s = sin(theta);
  let x0 = x[off]; let x1 = x[off+1u];
  out[off] = x0*c - x1*s;
  out[off+1u] = x0*s + x1*c;
}
`;
// Rotation is orthogonal, so the vector-Jacobian product is the inverse rotation.
const ROPE_BWD = /* wgsl */ `
struct P { groups:u32, t:u32, hd:u32, base:f32 };
@group(0) @binding(0) var<storage, read> gOut: array<f32>;
@group(0) @binding(1) var<storage, read_write> gIn: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>){
  let halfd = p.hd/2u;
  let i = g.x + g.y*8388608u; if(i >= p.groups*p.t*halfd){return;}
  let pair = i % halfd;
  let pos = (i/halfd) % p.t;
  let grp = i/(halfd*p.t);
  let off = (grp*p.t + pos)*p.hd + pair*2u;
  let theta = f32(pos) / pow(p.base, f32(2u*pair)/f32(p.hd));
  let c = cos(theta); let s = sin(theta);
  let g0 = gOut[off]; let g1 = gOut[off+1u];
  gIn[off] = gIn[off] + g0*c + g1*s;
  gIn[off+1u] = gIn[off+1u] - g0*s + g1*c;
}
`;

/** Embedding gather: out[n,:] = table[idx[n],:]. */
const GATHER = /* wgsl */ `
struct P { n:u32, d:u32,_a:u32,_b:u32 };
@group(0) @binding(0) var<storage, read> table: array<f32>;
@group(0) @binding(1) var<storage, read> idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let i=g.x+g.y*8388608u; if(i>=p.n*p.d){return;}
  let row=i/p.d; let col=i%p.d; out[i]=table[idx[row]*p.d+col]; }
`;
// One thread per vocab row scans all positions (cheap: V*N comparisons) and
// accumulates D features only on a match — no atomics, total feature-work N*D.
const GATHER_BWD = /* wgsl */ `
struct P { n:u32, d:u32, vocab:u32,_b:u32 };
@group(0) @binding(0) var<storage, read> idx: array<u32>;
@group(0) @binding(1) var<storage, read> gOut: array<f32>;
@group(0) @binding(2) var<storage, read_write> dTable: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let v=g.x; if(v>=p.vocab){return;}
  for(var i=0u;i<p.n;i=i+1u){ if(idx[i]==v){ for(var c=0u;c<p.d;c=c+1u){ dTable[v*p.d+c]=dTable[v*p.d+c]+gOut[i*p.d+c]; } } }
}
`;

/** Fused softmax cross-entropy: per-row loss, then reduced to a scalar. */
const CE_FWD = /* wgsl */ `
struct P { rows:u32, vocab:u32,_a:u32,_b:u32 };
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> perRow: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let r=g.x; if(r>=p.rows){return;}
  let off=r*p.vocab; var m=logits[off];
  for(var j=1u;j<p.vocab;j=j+1u){ m=max(m,logits[off+j]); }
  var s=0.0; for(var j=0u;j<p.vocab;j=j+1u){ s=s+exp(logits[off+j]-m); }
  perRow[r] = m + log(s) - logits[off+targets[r]];
}
`;
const CE_BWD = /* wgsl */ `
struct P { rows:u32, vocab:u32,_a:u32,_b:u32 };
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read> gLoss: array<f32>;
@group(0) @binding(3) var<storage, read_write> dLogits: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let r=g.x; if(r>=p.rows){return;}
  let off=r*p.vocab; var m=logits[off];
  for(var j=1u;j<p.vocab;j=j+1u){ m=max(m,logits[off+j]); }
  var s=0.0; for(var j=0u;j<p.vocab;j=j+1u){ s=s+exp(logits[off+j]-m); }
  let scale = gLoss[0]/f32(p.rows);
  for(var j=0u;j<p.vocab;j=j+1u){ let pr=exp(logits[off+j]-m)/s; dLogits[off+j]=dLogits[off+j]+scale*pr; }
  dLogits[off+targets[r]] = dLogits[off+targets[r]] - scale;
}
`;

/** Causal mask: set scores above the diagonal of each [T,T] block to -1e9. */
const MASK = /* wgsl */ `
struct P { groups:u32, t:u32,_a:u32,_b:u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let i=g.x+g.y*8388608u; if(i>=p.groups*p.t*p.t){return;}
  let within=i%(p.t*p.t); let row=within/p.t; let col=within%p.t;
  if(col>row){ out[i]=-1e9; } else { out[i]=x[i]; } }
`;
const MASK_BWD = /* wgsl */ `
struct P { groups:u32, t:u32,_a:u32,_b:u32 };
@group(0) @binding(0) var<storage, read> gOut: array<f32>;
@group(0) @binding(1) var<storage, read_write> gIn: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let i=g.x+g.y*8388608u; if(i>=p.groups*p.t*p.t){return;}
  let within=i%(p.t*p.t); let row=within/p.t; let col=within%p.t;
  if(col<=row){ gIn[i]=gIn[i]+gOut[i]; } }
`;

/** 4-D permute (used for attention head reshapes). perm[i] = source axis of out axis i. */
const PERMUTE4 = /* wgsl */ `
struct P { s0:u32,s1:u32,s2:u32,s3:u32, p0:u32,p1:u32,p2:u32,p3:u32, accum:u32, n:u32,_a:u32,_b:u32 };
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let o=g.x+g.y*8388608u; if(o>=p.n){return;}
  var inShape = array<u32,4>(p.s0,p.s1,p.s2,p.s3);
  var perm = array<u32,4>(p.p0,p.p1,p.p2,p.p3);
  var outShape = array<u32,4>(inShape[perm[0]],inShape[perm[1]],inShape[perm[2]],inShape[perm[3]]);
  // in strides (row-major)
  var inStride = array<u32,4>(inShape[1]*inShape[2]*inShape[3], inShape[2]*inShape[3], inShape[3], 1u);
  var outStride = array<u32,4>(outShape[1]*outShape[2]*outShape[3], outShape[2]*outShape[3], outShape[3], 1u);
  var rem = o; var inIdx = 0u;
  for (var ax=0u; ax<4u; ax=ax+1u){ let c = rem/outStride[ax]; rem = rem - c*outStride[ax]; inIdx = inIdx + c*inStride[perm[ax]]; }
  if (p.accum==0u){ out[o]=inp[inIdx]; } else { out[inIdx]=out[inIdx]+inp[o]; }
}
`;

/** AdamW update with resident m/v state. */
const ADAMW = /* wgsl */ `
struct P { n:u32,_a:u32,_b:u32,_c:u32, lr:f32, beta1:f32, beta2:f32, eps:f32, wd:f32, bc1:f32, bc2:f32,_d:f32 };
@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> param: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let i=g.x+g.y*8388608u; if(i>=p.n){return;}
  let gr = grad[i];
  m[i] = p.beta1*m[i] + (1.0-p.beta1)*gr;
  v[i] = p.beta2*v[i] + (1.0-p.beta2)*gr*gr;
  let mh = m[i]/p.bc1; let vh = v[i]/p.bc2;
  param[i] = param[i] - p.lr*(mh/(sqrt(vh)+p.eps) + p.wd*param[i]);
}
`;

interface Ctx {
  parents: GpuTensor[];
  backward: () => void;
}

const wg = (n: number): number => Math.ceil(n / 256);

export class GpuEngine {
  private pipelines = new Map<string, GPUComputePipeline>();
  private constructor(readonly device: GPUDevice) {}

  static async create(device?: GPUDevice): Promise<GpuEngine> {
    let dev = device;
    if (!dev) {
      const nav = (globalThis as unknown as { navigator?: { gpu?: { requestAdapter(): Promise<GPUAdapter | null> } } })
        .navigator;
      if (!nav?.gpu) throw new Error("No WebGPU device available");
      const adapter = await nav.gpu.requestAdapter();
      if (!adapter) throw new Error("No WebGPU adapter available");
      // Request the adapter's full limits. The default device grants only the
      // WebGPU baseline (128MB storage binding / 256MB buffer), which caps the
      // model size well below what the hardware allows; on Apple Silicon and
      // cloud NVIDIA these are gigabytes. Bump the four limits that gate GPT
      // scale: big weight matrices, the [N, vocab] logits, and the AdamW state.
      const al = adapter.limits;
      const requiredLimits: Record<string, number> = {
        maxBufferSize: al.maxBufferSize,
        maxStorageBufferBindingSize: al.maxStorageBufferBindingSize,
        maxComputeWorkgroupsPerDimension: al.maxComputeWorkgroupsPerDimension,
        maxComputeInvocationsPerWorkgroup: al.maxComputeInvocationsPerWorkgroup,
      };
      dev = await adapter.requestDevice({ requiredLimits });
    }
    return new GpuEngine(dev);
  }

  private pipeline(code: string): GPUComputePipeline {
    let p = this.pipelines.get(code);
    if (!p) {
      const module = this.device.createShaderModule({ code });
      p = this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
      this.pipelines.set(code, p);
    }
    return p;
  }

  buffer(elems: number): GPUBuffer {
    return this.device.createBuffer({
      size: Math.max(4, elems * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }

  private run(code: string, storages: GPUBuffer[], uniform: ArrayBufferView, gx: number, gy = 1, gz = 1): void {
    // Vulkan caps workgroups at 65535 per dimension (Metal is far higher), so
    // large flat dispatches fold into fixed 32768-wide rows; the flat kernels
    // add g.y*8388608 (= 32768 workgroups * 256 threads) to their index. Only
    // 1D dispatches can trip this: row-per-thread kernels stay orders of
    // magnitude below the cap, and matmul grids are bounded by dims/TILE.
    if (gy === 1 && gz === 1 && gx > 65535) {
      gy = Math.ceil(gx / 32768);
      gx = 32768;
    }
    const pipeline = this.pipeline(code);
    const ubuf = this.device.createBuffer({
      size: Math.max(16, uniform.byteLength),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(ubuf, 0, uniform.buffer, uniform.byteOffset, uniform.byteLength);
    const entries: GPUBindGroupEntry[] = storages.map((b, i) => ({ binding: i, resource: { buffer: b } }));
    entries.push({ binding: storages.length, resource: { buffer: ubuf } });
    const bind = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(gx, gy, gz);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    // Destroy the uniform now that it is submitted (pending work keeps it
    // alive). Vulkan drivers cap live memory allocations (~4k); leaving one
    // uniform per dispatch to the JS GC blows past that cap within a few
    // training steps on wgpu.
    ubuf.destroy();
  }

  matmulInto(
    a: GPUBuffer, b: GPUBuffer, out: GPUBuffer,
    M: number, K: number, N: number, batch: number,
    transA: boolean, transB: boolean, accum: boolean,
  ): void {
    const u = new Uint32Array([M, K, N, batch, transA ? 1 : 0, transB ? 1 : 0, accum ? 1 : 0, 0]);
    this.run(MATMUL, [a, b, out], u, Math.ceil(N / TILE), Math.ceil(M / TILE), batch);
  }
  ewise(a: GPUBuffer, b: GPUBuffer, out: GPUBuffer, n: number, bLen: number, op: number, accum: boolean): void {
    this.run(EWISE, [a, b, out], new Uint32Array([n, bLen, op, accum ? 1 : 0]), wg(n));
  }
  scale(a: GPUBuffer, out: GPUBuffer, n: number, s: number, accum: boolean): void {
    const u = new ArrayBuffer(32); new Uint32Array(u, 0, 2).set([n, accum ? 1 : 0]); new Float32Array(u, 16, 1)[0] = s;
    this.run(SCALE, [a, out], new Uint8Array(u), wg(n));
  }
  bcastAcc(src: GPUBuffer, out: GPUBuffer, n: number, bLen: number, s: number): void {
    const u = new ArrayBuffer(32); new Uint32Array(u, 0, 2).set([n, bLen]); new Float32Array(u, 16, 1)[0] = s;
    this.run(BCAST_ACC, [src, out], new Uint8Array(u), wg(n));
  }
  segsum(g: GPUBuffer, out: GPUBuffer, n: number, bLen: number, s: number, accum: boolean): void {
    const u = new ArrayBuffer(32); new Uint32Array(u, 0, 3).set([n, bLen, accum ? 1 : 0]); new Float32Array(u, 16, 1)[0] = s;
    this.run(SEGSUM, [g, out], new Uint8Array(u), wg(bLen));
  }
  total(a: GPUBuffer, out: GPUBuffer, n: number): void {
    this.run(TOTAL, [a, out], new Uint32Array([n, 0, 0, 0]), 1);
  }
  reluInto(x: GPUBuffer, out: GPUBuffer, n: number): void {
    this.run(RELU, [x, out], new Uint32Array([n, 0, 0, 0]), wg(n));
  }
  reluBwd(x: GPUBuffer, gOut: GPUBuffer, gIn: GPUBuffer, n: number): void {
    this.run(RELU_BWD, [x, gOut, gIn], new Uint32Array([n, 0, 0, 0]), wg(n));
  }
  sgd(grad: GPUBuffer, param: GPUBuffer, n: number, lr: number): void {
    const u = new ArrayBuffer(32); new Uint32Array(u, 0, 1)[0] = n; new Float32Array(u, 16, 1)[0] = lr;
    this.run(SGD, [grad, param], new Uint8Array(u), wg(n));
  }

  indexBuffer(data: Uint32Array): GPUBuffer {
    const buf = this.device.createBuffer({
      size: Math.max(4, data.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buf, 0, data);
    return buf;
  }

  geluInto(x: GPUBuffer, out: GPUBuffer, n: number): void {
    this.run(GELU, [x, out], new Uint32Array([n, 0, 0, 0]), wg(n));
  }
  geluBwd(x: GPUBuffer, gOut: GPUBuffer, gIn: GPUBuffer, n: number): void {
    this.run(GELU_BWD, [x, gOut, gIn], new Uint32Array([n, 0, 0, 0]), wg(n));
  }
  siluInto(x: GPUBuffer, out: GPUBuffer, n: number): void {
    this.run(SILU, [x, out], new Uint32Array([n, 0, 0, 0]), wg(n));
  }
  siluBwd(x: GPUBuffer, gOut: GPUBuffer, gIn: GPUBuffer, n: number): void {
    this.run(SILU_BWD, [x, gOut, gIn], new Uint32Array([n, 0, 0, 0]), wg(n));
  }
  rmsFwd(x: GPUBuffer, gamma: GPUBuffer, out: GPUBuffer, rstd: GPUBuffer, rows: number, d: number, eps: number): void {
    const u = new ArrayBuffer(16); new Uint32Array(u, 0, 2).set([rows, d]); new Float32Array(u, 8, 1)[0] = eps;
    this.run(RMS_FWD, [x, gamma, out, rstd], new Uint8Array(u), Math.ceil(rows / 64));
  }
  rmsDx(x: GPUBuffer, gOut: GPUBuffer, gamma: GPUBuffer, rstd: GPUBuffer, gIn: GPUBuffer, rows: number, d: number, eps: number): void {
    const u = new ArrayBuffer(16); new Uint32Array(u, 0, 2).set([rows, d]); new Float32Array(u, 8, 1)[0] = eps;
    this.run(RMS_DX, [x, gOut, gamma, rstd, gIn], new Uint8Array(u), Math.ceil(rows / 64));
  }
  rmsDg(x: GPUBuffer, gOut: GPUBuffer, rstd: GPUBuffer, dgamma: GPUBuffer, rows: number, d: number): void {
    const u = new ArrayBuffer(16); new Uint32Array(u, 0, 2).set([rows, d]);
    this.run(RMS_DG, [x, gOut, rstd, dgamma], new Uint8Array(u), Math.ceil(d / 64));
  }
  ropeInto(x: GPUBuffer, out: GPUBuffer, groups: number, t: number, hd: number, base: number): void {
    const u = new ArrayBuffer(16); new Uint32Array(u, 0, 3).set([groups, t, hd]); new Float32Array(u, 12, 1)[0] = base;
    this.run(ROPE, [x, out], new Uint8Array(u), wg(groups * t * (hd / 2)));
  }
  ropeBwd(gOut: GPUBuffer, gIn: GPUBuffer, groups: number, t: number, hd: number, base: number): void {
    const u = new ArrayBuffer(16); new Uint32Array(u, 0, 3).set([groups, t, hd]); new Float32Array(u, 12, 1)[0] = base;
    this.run(ROPE_BWD, [gOut, gIn], new Uint8Array(u), wg(groups * t * (hd / 2)));
  }
  softmaxInto(x: GPUBuffer, out: GPUBuffer, rows: number, d: number): void {
    this.run(SOFTMAX, [x, out], new Uint32Array([rows, d, 0, 0]), Math.ceil(rows / 64));
  }
  softmaxBwd(y: GPUBuffer, gOut: GPUBuffer, gIn: GPUBuffer, rows: number, d: number): void {
    this.run(SOFTMAX_BWD, [y, gOut, gIn], new Uint32Array([rows, d, 0, 0]), Math.ceil(rows / 64));
  }
  lnFwd(x: GPUBuffer, gamma: GPUBuffer, beta: GPUBuffer, out: GPUBuffer, mean: GPUBuffer, rstd: GPUBuffer, rows: number, d: number, eps: number): void {
    const u = new ArrayBuffer(16); new Uint32Array(u, 0, 2).set([rows, d]); new Float32Array(u, 8, 1)[0] = eps;
    this.run(LN_FWD, [x, gamma, beta, out, mean, rstd], new Uint8Array(u), Math.ceil(rows / 64));
  }
  lnDx(x: GPUBuffer, gOut: GPUBuffer, gamma: GPUBuffer, mean: GPUBuffer, rstd: GPUBuffer, gIn: GPUBuffer, rows: number, d: number, eps: number): void {
    const u = new ArrayBuffer(16); new Uint32Array(u, 0, 2).set([rows, d]); new Float32Array(u, 8, 1)[0] = eps;
    this.run(LN_DX, [x, gOut, gamma, mean, rstd, gIn], new Uint8Array(u), Math.ceil(rows / 64));
  }
  lnDgb(x: GPUBuffer, gOut: GPUBuffer, mean: GPUBuffer, rstd: GPUBuffer, dgamma: GPUBuffer, dbeta: GPUBuffer, rows: number, d: number): void {
    const u = new ArrayBuffer(16); new Uint32Array(u, 0, 2).set([rows, d]);
    this.run(LN_DGB, [x, gOut, mean, rstd, dgamma, dbeta], new Uint8Array(u), Math.ceil(d / 64));
  }
  gatherInto(table: GPUBuffer, idx: GPUBuffer, out: GPUBuffer, n: number, d: number): void {
    this.run(GATHER, [table, idx, out], new Uint32Array([n, d, 0, 0]), wg(n * d));
  }
  gatherBwd(idx: GPUBuffer, gOut: GPUBuffer, dTable: GPUBuffer, n: number, d: number, vocab: number): void {
    this.run(GATHER_BWD, [idx, gOut, dTable], new Uint32Array([n, d, vocab, 0]), Math.ceil(vocab / 64));
  }
  ceFwd(logits: GPUBuffer, targets: GPUBuffer, perRow: GPUBuffer, rows: number, vocab: number): void {
    this.run(CE_FWD, [logits, targets, perRow], new Uint32Array([rows, vocab, 0, 0]), Math.ceil(rows / 64));
  }
  ceBwd(logits: GPUBuffer, targets: GPUBuffer, gLoss: GPUBuffer, dLogits: GPUBuffer, rows: number, vocab: number): void {
    this.run(CE_BWD, [logits, targets, gLoss, dLogits], new Uint32Array([rows, vocab, 0, 0]), Math.ceil(rows / 64));
  }
  maskInto(x: GPUBuffer, out: GPUBuffer, groups: number, t: number): void {
    this.run(MASK, [x, out], new Uint32Array([groups, t, 0, 0]), wg(groups * t * t));
  }
  maskBwd(gOut: GPUBuffer, gIn: GPUBuffer, groups: number, t: number): void {
    this.run(MASK_BWD, [gOut, gIn], new Uint32Array([groups, t, 0, 0]), wg(groups * t * t));
  }
  permute4(inp: GPUBuffer, out: GPUBuffer, inShape: number[], perm: number[], n: number, accum: boolean): void {
    const u = new Uint32Array([inShape[0], inShape[1], inShape[2], inShape[3], perm[0], perm[1], perm[2], perm[3], accum ? 1 : 0, n, 0, 0]);
    this.run(PERMUTE4, [inp, out], u, wg(n));
  }
  adamw(grad: GPUBuffer, param: GPUBuffer, m: GPUBuffer, v: GPUBuffer, n: number, lr: number, b1: number, b2: number, eps: number, wd: number, bc1: number, bc2: number): void {
    const u = new ArrayBuffer(64);
    new Uint32Array(u, 0, 1)[0] = n;
    new Float32Array(u, 16, 8).set([lr, b1, b2, eps, wd, bc1, bc2, 0]);
    this.run(ADAMW, [grad, param, m, v], new Uint8Array(u), wg(n));
  }

  tensor(data: Float32Array, shape: number[], requiresGrad = false): GpuTensor {
    const buf = this.buffer(data.length);
    this.device.queue.writeBuffer(buf, 0, data);
    return new GpuTensor(this, buf, [...shape], requiresGrad);
  }
  zeros(shape: number[], requiresGrad = false): GpuTensor {
    return new GpuTensor(this, this.buffer(shape.reduce((a, b) => a * b, 1)), [...shape], requiresGrad);
  }

  async read(buf: GPUBuffer, elems: number): Promise<Float32Array> {
    const staging = this.device.createBuffer({ size: elems * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, elems * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }
  fill(buf: GPUBuffer, value: number, elems: number): void {
    this.device.queue.writeBuffer(buf, 0, new Float32Array(elems).fill(value));
  }
}

export class GpuTensor {
  grad: GPUBuffer | null = null;
  ctx: Ctx | null = null;

  constructor(
    readonly engine: GpuEngine,
    readonly buffer: GPUBuffer,
    readonly shape: number[],
    public requiresGrad = false,
  ) {}

  get size(): number { return this.shape.reduce((a, b) => a * b, 1); }
  get ndim(): number { return this.shape.length; }

  ensureGrad(): GPUBuffer {
    this.grad ??= this.engine.buffer(this.size);
    return this.grad;
  }
  zeroGrad(): void {
    this.grad?.destroy();
    this.grad = null;
  }
  /**
   * Destroy every non-leaf buffer (activations and their grads) in this
   * tensor's autograd graph. Call once the step's reads are done: a training
   * step allocates hundreds of intermediates, and waiting for the JS GC to
   * release them exceeds the driver's live-allocation cap (~4k on Vulkan)
   * within a few steps. Leaves — params and raw inputs — are untouched.
   */
  freeGraph(): void {
    const stack: GpuTensor[] = [this];
    const seen = new Set<GpuTensor>();
    while (stack.length) {
      const t = stack.pop()!;
      if (seen.has(t) || !t.ctx) continue;
      seen.add(t);
      for (const p of t.ctx.parents) stack.push(p);
      t.ctx = null;
      t.buffer.destroy();
      if (t.grad) {
        t.grad.destroy();
        t.grad = null;
      }
    }
  }

  private out(buffer: GPUBuffer, shape: number[], parents: GpuTensor[], backward: () => void): GpuTensor {
    const t = new GpuTensor(this.engine, buffer, shape, parents.some((p) => p.requiresGrad));
    if (t.requiresGrad) t.ctx = { parents, backward };
    return t;
  }

  matmul(b: GpuTensor): GpuTensor {
    const [M, K] = this.shape;
    const [K2, N] = b.shape;
    if (K !== K2) throw new Error(`matmul shape mismatch: [${this.shape}] @ [${b.shape}]`);
    const e = this.engine;
    const outBuf = e.buffer(M * N);
    e.matmulInto(this.buffer, b.buffer, outBuf, M, K, N, 1, false, false, false);
    const a = this;
    const t = this.out(outBuf, [M, N], [a, b], () => {
      const gC = t.grad!;
      if (a.requiresGrad) e.matmulInto(gC, b.buffer, a.ensureGrad(), M, N, K, 1, false, true, true);
      if (b.requiresGrad) e.matmulInto(a.buffer, gC, b.ensureGrad(), K, M, N, 1, true, false, true);
    });
    return t;
  }

  private binary(b: GpuTensor, op: 0 | 1 | 2): GpuTensor {
    const e = this.engine;
    const n = this.size;
    const bLen = b.size;
    const outBuf = e.buffer(n);
    e.ewise(this.buffer, b.buffer, outBuf, n, bLen, op, false);
    const a = this;
    const t = this.out(outBuf, [...this.shape], [a, b], () => {
      const gC = t.grad!;
      if (a.requiresGrad) {
        // add/sub: dA += gC ; mul: dA += gC * b
        if (op === 2) e.ewise(gC, b.buffer, a.ensureGrad(), n, bLen, 2, true);
        else e.scale(gC, a.ensureGrad(), n, 1, true);
      }
      if (b.requiresGrad) {
        const sign = op === 1 ? -1 : 1;
        if (op === 2) {
          if (bLen === n) {
            // Same-shape mul (SwiGLU's gate): no broadcast, so every segment has
            // exactly one element — write dB += gC * a straight through instead
            // of paying a temp buffer and a degenerate segsum.
            e.ewise(gC, a.buffer, b.ensureGrad(), n, n, 2, true);
          } else {
            // dB += segsum(gC * a)
            const tmp = e.buffer(n);
            e.ewise(gC, a.buffer, tmp, n, n, 2, false);
            e.segsum(tmp, b.ensureGrad(), n, bLen, 1, true);
            tmp.destroy();
          }
        } else {
          e.segsum(gC, b.ensureGrad(), n, bLen, sign, true);
        }
      }
    });
    return t;
  }
  add(b: GpuTensor): GpuTensor { return this.binary(b, 0); }
  sub(b: GpuTensor): GpuTensor { return this.binary(b, 1); }
  mul(b: GpuTensor): GpuTensor { return this.binary(b, 2); }

  scale(s: number): GpuTensor {
    const e = this.engine;
    const n = this.size;
    const outBuf = e.buffer(n);
    e.scale(this.buffer, outBuf, n, s, false);
    const a = this;
    const t = this.out(outBuf, [...this.shape], [a], () => {
      if (a.requiresGrad) e.scale(t.grad!, a.ensureGrad(), n, s, true);
    });
    return t;
  }

  relu(): GpuTensor {
    const e = this.engine;
    const n = this.size;
    const outBuf = e.buffer(n);
    e.reluInto(this.buffer, outBuf, n);
    const a = this;
    const t = this.out(outBuf, [...this.shape], [a], () => {
      if (a.requiresGrad) e.reluBwd(a.buffer, t.grad!, a.ensureGrad(), n);
    });
    return t;
  }

  sum(): GpuTensor {
    const e = this.engine;
    const n = this.size;
    const outBuf = e.buffer(1);
    e.total(this.buffer, outBuf, n);
    const a = this;
    const t = this.out(outBuf, [1], [a], () => {
      if (a.requiresGrad) e.bcastAcc(t.grad!, a.ensureGrad(), n, 1, 1);
    });
    return t;
  }
  mean(): GpuTensor {
    return this.sum().scale(1 / this.size);
  }

  /** Batched matmul: [g,m,k] @ [g,k,n] -> [g,m,n]. */
  bmm(b: GpuTensor): GpuTensor {
    const [G, M, K] = this.shape;
    const [G2, K2, N] = b.shape;
    if (G !== G2 || K !== K2) throw new Error(`bmm shape mismatch: [${this.shape}] @ [${b.shape}]`);
    const e = this.engine;
    const outBuf = e.buffer(G * M * N);
    e.matmulInto(this.buffer, b.buffer, outBuf, M, K, N, G, false, false, false);
    const a = this;
    const t = this.out(outBuf, [G, M, N], [a, b], () => {
      const gC = t.grad!;
      if (a.requiresGrad) e.matmulInto(gC, b.buffer, a.ensureGrad(), M, N, K, G, false, true, true);
      if (b.requiresGrad) e.matmulInto(a.buffer, gC, b.ensureGrad(), K, M, N, G, true, false, true);
    });
    return t;
  }

  /** Batched q @ k^T: [g,m,k] @ [g,n,k]^T -> [g,m,n]. Attention scores without transposing k. */
  bmmBT(k: GpuTensor): GpuTensor {
    const [G, M, K] = this.shape;
    const [G2, N, K2] = k.shape;
    if (G !== G2 || K !== K2) throw new Error(`bmmBT shape mismatch: [${this.shape}] @ [${k.shape}]^T`);
    const e = this.engine;
    const outBuf = e.buffer(G * M * N);
    e.matmulInto(this.buffer, k.buffer, outBuf, M, K, N, G, false, true, false);
    const a = this;
    const t = this.out(outBuf, [G, M, N], [a, k], () => {
      const gS = t.grad!;
      // dQ += dScores @ k ; dK += dScores^T @ q
      if (a.requiresGrad) e.matmulInto(gS, k.buffer, a.ensureGrad(), M, N, K, G, false, false, true);
      if (k.requiresGrad) e.matmulInto(gS, a.buffer, k.ensureGrad(), N, M, K, G, true, false, true);
    });
    return t;
  }

  /** this[N,D] @ other[V,D]^T -> [N,V]. Used for the weight-tied output head. */
  matmulBT(b: GpuTensor): GpuTensor {
    const [Nrows, D] = this.shape;
    const [V, D2] = b.shape;
    if (D !== D2) throw new Error(`matmulBT shape mismatch: [${this.shape}] @ [${b.shape}]^T`);
    const e = this.engine;
    const outBuf = e.buffer(Nrows * V);
    e.matmulInto(this.buffer, b.buffer, outBuf, Nrows, D, V, 1, false, true, false);
    const a = this;
    const t = this.out(outBuf, [Nrows, V], [a, b], () => {
      const gC = t.grad!;
      if (a.requiresGrad) e.matmulInto(gC, b.buffer, a.ensureGrad(), Nrows, V, D, 1, false, false, true);
      if (b.requiresGrad) e.matmulInto(gC, a.buffer, b.ensureGrad(), V, Nrows, D, 1, true, false, true);
    });
    return t;
  }

  /** Reshape (contiguous, shares the data buffer). */
  reshape(shape: number[]): GpuTensor {
    const e = this.engine;
    const a = this;
    const t = new GpuTensor(e, this.buffer, [...shape], this.requiresGrad);
    if (this.requiresGrad) {
      t.ctx = {
        parents: [a],
        backward: () => e.scale(t.grad!, a.ensureGrad(), a.size, 1, true),
      };
    }
    return t;
  }

  gelu(): GpuTensor {
    const e = this.engine;
    const n = this.size;
    const outBuf = e.buffer(n);
    e.geluInto(this.buffer, outBuf, n);
    const a = this;
    const t = this.out(outBuf, [...this.shape], [a], () => {
      if (a.requiresGrad) e.geluBwd(a.buffer, t.grad!, a.ensureGrad(), n);
    });
    return t;
  }

  /** SiLU / swish: x * sigmoid(x). */
  silu(): GpuTensor {
    const e = this.engine;
    const n = this.size;
    const outBuf = e.buffer(n);
    e.siluInto(this.buffer, outBuf, n);
    const a = this;
    const t = this.out(outBuf, [...this.shape], [a], () => {
      if (a.requiresGrad) e.siluBwd(a.buffer, t.grad!, a.ensureGrad(), n);
    });
    return t;
  }

  /** RMSNorm over the last dimension with a learnable gain (no bias). */
  rmsNorm(gamma: GpuTensor, eps = 1e-5): GpuTensor {
    const e = this.engine;
    const d = this.shape[this.ndim - 1];
    const rows = this.size / d;
    const outBuf = e.buffer(this.size);
    const rstd = e.buffer(rows);
    e.rmsFwd(this.buffer, gamma.buffer, outBuf, rstd, rows, d, eps);
    const a = this;
    const t = this.out(outBuf, [...this.shape], [a, gamma], () => {
      const gC = t.grad!;
      if (a.requiresGrad) e.rmsDx(a.buffer, gC, gamma.buffer, rstd, a.ensureGrad(), rows, d, eps);
      if (gamma.requiresGrad) e.rmsDg(a.buffer, gC, rstd, gamma.ensureGrad(), rows, d);
    });
    return t;
  }

  /** Rotary position embedding over a [groups, time, headDim] tensor. */
  rope(base = 10000): GpuTensor {
    if (this.ndim !== 3) throw new Error(`rope expects a [groups, time, headDim] tensor, got ${this.ndim}-D`);
    const [groups, tt, hd] = this.shape;
    if (hd % 2 !== 0) throw new Error(`rope requires an even headDim, got ${hd}`);
    const e = this.engine;
    const outBuf = e.buffer(this.size);
    e.ropeInto(this.buffer, outBuf, groups, tt, hd, base);
    const a = this;
    const t = this.out(outBuf, [...this.shape], [a], () => {
      if (a.requiresGrad) e.ropeBwd(t.grad!, a.ensureGrad(), groups, tt, hd, base);
    });
    return t;
  }

  softmax(): GpuTensor {
    const e = this.engine;
    const d = this.shape[this.ndim - 1];
    const rows = this.size / d;
    const outBuf = e.buffer(this.size);
    e.softmaxInto(this.buffer, outBuf, rows, d);
    const a = this;
    const t = this.out(outBuf, [...this.shape], [a], () => {
      if (a.requiresGrad) e.softmaxBwd(outBuf, t.grad!, a.ensureGrad(), rows, d);
    });
    return t;
  }

  maskedFillCausal(): GpuTensor {
    const e = this.engine;
    const t2 = this.shape[this.ndim - 1];
    const groups = this.size / (t2 * t2);
    const outBuf = e.buffer(this.size);
    e.maskInto(this.buffer, outBuf, groups, t2);
    const a = this;
    const t = this.out(outBuf, [...this.shape], [a], () => {
      if (a.requiresGrad) e.maskBwd(t.grad!, a.ensureGrad(), groups, t2);
    });
    return t;
  }

  /** 4-D axis permutation. perm[i] is the source axis of output axis i. */
  permute(perm: number[]): GpuTensor {
    if (this.ndim !== 4) throw new Error("GpuTensor.permute currently supports 4-D tensors");
    const e = this.engine;
    const inShape = this.shape;
    const outShape = perm.map((p) => inShape[p]);
    const n = this.size;
    const outBuf = e.buffer(n);
    e.permute4(this.buffer, outBuf, inShape, perm, n, false);
    const a = this;
    const t = this.out(outBuf, outShape, [a], () => {
      // backward scatters gOut back through the same permutation (accumulate).
      if (a.requiresGrad) e.permute4(t.grad!, a.ensureGrad(), inShape, perm, n, true);
    });
    return t;
  }

  /** LayerNorm over the last dimension with learnable gamma/beta. */
  layerNorm(gamma: GpuTensor, beta: GpuTensor, eps = 1e-5): GpuTensor {
    const e = this.engine;
    const d = this.shape[this.ndim - 1];
    const rows = this.size / d;
    const outBuf = e.buffer(this.size);
    const mean = e.buffer(rows);
    const rstd = e.buffer(rows);
    e.lnFwd(this.buffer, gamma.buffer, beta.buffer, outBuf, mean, rstd, rows, d, eps);
    const a = this;
    const t = this.out(outBuf, [...this.shape], [a, gamma, beta], () => {
      const gC = t.grad!;
      if (a.requiresGrad) e.lnDx(a.buffer, gC, gamma.buffer, mean, rstd, a.ensureGrad(), rows, d, eps);
      if (gamma.requiresGrad) e.lnDgb(a.buffer, gC, mean, rstd, gamma.ensureGrad(), beta.ensureGrad(), rows, d);
    });
    return t;
  }

  /** Embedding lookup: this is the [vocab, d] table; `indices` selects rows. */
  gatherRows(indices: Uint32Array): GpuTensor {
    const e = this.engine;
    const [vocab, d] = this.shape;
    const n = indices.length;
    const idxBuf = e.indexBuffer(indices);
    const outBuf = e.buffer(n * d);
    e.gatherInto(this.buffer, idxBuf, outBuf, n, d);
    const a = this;
    const t = this.out(outBuf, [n, d], [a], () => {
      if (a.requiresGrad) e.gatherBwd(idxBuf, t.grad!, a.ensureGrad(), n, d, vocab);
    });
    return t;
  }

  /** Fused softmax cross-entropy over logits [N, vocab] against integer targets. */
  crossEntropyLogits(targets: Uint32Array): GpuTensor {
    const e = this.engine;
    const [rows, vocab] = this.shape;
    const tgtBuf = e.indexBuffer(targets);
    const perRow = e.buffer(rows);
    e.ceFwd(this.buffer, tgtBuf, perRow, rows, vocab);
    // total() and scale() must use distinct buffers: binding one buffer as both
    // a read and a read_write resource is an aliasing hazard in WebGPU.
    const sumBuf = e.buffer(1);
    e.total(perRow, sumBuf, rows);
    const lossBuf = e.buffer(1);
    e.scale(sumBuf, lossBuf, 1, 1 / rows, false);
    // Forward-only scratch; the backward closure needs tgtBuf but not these.
    perRow.destroy();
    sumBuf.destroy();
    const a = this;
    const t = this.out(lossBuf, [1], [a], () => {
      if (a.requiresGrad) e.ceBwd(a.buffer, tgtBuf, t.grad!, a.ensureGrad(), rows, vocab);
    });
    return t;
  }

  /** Reverse-mode backprop from this scalar tensor. */
  backward(): void {
    if (this.size !== 1) throw new Error("backward() requires a scalar");
    const topo: GpuTensor[] = [];
    const seen = new Set<GpuTensor>();
    const visit = (t: GpuTensor): void => {
      if (seen.has(t)) return;
      seen.add(t);
      if (t.ctx) for (const p of t.ctx.parents) visit(p);
      topo.push(t);
    };
    visit(this);
    this.engine.fill(this.ensureGrad(), 1, 1);
    for (let i = topo.length - 1; i >= 0; i--) topo[i].ctx?.backward();
  }
}
