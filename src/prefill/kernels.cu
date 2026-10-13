// src/prefill/kernels.cu - see include/strata/prefill/kernels.hpp.
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill {
namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr int S = 128, HK = 16, HV = 48, C = 10240;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ uint16_t bf(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ uint16_t hf(float f) { return __half_as_ushort(__float2half_rn(f)); }
// A SwiGLU product for an FP16 GEMM, saturated: a token with a massive activation would turn into inf there and NaN
// after the down projection (decode's q8_1 has room to ~8e6). A NaN stays NaN; finite values round as before.
__device__ __forceinline__ uint16_t hf_sat(float f) { return hf(isnan(f) ? f : fminf(fmaxf(f, -65504.0f), 65504.0f)); }
// block-wide sum for blockDim.x <= 1024, result broadcast
__device__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float t = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
    if (w == 0) t = warp_sum(t);
    if (threadIdx.x == 0) sh[0] = t;
    __syncthreads();
    return sh[0];
}
void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "prefill %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
unsigned blocks_for(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

// ---------------------------------------------------------------- hyper-connection
__global__ void gr_norm_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                               float* __restrict__ xn, uint16_t* __restrict__ xn16) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int c = (int) (row % HC);
    const float* r = R + row * N;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < N; d += blockDim.x) ss += r[d] * r[d];
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    for (int d = threadIdx.x; d < N; d += blockDim.x) {
        const float v = r[d] * rs * w[c * N + d];
        xn[row * N + d] = v;
        xn16[row * N + d] = bf(v);
    }
}
// gr_norm_kernel's row scale and BF16 image; gr_mix_r_kernel recomputes r * rs * w in the same order, so the FP32
// copy of the normalized rows is neither written nor read
__global__ void gr_norm_rs_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                                  float* __restrict__ rs_out, uint16_t* __restrict__ xn16) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int c = (int) (row % HC);
    const float* r = R + row * N;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < N; d += blockDim.x) ss += r[d] * r[d];
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rs_out[row] = rs;
    for (int d = threadIdx.x; d < N; d += blockDim.x) xn16[row * N + d] = bf(r[d] * rs * w[c * N + d]);
}
__global__ void gr_mix_r_kernel(const float* __restrict__ R, const float* __restrict__ rs, const float* __restrict__ w,
                                const float* __restrict__ g, float* __restrict__ mixed, uint16_t* __restrict__ mixed16,
                                int64_t T, uint16_t* __restrict__ mixed_h) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const int64_t j = t * D + c * N + d;
        const float x = R[j] * rs[t * HC + c] * w[c * N + d];   // gr_norm_kernel's value, bit for bit
        s = fmaf(x, sigm(g[j]), s);
    }
    s /= (float) HC;
    if (mixed) mixed[i] = s;
    if (mixed16) mixed16[i] = bf(s);
    if (mixed_h) mixed_h[i] = hf(s);
}
// gr_write_kernel for one row (t, c), then gr_norm_rs_kernel's reduction over it with the next read's norm weights:
// the same thread-to-element mapping (256 threads, stride 256) and block_sum, so rs and the BF16 image are the same
// bits, and R is not read back
constexpr int GRW_PER = N / 256;
__global__ void __launch_bounds__(256)
gr_write_norm_rs_kernel(float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj,
                        int64_t inj_ld, const float* __restrict__ w, float eps, float* __restrict__ rs_out,
                        uint16_t* __restrict__ xn16, const uint16_t* __restrict__ partial) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int64_t t = row / HC;
    const int c = (int) (row % HC);
    float* r = R + row * N;
    const float sc = 2.0f * sigm(inj[t * inj_ld + c] / (float) HC);
    float v[GRW_PER];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < GRW_PER; ++k) {
        const int d = threadIdx.x + 256 * k;
        const float b = partial ? bo[t * N + d] + __half2float(__ushort_as_half(partial[t * N + d])) : bo[t * N + d];
        const float x = fmaf(b, sc, r[d]);
        r[d] = x;
        v[k] = x;
        ss += x * x;
    }
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rs_out[row] = rs;
#pragma unroll
    for (int k = 0; k < GRW_PER; ++k) {
        const int d = threadIdx.x + 256 * k;
        xn16[row * N + d] = bf(v[k] * rs * w[c * N + d]);
    }
}
__global__ void gr_silu_kernel(const float* __restrict__ lo, uint16_t* __restrict__ lo16, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = lo[i] / (float) HC;
    lo16[i] = bf(x / (1.0f + __expf(-x)));
}
__global__ void gr_mix_kernel(const float* __restrict__ xn, const float* __restrict__ g, float* __restrict__ mixed,
                              uint16_t* __restrict__ mixed16, int64_t T, uint16_t* __restrict__ mixed_h) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const int64_t j = t * D + c * N + d;
        s = fmaf(xn[j], sigm(g[j]), s);
    }
    s /= (float) HC;
    if (mixed) mixed[i] = s;
    if (mixed16) mixed16[i] = bf(s);
    if (mixed_h) mixed_h[i] = hf(s);
}
__global__ void gr_write_kernel(float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj,
                                int64_t inj_ld, int64_t T, const uint16_t* __restrict__ partial) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int64_t t = i / D, c = (i % D) / N, d = i % N;
    const float b = partial ? bo[t * N + d] + __half2float(__ushort_as_half(partial[t * N + d])) : bo[t * N + d];
    R[i] = fmaf(b, 2.0f * sigm(inj[t * inj_ld + c] / (float) HC), R[i]);
}
__global__ void gr_broadcast_kernel(const float* __restrict__ e, float* __restrict__ R, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int64_t t = i / D, d = i % N;
    R[i] = e[t * N + d];
}

// ---------------------------------------------------------------- GDN
__global__ void gdn_gates_kernel(const float* __restrict__ ab, const float* __restrict__ dt,
                                 const float* __restrict__ ssm_a, float* __restrict__ gate, float* __restrict__ beta,
                                 int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * HV) return;
    const int64_t t = i / HV, h = i % HV;
    const float v = ab[t * 2 * HV + h] + dt[h];
    gate[i] = (v > 20.0f ? v : log1pf(__expf(v))) * ssm_a[h];
    beta[i] = sigm(ab[t * 2 * HV + HV + h]);
}
// The conv history after a chunk: its last three inputs, older ones moved up when the chunk is shorter.
__global__ void gdn_conv_hist_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float old[3] = {hist[c * 3], hist[c * 3 + 1], hist[c * 3 + 2]};
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        const int64_t j = T - 3 + r;
        hist[c * 3 + r] = j >= 0 ? qkv[j * C + c] : old[r + T];
    }
}
// y16 = rmsnorm(o) * gamma * sigmoid(z) per (token, head) in FP16 (the out projection's input), from o in y; with
// Y32 its FP32 value too, in place over o
template <bool Y32>
__global__ void __launch_bounds__(S) gdn_norm_kernel(float* __restrict__ y, const float* __restrict__ z,
                                                     const float* __restrict__ gamma, float eps,
                                                     uint16_t* __restrict__ y16) {
    __shared__ float wsum[S / 32];
    const size_t i = ((size_t) blockIdx.y * HV + blockIdx.x) * S + threadIdx.x;
    const float oc = y[i];
    const float sp = warp_sum(oc * oc);
    if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = sp;
    __syncthreads();
    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
    const float v = oc * rsqrtf(ss / (float) S + eps) * gamma[threadIdx.x] * sigm(z[i]);
    if constexpr (Y32) y[i] = v;
    y16[i] = hf(v);
}

// ---------------------------------------------------------------- MoE
__global__ void route_kernel(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ wout,
                             int64_t T) {
    const int64_t t = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (t >= T) return;
    const int lane = threadIdx.x & 31;
    const float* lg = logits + t * 512;
    float v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = lg[lane + i * 32];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) mx = fmaxf(mx, v[i]);
    mx = warp_max(mx);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) { v[i] = expf(v[i] - mx); sum += v[i]; }
    const float rcp = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) { v[i] *= rcp; if (isnan(v[i])) v[i] = -FLT_MAX; }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = v[0];
        int ex = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) if (v[i] > best) { best = v[i]; ex = lane + i * 32; }
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, m);
            const int oi = __shfl_xor_sync(0xffffffffu, ex, m);
            if (ob > best || (ob == best && oi < ex)) { best = ob; ex = oi; }
        }
        if ((ex & 31) == lane) { v[ex / 32] = -INFINITY; selected_sum += best; }
        if (lane == 0) ids[t * 10 + rank] = ex;
        if (rank == lane) selected = best;
    }
    selected_sum = fmaxf(warp_sum(selected_sum), 6.103515625e-5f);
    if (lane < 10) wout[t * 10 + lane] = selected / selected_sum;
}
// Strata blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up scales [1280][40] f16, down scales [2560][10] f16
template <bool HALF>
__global__ void blob_dequant_kernel(const uint8_t* __restrict__ blob, uint16_t* __restrict__ gu16,
                                    uint16_t* __restrict__ d16) {
    constexpr size_t O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                     O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one thread per 4 weights (one code byte)
    const int64_t n_gu = 1280LL * 640, n_d = 2560LL * 160;
    if (i < n_gu) {
        const int64_t row = i / 640, byte = i % 640;
        const uint8_t c = blob[row * 640 + byte];
        const uint8_t* sp = blob + O_GU_SC + (size_t) (row * 40 + (byte * 4) / 64) * 2;
        const float d = __half2float(__ushort_as_half((uint16_t) (sp[0] | (sp[1] << 8))));
        uint16_t* o = gu16 + row * 2560 + byte * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float v = (float) (((c >> (2 * k)) & 3) - 1) * d; o[k] = HALF ? hf(v) : bf(v); }
    } else if (i < n_gu + n_d) {
        const int64_t j = i - n_gu, row = j / 160, byte = j % 160;
        const uint8_t c = blob[O_D_CODES + row * 160 + byte];
        const uint8_t* sp = blob + O_D_SC + (size_t) (row * 10 + (byte * 4) / 64) * 2;
        const float d = __half2float(__ushort_as_half((uint16_t) (sp[0] | (sp[1] << 8))));
        uint16_t* o = d16 + row * 640 + byte * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float v = (float) (((c >> (2 * k)) & 3) - 1) * d; o[k] = HALF ? hf(v) : bf(v); }
    }
}
__global__ void swiglu_il_kernel(const float* __restrict__ gu, uint16_t* __restrict__ h16, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const int64_t r = i / 640, k = i % 640;
    const float g = gu[r * 1280 + 2 * k], u = gu[r * 1280 + 2 * k + 1];
    h16[i] = hf_sat(g / (1.0f + __expf(-g)) * u);
}
__global__ void swiglu_pair_kernel(const float* __restrict__ g, const float* __restrict__ u, uint16_t* __restrict__ h16,
                                   int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const float a = g[i];
    h16[i] = hf_sat(a / (1.0f + __expf(-a)) * u[i]);
}
__global__ void gather_rows16_kernel(const uint16_t* __restrict__ x, const int32_t* __restrict__ src,
                                     uint16_t* __restrict__ dst, int64_t n, int64_t width) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one uint4 (8 bf16)
    const int64_t per = width / 8;
    if (i >= n * per) return;
    const int64_t r = i / per, j = i % per;
    reinterpret_cast<uint4*>(dst)[r * per + j] = reinterpret_cast<const uint4*>(x)[(int64_t) src[r] * per + j];
}
__global__ void gather_rows16_f32_kernel(const uint16_t* __restrict__ x, const int32_t* __restrict__ src,
                                         float* __restrict__ dst, int64_t n, int64_t width) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // 8 values
    const int64_t per = width / 8;
    if (i >= n * per) return;
    const int64_t r = i / per, j = i % per;
    const uint4 v = reinterpret_cast<const uint4*>(x)[(int64_t) src[r] * per + j];
    const __half2* h = reinterpret_cast<const __half2*>(&v);
    float4* o = reinterpret_cast<float4*>(dst + r * width + j * 8);
    const float2 a = __half22float2(h[0]), b = __half22float2(h[1]), c = __half22float2(h[2]), d = __half22float2(h[3]);
    o[0] = make_float4(a.x, a.y, b.x, b.y);
    o[1] = make_float4(c.x, c.y, d.x, d.y);
}
__global__ void moe_shared_gate_kernel(float* __restrict__ x, const float* __restrict__ sg, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    x[i] = x[i] * sigm(sg[i / N]);
}
__global__ void moe_scatter_add_kernel(float* __restrict__ sum, const float* __restrict__ rows,
                                       const float* __restrict__ w, const int32_t* __restrict__ src, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * N) return;
    const int64_t r = i / N, d = i % N;
    float* o = sum + (int64_t) src[r] * N + d;
    *o = fmaf(w[r], rows[i], *o);
}

// ---------------------------------------------------------------- QSA helpers
__global__ void rms_rows_kernel(float* __restrict__ x, const float* __restrict__ w, int64_t cols, int64_t ld, float eps) {
    __shared__ float sh[32];
    float* r = x + (int64_t) blockIdx.x * ld;
    float ss = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) ss += r[c] * r[c];
    const float s = rsqrtf(block_sum(ss, sh) / (float) cols + eps);
    __syncthreads();
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) r[c] = s * r[c] * w[c];
}
__global__ void rope_kernel(float* __restrict__ x, int64_t heads, int64_t dim, int64_t ld, int64_t pos0,
                            float theta_scale, const int32_t* __restrict__ mtab) {
    const int64_t row = blockIdx.x;             // t * heads + h
    const int pair = threadIdx.x;               // 0..31
    const int64_t t = row / heads, h = row % heads;
    float* p = x + t * ld + h * dim;
    const float theta = (float) strata::kernels::mrope_pos(mtab, (int) (pos0 + t), pair) * powf(theta_scale, (float) pair);
    const float c = cosf(theta), s = sinf(theta);
    const float a = p[pair], b = p[pair + 32];
    p[pair] = a * c - b * s;
    p[pair + 32] = a * s + b * c;
}
__global__ void split_q_kernel(const float* __restrict__ qf, float* __restrict__ q, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 24 * 256) return;
    const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    q[i] = qf[t * 24 * 512 + h * 512 + d];
}
__global__ void gate_attn_kernel(const float* __restrict__ a, const float* __restrict__ qf, uint16_t* __restrict__ o16,
                                 int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 24 * 256) return;
    const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    o16[i] = hf(a[i] * (1.0f / (1.0f + expf(-qf[t * 24 * 512 + h * 512 + 256 + d]))));
}

// one block per (token, kv head, 64-value group[, side]); 64 threads
__global__ void kv_append_kernel(const float* __restrict__ K, const float* __restrict__ V, int64_t pos0,
                                 const int32_t* __restrict__ table, int64_t page_size, uint16_t* k_pool,
                                 uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
                                 int sides) {
    const int64_t t = blockIdx.x;
    const int kvh = blockIdx.y, g = sides == 3 ? blockIdx.z >> 1 : blockIdx.z;
    const bool is_v = sides == 3 ? (blockIdx.z & 1) != 0 : sides == 2;
    const int d = g * 64 + threadIdx.x;
    const float x = (is_v ? V : K)[t * 512 + kvh * 256 + d];
    const int64_t pos = pos0 + t;
    const int64_t page = table[pos / page_size];
    const int64_t row = (page * 2 + kvh) * page_size + pos % page_size;
    if (k_pool != nullptr) {
        (is_v ? v_pool : k_pool)[row * 256 + d] = hf(x);
        return;
    }
    float a = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float wm[2];
    if ((threadIdx.x & 31) == 0) wm[threadIdx.x >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(wm[0], wm[1]);
    const uint16_t sb = hf(amax / 127.0f);
    const float sf = __half2float(__ushort_as_half(sb));
    int q = 0;
    if (sf > 0.0f) { q = __float2int_rn(x / sf); q = q < -127 ? -127 : (q > 127 ? 127 : q); }
    (is_v ? v_q : k_q)[row * 256 + d] = (int8_t) q;
    if (threadIdx.x == 0) (is_v ? v_scale : k_scale)[row * 4 + g] = sb;
}
__global__ void to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = hf(x[i]);
}
__global__ void q8_to_bf16_kernel(const int8_t* __restrict__ q, const uint16_t* __restrict__ s, int64_t n,
                                  uint16_t* __restrict__ y) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = bf((float) q[i] * __half2float(__ushort_as_half(s[i / 32])));
}
__global__ void to_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = bf(x[i]);
}

// ---------------------------------------------------------------- GDN, the chunkwise form
// Per value head, a chunk of GCL tokens from state S0 (rows: the key dimension, columns: the value dimension), with
// g_t = exp(gate_t) and gamma_t the gates' sum from the chunk's start to t:
//   A[t][i] = beta_t exp(gamma_t - gamma_i) k_t.k_i (i < t),   T = (I + A)^-1,
//   B_t = beta_t (v_t - exp(gamma_t) S0^T k_t),   D = T B (the delta rule's corrections),
//   o_t = exp(gamma_t) S0^T q_t + sum_{i <= t} exp(gamma_t - gamma_i) (q_t.k_i) D_i,
//   S = exp(gamma_L) S0 + sum_t exp(gamma_L - gamma_t) k_t D_t^T.
// gdn_chunk_prep_kernel: T, P (the decayed q.k) and the scales of every chunk (FP32, a block per chunk and key head);
// gdn_chunk_walk_kernel: a block per key head and GCW value columns, a warp per value head and 16 columns, its state
// slice S^T [16 x 128] in FP32 registers as mma accumulators, chunk after chunk; every product an m16n8k16 FP16 mma
// with FP32 sums (an accumulator tile pair is the next mma's A operand; K and Q staged in FP32, rounded as loaded).
constexpr int VPK = HV / HK;     // value heads a key head
constexpr int GCL = 16;          // tokens a chunk
constexpr int GCW = 32;          // value columns a walk block
constexpr int GKP = S + 8;       // padded FP32 K / Q rows: a half-warp's 64-bit loads on distinct banks
constexpr int GTP = GCL + 8;     // padded FP16 T / P rows
constexpr int GVP = GCW + 4;     // padded V rows
struct alignas(16) GdnTps {      // a (chunk, value head)
    __half t[GCL][GTP], p[GCL][GTP];
    float s[64];                 // exp(gamma_t), exp(gamma_L - gamma_t), beta_t, exp(gamma_L)
};
struct alignas(16) GdnStage {
    float k[GCL][GKP], q[GCL][GKP];
    GdnTps tp[VPK];
    float v[VPK][GCL][GVP];
};
__device__ __forceinline__ uint32_t gh2(float lo, float hi) {
    const __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ uint32_t gld32(const __half* p) { return *reinterpret_cast<const uint32_t*>(p); }
__device__ __forceinline__ void gmma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ga_of(const float (&x)[4], const float (&y)[4], uint32_t (&a)[4]) {
    a[0] = gh2(x[0], x[1]);
    a[1] = gh2(x[2], x[3]);
    a[2] = gh2(y[0], y[1]);
    a[3] = gh2(y[2], y[3]);
}
__device__ __forceinline__ void gcp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src));
}

__global__ void __launch_bounds__(128) gdn_chunk_prep_kernel(const float* __restrict__ h, const float* __restrict__ gate,
                                                             const float* __restrict__ beta, int T,
                                                             GdnTps* __restrict__ tp) {
    const int ch = blockIdx.x, kh = blockIdx.y, tid = threadIdx.x, j = tid >> 5, lane = tid & 31;
    const int t0 = ch * GCL, n = min(GCL, T - t0);
    __shared__ __align__(16) float sk[GCL][S + 4], sq[GCL][S + 4];
    __shared__ float kk[GCL][GCL + 1], qk[GCL][GCL + 1];
    __shared__ float a[VPK][GCL][GCL + 1], ti[VPK][GCL][GCL + 1], gm[VPK][GCL], bt[VPK][GCL];
    for (int i = tid; i < 2 * GCL * (S / 4); i += blockDim.x) {
        const int w = i / (GCL * (S / 4)), t = (i / (S / 4)) % GCL, r4 = (i % (S / 4)) * 4;
        const float4 v = t < n ? *reinterpret_cast<const float4*>(h + (size_t) (t0 + t) * C + (w ? HK * S : 0) + kh * S + r4)
                               : make_float4(0.f, 0.f, 0.f, 0.f);
        *reinterpret_cast<float4*>(w ? &sk[t][r4] : &sq[t][r4]) = v;
    }
    __syncthreads();
    for (int p = tid; p < 2 * GCL * GCL; p += blockDim.x) {
        const int w = p / (GCL * GCL), t = (p / GCL) % GCL, i = p % GCL;
        const float4* x = reinterpret_cast<const float4*>(w ? sq[t] : sk[t]);
        const float4* y = reinterpret_cast<const float4*>(sk[i]);
        float acc = 0.0f;
#pragma unroll 8
        for (int r = 0; r < S / 4; ++r) {
            const float4 u = x[r], v = y[r];
            acc = fmaf(u.x, v.x, acc);
            acc = fmaf(u.y, v.y, acc);
            acc = fmaf(u.z, v.z, acc);
            acc = fmaf(u.w, v.w, acc);
        }
        (w ? qk : kk)[t][i] = acc;
    }
    __syncthreads();
    if (j >= VPK) return;   // the fourth warp only helped with the loads and the dots
    const int head = kh + j * HK;
    float gv = lane < n ? gate[(size_t) (t0 + lane) * HV + head] : 0.0f;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const float y = __shfl_up_sync(0xffffffffu, gv, d);
        if (lane >= d) gv += y;
    }
    if (lane < GCL) {
        gm[j][lane] = gv;
        bt[j][lane] = lane < n ? beta[(size_t) (t0 + lane) * HV + head] : 0.0f;
    }
    __syncwarp();
    for (int p = lane; p < GCL * GCL; p += 32) {
        const int t = p / GCL, i = p % GCL;
        a[j][t][i] = i < t ? bt[j][t] * expf(gm[j][t] - gm[j][i]) * kk[t][i] : 0.0f;
        ti[j][t][i] = t == i ? 1.0f : 0.0f;
    }
    __syncwarp();
    for (int t = 1; t < GCL; ++t) {   // T = (I + A)^-1, row t from the rows before it
        if (lane < t) {
            float acc = 0.0f;
            for (int m = lane; m < t; ++m) acc = fmaf(a[j][t][m], ti[j][m][lane], acc);
            ti[j][t][lane] = -acc;
        }
        __syncwarp();
    }
    GdnTps& w = tp[(size_t) ch * HV + head];
    uint32_t* wt = reinterpret_cast<uint32_t*>(&w.t[0][0]);
    uint32_t* wp = reinterpret_cast<uint32_t*>(&w.p[0][0]);
    for (int p = lane; p < GCL * (GCL / 2); p += 32) {
        const int t = p / (GCL / 2), i = 2 * (p % (GCL / 2));
        wt[t * (GTP / 2) + i / 2] = gh2(ti[j][t][i], ti[j][t][i + 1]);
        wp[t * (GTP / 2) + i / 2] = gh2(i <= t ? expf(gm[j][t] - gm[j][i]) * qk[t][i] : 0.0f,
                                        i + 1 <= t ? expf(gm[j][t] - gm[j][i + 1]) * qk[t][i + 1] : 0.0f);
    }
    if (lane < GCL) {
        w.s[lane] = expf(gm[j][lane]);
        w.s[GCL + lane] = expf(gm[j][GCL - 1] - gm[j][lane]);
        w.s[2 * GCL + lane] = bt[j][lane];
    }
    if (lane == 0) w.s[3 * GCL] = expf(gm[j][GCL - 1]);
}

__global__ void __launch_bounds__(32 * 2 * VPK) gdn_chunk_walk_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                                   const GdnTps* __restrict__ tp, float* __restrict__ out,
                                                                   int T) {
    extern __shared__ __align__(16) unsigned char gdn_smem[];
    GdnStage* st = reinterpret_cast<GdnStage*>(gdn_smem);
    const int kh = blockIdx.x, cs = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, q = lane & 3, j = warp >> 1, head = kh + j * HK;
    const int cl = (warp & 1) * 16, cb = cs * GCW + cl;   // the warp's first column, in the block and in the head
    const int nch = (T + GCL - 1) / GCL;
    float s[S / 8][4];   // S^T: tile nt covers rows 8 nt .. 8 nt + 7
#pragma unroll
    for (int nt = 0; nt < S / 8; ++nt) {
        const int r = 8 * nt + 2 * q;
        s[nt][0] = state[((size_t) r * HV + head) * S + cb + g];
        s[nt][1] = state[((size_t) (r + 1) * HV + head) * S + cb + g];
        s[nt][2] = state[((size_t) r * HV + head) * S + cb + g + 8];
        s[nt][3] = state[((size_t) (r + 1) * HV + head) * S + cb + g + 8];
    }
    auto stage = [&](int ch) {
        GdnStage& b = st[ch & 1];
        const int t0 = ch * GCL, n = min(GCL, T - t0);
        for (int i = tid; i < 2 * GCL * (S / 4); i += blockDim.x) {
            const int w = i / (GCL * (S / 4)), t = (i / (S / 4)) % GCL, r4 = (i % (S / 4)) * 4;
            float* d = w ? &b.k[t][r4] : &b.q[t][r4];
            if (t < n) gcp16(d, h + (size_t) (t0 + t) * C + (w ? HK * S : 0) + kh * S + r4);
            else *reinterpret_cast<float4*>(d) = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        for (int i = tid; i < VPK * (int) (sizeof(GdnTps) / 16); i += blockDim.x) {
            const int jj = i / (int) (sizeof(GdnTps) / 16), p = i % (int) (sizeof(GdnTps) / 16);
            gcp16(reinterpret_cast<char*>(&b.tp[jj]) + 16 * p,
                  reinterpret_cast<const char*>(tp + (size_t) ch * HV + kh + jj * HK) + 16 * p);
        }
        for (int i = tid; i < VPK * GCL * (GCW / 4); i += blockDim.x) {
            const int jj = i / (GCL * (GCW / 4)), t = (i / (GCW / 4)) % GCL, c4 = (i % (GCW / 4)) * 4;
            float* d = &b.v[jj][t][c4];
            if (t < n) gcp16(d, h + (size_t) (t0 + t) * C + 2 * HK * S + (kh + jj * HK) * S + cs * GCW + c4);
            else *reinterpret_cast<float4*>(d) = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    stage(0);
    for (int ch = 0; ch < nch; ++ch) {
        if (ch + 1 < nch) stage(ch + 1);
        else asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        const GdnStage& b = st[ch & 1];
        const int t0 = ch * GCL, n = min(GCL, T - t0);
        const float* scj = b.tp[j].s;
        // W^T = S^T K^T and S^T Q^T [16 columns x GCL tokens]
        float w[GCL / 8][4] = {}, qs[GCL / 8][4] = {};
#pragma unroll
        for (int kt = 0; kt < S / 16; ++kt) {
            uint32_t a[4];
            ga_of(s[2 * kt], s[2 * kt + 1], a);
#pragma unroll
            for (int nt = 0; nt < GCL / 8; ++nt) {
                const float2 k0 = *reinterpret_cast<const float2*>(&b.k[8 * nt + g][16 * kt + 2 * q]);
                const float2 k1 = *reinterpret_cast<const float2*>(&b.k[8 * nt + g][16 * kt + 2 * q + 8]);
                const float2 q0 = *reinterpret_cast<const float2*>(&b.q[8 * nt + g][16 * kt + 2 * q]);
                const float2 q1 = *reinterpret_cast<const float2*>(&b.q[8 * nt + g][16 * kt + 2 * q + 8]);
                gmma(w[nt], a, gh2(k0.x, k0.y), gh2(k1.x, k1.y));
                gmma(qs[nt], a, gh2(q0.x, q0.y), gh2(q1.x, q1.y));
            }
        }
        // B^T = beta (V^T - exp(gamma) W^T); element e of tile nt: column g + 8 (e >> 1), token 8 nt + 2 q + (e & 1)
        float bt[GCL / 8][4];
#pragma unroll
        for (int nt = 0; nt < GCL / 8; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int t = 8 * nt + 2 * q + (e & 1);
                bt[nt][e] = scj[2 * GCL + t] * (b.v[j][t][cl + g + 8 * (e >> 1)] - scj[t] * w[nt][e]);
            }
        // D^T = B^T T^T, O^T = exp(gamma) S^T Q^T + D^T P^T
        uint32_t ab[4], ad[4];
        ga_of(bt[0], bt[1], ab);
        float d[GCL / 8][4] = {}, o[GCL / 8][4];
#pragma unroll
        for (int nt = 0; nt < GCL / 8; ++nt) {
            const __half* tr = &b.tp[j].t[8 * nt + g][2 * q];
            gmma(d[nt], ab, gld32(tr), gld32(tr + 8));
        }
        ga_of(d[0], d[1], ad);
        const float rs = rsqrtf((float) S);
#pragma unroll
        for (int nt = 0; nt < GCL / 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) o[nt][e] = scj[8 * nt + 2 * q + (e & 1)] * qs[nt][e];
            const __half* pr = &b.tp[j].p[8 * nt + g][2 * q];
            gmma(o[nt], ad, gld32(pr), gld32(pr + 8));
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int t = 8 * nt + 2 * q + (e & 1);
                if (t < n) out[(size_t) (t0 + t) * HV * S + head * S + cb + g + 8 * (e >> 1)] = o[nt][e] * rs;
            }
        }
        // S^T = exp(gamma_L) S^T + (exp(gamma_L - gamma) D^T) K
#pragma unroll
        for (int nt = 0; nt < GCL / 8; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) d[nt][e] *= scj[GCL + 8 * nt + 2 * q + (e & 1)];
        ga_of(d[0], d[1], ad);
        const float eL = scj[3 * GCL];
#pragma unroll
        for (int nt = 0; nt < S / 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) s[nt][e] *= eL;
            const int r = 8 * nt + g;
            gmma(s[nt], ad, gh2(b.k[2 * q][r], b.k[2 * q + 1][r]), gh2(b.k[2 * q + 8][r], b.k[2 * q + 9][r]));
        }
        __syncthreads();   // every warp is done with this stage before the next one fills it
    }
#pragma unroll
    for (int nt = 0; nt < S / 8; ++nt) {
        const int r = 8 * nt + 2 * q;
        state[((size_t) r * HV + head) * S + cb + g] = s[nt][0];
        state[((size_t) (r + 1) * HV + head) * S + cb + g] = s[nt][1];
        state[((size_t) r * HV + head) * S + cb + g + 8] = s[nt][2];
        state[((size_t) (r + 1) * HV + head) * S + cb + g + 8] = s[nt][3];
    }
}

}  // namespace

void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const int32_t* page_table, int64_t page_size,
               uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
               void* stream, int sides) {
    if (T <= 0 || sides < 1 || sides > 3) return;
    kv_append_kernel<<<dim3((unsigned) T, 2, sides == 3 ? 8 : 4), 64, 0, (cudaStream_t) stream>>>(
        K, V, pos0, page_table, page_size, k_pool, v_pool, k_q, v_q, k_scale, v_scale, sides);
    check("kv_append");
}
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_f16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("to_f16");
}
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_bf16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("to_bf16");
}

void q8_to_bf16(const int8_t* q, const uint16_t* scales, int64_t n, uint16_t* y, void* stream) {
    if (n <= 0) return;
    q8_to_bf16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(
        q, scales, n, y);
    check("q8_to_bf16");
}

void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream) {
    gr_norm_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, xn, xn16);
    check("gr_norm");
}
void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream) {
    gr_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, rs, xn16);
    check("gr_norm_rs");
}
void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed, uint16_t* mixed16,
              int64_t T, void* stream, uint16_t* mixed_h) {
    gr_mix_r_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(R, rs, w_norm, gated, mixed, mixed16, T,
                                                                          mixed_h);
    check("gr_mix_r");
}
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm, float eps,
                      float* rs, uint16_t* xn16, int64_t T, void* stream, const uint16_t* partial) {
    gr_write_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, bo, inj, inj_ld, w_norm, eps,
                                                                                     rs, xn16, partial);
    check("gr_write_norm_rs");
}
void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream) {
    gr_silu_kernel<<<blocks_for(T * LR), 256, 0, (cudaStream_t) stream>>>(lo, lo16, T * LR);
    check("gr_silu");
}
void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream,
            uint16_t* mixed_h) {
    gr_mix_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(xn, gated, mixed, mixed16, T, mixed_h);
    check("gr_mix");
}
void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream,
              const uint16_t* partial) {
    gr_write_kernel<<<blocks_for(T * D), 256, 0, (cudaStream_t) stream>>>(R, bo, inj, inj_ld, T, partial);
    check("gr_write");
}
void gr_broadcast(const float* e, float* R, int64_t T, void* stream) {
    gr_broadcast_kernel<<<blocks_for(T * D), 256, 0, (cudaStream_t) stream>>>(e, R, T);
    check("gr_broadcast");
}
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream) {
    gdn_gates_kernel<<<blocks_for(T * HV), 256, 0, (cudaStream_t) stream>>>(ab, dt, ssm_a, gate, beta, T);
    check("gdn_gates");
}
void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream) {
    for (int64_t t0 = 0; t0 < T; t0 += 65535)
        strata::kernels::gdn_conv_l2_multi(history, qkv, conv_w, h, C, 2 * HK, eps, (int) std::min<int64_t>(65535, T - t0),
                                           stream, (int) t0);
    gdn_conv_hist_kernel<<<C / 256, 256, 0, (cudaStream_t) stream>>>(history, qkv, T);
    check("gdn_conv");
}
size_t gdn_scan_scratch_bytes(int64_t T) { return (size_t) ((T + GCL - 1) / GCL) * HV * sizeof(GdnTps); }
void gdn_scan(float* state, const float* h, const float* gate, const float* beta, float* y, int64_t T, void* scratch,
              void* stream) {
    if (T <= 0) return;
    static const bool smem_set = [] {   // the walk's two stages: 58 KB
        return cudaFuncSetAttribute(gdn_chunk_walk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int) (2 * sizeof(GdnStage))) == cudaSuccess;
    }();
    if (!smem_set) {
        std::fprintf(stderr, "gdn_scan: the walk's shared memory\n");
        std::exit(1);
    }
    auto* tp = (GdnTps*) scratch;
    gdn_chunk_prep_kernel<<<dim3((unsigned) ((T + GCL - 1) / GCL), HK), 128, 0, (cudaStream_t) stream>>>(h, gate, beta,
                                                                                                       (int) T, tp);
    gdn_chunk_walk_kernel<<<dim3(HK, S / GCW), 32 * 2 * VPK, 2 * sizeof(GdnStage), (cudaStream_t) stream>>>(state, h, tp,
                                                                                                         y, (int) T);
    check("gdn_scan");
}
void gdn_out_norm(float* y, const float* z, const float* gamma, float eps, uint16_t* y16, int64_t T, void* stream,
                  bool y32) {
    for (int64_t t0 = 0; t0 < T; t0 += 65535) {
        const int64_t n = std::min<int64_t>(65535, T - t0);
        const dim3 grid(HV, (unsigned) n);
        if (y32)
            gdn_norm_kernel<true><<<grid, S, 0, (cudaStream_t) stream>>>(y + t0 * HV * S, z + t0 * HV * S, gamma, eps,
                                                                         y16 + t0 * HV * S);
        else
            gdn_norm_kernel<false><<<grid, S, 0, (cudaStream_t) stream>>>(y + t0 * HV * S, z + t0 * HV * S, gamma, eps,
                                                                          y16 + t0 * HV * S);
    }
    check("gdn_out_norm");
}
void route(const float* logits, int32_t* ids, float* weights, int64_t T, void* stream) {
    route_kernel<<<(unsigned) ((T + 7) / 8), 256, 0, (cudaStream_t) stream>>>(logits, ids, weights, T);
    check("route");
}
void blob_dequant(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_kernel<false><<<blocks_for(1280LL * 640 + 2560LL * 160), 256, 0, (cudaStream_t) stream>>>(blob, gu16, down16);
    check("blob_dequant");
}
void blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_kernel<true><<<blocks_for(1280LL * 640 + 2560LL * 160), 256, 0, (cudaStream_t) stream>>>(blob, gu16, down16);
    check("blob_dequant_f16");
}
void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    swiglu_il_kernel<<<blocks_for(n * 640), 256, 0, (cudaStream_t) stream>>>(gu, h16, n);
    check("swiglu_interleaved");
}
void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream) {
    swiglu_pair_kernel<<<blocks_for(n * 640), 256, 0, (cudaStream_t) stream>>>(g, u, h16, n);
    check("swiglu_pair");
}
void gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    gather_rows16_kernel<<<blocks_for(n * (width / 8)), 256, 0, (cudaStream_t) stream>>>(x16, src, dst16, n, width);
    check("gather_rows16");
}
void gather_rows16_f32(const uint16_t* x16, const int32_t* src, float* dst, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    gather_rows16_f32_kernel<<<blocks_for(n * (width / 8)), 256, 0, (cudaStream_t) stream>>>(x16, src, dst, n, width);
    check("gather_rows16_f32");
}
__global__ void sums_to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = hf(fminf(fmaxf(x[i], -65504.0f), 65504.0f));
}
void sums_to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    sums_to_f16_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("sums_to_f16");
}
void moe_shared_gate(float* x, const float* sg, int64_t T, void* stream) {
    moe_shared_gate_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(x, sg, T);
    check("moe_shared_gate");
}
__global__ void moe_gather_add_kernel(float* __restrict__ sum, const float* __restrict__ rows, int64_t r0,
                                      const float* __restrict__ w, const int32_t* __restrict__ tok,
                                      const int32_t* __restrict__ start, const int32_t* __restrict__ list) {
    const int64_t d = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= N) return;
    const int b = blockIdx.y;
    float* o = sum + (int64_t) tok[b] * N + d;
    float s = *o;
    for (int i = start[b]; i < start[b + 1]; ++i) {
        const int r = list[i];
        s = fmaf(w[r], rows[(r - r0) * N + d], s);
    }
    *o = s;
}
void moe_gather_add(float* sum, const float* rows, int64_t r0, const float* w, const int32_t* tok, const int32_t* start,
                    const int32_t* list, int64_t n_tok, void* stream) {
    if (n_tok <= 0) return;
    moe_gather_add_kernel<<<dim3((unsigned) ((N + 255) / 256), (unsigned) n_tok), 256, 0, (cudaStream_t) stream>>>(
        sum, rows, r0, w, tok, start, list);
    check("moe_gather_add");
}
void moe_scatter_add(float* sum, const float* rows, const float* w, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    moe_scatter_add_kernel<<<blocks_for(n * N), 256, 0, (cudaStream_t) stream>>>(sum, rows, w, src, n);
    check("moe_scatter_add");
}
void rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream) {
    if (rows <= 0) return;
    rms_rows_kernel<<<(unsigned) rows, 256, 0, (cudaStream_t) stream>>>(x, w, cols, ld, eps);
    check("rms_rows");
}
void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0, float freq_base, void* stream) {
    const float theta_scale = powf(freq_base, -2.0f / 64.0f);
    rope_kernel<<<(unsigned) (T * heads), 32, 0, (cudaStream_t) stream>>>(x, heads, dim, ld, pos0, theta_scale, strata::kernels::mrope_table());
    check("rope");
}
void split_q(const float* q_full, float* q, int64_t T, void* stream) {
    split_q_kernel<<<blocks_for(T * 24 * 256), 256, 0, (cudaStream_t) stream>>>(q_full, q, T);
    check("split_q");
}
void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream) {
    gate_attn_kernel<<<blocks_for(T * 24 * 256), 256, 0, (cudaStream_t) stream>>>(attn, q_full, out16, T);
    check("gate_attn");
}

}  // namespace strata::prefill
