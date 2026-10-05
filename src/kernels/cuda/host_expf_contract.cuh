#pragma once

#include <cmath>
#include <cstdint>

// Clean-room CUDA implementation of the host contract selected by Debian
// glibc 2.41 on this VM: the AVX2/FMA scalar expf implementation from
// sysdeps/ieee754/flt-32/e_expf.c.  The source provenance and LGPL notice are
// documented in the campaign result; this header contains only the algorithm
// and constants needed for the model's finite FP32 range.

namespace strata::cuda_contract {

static __device__ __constant__ uint64_t exp2_tab[32] = {
    0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full, 0x3fef9301d0125b51ull,
    0x3fef72b83c7d517bull, 0x3fef54873168b9aaull, 0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull,
    0x3fef06fe0a31b715ull, 0x3feef1a7373aa9cbull, 0x3feedea64c123422ull, 0x3feece086061892dull,
    0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull, 0x3feea47eb03a5585ull,
    0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull, 0x3feea11473eb0187ull, 0x3feea589994cce13ull,
    0x3feeace5422aa0dbull, 0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull,
    0x3feee89f995ad3adull, 0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull, 0x3fef3720dcef9069ull,
    0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full, 0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull,
};
static __device__ __constant__ double exp2_invln2_scaled = 0x1.71547652b82fep+5;
static __device__ __constant__ double exp2_c0 = 0x1.c6af84b912394p-20;
static __device__ __constant__ double exp2_c1 = 0x1.ebfce50fac4f3p-13;
static __device__ __constant__ double exp2_c2 = 0x1.62e42ff0c52d6p-6;
static __device__ __constant__ double exp2_shift = 0x1.8p+52;

__device__ __forceinline__ float host_contract_expf_f32(float x) {
    const uint32_t ux = __float_as_uint(x);
    const uint32_t abstop = (ux >> 20) & 0x7ffu;
    if (abstop >= ((__float_as_uint(88.0f) >> 20) & 0x7ffu)) {
        if (ux == 0xff800000u) return 0.0f;
        // Preserve quiet-NaN payloads.  NaN behavior is outside the model
        // contract; this matches the qNaN corpus used by the parity gate.
        if (abstop >= 0x7f8u) return x;
        if (x > 0x1.62e42ep6f) return __int_as_float(0x7f800000);
        if (x < -0x1.9fe368p6f) return 0.0f;
    }
    const double xd = (double) x;
    const double z = __dmul_rn(exp2_invln2_scaled, xd);
    const double kd0 = __dadd_rn(z, exp2_shift);
    const uint64_t ki = __double_as_longlong(kd0);
    const double kd = __dsub_rn(kd0, exp2_shift);
    const double r = __dsub_rn(z, kd);
    const uint64_t t = exp2_tab[ki & 31ull] + (ki << 47);
    const double s = __longlong_as_double((long long) t);
    const double p = fma(exp2_c0, r, exp2_c1);
    const double r2 = __dmul_rn(r, r);
    double y = fma(exp2_c2, r, 1.0);
    y = fma(p, r2, y);
    y = __dmul_rn(y, s);
    return __double2float_rn(y);
}

} // namespace strata::cuda_contract
