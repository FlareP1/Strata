// Standalone host/device expf contract gate.
// The host side is the exact linked glibc expf@GLIBC_2.27 call.  The device
// host_contract candidate mirrors glibc 2.41's AVX2/FMA scalar expf algorithm
// (double range reduction and FMA polynomial) without copying runtime code.
#include <cuda_runtime.h>
#include "cuda/host_expf_contract.cuh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

struct StageFixtureHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t header_bytes;
    uint64_t payload_bytes;
    uint64_t payload_fnv1a;
    int64_t pos0, layer, expert, route_index;
    float route_weight;
    uint64_t n_embd, n_ff, k, act_bytes, h_bytes, blob_bytes;
    uint32_t act_type, gu_type, d_type, gu_act, d_act;
    uint32_t reserved;
};
static_assert(sizeof(StageFixtureHeader) % 8 == 0, "fixture header alignment");

static uint64_t fnv1a(const std::vector<uint8_t>& p) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t x : p) { h ^= x; h *= 1099511628211ull; }
    return h;
}
static uint32_t bits(float x) { uint32_t u = 0; std::memcpy(&u, &x, sizeof u); return u; }
static float from_bits(uint32_t u) { float x; std::memcpy(&x, &u, sizeof x); return x; }
static bool fixture_error = false;

__global__ static void eval_kernel(const float* x, uint32_t* y, int n, int candidate) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const float v = x[i];
    float z;
    if (candidate == 0) z = strata::cuda_contract::host_contract_expf_f32(v);
    else if (candidate == 1) z = expf(v);
    else z = __expf(v);
    y[i] = __float_as_uint(z);
}

__global__ static void eval_swiglu_kernel(const float* g, const float* u, uint32_t* y, int n) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const float e = strata::cuda_contract::host_contract_expf_f32(-g[i]);
    const float den = __fadd_rn(1.0f, e);
    const float sil = __fdiv_rn(g[i], den);
    y[i] = __float_as_uint(__fmul_rn(sil, u[i]));
}

struct SwigluFixture {
    std::vector<float> cpu_g, cpu_u, cpu_h;
    std::vector<float> gpu_g, gpu_u, gpu_h;
};

static void add_fixture_gates(const std::string& path, std::vector<float>& xs,
                              std::vector<SwigluFixture>& swiglu) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "fixture_open_failed " << path << "\n"; fixture_error = true; return; }
    StageFixtureHeader h{};
    in.read((char*) &h, sizeof h);
    if (!in || h.magic != 0x3147545341525453ull || h.version != 1 || h.header_bytes != sizeof h) {
        std::cerr << "fixture_header_failed " << path << "\n"; fixture_error = true; return;
    }
    std::vector<uint8_t> p((size_t) h.payload_bytes);
    in.read((char*) p.data(), (std::streamsize) p.size());
    if (!in || fnv1a(p) != h.payload_fnv1a) { std::cerr << "fixture_checksum_failed " << path << "\n"; fixture_error = true; return; }
    const size_t ids = (size_t) h.k * sizeof(int32_t);
    const size_t weights = (size_t) h.k * sizeof(float);
    const size_t off_cpu_g = ids + weights + (size_t) h.act_bytes;
    const size_t off_gpu_g = off_cpu_g + (size_t) h.n_ff * sizeof(float);
    const size_t off_cpu_u = off_gpu_g + (size_t) h.n_ff * sizeof(float);
    const size_t off_gpu_u = off_cpu_u + (size_t) h.n_ff * sizeof(float);
    const size_t off_cpu_h = off_gpu_u + (size_t) h.n_ff * sizeof(float);
    const size_t off_gpu_h = off_cpu_h + (size_t) h.n_ff * sizeof(float);
    if (off_gpu_h + (size_t) h.n_ff * sizeof(float) > p.size()) { std::cerr << "fixture_layout_failed " << path << "\n"; fixture_error = true; return; }
    const float* cpu_g = (const float*) (p.data() + off_cpu_g);
    const float* gpu_g = (const float*) (p.data() + off_gpu_g);
    const float* cpu_u = (const float*) (p.data() + off_cpu_u);
    const float* gpu_u = (const float*) (p.data() + off_gpu_u);
    const float* cpu_h = (const float*) (p.data() + off_cpu_h);
    const float* gpu_h = (const float*) (p.data() + off_gpu_h);
    xs.insert(xs.end(), cpu_g, cpu_g + h.n_ff);
    xs.insert(xs.end(), gpu_g, gpu_g + h.n_ff);
    SwigluFixture f;
    f.cpu_g.assign(cpu_g, cpu_g + h.n_ff); f.cpu_u.assign(cpu_u, cpu_u + h.n_ff); f.cpu_h.assign(cpu_h, cpu_h + h.n_ff);
    f.gpu_g.assign(gpu_g, gpu_g + h.n_ff); f.gpu_u.assign(gpu_u, gpu_u + h.n_ff); f.gpu_h.assign(gpu_h, gpu_h + h.n_ff);
    swiglu.push_back(std::move(f));
    std::cout << "fixture=" << path << " layer=" << h.layer << " expert=" << h.expert
              << " pos0=" << h.pos0 << " gates=" << (2 * h.n_ff) << "\n";
}

static void add_corpus(std::vector<float>& xs) {
    const float ln2 = 0.6931471805599453094f;
    for (int k = -4096; k <= 4096; ++k) {
        const float c = (float) k * ln2 / 32.0f;
        xs.push_back(c); xs.push_back(std::nextafterf(c, -INFINITY)); xs.push_back(std::nextafterf(c, INFINITY));
        xs.push_back(c + 0.5f * std::numeric_limits<float>::epsilon());
        xs.push_back(c - 0.5f * std::numeric_limits<float>::epsilon());
    }
    uint64_t state = 0x8e4d6f2b91a7c553ull;
    for (int i = 0; i < 1000000; ++i) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        const uint32_t u = (uint32_t) (state >> 32);
        const float unit = (float) (u >> 8) * (1.0f / 16777216.0f);
        xs.push_back((unit * 2.0f - 1.0f) * 24.0f);
    }
    const uint32_t edge[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
                             0x007fffffu, 0x807fffffu, 0x7f7fffffu, 0xff7fffffu,
                             0x7f800000u, 0xff800000u, 0x7fc00000u};
    for (uint32_t u : edge) xs.push_back(from_bits(u));
}

int main(int argc, char** argv) {
    std::vector<float> xs;
    std::vector<SwigluFixture> swiglu;
    for (int i = 1; i < argc; ++i) add_fixture_gates(argv[i], xs, swiglu);
    if (fixture_error || swiglu.empty()) return 2;
    add_corpus(xs);
    std::vector<uint32_t> host(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) host[i] = bits(::expf(xs[i]));
    float* dx = nullptr; uint32_t* dy = nullptr;
    cudaError_t e = cudaMalloc(&dx, xs.size() * sizeof(float));
    if (e != cudaSuccess) { std::cerr << "cudaMalloc x failed: " << cudaGetErrorString(e) << "\n"; return 2; }
    e = cudaMalloc(&dy, xs.size() * sizeof(uint32_t));
    if (e != cudaSuccess) { std::cerr << "cudaMalloc y failed: " << cudaGetErrorString(e) << "\n"; return 2; }
    cudaMemcpy(dx, xs.data(), xs.size() * sizeof(float), cudaMemcpyHostToDevice);
    bool contract_ok = true;
    for (int candidate = 0; candidate < 3; ++candidate) {
        eval_kernel<<<(unsigned)((xs.size() + 255) / 256), 256>>>(dx, dy, (int) xs.size(), candidate);
        cudaDeviceSynchronize();
        std::vector<uint32_t> got(xs.size());
        cudaMemcpy(got.data(), dy, got.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        size_t mismatches = 0;
        double max_abs = 0.0;
        std::cout << "candidate=" << (candidate == 0 ? "host_contract" : candidate == 1 ? "cuda_expf" : "cuda___expf")
                  << " corpus=" << xs.size() << " mismatches=";
        for (size_t i = 0; i < xs.size(); ++i) {
            if (got[i] != host[i]) {
                if (mismatches < 100)
                    std::cout << "\n  i=" << i << " x_bits=0x" << std::hex << bits(xs[i])
                              << " host=0x" << host[i] << " got=0x" << got[i] << std::dec;
                ++mismatches;
                max_abs = std::max(max_abs, std::fabs((double) from_bits(got[i]) - (double) from_bits(host[i])));
            }
        }
        std::cout << "\n  count=" << mismatches << " max_abs=" << max_abs << "\n";
        if (candidate == 0 && mismatches != 0) contract_ok = false;
    }
    std::vector<float> sg, su, sh, gg, gu, gh;
    for (const auto& f : swiglu) {
        sg.insert(sg.end(), f.cpu_g.begin(), f.cpu_g.end());
        su.insert(su.end(), f.cpu_u.begin(), f.cpu_u.end());
        sh.insert(sh.end(), f.cpu_h.begin(), f.cpu_h.end());
        gg.insert(gg.end(), f.gpu_g.begin(), f.gpu_g.end());
        gu.insert(gu.end(), f.gpu_u.begin(), f.gpu_u.end());
        gh.insert(gh.end(), f.gpu_h.begin(), f.gpu_h.end());
    }
    auto swiglu_check = [&](const char* label, const std::vector<float>& g, const std::vector<float>& u,
                            const std::vector<float>& expected) {
        float *dg = nullptr, *du = nullptr; uint32_t *dy = nullptr;
        cudaMalloc(&dg, g.size() * sizeof(float)); cudaMalloc(&du, u.size() * sizeof(float)); cudaMalloc(&dy, g.size() * sizeof(uint32_t));
        cudaMemcpy(dg, g.data(), g.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(du, u.data(), u.size() * sizeof(float), cudaMemcpyHostToDevice);
        eval_swiglu_kernel<<<(unsigned)((g.size() + 255) / 256), 256>>>(dg, du, dy, (int) g.size());
        cudaDeviceSynchronize();
        std::vector<uint32_t> got(g.size()), want(g.size());
        cudaMemcpy(got.data(), dy, got.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < expected.size(); ++i) want[i] = bits(expected[i]);
        size_t mismatches = 0, first = 0;
        for (size_t i = 0; i < got.size(); ++i) if (got[i] != want[i]) { if (!mismatches) first = i; ++mismatches; }
        std::cout << "swiglu=" << label << " corpus=" << got.size() << " mismatches=" << mismatches;
        if (mismatches) std::cout << " first=" << first << " expected=0x" << std::hex << want[first] << " got=0x" << got[first] << std::dec;
        std::cout << "\n";
        cudaFree(dg); cudaFree(du); cudaFree(dy);
        return mismatches == 0;
    };
    const bool cpu_swiglu_ok = swiglu_check("cpu_inputs", sg, su, sh);
    const bool gpu_swiglu_ok = swiglu_check("gpu_inputs", gg, gu, gh);
    cudaFree(dx); cudaFree(dy);
    return (contract_ok && cpu_swiglu_ok && gpu_swiglu_ok) ? 0 : 1;
}
