// Debug-only replay of one live native expert row captured by the verifier's MoE sidecar.
// Usage:
//   native_live_expert_parity <model-shard.gguf> <details> <pos0> <layer> <expert> <router-index>
// It feeds the captured quantized gate/up activation to both the CPU native path and the grouped GPU path,
// preserving the exact Q5_1/Q8_1 contract used by the full verifier.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

struct DetailHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t T;
    int64_t pos0;
    int64_t layer;
    uint64_t n_embd;
    uint64_t n_ff;
    uint64_t k;
    uint64_t act_bytes;
    uint32_t act_type;
    uint32_t meta_words;
};

static void must(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "%s\n", what); std::exit(2); }
}
static void cuda_must(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}

// Versioned stage fixture.  The payload is deliberately self-describing and
// carries a checksum so later host/device expf experiments do not depend on a
// mutable sidecar or on rerunning the full model.
struct StageFixtureHeader {
    uint64_t magic;          // "STRASTG1" in little-endian
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
static_assert(sizeof(StageFixtureHeader) % 8 == 0, "stage fixture header alignment");

static uint64_t fnv1a(const std::vector<uint8_t>& p) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t x : p) { h ^= x; h *= 1099511628211ull; }
    return h;
}
template<typename T>
static void append_bytes(std::vector<uint8_t>& out, const T* p, size_t n) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(p);
    out.insert(out.end(), b, b + n * sizeof(T));
}
static uint32_t f32_bits(float x) {
    uint32_t u = 0; std::memcpy(&u, &x, sizeof u); return u;
}
static float f16_value(const uint8_t* p) {
    const uint16_t u = (uint16_t) p[0] | ((uint16_t) p[1] << 8);
    return ggml_fp16_to_fp32((ggml_fp16_t) u);
}
static void print_q81_block(const char* name, const std::vector<uint8_t>& hq, int block) {
    const size_t off = (size_t) block * 36;
    if (off + 36 > hq.size()) return;
    int sum = 0;
    std::cout << name << " block=" << block
              << " d=" << f16_value(hq.data() + off)
              << " s=" << f16_value(hq.data() + off + 2)
              << " d_bits=0x" << std::hex << ((unsigned) hq[off] | ((unsigned) hq[off + 1] << 8))
              << " s_bits=0x" << ((unsigned) hq[off + 2] | ((unsigned) hq[off + 3] << 8)) << std::dec << " q=";
    for (int i = 0; i < 32; ++i) { const int q = (int8_t) hq[off + 4 + (size_t) i]; sum += q; if (i) std::cout << ","; std::cout << q; }
    std::cout << " sumq=" << sum << "\n";
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: %s MODEL DETAILS POS0 LAYER EXPERT ROUTER_INDEX\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1], detail_path = argv[2];
    const int64_t want_pos = std::stoll(argv[3]), layer = std::stoll(argv[4]);
    const int expert = std::stoi(argv[5]), route_index = std::stoi(argv[6]);

    std::ifstream in(detail_path, std::ios::binary);
    must((bool) in, "cannot open details");
    DetailHeader dh{};
    std::vector<uint8_t> act;
    std::vector<float> weights;
    std::vector<int32_t> ids;
    std::vector<int32_t> meta;
    bool found = false;
    while (in.read((char*) &dh, sizeof(dh))) {
        must(dh.magic == 0x535452414d4f4532ull && dh.version == 1, "bad details header");
        weights.resize((size_t) dh.k);
        ids.resize((size_t) dh.k);
        act.resize((size_t) dh.act_bytes);
        meta.resize((size_t) dh.meta_words);
        in.read((char*) weights.data(), (std::streamsize) weights.size() * sizeof(float));
        in.read((char*) ids.data(), (std::streamsize) ids.size() * sizeof(int32_t));
        in.read((char*) act.data(), (std::streamsize) act.size());
        in.read((char*) meta.data(), (std::streamsize) meta.size() * sizeof(int32_t));
        must((bool) in, "truncated details record");
        if (dh.pos0 == want_pos && dh.layer == layer) { found = true; break; }
    }
    must(found, "requested position/layer not found in details");
    must(route_index >= 0 && (uint64_t) route_index < dh.k, "router index out of range");
    must(ids[(size_t) route_index] == expert, "router index does not name requested expert");
    must(dh.n_embd == 2560 && dh.n_ff == 640, "unexpected model geometry");
    must(dh.act_type == 15, "this replay currently requires Q8_K gate/up activation");

    const strata::GgufModel model(strata::gguf_split_paths(model_path));
    const strata::TensorInfo* t[3] = {};
    const uint8_t* data[3] = {};
    const char* roles[3] = {"gate", "up", "down"};
    for (int r = 0; r < 3; ++r) {
        size_t at = 0;
        t[r] = model.find("blk." + std::to_string(layer) + ".ffn_" + roles[r] + "_exps.weight", &at);
        if (t[r]) data[r] = model.shard(at).tensor_data(*t[r]);
    }
    must(t[0] && t[1] && t[2], "expert tensors not found");
    cpu::NativeFmt f;
    std::string err;
    must(cpu::native_fmt((int) t[0]->type, (int) t[2]->type, dh.n_embd, dh.n_ff, f, err), err.c_str());
    std::vector<uint8_t> blob(f.bytes);
    std::memcpy(blob.data(), data[0] + (size_t) expert * f.up_off, f.up_off);
    std::memcpy(blob.data() + f.up_off, data[1] + (size_t) expert * f.up_off, f.up_off);
    std::memcpy(blob.data() + f.down_off, data[2] + (size_t) expert * (f.bytes - f.down_off), f.bytes - f.down_off);
    must(f.gu_act == 15 && (f.d_type == 7 || f.d_type == 8) && (f.d_act == 9 || f.d_act == 8),
         "requested row is not a native Q4_K/Q5_1 or Q8_0 DOWN contract");
    must(act.size() >= f.act_bytes, "captured activation is shorter than Q8_K contract");

    std::vector<uint8_t> hq(f.h_bytes);
    std::vector<float> cpu_h((size_t) f.n_ff), cpu_out((size_t) f.n_embd);
    const void* actp[1] = {act.data()};
    float* hp[1] = {cpu_h.data()};
    float* op[1] = {cpu_out.data()};
    cpu::native_gu_rows(f, blob.data(), actp, 1, hp, 0, (int) f.n_ff);
    std::vector<float> cpu_g((size_t) f.n_ff), cpu_u((size_t) f.n_ff);
    const auto* tc = ggml_get_type_traits_cpu((ggml_type) f.gu_type);
    for (int64_t r = 0; r < f.n_ff; ++r) {
        tc->vec_dot((int) f.n_embd, &cpu_g[(size_t) r], 0, blob.data() + (size_t) r * f.gu_row, 0,
                    act.data(), 0, 1);
        tc->vec_dot((int) f.n_embd, &cpu_u[(size_t) r], 0, blob.data() + f.up_off + (size_t) r * f.gu_row, 0,
                    act.data(), 0, 1);
    }
    cpu::native_quant_h(f, cpu_h.data(), hq.data());
    const void* hqp[1] = {hq.data()};
    cpu::native_down_rows(f, blob.data(), hqp, 1, op, 0, (int) f.n_embd);

    cudaStream_t s = nullptr;
    cuda_must(cudaStreamCreate(&s), "cudaStreamCreate");
    void *dblob = nullptr, *dact = nullptr, *dscr = nullptr, *dhq_cpu = nullptr, *dhq_gpu = nullptr;
    float* dout = nullptr;
    float *dout_cpu_hidden = nullptr, *dout_gpu_hidden = nullptr;
    float* ddebug_h = nullptr;
    float* ddebug_gu = nullptr;
    unsigned long long* dptr = nullptr;
    int32_t *dstart = nullptr, *dn = nullptr, *ddst = nullptr, *dtok = nullptr;
    cuda_must(cudaMalloc(&dblob, blob.size()), "blob alloc");
    cuda_must(cudaMalloc(&dact, (size_t) dh.act_bytes), "activation alloc");
    cuda_must(cudaMalloc(&dscr, strata::kernels::native_expert_scratch_bytes(1, f.n_ff)), "scratch alloc");
    cuda_must(cudaMalloc((void**) &dout, (size_t) f.n_embd * sizeof(float)), "output alloc");
    cuda_must(cudaMalloc(&dhq_cpu, f.h_bytes), "cpu hidden alloc");
    cuda_must(cudaMalloc(&dhq_gpu, f.h_bytes), "gpu hidden alloc");
    cuda_must(cudaMalloc((void**) &dout_cpu_hidden, (size_t) f.n_embd * sizeof(float)), "cpu hidden output alloc");
    cuda_must(cudaMalloc((void**) &dout_gpu_hidden, (size_t) f.n_embd * sizeof(float)), "gpu hidden output alloc");
    cuda_must(cudaMalloc((void**) &ddebug_h, (size_t) f.n_ff * sizeof(float)), "debug hidden alloc");
    cuda_must(cudaMalloc((void**) &ddebug_gu, (size_t) 2 * f.n_ff * sizeof(float)), "debug GU alloc");
    cuda_must(cudaMalloc((void**) &dptr, sizeof(unsigned long long)), "ptr alloc");
    cuda_must(cudaMalloc((void**) &dstart, 2 * sizeof(int32_t)), "start alloc");
    cuda_must(cudaMalloc((void**) &dn, sizeof(int32_t)), "groups alloc");
    cuda_must(cudaMalloc((void**) &ddst, sizeof(int32_t)), "dst alloc");
    cuda_must(cudaMalloc((void**) &dtok, sizeof(int32_t)), "tok alloc");
    cuda_must(cudaMemcpy(dblob, blob.data(), blob.size(), cudaMemcpyHostToDevice), "blob copy");
    cuda_must(cudaMemcpy(dact, act.data(), (size_t) dh.act_bytes, cudaMemcpyHostToDevice), "activation copy");
    const unsigned long long p = (unsigned long long) dblob;
    const int32_t st[2] = {0, 1}, one = 1, zero = 0;
    cuda_must(cudaMemcpy(dptr, &p, sizeof(p), cudaMemcpyHostToDevice), "ptr copy");
    cuda_must(cudaMemcpy(dstart, st, sizeof(st), cudaMemcpyHostToDevice), "start copy");
    cuda_must(cudaMemcpy(dn, &one, sizeof(one), cudaMemcpyHostToDevice), "groups copy");
    cuda_must(cudaMemcpy(ddst, &zero, sizeof(zero), cudaMemcpyHostToDevice), "dst copy");
    cuda_must(cudaMemcpy(dtok, &zero, sizeof(zero), cudaMemcpyHostToDevice), "tok copy");
    const auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff, f.gu_act, f.d_act);
    strata::kernels::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, 1, 1, dact,
                                           (int64_t) dh.act_bytes, dscr, dout, s, ddebug_h, ddebug_gu);
    cuda_must(cudaStreamSynchronize(s), "grouped synchronize");
    std::vector<float> gpu_out((size_t) f.n_embd);
    cuda_must(cudaMemcpy(gpu_out.data(), dout, gpu_out.size() * sizeof(float), cudaMemcpyDeviceToHost), "output copy");
    std::vector<uint8_t> gpu_hq(f.h_bytes);
    cuda_must(cudaMemcpy(gpu_hq.data(), dscr, gpu_hq.size(), cudaMemcpyDeviceToHost), "hidden copy");
    // Four-way DOWN matrix: feed each hidden byte buffer through both DOWN
    // implementations, independently of GU/SwiGLU.
    std::vector<float> cpu_down_gpu_hidden((size_t) f.n_embd);
    const void* gpu_hqp[1] = {gpu_hq.data()};
    float* cpu_down_gpu_out[1] = {cpu_down_gpu_hidden.data()};
    cpu::native_down_rows(f, blob.data(), gpu_hqp, 1, cpu_down_gpu_out, 0, (int) f.n_embd);
    cuda_must(cudaMemcpy(dhq_cpu, hq.data(), hq.size(), cudaMemcpyHostToDevice), "cpu hidden upload");
    cuda_must(cudaMemcpy(dhq_gpu, gpu_hq.data(), gpu_hq.size(), cudaMemcpyHostToDevice), "gpu hidden upload");
    cuda_must(cudaMemsetAsync(dout_cpu_hidden, 0, (size_t) f.n_embd * sizeof(float), s), "cpu hidden output clear");
    cuda_must(cudaMemsetAsync(dout_gpu_hidden, 0, (size_t) f.n_embd * sizeof(float), s), "gpu hidden output clear");
    strata::kernels::native_expert_down_only(L, dptr, dstart, dn, ddst, dhq_cpu, 1, dout_cpu_hidden, s);
    strata::kernels::native_expert_down_only(L, dptr, dstart, dn, ddst, dhq_gpu, 1, dout_gpu_hidden, s);
    cuda_must(cudaStreamSynchronize(s), "four-way DOWN synchronize");
    std::vector<float> gpu_down_cpu_hidden((size_t) f.n_embd);
    std::vector<float> gpu_down_gpu_hidden((size_t) f.n_embd);
    cuda_must(cudaMemcpy(gpu_down_cpu_hidden.data(), dout_cpu_hidden, gpu_down_cpu_hidden.size() * sizeof(float), cudaMemcpyDeviceToHost), "cpu hidden GPU DOWN copy");
    cuda_must(cudaMemcpy(gpu_down_gpu_hidden.data(), dout_gpu_hidden, gpu_down_gpu_hidden.size() * sizeof(float), cudaMemcpyDeviceToHost), "gpu hidden GPU DOWN copy");
    std::vector<float> gpu_h((size_t) f.n_ff);
    cuda_must(cudaMemcpy(gpu_h.data(), ddebug_h, gpu_h.size() * sizeof(float), cudaMemcpyDeviceToHost), "debug hidden copy");
    std::vector<float> gpu_gu((size_t) 2 * f.n_ff);
    cuda_must(cudaMemcpy(gpu_gu.data(), ddebug_gu, gpu_gu.size() * sizeof(float), cudaMemcpyDeviceToHost), "debug GU copy");

    if (const char* dump = std::getenv("STRATA_STAGE_DUMP"); dump != nullptr && *dump != '\0') {
        std::vector<uint8_t> payload;
        append_bytes(payload, ids.data(), ids.size());
        append_bytes(payload, weights.data(), weights.size());
        append_bytes(payload, act.data(), act.size());
        append_bytes(payload, cpu_g.data(), cpu_g.size());
        append_bytes(payload, gpu_gu.data(), f.n_ff);
        append_bytes(payload, cpu_u.data(), cpu_u.size());
        append_bytes(payload, gpu_gu.data() + f.n_ff, f.n_ff);
        append_bytes(payload, cpu_h.data(), cpu_h.size());
        append_bytes(payload, gpu_h.data(), gpu_h.size());
        append_bytes(payload, hq.data(), hq.size());
        append_bytes(payload, gpu_hq.data(), gpu_hq.size());
        append_bytes(payload, cpu_out.data(), cpu_out.size());
        append_bytes(payload, gpu_out.data(), gpu_out.size());
        append_bytes(payload, cpu_down_gpu_hidden.data(), cpu_down_gpu_hidden.size());
        append_bytes(payload, gpu_down_cpu_hidden.data(), gpu_down_cpu_hidden.size());
        append_bytes(payload, gpu_down_gpu_hidden.data(), gpu_down_gpu_hidden.size());
        const bool embed_blob = std::getenv("STRATA_STAGE_EMBED_BLOB") != nullptr;
        if (embed_blob) append_bytes(payload, blob.data(), blob.size());
        StageFixtureHeader sh{};
        sh.magic = 0x3147545341525453ull;
        sh.version = embed_blob ? 2 : 1;
        sh.header_bytes = sizeof(sh);
        sh.payload_bytes = payload.size();
        sh.payload_fnv1a = fnv1a(payload);
        sh.pos0 = dh.pos0; sh.layer = layer; sh.expert = expert; sh.route_index = route_index;
        sh.route_weight = weights[(size_t) route_index];
        sh.n_embd = f.n_embd; sh.n_ff = f.n_ff; sh.k = dh.k; sh.act_bytes = dh.act_bytes;
        sh.h_bytes = f.h_bytes; sh.blob_bytes = f.bytes;
        sh.act_type = dh.act_type; sh.gu_type = f.gu_type; sh.d_type = f.d_type;
        sh.gu_act = f.gu_act; sh.d_act = f.d_act;
        std::ofstream out(dump, std::ios::binary);
        must((bool) out, "cannot create stage fixture");
        out.write((const char*) &sh, sizeof sh);
        out.write((const char*) payload.data(), (std::streamsize) payload.size());
        must((bool) out, "cannot write stage fixture");
        std::cout << "stage_fixture=" << dump << " payload_bytes=" << payload.size()
                  << " fnv1a=0x" << std::hex << sh.payload_fnv1a << std::dec << "\n";
    }

    if ((expert == 148 && layer == 35 && want_pos == 12) || std::getenv("STRATA_PRINT_STAGE") != nullptr) {
        const int idx = std::min<int>(542, (int) f.n_ff - 1);
        const float host_e = ::expf(-cpu_g[(size_t) idx]);
        const float host_silu = (cpu_g[(size_t) idx] / (1.0f + host_e)) * cpu_u[(size_t) idx];
        const float candidate_e = gpu_h[(size_t) idx] != 0.0f
            ? cpu_g[(size_t) idx] * cpu_u[(size_t) idx] / gpu_h[(size_t) idx] - 1.0f : 0.0f;
        std::cout << "stage542 gate_cpu=" << cpu_g[(size_t) idx] << " bits=0x" << std::hex << f32_bits(cpu_g[(size_t) idx])
                  << " gate_gpu_bits=0x" << f32_bits(gpu_gu[(size_t) idx]) << " up_cpu_bits=0x"
                  << f32_bits(cpu_u[(size_t) idx]) << " up_gpu_bits=0x" << f32_bits(gpu_gu[f.n_ff + (size_t) idx])
                  << std::dec << "\n";
        std::cout << "stage542 host_expf_neg_gate=" << host_e << " bits=0x" << std::hex << f32_bits(host_e)
                  << " candidate_expf_estimate=" << candidate_e << " bits=0x" << f32_bits(candidate_e)
                  << " host_hidden=" << host_silu << " bits=0x" << f32_bits(host_silu)
                  << " cpu_hidden=" << cpu_h[(size_t) idx] << " bits=0x" << f32_bits(cpu_h[(size_t) idx])
                  << " gpu_hidden=" << gpu_h[(size_t) idx] << " bits=0x" << f32_bits(gpu_h[(size_t) idx]) << std::dec << "\n";
        auto block_stats = [](const std::vector<float>& h, int b) {
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(h[(size_t) b * 32 + i]));
            return amax;
        };
        const int b = 16;
        const float cpu_amax = block_stats(cpu_h, b), gpu_amax = block_stats(gpu_h, b);
        std::cout << "block16_cpu amax=" << cpu_amax << " bits=0x" << std::hex << f32_bits(cpu_amax)
                  << " d32_bits=0x" << f32_bits(cpu_amax / 127.0f)
                  << " id_bits=0x" << f32_bits(cpu_amax == 0.0f ? 0.0f : 127.0f / cpu_amax) << std::dec << "\n";
        std::cout << "block16_gpu amax=" << gpu_amax << " bits=0x" << std::hex << f32_bits(gpu_amax)
                  << " d32_bits=0x" << f32_bits(gpu_amax / 127.0f)
                  << " id_bits=0x" << f32_bits(gpu_amax == 0.0f ? 0.0f : 127.0f / gpu_amax) << std::dec << "\n";
        print_q81_block("block16_cpu_q81", hq, b);
        print_q81_block("block16_gpu_q81", gpu_hq, b);
    }

    double max_abs = 0, sum2 = 0, sumabs = 0, ref2 = 0;
    for (size_t i = 0; i < cpu_out.size(); ++i) {
        const double d = (double) gpu_out[i] - cpu_out[i];
        max_abs = std::max(max_abs, std::fabs(d)); sum2 += d * d; sumabs += std::fabs(d); ref2 += (double) cpu_out[i] * cpu_out[i];
    }
    size_t hdiff = 0;
    for (size_t i = 0; i < hq.size(); ++i) hdiff += hq[i] != gpu_hq[i];
    double hmax = 0, hsum2 = 0;
    for (size_t i = 0; i < cpu_h.size(); ++i) {
        const double d = (double) gpu_h[i] - cpu_h[i];
        hmax = std::max(hmax, std::fabs(d)); hsum2 += d * d;
    }
    double gmax = 0, umax = 0;
    for (size_t i = 0; i < cpu_g.size(); ++i) {
        gmax = std::max(gmax, std::fabs((double) gpu_gu[i] - cpu_g[i]));
        umax = std::max(umax, std::fabs((double) gpu_gu[cpu_g.size() + i] - cpu_u[i]));
    }
    auto row_error = [](const std::vector<float>& a, const std::vector<float>& b) {
        double mx = 0, ss = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double d = (double) a[i] - b[i];
            mx = std::max(mx, std::fabs(d)); ss += d * d;
        }
        return std::pair<double, double>(mx, std::sqrt(ss / a.size()));
    };
    const auto cpu_cpu = row_error(cpu_out, cpu_out);
    const auto cpu_gpu = row_error(cpu_out, cpu_down_gpu_hidden);
    const auto gpu_cpu = row_error(cpu_out, gpu_down_cpu_hidden);
    const auto gpu_gpu = row_error(cpu_out, gpu_down_gpu_hidden);
    const double weight = weights[(size_t) route_index];
    std::cout << "pos0=" << dh.pos0 << " layer=" << layer << " expert=" << expert << " route_index=" << route_index
              << " gu_type=" << f.gu_type << " down_type=" << f.d_type << " weight=" << weight << "\n"
              << "hidden_bytes=" << hq.size() << " hidden_byte_diffs=" << hdiff << "\n"
              << "hidden_float_max_abs=" << hmax << " hidden_float_rms=" << std::sqrt(hsum2 / cpu_h.size()) << "\n"
              << "gate_float_max_abs=" << gmax << " up_float_max_abs=" << umax << "\n"
              << "raw_max_abs=" << max_abs << " raw_rms=" << std::sqrt(sum2 / cpu_out.size())
              << " raw_mean_abs=" << sumabs / cpu_out.size() << " raw_relative_l2=" << std::sqrt(sum2 / (ref2 + 1e-30)) << "\n"
              << "weighted_max_abs=" << max_abs * weight << " weighted_rms=" << std::sqrt(sum2 / cpu_out.size()) * weight << "\n";
    std::cout << "four_way_down cpu_cpu_max_abs=" << cpu_cpu.first << " cpu_cpu_rms=" << cpu_cpu.second
              << " cpu_gpu_max_abs=" << cpu_gpu.first << " cpu_gpu_rms=" << cpu_gpu.second
              << " gpu_cpu_max_abs=" << gpu_cpu.first << " gpu_cpu_rms=" << gpu_cpu.second
              << " gpu_gpu_max_abs=" << gpu_gpu.first << " gpu_gpu_rms=" << gpu_gpu.second << "\n";
    std::cout << "cpu_first=";
    for (int i = 0; i < 8; ++i) std::cout << (i ? "," : "") << cpu_out[(size_t) i];
    std::cout << "\ngpu_first=";
    for (int i = 0; i < 8; ++i) std::cout << (i ? "," : "") << gpu_out[(size_t) i];
    std::cout << "\n";
    std::cout << "hidden_byte_diff_indices=";
    bool first_diff = true;
    for (size_t i = 0; i < hq.size(); ++i) if (hq[i] != gpu_hq[i]) {
        if (!first_diff) std::cout << ",";
        first_diff = false;
        std::cout << i << ":" << (unsigned) hq[i] << ">" << (unsigned) gpu_hq[i];
    }
    std::cout << "\n";
    if (gmax != 0.0 || umax != 0.0 || hmax != 0.0) {
        std::cout << "float_diff_samples=";
        bool printed = false;
        int sample_count = 0;
        for (size_t i = 0; i < cpu_g.size(); ++i) {
            const float cg = cpu_g[i], gg = gpu_gu[i];
            const float cu = cpu_u[i], gu = gpu_gu[cpu_g.size() + i];
            if (f32_bits(cg) != f32_bits(gg) || f32_bits(cu) != f32_bits(gu) ||
                f32_bits(cpu_h[i]) != f32_bits(gpu_h[i])) {
                if (printed) std::cout << ";";
                printed = true;
                std::cout << i << ":g=" << std::hex << f32_bits(cg) << ">" << f32_bits(gg)
                          << ",u=" << f32_bits(cu) << ">" << f32_bits(gu)
                          << ",h=" << f32_bits(cpu_h[i]) << ">" << f32_bits(gpu_h[i]) << std::dec;
                if (++sample_count == 16) break;
            }
        }
        std::cout << "\n";
    }
    if (const char* dump = std::getenv("STRATA_LIVE_PARITY_DUMP"); dump != nullptr && *dump != '\0') {
        std::ofstream d(dump, std::ios::binary);
        if (d) {
            d.write((const char*) cpu_out.data(), (std::streamsize) cpu_out.size() * sizeof(float));
            d.write((const char*) gpu_out.data(), (std::streamsize) gpu_out.size() * sizeof(float));
        }
    }
    if (const char* dump = std::getenv("STRATA_LIVE_PARITY_HIDDEN_DUMP"); dump != nullptr && *dump != '\0') {
        std::ofstream d(dump, std::ios::binary);
        if (d) {
            const uint32_t nff = (uint32_t) f.n_ff, hbytes = (uint32_t) hq.size();
            d.write((const char*) &nff, sizeof nff);
            d.write((const char*) &hbytes, sizeof hbytes);
            d.write((const char*) cpu_h.data(), (std::streamsize) cpu_h.size() * sizeof(float));
            d.write((const char*) gpu_h.data(), (std::streamsize) gpu_h.size() * sizeof(float));
            d.write((const char*) hq.data(), (std::streamsize) hq.size());
            d.write((const char*) gpu_hq.data(), (std::streamsize) gpu_hq.size());
        }
    }
    const bool strict = std::getenv("STRATA_STRICT_CONTRACT") != nullptr;
    const bool pass = hdiff == 0 && gmax == 0.0 && umax == 0.0 && max_abs == 0.0 &&
                      cpu_gpu.first == 0.0 && gpu_cpu.first == 0.0 && gpu_gpu.first == 0.0;
    cudaFree(dblob); cudaFree(dact); cudaFree(dscr); cudaFree(dhq_cpu); cudaFree(dhq_gpu); cudaFree(dout);
    cudaFree(dout_cpu_hidden); cudaFree(dout_gpu_hidden); cudaFree(ddebug_h); cudaFree(ddebug_gu); cudaFree(dptr); cudaFree(dstart);
    cudaFree(dn); cudaFree(ddst); cudaFree(dtok); cudaStreamDestroy(s);
    return strict && !pass ? 1 : 0;
}
