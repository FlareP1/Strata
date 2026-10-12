# 2026-10-04: `--expert-math fast|exact`

Question: the bit-exact GPU expert kernels (the GPU reproducing the CPU pool's ggml contract: Q8_K / Q8_0 activations,
the AVX2 accumulation and reduction order, glibc's `expf`) made decode about 2.4x slower than llama.cpp's CUDA
arithmetic. Is the exactness worth it by default, and can the exact kernels be faster?

Change: `--expert-math fast` (the default) runs the GPU's resident experts with llama.cpp's q8_1 kernels, as the engine
did before the exact kernels; `--expert-math exact` keeps the bit-exact ones. The exact dots now take four rows per warp
(one per eight-lane group, each group in the CPU's own order), where they used to leave 24 of 32 lanes idle.

UD-Q4_K_XL_2 native pack, RTX 3090 (main) + RTX 3080 Ti (second GPU), i5-14600K (AVX2, no AVX-512), CUDA 13.1.

## Exactness

Teacher-forced decode (64-token prefix, 960 followed tokens), production expert arguments, `--deterministic-adapt
--no-prompt-offload`, `--window-hashes` (a hash of the hidden state per verify window) and `--window-logits`:

| Run | Resident experts | Window hashes | Logits |
|---|---:|---|---|
| `--disable-hit` (every routed expert on the CPU) | 0 | reference | reference |
| before this change (exact, one row per warp) | 5,505 | identical (627/627) | identical |
| `--expert-math exact` (four rows per warp) | 5,505 | identical (627/627) | identical |
| `--expert-math fast`, twice | 5,505 | differ from the CPU, identical to each other | see below |

`native_expert_parity` on real experts (layers 0-5, 10, 20, 30, 40, 46: Q4_K/Q5_1, Q5_K/Q8_0, Q4_K/Q8_0) and on
synthetic Q4_K/Q5_K x Q5_1/Q8_0: CPU - GPU = 0 and 0 differing hidden bytes, as before. `expert_grouped_parity`: PASS
(15 format pairs).

## Quality of `fast`

Same method, CPU (`--disable-hit`) vs `--expert-math fast`, three texts:

| Text | Positions | Same top-1 | Flips at a near-tie (< 0.5 nats) | Fast - CPU (nats per token) |
|---|---:|---:|---:|---:|
| StrataKV task document | 885 | 91.9% | 74 of 78 | +0.0055 +- 0.0060 |
| C++ source | 1,941 | 97.8% | 41 of 44 | +0.0007 +- 0.0032 |
| Prose (docs) | 1,887 | 93.9% | 111 of 121 | -0.0048 +- 0.0048 |
| **All 4,713** | | **95.1%** | **226 of 243** | **+0.0001 +- 0.0024** |

Perplexity ratio fast / CPU: 95% interval 0.995 - 1.005. As in `2026-09-27-cache-parity`: the two change which of two
nearly equal candidates wins, and neither is more accurate. For scale, the per-expert error of either path against
fp64 arithmetic on the dequantized weights is ~1.3% (`native_expert_parity`, "cpu rel" / "gpu rel"), from rounding
the activations to 8 bits; fast and exact are two equally good roundings of the same quantized expert.

## Speed

The 2026-10-04 regression workload: 8,352-token prompt (`short-8192` + suffix), 256 greedy tokens, MTP spec 4, async
adaptation, prompt offload on, prompt cache off; one server per arm, warmup, three cycles, medians (tokens/s):

| Arm | Prompt (cold) | Decode (cold) | Decode (repeat) | Decode (continuation) |
|---|---:|---:|---:|---:|
| exact, one row per warp (before) | 2,156 | 54.5 | 55.8 | 55.0 |
| `--expert-math exact`, four rows per warp | 2,155 | 92.7 | 96.5 | 94.2 |
| `--expert-math fast` | 2,158 | 127.6 | 134.1 | 163.7 |

All arms emitted the same text. Production configuration (deterministic adaptation, prompt cache 16 / 16 GiB, vision):

| Arm | Prompt (first, uncached) | Decode, median of 9 requests | Range |
|---|---:|---:|---:|
| exact, one row per warp (before) | 1,893 | 43.6 | 27.2 - 56.5 |
| `--expert-math exact`, four rows per warp | 1,935 | 58.9 | 48.5 - 96.1 |
| `--expert-math fast` | 1,943 | 122.9 | 91.9 - 146.2 |

(With the prompt cache on, the later requests reuse the prompt, so only the first one measures prompt processing.
Deterministic adaptation moves thousands of experts per request here, which makes decode vary from request to request.)

`exact` reproduces ggml's AVX2 kernels. On a CPU where ggml uses another kernel set (AVX-512, ARM) the CPU pool sums in
another order, and `exact` is only as exact as a check against `--disable-hit` shows.

## On `custom` 3a19944 (2026-10-05)

The same checks after porting onto the current `custom` (prefix 64 + 960 followed tokens, production expert arguments,
default asynchronous adaptation, `--no-prompt-offload`):

| Run | Window hashes / logits |
|---|---|
| `--disable-hit` (all on the CPU) | reference |
| `--expert-math exact`, twice | identical to the reference (3,235 moves dropped by the async tier: residency differs, output does not) |
| `--expert-math fast` vs this `custom` without the PR, `--adapt-every 0` | identical to each other: the default is upstream's arithmetic |

Speed, 8,352-token prompt, medians of three cycles (tok/s): `custom` 130.5, `--expert-math fast` 130.8,
`--expert-math exact` 97.9 (repeat decode); prompt processing 2,152-2,158 in all three.
