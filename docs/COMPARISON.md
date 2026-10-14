# This fork, upstream Strata and llama.cpp on one PC

Qwen3.8-Flash-Next in two quantizations, IQ3_S and UD-Q4_K_XL, through three engines on an RTX 3090 + RTX 5070 Ti PC,
measured on 2026-10-13/14: llama.cpp's mainline build 11606, upstream Strata a0116b27 (engine 0.1.43.1) and this fork
at e1ccfcbb (IQ3_S on the 3090 alone and for memory, and UD-Q4_K_XL's agreement with llama.cpp, at 2ae47d0f, whose
arithmetic there is the same).

**In short**, with both GPUs:

- **IQ3_S:** this fork reads prompts 3.5-4.0x as fast as llama.cpp and writes answers 4.9-9.4x as fast. Against
  upstream Strata's layer split: prompts 1.33x / 1.14x / 1.03x as fast at 16K / 32K / 64K tokens, 3% slower at 128K;
  answers 1.3-2.0x (the most on answers that quote the prompt). On the 3090 alone, against upstream there:
  prompts 1.7-1.8x, answers 1.1-1.7x.
- **UD-Q4_K_XL:** prompts 11.8-13.6x and answers 5.0-6.8x as fast as llama.cpp; against upstream's layer split prompts
  2.2-3.7x and answers 1.9-2.4x, on the 3090 alone prompts 1.9-2.6x and answers 1.5-2.0x.
- A follow-up turn after a 16K-token prompt starts in 0.17-0.23 s (llama.cpp 0.61-0.72 s; upstream 0.26-0.42 s on
  the 3090, 0.45-1.06 s on both GPUs).
- The next-token distributions match llama.cpp's about as closely as llama.cpp's own CPU and CUDA backends match each
  other: UD-Q4_K_XL 2.0-2.6x as far as those backends are apart (KL), from llama.cpp's 8-bit rounding of the file's
  Q8_0 hyper-connection weights; IQ3_S with the 4-bit V cache its config runs (k8v4) 1.3-3.8x (with 8-bit KV as
  close as the backends).

## The PC and the engines

- **PC:** Ryzen 7 9700X (8 cores), 160 GB DDR5-5600 (2 x 32 + 2 x 48 GB), Windows 11. An RTX 3090 (24 GB, PCIe 4.0
  x4, nothing else on it) and an RTX 5070 Ti (16 GB, PCIe 5.0 x16, driving the display).
- **Models:** ISTA-DASLab's GSQ-RCO **IQ3_S** (shard 1 54.8 GB: experts 46.8 GiB, i-quants with Q2_0 and IQ4_NL down
  projections) and unsloth's **UD-Q4_K_XL** (111 GB: experts 71.7 GiB, Q4_K / Q5_K with Q5_1 / Q8_0 downs). Each has
  a 28.8 GB IQ4_NL n-gram table (the per-layer embeddings) besides.
- **llama.cpp** build 11606 (1b70350c9, MSVC), as mainline builds it. Both GPUs with the layers split (the 5070 Ti
  first), the experts that do not fit in VRAM in RAM: UD-Q4_K_XL by `--fit`, IQ3_S by fit's placement with its Q2_0 down
  projections kept on the GPUs (llama.cpp has no x86 SIMD kernel for Q2_0: 30.5 instead of 27.2 t/s). 8-bit KV cache, 8
  threads, 1024-token micro-batches, the n-gram table in RAM. No speculative decoding: llama.cpp has no graph for this
  model's MTP layer.
- **Upstream Strata** a0116b27 (engine 0.1.43.1) with the configs its setup writes for each file on this PC: MTP
  drafts, 8-bit KV kept in RAM past 32K cells, the expert cache sized by the free VRAM; for UD-Q4_K_XL every expert in
  RAM (its RAM budget of 71 GiB) and its own pack (`tools/iq_pack.py --compat-bf16`). On the 3090, and on both GPUs
  with a layer split, the newer 5070 Ti first (its setup's choice here for IQ3_S; for UD-Q4_K_XL it recommends one GPU
  and offers both). Its route tail skip is on, as by default on CUDA: in a verify window, a missed expert that every
  token ranks 7th or lower is not computed (an approximation; this fork computes every routed expert).
- **This fork** with its configs ([examples](../examples/README.md)): the 3090 runs the model and the 5070 Ti is a
  second expert tier (12.2-12.4 GiB of experts; its share of each layer's experts, the next layer's likely experts
  copied ahead, the prompt path's experts the 3090 lacks, and with IQ3_S part of the output head). MTP drafts and
  prompt lookup, KV in VRAM (IQ3_S 8-bit keys and 4-bit values, k8v4; UD-Q4_K_XL 8-bit), 16K-token prompt chunks.
  For UD-Q4_K_XL the experts the CPU reads are kept out of the RAM's slower region (`--arena-place`: the unpaired
  DIMMs leave 32 GB on one channel; timed at the start, placed again between requests); for IQ3_S the 5070 Ti also
  computes a fifth of the experts the 3090 holds in long prompts (`--prompt-shift 0.2`). And on the 3090 alone.

## Speed through the API

The same requests to each server's OpenAI API: greedy, one at a time, the prompt caches off except for the follow-up
turn; this fork's UD-Q4_K_XL server got them 4 s apart, the time it uses to place its RAM arena again. Where that
arena lands varies from start to start, and its generation with it by 5-10% (these runs: 26-27 GiB of the 72 in the
RAM's slower region, among the worst seen). Prompts: 16K-128K tokens of llama.cpp's sources with a question (400 tokens of answer), a 28-token coding request
(up to 300), a coding task with thinking on (3,000), and a Python module to return with a class renamed (the edit, up
to 3,000 tokens that mostly quote the prompt). The follow-up turn sends the 16K prompt, its answer and a new question:
its prompt time is the processing of what the engine did not keep. Each engine writes its own text and the texts part
at the first near-tie, so the generation speeds compare different texts of one kind, one run each. The memory rows
come from a second start of each server (the model file then in the OS cache, as far as it keeps it: this fork read
IQ3_S's experts from it at 6.8-7.0 GiB/s and UD-Q4_K_XL's, partly cached, at 4.9-5.6, against 4.1-4.4 from the SSD):
the GPUs' memory taken, and the RAM taken as the drop in the system's available memory.

### IQ3_S

<!-- matrix:api-iq3_s -->
| | llama.cpp, both GPUs | upstream Strata, 3090 | upstream Strata, both GPUs | this fork, both GPUs | this fork, 3090 |
|---|---|---|---|---|---|
| Prompt, 16K tokens (t/s) | 885 | 1172 | 2308 | 3077 | 2016 |
| Prompt, 32K tokens (t/s) | 851 | 1236 | 2927 | 3324 | 2228 |
| Prompt, 64K tokens (t/s) | 837 | 1232 | 3250 | 3345 | 2144 |
| Prompt, 128K tokens (t/s) | 805 | 1203 | 3371 | 3258 | 2132 |
| Generation: code, 28-token prompt (t/s) | 29.6 | 93.0 | 117.1 | 194.0 | 101.7 |
| Generation: thinking, 3,000 tokens | 31.2 | 82.0 | 117.8 | 153.5 | 128.8 |
| Generation: edit, the prompt's file returned (2.4K tokens) | 31.1 | 115.8 | 148.3 | 290.9 | 200.7 |
| Generation after the 16K prompt | 30.8 | 84.3 | 117.6 | 184.7 | 134.6 |
| Generation after the 32K prompt | 28.9 | 91.6 | 125.6 | 179.4 | 146.7 |
| Generation after the 64K prompt | 25.7 | 80.5 | 122.3 | 156.8 | 119.2 |
| Generation after the 128K prompt | 21.4 | 84.4 | 113.9 | 145.4 | 112.7 |
| Follow-up turn after the 16K prompt: new tokens, prompt time | 34 in 0.61 s | 33 in 0.26 s | 440 in 1.06 s | 34 in 0.18 s | 33 in 0.28 s |
| Start to the API, the model file cached (s) | 20 | 21 | 25 | 22 | 20 |
| VRAM taken: 3090 / 5070 Ti (GiB) | 22.5 / 12.4 | 22.7 / 0.0 | 22.5 / 14.6 | 22.6 / 13.1 | 22.7 / 0.0 |
| RAM taken (GB) | 57 | 56 | 58 | 53 | 53 |
<!-- /matrix:api-iq3_s -->

### UD-Q4_K_XL

<!-- matrix:api-ud-q4_k_xl -->
| | llama.cpp, both GPUs | upstream Strata, 3090 | upstream Strata, both GPUs | this fork, both GPUs | this fork, 3090 |
|---|---|---|---|---|---|
| Prompt, 16K tokens (t/s) | 258 | 558 | 828 | 3043 | 1424 |
| Prompt, 32K tokens (t/s) | 258 | 746 | 1416 | 3407 | 1525 |
| Prompt, 64K tokens (t/s) | 253 | 750 | 1484 | 3438 | 1427 |
| Prompt, 128K tokens (t/s) | 248 | 743 | 1519 | 3349 | 1433 |
| Generation: code, 28-token prompt (t/s) | 23.4 | 49.3 | 65.5 | 126.5 | 79.9 |
| Generation: thinking, 3,000 tokens | 24.2 | 39.5 | 54.4 | 121.0 | 79.7 |
| Generation: edit, the prompt's file returned (2.4K tokens) | 24.1 | 68.1 | 85.3 | 160.3 | 115.6 |
| Generation after the 16K prompt | 23.6 | 44.1 | 58.6 | 143.4 | 77.1 |
| Generation after the 32K prompt | 22.6 | 44.3 | 64.0 | 152.8 | 87.0 |
| Generation after the 64K prompt | 21.6 | 42.4 | 63.7 | 123.9 | 68.4 |
| Generation after the 128K prompt | 18.5 | 41.8 | 59.2 | 113.9 | 64.1 |
| Follow-up turn after the 16K prompt: new tokens, prompt time | 34 in 0.72 s | 33 in 0.42 s | 33 in 0.45 s | 33 in 0.23 s | 34 in 0.48 s |
| Start to the API, the model file cached (s) | 53 | 51 | 61 | 41 | 36 |
| VRAM taken: 3090 / 5070 Ti (GiB) | 22.3 / 12.9 | 22.7 / 0.0 | 22.8 / 14.6 | 22.7 / 13.1 | 22.7 / 0.0 |
| RAM taken (GB) | 84 | 69 | 69 | 81 | 80 |
<!-- /matrix:api-ud-q4_k_xl -->

## Generation on fixed text (this fork)

Each run emits the model's reference text (`--spec-follow`), so runs and builds compare on the same tokens and
experts: two runs, averaged, tokens a verify pass in parentheses. The code text's reference repeats its answer after
the end of the turn, which prompt lookup drafts and a server never generates.

<!-- matrix:fixed -->
| Text | IQ3_S t/s (tokens a pass) | UD-Q4_K_XL t/s (tokens a pass) |
|---|---|---|
| code (400 tokens) | 199.1 (4.06) | 121.7 (3.69) |
| thinking (1,200) | 161.1 (2.31) | 116.2 (1.90) |
| after the 16K prompt (400) | 170.0 (2.86) | 131.9 (2.82) |
| edit (2,397) | 290.4 (7.49) | 173.1 (7.51) |
<!-- /matrix:fixed -->

## Agreement with llama.cpp

Argmax agreement and mean KL divergence (nats) of the next-token distributions along llama.cpp's greedy text, which
this fork follows (`--spec-follow`, `--window-logits`), with `--adapt-every 0`. The two baselines differ from each
other only in rounding: llama.cpp's CPU backend against its CUDA backend, and this fork in one-token windows against
its four-token windows. Any change of rounding moves 2-4% of the thinking text's argmaxes, all near-ties.

<!-- matrix:parity -->
| | Code, ~260 tokens | Thinking, 1,200 | After the 16K prompt, 300 |
|---|---|---|---|
| IQ3_S: this fork vs llama.cpp | 100.0%, 5.7e-04 | 97.0%, 3.7e-03 | 98.7%, 2.0e-03 |
| llama.cpp's CPU vs CUDA backend | 100.0%, 1.5e-04 | 97.7%, 2.6e-03 | 98.7%, 1.5e-03 |
| this fork, one-token vs four-token windows | 99.6%, 2.1e-04 | 97.3%, 2.6e-03 | 98.7%, 5.9e-04 |
| UD-Q4_K_XL: this fork vs llama.cpp | 99.6%, 3.4e-04 | 96.1%, 6.4e-03 | 98.3%, 3.9e-03 |
| llama.cpp's CPU vs CUDA backend | 100.0%, 1.7e-04 | 97.7%, 2.8e-03 | 98.7%, 1.5e-03 |
| this fork, one-token vs four-token windows | 100.0%, 1.1e-04 | 97.9%, 2.2e-03 | 100.0%, 4.4e-04 |
<!-- /matrix:parity -->

## The two files

llama.cpp's perplexity (build 11116, 2026-09-30) over 28 chunks of 4,096 tokens of llama.cpp's own docs, C++ and
Python sources (lower is better): UD-Q4_K_XL 1.8849, IQ3_S 1.8946 (+0.5%), and ISTA-DASLab's IQ3_XXS, which IQ3_S
replaced in this fork's configs, 1.9215 (+1.9%). IQ3_S's experts take 46.8 GiB, UD-Q4_K_XL's 71.7: on this PC IQ3_S
generates 29-68% faster on fixed text; with both GPUs it reads prompts as fast to 3% slower, on the 3090 alone
1.4-1.5x as fast.

## Features

| | llama.cpp | Upstream Strata | This fork |
|---|---|---|---|
| Model files | any GGUF of the model | ISTA-DASLab's GSQ-RCO sizes (Q2_0 to IQ3_S), their Coder, Swift 1.5 (i-quant and Q2_0 experts); unsloth's UD-Q4_K_XL and UD-IQ4_XS | those, and every quant of unsloth's (K-quant, Q5_1 and Q8_0 experts) |
| Speculative decoding | none: no graph for this model's MTP layer | the MTP layer's drafts, up to 3 a pass; prompt lookup (up to 5 tokens where it pays) | the MTP layer's drafts; prompt lookup (up to 7 tokens after a 16+-token match) |
| Several GPUs | a layer or row split | a layer split, each card holding its layers' experts; helper expert caches on up to three more cards (experimental) | the second card as an expert tier: its share of each layer, the next layer's likely experts copied ahead, the prompt's experts the first card lacks, part of the output head |
| Experts in VRAM | a fixed placement | ranked by use, updated every 4 verify passes | ranked by use on both cards, updated every pass |
| KV cache | 8-bit, in VRAM | 8-bit, in RAM past 32K cells; 4-bit options | 8-bit, in VRAM; 4-bit options (q4_0, k8v4) in the tensor-core attention |
| Prompt cache | the live slot, states saved to RAM | the live session, checkpoints at turns and every 16K tokens, a replaced conversation parked in RAM | the live sequence, checkpoints at turns and chunks, replaced conversations kept in RAM with their K/V |
| Sampling, APIs | full; OpenAI and Anthropic | per request; OpenAI and Anthropic, a web app, MCP tools | upstream's |
