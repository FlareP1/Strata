# PR: adaptive tier: pin the refill copies so the pump cannot block the second GPU's launcher

**Base:** `eddoursul/Strata` `custom`  <-  **Head:** `FlareP1/Strata` `fix/dual-gpu-pump-deadlock`

---

## The problem

Dual-GPU decode on UD-Q4_K_XL froze completely after a few tokens (one GPU spinning ~100% on the second-GPU flag wait, the other ~0%). The IQ3_S model worked fine on the same code and the same dual-GPU config - it was only UD-Q4_K_XL that locked up.

Test machine: two RTX 3090s on an **asymmetric link setup** - the main GPU on a full Gen4 x16 slot, the second GPU on a x4 link (~3.3 GB/s measured).

Repro: `--main-gpu 1 --second-gpu 0 --adapt-every 1`; the trace always ended the same way:

```
[vtrace] rows_in enter layer 5
[vtrace] tier send enter n=6 bytes=16035840      <- cudaMemcpyBatchAsync, never returns
```

With `--adapt-every 0` the run always completed, so the adaptive-tier pump was the blocker. Free VRAM headroom (`--vram-reserve-mib 2560`) did not prevent it, so it was not paging. The lockup also appeared on the swapped config (second GPU on the fast x16 link), so it was not the slow link itself.

## Root cause

A four-party circular wait:

1. GPU0's verify graph spins on the host-pinned second-GPU flag (layer l+1 wants flag l+1), so the main stream stalls.
2. The host's per-layer `rows_in` callback pumps adaptive-tier refills via `cudaMemcpyBatchAsync` from the arena.
3. XL pins only 39/48 arena layers (`cudaHostAlloc` fails beyond that). A pump batch with blobs from the 9 **pageable** layers needs driver staging, and the enqueue **blocks** while pending copies cannot complete against the spinning main stream.
4. That same host thread launches GPU1's next-layer graph - so flag l+1 is never written, GPU0 never drains, and the copies never complete.

IQ3_S never locks because all 48 arena layers pin successfully, so the enqueue never needs staging. The swap config (fast link) locks too - the cycle only needs GPU1 to fall behind once. Freezes clustered at layers 39-47 (the pageable tail) with 9-45 MB batches; small pinned-layer batches always returned in ~0.2 ms.

## The fix - three layers of defense

1. **Pinned staging ring** (the actual deadlock fix, `adaptive_tier.cpp`): an 8 x 8 MiB `cudaHostAlloc` staging arena; every blob from a pageable source layer is memcpy'd into a pinned block before the batch copy, so `cudaMemcpyBatchAsync` sources are always pinned. Blocks recycle when their batch event retires; if all 8 are busy the pump stops early (never blocks). Falls back to the old path if the arena can't be allocated.
2. **Bounded flight** (`kMaxFlying = 2` + `retire()`): at most 2 copy batches outstanding; the pump is opportunistic, so a skipped tick is harmless.
3. **Pump skip while the flag is behind** (`verify.cpp`): skip the per-layer `rows_in` pump when the second-GPU flag is behind, so pump host time never delays the launch it is waiting on.

A pinned-only A/B variant confirmed defense 1 alone stops the lockup; adding defenses 2-3 on top gave measurably improved decode performance (without them the pump's host time delays the very launch it waits on), so they are performance guards, not correctness-critical. The staging is a byte-identical pass-through (same source, length, destination), so output is fix-neutral.

## Verification (2026-10-02)

- Repro A (`--second-gpu 1 --adapt-every 1`, the config that froze at window 3): 3/3 complete, but slower TK/s
- Repro B (`--second-gpu 0 --adapt-every 1`): 3/3 complete, faster tk/s
- Frontend `--second-gpu 0` + `--adapt-every 1`: working, ~70-90 tok/s at ~96% expert-cache hits.

## Working config (UD-Q4_K_XL, second GPU on the fast link)

```json
{
  "exe": "./build/strata.exe",
  "args": [
    "--pack",
    "d:/Strata-data/packs/ud-q4_k_xl",
    "--native",
    "d:/Strata-data/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
    "--expert-profile",
    "./data/expert-profile.bin",
    "--expert-cache",
    "auto",
    "--vram-reserve-mib",
    "700",
    "--prefill",
    "16384",
    "--feed-max",
    "52",
    "--spec",
    "4",
    "--spec-min-p",
    "0.5",
    "--spec-lookup",
    "16",
    "--adapt-every",
    "1",
    "--mtp",
    "D:/Strata-data/mtp/rt",
    "--max-context",
    "200192",
    "--kv",
    "int8",
    "--pcie-mode",
    "kernel",
    "--pcie-frac",
    "0",
    "--second-gpu",
    "0",
    "--main-gpu",
    "1",
    "--head-split",
    "0"
  ],
  "cwd": ".",
  "tokenizer": "d:/Strata-data/packs/ud-q4_k_xl/tokenizer",
  "model_name": "qwen3.8-flash-next-ud-q4_k_xl",
  "log": "./strata-ud-q4_k_xl_dual_swap.log",
  "sampling": {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0
  },
  "lib_dirs": []
}
```
