# This fork's configs

The configs this fork runs with on a PC with an RTX 3090, an RTX 5070 Ti and 160 GB of RAM, for `serve/server.py`
(their speeds: [COMPARISON.md](../docs/COMPARISON.md)):

- `iq3_s.json`: ISTA-DASLab's GSQ-RCO IQ3_S (~53 GB of RAM taken).
- `ud-q4_k_xl.json`: unsloth's UD-Q4_K_XL (~81 GB of RAM taken).

This branch's engine leaves out some of upstream's options, which [DETAILS.md](../docs/DETAILS.md) describes and setup
may write into its own configs: the layer split and the helper GPUs' caches (`--peer-device`; the second GPU here is
`--second-gpu`), KV streaming (`--kv-resident`: the KV cache stays in VRAM), the VRAM resize (`--vram-elastic`), the
low-RAM modes, the file tier and the RAM budget (`--resident-budget-gib`), the experimental speed projection (control
vectors), upstream's conversation cache (`--conversation-cache-mib`; the prompt cache here keeps replaced conversations
in RAM), session files, parallel requests (`--batch`), token biases, prompt lookup chained after the drafts
(`--lookup-chain`), upstream's routing approximations (the route tail skip, on by default in upstream's CUDA server),
the adaptive tier's overlapped copies
(`STRATA_ADAPT_OVERLAP`), the engine's own watchdog and stall report, the PLE read probe, 0.1.44's NVFP4 experts,
Qwen3.6-35B-A3B engine and batched commit graph, and the AMD, Intel, Metal (Mac), Turing, Volta and Pascal builds. A
server config with `parallel` or `vram_elastic` passes the engine a flag it refuses; with `slot_save_path`, a save
request ends the engine.

The configs keep what each model's run learned (`expert_profile_save`, upstream's #477): the engine writes a profile
(`<model>-learned.bin` in the Strata folder) at exit, on Ctrl+C in the server's window and between requests at most once
a minute, and the next start fills both GPUs' expert caches from it instead of the shipped profile. It ranks the
experts by this run's routing and the profile it started from at equal weight. Against the shipped profile, after a
session of other code and reasoning requests, a round of fixed text took 9-14% less time on code, -3.8 to +0.9% on
thinking and edits, 0 to +2.0% after a 16K-token prompt. Remove the two keys to start from the shipped profile every
time.

Their paths are relative to the Strata folder, where the server starts: the engine in `build/`; the model files, the
packs and the MTP draft layer in setup's data folder, `../Strata-data` (unless setup was given another: `--data-dir`).
A pack is what `tools/iq_pack.py` makes from a model's GGUF files in seconds (~1.5 GB): the tokenizer with the chat
template, the model's float tensors in the forms the engine takes, and an index of the rest; the engine reads the
quantized tensors, the experts and most projections, from the GGUF files at start.

1. Set up IQ3_S once: `START-HERE.bat --setup --model IQ3_S --no-start` downloads the model into
   `../Strata-data/models/IQ3_S`, makes its pack in `../Strata-data/packs/iq3_s`, and makes the MTP draft layer in
   `../Strata-data/mtp/rt` from the ~5 GB of MTP tensors of Qwen's original checkpoint. The configs' draft layer is
   `../Strata-data/mtp/rt-tuned`: a copy of `mtp/rt` with its `dense.bin` replaced by the self-distilled one published
   with jesdga95's basalt fork ([jesdga/Qwen3.8-Flash-Next-Basalt](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt),
   `convert/mtp/rt/dense.bin`, sha256 `dfe18647e0c95d05302907be88f021590119645a1567fcd770c8d309df852d75`, under Qwen's
   license). It drafts better (0.2 more tokens per verify pass on thinking text and after long prompts: 4-6% faster
   there), and the output stays the same: every token is verified.
2. Build the engine from this branch with code for both cards (the ready-made engine setup puts in `engine/` is
   upstream's, without this fork's options), with MSVC, CUDA 13, CMake and Ninja (setup puts the last two in
   `.venv\Scripts`):

       cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF ^
         "-DCMAKE_CUDA_ARCHITECTURES=86;120"
       cmake --build build --target strata

   `86;120` is an RTX 30 card and an RTX 50 card. The engine needs the CUDA runtime DLLs (`cudart64_13.dll`,
   `cublas64_13.dll`, `cublasLt64_13.dll`) beside it, on the PATH, or their folder in the config's `lib_dirs`.
3. For UD-Q4_K_XL, download unsloth's four `Qwen3.8-Flash-Next-UD-Q4_K_XL-0000N-of-00004.gguf` files into
   `../Strata-data/models/UD-Q4_K_XL` and make its pack:

       .venv\Scripts\python.exe tools\iq_pack.py --out ../Strata-data/packs/ud-q4_k_xl ^
         --gguf ../Strata-data/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

4. Set the GPU numbers, as `nvidia-smi` numbers the cards (here the 5070 Ti is 0 and the 3090 is 1). `--main-gpu`
   runs the model, and nothing else may use that card: a display or another program on it slows generation to a few
   tokens per second. `--second-gpu` holds more experts and may drive the display.
5. Start a config with its launcher, `run-iq3_s.bat` (port 8081) or `run-ud-q4_k_xl.bat` (8082), or from the Strata
   folder:

       .venv\Scripts\python.exe serve\server.py --engine strata --config examples\iq3_s.json --port 8081

   (`START-HERE.bat` starts setup's own config.)

This PC's UD-Q4_K_XL config also has `--arena-place`, left out here: for RAM whose regions read at different rates,
as unpaired DIMMs make (two 32 GB and two 48 GB modules), the engine times its RAM arena at the start and keeps the
experts the CPU reads out of the slow region, again between requests ([DETAILS.md](../docs/DETAILS.md)). On such a
PC add it to a config's arguments; elsewhere it finds nothing to move and costs ~2 s at the start.

On Linux the engine is `./build/strata` and Python `.venv/bin/python`.
