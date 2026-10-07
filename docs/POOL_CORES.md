# `--pool-cores`: pin the CPU expert pool to chosen cores

Branch `feat/pool-cores` (engine 0.1.30 fork). Adds one engine flag and teaches setup's
calibrator about it.

## Why

The CPU expert pool is DRAM-bandwidth bound, and on multi-CCD CPUs (Threadripper 3970X:
4 CCDs x 8 cores) **each CCD has its own bandwidth ceiling**. The automatic placement
(`CorePlacement`) takes physical cores in enumeration order, so a low worker count packs
them onto the first CCDs and pays one CCD's ceiling. `--pool-cores` lets you place the
workers deliberately - one (or two) cores per CCD - so a small pool spans all four.

Measured on the 3970X + 2x3090 (IQ3_S, dual-swap config): with the expert tiers covering
most routing, decode is GPU-bound and the layout is flat; with the calibrator's
`--pcie-frac 0.0` recommendation (more expert work on the CPU pool) the CCD spread is
part of the best setup (102.9 tok/s sustained at ~150K context, up from 100.7).

## The flag

    --pool-cores LIST    comma-separated logical processor ids, worker order

- Worker `i` is pinned to id `i` of the list; the worker count becomes the list length.
- The host thread takes the last **physical** core that holds none of the listed ids
  (never an SMT sibling of a worker).
- The interrupt-driven quiet-core moves (`CorePlacement::tick`) are **frozen**: the list
  was chosen on purpose, moving a worker out of it would undo the CCD spread.
- Validation is strict: every id must exist on the machine, no repeats, and
  `--pool-workers` may only be 0 or the list length.

### Finding the right ids (Windows)

The pinning uses the OS's logical numbering, which differs by OS generation: Win11-style
machines number physical cores first (0-31) with SMT siblings above, Win10-style pairs
siblings consecutively (core `n` = `2n, 2n+1`). Check what the engine itself says: the
automatic placement line in the log lists one worker per physical core - an even-only
sequence like `2,4,...,60` means pair numbering. On the 3970X with pair numbering and
contiguous CCD numbering, 16 workers spread 4-per-CCD are:

    --pool-cores 0,2,4,6,16,18,20,22,32,34,36,38,48,50,52,54

## Calibration

`tools/calibrate.py <config.json>` (any config path, not just the root ones) now sweeps
an explicit `--pool-cores` list as **evenly-spaced subsets of itself** (16 -> 11 -> 8,
still across all CCDs) instead of mixing in `--pool-workers`, which is a startup error
beside an explicit list.

## Files

- `include/strata/kernels/cpu/pool.hpp`, `src/kernels/cpu/pool.cpp` - the explicit
  placement constructor, host-core pick, frozen `tick`.
- `src/program/generate.cpp` - the flag, validation, wiring.
- `tools/calibrate.py` - subset sweep for explicit lists.
- `examples/iq3_s_dual_swap_vision_threadripper-ccd16.json` - a working 3970X config.
