"""Offline check: how predictable are the first tokens of wiki.test.raw by pure repetition?

Tokenizes the first part of the corpus with the user's llama-tokenize (CPU only) and measures, for a scored span,
how many positions have their exact 16-gram (and 8-gram, 4-gram) context seen earlier in the sequence, plus a
unigram surprise baseline. A model CANNOT score median NLL 0.002 unless the text repeats itself or it leaks.
"""
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

LLAMA_DIR = Path(r"C:\Users\Chris\Documents\llama.cpp\llama-b11053-bin-win-cuda-13.4-x64")
GGUF = r"d:\Strata-data\models\UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
CORPUS = Path(__file__).resolve().parent / "corpora" / "wiki.test.raw"
CHUNK, HALF, LIMIT = 4096, 2048, 8192

text = CORPUS.read_bytes().decode("utf-8", errors="replace")
text = text[:-1] if text.endswith("\n") else text
tmp = Path(__file__).resolve().parent / "results" / "ppl" / "head.txt"
tmp.parent.mkdir(parents=True, exist_ok=True)
tmp.write_bytes(text[:400_000].encode("utf-8"))  # a prefix is enough to test window 0
r = subprocess.run([str(LLAMA_DIR / "llama-tokenize.exe"), "-m", GGUF, "-f", str(tmp), "--ids", "--no-escape"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
m = re.search(r"\[([0-9,\s]*)\]", r.stdout)
if m is None:
    sys.exit(f"tokenize failed: {(r.stderr or r.stdout)[-500:]}")
ids = [int(x) for x in m.group(1).replace(" ", "").split(",") if x]
print(f"tokenized {len(ids):,} tokens")

n = min(LIMIT + 1, len(ids))
seq = ids[:n]
for k in (4, 8, 16):
    seen, hits = set(), 0
    for p in range(HALF, CHUNK):
        if p >= k and tuple(seq[p - k:p]) in seen:
            hits += 1
        if p - k >= 0:
            seen.add(tuple(seq[p - k:p]))
    print(f"{k:2d}-gram context seen before (scored span {HALF}..{CHUNK - 1}): {hits}/{CHUNK - HALF} "
          f"= {100.0 * hits / (CHUNK - HALF):.1f}%")

uni = Counter(seq[:HALF // 2])
tot = sum(uni.values())
surprise = sum(-math.log2(uni.get(seq[p], 1) / tot) for p in range(HALF, CHUNK)) / (CHUNK - HALF)
print(f"unigram-only surprise over the scored span: {surprise:.2f} bits/token (a model beating THIS is memorizing)")
dup = sum(c - 1 for c in Counter(seq).values() if c > 1) / len(seq)
print(f"repeated-token share of window 0: {100.0 * dup:.1f}%")
