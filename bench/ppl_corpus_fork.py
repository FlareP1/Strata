"""bench/ppl_corpus_fork.py - this fork's own text as a perplexity corpus (bench/corpora/fork-sources.txt).

WikiText-2 is Wikipedia, and Wikipedia is in every frontier pretraining set: a model can QUOTE it (probe: median next
token prob 0.998 on window 0, shuffled control honest).  Post-cutoff sources and docs of this repository are not in
any training set, so a low PPL there means modeling, not memory.  Plain text concatenation, sorted, LF endings.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "corpora" / "fork-sources.txt"

parts = [f.read_text(encoding="utf-8", errors="replace")
         for pat in ("serve/*.py", "tools/*.py", "docs/*.md", "*.md") for f in sorted(ROOT.glob(pat))]
OUT.write_text("\n\n".join(parts), encoding="utf-8", newline="\n")
print(f"{OUT}: {OUT.stat().st_size:,} bytes from {len(parts)} files")
