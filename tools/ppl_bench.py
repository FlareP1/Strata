"""tools/ppl_bench.py - teacher-forced perplexity of the engine, one position at a time.

The engine reads a prompt through verify windows when --feed-max covers it, and STRATA_LOGPOS=1 makes every window
print the log prob it gave the token that actually followed, of the token two ahead, and of its own argmax:
`LOGPOS <pos> <target id> <log prob> <two-ahead id or -1> <its log prob> <argmax id> <its log prob>`.  So each scored
position is a forward pass the engine really ran over the corpus's own prefix, and nothing is sampled.  The windows
mirror llama-perplexity - the second half of each window is scored, with stride chunk/2 by default and stride chunk
with --no-overlap, which is what llama-perplexity itself does (its --ppl-stride scores a different span, so the two
tools agree exactly in --no-overlap mode).

    .venv\\Scripts\\python.exe tools\\ppl_bench.py --config examples\\ud-q4_k_xl_dual_swap_vision.json --dry-run
    .venv\\Scripts\\python.exe tools\\ppl_bench.py --config examples\\ud-q4_k_xl_dual_swap_vision.json --out bench\\results\\ppl.json
    .venv\\Scripts\\python.exe tools\\ppl_bench.py --config examples\\ud-q4_k_xl_dual_swap_vision.json --corpus tiny.txt --tokenize-only
    .venv\\Scripts\\python.exe tools\\ppl_bench.py --config examples\\ud-q4_k_xl_dual_swap_vision.json --run-llama --no-overlap ^
        --llama-extra "-ngl 99 --fit on"
    .venv\\Scripts\\python.exe tools\\ppl_bench.py --config examples\\ud-q4_k_xl_dual_swap_vision.json --shuffle

With --shuffle the planned windows run a second time over a fixed-seed shuffle (--shuffle-seed) of the whole id
list.  Shuffled text has no sequential structure, so a correct engine's shuffled PPL must be enormous; one close to
the real PPL means the LOGPOS rows leak.  The shuffle catches context-reuse leaks (a row that reused real context
would still score the shuffled targets honestly) but NOT a +1 row shift: a shifted row is just as lost on shuffled
text, so both passes score enormous and look honest - that is what the nll2 and am=+1/+2/+0 columns are for.

Without --corpus the wikitext-2 test set is fetched into bench/corpora/ (the zip llama.cpp's perplexity tool points
at).  The preprocessing is what llama-perplexity's -f handler does to it - the bytes as UTF-8 with a single trailing
newline dropped - and llama-tokenize is then run with --no-escape and the same model, so both tools see the same ids;
this llama.cpp version replaces nothing else and has no wikitext-specific handling.  The engine runs the config's own
arguments with --prompt-cache 0 (nothing reused, so every request starts at position 0) and --feed-max at least
chunk+1 (every prompt position goes through the verify windows, which is where LOGPOS comes from).

Each pass wants the GPU to itself: the llama-perplexity baseline runs before the engine is started, and the 101 GB
model needs its memory placement flags (-ngl, --fit and friends), which is what --llama-extra carries.  That binary
has no --special and no --no_overlap in b11053 (its defaults already are: specials unparsed, windows not overlapped).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import random
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from serve.server import child_env           # the engine's environment (the config's lib_dirs on the path)
from serve.winjob import contain             # the engine ends with this script, however it ends (Windows)

WIKITEXT_ZIP = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
CORPORA = ROOT / "bench" / "corpora"
LLAMA_DIR = Path(r"C:\Users\Chris\Documents\llama.cpp\llama-b11053-bin-win-cuda-13.4-x64")


def arg_value(args: list[str], flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None


def with_arg(args: list[str], flag: str, value: str) -> list[str]:
    """`args` with `flag value` set (replaced if present) - tools/calibrate.py's helper."""
    out = list(args)
    while flag in out:
        i = out.index(flag)
        del out[i:i + 2]
    return out + [flag, value]


def fetch_wikitext(dest: Path) -> Path:
    """The wikitext-2 test set: wiki.test.raw out of the zip llama.cpp's perplexity comments name."""
    out = dest / "wiki.test.raw"
    if out.exists():
        return out
    dest.mkdir(parents=True, exist_ok=True)
    print(f"fetching {WIKITEXT_ZIP} ...", flush=True)
    req = urllib.request.Request(WIKITEXT_ZIP, headers={"User-Agent": "strata-ppl-bench"})
    with urllib.request.urlopen(req, timeout=300) as r:
        blob = r.read()
    with zipfile.ZipFile(BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.endswith("wiki.test.raw"))
        out.write_bytes(z.read(name))
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return out


def preprocess(raw: bytes) -> str:
    """llama-perplexity's -f handler: the file's bytes as UTF-8, one trailing newline dropped, nothing else."""
    text = raw.decode("utf-8", errors="replace")
    return text[:-1] if text.endswith("\n") else text


def tokenize(llama_dir: Path, gguf: str, src: Path) -> list[int]:
    """llama-tokenize over the preprocessed corpus: `--ids` prints them as one line `[14556, 1814]`, everything else
    it says is stderr.  --no-escape because the default turns literal backslash pairs into characters, which
    llama-perplexity does not (it sets escape off), and the corpus is read verbatim."""
    r = subprocess.run([str(llama_dir / "llama-tokenize.exe"), "-m", gguf, "-f", str(src), "--ids", "--no-escape"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r"\[([0-9,\s]*)\]", r.stdout)
    if r.returncode != 0 or m is None:
        raise SystemExit(f"llama-tokenize failed (code {r.returncode}):\n{(r.stderr or r.stdout)[-1500:]}")
    return [int(x) for x in m.group(1).replace(" ", "").split(",") if x]


def window_starts(n_tokens: int, chunk: int, stride: int, limit: int) -> list[int]:
    """Window starts: the last scored position of a window is start+chunk-1 and needs the token after it."""
    out = []
    for start in range(0, max(0, n_tokens - chunk), stride):
        out.append(start)
        if limit and len(out) >= limit:
            break
    return out


def shuffled_ids(ids: list[int], seed: int) -> list[int]:
    """The whole id list in a fixed-seed order: a permutation, so the planned windows still fit and the target
    check of the shuffled pass reads the shuffled list."""
    out = list(ids)
    random.Random(seed).shuffle(out)
    return out


class Serve:
    """`strata --serve` with STRATA_LOGPOS: the prompt positions' log probs, request by request.  Its stdout is read
    on a thread so a quiet engine (a long prompt, no PP lines) is still caught by the idle timeout."""

    def __init__(self, exe: Path, args: list[str], cwd: Path, log: Path, env: dict, startup_s: float,
                 idle_s: float):
        log.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log
        self.log = log.open("a", encoding="utf-8")
        self.idle_s = idle_s
        self.proc = subprocess.Popen([str(exe), *args], cwd=str(cwd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, encoding="utf-8", bufsize=1, env=env)
        contain(self.proc)
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.max_context = 0
        self.info: dict[str, str] = {}
        for line in iter(lambda: self._line(startup_s, "the engine never became ready"), ""):
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = v
            if line.startswith("READY"):
                self.max_context = int(line.split()[1])
                break

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def _line(self, timeout: float, what: str) -> str:
        """One line of the engine's stdout; `timeout` is the silence it may take, not the whole wait."""
        quiet = time.time() + timeout
        while True:
            try:
                line = self.lines.get(timeout=5)
            except queue.Empty:
                if time.time() > quiet:
                    raise SystemExit(f"{what}: silent for {timeout:.0f} s.  The log ends:\n{self.tail()}")
                continue
            if line is None:
                raise SystemExit(f"{what}: it exited (code {self.proc.poll()}).  The log ends:\n{self.tail()}")
            quiet = time.time() + timeout
            return line.strip()

    def tail(self, n: int = 3000) -> str:
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - n))
                return f.read().decode("utf-8", "replace")
        except OSError:
            return "(no log)"

    def logprobs(self, ids: list[int]) -> tuple[dict[int, tuple[int, float, int, float, int, float]], list[str]]:
        """One `GEN 1` request over `ids`: the LOGPOS lines by prompt position, and the DONE fields.  Each position
        carries (target, log P(target), two-ahead id, log P(two-ahead), argmax id, log P(argmax))."""
        try:
            self.proc.stdin.write("GEN 1 temperature=0 " + ",".join(str(int(t)) for t in ids) + "\n")
            self.proc.stdin.flush()
        except OSError:
            raise SystemExit(f"the engine is gone (code {self.proc.poll()}).  The log ends:\n{self.tail()}") from None
        got: dict[int, tuple[int, float, int, float, int, float]] = {}
        done = None
        while done is None:
            line = self._line(self.idle_s, "the engine stopped answering")
            if line.startswith("LOGPOS "):
                f = line.split()
                if len(f) < 8:
                    raise SystemExit(f"short LOGPOS line (engine built before the row-shift probe?): {line!r}")
                got[int(f[1])] = (int(f[2]), float(f[3]), int(f[4]), float(f[5]), int(f[6]), float(f[7]))
            elif line.startswith("ERR "):
                raise SystemExit("the engine refused the request: " + line[4:])
            elif line.startswith("DONE"):
                done = line.split()
        return got, done

    def stop(self):
        try:
            self.proc.stdin.write("STOP\n")
            self.proc.stdin.flush()
        except OSError:
            pass

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            self.proc.kill()
        self.log.close()


def median(vals: list[float]) -> float:
    """The middle of `vals` (the mean of the two middles for an even count)."""
    s = sorted(vals)
    n = len(s)
    return (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]


def run_windows(eng: Serve, series: list[int], starts: list[int], chunk: int, half: int, rows: list[dict],
                label: str) -> dict:
    """One pass over `series` at the planned window starts, appending its table rows as it goes so an interrupt
    keeps what it scored.  Every target is checked against `series`, so the shuffled pass checks against the
    shuffled list.  The median, p90 and max of -logprob tell a few leaked rows apart from every row at ~0.  Alongside
    the target's NLL it tracks the two-ahead token's NLL and which token the row's argmax is (the next, the one after
    it, or the input): if the rows are shifted a position ahead their argmax is the two-ahead token, not the target.
    Returns the raw values and argmax counts so the overall verdict can take a true median across all windows."""
    print(f"\n{label} pass: {len(starts)} windows of {chunk}, {half} scored positions each", flush=True)
    print(f"{'window':>6} {'start':>9} {'tokens':>7} {'mean NLL':>10} {'med NLL':>9} {'p90 NLL':>9} {'max NLL':>9} "
          f"{'PPL':>9} {'nll2':>7} {'am=+1/+2/+0':>13} {'prompt':>8}", flush=True)
    agg = {"vals": [], "vals2": [], "c1": 0, "c2": 0, "c0": 0, "d1": 0, "d2": 0, "d0": 0}
    for w, start in enumerate(starts):
        got, done = eng.logprobs(series[start:start + chunk + 1])
        if len(got) < chunk:
            raise SystemExit(f"{label} window {w}: {len(got)} LOGPOS lines for a prompt of {chunk + 1} tokens - the "
                             "engine did not read all of it through verify windows (--prompt-cache 0 and "
                             "--feed-max are set by this tool; a GENI/vision prompt would not work)")
        nll, vals, worst = 0.0, [], None
        vals2 = []
        c1 = c2 = c0 = d1 = d2 = d0 = 0
        for sent in range(half, chunk):                 # llama-perplexity scores the second half of the window
            target = int(series[start + sent + 1])
            pos_target, lp, two_id, lp2, am_id, _lp_am = got[sent]
            if pos_target != target:
                raise SystemExit(f"{label} window {w}, position {start + sent}: the engine scored token "
                                 f"{pos_target}, the series has {target}")
            nll -= lp
            vals.append(-lp)
            if worst is None or lp < worst[1]:
                worst = (start + sent, lp)
            # the row-shift probe: is the row's argmax the honest next token (+1), the two-ahead token (+2) or the
            # input token it was fed (+0)?  two_id/lp2 already carry the two-ahead token's own log prob
            d1 += 1
            d0 += 1
            c1 += int(am_id == target)
            c0 += int(am_id == int(series[start + sent]))
            if start + sent + 2 < len(series):
                d2 += 1
                c2 += int(am_id == int(series[start + sent + 2]))
            if two_id >= 0:
                vals2.append(-lp2)
        vals.sort()
        n = len(vals)                                 # chunk - half scored values: half, unless the chunk is odd
        med = median(vals)
        med2 = median(vals2) if vals2 else None
        mean2 = sum(vals2) / len(vals2) if vals2 else None
        sh1 = 100.0 * c1 / d1 if d1 else 0.0
        sh2 = 100.0 * c2 / d2 if d2 else 0.0
        sh0 = 100.0 * c0 / d0 if d0 else 0.0
        row = {"window": w, "start": start, "tokens": half, "mean_nll": nll / half, "median_nll": med,
               "p90_nll": vals[math.ceil(0.9 * n) - 1], "max_nll": vals[-1], "ppl": math.exp(nll / half),
               "median_nll2": med2, "mean_nll2": mean2, "am_next": sh1, "am_two": sh2, "am_self": sh0,
               "prompt_ms": float(done[3]), "finish": done[5], "worst_position": worst[0],
               "worst_logprob": worst[1]}
        rows.append(row)
        agg["vals"] += vals
        agg["vals2"] += vals2
        agg["c1"] += c1
        agg["c2"] += c2
        agg["c0"] += c0
        agg["d1"] += d1
        agg["d2"] += d2
        agg["d0"] += d0
        print(f"{w:>6} {start:>9,} {half:>7,} {row['mean_nll']:>10.5f} {med:>9.4f} {row['p90_nll']:>9.4f} "
              f"{row['max_nll']:>9.4f} {row['ppl']:>9.4f} "
              f"{('-' if med2 is None else f'{med2:.4f}'):>7} {f'{sh1:.0f}/{sh2:.0f}/{sh0:.0f}':>13} "
              f"{row['prompt_ms'] / 1000:>7.2f}s", flush=True)
    return agg


def overall_of(rows: list[dict]) -> dict:
    """The scored-position mean NLL over `rows` and its PPL."""
    tokens = sum(r["tokens"] for r in rows)
    mean_nll = sum(r["mean_nll"] * r["tokens"] for r in rows) / tokens
    return {"tokens": tokens, "mean_nll": mean_nll, "ppl": math.exp(mean_nll)}


def run_llama(llama_dir: Path, gguf: str, corpus: Path, chunk: int, extra: str) -> dict:
    """llama-perplexity over the same corpus and window size, its output streamed; the `Final estimate: PPL` line is
    the number, and the `[N]ppl,` values are its running means per window."""
    cmd = [str(llama_dir / "llama-perplexity.exe"), "-m", gguf, "-f", str(corpus), "-c", str(chunk), "-b", "2048"]
    if extra:
        cmd += shlex.split(extra)
    print("$ " + " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", bufsize=1)
    final, running = None, []
    for raw in p.stdout:
        line = raw.rstrip()
        print("  llama | " + line, flush=True)
        m = re.search(r"Final estimate: PPL = ([0-9.]+)(?: \+/- ([0-9.]+))?", line)
        if m:
            final = {"ppl": float(m.group(1)), "plus_minus": float(m.group(2)) if m.group(2) else None}
        running += [float(x) for x in re.findall(r"\[\d+\]([0-9.]+),", line)]
    rc = p.wait()
    if final is None:
        raise SystemExit(f"llama-perplexity ended (code {rc}) without a final estimate")
    return {"cmd": cmd, "exit": rc, **final, "running_ppl": running[-3:]}


def main() -> int:
    ap = argparse.ArgumentParser(description="teacher-forced perplexity of the engine, with a llama-perplexity baseline")
    ap.add_argument("--config", required=True, type=Path, help="the engine's config json (exe, args, cwd, --native)")
    ap.add_argument("--corpus", type=Path, help="text file; default: the wikitext-2 test set, fetched to bench/corpora")
    ap.add_argument("--llama-dir", type=Path, default=LLAMA_DIR, help="the llama.cpp release to tokenize with")
    ap.add_argument("--chunk", type=int, default=8192, help="window size in tokens (llama-perplexity's -c)")
    ap.add_argument("--no-overlap", action="store_true", help="stride chunk instead of chunk/2 (llama-perplexity's own)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N windows")
    ap.add_argument("--shuffle", action="store_true",
                    help="run every window a second time over a fixed-seed shuffle of the whole id list: shuffled "
                         "text has no sequential structure, so a PPL close to the real one means LOGPOS leaks")
    ap.add_argument("--shuffle-seed", type=int, default=20261004, help="the --shuffle seed")
    ap.add_argument("--out", type=Path, help="write the results as JSON here")
    ap.add_argument("--dry-run", action="store_true", help="tokenize and print the plan and the commands, start nothing")
    ap.add_argument("--tokenize-only", action="store_true", help="tokenize and stop")
    ap.add_argument("--run-llama", action="store_true", help="run llama-perplexity first, for the comparison table")
    ap.add_argument("--llama-extra", default="", help='llama-perplexity arguments as one string, e.g. "-ngl 99 --fit on"')
    ap.add_argument("--log", type=Path, default=ROOT / "bench" / "results" / "ppl" / "strata.log",
                    help="where the engine's stderr goes")
    ap.add_argument("--startup-timeout", type=float, default=3600, help="seconds the engine may take to load and say READY")
    ap.add_argument("--idle-timeout", type=float, default=900, help="seconds the engine may stay silent inside a request")
    a = ap.parse_args()

    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    exe = Path(cfg["exe"])
    if not exe.is_absolute() and not exe.exists():      # the configs spell these relative to the repository root
        exe = ROOT / exe
    cwd = Path(cfg.get("cwd") or ".")
    if not cwd.is_absolute():
        cwd = ROOT / cwd
    gguf = arg_value(cfg["args"], "--native")
    if not gguf:
        raise SystemExit(f"{a.config} has no --native GGUF, so there is nothing to tokenize with")

    corpus = a.corpus or fetch_wikitext(CORPORA)
    if not corpus.exists():
        raise SystemExit(f"no such corpus: {corpus}  (drop --corpus to fetch the wikitext-2 test set)")

    with tempfile.TemporaryDirectory(prefix="strata-ppl-") as td:
        work = Path(td)
        text = work / "corpus.txt"                      # what both tools read: the preprocessed corpus, as bytes so
        text.write_bytes(preprocess(corpus.read_bytes()).encode("utf-8"))   # Windows writes no CRs into it
        ids = tokenize(a.llama_dir, gguf, text)
        stride = a.chunk if a.no_overlap else a.chunk // 2
        starts = window_starts(len(ids), a.chunk, stride, a.limit)
        half = a.chunk // 2
        print(f"{corpus.name}: {len(ids):,} tokens, chunk {a.chunk}, stride {stride}"
              f"{' (no overlap)' if a.no_overlap else ''}, {len(starts)} windows x {half} scored positions", flush=True)
        if not starts:
            raise SystemExit(f"the corpus gives no full window of {a.chunk} tokens")
        if a.tokenize_only:
            print(f"first ids: {ids[:16]}")
            print(f"window 0: GEN 1 over ids[0:{a.chunk + 1}], scored positions {half}..{a.chunk - 1}")
            return 0

        eng_args = with_arg(with_arg([x for x in cfg["args"] if x != "--serve"], "--prompt-cache", "0"),
                            "--feed-max", str(a.chunk + 1)) + ["--serve"]
        env = child_env(cfg)
        env["STRATA_LOGPOS"] = "1"
        llama_cmd = [str(a.llama_dir / "llama-perplexity.exe"), "-m", gguf, "-f", str(text), "-c", str(a.chunk),
                     "-b", "2048"] + (shlex.split(a.llama_extra) if a.llama_extra else [])
        if a.dry_run:
            if a.shuffle:
                print(f"shuffle pass: the same {len(starts)} window starts over the {len(ids):,} ids in seed "
                      f"{a.shuffle_seed} order")
            print("$ " + " ".join([str(exe), *eng_args]))
            print("$ " + " ".join(llama_cmd) + "        (with --run-llama)")
            return 0

        llama = None
        if a.run_llama:                                 # before the engine: one of them has the GPUs to itself
            llama = run_llama(a.llama_dir, gguf, text, a.chunk, a.llama_extra)

        print(f"engine: {exe}\nengine stderr: {a.log}", flush=True)
        eng = Serve(exe, eng_args, cwd, a.log, env, a.startup_timeout, a.idle_timeout)
        if a.chunk + 10 > eng.max_context:
            eng.close()
            raise SystemExit(f"a window of {a.chunk} tokens does not fit the engine's context ({eng.max_context})")
        print(f"engine ready: context {eng.max_context:,}, spec {eng.info.get('spec')}", flush=True)

        rows: list[dict] = []
        srows: list[dict] = []
        agg: dict | None = None
        sagg: dict | None = None
        try:
            agg = run_windows(eng, ids, starts, a.chunk, half, rows, "corpus")
            if a.shuffle:
                sagg = run_windows(eng, shuffled_ids(ids, a.shuffle_seed), starts, a.chunk, half, srows, "shuffled")
        except KeyboardInterrupt:
            print("interrupted: the request is stopped", flush=True)
            eng.stop()
        finally:
            eng.close()

    if not rows:
        return 1
    overall = overall_of(rows)
    soverall = overall_of(srows) if srows else None
    print(f"{'overall':>6} {'':>9} {overall['tokens']:>7,} {overall['mean_nll']:>10.5f} {'':>9} {'':>9} {'':>9} "
          f"{overall['ppl']:>9.4f}")
    if soverall:
        print(f"{'shuffled':>6} {'':>9} {soverall['tokens']:>7,} {soverall['mean_nll']:>10.5f} {'':>9} {'':>9} "
              f"{'':>9} {soverall['ppl']:>9.4f}")
        if soverall["ppl"] < 2 * overall["ppl"]:
            print("LEAK: the shuffled text scores like the real one")
        else:
            print("ok: shuffled text scores far worse, as it must")
    if llama:
        same = "the same windows" if a.no_overlap else "its windows are not overlapped, so this is not like for like"
        print(f"llama-perplexity: PPL {llama['ppl']:.4f}"
              + (f" +/- {llama['plus_minus']:.4f}" if llama["plus_minus"] is not None else "") + f" ({same})")

    # the row-shift verdict over the corpus pass: the two-ahead token scores like the target and the row's argmax is
    # the two-ahead token at most positions => every row is really the NEXT cell's distribution, not this one's
    if agg and agg["vals"] and agg["vals2"] and agg["d2"]:
        med_all = median(agg["vals"])
        med2_all = median(agg["vals2"])
        shift = agg["c2"] / agg["d2"]
        if med2_all < med_all and shift > 0.5:
            print("LEAK: the rows are shifted one position ahead (they score what the NEXT cell predicts)")
        else:
            print("no +1 shift detected: rows look honestly positioned")

    args_hash = hashlib.sha256(" ".join([str(exe), *eng_args]).encode()).hexdigest()[:16]
    out = {"corpus": {"path": str(corpus), "tokens": len(ids),
                      "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest()[:16]},
           "windows": {"chunk": a.chunk, "stride": stride, "overlap": not a.no_overlap, "scored_per_window": half,
                       "count": len(starts)},
           "engine": {"config": str(a.config), "exe": str(exe), "cwd": str(cwd), "args": eng_args,
                      "args_sha256": args_hash, "max_context": eng.max_context, "info": eng.info,
                      "log": str(a.log)},
           "per_window": rows, "overall": overall}
    if a.shuffle:
        out["shuffled"] = {"seed": a.shuffle_seed, "per_window": srows, "overall": soverall}
    if llama:
        out["llama_perplexity"] = llama
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
