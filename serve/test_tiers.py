"""serve/test_tiers.py - the expert tiers' per-request counters (the engine's TIERS line, PR #11) in the DONE figures, a
request's timings and its /metrics record, against mock engines (no GPU, no pack).

    python -m unittest serve.test_tiers -v
"""
from __future__ import annotations

import io
import queue
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import (ByteTokenizer, MockEngine, Service, StrataEngine, request_stats,  # noqa: E402
                          request_timings)

TIERS = ("TIERS main_swaps=11 main_fills=12 main_dropped=0 second_entries=13 second_experts=14 second_layers=15 "
         "second_cpu_layers=1 second_swaps=16 second_fills=17 second_dropped=0 second_empty_slots=18 "
         "second_prefetched=19 second_prefetch_used=20 deterministic_adapt=1")


def engine(*lines):
    eng = StrataEngine.__new__(StrataEngine)
    eng.proc = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
    eng.lines = queue.Queue()
    eng.can_stop = True
    eng.max_context = 4096
    for line in lines:
        eng.lines.put(line + "\n")
    return eng


class TiersLine(unittest.TestCase):
    def test_done_figures(self):
        eng = engine("T 65", TIERS, "DONE 1 24 12.5 7.5 length 1 2 4 8 9")
        self.assertEqual(list(eng.generate([1], 1, {}, threading.Event())), [65])
        t = eng.last["expert_tiers"]
        self.assertEqual((t["main_swaps"], t["second_prefetch_used"], t["deterministic_adapt"]), (11, 20, 1))
        self.assertEqual((eng.last["hits"], eng.last["lookups"]), (8, 9))          # DONE's fields as before
        self.assertEqual(request_timings(24, 1, eng.last)["expert_tiers"], t)

    def test_malformed_fields_are_left_out(self):
        eng = engine("TIERS main_swaps=bad future=99 x second_layers=7", "DONE 1 24 12.5 7.5 length 1 2 4 8 9")
        list(eng.generate([1], 1, {}, threading.Event()))
        self.assertEqual(eng.last["expert_tiers"], {"future": 99, "second_layers": 7})

    def test_without_the_line(self):
        eng = engine("T 65", "DONE 1 24 12.5 7.5 length 1 2 4 8 9")
        list(eng.generate([1], 1, {}, threading.Event()))
        self.assertNotIn("expert_tiers", eng.last)
        self.assertNotIn("expert_tiers", request_timings(24, 1, eng.last))

    def test_a_consumer_that_stops_early(self):
        eng = engine("T 65", "T 66", TIERS, "DONE 2 24 12.5 7.5 cancel 1 2 4 8 9")
        gen = eng.generate([1], 8, {}, threading.Event())
        self.assertEqual(next(gen), 65)
        gen.close()                                  # STOP, then the drain reads to DONE
        self.assertIn("STOP", eng.proc.stdin.getvalue())
        self.assertEqual(eng.last["expert_tiers"]["second_entries"], 13)

    def test_passes_are_summed(self):
        a = {"generated": 1, "expert_tiers": {"main_swaps": 2, "second_empty_slots": 9, "deterministic_adapt": 1}}
        b = {"generated": 2, "expert_tiers": {"main_swaps": 3, "second_empty_slots": 4, "deterministic_adapt": 1}}
        self.assertEqual(request_stats([a, b])["expert_tiers"],
                         {"main_swaps": 5, "second_empty_slots": 4, "deterministic_adapt": 1})


class TiersEngine(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        yield from super().generate(ids, 1, sampling, cancel, embeddings)
        self.last = {"generated": 1, "prompt_tokens": len(ids), "prompt_ms": 1.0, "decode_ms": 1.0, "reused": 0,
                     "hits": 0, "lookups": 0, "expert_tiers": {"main_swaps": 3, "second_entries": 4}}


class TiersInTheRecords(unittest.TestCase):
    def test_history_metrics_and_timings(self):
        tok = ByteTokenizer()
        svc = Service(TiersEngine(tok, "x"), tok, object())
        events = list(svc.run([120], False, None, 1, {}, threading.Event()))
        want = {"main_swaps": 3, "second_entries": 4}
        self.assertEqual(events[-1][1]["timings"]["expert_tiers"], want)
        self.assertEqual(svc.history[-1]["expert_tiers"], want)
        self.assertEqual(svc.metrics(all_requests=True)["requests"][0]["expert_tiers"], want)


if __name__ == "__main__":
    unittest.main()
