"""serve/test_choose.py - POST /v1/choose, the engine's score= key and LP line, and the model's text verbatim for a request
without tools, against mock engines (no GPU, no pack).

    python -m unittest serve.test_choose -v
"""
from __future__ import annotations

import io
import json
import math
import queue
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, OutputParser  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class ScoringEngine(MockEngine):
    """Answers a scored request as the engine does: one token, and the candidates' log-probabilities in its figures."""

    def __init__(self, tokenizer, logprobs: dict[int, float] | None):
        super().__init__(tokenizer, "B")
        self.logprobs = logprobs
        self.calls = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls.append((list(ids), max_new, dict(sampling)))
        last = {"generated": 1, "prompt_tokens": len(ids), "finish": "length", "reused": 0}
        if self.logprobs is not None and sampling.get("score"):
            last["scores"] = {t: self.logprobs.get(t, -30.0) for t in sampling["score"]}
        yield from super().generate(ids, 1, sampling, cancel, embeddings)
        self.last = last


class Choose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = ScoringEngine(cls.tok, {ord("A"): math.log(0.6), ord("B"): math.log(0.3)})
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, body):
        req = urllib.request.Request(self.base + "/v1/choose", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_probabilities_over_the_options(self):
        msgs = [{"role": "user", "content": "Pick one. Answer with the letter."}]
        s, b = self.post({"messages": msgs, "options": ["A", "B"], "answer_prefix": "Answer: "})
        self.assertEqual(s, 200, b)
        self.assertAlmostEqual(b["probs"]["A"], 2 / 3, places=6)
        self.assertAlmostEqual(b["probs"]["B"], 1 / 3, places=6)
        self.assertAlmostEqual(b["mass"], 0.9, places=6)
        self.assertEqual((b["choice"], round(b["confidence"], 6)), ("A", round(2 / 3, 6)))
        ids, max_new, sampling = self.engine.calls[-1]
        self.assertEqual((max_new, sampling), (1, {"score": [ord("A"), ord("B")]}))
        # the chat with thinking off, as /v1/chat/completions renders it, then the prefix
        prompt = self.tok.decode(ids)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: "), prompt[-80:])
        self.assertEqual(b["prompt_tokens"], len(ids))

    def test_config_sampling_does_not_reach_a_scored_request(self):
        self.svc.sampling_defaults = {"temperature": 0.7, "repetition_penalty": 1.1}
        try:
            s, b = self.post({"messages": [{"role": "user", "content": "q"}], "options": ["A", "B"]})
        finally:
            self.svc.sampling_defaults = {}
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.calls[-1][2], {"score": [ord("A"), ord("B")]})

    def test_bad_options_are_400(self):
        msgs = [{"role": "user", "content": "q"}]
        for options in (None, [], ["A", ""], ["A", 5], ["yes", "no"], ["A", "A"]):
            s, b = self.post({"messages": msgs, "options": options})
            self.assertEqual(s, 400, options)
            self.assertIn("choose:", b["error"]["message"])

    def test_no_logprobs_is_an_error(self):
        logprobs, self.engine.logprobs = self.engine.logprobs, None
        try:
            s, b = self.post({"messages": [{"role": "user", "content": "q"}], "options": ["A", "B"]})
        finally:
            self.engine.logprobs = logprobs
        self.assertEqual(s, 400)
        self.assertIn("no log-probabilities", b["error"]["message"])


class ScoreProtocol(unittest.TestCase):
    def engine(self, *lines):
        eng = StrataEngine.__new__(StrataEngine)
        eng.proc = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
        eng.lines = queue.Queue()
        eng.can_stop = False
        eng.max_context = 4096
        for line in lines:
            eng.lines.put(line + "\n")
        return eng

    def test_score_key_and_lp_line(self):
        self.assertIn(" score=65:66", StrataEngine.sampling_keys({"score": [65, 66]}))
        eng = self.engine("LP 65:-0.25 66:-1.5", "T 65", "DONE 1 10 5.0 1.0 length 0 0 0")
        self.assertEqual(list(eng.generate([1, 2], 1, {"score": [65, 66]}, threading.Event())), [65])
        self.assertIn("score=65:66 1,2", eng.proc.stdin.getvalue())
        self.assertEqual(eng.last["scores"], {65: -0.25, 66: -1.5})

    def test_unread_row(self):
        eng = self.engine("LP", "T 65", "DONE 1 10 5.0 1.0 length 0 0 0")
        list(eng.generate([1], 1, {"score": [65]}, threading.Event()))
        self.assertEqual(eng.last["scores"], {})


class VerbatimWithoutTools(unittest.TestCase):
    TEXT = ("Let me look.\n\n<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n"
            "</tool_call>")

    def events(self, tools, step):
        p = OutputParser(thinking=False, tools=tools, stream_tools=True)
        out = []
        for i in range(0, len(self.TEXT), step):
            out += p.feed(self.TEXT[i:i + step])
        return out + p.finish()

    def test_no_tools_keeps_the_tags(self):
        for step in (1, 7, len(self.TEXT)):
            evs = self.events(None, step)
            self.assertEqual({e.kind for e in evs}, {"content"})
            self.assertEqual("".join(e.text for e in evs), self.TEXT)

    def test_declared_tools_still_parse(self):
        evs = self.events([{"name": "read", "parameters": {"properties": {"path": {"type": "string"}}}}], 5)
        calls = [e.call for e in evs if e.kind == "tool_call"]
        self.assertEqual([(c.name, c.arguments) for c in calls], [("read", {"path": "a.txt"})])


if __name__ == "__main__":
    unittest.main()
