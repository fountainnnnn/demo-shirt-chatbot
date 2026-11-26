"""Offline fake for app.models (chat / chat_stream / embed) used by test_agent and test_api.

The fake extractor understands a few phrases, the fake picker returns a
hallucinated id first (the agent must drop it), and the reply stream is fixed.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pytest

HALLUCINATED_ID = 999999


class FakeProvider:
    def __init__(self):
        self.calls: list[str] = []      # "extract" | "pick" | "stream"
        self.pick_override = None       # callable(candidate_ids) -> dict

    # -- chat with schema
    def chat(self, messages, schema=None, max_tokens=512, temperature=0.2):
        kind = (schema or {}).get("title", "free")
        self.calls.append(kind)
        if kind == "extract":
            text = messages[-1]["content"].split("Shopper's new message:", 1)[1].strip()
            return json.dumps(self._extract(text))
        if kind == "pick":
            cands = [c["id"] for c in json.loads(messages[-1]["content"])["candidates"]]
            if self.pick_override:
                return json.dumps(self.pick_override(cands))
            picks = [{"id": HALLUCINATED_ID, "reason": "Invented shirt."}]
            picks += [{"id": c, "reason": "Fits the brief."} for c in cands[1:3]]
            return json.dumps({"picks": picks, "best_id": cands[1] if len(cands) > 1 else cands[0]})
        return "ok"

    @staticmethod
    def _extract(t: str) -> dict:
        low = t.lower()
        out: dict = {"intent": "other"}
        if re.search(r"shirt|tee|wedding|office|looking for|under \d+", low):
            out["intent"] = "search"
        if "wedding" in low:
            out["occasion"] = "a beach wedding" if "beach" in low else "a wedding"
            out["title"] = "Beach wedding in June" if "beach" in low else "Wedding"
        if "office" in low:
            out["occasion"], out["usage"] = "the office", "formal"
        if re.search(r"\bshort", low):
            out["sleeve"], out["intent"] = "short", out["intent"] if out["intent"] == "search" else "refine"
        if re.search(r"\blong", low):
            out["sleeve"] = "long"
        if re.search(r"\b(tee|t-shirt)", low):
            out["type"] = "tshirt"
        if "button shirt" in low:
            out["type"] = "shirt"
        m = re.search(r"\b(?:no|not|nothing)\s+(white|black|red|blue)", low)
        if m:
            out["avoid_colours"] = [m.group(1)]
            out.setdefault("intent", "refine")
        wanted = [c for c in re.findall(r"(?<!no )(?<!not )\b(black|blue|green|red)\b", low)
                  if c not in out.get("avoid_colours", [])]
        if wanted:
            out["colours"] = wanted
        if "plain" in low:
            out["pattern"] = "solid"
        if "formal" in low:
            out["usage"] = "formal"   # the agent must drop this for "formal casual"
        if "gender" in low:
            out["gender"] = "men"     # a guess without any cue: the agent must ignore it
        m = re.search(r"\b(\d{3})\s*(?:and|,)\s*(\d{2,3})\b", low)  # "178 and 76": bare numbers
        if m:
            out.update(height_value=float(m.group(1)), height_unit="cm",
                       weight_value=float(m.group(2)), weight_unit="kg", intent="size")
        m = re.search(r"(\d)\s*foot\s*(\d+)", low)  # "5 foot 10": raw units for code to convert
        if m:
            out.update(height_value=float(m.group(1)), height_unit="ft", height_inches=float(m.group(2)))
        m = re.search(r"(\d{3})\s*pounds", low)
        if m:
            out.update(weight_value=float(m.group(1)), weight_unit="lb", intent="size")
        if "loose" in low:
            out["fit"] = "loose"
        m = re.search(r"the (first|second|third|blue|last) one", low)
        if m:
            out["reference"], out["intent"] = m.group(0), "select"
        if "any colour" in low:
            out["clear"] = ["colours", "avoid_colours"]
        return out

    def chat_stream(self, messages, max_tokens=512, temperature=0.4):
        self.calls.append("stream")
        facts = json.loads(messages[-1]["content"][len("FACTS: "):])
        yield "Here "
        yield "you go."
        if "ask_height_weight" in facts.get("todo", ""):
            yield " What's your height and weight?"

    def embed(self, texts):
        rng = np.random.default_rng(0)
        v = rng.normal(size=(len(texts), 8)).astype("float32")
        return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    from app import models, store

    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("TRYON_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "state.sqlite")
    fp = FakeProvider()
    monkeypatch.setattr(models, "chat", fp.chat)
    monkeypatch.setattr(models, "chat_stream", fp.chat_stream)
    monkeypatch.setattr(models, "embed", fp.embed)
    monkeypatch.setattr(models, "search_embed", lambda: None)  # keyword ranking, no network
    return fp
