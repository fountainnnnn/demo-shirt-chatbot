"""Local mode (the default): no OPENROUTER key needed, and zero network calls at runtime.

app.llm and app.tryon_dmvton are replaced by fakes (no model load), every socket connection
attempt raises, and a full session runs through the HTTP API: status, a conversation with a
clarifying question, results, sizing, shortlist, a photo upload and a try-on.
"""
import io
import json
import socket
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.fakes import FakeProvider


class NetworkUsed(AssertionError):
    pass


@pytest.fixture
def local(monkeypatch, tmp_path):
    from app import llm, main, store, tryon_dmvton

    for var in ("OPENROUTER_API_KEY", "LLM_PROVIDER", "TRYON_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "state.sqlite")
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(main, "TRYON_DIR", tmp_path / "tryon")

    fp = FakeProvider()
    monkeypatch.setattr(llm, "chat", fp.chat)
    monkeypatch.setattr(llm, "chat_stream", fp.chat_stream)
    dim = np.load(main.catalog.CATALOG_DIR / "embeddings.npy").shape[1]

    def embed(texts):
        v = np.random.default_rng(0).normal(size=(len(texts), dim)).astype("float32")
        fp.calls.append("embed")
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    monkeypatch.setattr(llm, "embed", embed)
    monkeypatch.setattr(llm, "device", lambda: "metal")

    stages = []

    def fake_dmvton(person, garment_path, out, quality="balanced", garment=None, on_stage=None):
        for s in ("Reading your photo", "Preparing the shirt", "Fitting the shirt to your photo",
                  "Blending it into your photo"):
            stages.append(s)
            if on_stage:
                on_stage(s)
        Image.new("RGB", (300, 400), "navy").save(out)
        return {"seconds": 0.5, "device": "mps", "model": "dm-vton"}

    monkeypatch.setattr(tryon_dmvton, "run_tryon", fake_dmvton)

    def no_network(*a, **k):
        raise NetworkUsed(f"network call attempted in local mode: {a[:2]}")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    with TestClient(main.app) as c:
        c.fake, c.stages = fp, stages
        yield c


def sse(resp):
    out = []
    for block in resp.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        out.append((lines.get("event"), json.loads(lines.get("data", "{}"))))
    return out


def chat(c, msg, cid=None):
    r = c.post("/api/chat", json={"conversation_id": cid, "message": msg})
    assert r.status_code == 200
    ev = sse(r)
    return ev, next(d["id"] for e, d in ev if e == "conversation")


def test_local_is_default_even_with_a_key(monkeypatch):
    from app import models, tryon

    monkeypatch.setenv("OPENROUTER_API_KEY", "set-but-unused")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("TRYON_PROVIDER", raising=False)
    assert models.provider() == "local" and tryon.provider() == "dmvton"
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    assert models.provider() == "openrouter"


def test_full_local_session_makes_no_network_calls(local):
    s = local.get("/api/status").json()
    assert s["mode"] == "local" and s["tryon_provider"] == "dmvton" and s["device"] == "metal"
    assert s["chat_model"].startswith("Qwen3-4B") and "offline" in s["privacy"]

    ev, cid = chat(local, "A shirt for a beach wedding in June")
    assert dict(ev)["question"]["text"] == "Who is it for?"
    for answer in ("Men", "Short sleeves"):
        chat(local, answer, cid)
    ev, _ = chat(local, "No limit", cid)
    res = dict(ev)["results"]
    assert len(res["items"]) == 3 and all(i["gender"] in ("men", "unisex") for i in res["items"])
    assert "embed" in local.fake.calls, "search used the local embedder"
    ev, _ = chat(local, "178 and 76", cid)
    assert "size" in dict(ev)

    item = res["items"][0]["id"]
    assert local.post("/api/size", json={"item_id": item, "profile": {"chest_cm": 100}}).status_code == 200
    assert local.post("/api/shortlist", json={"item_id": item}).status_code == 200

    buf = io.BytesIO()
    Image.new("RGB", (600, 800), "white").save(buf, "JPEG")
    photo = local.post("/api/photo", files={"file": ("me.jpg", buf.getvalue(), "image/jpeg")}).json()
    assert photo["check"]["ok"]
    job = local.post("/api/tryon", json={"photo_id": photo["photo_id"], "item_id": item}).json()["job_id"]
    for _ in range(200):
        j = local.get(f"/api/tryon/{job}").json()
        if j["status"] == "done":
            break
        time.sleep(0.01)
    assert j["status"] == "done" and local.get(j["url"]).status_code == 200
    assert local.stages[-1] == "Blending it into your photo"


def test_no_rate_limit_in_local_mode(local, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "LIMITS", {"chat": (1, 1), "tryon": (1, 1)})
    for _ in range(3):
        assert local.post("/api/chat", json={"conversation_id": None, "message": "hello"}).status_code == 200
