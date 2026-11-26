"""HTTP API with FastAPI's TestClient and the fake provider (offline)."""
import io
import json
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.fakes import fake  # noqa: F401  (fixture)


@pytest.fixture
def api(fake, monkeypatch, tmp_path):
    from app import main, tryon_openrouter

    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(main, "TRYON_DIR", tmp_path / "tryon")
    monkeypatch.setattr(main, "LIMITS", {"chat": (60, 600), "tryon": (5, 60)})
    calls = []

    def fake_tryon(person, garment_path, out, quality="balanced", **kw):
        calls.append((person, garment_path))
        Image.new("RGB", (600, 800), "lightblue").save(out)
        return {"seconds": 0.01, "device": "openrouter", "model": "fake", "cost_usd": 0.0}

    monkeypatch.setattr(tryon_openrouter, "run_tryon", fake_tryon)
    with TestClient(main.app) as c:
        c.tryon_calls = calls
        yield c


def sse(resp):
    out = []
    for block in resp.text.strip().split("\n\n"):
        ev = data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        out.append((ev, data))
    return out


def chat(c, message, conv=None, **extra):
    r = c.post("/api/chat", json={"conversation_id": conv, "message": message, **extra})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    return sse(r)


def jpeg(w=600, h=800, colour="white", exif=False) -> bytes:
    buf = io.BytesIO()
    im = Image.new("RGB", (w, h), colour)
    kw = {}
    if exif:
        ex = Image.Exif()
        ex[0x010F] = "SecretCam"  # Make
        kw["exif"] = ex
    im.save(buf, "JPEG", **kw)
    return buf.getvalue()


FULL = "Men's short-sleeve shirt for a wedding, nothing white, under $60"


# ------------------------------------------------------------------ status, items, images
def test_status(api):
    s = api.get("/api/status").json()
    assert s["ready"] is True
    assert s["mode"] == "hosted" and s["tryon_provider"] == "openrouter"
    assert "OpenRouter" in s["privacy"]
    assert {"chat_model", "tryon_device"} <= s.keys()


def test_item_and_image(api):
    it = api.get("/api/items/30802").json()
    assert it["id"] == 30802 and "size_chart" in it
    assert api.get("/api/items/1").status_code == 404
    r = api.get("/images/30802.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    for bad in ("/images/1.jpg", "/images/..%2Fcatalog.json", "/images/30802.png"):
        assert api.get(bad).status_code == 404


def test_static_index(api):
    r = api.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()


# ------------------------------------------------------------------ chat
def no_steps(ev):
    return [e for e, _ in ev if e != "step"]


def test_chat_event_order_question_then_results(api):
    ev = chat(api, "A shirt for a beach wedding in June")
    assert ev[0][0] == "conversation"
    assert [(d["id"], d["status"]) for e, d in ev if e == "step"] == [
        ("understand", "start"), ("understand", "done"), ("decide", "start"), ("decide", "done")]
    assert no_steps(ev)[-3:] == ["question", "state", "done"]
    assert dict(ev)["question"]["text"] == "Who is it for?"
    cid = ev[0][1]["id"]
    for answer in ("Men", "Short sleeves"):
        ev2 = chat(api, answer, cid)
        assert "question" in no_steps(ev2)
    ev3 = chat(api, "No limit", cid)
    names3 = no_steps(ev3)
    assert names3[0] == "conversation" and names3[1] == "results"
    assert names3[-2:] == ["state", "done"] and "text" in names3
    res = dict(ev3)["results"]
    assert len(res["items"]) == 3 and all({"reason", "best", "price", "name", "exact"} <= i.keys()
                                          for i in res["items"])
    assert "exact_count" in res
    steps = [d for e, d in ev3 if e == "step" and d["status"] == "done"]
    assert [s["id"] for s in steps][:3] == ["understand", "decide", "search"] and steps[-1]["id"] == "write"


def test_steps_are_persisted_with_messages(api):
    cid = chat(api, FULL)[0][1]["id"]
    msgs = api.get(f"/api/conversations/{cid}").json()["messages"]
    assert "steps" not in msgs[0]
    assert [s["id"] for s in msgs[1]["steps"]] == ["understand", "decide", "search", "pick", "write"]
    assert all(set(s) == {"id", "label", "detail", "ms"} for s in msgs[1]["steps"])


def test_chat_size_and_conversation_view(api):
    ev = chat(api, FULL)
    cid = ev[0][1]["id"]
    ev2 = chat(api, "178 and 76", cid)
    names = no_steps(ev2)
    assert names.index("text") < names.index("size") < names.index("state")
    conv = api.get(f"/api/conversations/{cid}").json()
    assert conv["id"] == cid and conv["title"]
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant", "user", "assistant"]
    assert all(isinstance(m["text"], str) for m in conv["messages"])
    assert conv["results"]["items"] and conv["size"]["item_id"]
    assert conv["state"]["profile"]["height_cm"] == 178.0
    lst = api.get("/api/conversations").json()
    assert lst[0]["id"] == cid and lst[0]["updated"].endswith("Z")


def test_question_message_has_options_in_history(api):
    cid = chat(api, "A shirt for a beach wedding in June")[0][1]["id"]
    msgs = api.get(f"/api/conversations/{cid}").json()["messages"]
    assert msgs[1]["role"] == "assistant" and msgs[1]["options"]


def test_remove_filter_field(api):
    cid = chat(api, FULL)[0][1]["id"]
    ev = chat(api, "", cid, remove_filter="max_price")
    d = dict(ev)
    assert "results" in d and "max_price" not in d["state"]["wants"]


def test_conversations_are_private_per_visitor(api, fake):
    from app import main

    cid = chat(api, FULL)[0][1]["id"]
    with TestClient(main.app) as other:
        assert other.get("/api/conversations").json() == []
        assert other.get(f"/api/conversations/{cid}").status_code == 404


def test_unknown_conversation_404(api):
    assert api.get("/api/conversations/ffffffffffffffff").status_code == 404
    assert api.get("/api/conversations/..%2F..%2Fetc").status_code == 404


def test_empty_message_rejected(api):
    assert api.post("/api/chat", json={"conversation_id": None, "message": "  "}).status_code == 422


def test_chat_rate_limit_429(api, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "LIMITS", {"chat": (2, 600), "tryon": (5, 60)})
    chat(api, "hello")
    chat(api, "hello")
    r = api.post("/api/chat", json={"conversation_id": None, "message": "hello"})
    assert r.status_code == 429 and "limit" in r.json()["detail"]


def test_global_rate_limit_429(api, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "LIMITS", {"chat": (100, 1), "tryon": (5, 60)})
    chat(api, "hello")
    r = api.post("/api/chat", json={"conversation_id": None, "message": "hello"})
    assert r.status_code == 429 and "demo" in r.json()["detail"]


# ------------------------------------------------------------------ size
def test_size_endpoint(api):
    r = api.post("/api/size", json={"item_id": 30802, "profile": {"height_cm": 178, "weight_kg": 76, "fit": "loose"}})
    assert r.status_code == 200
    s = r.json()
    assert s["moved"] == 1 and s["size"] != s["chart_size"]
    assert {"chart_room_cm", "room_cm", "ranges", "garment", "notes", "length_note"} <= s.keys()
    assert api.post("/api/size", json={"item_id": 30802, "profile": {}}).status_code == 422
    assert api.post("/api/size", json={"item_id": 30802, "profile": {"chest_cm": 500}}).status_code == 422
    assert api.post("/api/size", json={"item_id": 1, "profile": {"chest_cm": 100}}).status_code == 404


def test_size_with_conversation_stores_profile(api):
    cid = chat(api, FULL)[0][1]["id"]
    api.post("/api/size", json={"item_id": 30802, "conversation_id": cid,
                                "profile": {"chest_cm": 101, "fit": "slim"}})
    prof = api.get(f"/api/conversations/{cid}").json()["state"]["profile"]
    assert prof["chest_cm"] == 101 and prof["fit"] == "slim"


# ------------------------------------------------------------------ shortlist
def test_shortlist_round_trip_and_cap(api):
    assert api.get("/api/shortlist").json() == []
    ids = [30802, 14052, 15970]
    for iid in ids:
        r = api.post("/api/shortlist", json={"item_id": iid})
        assert r.status_code == 200
    lst = api.get("/api/shortlist").json()
    assert [i["id"] for i in lst] == ids and all("size" in i for i in lst)
    r = api.post("/api/shortlist", json={"item_id": 4943})
    assert r.status_code == 409 and "three" in r.json()["detail"]
    assert api.post("/api/shortlist", json={"item_id": 30802}).status_code == 200  # already in: fine
    after = api.delete("/api/shortlist/14052").json()
    assert [i["id"] for i in after] == [30802, 15970]
    assert api.post("/api/shortlist", json={"item_id": 1}).status_code == 404


def test_shortlist_size_is_null_without_profile(api):
    api.post("/api/shortlist", json={"item_id": 30802})
    assert api.get("/api/shortlist").json()[0]["size"] is None


def test_shortlist_sized_with_latest_profile(api):
    cid = chat(api, FULL)[0][1]["id"]
    chat(api, "178 and 76", cid)
    api.post("/api/shortlist", json={"item_id": 30802})
    assert api.get("/api/shortlist").json()[0]["size"]["size"]


# ------------------------------------------------------------------ photos and try-on
def test_photo_upload_reencodes_and_strips_metadata(api):
    r = api.post("/api/photo", files={"file": ("me.jpg", jpeg(exif=True), "image/jpeg")})
    assert r.status_code == 200
    p = r.json()
    assert p["check"] == {"ok": True, "issues": []}
    img = api.get(p["url"])
    assert img.status_code == 200
    with Image.open(io.BytesIO(img.content)) as im:
        assert not im.getexif()


def test_photo_validation(api):
    assert api.post("/api/photo", files={"file": ("x.txt", b"not an image", "text/plain")}).status_code == 415
    big = b"\xff\xd8" + b"0" * (8 * 1024 * 1024 + 10)
    assert api.post("/api/photo", files={"file": ("big.jpg", big, "image/jpeg")}).status_code == 413
    small = api.post("/api/photo", files={"file": ("s.jpg", jpeg(100, 120), "image/jpeg")}).json()
    assert small["check"]["ok"] is False and small["check"]["issues"][0].endswith(".")
    wide = api.post("/api/photo", files={"file": ("w.jpg", jpeg(1200, 600), "image/jpeg")}).json()
    assert wide["check"]["ok"] is False


def wait_job(c, job_id):
    for _ in range(100):
        j = c.get(f"/api/tryon/{job_id}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_tryon_flow_cache_and_list(api):
    p = api.post("/api/photo", files={"file": ("me.jpg", jpeg(), "image/jpeg")}).json()
    r = api.post("/api/tryon", json={"photo_id": p["photo_id"], "item_id": 30802})
    assert r.status_code == 200
    job = r.json()["job_id"]
    first = api.get(f"/api/tryon/{job}").json()
    assert first["status"] in ("queued", "running", "done") and first["stage"]
    j = wait_job(api, job)
    assert j["status"] == "done" and j["url"] == f"/media/tryon/{job}.png" and j["error"] is None
    assert api.get(j["url"]).status_code == 200
    again = api.post("/api/tryon", json={"photo_id": p["photo_id"], "item_id": 30802}).json()
    assert again["job_id"] == job and len(api.tryon_calls) == 1  # cached per (photo, item)
    assert api.get("/api/tryons").json() == [{"job_id": job, "item_id": 30802, "url": j["url"], "seconds": 0.01}]


def test_tryon_validation(api):
    assert api.post("/api/tryon", json={"photo_id": "../../etc/passwd", "item_id": 30802}).status_code == 404
    assert api.post("/api/tryon", json={"photo_id": "0" * 32, "item_id": 30802}).status_code == 404
    bad = api.post("/api/photo", files={"file": ("s.jpg", jpeg(100, 120), "image/jpeg")}).json()
    assert api.post("/api/tryon", json={"photo_id": bad["photo_id"], "item_id": 30802}).status_code == 422
    assert api.get("/api/tryon/nope").status_code == 404


def test_tryon_error_status(api, monkeypatch):
    from app import tryon_openrouter

    def broken(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(tryon_openrouter, "run_tryon", broken)
    p = api.post("/api/photo", files={"file": ("me.jpg", jpeg(), "image/jpeg")}).json()
    job = api.post("/api/tryon", json={"photo_id": p["photo_id"], "item_id": 14052}).json()["job_id"]
    j = wait_job(api, job)
    assert j["status"] == "error" and j["error"].endswith(".") and j["url"] is None


def test_tryon_rate_limit_429(api, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "LIMITS", {"chat": (60, 600), "tryon": (1, 60)})
    p = api.post("/api/photo", files={"file": ("me.jpg", jpeg(), "image/jpeg")}).json()
    assert api.post("/api/tryon", json={"photo_id": p["photo_id"], "item_id": 30802}).status_code == 200
    r = api.post("/api/tryon", json={"photo_id": p["photo_id"], "item_id": 14052})
    assert r.status_code == 429


def test_cleanup_deletes_old_photos(api, monkeypatch):
    from app import main, store

    p = api.post("/api/photo", files={"file": ("me.jpg", jpeg(), "image/jpeg")}).json()
    store.db().execute("UPDATE photos SET created=created-90000")
    assert main.cleanup() >= 1
    assert api.get(p["url"]).status_code == 404
    assert api.post("/api/tryon", json={"photo_id": p["photo_id"], "item_id": 30802}).status_code == 404


def test_unbuilt_tryon_backends_raise(monkeypatch):
    from app import tryon

    for name in ("oxygen",):
        monkeypatch.setenv("TRYON_PROVIDER", name)
        with pytest.raises(NotImplementedError, match="docs"):
            tryon.run_tryon("a", "b", "c")
