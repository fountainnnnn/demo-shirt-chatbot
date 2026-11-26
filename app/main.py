"""HTTP API (docs/contracts.md "HTTP API") plus the hosted-mode additions:

- GET /api/status adds "mode", "tryon_provider", "privacy".
- POST /api/chat accepts "remove_filter": <chip key> (clears that criterion, no extract call).
- POST /api/size accepts "conversation_id" (stores fit/chest/height/weight in that conversation)
  and returns "chart_room_cm".
- Shortlist holds at most 3 (HTTP 409 beyond that).
- Daily rate limits per IP and global (HTTP 429), uploads <= 8 MB, re-encoded, deleted after 24 h.
- Each visitor is identified by an opaque random cookie; conversations, shortlist, photos
  and try-ons are scoped to it.

Run: .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import secrets
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from app import llm  # noqa: F401  (must come before anything imports llama_cpp: it picks Metal or CPU)
from app import agent, catalog, models, sizing, store, tryon

log = logging.getLogger("fitting-room")

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
IMAGES_DIR = catalog.CATALOG_DIR / "images"
UPLOAD_DIR = ROOT / "data" / "uploads"
TRYON_DIR = ROOT / "data" / "tryon"

MAX_UPLOAD = 8 * 1024 * 1024
MAX_SIDE = 2048
PHOTO_TTL_S = 24 * 3600
SHORTLIST_MAX = 3
COOKIE = "fr_client"

_ID = re.compile(r"^[0-9a-f]{16,32}$")
Image.MAX_IMAGE_PIXELS = 40_000_000  # refuse decompression bombs


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


LIMITS = {
    "chat": (_env_int("RATE_CHAT_PER_IP", 60), _env_int("RATE_CHAT_GLOBAL", 600)),
    "tryon": (_env_int("RATE_TRYON_PER_IP", 5), _env_int("RATE_TRYON_GLOBAL", 60)),
}


def chat_hosted() -> bool:
    return models.provider() == "openrouter"


def tryon_hosted() -> bool:
    return tryon.provider() == "openrouter"


def mode() -> str:
    return "hosted" if (chat_hosted() or tryon_hosted()) else "local"


# ------------------------------------------------------------------ background work
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tryon")
_stop = threading.Event()


def cleanup() -> int:
    """Delete uploads and try-on outputs older than 24 h. Returns the number of files removed."""
    removed = 0
    for entry in store.expire(PHOTO_TTL_S):
        if entry.startswith("tryon:"):
            path = TRYON_DIR / f"{entry[6:]}.png"
        else:
            path = Path(entry)
        try:
            if path.resolve().parent in (UPLOAD_DIR.resolve(), TRYON_DIR.resolve()) and path.exists():
                path.unlink()
                removed += 1
        except OSError:
            pass
    # Files without a row (e.g. a crash between save and insert) also go after 24 h.
    cutoff = time.time() - PHOTO_TTL_S
    for folder in (UPLOAD_DIR, TRYON_DIR):
        if folder.exists():
            for p in folder.iterdir():
                if p.is_file() and _ID.match(p.stem) and p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
                    removed += 1
    return removed


def _cleanup_loop():
    while not _stop.wait(3600):
        try:
            cleanup()
        except Exception:
            log.exception("cleanup failed")


def _warm():
    """Load the local models in the background so the first request does not pay for it."""
    t0 = time.time()
    try:
        if not chat_hosted():
            llm.chat([{"role": "user", "content": "hi"}], max_tokens=1)
            llm.embed(["warm up"])
        if tryon.provider() in ("dmvton", "fashn"):
            from app import tryon_composite
            tryon_composite.available()
            if tryon.provider() == "dmvton":
                from app import tryon_dmvton
                tryon_dmvton.load()
        log.info("models warm in %.1fs", time.time() - t0)
    except Exception:
        log.exception("warm-up failed (the first request will load the models)")


@asynccontextmanager
async def lifespan(_app):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    TRYON_DIR.mkdir(parents=True, exist_ok=True)
    store.fail_unfinished_tryons()
    cleanup()
    t = threading.Thread(target=_cleanup_loop, daemon=True, name="cleanup")
    t.start()
    if os.environ.get("WARM_MODELS", "1") != "0":
        threading.Thread(target=_warm, daemon=True, name="warm").start()
    yield
    _stop.set()


app = FastAPI(title="Fitting Room", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def client_cookie(request: Request, call_next):
    cid = request.cookies.get(COOKIE, "")
    fresh = not re.fullmatch(r"[A-Za-z0-9_-]{20,64}", cid or "")
    if fresh:
        cid = secrets.token_urlsafe(24)
    request.state.client = cid
    response = await call_next(request)
    if fresh:
        response.set_cookie(COOKIE, cid, max_age=180 * 86400, httponly=True, samesite="lax",
                            secure=os.environ.get("COOKIE_SECURE") == "1")
    return response


def client_of(request: Request) -> str:
    return request.state.client


def ip_of(request: Request) -> str:
    if os.environ.get("TRUST_PROXY") == "1":
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd.strip():
            # The rightmost entry is the one our own proxy appended; earlier ones are client-supplied.
            return fwd.split(",")[-1].strip()[:64]
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request, kind: str) -> None:
    per_ip, global_ = LIMITS[kind]
    full = store.take(kind, [(f"ip:{ip_of(request)}", per_ip), ("global", global_)])
    if full is None:
        return
    what = "messages" if kind == "chat" else "try-ons"
    if full == "global":
        msg = f"The demo has used up today's {what}. Please come back tomorrow."
    else:
        msg = f"You've reached today's limit of {per_ip} {what}. Please come back tomorrow."
    raise HTTPException(status_code=429, detail=msg)


def _item_or_404(item_id) -> dict:
    item = catalog.get(item_id)
    if item is None:
        raise HTTPException(404, "No shirt with that id.")
    return item


# ------------------------------------------------------------------ status
@app.get("/api/status")
def status():
    hosted = mode() == "hosted"
    chat_device = "openrouter"
    if not chat_hosted():
        try:
            chat_device = llm.device()
        except Exception:
            chat_device = "unknown"
    return {
        "chat_model": models.chat_model_name(),
        "chat_device": chat_device,
        "device": chat_device,
        "tryon_device": tryon.device(),
        "tryon_model": tryon.model_name(),
        "ready": True,
        "mode": mode(),
        "tryon_provider": tryon.provider(),
        "privacy": ("Hosted demo. Your messages and try-on photos are sent to OpenRouter." if hosted
                    else "Runs on this laptop, offline."),
    }


# ------------------------------------------------------------------ chat
class ChatIn(BaseModel):
    conversation_id: str | None = None
    message: str = Field("", max_length=4000)
    remove_filter: str | None = Field(None, max_length=64)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
def chat(body: ChatIn, request: Request):
    if not body.message.strip() and body.remove_filter is None:
        raise HTTPException(422, "Type a message first.")
    conv_id = body.conversation_id if body.conversation_id and _ID.match(body.conversation_id) else None
    if chat_hosted():
        rate_limit(request, "chat")
    client = client_of(request)

    def gen():
        # Sync generator: Starlette iterates it in a worker thread, so the
        # blocking provider calls never block the event loop.
        try:
            for event, data in agent.run_turn(conv_id, body.message, client, remove_filter=body.remove_filter):
                yield _sse(event, data)
        except Exception:
            log.exception("chat turn failed")
            yield _sse("text", {"delta": "Sorry, something went wrong on my side. Try that again in a moment."})
            yield _sse("done", {})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/conversations")
def conversations(request: Request):
    return store.list_conversations(client_of(request))


@app.get("/api/conversations/{conv_id}")
def conversation(conv_id: str, request: Request):
    if not _ID.match(conv_id):
        raise HTTPException(404, "Conversation not found.")
    view = agent.conversation_view(conv_id, client_of(request))
    if view is None:
        raise HTTPException(404, "Conversation not found.")
    return view


# ------------------------------------------------------------------ items and sizing
WORN_OPEN_NOTE = ("In the catalog photo this shirt is worn open over a T-shirt, so the picture may "
                  "include the layer underneath.")


@app.get("/api/items/{item_id}")
def item(item_id: int):
    it = _item_or_404(item_id)
    return {**it, "worn_open": catalog.worn_open(it["id"])}


class SizeIn(BaseModel):
    item_id: int
    profile: dict = {}
    conversation_id: str | None = None


def _clean_profile(p: dict) -> dict:
    out = {}
    for key in ("height_cm", "weight_kg", "chest_cm"):
        v = p.get(key)
        if v not in (None, ""):
            try:
                out[key] = float(v)
            except (TypeError, ValueError):
                raise HTTPException(422, f"{key} must be a number.")
    if p.get("fit") not in (None, ""):
        out["fit"] = str(p["fit"])
    if p.get("gender") in ("men", "women", "boys", "girls", "unisex"):
        out["gender"] = p["gender"]
    return out


@app.post("/api/size")
def size(body: SizeIn, request: Request):
    item = _item_or_404(body.item_id)
    prof = _clean_profile(body.profile or {})
    if not agent.has_body(prof):
        raise HTTPException(422, "I need your chest measurement, or your height and weight.")
    try:
        res = agent.size_for(item, prof)
    except ValueError as e:
        raise HTTPException(422, f"Those numbers don't look right: {e}.")
    if body.conversation_id and _ID.match(body.conversation_id):
        agent.set_profile(body.conversation_id, client_of(request), body.profile or {}, item["id"], res)
    return res


# ------------------------------------------------------------------ shortlist
def _shortlist_view(client: str) -> list[dict]:
    prof = store.latest_profile(client)
    out = []
    for iid in store.shortlist(client):
        it = catalog.get(iid)
        if it is None:
            continue
        try:
            sz = agent.size_for(it, prof)
        except ValueError:
            sz = None
        out.append({**it, "worn_open": catalog.worn_open(it["id"]), "size": sz})
    return out


class ShortlistIn(BaseModel):
    item_id: int


@app.get("/api/shortlist")
def shortlist_get(request: Request):
    return _shortlist_view(client_of(request))


@app.post("/api/shortlist")
def shortlist_add(body: ShortlistIn, request: Request):
    _item_or_404(body.item_id)
    client = client_of(request)
    current = store.shortlist(client)
    if body.item_id not in current and len(current) >= SHORTLIST_MAX:
        raise HTTPException(409, "The shortlist holds three shirts. Take one off to add another.")
    store.shortlist_add(client, body.item_id)
    return _shortlist_view(client)


@app.delete("/api/shortlist/{item_id}")
def shortlist_remove(item_id: int, request: Request):
    store.shortlist_remove(client_of(request), item_id)
    return _shortlist_view(client_of(request))


# ------------------------------------------------------------------ photos
@app.post("/api/photo")
async def photo(request: Request, file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "That photo is over 8 MB. Use a smaller one.")
    if not data:
        raise HTTPException(422, "That file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ("JPEG", "PNG", "WEBP", "HEIF", "MPO", "GIF", "BMP"):
                raise ValueError("format")
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            clean = Image.new("RGB", im.size)  # fresh image: no EXIF, GPS or other metadata
            clean.paste(im)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(415, "That file is not a photo we can read. Use a JPG or PNG.")
    photo_id = uuid.uuid4().hex
    path = UPLOAD_DIR / f"{photo_id}.jpg"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    clean.save(path, "JPEG", quality=90)
    check = tryon.check_photo(str(path))
    store.add_photo(photo_id, client_of(request), str(path), check)
    return {"photo_id": photo_id, "url": f"/media/uploads/{photo_id}.jpg", "check": check}


# ------------------------------------------------------------------ try-on
class TryonIn(BaseModel):
    photo_id: str
    item_id: int
    quality: str | None = "balanced"


STAGE_QUEUED = "Waiting in line"


def _run_job(job_id: str, person: str, item: dict, quality: str):
    store.update_tryon(job_id, status="running", stage="Reading your photo")
    try:
        res = tryon.run_tryon(person, str(catalog.CATALOG_DIR / item["image"]), str(TRYON_DIR / f"{job_id}.png"),
                              quality=quality, garment=item,
                              on_stage=lambda text: store.update_tryon(job_id, stage=text))
        store.update_tryon(job_id, status="done", stage="Done", seconds=res.get("seconds"),
                           url=f"/media/tryon/{job_id}.png", model=res.get("model") or res.get("device"),
                           cost_usd=res.get("cost_usd"), drift=res.get("drift"), retried=res.get("retried"))
    except NotImplementedError as e:
        store.update_tryon(job_id, status="error", stage="The try-on didn't finish.", error=str(e))
    except Exception as e:
        log.warning("try-on %s failed: %s", job_id, type(e).__name__)
        store.update_tryon(job_id, status="error", stage="The try-on didn't finish.",
                           error="The try-on service did not return a picture. Try again.")


@app.post("/api/tryon")
def start_tryon(body: TryonIn, request: Request):
    if not _ID.match(body.photo_id):
        raise HTTPException(404, "Photo not found. Upload it again.")
    client = client_of(request)
    ph = store.get_photo(body.photo_id, client)
    if ph is None:
        raise HTTPException(404, "Photo not found. It may have expired; upload it again.")
    if not ph["check"].get("ok"):
        raise HTTPException(422, "This photo won't work for a try-on. Use another one.")
    item = _item_or_404(body.item_id)
    quality = body.quality if body.quality in ("fast", "balanced", "best") else "balanced"
    cached = store.find_tryon(body.photo_id, item["id"])
    if cached:
        return {"job_id": cached["job_id"]}
    if tryon_hosted():
        rate_limit(request, "tryon")
    job_id = uuid.uuid4().hex
    store.add_tryon(job_id, client, body.photo_id, item["id"], STAGE_QUEUED)
    _executor.submit(_run_job, job_id, ph["path"], item, quality)
    return {"job_id": job_id}


@app.get("/api/tryon/{job_id}")
def tryon_status(job_id: str, request: Request):
    job = store.get_tryon(job_id, client_of(request)) if _ID.match(job_id) else None
    if job is None:
        raise HTTPException(404, "Try-on not found.")
    note = WORN_OPEN_NOTE if catalog.worn_open(job["item_id"]) and tryon.provider() != "openrouter" else None
    return {"status": job["status"], "stage": job["stage"], "seconds": job["seconds"], "note": note,
            "url": None if job["deleted"] else job["url"], "error": job["error"],
            "drift": job.get("drift"), "retried": bool(job.get("retried")) if job.get("retried") is not None else None}


@app.get("/api/tryons")
def tryons(request: Request):
    return store.list_tryons(client_of(request))


# ------------------------------------------------------------------ files
@app.get("/images/{name}")
def catalog_image(name: str):
    m = re.fullmatch(r"(\d{1,9})\.jpg", name)
    item = catalog.get(int(m.group(1))) if m else None
    if item is None:
        raise HTTPException(404, "Not found.")
    path = IMAGES_DIR / f"{item['id']}.jpg"
    if not path.exists():
        raise HTTPException(404, "Not found.")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/media/uploads/{name}")
def media_upload(name: str):
    m = re.fullmatch(r"([0-9a-f]{32})\.jpg", name)
    ph = store.get_photo(m.group(1)) if m else None
    if ph is None or not Path(ph["path"]).exists():
        raise HTTPException(404, "Not found.")
    return FileResponse(UPLOAD_DIR / f"{m.group(1)}.jpg", media_type="image/jpeg",
                        headers={"Cache-Control": "private, no-store"})


@app.get("/media/tryon/{name}")
def media_tryon(name: str):
    m = re.fullmatch(r"([0-9a-f]{32})\.png", name)
    job = store.get_tryon(m.group(1)) if m else None
    path = TRYON_DIR / f"{m.group(1)}.png" if m else None
    if job is None or job["deleted"] or job["status"] != "done" or not path.exists():
        raise HTTPException(404, "Not found.")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, no-store"})


@app.exception_handler(HTTPException)
async def http_error(_request: Request, exc: HTTPException):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
