"""SQLite persistence (stdlib sqlite3) at data/state.sqlite.

Everything a visitor creates is scoped by `client`, an opaque random id kept in
a cookie, so visitors of the hosted demo never see each other's conversations,
shortlist, photos or try-ons.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("STATE_DB", ROOT / "data" / "state.sqlite"))

_local = threading.local()
_init_lock = threading.Lock()
_initialised: set[str] = set()

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY, client TEXT NOT NULL, title TEXT NOT NULL DEFAULT '',
    created REAL NOT NULL, updated REAL NOT NULL, state TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS conv_client ON conversations(client, updated);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, conv_id TEXT NOT NULL, role TEXT NOT NULL,
    text TEXT NOT NULL, options TEXT, created REAL NOT NULL, steps TEXT
);
CREATE INDEX IF NOT EXISTS msg_conv ON messages(conv_id, id);
CREATE TABLE IF NOT EXISTS shortlist (
    client TEXT NOT NULL, item_id INTEGER NOT NULL, added REAL NOT NULL,
    PRIMARY KEY (client, item_id)
);
CREATE TABLE IF NOT EXISTS photos (
    id TEXT PRIMARY KEY, client TEXT NOT NULL, path TEXT NOT NULL, created REAL NOT NULL,
    check_json TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS tryons (
    job_id TEXT PRIMARY KEY, client TEXT NOT NULL, photo_id TEXT NOT NULL, item_id INTEGER NOT NULL,
    status TEXT NOT NULL, stage TEXT NOT NULL, seconds REAL, url TEXT, error TEXT,
    model TEXT, cost_usd REAL, created REAL NOT NULL, deleted INTEGER NOT NULL DEFAULT 0,
    drift REAL, retried INTEGER
);
CREATE INDEX IF NOT EXISTS tryon_pair ON tryons(photo_id, item_id);
CREATE TABLE IF NOT EXISTS usage (
    day TEXT NOT NULL, scope TEXT NOT NULL, kind TEXT NOT NULL, count INTEGER NOT NULL,
    PRIMARY KEY (day, scope, kind)
);
"""


def db() -> sqlite3.Connection:
    path = str(DB_PATH)
    conn = getattr(_local, "conns", {}).get(path)
    if conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        with _init_lock:
            if path not in _initialised:
                conn.executescript(SCHEMA)
                cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
                if "steps" not in cols:  # databases made before step events existed
                    conn.execute("ALTER TABLE messages ADD COLUMN steps TEXT")
                tcols = {r[1] for r in conn.execute("PRAGMA table_info(tryons)")}
                for col, typ in (("drift", "REAL"), ("retried", "INTEGER")):
                    if col not in tcols:
                        conn.execute(f"ALTER TABLE tryons ADD COLUMN {col} {typ}")
                _initialised.add(path)
        if not hasattr(_local, "conns"):
            _local.conns = {}
        _local.conns[path] = conn
    return conn


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ------------------------------------------------------------------ conversations
def create_conversation(conv_id: str, client: str, title: str, state: dict) -> None:
    now = time.time()
    db().execute("INSERT INTO conversations (id, client, title, created, updated, state) VALUES (?,?,?,?,?,?)",
                 (conv_id, client, title, now, now, json.dumps(state)))


def get_conversation(conv_id: str, client: str) -> dict | None:
    row = db().execute("SELECT * FROM conversations WHERE id=? AND client=?", (conv_id, client)).fetchone()
    if row is None:
        return None
    return {"id": row["id"], "title": row["title"], "updated": row["updated"], "state": json.loads(row["state"])}


def save_conversation(conv_id: str, state: dict, title: str | None = None) -> None:
    if title is None:
        db().execute("UPDATE conversations SET state=?, updated=? WHERE id=?",
                     (json.dumps(state), time.time(), conv_id))
    else:
        db().execute("UPDATE conversations SET state=?, title=?, updated=? WHERE id=?",
                     (json.dumps(state), title, time.time(), conv_id))


def list_conversations(client: str) -> list[dict]:
    rows = db().execute("SELECT id, title, updated FROM conversations WHERE client=? ORDER BY updated DESC",
                        (client,)).fetchall()
    return [{"id": r["id"], "title": r["title"], "updated": iso(r["updated"])} for r in rows]


def latest_profile(client: str) -> dict:
    """Profile of the visitor's most recently updated conversation that has one."""
    for r in db().execute("SELECT state FROM conversations WHERE client=? ORDER BY updated DESC LIMIT 20",
                          (client,)):
        prof = json.loads(r["state"]).get("profile") or {}
        if (prof.get("height_cm") and prof.get("weight_kg")) or prof.get("chest_cm"):
            return prof
    return {}


def add_message(conv_id: str, role: str, text: str, options: list | None = None,
                steps: list | None = None) -> None:
    db().execute("INSERT INTO messages (conv_id, role, text, options, created, steps) VALUES (?,?,?,?,?,?)",
                 (conv_id, role, text, json.dumps(options) if options else None, time.time(),
                  json.dumps(steps) if steps else None))


def messages(conv_id: str) -> list[dict]:
    out = []
    for r in db().execute("SELECT role, text, options, steps FROM messages WHERE conv_id=? ORDER BY id",
                          (conv_id,)):
        m = {"role": r["role"], "text": r["text"]}
        if r["options"]:
            m["options"] = json.loads(r["options"])
        if r["steps"]:
            m["steps"] = json.loads(r["steps"])
        out.append(m)
    return out


# ------------------------------------------------------------------ shortlist
def shortlist(client: str) -> list[int]:
    return [r["item_id"] for r in db().execute(
        "SELECT item_id FROM shortlist WHERE client=? ORDER BY added", (client,))]


def shortlist_add(client: str, item_id: int) -> None:
    db().execute("INSERT OR IGNORE INTO shortlist (client, item_id, added) VALUES (?,?,?)",
                 (client, item_id, time.time()))


def shortlist_remove(client: str, item_id: int) -> None:
    db().execute("DELETE FROM shortlist WHERE client=? AND item_id=?", (client, item_id))


# ------------------------------------------------------------------ photos
def add_photo(photo_id: str, client: str, path: str, check: dict) -> None:
    db().execute("INSERT INTO photos (id, client, path, created, check_json) VALUES (?,?,?,?,?)",
                 (photo_id, client, path, time.time(), json.dumps(check)))


def get_photo(photo_id: str, client: str | None = None) -> dict | None:
    if client is None:
        row = db().execute("SELECT * FROM photos WHERE id=? AND deleted=0", (photo_id,)).fetchone()
    else:
        row = db().execute("SELECT * FROM photos WHERE id=? AND client=? AND deleted=0",
                           (photo_id, client)).fetchone()
    if row is None:
        return None
    return {"id": row["id"], "path": row["path"], "check": json.loads(row["check_json"]), "created": row["created"]}


# ------------------------------------------------------------------ try-on jobs
_TRYON_COLS = ("job_id", "client", "photo_id", "item_id", "status", "stage", "seconds", "url", "error",
               "model", "cost_usd", "created", "drift", "retried")


def add_tryon(job_id: str, client: str, photo_id: str, item_id: int, stage: str) -> None:
    db().execute("INSERT INTO tryons (job_id, client, photo_id, item_id, status, stage, created) "
                 "VALUES (?,?,?,?,?,?,?)", (job_id, client, photo_id, item_id, "queued", stage, time.time()))


def update_tryon(job_id: str, **fields) -> None:
    keys = [k for k in fields if k in _TRYON_COLS]
    if keys:
        db().execute(f"UPDATE tryons SET {', '.join(k + '=?' for k in keys)} WHERE job_id=?",
                     [fields[k] for k in keys] + [job_id])


def get_tryon(job_id: str, client: str | None = None) -> dict | None:
    q, args = "SELECT * FROM tryons WHERE job_id=?", [job_id]
    if client is not None:
        q, args = q + " AND client=?", args + [client]
    row = db().execute(q, args).fetchone()
    return dict(row) if row else None


def find_tryon(photo_id: str, item_id: int) -> dict | None:
    """A finished or in-flight job for this (photo, item), for the cache."""
    row = db().execute("SELECT * FROM tryons WHERE photo_id=? AND item_id=? AND deleted=0 "
                       "AND status IN ('queued','running','done') ORDER BY created DESC LIMIT 1",
                       (photo_id, item_id)).fetchone()
    return dict(row) if row else None


def list_tryons(client: str) -> list[dict]:
    rows = db().execute("SELECT job_id, item_id, url, seconds FROM tryons WHERE client=? AND status='done' "
                        "AND deleted=0 ORDER BY created DESC", (client,)).fetchall()
    return [dict(r) for r in rows]


def fail_unfinished_tryons() -> None:
    """Jobs left queued/running by a previous process will never finish."""
    db().execute("UPDATE tryons SET status='error', stage='The try-on was interrupted.', "
                 "error='The server restarted. Try again.' WHERE status IN ('queued','running')")


# ------------------------------------------------------------------ cleanup
def expire(max_age_s: float) -> list[str]:
    """Mark photos and try-on outputs older than max_age_s deleted; return their file paths."""
    cutoff = time.time() - max_age_s
    paths = [r["path"] for r in db().execute("SELECT path FROM photos WHERE created<? AND deleted=0", (cutoff,))]
    db().execute("UPDATE photos SET deleted=1 WHERE created<? AND deleted=0", (cutoff,))
    jobs = [r["job_id"] for r in db().execute(
        "SELECT job_id FROM tryons WHERE created<? AND deleted=0", (cutoff,))]
    db().execute("UPDATE tryons SET deleted=1, url=NULL WHERE created<? AND deleted=0", (cutoff,))
    db().execute("DELETE FROM usage WHERE day < ?", (time.strftime("%Y-%m-%d", time.gmtime(cutoff - 86400 * 7)),))
    return paths + [f"tryon:{j}" for j in jobs]


# ------------------------------------------------------------------ rate limits
_usage_lock = threading.Lock()


def take(kind: str, scopes: list[tuple[str, int]]) -> str | None:
    """Count one use of `kind` against every (scope, limit); return the scope that is full, else None.

    Nothing is counted when any scope is already at its limit.
    """
    day = time.strftime("%Y-%m-%d", time.gmtime())
    with _usage_lock:
        conn = db()
        for scope, limit in scopes:
            row = conn.execute("SELECT count FROM usage WHERE day=? AND scope=? AND kind=?",
                               (day, scope, kind)).fetchone()
            if (row["count"] if row else 0) >= limit:
                return scope
        for scope, _ in scopes:
            conn.execute("INSERT INTO usage (day, scope, kind, count) VALUES (?,?,?,1) "
                         "ON CONFLICT(day, scope, kind) DO UPDATE SET count=count+1", (day, scope, kind))
    return None
