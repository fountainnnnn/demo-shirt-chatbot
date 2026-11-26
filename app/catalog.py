"""Catalog access and search (contract: docs/contracts.md).

    load() -> list[dict]
    get(item_id) -> dict | None
    search(filters, query="", k=8, embed=None) -> list[dict]

Hard filters first, then rank: cosine similarity of the query to the item
descriptions when `embed` is given and data/catalog/embeddings.npy exists,
otherwise IDF-weighted keyword overlap with the description. If the hard
filters leave fewer than 3 items they are relaxed in a fixed order.
"""
from __future__ import annotations

import json
import math
import pathlib
import re

CATALOG_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "catalog"
MIN_RESULTS = 3

_cache: dict = {}


def load() -> list[dict]:
    path = CATALOG_DIR / "catalog.json"
    key = (str(path), path.stat().st_mtime_ns)
    if _cache.get("key") != key:
        items = json.loads(path.read_text())
        _cache.clear()
        _cache.update(key=key, items=items, by_id={i["id"]: i for i in items})
    return _cache["items"]


def worn_open(item_id) -> bool:
    """True when the catalog photo shows the shirt worn open over another top
    (data/catalog/worn_open.json, made by scripts/detect_worn_open.py)."""
    path = CATALOG_DIR / "worn_open.json"
    try:
        key = path.stat().st_mtime_ns
    except OSError:
        return False
    if _cache.get("worn_key") != key:
        _cache["worn_key"], _cache["worn"] = key, set(json.loads(path.read_text()))
    try:
        return int(item_id) in _cache["worn"]
    except (TypeError, ValueError):
        return False


def get(item_id: int) -> dict | None:
    load()
    try:
        return _cache["by_id"].get(int(item_id))
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ colours
_COLOUR_NOISE = {"off", "light", "dark", "pale", "bright", "sea", "deep", "dull"}
_EXTRA_FAMILIES = {
    "teal": {"blue", "green"},
    "turquoise": {"blue", "teal", "green"},
    "burgundy": {"maroon", "red"},
    "maroon": {"burgundy", "red"},
    "cream": {"white", "beige"},
    "off white": {"white", "cream"},
    "beige": {"cream", "brown"},
    "charcoal": {"grey"},
    "khaki": {"beige", "brown"},
    "tan": {"brown", "beige"},
    "mustard": {"yellow"},
    "lavender": {"purple"},
    "magenta": {"pink", "purple"},
    "rose": {"pink"},
    "peach": {"pink", "orange"},
    "rust": {"orange", "red", "brown"},
    "coffee brown": {"brown"},
    "olive": {"green"},
    "lime": {"green"},
}
_QUERY_ALIASES = {"gray": "grey", "navy": "navy blue", "off-white": "white", "offwhite": "white"}


def _norm_colour(c: str) -> str:
    c = re.sub(r"\s+", " ", str(c).lower().strip())
    c = _QUERY_ALIASES.get(c, c).replace("gray", "grey")
    words = [w for w in c.split() if w not in _COLOUR_NOISE]
    return " ".join(words) or c


def colour_families(colour: str) -> set[str]:
    """Every name an item of this dataset colour answers to, e.g.
    "navy blue" -> {"navy blue", "navy", "blue"}; "cream" -> {"cream", "white", "beige"}."""
    raw = re.sub(r"\s+", " ", str(colour).lower().strip())
    fam = {raw, _norm_colour(raw)}
    fam.update(w for w in raw.split() if w not in _COLOUR_NOISE)
    fam.update(_EXTRA_FAMILIES.get(raw, ()))
    fam.update(_EXTRA_FAMILIES.get(_norm_colour(raw), ()))
    return fam


def colour_matches(item_colour: str, wanted: str) -> bool:
    return _norm_colour(wanted) in {_norm_colour(f) for f in colour_families(item_colour)}


# ------------------------------------------------------------------ filters
_GENDER = {"men": "men", "man": "men", "male": "men", "mens": "men", "women": "women", "woman": "women",
           "female": "women", "womens": "women", "ladies": "women", "boy": "boys", "boys": "boys",
           "girl": "girls", "girls": "girls", "unisex": "unisex"}
_TYPE = {"shirt": "shirt", "shirts": "shirt", "tshirt": "tshirt", "tshirts": "tshirt", "t-shirt": "tshirt",
         "t-shirts": "tshirt", "tee": "tshirt", "tees": "tshirt", "t shirt": "tshirt"}


def _as_set(v, mapping=None) -> set:
    vals = v if isinstance(v, (list, tuple, set)) else [v]
    out = set()
    for x in vals:
        x = str(x).lower().strip()
        out.add(mapping.get(x, x) if mapping else x)
    return out


def _passes(item: dict, f: dict) -> bool:
    if f.get("gender"):
        g = _as_set(f["gender"], _GENDER)
        if item["gender"] not in g and not (item["gender"] == "unisex" and g & {"men", "women"}):
            return False
    if f.get("type") and item["type"] not in _as_set(f["type"], _TYPE):
        return False
    for key in ("sleeve", "pattern", "usage"):
        if f.get(key) and item[key] not in _as_set(f[key]):
            return False
    if f.get("colours") and not any(colour_matches(item["colour"], c) for c in f["colours"]):
        return False
    if f.get("avoid_colours") and any(colour_matches(item["colour"], c) for c in f["avoid_colours"]):
        return False
    if f.get("max_price") is not None and item["price"] > f["max_price"]:
        return False
    if f.get("min_price") is not None and item["price"] < f["min_price"]:
        return False
    return True


def _filter(items: list[dict], f: dict) -> list[dict]:
    return [i for i in items if _passes(i, f)]


def _relax(f: dict, step: str) -> dict | None:
    """Return f with `step` relaxed, or None if f does not use that filter."""
    g = dict(f)
    if step == "pattern" and g.get("pattern"):
        del g["pattern"]
    elif step == "usage" and g.get("usage"):
        del g["usage"]
    elif step == "colours" and g.get("colours"):
        del g["colours"]
    elif step == "price" and g.get("max_price") is not None:
        g["max_price"] = g["max_price"] * 1.25
    else:
        return None
    return g


# ------------------------------------------------------------------ ranking
_STOP = {"a", "an", "the", "for", "in", "by", "of", "and", "or", "with", "to", "me", "i", "want", "need",
         "something", "looking", "show", "find", "some", "that", "is", "it", "my", "on", "at", "like",
         "please", "shirt", "shirts", "tshirts"}  # "shirt" is in every description


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3 and not (suf == "s" and w.endswith("ss")):
            w = w[: -len(suf)]
            break
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    return w


def _tokens(text: str, drop_stop: bool = False) -> list[str]:
    t = text.lower()
    t = re.sub(r"\bt[- ]?shirts?\b|\btees?\b", " tshirt ", t)
    out = []
    for w in re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", t):
        w = {"gray": "grey", "mens": "men", "womens": "women", "ladies": "women"}.get(w, w)
        if drop_stop and w in _STOP:
            continue
        out.append(_stem(w))
    return out


def _keyword_scores(query: str, items: list[dict]) -> dict[int, float]:
    q = set(_tokens(query, drop_stop=True))
    if not q:
        return {i["id"]: 0.0 for i in items}
    docs = {i["id"]: set(_tokens(i["description"])) for i in load()}
    n = len(docs)
    idf = {t: math.log(1 + n / (1 + sum(t in d for d in docs.values()))) for t in q}
    total = sum(idf.values())
    return {i["id"]: sum(idf[t] for t in q if t in docs[i["id"]]) / total for i in items}


def _embedding_scores(query: str, items: list[dict], embed) -> dict[int, float] | None:
    import numpy as np

    npy, ids_path = CATALOG_DIR / "embeddings.npy", CATALOG_DIR / "embeddings_ids.json"
    if embed is None or not query.strip() or not npy.exists() or not ids_path.exists():
        return None
    ids = json.loads(ids_path.read_text())
    mat = np.load(npy)
    if mat.shape[0] != len(ids):
        return None
    row = {iid: r for r, iid in enumerate(ids)}
    if any(i["id"] not in row for i in items):
        return None
    qv = np.asarray(embed([query]), dtype="float32").reshape(-1)
    qv = qv / (np.linalg.norm(qv) or 1.0)
    sims = mat @ qv
    return {i["id"]: float(sims[row[i["id"]]]) for i in items}


# ------------------------------------------------------------------ near matches
# Hard filters are never relaxed: gender, type (shirt vs T-shirt), colours to avoid, min price.
# Everything else is scored per item: an item that breaks fewer criteria always ranks above
# one that breaks more, then lower total severity, then relevance.
SMALL, MEDIUM = 1, 2
HARD_KEYS = ("gender", "type", "avoid_colours", "min_price")
COLOUR_NEAR = {  # wanted colour -> item colours that are a near miss (small, not medium)
    "black": {"grey", "charcoal"}, "grey": {"black", "charcoal", "silver"}, "charcoal": {"black", "grey"},
    "white": {"cream", "beige", "off white", "grey"}, "cream": {"white", "beige"}, "beige": {"cream", "brown", "khaki"},
    "blue": {"teal", "turquoise"}, "navy blue": {"blue"}, "navy": {"blue"}, "teal": {"blue", "green"},
    "red": {"maroon", "burgundy", "pink"}, "maroon": {"red", "burgundy", "brown"}, "burgundy": {"red", "maroon"},
    "pink": {"red", "purple"}, "purple": {"pink"}, "green": {"teal", "olive"}, "yellow": {"orange", "mustard"},
    "brown": {"beige", "maroon"},
}


def _hard(f: dict) -> dict:
    return {k: f[k] for k in HARD_KEYS if f.get(k) not in (None, "", [])}


def violations(item: dict, f: dict) -> list[dict] | None:
    """Soft criteria this item breaks, each {"key", "severity"}; None if it is too far off to show."""
    out = []
    if f.get("max_price") is not None and item["price"] > f["max_price"]:
        over = (item["price"] - f["max_price"]) / max(f["max_price"], 1)
        if over > 0.60:
            return None
        out.append({"key": "price", "severity": SMALL if over <= 0.35 else MEDIUM})
    if f.get("colours") and not any(colour_matches(item["colour"], c) for c in f["colours"]):
        fam = colour_families(item["colour"])
        near = any(_norm_colour(c) in COLOUR_NEAR and fam & COLOUR_NEAR[_norm_colour(c)] for c in f["colours"])
        out.append({"key": "colours", "severity": SMALL if near else MEDIUM})
    for key, sev in (("pattern", MEDIUM), ("sleeve", MEDIUM), ("usage", SMALL)):
        if f.get(key) and item[key] not in _as_set(f[key]):
            out.append({"key": key, "severity": sev})
    return out


def search(filters: dict | None = None, query: str = "", k: int = 8, embed=None) -> list[dict]:
    """Exact matches first. If fewer than MIN_RESULTS items meet every criterion, near matches
    fill up to k, ranked by (number of violations, total severity, relevance). Each result has
    "score", "exact", "violations" [{"key","severity"}] and "relaxed" (the violated keys)."""
    filters = {key: v for key, v in (filters or {}).items() if v not in (None, "", [])}
    items = _filter(load(), _hard(filters))
    scored = []
    for i in items:
        v = violations(i, filters)
        if v is not None:
            scored.append((i, v))
    exact = [i for i, v in scored if not v]
    pool = scored if len(exact) < MIN_RESULTS else [(i, []) for i in exact]
    cand = [i for i, _ in pool]
    scores = _embedding_scores(query, cand, embed) or _keyword_scores(query, cand)
    def rank(entry):
        i, v = entry
        return (len(v), sum(x["severity"] for x in v), -scores[i["id"]], i["price"], i["id"])
    ranked = sorted(pool, key=rank)[:k]
    return [{**i, "score": round(scores[i["id"]], 4), "exact": not v, "violations": v,
             "relaxed": [x["key"] for x in v]} for i, v in ranked]
