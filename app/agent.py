"""Agent loop (docs/architecture.md, "Agent"). Code owns the control flow; the model fills in language.

    run_turn(conversation_id, message, client, remove_filter=None) -> Iterator[(event, data)]

Per turn: EXTRACT (schema call) -> DECIDE (code: ask at most two clarifying
questions per search) -> SEARCH (catalog) -> PICK (schema call, ids checked
against the candidates) -> SIZE (app.sizing) -> REPLY (streamed text).
Events follow docs/contracts.md: conversation, question, text, results, size, state, done.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from contextlib import closing
from typing import Iterator

from app import catalog, models, sizing, store

log = logging.getLogger("fitting-room.agent")

MAX_QUESTIONS = 3          # per search: who it is for, then at most two details
MAX_DETAIL_QUESTIONS = 2
MAX_MESSAGE = 1000
DETAIL_SLOTS = ("sleeve", "max_price", "avoid_colours")
SLOTS_TO_ASK = ("gender",) + DETAIL_SLOTS
WANT_KEYS = ("occasion", "type", "sleeve", "pattern", "colours", "avoid_colours", "max_price", "usage", "gender")
PROFILE_KEYS = ("height_cm", "weight_kg", "chest_cm", "fit", "gender")
COLOURS = sorted({c for i in catalog.load() for c in catalog.colour_families(i["colour"])} |
                 {"orange", "olive", "khaki", "lavender", "mustard", "navy"})

QUESTIONS = {
    "gender": ("Who is it for?", ["Men", "Women", "Kids"]),
    "sleeve": ("Short sleeves or long?", ["Short sleeves", "Long sleeves", "Either"]),
    "max_price": ("What would you like to spend at most?", ["Under $30", "Under $50", "Under $70", "No limit"]),
    "avoid_colours": ("Any colour you'd rather avoid?", ["Not white", "Not black", "Nothing too bright", "No preference"]),
}
_BRIGHT = ["red", "yellow", "pink", "orange", "purple"]


# ================================================================== state
def new_state() -> dict:
    return {
        "wants": {}, "profile": {}, "results": None, "size": None, "selected": None,
        "questions": 0, "asked": [], "answered": [], "pending": None, "asked_profile": False,
    }


def public_state(st: dict) -> dict:
    return {"wants": dict(st["wants"]), "profile": dict(st["profile"])}


# ================================================================== units and code-side parsing
def _num(s: str) -> float:
    return float(s.replace(",", "."))


def parse_units(text: str) -> dict:
    """Height/weight/chest/fit/budget written with units, converted to cm/kg in code."""
    t = text.lower()
    out: dict = {}
    m = re.search(r"\b([4-7])\s*(?:'|’|ft|feet|foot)\s*(\d{1,2}(?:\.\d)?)?\s*(?:\"|”|''|in|inches|inch)?", t)
    if m:
        out["height_cm"] = round(int(m.group(1)) * 30.48 + (_num(m.group(2)) if m.group(2) else 0) * 2.54, 1)
    else:
        m = re.search(r"\b(1[0-9]{2}|2[0-2][0-9])(?:[.,]\d)?\s*(?:cm|centimet(?:er|re)s?)\b", t)
        if m:
            out["height_cm"] = float(m.group(1))
        else:
            m = re.search(r"\b([12][.,]\d{1,2})\s*(?:m|metres?|meters?)\b", t)
            if m:
                out["height_cm"] = round(_num(m.group(1)) * 100, 1)
    if "height_cm" not in out:  # "178 and 76": a bare height and weight
        m = re.search(r"\b(1[4-9]\d|20\d)\s*(?:cm)?\s*(?:and|,|/|&)\s*(\d{2,3})\s*(?:kg)?\b", t)
        if m and 35 <= int(m.group(2)) <= 160:
            out["height_cm"], out["weight_kg"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"\b(\d{2,3}(?:[.,]\d)?)\s*(kg|kgs|kilos?|kilograms?|lb|lbs|pounds?)\b", t)
    if m:
        v = _num(m.group(1))
        out["weight_kg"] = round(v * 0.45359237, 1) if m.group(2).startswith(("lb", "pound")) else v
    else:
        m = re.search(r"\b(\d{1,2})\s*(?:st|stone)\s*(\d{1,2})?\b", t)
        if m:
            out["weight_kg"] = round((int(m.group(1)) * 14 + int(m.group(2) or 0)) * 0.45359237, 1)
    m = re.search(r"chest(?:\s+is|\s+of|:)?\s*(?:about\s+)?(\d{2,3}(?:[.,]\d)?)\s*(cm|in|inch|inches|\"|”)?", t)
    if m:
        v = _num(m.group(1))
        unit = m.group(2) or ("in" if v < 60 else "cm")
        out["chest_cm"] = round(v * 2.54, 1) if unit != "cm" else v
    if re.search(r"\b(loose|looser|relaxed|roomy|baggy|oversized?)\b", t):
        out["fit"] = "loose"
    elif re.search(r"\b(slim|slimmer|fitted|tight|snug)\b(?!\s+budget)", t):
        out["fit"] = "slim"
    elif re.search(r"\bregular fit\b", t):
        out["fit"] = "regular"
    m = re.search(r"(?:under|below|less than|max(?:imum)?|up to|no more than|not more than|at most|within|"
                  r"cheaper than|nothing (?:over|above|more than)|not over|budget(?: is| of)?|spend(?:ing)?)"
                  r"\s*(?:about\s+)?\$?\s*(\d{2,4})", t)
    if not m:  # "rather not spend more than 70 dollars", "nothing over $40", "60 bucks max"
        m = re.search(r"(?:not|n't|never|nothing|no)\b[\w\s']{0,20}?(?:more than|over|above)\s*\$?\s*(\d{2,4})", t) \
            or re.search(r"\$?(\d{2,4})\s*(?:dollars|bucks|usd)?\s*(?:max|tops|or less|at most)\b", t)
    if m:
        out["max_price"] = int(m.group(1))
    return out


def convert_model_units(p: dict) -> dict:
    """Turn the extractor's raw values + units into cm/kg (code, not the model, does the maths)."""
    out: dict = {}
    hv, hu = p.get("height_value"), p.get("height_unit")
    if isinstance(hv, (int, float)) and hv > 0:
        if hu == "ft":
            out["height_cm"] = round(hv * 30.48 + float(p.get("height_inches") or 0) * 2.54, 1)
        elif hu == "in":
            out["height_cm"] = round(hv * 2.54, 1)
        elif hu == "m" or (hu is None and hv < 3):
            out["height_cm"] = round(hv * 100, 1)
        elif hu is None and hv < 8:
            out["height_cm"] = round(hv * 30.48 + float(p.get("height_inches") or 0) * 2.54, 1)
        else:
            out["height_cm"] = float(hv)
    wv, wu = p.get("weight_value"), p.get("weight_unit")
    if isinstance(wv, (int, float)) and wv > 0:
        if wu == "lb" or (wu is None and wv > 160):
            out["weight_kg"] = round(wv * 0.45359237, 1)
        elif wu == "st":
            out["weight_kg"] = round(wv * 6.35029318, 1)
        else:
            out["weight_kg"] = float(wv)
    cv, cu = p.get("chest_value"), p.get("chest_unit")
    if isinstance(cv, (int, float)) and cv > 0:
        out["chest_cm"] = round(cv * 2.54, 1) if (cu == "in" or (cu is None and cv < 60)) else float(cv)
    return out


def _valid_profile_values(p: dict) -> dict:
    ok = {}
    if p.get("height_cm") is not None and 100 <= p["height_cm"] <= 230:
        ok["height_cm"] = p["height_cm"]
    if p.get("weight_kg") is not None and 25 <= p["weight_kg"] <= 250:
        ok["weight_kg"] = p["weight_kg"]
    if p.get("chest_cm") is not None and 60 <= p["chest_cm"] <= 150:
        ok["chest_cm"] = p["chest_cm"]
    if p.get("fit") in ("slim", "regular", "loose"):
        ok["fit"] = p["fit"]
    return ok


def keyword_wants(text: str) -> dict:
    """Crude extract used only when the model call fails, so a turn still makes progress."""
    t = text.lower()
    x: dict = {"intent": "search" if re.search(r"\b(shirts?|tees?|t-shirts?|polos?)\b", t) else "other"}
    m = re.search(r"\b(short|long)[- ]sleeve", t)
    if m:
        x["sleeve"] = m.group(1)
    if re.search(r"\b(t-?shirts?|tees?|polos?)\b", t):
        x["type"] = "tshirt"
    x["avoid_colours"] = re.findall(r"\b(?:no|not|nothing|avoid)\s+(" + "|".join(map(re.escape, COLOURS)) + r")\b", t)
    for word, pattern in (("plain", "solid"), ("striped", "striped"), ("checked", "checked"), ("printed", "printed")):
        if word in t:
            x["pattern"] = pattern
    if re.search(r"\b(office|work|interview|formal)\b", t):
        x["usage"] = "formal"
    if x["avoid_colours"] or x.get("sleeve"):
        x["intent"] = x["intent"] if x["intent"] == "search" else "refine"
    return x


_MEN = (r"\b(men|men'?s|mens|man|male|males|guy|guys|gents?|gentleman|gentlemen|husband|boyfriend|dad|father|"
        r"son|brother|grandad|grandpa|grandfather|uncle|groom|groomsman|he|him|his)\b")
_WOMEN = (r"\b(women|women'?s|womens|woman|female|females|lady|ladies|girlfriend|wife|mum|mom|mother|daughter|"
          r"sister|grandma|grandmother|aunt|bride|bridesmaid|she|her|hers)\b")
_KIDS = r"\b(kids?|kid'?s|child|children|children'?s|boys?|boy'?s|girls?|girl'?s|toddler|teen|teenager|school ?boy)\b"
_OTHER_CUE = r"\b(partner|myself|unisex)\b"


def detect_gender(text: str) -> str | None:
    """'men' | 'women' | 'kids' when the message clearly says who the shirt is for, else None."""
    t = text.lower()
    found = {g for g, rx in (("men", _MEN), ("women", _WOMEN), ("kids", _KIDS)) if re.search(rx, t)}
    if found == {"kids", "men"} and re.search(r"\bboys?\b", t):
        found = {"kids"}
    if found == {"kids", "women"} and re.search(r"\bgirls?\b", t):
        found = {"kids"}
    return found.pop() if len(found) == 1 else None


def has_gender_cue(text: str) -> bool:
    t = text.lower()
    return any(re.search(rx, t) for rx in (_MEN, _WOMEN, _KIDS, _OTHER_CUE))


def match_option(st: dict, text: str) -> dict | None:
    """Answer to the pending clarifying question given by tapping an option: patch without a model call."""
    slot = st.get("pending")
    if not slot:
        return None
    t = text.strip().lower().rstrip(".!")
    _, options = QUESTIONS[slot]
    if t not in [o.lower() for o in options]:
        return None
    if slot == "gender":
        return {"gender": t, "_trusted_gender": True}
    if slot == "sleeve":
        return {"sleeve": {"short sleeves": "short", "long sleeves": "long", "either": "any"}[t]}
    if slot == "max_price":
        m = re.search(r"\d+", t)
        return {"max_price": int(m.group()) if m else None, "_answered": "max_price"}
    if slot == "avoid_colours":
        return {"not white": {"avoid_colours": ["white"]}, "not black": {"avoid_colours": ["black"]},
                "nothing too bright": {"avoid_colours": list(_BRIGHT)},
                "no preference": {"_answered": "avoid_colours"}}[t]
    return None


# ================================================================== extract
# The model reads only what code cannot: intent, title, occasion, references, clearing, and
# bare or unusual numbers. Sleeve, colours, pattern, type, usage, budget, gender and fit come
# from code_extract()/parse_units() (small local models add fields nobody stated).
EXTRACT_SCHEMA = {
    "title": "extract",
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["search", "refine", "size", "select", "smalltalk", "other"]},
        "title": {"type": ["string", "null"]},
        "occasion": {"type": ["string", "null"]},
        "colours": {"type": "array", "items": {"type": "string"}},
        "avoid_colours": {"type": "array", "items": {"type": "string"}},
        "reference": {"type": ["string", "null"]},
        "clear": {"type": "array", "items": {"type": "string", "enum": [
            "occasion", "type", "sleeve", "pattern", "colours", "avoid_colours", "max_price", "usage", "gender"]}},
        "height_value": {"type": ["number", "null"]},
        "height_unit": {"type": ["string", "null"], "enum": ["cm", "m", "ft", "in", None]},
        "height_inches": {"type": ["number", "null"]},
        "weight_value": {"type": ["number", "null"]},
        "weight_unit": {"type": ["string", "null"], "enum": ["kg", "lb", "st", None]},
        "chest_value": {"type": ["number", "null"]},
        "chest_unit": {"type": ["string", "null"], "enum": ["cm", "in", None]},
    },
    "required": ["intent", "title", "occasion", "colours", "avoid_colours", "reference", "clear",
                 "height_value", "height_unit", "height_inches", "weight_value", "weight_unit",
                 "chest_value", "chest_unit"],
    "additionalProperties": False,
}

EXTRACT_SYSTEM = """Read the shopper's new message to a shirt shop assistant and return JSON. Only record what
THIS message says; null or [] for anything else. Never guess.
intent: search (asks for shirts / new request), refine (changes criteria of the shown shirts), size (gives
height, weight, chest or fit), select (points at a shown shirt), smalltalk, other.
title: only if there is no conversation yet: 2-5 words naming the request, e.g. "Beach wedding in June".
occasion: what or where it is for, lower case with an article: "a beach wedding", "the office". Not a garment:
"a shirt" is not an occasion (null).
colours / avoid_colours: colour words they want / do not want, as written.
reference: their words pointing at a shown shirt ("the first one", "the blue one").
clear: criteria they want dropped ("any colour is fine" -> ["colours", "avoid_colours"]).
height/weight/chest: copy numbers and units as written, do not convert ("5 ft 10" -> 5, "ft", 10). After being
asked for height and weight, a bare "178 and 76" means 178 cm and 76 kg."""


def _context(st: dict, history: list[dict]) -> str:
    lines = []
    for m in history[-6:]:
        lines.append(f"{m['role']}: {m['text'][:300]}")
    shown = ""
    if st.get("results"):
        shown = "; ".join(f"{n + 1}. {i['name']} ({i['brand']})" for n, i in enumerate(st["results"]["items"]))
    return (f"Conversation so far:\n{chr(10).join(lines) or '(none)'}\n"
            f"Current criteria: {json.dumps(st['wants'])}\n"
            f"Shirts on screen: {shown or '(none)'}\n"
            f"Question the assistant just asked: {QUESTIONS[st['pending']][0] if st.get('pending') else '(none)'}")


def extract(st: dict, history: list[dict], message: str) -> dict:
    msgs = [{"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": f"{_context(st, history)}\n\nShopper's new message: {message}"}]
    return json.loads(models.chat(msgs, schema=EXTRACT_SCHEMA, max_tokens=200, temperature=0.0))


def _norm_colours(vals) -> list[str]:
    out = []
    for v in vals or []:
        v = str(v).lower().strip()
        v = re.sub(r"^(no|not|nothing|avoid)\s+", "", v)
        if v and v not in out and len(v) < 25:
            out.append(v)
    return out


# ------------------------------------------------------------------ code-side extraction
_NEG = r"(?:no|not|nothing|avoid|without|except|anything but|hate|never)"
_CUES = {  # a model value for these fields is kept only if the message has one of these words
    "sleeve": r"sleeve|short|long|either|any|arm",
    "pattern": r"plain|solid|stripe|check|plaid|gingham|tartan|print|floral|graphic|pattern",
    "type": r"shirt|tee\b|tees\b|polo",
    "usage": r"formal|office|work|interview|business|smart|meeting|casual|gym|sport|running|run\b|training|"
             r"workout|ethnic|traditional",
}


def code_extract(message: str) -> dict:
    """Deterministic reading of the words small models get wrong: sleeve, colours, pattern,
    type and usage. Anything found here overrides the model."""
    t = message.lower()
    out: dict = {}
    if re.search(r"short[- ]?sleeve|short sleeves|short ones", t):
        out["sleeve"] = "short"
    elif re.search(r"long[- ]?sleeve|long sleeves|long ones", t):
        out["sleeve"] = "long"
    elif re.search(r"\b(either|any|whatever)\s+sleeve|sleeves? (?:don'?t|doesn'?t|do not|does not) matter", t):
        out["sleeve"] = "any"
    names = sorted(COLOURS, key=len, reverse=True)
    col = r"\b(" + "|".join(map(re.escape, names)) + r")\b"
    avoid = re.findall(_NEG + r"\s+(?:in\s+|too\s+|more\s+)?" + col, t)
    for m in re.finditer(_NEG + r"\s+(?:in\s+)?" + col + r"((?:\s*(?:,|or|and)\s*" + col + r")+)", t):
        avoid += re.findall(col, m.group(2))
    avoid = list(dict.fromkeys(avoid))
    wanted = [c for c in dict.fromkeys(re.findall(col, t)) if c not in avoid
              and not any(c != a and c in a.split() for a in avoid)]
    wanted = [c for c in wanted if not any(c != w and c in w.split() for w in wanted)]  # "navy blue" not "blue"
    if avoid:
        out["avoid_colours"] = avoid
    if wanted:
        out["colours"] = wanted
    if re.search(r"\b(plain|solid)\b|no pattern", t):
        out["pattern"] = "solid"
    elif re.search(r"\b(striped|stripes|stripey|pinstripes?)\b", t):
        out["pattern"] = "striped"
    elif re.search(r"\b(checked|checks|checkered|plaid|gingham|tartan)\b|\bcheck (?:shirt|pattern)", t):
        out["pattern"] = "checked"
    elif re.search(r"\b(printed|prints?|floral|graphic)\b", t):
        out["pattern"] = "printed"
    tee = re.search(r"\b(t-?shirts?|t shirts?|tee shirts?|tees?|polos?|jerseys?)\b", t)
    shirt = re.search(r"(?<![-\w])(?<!t )(?<!tee )shirts?\b|\b(button[- ]?(?:down|up)|oxford|flannel)\b", t)
    if tee and not shirt:
        out["type"] = "tshirt"
    elif shirt and not tee:
        out["type"] = "shirt"  # "shirt" alone means a shirt, never both
    clear = []
    if re.search(r"\bany colou?rs?\b|colou?rs? (?:doesn'?t|don'?t|does not|do not) matter|whatever colou?r", t):
        clear += ["colours", "avoid_colours"]
    if re.search(r"\bany price\b|price (?:doesn'?t|does not) matter|\bno (?:budget|limit|price limit)\b", t):
        clear.append("max_price")
    if re.search(r"\bany pattern\b|pattern (?:doesn'?t|does not) matter", t):
        clear.append("pattern")
    if clear:
        out["clear"] = clear
        for k in clear:
            out.pop(k, None)
    formal = re.search(r"\b(formal|office|work|interview|business|meeting)\b", t)
    casual = re.search(r"\bcasual\b", t)
    sporty = re.search(r"\b(gym|running|sports?|training|workout)\b", t)
    if formal and casual:
        out["usage"] = None  # contradictory: no hard usage filter
    elif formal:
        out["usage"] = "formal"
    elif sporty:
        out["usage"] = "sports"
    elif casual:
        out["usage"] = "casual"
    return out


_FILLER = {"s", "am", "'s", "i", "im", "i'm", "its", "it's", "id", "i'd", "want", "wanna", "need", "looking", "look", "for", "a", "an", "the", "some",
           "something", "please", "me", "my", "show", "find", "get", "like", "would", "with", "and", "or", "in",
           "of", "to", "is", "it", "that", "one", "ones", "any", "anything", "just", "also", "too", "actually",
           "can", "you", "could", "hi", "hey", "hello", "thanks", "thank", "ok", "okay", "yes", "now", "then",
           "but", "maybe", "preferably", "ideally", "really", "very", "much", "more", "less", "than", "only",
           "colour", "color", "colours", "colors", "sleeve", "sleeves", "sleeved", "fit", "size", "budget",
           "dollars", "dollar", "bucks", "usd", "price", "cost", "max", "maximum", "under", "below", "over",
           "above", "around", "about", "up", "at", "most", "within", "cheaper", "spend", "pattern", "a", "be",
           "should", "is", "are", "do", "does", "don't", "doesnt", "doesn't", "matter", "fine", "either", "whatever",
           "nothing", "no", "not", "avoid", "without", "except", "never", "hate", "but", "this", "these"}
_COVER_RX = None


def code_covers(message: str) -> bool:
    """True when every content word is something code already reads (so no model call is needed)."""
    global _COVER_RX
    if _COVER_RX is None:
        names = sorted(COLOURS, key=len, reverse=True)
        parts = [r"\$?\d+(?:[.,]\d+)?\s*(?:cm|kg|kgs|lbs?|pounds?|m|in|inches|ft|feet|'|\")?", _MEN, _WOMEN, _KIDS,
                 r"\b(" + "|".join(map(re.escape, names)) + r")\b",
                 r"\b(plain|solid|striped|stripes|stripey|checked|checks|checkered|plaid|gingham|tartan|printed|"
                 r"prints?|floral|graphic|short|long|loose|looser|relaxed|roomy|baggy|oversized?|slim|slimmer|"
                 r"fitted|tight|snug|regular|t-?shirts?|t shirts?|tee shirts?|tees?|polos?|jerseys?|shirts?|button|"
                 r"buttoned|down|oxford|flannel|formal|office|work|interview|business|meeting|casual|gym|running|"
                 r"sports?|training|workout|tall|height|weight|chest|heavy|light)\b"]
        _COVER_RX = re.compile("|".join(parts))
    rest = _COVER_RX.sub(" ", message.lower())
    words = [w for w in re.findall(r"[a-z']+", rest) if w not in _FILLER]
    return not words


def code_title(wants: dict) -> str:
    bits = [{"solid": "plain"}.get(wants.get("pattern"), wants.get("pattern") or ""),
            (wants.get("colours") or [""])[0],
            "T-shirt" if wants.get("type") == "tshirt" else "shirt"]
    t = " ".join(b for b in bits if b)
    if wants.get("gender") in ("men", "women", "kids"):
        t = {"men": "Men's", "women": "Women's", "kids": "Kids'"}[wants["gender"]] + " " + t
    return t[:1].upper() + t[1:]


def guard_model(x: dict, message: str) -> dict:
    """Drop model values the message gives no basis for, then let code readings override."""
    x = dict(x)
    t = message.lower()
    for key, cue in _CUES.items():
        if x.get(key) and not re.search(cue, t):
            x[key] = None
    for key in ("colours", "avoid_colours"):
        if x.get(key):
            x[key] = [c for c in x[key] if str(c).strip()  # a real colour word that the message uses
                      and str(c).lower().split()[-1] in t
                      and (str(c).lower().strip() in COLOURS or str(c).lower().split()[-1] in COLOURS)]
    if x.get("max_price") is not None and not re.search(r"\d", t):
        x["max_price"] = None
    code = code_extract(message)
    if code.get("clear"):
        code = {**code, "clear": list(dict.fromkeys((x.get("clear") or []) + code["clear"]))}
    x.update(code)
    if "usage" in code and code["usage"] is None:
        x["usage"] = None
    return x


def apply_patch(st: dict, x: dict, message: str) -> tuple[bool, bool]:
    """Merge an extract result (model + code parsing) into state. Returns (wants_changed, profile_changed)."""
    if not x.get("_option"):
        x = guard_model(x, message)
    if x.get("occasion") and _GARMENT_ONLY.match(str(x["occasion"]).strip()):
        x = {**x, "occasion": None}
    wants, prof = st["wants"], st["profile"]
    before_w, before_p = json.dumps(wants, sort_keys=True), json.dumps(prof, sort_keys=True)

    for key in x.get("clear") or []:
        wants.pop(key, None)
        if key in SLOTS_TO_ASK and key not in st["answered"] and key != "gender":
            st["answered"].append(key)
    if re.search(r"\bformal\b", message, re.I) and re.search(r"\bcasual\b", message, re.I):
        x = {**x, "usage": None}  # "formal casual": contradictory, so no hard usage filter
    for key in ("occasion", "type", "sleeve", "pattern", "usage"):
        v = x.get(key)
        if isinstance(v, str) and v.strip():
            wants[key] = v.strip().lower() if key != "occasion" else v.strip()[:60]
    if x.get("max_price") is not None and isinstance(x["max_price"], (int, float)) and x["max_price"] > 0:
        wants["max_price"] = int(round(x["max_price"]))
    cols, avoid = _norm_colours(x.get("colours")), _norm_colours(x.get("avoid_colours"))
    if cols:
        wants["colours"] = cols
        wants["avoid_colours"] = [c for c in wants.get("avoid_colours", []) if c not in cols]
    if avoid:
        merged = list(dict.fromkeys(wants.get("avoid_colours", []) + avoid))
        wants["avoid_colours"] = merged
        if wants.get("colours"):
            wants["colours"] = [c for c in wants["colours"] if c not in avoid]
    for key in ("colours", "avoid_colours"):
        if key in wants and not wants[key]:
            del wants[key]
    if x.get("_answered") and x["_answered"] not in st["answered"]:
        st["answered"].append(x["_answered"])
    if x.get("_answered") == "max_price" and x.get("max_price") is None:
        wants.pop("max_price", None)

    # Profile: model's raw numbers converted in code, then code parsing of the text wins.
    units = convert_model_units(x)
    if x.get("fit"):
        units["fit"] = x["fit"]
    units.update(parse_units(message))
    if "max_price" in units:
        wants["max_price"] = units.pop("max_price")
    prof.update(_valid_profile_values(units))

    # Who it is for: the code's reading of the message wins; the model's is only trusted
    # when the message has a person cue (small models like to guess "men").
    g = detect_gender(message) if not x.get("_trusted_gender") else x.get("gender")
    if g is None and x.get("gender") and has_gender_cue(message):
        g = x["gender"]
    g = {"boys": "kids", "girls": "kids"}.get(g, g)
    if g in ("men", "women", "kids"):
        wants["gender"] = g
    if wants.get("gender") in ("men", "women"):
        prof["gender"] = wants["gender"]  # the size engine uses the matching body model
    elif wants.get("gender") == "kids":
        prof.pop("gender", None)

    return (json.dumps(wants, sort_keys=True) != before_w, json.dumps(prof, sort_keys=True) != before_p)


# ================================================================== decide / search
def _slot_known(st: dict, slot: str) -> bool:
    w = st["wants"]
    if slot in st["asked"] or slot in st["answered"]:
        return True
    if slot == "gender":
        return bool(w.get("gender"))
    if slot == "sleeve":
        return bool(w.get("sleeve"))
    if slot == "max_price":
        return w.get("max_price") is not None
    return bool(w.get("avoid_colours") or w.get("colours"))


def unasked_detail(st: dict) -> str | None:
    """First detail slot still unknown, ignoring the question cap (for 'I assumed ...')."""
    return next((s for s in DETAIL_SLOTS if not _slot_known(st, s)), None)


def missing_slot(st: dict) -> str | None:
    """Next clarifying question: who it is for first, then at most two details."""
    if not _slot_known(st, "gender"):
        return "gender"
    if sum(1 for s in st["asked"] if s in DETAIL_SLOTS) >= MAX_DETAIL_QUESTIONS:
        return None
    return unasked_detail(st)


GENDER_FILTER = {"men": ["men"], "women": ["women"], "kids": ["boys", "girls"]}


def filters_for(wants: dict) -> dict:
    # Gender is a hard filter that catalog.search never relaxes; unisex items match men and women.
    # Unknown: adults only (a boys' chart cannot fit an adult).
    f = {"gender": GENDER_FILTER.get(wants.get("gender"), ["men", "women", "unisex"])}
    for key in ("type", "pattern", "usage"):
        if wants.get(key):
            f[key] = wants[key]
    if wants.get("sleeve") in ("short", "long"):
        f["sleeve"] = wants["sleeve"]
    for key in ("colours", "avoid_colours"):
        if wants.get(key):
            f[key] = list(wants[key])
    if wants.get("max_price") is not None:
        f["max_price"] = wants["max_price"]
    return f


def filter_chips(wants: dict) -> list[dict]:
    chips = []
    if wants.get("gender") in ("men", "women", "kids"):
        chips.append({"key": "gender", "label": {"men": "Men's", "women": "Women's", "kids": "Kids'"}[wants["gender"]]})
    if wants.get("occasion"):
        occ = re.sub(r"^(a|an|the)\s+", "", wants["occasion"], flags=re.I)
        chips.append({"key": "occasion", "label": occ[:1].upper() + occ[1:]})
    if wants.get("type"):
        chips.append({"key": "type", "label": "T-shirts" if wants["type"] == "tshirt" else "Button shirts"})
    if wants.get("sleeve") in ("short", "long"):
        chips.append({"key": "sleeve", "label": f"{wants['sleeve'].capitalize()} sleeves"})
    if wants.get("pattern"):
        chips.append({"key": "pattern", "label": {"solid": "Plain"}.get(wants["pattern"], wants["pattern"].capitalize())})
    if wants.get("usage"):
        chips.append({"key": "usage", "label": wants["usage"].capitalize()})
    for c in wants.get("colours", []):
        chips.append({"key": f"colours:{c}", "label": c.capitalize()})
    for c in wants.get("avoid_colours", []):
        chips.append({"key": f"avoid_colours:{c}", "label": f"Not {c}"})
    if wants.get("max_price") is not None:
        chips.append({"key": "max_price", "label": f"Under ${wants['max_price']}"})
    return chips


def remove_chip(st: dict, key: str) -> bool:
    """Clear one criterion by chip key ('sleeve', 'avoid_colours:white', ...). True if something changed."""
    wants = st["wants"]
    slot, _, value = key.partition(":")
    if slot not in WANT_KEYS:
        return False
    if value and slot in ("colours", "avoid_colours"):
        if value not in wants.get(slot, []):
            return False
        wants[slot] = [c for c in wants[slot] if c != value]
        if not wants[slot]:
            del wants[slot]
    elif slot in wants:
        del wants[slot]
    else:
        return False
    if slot in SLOTS_TO_ASK and slot not in st["answered"]:
        st["answered"].append(slot)  # removing a criterion is an answer: never ask for it again
    if slot == "gender":
        st["profile"].pop("gender", None)
    return True


def chip_for_message(st: dict, message: str) -> str | None:
    """The UI removes a chip by sending `Remove the "<label>" filter.`; map that to the chip key."""
    m = re.match(r'^\s*remove the\s+["“](.+?)["”]\s+filter\.?\s*$', message, flags=re.I)
    if not m:
        return None
    label = m.group(1).strip().lower()
    chips = (st.get("results") or {}).get("filters") or filter_chips(st["wants"])
    for chip in chips:
        if chip["label"].lower() == label:
            return chip["key"]
    return ""  # a removal request whose label is not active


def search_query(wants: dict, message: str) -> str:
    parts = [wants.get("occasion") or ""]
    for key in ("usage", "pattern"):
        if wants.get(key):
            parts.append(wants[key])
    parts.extend(wants.get("colours", []))
    parts.append(message[:200])
    return " ".join(p for p in parts if p).strip()


def run_search(wants: dict, query: str) -> list[dict]:
    try:
        return catalog.search(filters_for(wants), query, k=8, embed=models.search_embed())
    except Exception as e:  # embedding provider down: rank by keywords instead
        log.warning("embedding search failed (%s); keyword ranking", type(e).__name__)
        return catalog.search(filters_for(wants), query, k=8, embed=None)


# ================================================================== pick
PICK_SCHEMA = {
    "title": "pick",
    "type": "object",
    "properties": {
        "picks": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "reason": {"type": "string"}},
            "required": ["id", "reason"], "additionalProperties": False}},
        "best_id": {"type": "integer"},
    },
    "required": ["picks", "best_id"],
    "additionalProperties": False,
}

PICK_SYSTEM = """You help a shopper choose shirts. From the candidates, choose the 3 that best suit what the shopper
asked for, and mark the single best one with best_id. Use only ids from the list.
For each pick write a reason: one short plain sentence, at most 8 words, saying why this shirt suits their
request (the occasion, the weather, the look they want), using only the item's listed fields.
Do not repeat the shirt's name, its colour or its sleeve length: the card already shows them.
No price, no sizes, no brand, no emoji. Do not invent facts (no "bestseller", no "breathable" or other fabric
properties; name the fabric only as listed). The three reasons must be different from each other.
Sentence case, ending with a full stop.
Examples of the register: "Light cotton, plain enough for a ceremony." "The most relaxed of the three."
"Darker, better for the evening." """

_SLIM = ("id", "name", "brand", "colour", "sleeve", "pattern", "fabric", "usage", "season", "price")


_USAGE_WORD = {"casual": "relaxed", "formal": "smart", "sports": "sporty", "ethnic": "traditional"}
_PATTERN_WORD = {"solid": "Plain", "striped": "Striped", "checked": "Checked", "printed": "Printed"}
_REASON_STOP = {"a", "an", "the", "and", "with", "for", "in", "of", "to", "is", "it", "this", "that", "very", "its",
                "on", "at", "your", "you", "great", "nice", "good", "perfect", "ideal", "choice", "option", "pick"}
_SLEEVE_WORDS = {"short", "long", "sleeve", "sleeves", "sleeved", "shirt", "shirts", "tee", "tshirt", "t"}


def code_reason(item: dict, occasion: str | None, variant: int = 0) -> str:
    """Deterministic reason from fabric + usage + the occasion (used when the model's is unusable)."""
    fabric = item["fabric"][:1].upper() + item["fabric"][1:]
    usage = _USAGE_WORD.get(item["usage"], "easy")
    occ = (occasion or "").strip()
    occ = occ[:1].lower() + occ[1:] if occ else ""
    options = [
        f"{fabric}, {usage} enough for {occ}." if occ else f"{fabric}, {usage} for everyday wear.",
        f"{_PATTERN_WORD.get(item['pattern'], 'Plain')} {item['fabric']} keeps it {usage}.",
        f"{fabric} for a {usage} look.",
    ]
    options = options[variant % 3:] + options[:variant % 3]
    return next((o for o in options if len(o) <= 70), options[-1])


def _restates(reason: str, item: dict) -> bool:
    """True when most of the reason's content words just repeat the name, colour or sleeve length."""
    words = [w for w in re.findall(r"[a-z]+", reason.lower()) if w not in _REASON_STOP]
    if not words:
        return True
    echo = set(re.findall(r"[a-z]+", f"{item['name']} {item['colour']} {item['brand']}".lower())) | _SLEEVE_WORDS
    return sum(w in echo for w in words) / len(words) >= 0.5


_PATTERN_TERMS = {"solid": {"solid", "plain"}, "striped": {"striped", "stripe", "stripes", "pinstripe"},
                  "checked": {"checked", "check", "checks", "plaid", "gingham", "tartan"},
                  "printed": {"printed", "print", "prints", "floral", "graphic"}}


def clean_reason(text: str, item: dict) -> str | None:
    """The model's reason tidied up, or None when it must be replaced."""
    t = re.sub(r"\s+", " ", str(text or "")).strip().strip('"').strip()
    if not t or len(t) > 70 or "$" in t or re.search(r"\d", t) or _restates(t, item):
        return None
    if re.search(r"\b(our|we|bestsell\w*|best-sell\w*|popular)\b", t, flags=re.I):
        return None
    words = {w: p for p, ws in _PATTERN_TERMS.items() for w in ws}
    said = {words[w] for w in re.findall(r"[a-z]+", t.lower()) if w in words}
    if item.get("pattern") and said and said != {item["pattern"]}:
        return None  # e.g. "a solid look" for a checked shirt
    if not t.endswith((".", "!", "?")):
        t += "."
    return t[0].upper() + t[1:]


class _SkipModel(Exception):
    pass


def pick(candidates: list[dict], wants: dict, request: str, exact_ids: set | None = None,
         use_model: bool = True) -> list[dict]:
    """Three items from `candidates` with reason + best. Unknown ids are dropped and backfilled.

    Exact matches (ids in `exact_ids`) always come before near matches, and the best pick
    is an exact match whenever one exists.
    """
    if not candidates:
        return []
    exact_ids = set(exact_ids if exact_ids is not None else (c["id"] for c in candidates))
    by_id = {c["id"]: c for c in candidates}
    model_picks: list[tuple[int, str]] = []
    best_id = None
    try:
        if not use_model:
            raise _SkipModel
        msgs = [{"role": "system", "content": PICK_SYSTEM},
                {"role": "user", "content": json.dumps({
                    "request": request[-400:], "occasion": wants.get("occasion"),
                    "criteria": {k: v for k, v in wants.items() if k != "occasion"},
                    "candidates": [{**{k: c[k] for k in _SLIM}, "fit": c["size_chart"]["fit"]} for c in candidates],
                })}]
        out = json.loads(models.chat(msgs, schema=PICK_SCHEMA, max_tokens=300, temperature=0.2))
        for p in out.get("picks") or []:
            try:
                pid = int(p.get("id"))
            except (TypeError, ValueError):
                continue
            if pid in by_id and pid not in [c for c, _ in model_picks]:
                model_picks.append((pid, p.get("reason") or ""))
        try:
            best_id = int(out.get("best_id"))
        except (TypeError, ValueError):
            best_id = None
    except _SkipModel:
        pass
    except Exception as e:
        log.warning("pick failed (%s); using ranking order", type(e).__name__)
    reasons = dict(model_picks)
    want = min(3, len(candidates))
    ranked = [c["id"] for c in candidates]
    # Exact first (the model chooses among them), then near matches strictly in the catalog's
    # order: fewest and mildest violations first, so the closest alternative is never skipped.
    order = ([i for i, _ in model_picks if i in exact_ids] + [i for i in ranked if i in exact_ids]
             + [i for i in ranked if i not in exact_ids])
    ids = list(dict.fromkeys(order))[:want]
    if not any(i in exact_ids for i in ids):
        best_id = ids[0]  # nothing matches exactly: the closest one is the pick
    elif best_id not in ids or best_id not in exact_ids:
        best_id = next(i for i in ids if i in exact_ids)
    ids.sort(key=lambda i: (i != best_id, i not in exact_ids))
    items, used = [], set()
    for n, iid in enumerate(ids):
        it = {k: v for k, v in by_id[iid].items() if k not in ("score",)}
        r = clean_reason(reasons.get(iid, ""), it)
        if r is None or r.lower() in used:
            r = next(c for v in range(n, n + 3)
                     if (c := code_reason(it, wants.get("occasion"), v)).lower() not in used or v == n + 2)
        used.add(r.lower())
        it["reason"] = r
        it["best"] = iid == best_id
        it["worn_open"] = catalog.worn_open(iid)
        items.append(it)
    return items


_PATTERN_LOWER = {"solid": "plain", "striped": "striped", "checked": "checked", "printed": "printed"}


def _one(v):
    return v[0] if isinstance(v, (list, tuple, set)) else v


_BROKE = {"price": "the budget", "colours": "the colour", "pattern": "the pattern", "sleeve": "the sleeve length",
          "usage": "the style"}


def near_phrase(item: dict, f: dict, violations: list | None = None) -> str:
    """How a near match differs from the user's criteria, from its real fields."""
    keys = [v["key"] for v in violations] if violations is not None else [
        v["key"] for v in (catalog.violations(item, f) or [])]
    sev = {v["key"]: v["severity"] for v in (violations or [])}
    parts = []
    for key in keys:
        if key == "pattern":
            parts.append(f"{_PATTERN_LOWER[item['pattern']]}, not {_PATTERN_LOWER.get(_one(f['pattern']), _one(f['pattern']))}")
        elif key == "usage":
            parts.append(f"{item['usage']}, not {_one(f['usage'])}")
        elif key == "colours":
            parts.append(f"{item['colour']}, not {' or '.join(f['colours'])}")
        elif key == "sleeve":
            parts.append(f"{item['sleeve']} sleeves, not {_one(f['sleeve'])}")
        elif key == "price":
            small = sev.get("price", 1) == 1 if sev else item["price"] <= f["max_price"] * 1.35
            parts.append(f"{'a little ' if small else ''}over budget at ${item['price']}")
    text = "; ".join(parts) or "close to what you asked for"
    return text[:1].upper() + text[1:]


_NUM = {1: "One", 2: "Two", 3: "Three"}
_RELAX_WORDS = {"pattern": "the pattern", "usage": "the style", "colours": "the colour", "price": "the budget a little"}


def match_sentence(items: list[dict], relaxed: list[str]) -> str:
    """The code-written sentence about exact vs near matches ('' when all shown items match)."""
    exact = [i for i in items if i["exact"]]
    near = [i for i in items if not i["exact"]]
    if not items or not near:
        return ""
    lc = [n["near"][:1].lower() + n["near"][1:] for n in near]
    if not exact:
        s = f"Nothing matches everything you asked for. The closest is {lc[0]}."
        rest = lc[1:]
        if len(rest) == 2 and rest[0] == rest[1]:
            s += f" The other two are {rest[0]}."
        elif len(rest) == 2:
            s += f" The others are {rest[0]}, and {rest[1]}."
        elif rest:
            s += f" The other one is {rest[0]}."
        return s
    s = (f"{_NUM[len(exact)]} shirt{'s' if len(exact) > 1 else ''} "
         f"match{'' if len(exact) > 1 else 'es'} everything you asked for. ")
    if len(near) == 1:
        return s + f"I added one close one: it is {lc[0]}."
    if lc[0] == lc[1]:
        return s + f"I added two close ones: both are {lc[0]}."
    return s + f"I added two close ones: one is {lc[0]}, the other {lc[1]}."


def results_title(wants: dict, n: int, items: list[dict] | None = None) -> str:
    """Built in code from the shown items' real type and the occasion; never from model text."""
    if n == 0:
        return "Nothing matches yet"
    word = {1: "One", 2: "Two", 3: "Three"}.get(n, str(n))
    types = {i["type"] for i in (items or [])} or ({wants["type"]} if wants.get("type") else {"shirt"})
    noun = "T-shirts" if types == {"tshirt"} else "shirts"
    if n == 1:
        noun = noun[:-1]
    occ = (wants.get("occasion") or "").strip()
    if occ and not _GARMENT_ONLY.match(occ):
        if not re.match(r"^(a|an|the|my|your|work|office|summer|winter|everyday|weekend)\b", occ, flags=re.I):
            occ = ("an " if occ[:1].lower() in "aeiou" else "a ") + occ
        return f"{word} {noun} for {occ[:1].lower() + occ[1:]}"
    if wants.get("sleeve") in ("short", "long"):
        return f"{word} {wants['sleeve']}-sleeved {noun}"
    return f"{word} {noun} for you"


# ================================================================== size
def has_body(profile: dict) -> bool:
    return bool((profile.get("height_cm") and profile.get("weight_kg")) or profile.get("chest_cm"))


def size_for(item: dict, profile: dict) -> dict | None:
    if not has_body(profile):
        return None
    p = {k: profile[k] for k in ("height_cm", "weight_kg", "chest_cm", "gender") if profile.get(k) is not None}
    p["fit"] = profile.get("fit") or "regular"
    if p.get("gender") not in ("men", "women", "boys", "girls", "unisex"):
        p.pop("gender", None)
    if not (p.get("height_cm") and p.get("weight_kg")):
        p.pop("height_cm", None)
        p.pop("weight_kg", None)
    res = sizing.recommend_size(item, p)
    chart = next(s for s in item["size_chart"]["sizes"] if s["size"] == res["chart_size"])
    res["chart_room_cm"] = round(chart["garment_chest"] - res["chest_cm"], 1)
    return {"item_id": item["id"], **res}


def resolve_reference(ref: str | None, results: dict | None) -> int | None:
    if not ref or not results or not results.get("items"):
        return None
    items = results["items"]
    r = ref.lower()
    ordinals = [("first", 0), ("1st", 0), ("second", 1), ("2nd", 1), ("third", 2), ("3rd", 2),
                ("middle", 1), ("last", len(items) - 1), ("best", next((n for n, i in enumerate(items) if i["best"]), 0))]
    for word, idx in ordinals:
        if re.search(rf"\b{word}\b", r) and idx < len(items):
            return items[idx]["id"]
    m = re.search(r"\b(?:number|no\.?|#)\s*([1-9])\b", r)
    if m and int(m.group(1)) <= len(items):
        return items[int(m.group(1)) - 1]["id"]
    words = set(re.findall(r"[a-z]+", r)) - {"the", "one", "a", "that", "this", "shirt", "tee", "i", "like"}
    best, best_score = None, 0
    for it in items:
        hay = set(re.findall(r"[a-z]+", f"{it['name']} {it['brand']} {it['colour']} {it['pattern']}".lower()))
        score = len(words & hay)
        if score > best_score:
            best, best_score = it["id"], score
    return best


# ================================================================== reply
REPLY_SYSTEM = """You are the voice of Fitting Room, a shirt shop assistant. Write the reply to the shopper.
Rules: 1 to 3 short sentences. Plain, warm, direct. No lists, no markdown, no emoji.
Speak in the first person singular ("I'd pick", "I found", "I widened"). Never say "we", "us" or "our".
No praise the facts do not support: never "quality", "perfect", "great choice", "stylish", "premium", "elegant".
If FACTS.already_said is given, it is shown just before your words: continue from it without repeating it.
Only mention shirts listed in FACTS, by their name or colour; never invent a shirt, a brand, a price or a
measurement. Any number you write must appear in FACTS. The screen already shows prices, sizes and
measurements, so mention at most one price.
Do exactly what FACTS.todo says."""


def reply_messages(history: list[dict], message: str, facts: dict) -> list[dict]:
    msgs = [{"role": "system", "content": REPLY_SYSTEM}]
    for m in history[-4:]:
        msgs.append({"role": m["role"], "content": m["text"][:400]})
    msgs.append({"role": "user", "content": message[:MAX_MESSAGE]})
    msgs.append({"role": "system", "content": "FACTS: " + json.dumps(facts)})
    return msgs


SIZE_WORD = {"XS": "extra small", "S": "small", "M": "medium", "L": "large", "XL": "extra large", "XXL": "extra extra large"}


def fallback_reply(facts: dict) -> str:
    parts = []
    if facts.get("results"):
        best = next((r for r in facts["results"] if r.get("best")), facts["results"][0])
        parts.append(f"I'd start with the {best['name'].lower()}.")
    if facts.get("size"):
        parts.append(f"You're a {SIZE_WORD.get(facts['size']['size'], facts['size']['size'])} in it.")
    if "ask_height_weight" in facts.get("todo", ""):
        parts.append("Tell me your height and weight and I'll work out your size.")
    return " ".join(parts) or "Tell me what you need a shirt for and I'll find a few."


# ================================================================== the turn
_REPLACE = [(r"\bperfectly\b", "well"), (r"\bperfect\b", "good"), (r"\bgreat choice\b", "good choice"),
            (r"\bstylish\b", "smart"), (r"\belegant\b", "smart"), (r"\bpremium\s+", ""),
            (r"\bhigh[- ]quality\s+", ""), (r"\bquality\s+", ""),
            (r"\bWe\b", "I"), (r"\bwe\b", "I"), (r"\bOur\b", "My"), (r"\bour\b", "my"), (r"\bus\b", "me")]


def clean_words(text: str) -> str:
    for rx, sub in _REPLACE:
        text = re.sub(rx, sub, text)
    return text


def filtered_stream(stream) -> Iterator[str]:
    """Pass the model's reply through clean_words() while still streaming: the last two words are
    held back so a word (or a two-word phrase) is cleaned whole."""
    buf = ""
    for delta in stream:
        buf += delta
        parts = buf.split(" ")
        if len(parts) > 3:
            release = " ".join(parts[:-2]) + " "
            buf = " ".join(parts[-2:])
            yield clean_words(release)
    if buf:
        yield clean_words(buf)


def _emit_text(text: str) -> Iterator[tuple[str, dict]]:
    for chunk in re.findall(r"\S+\s*|\s+", text):
        yield "text", {"delta": chunk}


_GARMENT_ONLY = re.compile(r"^(a |an |the |some )?(shirt|shirts|t-?shirts?|tees?|polos?|tops?|something|one|"
                           r"button[- ]?(?:up|down)? shirt)$", re.I)


_DESCRIBE = {
    "gender": lambda v: {"men": "men's", "women": "women's", "kids": "kids'"}.get(v, v),
    "type": lambda v: "T-shirt" if v == "tshirt" else "button shirt",
    "sleeve": lambda v: {"short": "short sleeves", "long": "long sleeves", "any": "any sleeve length"}.get(v, v),
    "pattern": lambda v: _PATTERN_LOWER.get(v, v),
    "colours": lambda v: " or ".join(v),
    "avoid_colours": lambda v: "not " + " or ".join(v),
    "max_price": lambda v: f"under ${v}",
    "usage": lambda v: v,
    "occasion": lambda v: v,
}


def describe_changes(before: dict, st: dict) -> str:
    """What this turn added or changed, for the 'understand' step."""
    parts = []
    w0, w1 = before["wants"], st["wants"]
    for k in ("gender", "occasion", "colours", "avoid_colours", "pattern", "type", "sleeve", "usage", "max_price"):
        if w1.get(k) is not None and w1.get(k) != w0.get(k):
            parts.append(_DESCRIBE[k](w1[k]))
        elif w0.get(k) is not None and w1.get(k) is None:
            parts.append(f"dropped {_DESCRIBE[k](w0[k])}")
    p0, p1 = before["profile"], st["profile"]
    for k, fmt in (("height_cm", "height {:g} cm"), ("weight_kg", "weight {:g} kg"), ("chest_cm", "chest {:g} cm")):
        if p1.get(k) is not None and p1.get(k) != p0.get(k):
            parts.append(fmt.format(p1[k]))
    if p1.get("fit") and p1.get("fit") != p0.get("fit"):
        parts.append(f"{p1['fit']} fit")
    return ", ".join(parts)


_NEED = {"gender": "who it's for", "sleeve": "sleeve length", "max_price": "budget", "avoid_colours": "colours to avoid"}


class _Steps:
    """Times the real pipeline stages and yields `step` events; keeps the finished ones for the transcript."""

    def __init__(self):
        self.done: list[dict] = []
        self._t: dict[str, float] = {}
        self._label: dict[str, str] = {}

    def start(self, sid: str, label: str) -> tuple[str, dict]:
        self._t[sid], self._label[sid] = time.perf_counter(), label
        return "step", {"id": sid, "status": "start", "label": label, "detail": None}

    def finish(self, sid: str, detail: str | None) -> tuple[str, dict]:
        ms = int(round((time.perf_counter() - self._t[sid]) * 1000))
        rec = {"id": sid, "label": self._label[sid], "detail": detail or None, "ms": ms}
        self.done.append(rec)
        return "step", {**rec, "status": "done"}


def _count(n: int, one: str, many: str) -> str:
    return f"{'None' if n == 0 else n} {one if n == 1 else many}"


def run_turn(conversation_id: str | None, message: str, client: str,
             remove_filter: str | None = None) -> Iterator[tuple[str, dict]]:
    message = (message or "").strip()[:MAX_MESSAGE]
    conv = store.get_conversation(conversation_id, client) if conversation_id else None
    is_new = conv is None
    st = conv["state"] if conv else new_state()
    for k, v in new_state().items():
        st.setdefault(k, v)
    history = store.messages(conv["id"]) if conv else []
    conv_id = conv["id"] if conv else uuid.uuid4().hex[:16]
    title = conv["title"] if conv else (" ".join(message.split()[:5])[:60] or "New search")
    steps = _Steps()

    if is_new:
        store.create_conversation(conv_id, client, title, st)
    yield "conversation", {"id": conv_id, "title": title}
    if message:
        store.add_message(conv_id, "user", message)

    # ---- UNDERSTAND
    before = {"wants": json.loads(json.dumps(st["wants"])), "profile": dict(st["profile"])}
    yield steps.start("understand", "Reading your request")
    chip = remove_filter if remove_filter is not None else (chip_for_message(st, message) if not is_new else None)
    x: dict | None = None
    how = ""
    if chip is None:
        opt = match_option(st, message)
        if opt is not None:
            x = {"intent": "refine", **opt, "_option": True}
        else:
            try:
                if code_covers(message) and not st.get("results"):
                    # Every word is something code reads: no model call (fast, and nothing invented).
                    x = {"intent": "search" if code_extract(message) or detect_gender(message) else "other"}
                    how = " (read by rules, no model call)"
                elif code_covers(message):
                    x = {"intent": "refine"}
                    how = " (read by rules, no model call)"
                else:
                    x = extract(st, history, message)
            except Exception as e:
                log.warning("extract failed (%s); code parsing only", type(e).__name__)
                x, how = keyword_wants(message), " (keyword rules; the model did not answer)"
    st["pending"], pending_before = None, st.get("pending")
    selection_changed = False
    removed_label = ""
    if chip is not None:
        removed_label = next((c["label"] for c in (st.get("results") or {}).get("filters", []) if c["key"] == chip),
                             chip)
        changed = bool(chip) and remove_chip(st, chip)
        wants_changed, profile_changed, intent = changed, False, "refine"
        detail = f"Remove the \u201c{removed_label}\u201d filter" if changed else "That filter is not active"
    else:
        if pending_before and pending_before not in st["answered"]:
            st["answered"].append(pending_before)  # whatever they said, don't ask that again
        wants_changed, profile_changed = apply_patch(st, x, message)
        intent = x.get("intent") or "other"
        if st.get("results") and x.get("reference"):
            sel = resolve_reference(x["reference"], st["results"])
            if sel is not None and sel != st.get("selected"):
                st["selected"], selection_changed = sel, True
        detail = describe_changes(before, st)
        if selection_changed:
            name = next(i["name"] for i in st["results"]["items"] if i["id"] == st["selected"])
            detail = ", ".join(filter(None, [detail, f"the {name.lower()}"]))
        detail = (detail or "Nothing new to note") + how
    yield steps.finish("understand", detail[:1].upper() + detail[1:])
    if is_new and x is not None and not (x.get("title") or "").strip() and any(st["wants"].values()):
        x = {**x, "title": code_title(st["wants"])}
    if is_new and x and (x.get("title") or "").strip():
        t2 = x["title"].strip()[:60]
        t2 = t2[:1].upper() + t2[1:]
        if t2 != title:
            title = t2
            yield "conversation", {"id": conv_id, "title": title}

    # ---- DECIDE
    yield steps.start("decide", "Checking what I still need")
    has_results = bool(st.get("results"))
    has_wants = any(st["wants"].get(k) for k in WANT_KEYS)
    want_search = wants_changed or (not has_results and (
        has_wants or intent in ("search", "refine") or bool(pending_before)))
    slot = None
    if want_search and not has_results and chip is None and st["questions"] < MAX_QUESTIONS:
        slot = missing_slot(st)
    if slot:
        yield steps.finish("decide", f"Need: {_NEED[slot]}")
        text, options = QUESTIONS[slot]
        occ = st["wants"].get("occasion")
        if occ and slot == "sleeve":
            text = f"For {occ[:1].lower() + occ[1:]}, short sleeves or long?"
        st["questions"] += 1
        st["asked"].append(slot)
        st["pending"] = slot
        store.add_message(conv_id, "assistant", text, options, steps.done)
        store.save_conversation(conv_id, st, title if is_new else None)
        yield "question", {"text": text, "options": options}
        yield "state", public_state(st)
        yield "done", {}
        return
    yield steps.finish("decide", "Enough to search" if want_search else "No new search needed")

    events: list[tuple[str, dict]] = []
    facts: dict = {"todo": ""}
    lead = ""  # sentence written by code, sent before the model's words
    new_results = False
    if want_search:
        # ---- SEARCH: hard filters + every soft criterion (real count), then rank
        f = {k: v for k, v in filters_for(st["wants"]).items() if v not in (None, "", [])}
        all_items = catalog.load()
        yield steps.start("search", f"Searching {len(all_items)} shirts")
        hard_ok = [i for i in all_items if catalog._passes(i, catalog._hard(f))]
        viol = {i["id"]: catalog.violations(i, f) for i in hard_ok}
        exact_total = sum(1 for v in viol.values() if v == [])
        query = search_query(st["wants"], message)
        if exact_total >= catalog.MIN_RESULTS:
            candidates = run_search(st["wants"], query)
        yield steps.finish("search", _count(exact_total, "matches every filter", "match every filter"))
        if exact_total < catalog.MIN_RESULTS:
            # ---- WIDEN: near matches ranked by how many criteria they break, then how badly
            yield steps.start("widen", "Too few, widening the search")
            candidates = run_search(st["wants"], query)
            n_near = sum(1 for v in viol.values() if v)
            first_near = next((c for c in candidates if not c["exact"]), None)
            detail = f"{n_near} near match{'' if n_near == 1 else 'es'}"
            if first_near:
                broke = " and ".join(_BROKE[v["key"]] for v in first_near["violations"])
                detail += f"; the closest breaks only {broke}" if len(first_near["violations"]) == 1 \
                    else f"; the closest breaks {broke}"
            yield steps.finish("widen", detail)
        exact_ids = {c["id"] for c in candidates if c.get("exact", True)}
        relaxed = list(dict.fromkeys(k for c in candidates for k in c.get("relaxed", [])))

        # ---- PICK
        items = []
        if candidates:
            yield steps.start("pick", f"Comparing the top {len(candidates)}")
            request = " / ".join([m["text"] for m in history if m["role"] == "user"][-3:] + [message])
            # With 3 or fewer exact matches the shown set is fixed by the ranking; locally, skip the
            # model and write the reasons in code (saves several seconds on a laptop).
            use_model = models.provider() == "openrouter" or len(exact_ids) > 3
            items = pick(candidates, st["wants"], request, exact_ids, use_model=use_model)
            by_id = {c["id"]: c for c in candidates}
            for it in items:
                it["exact"] = it["id"] in exact_ids
                it.pop("violations", None)
                if not it["exact"]:
                    it["near"] = near_phrase(it, f, by_id[it["id"]].get("violations"))
            n_exact = sum(i["exact"] for i in items)
            yield steps.finish("pick", f"Chose {len(items)}" + (f", {n_exact} exact" if n_exact < len(items) else ""))
        n_exact = sum(i["exact"] for i in items)
        st["results"] = {"title": results_title(st["wants"], len(items), items), "filters": filter_chips(st["wants"]),
                         "items": items, "exact_count": n_exact}
        st["selected"] = next((i["id"] for i in items if i["best"]), items[0]["id"] if items else None)
        new_results = True
        events.append(("results", st["results"]))
        facts["results"] = [{"name": i["name"], "colour": i["colour"], "reason": i["reason"], "best": i["best"],
                             "exact_match": i["exact"]} for i in items]
        lead = match_sentence(items, relaxed)
        if not items:
            facts["todo"] += ("Say nothing in the catalog comes close to all of that, and suggest raising the "
                              "budget or dropping one filter (name one from criteria). ")
            facts["criteria"] = [c["label"] for c in st["results"]["filters"]]
        else:
            facts["todo"] += "Say which one you'd pick and why, using its reason, in one sentence. "
            if lead:
                facts["already_said"] = lead
                facts["todo"] += "Do not talk about matching, widening or relaxing: that is already said. "
        if chip is None and not has_results and unasked_detail(st) and items:
            facts["todo"] += "Mention in a few words that you assumed no limit on " + {
                "sleeve": "sleeve length", "max_price": "price", "avoid_colours": "colour"}[unasked_detail(st)] + ". "
        if chip is not None and removed_label:
            facts["todo"] = f"Say you removed the '{removed_label}' filter. " + facts["todo"]

    # ---- SIZE
    if st.get("results") and st["results"]["items"] and has_body(st["profile"]) and (
            new_results or profile_changed or selection_changed or st.get("size") is None):
        sel = st.get("selected")
        item = catalog.get(sel) if sel is not None else None
        if item is not None:
            yield steps.start("size", "Working out your size")
            size_event = None
            try:
                size_event = size_for(item, st["profile"])
            except ValueError as e:
                log.info("size not possible: %s", e)
            if size_event:
                fit = st["profile"].get("fit") or "regular"
                chest = (f"Chest about {size_event['chest_cm']:.0f} cm" if size_event["chest_estimated"]
                         else f"Chest {size_event['chest_cm']:g} cm")
                sd = f"{chest}, {size_event['chart_size']} on the chart"
                if size_event["moved"]:
                    sd += f", {size_event['size']} for a {fit} fit"
                yield steps.finish("size", sd)
                st["size"] = size_event
                events.append(("size", size_event))
                facts["size"] = {"shirt": item["name"], "size": size_event["size"],
                                 "chart_size": size_event["chart_size"], "moved": size_event["moved"], "fit": fit}
                facts["todo"] += ("Tell them their size letter for that shirt in one sentence"
                                  + (" and that it is one up from the chart for the looser fit"
                                     if size_event["moved"] > 0 else "")
                                  + (" and that it is one down for the slimmer fit" if size_event["moved"] < 0 else "")
                                  + ". ")
            else:
                yield steps.finish("size", "Those numbers are outside the size chart")

    if st.get("results") and st["results"]["items"] and not has_body(st["profile"]) and not st["asked_profile"] \
            and new_results:
        st["asked_profile"] = True
        facts["todo"] += "End by asking for their height and weight so you can size it (ask_height_weight). "

    if not events and not facts["todo"]:
        if chip is not None:
            facts["todo"] = "Say that filter is not active, and they can tap another one to remove it."
        elif not st.get("results"):
            facts["todo"] = "Ask what they need a shirt for (occasion, weather or colour), in one sentence."
        elif profile_changed and not has_body(st["profile"]):
            facts["todo"] = "Say you noted it and ask for the missing height or weight."
        else:
            facts["todo"] = ("Answer briefly using only the shown shirts. You can change filters, "
                             "pick another shirt or size one if they give height and weight.")
    if st.get("results") and "results" not in facts:
        facts["shown"] = [{"name": i["name"], "colour": i["colour"], "reason": i["reason"], "best": i["best"]}
                          for i in st["results"]["items"]]

    for ev in events:
        if ev[0] == "results":
            yield ev

    # ---- WRITE
    yield steps.start("write", "Writing the reply")
    text, first = "", True

    def sent():
        nonlocal first
        if first:
            first = False
            return [steps.finish("write", None)]
        return []

    if lead:
        text = lead + " "
        yield "text", {"delta": text}
        yield from sent()
    try:
        # closing(): the local provider holds a lock until its stream is exhausted or closed.
        with closing(models.chat_stream(reply_messages(history, message, facts), max_tokens=160,
                                        temperature=0.4)) as stream:
            for delta in filtered_stream(stream):
                text += delta
                yield "text", {"delta": delta}
                yield from sent()
    except Exception as e:
        log.warning("reply stream failed (%s)", type(e).__name__)
        if not text.strip():
            text = fallback_reply(facts)
            yield from _emit_text(text)
    yield from sent()
    for ev in events:
        if ev[0] != "results":
            yield ev
    store.add_message(conv_id, "assistant", text.strip() or fallback_reply(facts), None, steps.done)
    store.save_conversation(conv_id, st, title)
    yield "state", public_state(st)
    yield "done", {}


def conversation_view(conv_id: str, client: str) -> dict | None:
    conv = store.get_conversation(conv_id, client)
    if conv is None:
        return None
    st = conv["state"]
    return {"id": conv["id"], "title": conv["title"], "messages": store.messages(conv_id),
            "results": st.get("results"), "size": st.get("size"), "state": public_state({**new_state(), **st})}


def set_profile(conv_id: str, client: str, profile: dict, item_id: int, size: dict | None) -> None:
    """POST /api/size with a conversation_id: remember fit / chest / height / weight for later turns."""
    conv = store.get_conversation(conv_id, client)
    if conv is None:
        return
    st = conv["state"]
    prof = st.setdefault("profile", {})
    prof.update(_valid_profile_values(profile))
    if "chest_cm" in profile and profile["chest_cm"] in (None, ""):
        prof.pop("chest_cm", None)
    if size is not None:
        st["size"] = {"item_id": item_id, **size}
        st["selected"] = item_id
    store.save_conversation(conv_id, st)
