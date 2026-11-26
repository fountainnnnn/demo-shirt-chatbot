"""Enrich data/catalog/raw.json into data/catalog/catalog.json (contract format).

Real fields (from the dataset): brand, gender, type, colour, usage, season, image.
Image-derived (zero-shot CLIP on the photo, product name used as a prior):
    sleeve, pattern.
Synthetic (seeded by id, listed in item["synthetic"]): fabric, price, size_chart.

CLIP predictions are cached in data/catalog/clip_predictions.json so the build
is repeatable without the model. Re-run CLIP with --rerun-clip. Hand fixes live
in data/catalog/overrides.json ({id: {field: value}}) and are applied last.

    HF_HOME=$PWD/models/hf .venv/bin/python scripts/enrich_catalog.py [--rerun-clip]
    .venv/bin/python scripts/enrich_catalog.py --sheets      # contact sheets for review
"""
import json
import os
import pathlib
import random
import re
import sys
import zlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CAT = ROOT / "data" / "catalog"
RAW = CAT / "raw.json"
OUT = CAT / "catalog.json"
OVERRIDES = CAT / "overrides.json"
CLIP_CACHE = CAT / "clip_predictions.json"
SHEETS = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "catalog_sheets"
LOCAL_CLIP = ROOT / "models" / "clip-b32"

# Longest match wins. value = display brand (None -> keep the matched text).
BRANDS = {
    "United Colors of Benetton": "United Colors of Benetton",
    "UCB": "United Colors of Benetton",
    "U.S. Polo Assn.": "U.S. Polo Assn.",
    "John Players": "John Players", "John Miller": "John Miller",
    "Indigo Nation": "Indigo Nation", "Reid & Taylor": "Reid & Taylor",
    "Gini and Jony": "Gini and Jony", "Peter England": "Peter England",
    "Flying Machine": "Flying Machine", "Classic Polo": "Classic Polo",
    "Manchester United": "Manchester United", "Ed Hardy": "Ed Hardy",
    "French Connection": "French Connection", "Indian Terrain": "Indian Terrain",
    "Mark Taylor": "Mark Taylor", "Jealous 21": "Jealous 21",
    "Urban Yoga": "Urban Yoga", "Little Miss": "Little Miss",
    "Palm Tree": "Palm Tree", "Scullers For Her": "Scullers",
    "Arrow Sport": "Arrow Sport", "Arrow": "Arrow", "s.Oliver": "s.Oliver",
    "Turtle": "Turtle", "Scullers": "Scullers", "Doodle": "Doodle",
    "Fabindia": "Fabindia", "Locomotive": "Locomotive", "Spykar": "Spykar",
    "Levis": "Levi's", "Lee": "Lee", "Wrangler": "Wrangler", "Puma": "Puma",
    "Fila": "Fila", "Nike": "Nike", "ADIDAS": "Adidas", "Myntra": "Myntra",
    "Tantra": "Tantra", "Inkfruit": "Inkfruit", "Superman": "Superman",
    "Basics": "Basics", "Music": "Music",
}
PREMIUM = {"United Colors of Benetton", "Arrow", "French Connection", "Reid & Taylor",
           "Fabindia", "Levi's", "Lee", "s.Oliver", "Indian Terrain", "Ed Hardy", "Nike", "Adidas"}

KID_WORDS = re.compile(r"\b(boy'?s?|girl'?s?|kids?|kidswear|infant)\b", re.I)
GENDER_WORDS = re.compile(r"\b(men'?s?|women'?s?|woman|boy'?s?|girl'?s?|kids?|kidswear|infant|for her)\b", re.I)


def parse_brand(display: str) -> tuple[str, str]:
    """-> (brand, remainder of the product name)."""
    s = re.sub(r"\s+", " ", display).strip()
    low = s.lower()
    for key in sorted(BRANDS, key=len, reverse=True):
        if low.startswith(key.lower() + " ") or low == key.lower():
            return BRANDS[key], s[len(key):].strip()
    first, _, rest = s.partition(" ")
    return first, rest


def parse_gender(raw_gender: str, display: str) -> str:
    low = display.lower()
    if re.search(r"\bgirl", low):
        return "girls"
    if KID_WORDS.search(display) or raw_gender.lower() in ("boys", "girls"):
        return "girls" if raw_gender.lower() == "girls" else "boys"
    return {"men": "men", "women": "women"}.get(raw_gender.lower(), "unisex")


def name_prior(display: str) -> dict:
    """Explicit cues in the product name."""
    low = display.lower()
    out = {}
    if re.search(r"\bcheck(s|ed)?\b", low):
        out["pattern"] = "checked"
    elif re.search(r"\bstripe[sd]?\b", low):
        out["pattern"] = "striped"
    elif re.search(r"\bprinted\b|\bgraphic\b|\blogo\b", low):
        out["pattern"] = "printed"
    elif re.search(r"\bsolid\b", low):
        out["pattern"] = "solid"
    return out


# ----------------------------------------------------------------- CLIP
SLEEVE_PROMPTS = {
    "short": ["a photo of a short-sleeved shirt", "a short sleeve top with bare forearms, sleeves ending above the elbow"],
    "long": ["a photo of a long-sleeved shirt", "a long sleeve shirt with sleeves reaching the wrists"],
}
PATTERN_PROMPTS = {
    "solid": ["a plain solid colour shirt with no pattern"],
    "striped": ["a striped shirt with vertical or horizontal stripes"],
    "checked": ["a checked plaid shirt with a check pattern"],
    "printed": ["a shirt with a printed graphic, logo or text on it"],
}


def run_clip(items: list[dict]) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))
    src = str(LOCAL_CLIP) if LOCAL_CLIP.exists() else "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(src).eval()
    proc = CLIPProcessor.from_pretrained(src)

    def text_feats(prompts: dict):
        labels, feats = list(prompts), []
        for lab in labels:
            inp = proc(text=prompts[lab], return_tensors="pt", padding=True)
            with torch.no_grad():
                f = model.get_text_features(**inp)
            f = f if isinstance(f, torch.Tensor) else f.pooler_output
            f = f / f.norm(dim=-1, keepdim=True)
            m = f.mean(0)
            feats.append(m / m.norm())
        return labels, torch.stack(feats)

    groups = {"sleeve": text_feats(SLEEVE_PROMPTS), "pattern": text_feats(PATTERN_PROMPTS)}
    preds = {}
    for it in items:
        img = Image.open(CAT / it["image"]).convert("RGB")
        inp = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            f = model.get_image_features(**inp)
        f = f if isinstance(f, torch.Tensor) else f.pooler_output
        f = f / f.norm(dim=-1, keepdim=True)
        rec = {}
        for g, (labels, tf) in groups.items():
            p = (100 * f @ tf.T).softmax(-1)[0].tolist()
            rec[g] = {l: round(v, 4) for l, v in zip(labels, p)}
        preds[str(it["id"])] = rec
    CLIP_CACHE.write_text(json.dumps(preds, indent=1))
    return preds


# ------------------------------------------------------------ synthetic
def rng_for(item_id: int, salt: str = "") -> random.Random:
    return random.Random(f"{item_id}:{salt}")


def pick_fabric(a: dict, rng: random.Random) -> tuple[str, str]:
    """-> (fabric, short cue used in the item name)."""
    low = a["source_name"].lower()
    if a["type"] == "tshirt":
        if "jersey" in low:
            return "polyester jersey", "jersey"
        if a["usage"] == "sports":
            return rng.choice([("polyester mesh", "mesh"), ("dry-fit polyester", "dry-fit")])
        if "polo" in low:
            return "cotton pique", "pique"
        return rng.choice([("cotton jersey", "cotton"), ("combed cotton", "cotton"), ("soft cotton", "cotton")])
    if "satin" in low:
        return "cotton satin", "satin"
    if a["usage"] == "ethnic":
        return "handloom cotton", "handloom"
    if a["usage"] == "formal":
        return rng.choice([("cotton poplin", "poplin"), ("fine cotton twill", "twill")])
    if a["pattern"] == "checked":
        return rng.choice([("brushed cotton", "brushed cotton"), ("cotton flannel", "flannel"), ("cotton twill", "twill")])
    if a["pattern"] == "striped":
        return rng.choice([("cotton poplin", "poplin"), ("oxford cotton", "oxford")])
    return rng.choice([("light cotton", "cotton"), ("oxford cotton", "oxford"), ("linen blend", "linen-blend"),
                       ("cotton poplin", "poplin"), ("cotton chambray", "chambray")])


def pick_price(a: dict, rng: random.Random) -> int:
    if a["type"] == "tshirt":
        lo, hi = (22, 32) if "polo" in a["source_name"].lower() else (14, 28)
    elif a["usage"] == "formal":
        lo, hi = 52, 75
    else:
        lo, hi = 28, 58
    price = rng.randint(lo, hi)
    if a["brand"] in PREMIUM:
        price = min(price + 5, 75 if a["type"] == "shirt" else 32)
    if a["gender"] in ("boys", "girls"):
        price = round(price * 0.75)
    return int(price)


def pick_fit(a: dict, rng: random.Random) -> str:
    low = a["source_name"].lower()
    if "slim fit" in low:
        return "slim"
    if a["gender"] in ("boys", "girls") or a["type"] == "tshirt":
        return "regular"
    if a["usage"] == "formal":
        return "slim" if rng.random() < 0.6 else "regular"
    if a["usage"] == "ethnic":
        return "relaxed"
    return "regular" if rng.random() < 0.55 else "relaxed"


# body chest ranges (cm) per template, contiguous; ease by fit; length; shoulder
TEMPLATES = {
    "men": {"lo": 86, "step": 8, "ease": {"slim": 6, "regular": 10, "relaxed": 14},
            "length": {"shirt": [72, 74, 76, 78], "tshirt": [68, 70, 72, 74]},
            "shoulder": [44, 46, 48, 50]},
    "women": {"lo": 78, "step": 8, "ease": {"slim": 4, "regular": 8, "relaxed": 12},
              "length": {"shirt": [62, 64, 66, 68], "tshirt": [58, 60, 62, 64]},
              "shoulder": [37, 38, 40, 42]},
    "boys": {"lo": 60, "step": 6, "ease": {"slim": 6, "regular": 8, "relaxed": 10},
             "length": {"shirt": [50, 54, 58, 62], "tshirt": [48, 52, 56, 60]},
             "shoulder": [33, 35, 37, 39]},
    "girls": {"lo": 58, "step": 6, "ease": {"slim": 5, "regular": 7, "relaxed": 9},
              "length": {"shirt": [48, 52, 56, 60], "tshirt": [46, 50, 54, 58]},
              "shoulder": [31, 33, 35, 37]},
}
FIT_ADJ = {  # (length, shoulder) adjustment vs regular
    "slim": (-1, -1), "regular": (0, 0), "relaxed": (2, 1),
}


def brand_offset(brand: str, what: str) -> int:
    return zlib.crc32(f"{brand}:{what}".encode()) % 3 - 1  # -1, 0, +1


def build_size_chart(a: dict) -> dict:
    t = TEMPLATES["men" if a["gender"] == "unisex" else a["gender"]]
    fit = a["fit"]
    dl, ds = FIT_ADJ[fit]
    sizes = []
    for i, name in enumerate(["S", "M", "L", "XL"]):
        lo = t["lo"] + i * t["step"]
        hi = lo + t["step"]
        mid = (lo + hi) / 2
        sizes.append({
            "size": name,
            "body_chest": [lo, hi],
            "garment_chest": int(round(mid + t["ease"][fit])) + brand_offset(a["brand"], "chest"),
            "length": t["length"][a["type"]][i] + dl + brand_offset(a["brand"], "length"),
            "shoulder": t["shoulder"][i] + ds + brand_offset(a["brand"], "shoulder"),
        })
    return {"fit": fit, "sizes": sizes}


def type_word(a: dict) -> str:
    if a["type"] == "tshirt":
        return "polo t-shirt" if "polo" in a["source_name"].lower() else "t-shirt"
    return "shirt"


def build_name(a: dict) -> str:
    parts = [a["colour"]]
    if a["pattern"] != "solid":
        parts.append({"striped": "striped", "checked": "checked", "printed": "printed"}[a["pattern"]])
    if a.get("_sleeveless"):
        parts.append("sleeveless")
    parts.append(a["fabric_cue"])
    parts.append(type_word(a))
    s = " ".join(parts)
    return s[0].upper() + s[1:]


def build_description(a: dict) -> str:
    pat = {"solid": "solid colour", "striped": "striped", "checked": "checked", "printed": "printed"}[a["pattern"]]
    who = {"men": "men", "women": "women", "boys": "boys", "girls": "girls", "unisex": "everyone"}[a["gender"]]
    colour = a["colour"][0].upper() + a["colour"][1:]
    sl = "sleeveless" if a.get("_sleeveless") else f"{a['sleeve']}-sleeved"  # contract has no sleeveless value
    return (f"{colour} {sl} {a['usage']} {type_word(a)} in {a['fabric']} by {a['brand']}, "
            f"{pat}, {a['fit']} fit, for {who}, {a['season']}.")


# ---------------------------------------------------------------- build
def decide(item: dict, clip: dict | None, prior: dict) -> dict:
    """sleeve and pattern: name prior for pattern, CLIP otherwise."""
    out = {"pattern_src": "clip", "sleeve_src": "clip"}
    if "pattern" in prior:
        out["pattern"], out["pattern_src"] = prior["pattern"], "name"
    elif clip:
        out["pattern"] = max(clip["pattern"], key=clip["pattern"].get)
    else:
        out["pattern"] = "solid"
    if clip:
        out["sleeve"] = max(clip["sleeve"], key=clip["sleeve"].get)
    else:
        out["sleeve"] = "long" if item["articleType"] == "Shirts" else "short"
    return out


def build(raw: list[dict], preds: dict, overrides: dict) -> list[dict]:
    items = []
    for r in raw:
        iid = r["id"]
        src = re.sub(r"\s+", " ", r["productDisplayName"]).strip()
        brand, rest = parse_brand(src)
        a = {
            "id": iid, "brand": brand, "source_name": src,
            "gender": parse_gender(r["gender"], src),
            "type": "shirt" if r["articleType"] == "Shirts" else "tshirt",
            "colour": r["baseColour"].lower(),
            "usage": r["usage"].lower(), "season": r["season"].lower(),
            "image": r["image"],
        }
        a.update({k: v for k, v in decide(r, preds.get(str(iid)), name_prior(src)).items()})
        a.update({k: v for k, v in overrides.get(str(iid), {}).items()})
        rng = rng_for(iid)
        a["fabric"], a["fabric_cue"] = pick_fabric(a, rng)
        if "fabric" in overrides.get(str(iid), {}):
            a["fabric"] = overrides[str(iid)]["fabric"]
            a["fabric_cue"] = a["fabric"].split()[-1] if a["type"] == "shirt" else a["fabric_cue"]
        a["price"] = pick_price(a, rng_for(iid, "price"))
        a["fit"] = pick_fit(a, rng_for(iid, "fit"))
        a.update({k: v for k, v in overrides.get(str(iid), {}).items() if k in ("price", "fit")})
        item = {
            "id": iid,
            "name": build_name(a),
            "brand": brand,
            "gender": a["gender"],
            "type": a["type"],
            "colour": a["colour"],
            "sleeve": a["sleeve"],
            "pattern": a["pattern"],
            "usage": a["usage"],
            "season": a["season"],
            "fabric": a["fabric"],
            "price": a["price"],
            "image": a["image"],
            "description": build_description(a),
            "size_chart": build_size_chart(a),
            "source_name": src,
            "synthetic": ["price", "fabric", "size_chart"],
        }
        # final overrides: any field, applied last
        for k, v in overrides.get(str(iid), {}).items():
            if k in item:
                item[k] = v
        items.append(item)
    return items


# ---------------------------------------------------------- contact sheets
def make_sheets(raw, items, preds, cols=4, rows=3, cw=330, ch=440):
    from PIL import Image, ImageDraw, ImageFont

    SHEETS.mkdir(parents=True, exist_ok=True)
    for f in SHEETS.glob("sheet_*.jpg"):
        f.unlink()
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
    except Exception:
        font = ImageFont.load_default()
    per = cols * rows
    lab_h = 46
    for s in range(0, len(items), per):
        sheet = Image.new("RGB", (cols * cw, rows * (ch + lab_h)), "white")
        d = ImageDraw.Draw(sheet)
        for i, it in enumerate(items[s:s + per]):
            x, y = (i % cols) * cw, (i // cols) * (ch + lab_h)
            im = Image.open(CAT / it["image"]).convert("RGB")
            im.thumbnail((cw, ch))
            sheet.paste(im, (x + (cw - im.width) // 2, y))
            p = preds.get(str(it["id"]))
            conf = f" {p['sleeve'][it['sleeve']]:.2f}" if p else ""
            d.text((x + 4, y + ch + 2), f"{it['id']}  SLEEVE={it['sleeve'].upper()}{conf}", fill="red", font=font)
            d.text((x + 4, y + ch + 24), f"{it['pattern']} | {it['type']} | {it['brand'][:18]}", fill="black", font=font)
        sheet.save(SHEETS / f"sheet_{s // per:02d}.jpg", quality=88)
    print("sheets in", SHEETS)


def main():
    raw = json.loads(RAW.read_text())
    overrides = json.loads(OVERRIDES.read_text()) if OVERRIDES.exists() else {}
    if "--rerun-clip" in sys.argv or not CLIP_CACHE.exists():
        preds = run_clip(raw)
    else:
        preds = json.loads(CLIP_CACHE.read_text())
    items = build(raw, preds, overrides)
    OUT.write_text(json.dumps(items, indent=1))
    print(len(items), "items written to", OUT.relative_to(ROOT))
    if "--sheets" in sys.argv:
        make_sheets(raw, items, preds)


if __name__ == "__main__":
    main()
