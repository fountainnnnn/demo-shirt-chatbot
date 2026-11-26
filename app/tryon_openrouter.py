"""Hosted try-on: an image-editing model on OpenRouter replaces only the shirt in the person photo.

    check_photo(path) -> {"ok": bool, "issues": [str]}     # local checks only, no model
    run_tryon(person_path, garment_path, out_path, quality="balanced", garment=None)
        -> {"seconds", "device": "openrouter", "model", "cost_usd", "drift", "retried"}

`garment` is the catalog item; its name/colour/pattern/sleeve go into the prompt so the
model knows which garment in the catalog photo is meant (catalog models sometimes wear a
T-shirt under an open shirt).

Env: OPENROUTER_TRYON_MODEL (default below), TRYON_DRIFT_MAX (retry threshold).
The person photo goes first (the image to edit), the garment photo second, both as
base64 data URLs. The output must keep the input's aspect ratio; if it comes back in a
different one it is re-requested once and never stretched. A drift check compares the
zones that should not change (bottom 30 %, outer 12 % left/right, top 15 %); above the
threshold the edit is retried once with a stricter prompt and the better result kept.
"""
from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from app import llm_openrouter as orr

# Picked 2026-11-26 on a real selfie (two garments): faithful body, trousers, straps, watch and
# background at about $0.04 and 11 s per image. flash-lite redrew the body and trousers.
DEFAULT_MODEL = "google/gemini-nano-banana-2.1"
PERSON_MAX_SIDE = 1536
GARMENT_MAX_SIDE = 1024
MIN_SIDE = 256
# Calibrated on those runs: faithful edits 0.007-0.012, edits that redrew body/trousers 0.026-0.041.
DRIFT_MAX = float(os.environ.get("TRYON_DRIFT_MAX", "0.018"))
RATIOS = {"1:1": 1.0, "2:3": 2 / 3, "3:2": 1.5, "3:4": 0.75, "4:3": 4 / 3, "4:5": 0.8, "5:4": 1.25,
          "9:16": 9 / 16, "16:9": 16 / 9}

PROMPT = """Edit the FIRST image. It is a photo of a real person; the output must be this same photo with one change.
The SECOND image is a catalog photo, cropped to the torso of a model, showing the shirt to put on: {garment}.
Ignore everything else in the second image: the model's body, any T-shirt or vest under the shirt, trousers and accessories.

Replace only the shirt the person in the first image is wearing with that shirt. Change nothing else.
- Keep the person's body shape, width, shoulders, arms, hands, face, hair, glasses and skin exactly as they are.
  The new shirt follows the existing body outline and pose. Do not make the person larger, broader, slimmer or more muscular.
- Keep their trousers, belt, watch, bag and bag straps, phone and every other item exactly as they are,
  including anything that overlaps the shirt (straps stay on top of the new shirt, hands stay in front of it).
- Keep whatever the person wears under or around their shirt unless the new shirt would cover it.
- {buttoned}
- Keep the background, reflections, lighting, camera angle, crop and aspect ratio identical.
- Copy the shirt's colour, pattern, collar, buttons, pocket and sleeve length from the second image.
- {tuck}
Photorealistic. No text, no watermark, no border. Return only the edited photo."""

STRICT = ("\n\nYour previous edit changed parts of the photo other than the shirt (the trousers, the body size or the "
          "background). Redo it changing only the shirt; every pixel outside the shirt must stay as in the first image.")


def model_name() -> str:
    return os.environ.get("OPENROUTER_TRYON_MODEL") or DEFAULT_MODEL


def check_photo(path: str) -> dict:
    """Basic local checks: decodes, big enough, roughly portrait. No model, no network."""
    issues: list[str] = []
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            w, h = im.size
            im.convert("RGB").load()
    except Exception:
        return {"ok": False, "issues": ["This file could not be read as a photo."]}
    if min(w, h) < MIN_SIDE:
        issues.append(f"This photo is too small. Use one at least {MIN_SIDE * 2} pixels on the short side.")
    if w > h * 1.2:
        issues.append("This photo is wide. Use an upright photo with your upper body in view.")
    return {"ok": not issues, "issues": issues}


def _open(path: str) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def _data_url(im: Image.Image, max_side: int) -> str:
    im = im.copy()
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _garment_crop(im: Image.Image) -> Image.Image:
    # Catalog photos are on-model with the head in the top quarter. Cropping the head away
    # leaves one face in the request, so the model keeps the user's.
    w, h = im.size
    return im.crop((0, int(h * 0.24), w, int(h * 0.92)))


def describe_garment(item: dict | None) -> tuple[str, str]:
    """(garment phrase, buttoning rule) from the catalog item."""
    if not item:
        return "the shirt", "Render the shirt buttoned unless the person already wears a visible under-layer."
    kind = "T-shirt" if item.get("type") == "tshirt" else "button-up shirt"
    pattern = {"solid": "plain"}.get(item.get("pattern"), item.get("pattern", ""))
    sleeve = {"short": "short-sleeved", "long": "long-sleeved"}.get(item.get("sleeve"), "")
    phrase = f"the {item.get('colour', '')} {pattern} {sleeve} {kind} (\"{item.get('name', '')}\")"
    if kind == "T-shirt":
        rule = "It is a T-shirt: no buttons, no collar unless the second image shows one."
    else:
        rule = ("If the shirt is shown open over a T-shirt in the second image, render it buttoned on the person, "
                "unless the person already wears a visible under-layer.")
    return " ".join(phrase.split()), rule


def nearest_ratio(w: int, h: int) -> str:
    r = w / h
    return min(RATIOS, key=lambda k: abs(np.log(RATIOS[k] / r)))


def drift(before: Image.Image, after: Image.Image) -> float:
    """Mean absolute difference (0..1) in the zones that should not change: bottom 30 %,
    left/right outer 12 % and top 15 %, after scaling both to the same small size."""
    size = (96, int(round(96 * before.height / before.width)))
    a = np.asarray(before.resize(size, Image.BILINEAR), dtype=np.float32) / 255.0
    b = np.asarray(after.resize(size, Image.BILINEAR), dtype=np.float32) / 255.0
    h, w = a.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mask[int(h * 0.70):, :] = True
    mask[:, : int(w * 0.12)] = True
    mask[:, w - int(w * 0.12):] = True
    mask[: int(h * 0.15), :] = True
    return float(np.abs(a - b)[mask].mean())


def _first_image(body: dict) -> bytes:
    try:
        msg = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise orr.ProviderError("try-on model returned no choices") from None
    for img in msg.get("images") or []:
        url = (img.get("image_url") or {}).get("url") or ""
        if url.startswith("data:image/") and "," in url:
            return base64.b64decode(url.split(",", 1)[1])
    raise orr.ProviderError("try-on model returned no image")


def _call(model: str, person_url: str, garment_url, prompt: str, ratio: str) -> tuple[Image.Image, float]:
    """garment_url: one data URL, or a list of (caption, data URL) for several reference images."""
    refs = garment_url if isinstance(garment_url, list) else [
        ("Second image: the catalog shirt (reference only).", garment_url)]
    content = [{"type": "text", "text": "First image: the person (edit this one)."},
               {"type": "image_url", "image_url": {"url": person_url}}]
    for caption, url in refs:
        content += [{"type": "text", "text": caption}, {"type": "image_url", "image_url": {"url": url}}]
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": model,
        "modalities": ["image", "text"],
        "messages": [{"role": "user", "content": content}],
        "image_config": {"aspect_ratio": ratio},
        "usage": {"include": True},
    }
    if not model.startswith("openai/"):
        payload["temperature"] = 0.2
    body = orr.post("/chat/completions", payload, timeout=240.0)
    with Image.open(io.BytesIO(_first_image(body))) as im:
        out = im.convert("RGB")
    return out, float((body.get("usage") or {}).get("cost") or 0.0)


def _same_ratio(a: Image.Image, b: Image.Image) -> bool:
    return abs(np.log((a.width / a.height) / (b.width / b.height))) < 0.03


NOTES_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog" / "garment_notes.json"
NOTES_MODEL_DEFAULT = "google/gemini-3.1-flash-lite"
NOTES_SCHEMA = {
    "title": "garment_notes",
    "type": "object",
    "properties": {
        "pattern_scale": {"type": "string"},
        "checks_across_chest": {"type": ["integer", "null"]},
        "colours": {"type": "array", "items": {"type": "string"}},
        "buttons": {"type": "string"},
        "pocket": {"type": "string"},
        "logo": {"type": "string"},
        "collar": {"type": "string"},
        "sleeve_style": {"type": "string"},
        "hem": {"type": "string"},
    },
    "required": ["pattern_scale", "checks_across_chest", "colours", "buttons", "pocket", "logo", "collar",
                 "sleeve_style", "hem"],
    "additionalProperties": False,
}
NOTES_PROMPT = """Describe the shirt in this product photo for a tailor who must copy it exactly.
Ignore any T-shirt or vest visible underneath it. Be literal and short; say "none" when absent.
pattern_scale: the pattern and its size, e.g. "fine gingham, checks about 3 mm", "wide stripes 1 cm apart", "solid".
checks_across_chest: for a check or stripe, how many repeats across the front from side seam to side seam; else null.
colours: the exact colours, darkest first, e.g. ["deep black"] or ["black", "white"].
buttons: colour, size and material look. pocket: "none" or how many and where, as seen in the photo (image left/right).
logo: any logo, crest or embroidery, what it looks like and where it sits (on the pocket? image left/right?).
collar: style. sleeve_style: length and finish (cuffs, roll-up tabs, rolled). hem: straight or curved, length."""

_notes_lock = __import__("threading").Lock()


def garment_notes(item: dict, cutout: Image.Image) -> dict | None:
    """One-off description of the product photo by a cheap vision model, cached per item id."""
    import json

    key = str(item["id"])
    with _notes_lock:
        try:
            cache = json.loads(NOTES_PATH.read_text())
        except (OSError, ValueError):
            cache = {}
        if key in cache:
            return cache[key]
    model = os.environ.get("OPENROUTER_VISION_MODEL") or NOTES_MODEL_DEFAULT
    body = orr.post("/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": NOTES_PROMPT + f'\nCatalog name: "{item.get("name", "")}".'},
            {"type": "image_url", "image_url": {"url": _data_url(cutout, 1024)}}]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "garment_notes", "strict": True,
                                                                   "schema": NOTES_SCHEMA}},
        "temperature": 0, "max_tokens": 400, "usage": {"include": True},
    }, timeout=60.0)
    try:
        text = orr._strip_fences(body["choices"][0]["message"]["content"])
        notes = json.loads(text)
        orr.validate(notes, NOTES_SCHEMA)
    except Exception:
        return None
    notes["_model"] = model
    with _notes_lock:
        try:
            cache = json.loads(NOTES_PATH.read_text())
        except (OSError, ValueError):
            cache = {}
        cache[key] = notes
        NOTES_PATH.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
    return notes


def notes_text(item: dict | None, notes: dict | None) -> str:
    if not item:
        return ""
    lines = [f"Exact colour: {item.get('colour')} (true, deep colour as in the reference; do not lighten, grey or "
             f"wash it out)."]
    if notes:
        pat = notes.get("pattern_scale") or ""
        if notes.get("checks_across_chest"):
            pat += f", about {notes['checks_across_chest']} repeats across the chest"
        lines += [f"Pattern: {pat} (keep this exact scale; do not enlarge the check).",
                  f"Colours: {', '.join(notes.get('colours') or [])}.",
                  f"Buttons: {notes.get('buttons')}.", f"Pocket: {notes.get('pocket')}.",
                  f"Logo or embroidery: {notes.get('logo')}.", f"Collar: {notes.get('collar')}.",
                  f"Sleeves: {notes.get('sleeve_style')}.", f"Hem: {notes.get('hem')}."]
    lines.append("Pockets, logos and embroidery go on the same side of the wearer's body as in the reference "
                 "(the reference faces the camera; the person's photo may be a mirror selfie).")
    return "\nDetails of the shirt to copy exactly:\n- " + "\n- ".join(lines)


def detail_zoom(cutout: Image.Image) -> Image.Image:
    """Close-up of the upper front (collar, buttons, pocket, logo, fabric) for the true pattern scale."""
    w, h = cutout.size
    return cutout.crop((int(w * 0.12), int(h * 0.06), int(w * 0.88), int(h * 0.52)))


def _up(im: Image.Image, side: int = 1024) -> Image.Image:
    if max(im.size) >= side:
        return im
    k = side / max(im.size)
    return im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)


CROP_NOTE = ("\nThe first image is a close crop of a larger photo, centred on the shirt. Replace the shirt within "
             "this crop and keep everything else in the crop unchanged, with exactly the same framing; the edges of "
             "the crop must line up with the rest of the photo.")
CUTOUT_NOTE = ("\nThe second image shows the shirt cut out on a plain grey background. If an inner T-shirt is still "
               "visible inside it, that T-shirt is not part of the product: ignore it.")


def _flag(name: str) -> bool:
    return os.environ.get(name, "1") != "0"


def run_tryon(person_path: str, garment_path: str, out_path: str, quality: str = "balanced",
              garment: dict | None = None, on_stage=None) -> dict:
    """Replace the person's shirt with the catalog garment; save the result (PNG) to out_path.

    Stages (each can be switched off): TRYON_EXTRACT_GARMENT (send the parsed garment cut-out),
    TRYON_CROP (send only the torso crop), TRYON_COMPOSITE (paste back only the shirt region).
    """
    from app import tryon_composite as tc

    stage = on_stage or (lambda _s: None)
    t0 = time.perf_counter()
    model = model_name()
    use_extract, use_crop, use_comp = _flag("TRYON_EXTRACT_GARMENT"), _flag("TRYON_CROP"), _flag("TRYON_COMPOSITE")
    parser_ok = (use_extract or use_crop or use_comp) and tc.available()

    stage("Reading your photo")
    person = _open(person_path)
    seg_orig = None
    if parser_ok:
        seg_orig = tc.segment(person, tc.seg_size(*person.size))

    stage("Preparing the shirt")
    phrase, rule = describe_garment(garment)
    tuck = ("The shirt is worn untucked, hanging over the waistband to about the top of the thighs."
            if _flag("TRYON_UNTUCKED") else "Keep the shirt tucked in exactly as the original one.")
    prompt = PROMPT.format(garment=phrase, buttoned=rule, tuck=tuck)
    garment_img = None
    if parser_ok and use_extract and garment and garment.get("id") is not None:
        try:
            garment_img = tc.garment_cutout(garment["id"], garment_path)
            prompt = prompt.replace("cropped to the torso of a model, showing the shirt", "showing the shirt") \
                + CUTOUT_NOTE
        except Exception as e:
            garment_img = None
            log_warn(f"garment cut-out failed ({type(e).__name__}); using the catalog crop")
    if garment_img is None:
        garment_img = _garment_crop(_open(garment_path))
    refs = [("Second image: the catalog shirt (reference only).", _data_url(_up(garment_img), GARMENT_MAX_SIDE))]
    notes = None
    if garment_img is not None and parser_ok and use_extract and garment:
        refs.append(("Third image: a close-up of the same shirt's collar, buttons, pocket, logo and fabric "
                     "(reference only; it shows the true size of the pattern).",
                     _data_url(_up(detail_zoom(garment_img)), GARMENT_MAX_SIDE)))
        try:
            notes = garment_notes(garment, garment_img)
        except Exception as e:
            log_warn(f"garment notes failed ({type(e).__name__})")
    prompt += notes_text(garment, notes)
    garment_url = refs

    box = (0, 0, person.width, person.height)
    ratio = nearest_ratio(person.width, person.height)
    if parser_ok and use_crop:
        box, ratio = tc.crop_box(seg_orig, person.size, RATIOS)
        prompt += CROP_NOTE
    region = person.crop(box)
    if region.width < 1024 and region.height < 1024:  # upscale small crops so the long side is >= 1024
        k = 1024 / max(region.size)
        region = region.resize((round(region.width * k), round(region.height * k)), Image.LANCZOS)
    person_url = _data_url(region, PERSON_MAX_SIDE)

    stage("Painting the shirt")
    cost, calls, best = 0.0, 0, None
    for attempt in range(2):
        p = prompt if attempt == 0 else prompt + STRICT
        out, c = _call(model, person_url, garment_url, p, ratio)
        cost, calls = cost + c, calls + 1
        if not _same_ratio(out, region):
            # Never stretch: ask once more with the ratio spelled out in the prompt.
            out, c = _call(model, person_url, garment_url,
                           p + f"\nThe output must have aspect ratio {ratio}, like the first image.", ratio)
            cost, calls = cost + c, calls + 1
            if not _same_ratio(out, region):
                continue
        full = person.copy()
        full.paste(out.resize((box[2] - box[0], box[3] - box[1]), Image.LANCZOS), box[:2])
        d = drift(person, full)
        iou = None
        if parser_ok:
            iou = tc.person_iou(seg_orig, tc.segment(full, tc.seg_size(*person.size)))
        score = (iou if iou is not None else 1.0) - d
        if best is None or score > best["score"]:
            best = {"full": full, "drift": d, "iou": iou, "score": score}
        ok = (iou >= tc.ALIGN_MIN) if iou is not None else (d <= DRIFT_MAX)
        if ok:
            break
    if best is None:
        raise orr.ProviderError("the image model did not keep the photo's aspect ratio")

    stage("Blending it into your photo")
    result, composited, aligned, colour_matched = best["full"], False, None, False
    if parser_ok and use_comp:
        ref = None
        if garment_img is not None and use_extract and parser_ok:
            ref = tc.garment_pixels(garment_img)
        comp = tc.composite(person, best["full"], seg_orig=seg_orig, ref_pixels=ref)
        result, composited, aligned = comp["image"], True, comp["aligned"]
        colour_matched = comp["colour_matched"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    return {
        "seconds": round(time.perf_counter() - t0, 2),
        "device": "openrouter",
        "model": model,
        "cost_usd": round(cost, 5),
        "drift": round(best["drift"], 4),
        "iou": None if best["iou"] is None else round(best["iou"], 3),
        "retried": calls > 1,
        "composited": composited,
        "colour_matched": colour_matched,
        "aligned": aligned,
        "stages": {"extract": parser_ok and use_extract, "crop": parser_ok and use_crop, "composite": composited},
    }


def log_warn(msg: str) -> None:
    import logging
    logging.getLogger("fitting-room.tryon").warning(msg)


def unload() -> None:
    return None
