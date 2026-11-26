"""Mask-and-composite for hosted try-on: only the shirt region comes from the generated image.

The image model is asked to change only the shirt, but it redraws the whole frame (trousers
shift, the body grows). This step makes the rest of the photo pixel-identical to the
original: segment both images with the FASHN human parser (SegFormer, 18 classes, local,
already downloaded), build an edit mask around the upper-body garment, and blend

    output = original * (1 - mask) + generated * mask

Edit mask = (top in original  UNION  top in generated), dilated a few pixels,
            MINUS what the ORIGINAL shows as face, hair, hands, glasses, hat, jewelry,
            bag, belt, pants, skirt, legs or feet (hard, never overwritten),
            feathered with a Gaussian of about 0.75 % of the image width.
If the generated person is misaligned (silhouette IoU < ALIGN_MIN), only the ORIGINAL top
mask is used, so the body outline cannot grow.

The parser loads lazily (torch + transformers) and stays loaded. If it is unavailable the
caller keeps the un-composited image.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

log = logging.getLogger("fitting-room.composite")

ROOT = Path(__file__).resolve().parent.parent
PARSER_REPO = ROOT / "models" / "hf" / "hub" / "models--fashn-ai--fashn-human-parser" / "snapshots"
SEG_LONG_SIDE = 768          # segmentation resolution; the mask is upscaled afterwards
ALIGN_MIN = 0.85             # person-silhouette IoU below this = the model moved or resized the body

# FASHN human parser labels
BACKGROUND, FACE, HAIR, TOP, DRESS, SKIRT, PANTS, BELT, BAG, HAT, SCARF, GLASSES, ARMS, HANDS, LEGS, FEET, \
    TORSO, JEWELRY = range(18)
PROTECTED = (FACE, HAIR, SKIRT, PANTS, BELT, BAG, HAT, GLASSES, HANDS, LEGS, FEET, JEWELRY)
GARMENT = (TOP,)

_lock = threading.Lock()
_parser = None
_parser_failed = False


def enabled() -> bool:
    return os.environ.get("TRYON_COMPOSITE", "1") != "0"


def _load_parser():
    global _parser, _parser_failed
    if _parser is not None or _parser_failed:
        return _parser
    try:
        snaps = sorted(PARSER_REPO.glob("*/config.json"))
        if not snaps:
            raise FileNotFoundError("human parser weights missing; run scripts/fetch_tryon_model.sh")
        from fashn_human_parser import FashnHumanParser

        import torch

        want = os.environ.get("TRYON_PARSER_DEVICE", "auto").strip().lower()
        device = want if want in ("mps", "cpu") else ("mps" if torch.backends.mps.is_available() else "cpu")
        t0 = time.perf_counter()
        try:
            _parser = FashnHumanParser(model_id=str(snaps[-1].parent), device=device)
            _parser.predict(Image.new("RGB", (64, 64)))  # fail here, not mid-request, if MPS is unusable
        except Exception as e:
            if device == "cpu":
                raise
            log.warning("human parser on MPS failed (%s); using CPU", type(e).__name__)
            device = "cpu"
            _parser = FashnHumanParser(model_id=str(snaps[-1].parent), device=device)
        log.info("human parser loaded on %s in %.1fs", device, time.perf_counter() - t0)
    except Exception as e:  # torch/transformers missing, weights missing, ...
        _parser_failed = True
        log.warning("human parser unavailable (%s: %s); try-on results will not be composited",
                    type(e).__name__, e)
    return _parser


def available() -> bool:
    with _lock:
        return _load_parser() is not None


def segment(im: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Class map (H, W) at `size` (w, h) for an RGB image."""
    with _lock:
        parser = _load_parser()
        if parser is None:
            raise RuntimeError("human parser unavailable")
        return np.asarray(parser.predict(im.convert("RGB").resize(size, Image.BILINEAR)), dtype=np.uint8)


def seg_size(w: int, h: int) -> tuple[int, int]:
    s = SEG_LONG_SIDE / max(w, h)
    return max(1, round(w * s)), max(1, round(h * s))


def person_iou(seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    a, b = seg_a != BACKGROUND, seg_b != BACKGROUND
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(im.filter(ImageFilter.MaxFilter(2 * px + 1))) > 127


def edit_mask(seg_orig: np.ndarray, seg_gen: np.ndarray | None, aligned: bool = True,
              dilate_px: int = 3, untucked: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(region, protected) boolean masks at segmentation size."""
    region = np.isin(seg_orig, GARMENT)
    if aligned and seg_gen is not None:
        region |= np.isin(seg_gen, GARMENT)
    region = _dilate(region, dilate_px)
    protected = np.isin(seg_orig, PROTECTED)
    if untucked and aligned and seg_gen is not None:
        # the new shirt may hang over the waistband and the top of the thighs, nothing lower
        ys = np.nonzero((seg_orig == TOP).any(1))[0]
        if len(ys):
            limit = int(ys.max() + 0.22 * (ys.max() - ys.min()))
            hem = np.zeros_like(protected)
            hem[:limit] = True
            protected &= ~(np.isin(seg_orig, (PANTS, BELT)) & (seg_gen == TOP) & hem)
    return region & ~protected, protected


def blend(original: Image.Image, generated: Image.Image, region: np.ndarray, protected: np.ndarray,
          feather_frac: float = 0.0075) -> tuple[Image.Image, np.ndarray]:
    """Composite at the original's resolution. Pixels with mask 0 (everything outside the
    feathered region, and every protected pixel) are exactly the original's."""
    W, H = original.size
    gen = generated.convert("RGB").resize((W, H), Image.BICUBIC) if generated.size != (W, H) else generated.convert("RGB")
    m = Image.fromarray(region.astype(np.uint8) * 255).resize((W, H), Image.BILINEAR)
    radius = max(1.0, feather_frac * W)
    m = np.asarray(m.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0
    # Protected classes: upscale smoothly (nearest-neighbour leaves a staircase along straps and
    # hands), soften the edge by ~1.5 px, never let the shirt through where they are solid.
    prot = Image.fromarray(protected.astype(np.uint8) * 255).resize((W, H), Image.BILINEAR)
    prot = np.asarray(prot.filter(ImageFilter.GaussianBlur(max(1.0, W / 1000))), dtype=np.float32) / 255.0
    m = m * (1.0 - prot)
    m[prot >= 0.5] = 0.0          # protected classes are never overwritten, not even by the feather
    m[m < 1.0 / 255] = 0.0        # no faint haze outside the region
    o = np.asarray(original.convert("RGB"), dtype=np.float32)
    g = np.asarray(gen, dtype=np.float32)
    out = o * (1.0 - m[..., None]) + g * m[..., None]
    out = np.where(m[..., None] == 0.0, o, out)  # exact original outside the mask
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8)), m


def composite(original: Image.Image, generated: Image.Image, seg_orig: np.ndarray | None = None,
              ref_pixels: np.ndarray | None = None) -> dict:
    """Segment, check alignment, (colour-match), blend.
    Returns {"image", "aligned", "iou", "mask_frac", "seconds", "colour_matched"}."""
    t0 = time.perf_counter()
    size = seg_size(*original.size)
    if seg_orig is None:
        seg_orig = segment(original, size)
    generated = generated.convert("RGB").resize(original.size, Image.BICUBIC)
    seg_gen = segment(generated, size)
    iou = person_iou(seg_orig, seg_gen)
    aligned = iou >= ALIGN_MIN
    region, protected = edit_mask(seg_orig, seg_gen, aligned=aligned,
                                  untucked=os.environ.get("TRYON_UNTUCKED", "1") != "0")
    matched = False
    if ref_pixels is not None and os.environ.get("TRYON_COLOUR_MATCH", "1") != "0":
        # only the generated shirt's own pixels inside the edit region, at full resolution
        shirt = region & (seg_gen == TOP)
        full = np.asarray(Image.fromarray(shirt.astype(np.uint8) * 255).resize(original.size, Image.NEAREST)) > 127
        generated = Image.fromarray(colour_match(np.asarray(generated), full, ref_pixels))
        matched = True
    out, m = blend(original, generated, region, protected)
    return {"image": out, "aligned": aligned, "iou": round(iou, 3), "mask_frac": round(float((m > 0).mean()), 3),
            "seconds": round(time.perf_counter() - t0, 2), "seg_orig": seg_orig, "colour_matched": matched}


# ------------------------------------------------------------------ stage 1: clean garment image
GARMENT_DIR = ROOT / "data" / "catalog" / "garments"
GARMENT_BG = (232, 232, 232)


def garment_cutout(item_id: int, catalog_photo: str) -> Image.Image:
    """The catalog photo's top garment on plain light grey, cropped tight; cached per item id.

    The parser labels an open shirt and the T-shirt under it both as `top`, so for such
    photos both stay in the cut-out (the prompt tells the model to ignore the under-layer).
    """
    cache = GARMENT_DIR / f"{int(item_id)}.png"
    if cache.exists():
        with Image.open(cache) as im:
            return im.convert("RGB")
    with Image.open(catalog_photo) as im:
        photo = im.convert("RGB")
    seg = segment(photo, photo.size)
    mask = np.isin(seg, GARMENT)
    if mask.sum() < 0.02 * mask.size:
        raise RuntimeError("no garment found in the catalog photo")
    soft = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(1.2))
    out = Image.composite(photo, Image.new("RGB", photo.size, GARMENT_BG), soft)
    ys, xs = np.nonzero(mask)
    m = int(0.04 * max(photo.size))
    box = (max(0, xs.min() - m), max(0, ys.min() - m), min(photo.width, xs.max() + m), min(photo.height, ys.max() + m))
    out = out.crop(box)
    GARMENT_DIR.mkdir(parents=True, exist_ok=True)
    out.save(cache)
    return out


# ------------------------------------------------------------------ stage 2: torso crop
def crop_box(seg_orig: np.ndarray, full_size: tuple[int, int], ratios: dict[str, float],
             expand: float = 0.15) -> tuple[tuple[int, int, int, int], str]:
    """Box (in full-image pixels) around top + arms, grown `expand` per side, clamped, and
    adjusted to the nearest supported aspect ratio. Returns (box, ratio name)."""
    W, H = full_size
    sh, sw = seg_orig.shape
    ys, xs = np.nonzero(np.isin(seg_orig, (TOP, ARMS)))
    if len(xs) == 0:
        return (0, 0, W, H), min(ratios, key=lambda k: abs(np.log(ratios[k] * H / W)))
    sx, sy = W / sw, H / sh
    x0, x1, y0, y1 = xs.min() * sx, (xs.max() + 1) * sx, ys.min() * sy, (ys.max() + 1) * sy
    bw, bh = x1 - x0, y1 - y0
    x0, x1, y0, y1 = x0 - expand * bw, x1 + expand * bw, y0 - expand * bh, y1 + expand * bh
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    name = min(ratios, key=lambda k: abs(np.log(ratios[k] / ((x1 - x0) / (y1 - y0)))))
    r = ratios[name]
    w, h = x1 - x0, y1 - y0
    if w / h < r:   # too narrow: widen (else shrink height if the image is too narrow)
        w = min(W, h * r)
        h = w / r
    else:
        h = min(H, w / r)
        w = h * r
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = int(round(min(max(0, cx - w / 2), W - w)))
    y0 = int(round(min(max(0, cy - h / 2), H - h)))
    return (x0, y0, int(round(x0 + w)), int(round(y0 + h))), name


# ------------------------------------------------------------------ colour match (inside the shirt only)
def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16) / 116
    fx, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883], np.float32)
    m = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]], np.float32)
    c = np.clip(xyz @ m.T, 0, 1)
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)
    return np.clip(np.rint(c * 255), 0, 255).astype(np.uint8)


def garment_pixels(cutout: Image.Image) -> np.ndarray:
    """RGB pixels of the garment in a cut-out (everything that is not the grey background)."""
    a = np.asarray(cutout.convert("RGB"), dtype=np.int16)
    keep = np.abs(a - np.array(GARMENT_BG, np.int16)).max(-1) > 14
    return a[keep].astype(np.uint8)


def colour_match(gen: np.ndarray, mask: np.ndarray, ref: np.ndarray, strength: float = 0.8) -> np.ndarray:
    """Move the masked pixels' LAB mean/std toward the catalog garment's.

    Lightness: the mean is shifted by `strength` of the gap and the spread is scaled at most
    x0.75-1.33, so folds and the scene's shading survive. a/b (hue): mean and spread matched
    the same way. Pixels outside the mask are untouched.
    """
    if mask.sum() < 50 or len(ref) < 50:
        return gen
    out = gen.copy()
    lab = _srgb_to_lab(gen[mask])
    rlab = _srgb_to_lab(ref)
    mu, sd = lab.mean(0), lab.std(0) + 1e-3
    rmu, rsd = rlab.mean(0), rlab.std(0) + 1e-3
    ratio = np.clip(rsd / sd, 0.75, 1.33)
    ratio = 1 + strength * (ratio - 1)
    new = (lab - mu) * ratio + mu + strength * (rmu - mu)
    out[mask] = _lab_to_srgb(new)
    return out
