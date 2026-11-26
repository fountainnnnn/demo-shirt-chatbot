"""Local try-on with DM-VTON (Distilled Mobile Real-time Virtual Try-On, 2023), ported to Apple GPU / CPU.

    check_photo(path) -> {"ok": bool, "issues": [str]}
    run_tryon(person_path, garment_path, out_path, quality="balanced")
        -> {"seconds", "device", "model": "dm-vton", "composited", "aligned", "iou", "stages", "timings"}
    unload()

Code: vendor/dm-vton (github.com/KiseKloset/DM-VTON, CC BY-NC-SA 4.0, academic use only).
Weights: models/dmvton/dmvton_pf_{warp,gen}.pt (scripts/fetch_dmvton.sh), same licence, trained
on VITON (not VITON-HD) at 192x256.

Port: the reference flow module imports a cupy CUDA kernel for its 7x7 local cost volume
(models/common/correlation.py). The STUDENT warp module (MobileAFWM) uses it at inference, so a
pure-PyTorch `correlation()` below is registered under that module name before the vendor code is
imported; cupy is never imported and vendor/ is unmodified. The only other CUDA call
(torch.cuda.FloatTensor) is in the training branch, which inference never runs.

Pipeline (reuses app/tryon_composite.py; nothing there is modified):
  1. parse the original photo (FASHN human parser)
  2. 3:4 crop around top + arms (tc.crop_box), optional white background like VITON
  3. garment: parser `top` mask of the catalog photo -> garment on white, centred in 192x256,
     plus its binary edge mask (VITON test_color / test_edge)
  4. DM-VTON at 192x256 (or 2x with DMVTON_SCALE=2)
  5. Lanczos upscale (+ light unsharp mask) back to the crop box
  6. blend into the full-resolution original inside the ORIGINAL shirt mask, never over
     face/hands/bag/trousers (tc.edit_mask + tc.blend)

Env: TRYON_DMVTON_DEVICE (auto|mps|cpu), DMVTON_SCALE (1|2), DMVTON_WHITE_BG (1|0),
     DMVTON_MASK_HANDS (1|0), DMVTON_FRAMING (torso|tight), DMVTON_SHARPEN (1|0).
Polish (see the section at the end; defaults in brackets):
     DMVTON_HIRES [1]     warp the 4x catalog garment with DM-VTON's own flow (sharp fabric)
     DMVTON_POLISH [1]    full-resolution paste: smooth class masks, inpaint old-shirt gaps,
                          decontaminated edge, no white halo
     DMVTON_DESPECKLE [1] median clean-up inside the garment (collar specks)
     DMVTON_GRAIN [1]     add the photo's noise level inside the garment
     TRYON_UNTUCKED [1]   the hem may cover the waistband/top of the thighs (DM-VTON's warp
                          itself ends near the original waistline, so the effect is small)
     DMVTON_STRAPS [0]    keep bag straps by colour + shape: not robust (also catches piping,
                          shadows; nothing on dark shirts), off by default
     DMVTON_ARM [0]       skin colour/texture for invented arm: made it worse, off by default
     DMVTON_CLOSE_FRONT [0] fill an under-layer band with shirt texture: duplicates the
                          placket, off; such items are listed in data/catalog/worn_open.json

Licence: DM-VTON code and weights are CC BY-NC-SA 4.0, for academic purposes only.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "dm-vton"
WEIGHTS = ROOT / "models" / "dmvton"
GARMENT_CACHE = ROOT / "data" / "tryon" / "dmvton_garments"
W0, H0 = 192, 256          # native input size (width, height)
MIN_SIDE = 256

_lock = threading.Lock()
_model = None
_device = None
last_debug: dict = {}   # inputs/raw output of the last run, for side-by-sides


# ------------------------------------------------------------------ pure-PyTorch correlation
def correlation(first, second, max_disp: int = 3):
    """Local cost volume, identical to the FlowNet2/PF-AFN cupy kernel at stride 1:
    out[:, (dy+md)*(2md+1) + (dx+md), y, x] = mean_c first[:, c, y, x] * second[:, c, y+dy, x+dx],
    zero outside the image."""
    import torch
    import torch.nn.functional as F

    _, _, h, w = first.shape
    pad = F.pad(second, (max_disp,) * 4)
    outs = []
    for dy in range(2 * max_disp + 1):
        for dx in range(2 * max_disp + 1):
            outs.append((first * pad[:, :, dy:dy + h, dx:dx + w]).mean(1))
    return torch.stack(outs, 1)


def _function_correlation(tenFirst, tenSecond, intStride=1):
    if intStride != 1:
        raise NotImplementedError("only intStride=1 is used by DM-VTON")
    return correlation(tenFirst, tenSecond, 3)


def _import_vendor():
    """Import the vendor pipeline with the cupy correlation replaced (no vendor patch)."""
    if "models.common.correlation" not in sys.modules:
        shim = types.ModuleType("models.common.correlation")
        shim.FunctionCorrelation = _function_correlation
        sys.modules["models.common.correlation"] = shim
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from pipelines.dmvton_pipeline import DMVTONPipeline  # noqa: E402
    return DMVTONPipeline


def pick_device() -> str:
    import torch

    want = os.environ.get("TRYON_DMVTON_DEVICE", "auto").strip().lower()
    if want in ("mps", "cpu"):
        return want
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load_weights(path: Path) -> dict:
    """The checkpoints hold {"epoch", "best_fid" (a numpy scalar), "model", "optimizer"}.
    Load with weights_only=True, allow-listing only the numpy scalar types."""
    import torch

    core = getattr(np, "_core", None) or np.core
    # the pickle names it numpy.core.multiarray.scalar (numpy 1.x); under numpy 2 it lives in numpy._core
    allowed = [(core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype]
    allowed += [type(np.dtype(t)) for t in ("float64", "float32")]
    with torch.serialization.safe_globals(allowed):
        ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = dict(ck.get("model", ck))
    sd.pop("aflow_net.weight", None)  # fixed Laplacian filter created only in the training branch
    return sd


def load():
    global _model, _device
    with _lock:
        if _model is None:
            import torch

            for f in ("dmvton_pf_warp.pt", "dmvton_pf_gen.pt"):
                if not (WEIGHTS / f).exists():
                    raise FileNotFoundError(f"{WEIGHTS / f} missing; run scripts/fetch_dmvton.sh")
            DMVTONPipeline = _import_vendor()
            _device = pick_device()
            m = DMVTONPipeline(align_corners=True, checkpoints=None)
            m.warp_model.load_state_dict(_load_weights(WEIGHTS / "dmvton_pf_warp.pt"))
            m.gen_model.load_state_dict(_load_weights(WEIGHTS / "dmvton_pf_gen.pt"))
            _model = m.to(_device).eval()
            torch.set_grad_enabled(False)
        return _model


def unload() -> None:
    global _model
    with _lock:
        _model = None
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


def check_photo(path: str) -> dict:
    from app.tryon_openrouter import check_photo as _check
    return _check(path)


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) != "0"


def _open(path: str) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


# ------------------------------------------------------------------ garment: VITON-style cloth + edge
def garment_inputs(garment_path: str) -> tuple[Image.Image, Image.Image]:
    """(cloth RGB on white, edge L) at 192x256, like VITON test_color / test_edge. Cached."""
    from app import tryon_composite as tc

    p = Path(garment_path)
    key = f"{p.stem}_{int(p.stat().st_mtime)}"
    cc, ce = GARMENT_CACHE / f"{key}_cloth.png", GARMENT_CACHE / f"{key}_edge.png"
    if cc.exists() and ce.exists():
        return Image.open(cc).convert("RGB"), Image.open(ce).convert("L")
    photo = _open(garment_path)
    seg = tc.segment(photo, photo.size)
    mask = np.isin(seg, tc.GARMENT)
    if mask.sum() < 0.02 * mask.size:
        raise RuntimeError("no garment found in the catalog photo")
    m = Image.fromarray(mask.astype(np.uint8) * 255)
    m = m.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))  # close small gaps
    ys, xs = np.nonzero(np.asarray(m) > 127)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    cloth = Image.composite(photo, Image.new("RGB", photo.size, (255, 255, 255)), m).crop(box)
    edge = m.crop(box)
    # VITON cloth images: garment centred, filling most of the 192x256 frame on white
    k = min(W0 * 0.86 / cloth.width, H0 * 0.86 / cloth.height)
    size = (max(1, round(cloth.width * k)), max(1, round(cloth.height * k)))
    cloth, edge = cloth.resize(size, Image.LANCZOS), edge.resize(size, Image.BILINEAR)
    canvas, ecanvas = Image.new("RGB", (W0, H0), (255, 255, 255)), Image.new("L", (W0, H0), 0)
    off = ((W0 - size[0]) // 2, (H0 - size[1]) // 2)
    canvas.paste(cloth, off)
    ecanvas.paste(edge.point(lambda v: 255 if v > 127 else 0), off)
    GARMENT_CACHE.mkdir(parents=True, exist_ok=True)
    canvas.save(cc)
    ecanvas.save(ce)
    return canvas, ecanvas


# ------------------------------------------------------------------ person crop
def person_box(seg_orig: np.ndarray, size: tuple[int, int]) -> tuple[int, int, int, int]:
    from app import tryon_composite as tc

    expand = 0.15 if os.environ.get("DMVTON_FRAMING", "torso") == "torso" else 0.05
    box, _ = tc.crop_box(seg_orig, size, {"3:4": W0 / H0}, expand=expand)
    return box


def person_input(person: Image.Image, seg_full: np.ndarray, box) -> tuple[Image.Image, np.ndarray]:
    """The crop, with the background painted white (VITON studio backgrounds) and optionally
    the hands painted white so the warp does not try to put the shirt over them.
    Returns (image, painted mask at crop size)."""
    from app import tryon_composite as tc

    crop = person.crop(box)
    seg = seg_full[box[1]:box[3], box[0]:box[2]]
    paint = np.zeros(seg.shape, bool)
    if _flag("DMVTON_WHITE_BG"):
        paint |= seg == tc.BACKGROUND
    if _flag("DMVTON_MASK_HANDS", "0"):
        paint |= seg == tc.HANDS
    if paint.any():
        a = np.asarray(crop).copy()
        a[paint] = 255
        crop = Image.fromarray(a)
    return crop, paint


def _to_tensor(im: Image.Image, device, normalize=True):
    import torch

    a = np.asarray(im, dtype=np.float32) / 255.0
    if a.ndim == 2:
        a = a[..., None]
    t = torch.from_numpy(a).permute(2, 0, 1)[None]
    return (t * 2 - 1 if normalize else t).to(device)


def infer(person_im: Image.Image, cloth: Image.Image, edge: Image.Image, scale: int = 1) -> Image.Image:
    """Run DM-VTON on prepared inputs; returns the try-on at (192*scale, 256*scale)."""
    import torch

    model = load()
    size = (W0 * scale, H0 * scale)
    p = _to_tensor(person_im.resize(size, Image.BICUBIC), _device)
    c = _to_tensor(cloth.resize(size, Image.BICUBIC), _device)
    e = _to_tensor(edge.resize(size, Image.NEAREST), _device, normalize=False)
    with torch.no_grad():
        out, _ = model(p, c, e, phase="test")
    out = ((out[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(out)


def run_tryon(person_path: str, garment_path: str, out_path: str, quality: str = "balanced",
              garment: dict | None = None, on_stage=None) -> dict:
    from app import tryon_composite as tc

    stage = on_stage or (lambda _s: None)
    t0 = time.perf_counter()
    timings = {}
    scale = int(os.environ.get("DMVTON_SCALE", "1"))

    t = time.perf_counter()
    load()
    timings["load"] = time.perf_counter() - t

    stage("Reading your photo")
    t = time.perf_counter()
    person = _open(person_path)
    seg_orig = tc.segment(person, tc.seg_size(*person.size))
    seg_full = np.asarray(Image.fromarray(seg_orig).resize(person.size, Image.NEAREST))
    timings["parse"] = time.perf_counter() - t

    stage("Preparing the shirt")
    t = time.perf_counter()
    cloth, edge = garment_inputs(garment_path)
    box = person_box(seg_orig, person.size)
    pin, painted = person_input(person, seg_full, box)
    timings["prepare"] = time.perf_counter() - t

    stage("Fitting the shirt to your photo")
    t = time.perf_counter()
    bw, bh = box[2] - box[0], box[3] - box[1]
    info = {}
    if _flag("DMVTON_HIRES"):
        _, _, cloth_hi, edge_hi = garment_inputs_hr(garment_path)
        parts = forward_parts(pin, cloth, edge)
        rgb, alpha = render_hires(parts, cloth_hi, edge_hi, (bw, bh))
        gen = Image.fromarray(rgb)
    else:
        gen = infer(pin, cloth, edge, scale)
        parts = None
    if _device == "mps":
        import torch
        torch.mps.synchronize()
    timings["dmvton"] = time.perf_counter() - t
    global last_debug
    last_debug = {"person_in": pin, "cloth": cloth, "edge": edge, "raw": gen}

    stage("Blending it into your photo")
    t = time.perf_counter()
    if parts is None:
        up = gen.resize((bw, bh), Image.LANCZOS)
        if _flag("DMVTON_SHARPEN"):
            up = up.filter(ImageFilter.UnsharpMask(radius=max(1.0, bw / 400), percent=60, threshold=2))
        rgb = np.asarray(up)
        alpha = None
    if alpha is None or not _flag("DMVTON_POLISH"):
        full = person.copy()
        up_im = Image.fromarray(rgb)
        if painted.any():
            up_im = Image.composite(person.crop(box), up_im, Image.fromarray(painted.astype(np.uint8) * 255))
        full.paste(up_im, box[:2])
        timings["upscale"] = time.perf_counter() - t
        t = time.perf_counter()
        region, protected = tc.edit_mask(seg_orig, None, aligned=False)
        result, _ = tc.blend(person, full, region, protected)
        iou = None
    else:
        result, info = polish_composite(person, seg_full, box, rgb, alpha, painted)
        iou = None
    timings["composite"] = time.perf_counter() - t

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    return {
        "seconds": round(time.perf_counter() - t0, 2),
        "device": _device,
        "model": "dm-vton",
        "composited": True,
        "polish": info,
        "stages": {"extract": True, "crop": True, "composite": True},
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "box": list(box),
        "scale": scale,
    }


# ================================================================== polish (all switchable)
# DMVTON_HIRES      warp the high-resolution catalog garment with DM-VTON's own flow field
# DMVTON_STRAPS     keep bag straps that the parser files under "top"
# DMVTON_REFINE     full-resolution paste mask: decontaminated, eroded, feathered; inpaint where the
#                   old shirt shows outside the new one (no white outline, no ghost)
# DMVTON_ARM        skin colour + texture for arm areas DM-VTON had to invent
# DMVTON_GRAIN      match the photo's noise inside the pasted shirt
# DMVTON_DESPECKLE  median clean-up of the warped garment (collar specks)
# TRYON_UNTUCKED    the shirt hangs over the waistband instead of being tucked in
HIRES = 4  # garment canvas at 4x the model input (768x1024)


def garment_inputs_hr(garment_path: str):
    """(cloth, edge) at 192x256 for the model, plus the same layout at HIRES x for the final warp."""
    from app import tryon_composite as tc

    p = Path(garment_path)
    key = f"{p.stem}_{int(p.stat().st_mtime)}_x{HIRES}"
    paths = [GARMENT_CACHE / f"{key}_{n}.png" for n in ("cloth", "edge", "cloth_hi", "edge_hi")]
    if all(q.exists() for q in paths):
        c, e, ch, eh = (Image.open(q) for q in paths)
        return c.convert("RGB"), e.convert("L"), ch.convert("RGB"), eh.convert("L")
    photo = _open(garment_path)
    seg = tc.segment(photo, photo.size)
    mask = np.isin(seg, tc.GARMENT)
    if mask.sum() < 0.02 * mask.size:
        raise RuntimeError("no garment found in the catalog photo")
    m = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    ys, xs = np.nonzero(np.asarray(m) > 127)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    if _flag("DMVTON_CLOSE_FRONT", "0"):  # off: the fill duplicates the placket; see worn_open
        photo, _ = close_front(photo, np.asarray(m) > 127, box)
    cloth = Image.composite(photo, Image.new("RGB", photo.size, (255, 255, 255)), m).crop(box)
    edge = m.crop(box)
    out = []
    for s in (HIRES, 1):
        W, H = W0 * s, H0 * s
        k = min(W * 0.86 / cloth.width, H * 0.86 / cloth.height)
        size = (max(1, round(cloth.width * k)), max(1, round(cloth.height * k)))
        c2, e2 = cloth.resize(size, Image.LANCZOS), edge.resize(size, Image.BILINEAR)
        canvas, ecanvas = Image.new("RGB", (W, H), (255, 255, 255)), Image.new("L", (W, H), 0)
        off = ((W - size[0]) // 2, (H - size[1]) // 2)
        canvas.paste(c2, off)
        ecanvas.paste(e2.point(lambda v: 255 if v > 127 else 0), off)
        out.append((canvas, ecanvas))
    (ch, eh), (c, e) = out
    GARMENT_CACHE.mkdir(parents=True, exist_ok=True)
    for im, q in zip((c, e, ch, eh), paths):
        im.save(q)
    return c, e, ch, eh


def forward_parts(person_im, cloth, edge):
    """DM-VTON forward (same maths as the vendor pipeline) returning its intermediate parts."""
    import torch
    import torch.nn.functional as F

    model = load()
    p = _to_tensor(person_im.resize((W0, H0), Image.BICUBIC), _device)
    c = _to_tensor(cloth.resize((W0, H0), Image.BICUBIC), _device)
    e = (_to_tensor(edge.resize((W0, H0), Image.NEAREST), _device, normalize=False) > 0.5).float()
    with torch.no_grad():
        warped_cloth, last_flow = model.warp_model(p, c * e, phase="test")
        warped_edge = F.grid_sample(e, last_flow.permute(0, 2, 3, 1), mode="bilinear", padding_mode="zeros",
                                    align_corners=model.align_corners)
        g = model.gen_model(torch.cat([p, warped_cloth, warped_edge], 1))
        p_rendered, m_comp = torch.split(g, [3, 1], 1)
        p_rendered, m_comp = torch.tanh(p_rendered), torch.sigmoid(m_comp) * warped_edge
    return {"flow": last_flow, "m": m_comp, "rendered": p_rendered, "warped_edge": warped_edge,
            "align_corners": model.align_corners}


def render_hires(parts, cloth_hi, edge_hi, size):
    """Try-on at `size` (w, h): the HIGH-RES garment warped with the upsampled flow, composited over
    DM-VTON's rendered body with its own (upsampled) mask. Returns (rgb uint8, garment alpha 0..1)."""
    import torch
    import torch.nn.functional as F

    w, h = size
    dev = parts["flow"].device
    grid = F.interpolate(parts["flow"], size=(h, w), mode="bilinear", align_corners=True).permute(0, 2, 3, 1)
    ch = _to_tensor(cloth_hi, dev)
    eh = (_to_tensor(edge_hi, dev, normalize=False) > 0.5).float()
    with torch.no_grad():
        wc = F.grid_sample(ch * eh, grid, mode="bilinear", padding_mode="zeros", align_corners=parts["align_corners"])
        we = F.grid_sample(eh, grid, mode="bilinear", padding_mode="zeros", align_corners=parts["align_corners"])
        m = F.interpolate(parts["m"], size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)
        m = torch.minimum(m, we)
        rend = F.interpolate(parts["rendered"], size=(h, w), mode="bicubic", align_corners=False).clamp(-1, 1)
        out = wc * m + rend * (1 - m)
    rgb = ((out[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    alpha = m[0, 0].float().cpu().numpy()
    return rgb, alpha


def strap_mask(orig: np.ndarray, seg_full: np.ndarray) -> np.ndarray:
    """Bag straps the parser filed under `top`: pixels inside the original shirt that are much darker
    than the shirt and form elongated pieces or touch the parser's bag pixels. Works when the shirt is
    clearly lighter than the straps; a dark shirt gives no contrast and the mask comes out empty."""
    import cv2
    from app import tryon_composite as tc

    top = seg_full == tc.TOP
    if top.sum() < 500:
        return np.zeros(top.shape, bool)
    lab = tc._srgb_to_lab(orig[top])
    L_med = float(np.median(lab[:, 0]))
    L = np.zeros(top.shape, np.float32)
    L[top] = lab[:, 0]
    dark = top & (L < L_med - 22)
    if L_med < 35 or dark.sum() < 0.002 * top.sum():  # dark shirt: no contrast to work with
        return np.zeros(top.shape, bool)
    bag = cv2.dilate((seg_full == tc.BAG).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    n, lab_im, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), connectivity=8)
    keep = np.zeros(top.shape, bool)
    min_area = 0.004 * top.sum()
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        comp = lab_im == i
        elongated = max(w, h) / max(1, min(w, h)) >= 2.5 or area / max(1, w * h) < 0.45
        if elongated or (comp & bag).any():
            keep |= comp
    keep = cv2.dilate(keep.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return keep & (top | (seg_full == tc.BAG))


def noise_sigma(img: np.ndarray, mask: np.ndarray) -> float:
    """Photo noise level: std of the high-pass layer in a region (median-robust)."""
    import cv2

    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.5)
    v = hp[mask]
    if v.size < 100:
        return 0.0
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def _flagd(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) != "0"


def _smooth(mask: np.ndarray, sigma: float) -> np.ndarray:
    """A class mask upscaled from parser resolution has stair-step edges; blur and re-threshold."""
    import cv2

    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma) > 0.5


def polish_composite(person: Image.Image, seg_full: np.ndarray, box, rgb_crop: np.ndarray,
                     alpha_crop: np.ndarray, painted: np.ndarray) -> tuple[Image.Image, dict]:
    """Paste the try-on crop into the full photo. Pixels outside the final mask are the original's.
    Returns (image, info)."""
    import cv2
    from app import tryon_composite as tc

    info = {}
    orig = np.asarray(person, dtype=np.uint8)
    H, W = orig.shape[:2]
    x0, y0, x1, y1 = box
    gen = orig.copy()
    gen[y0:y1, x0:x1] = rgb_crop
    garment = np.zeros((H, W), np.float32)
    garment[y0:y1, x0:x1] = alpha_crop
    G = garment > 0.5

    sig = max(1.0, W / 768)  # one parser pixel
    top_o = _smooth(seg_full == tc.TOP, sig)
    prot_classes = [tc.FACE, tc.HAIR, tc.HANDS, tc.GLASSES, tc.HAT, tc.JEWELRY, tc.BAG, tc.FEET, tc.SKIRT, tc.LEGS]
    untucked = _flagd("TRYON_UNTUCKED")
    raw_pants = np.isin(seg_full, (tc.PANTS, tc.BELT))
    pants = raw_pants | _smooth(raw_pants, sig)  # strict: trousers are never painted unless untucked
    protected = _smooth(np.isin(seg_full, prot_classes), sig)
    if untucked:
        # the hem may cover the waistband and the top of the thighs, nothing lower
        ys = np.nonzero(top_o.any(1))[0]
        if len(ys):
            shirt_bottom = ys.max()
            torso = shirt_bottom - ys.min()
            limit = int(shirt_bottom + 0.22 * torso)
            hem_zone = np.zeros_like(pants)
            hem_zone[:limit] = True
            protected |= pants & ~(G & hem_zone)
            info["hem_limit_px"] = int(limit - shirt_bottom)
        else:
            protected |= pants
    else:
        protected |= pants
    if _flagd("DMVTON_STRAPS", "0"):  # off: on a real photo it also caught piping and shadows
        straps = strap_mask(orig, seg_full)
        protected |= straps
        info["strap_px"] = int(straps.sum())

    if _flagd("DMVTON_DESPECKLE"):
        med = cv2.medianBlur(gen, 5)
        gen = np.where(G[..., None], med, gen)

    region = (top_o | G) & ~protected
    if _flagd("DMVTON_REFINE"):
        # old shirt showing outside the new silhouette: keep DM-VTON's pixels there only where it drew
        # skin (a shorter sleeve shows the arm); everything else is inpainted from the real background,
        # so there is no white fringe and no ghost of the old shirt
        painted_full = np.zeros((H, W), bool)
        painted_full[y0:y1, x0:x1] = painted
        skin_ref = orig[_smooth(seg_full == tc.ARMS, sig) | (seg_full == tc.HANDS)]
        skin_like = np.zeros((H, W), bool)
        if len(skin_ref) > 200:
            ref = tc._srgb_to_lab(skin_ref).mean(0)
            cand = top_o & ~G
            d = np.linalg.norm(tc._srgb_to_lab(gen[cand]) - ref, axis=1)
            skin_like[cand] = d < 22
        hole = top_o & ~G & ~protected & ~skin_like
        hole = cv2.dilate(hole.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        hole &= ~protected & (top_o | painted_full)
        if hole.any():
            known = orig.copy()
            inp = cv2.inpaint(cv2.cvtColor(known, cv2.COLOR_RGB2BGR), hole.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
            gen[hole] = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)[hole]
        info["inpainted_px"] = int(hole.sum())
        # decontaminate the garment edge: its outer pixels take colours from just inside
        Gi = cv2.erode(G.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        band = G & ~Gi
        if band.any():
            # normalised convolution: average of interior pixels only, so no white can leak in
            wgt = cv2.GaussianBlur(Gi.astype(np.float32), (0, 0), 3)
            col = cv2.GaussianBlur(gen.astype(np.float32) * Gi[..., None], (0, 0), 3)
            fill = col / np.maximum(wgt[..., None], 1e-3)
            ok = band & (wgt > 0.02)
            gen[ok] = np.clip(fill[ok], 0, 255).astype(np.uint8)
        region = (top_o | G | hole) & ~protected
        # outside the new garment, the inpainted gap and DM-VTON's skin, the "generated" layer is the
        # original photo, so the feathered edge blends photo with photo (no white halo from DM-VTON's
        # painted background)
        keep_gen = G | hole | skin_like
        gen = np.where(keep_gen[..., None], gen, orig)

    if _flagd("DMVTON_ARM", "0"):  # off: the colour pass caught the neck and left blotches
        # arm areas DM-VTON invented (inside the old shirt, outside the new garment, not background)
        arm = top_o & ~G & ~protected & ~((gen.min(-1) > 225) & (np.ptp(gen, -1) < 18))
        if _flagd("DMVTON_REFINE"):
            arm &= ~hole
        skin = seg_full == tc.ARMS
        if arm.sum() > 200 and skin.sum() > 500:
            gen[arm] = tc.colour_match(gen, arm, orig[skin], strength=0.9)[arm]
            g = orig.astype(np.float32)
            hp = g - cv2.GaussianBlur(g, (0, 0), 2.0)
            ys, xs = np.nonzero(skin)
            cy, cx = int(np.median(ys)), int(np.median(xs))
            r = 24
            patch = hp[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
            if patch.size:
                reps = (H // patch.shape[0] + 1, W // patch.shape[1] + 1, 1)
                tile = np.tile(patch, reps)[:H, :W]
                gen = np.where(arm[..., None], np.clip(gen.astype(np.float32) + tile, 0, 255), gen).astype(np.uint8)
            info["arm_px"] = int(arm.sum())

    if _flagd("DMVTON_GRAIN"):
        bg = seg_full == tc.BACKGROUND
        sig = noise_sigma(orig, bg)
        sig_gen = noise_sigma(gen, G)
        add = max(0.0, (sig ** 2 - sig_gen ** 2)) ** 0.5
        if add > 0.3:
            rng = np.random.default_rng(0)
            n = rng.normal(0, add, (H, W, 1)).astype(np.float32)
            gen = np.where(G[..., None], np.clip(gen.astype(np.float32) + n, 0, 255), gen).astype(np.uint8)
        info["grain_sigma"] = round(add, 2)

    # final mask: full resolution, slightly eroded at the garment silhouette, feathered, then
    # hard zero on protected classes
    m = region.astype(np.float32)
    feather = max(1.0, W / 1000)
    m = cv2.GaussianBlur(m, (0, 0), feather)
    m[protected] = 0.0
    m[m < 1.0 / 255] = 0.0
    out = orig.astype(np.float32) * (1 - m[..., None]) + gen.astype(np.float32) * m[..., None]
    out = np.where(m[..., None] == 0, orig, np.clip(np.rint(out), 0, 255)).astype(np.uint8)
    info["mask_px"] = int((m > 0).sum())
    return Image.fromarray(out), info


def under_layer(photo: np.ndarray, mask: np.ndarray, box) -> np.ndarray:
    """Pixels of an inner T-shirt seen through an open shirt: inside the garment mask, in the middle
    of the garment, light and smooth (low local contrast), as one vertical band."""
    import cv2

    x0, y0, x1, y1 = box
    g = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(g, (9, 9))
    std = np.sqrt(np.maximum(cv2.blur(g * g, (9, 9)) - mean * mean, 0))
    w = x1 - x0
    central = np.zeros(mask.shape, bool)
    central[:, x0 + int(0.28 * w):x0 + int(0.72 * w)] = True
    cand = mask & central & (std < 14) & (mean > 120)
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8)) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8))
    if n <= 1:
        return np.zeros(mask.shape, bool)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    band = (lab == i) & mask
    tall = stats[i, cv2.CC_STAT_HEIGHT] > 0.35 * (y1 - y0)
    if not (tall and band.sum() > 0.04 * mask.sum()):
        return np.zeros(mask.shape, bool)
    # it must look unlike the shirt right beside it (a plain light shirt is smooth and light too)
    ring = cv2.dilate(band.astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool) & mask & ~band
    if ring.sum() < 500:
        return np.zeros(mask.shape, bool)
    from app import tryon_composite as tc
    texture_gap = std[ring].mean() > max(2.0 * std[band].mean(), 18.0)
    colour_gap = np.linalg.norm(tc._srgb_to_lab(photo[ring]).mean(0) - tc._srgb_to_lab(photo[band]).mean(0)) > 18
    return band if (texture_gap or colour_gap) else np.zeros(mask.shape, bool)


def close_front(photo: Image.Image, mask: np.ndarray, box) -> tuple[Image.Image, bool]:
    """Remove an inner T-shirt from the catalog photo by filling its band with shirt texture copied
    from just left of it (a check or plain fabric repeats), so the try-on shows the shirt closed."""
    a = np.asarray(photo).copy()
    band = under_layer(a, mask, box)
    if not band.any():
        return photo, False
    for y in np.nonzero(band.any(1))[0]:
        xs = np.nonzero(band[y])[0]
        x_a, x_b = xs.min(), xs.max()
        width = x_b - x_a + 1
        src = np.arange(x_a, x_b + 1) - width - 4
        ok = src >= 0
        src = np.where(ok, src, np.arange(x_a, x_b + 1) + width + 4)
        src = np.clip(src, 0, a.shape[1] - 1)
        a[y, x_a:x_b + 1] = a[y, src]
    return Image.fromarray(a), True
