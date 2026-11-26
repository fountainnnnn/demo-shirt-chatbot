"""Local try-on with Mobile-VTON (CVPR 2026, "High-Fidelity On-Device Virtual Try-On"), on Apple GPU / CPU.

    check_photo(path) -> {"ok": bool, "issues": [str]}
    run_tryon(person_path, garment_path, out_path, quality="balanced", garment=None, on_stage=None)
        -> {"seconds", "device", "model": "mobile-vton", "composited", "aligned", "iou", "stages", "timings", ...}
    unload()

Code: vendor/mobile-vton (github.com/tmllab/2026_CVPR_Mobile-VTON), weights: models/mobile-vton/checkpoint
(huggingface.co/FlashStight/Mobile-VTON), both CC BY-NC-SA 4.0 (non-commercial). scripts/fetch_mobilevton.sh.

What the model needs (read from the vendor inference.py / pipeline, not the README): the person image,
the garment image at the SAME size (the two latents are concatenated along the height), a caption
"Replace the upper body with ..." for the try-on net, the garment description for the garment net,
and the garment again through DINOv2 for the IP-adapter tokens. It is mask-free: no agnostic mask and
no DensePose are read anywhere in the inference code, so none are built here.

Port (vendor/ is unmodified):
  * diffusers 0.43 moved MultiControlNetModel; it is re-exported under the old name before import
    (only ip_adapter/ip_adapter.py imports it, and only Resampler is used from that file).
  * fp16 weights on MPS instead of bf16 autocast on CUDA; the small VAE decoder stays fp32 as in the
    vendor script.

Pipeline (reuses app/tryon_composite.py; nothing there is modified):
  1. parse the original photo (FASHN human parser)
  2. 3:4 crop around top + arms (torso; default) or head to hips (viton); optional white background
     (white background broke the result: the model drew the garment over the head)
  3. garment: parser `top` mask of the catalog photo -> garment on white at the model size
     (or the raw catalog photo, MOBILEVTON_GARMENT=raw)
  4. Mobile-VTON at 768x1024 (or 576x768 / 384x512), N flow-matching Euler steps, CFG
  5. paste the crop back; parse the result; edit region = original top UNION generated top (if the
     body did not move), minus face/hair/hands/bag/belt/trousers of the ORIGINAL; feathered blend

Env: TRYON_MOBILEVTON_DEVICE (auto|mps|cpu), MOBILEVTON_DTYPE (fp16|bf16|fp32), MOBILEVTON_STEPS,
     MOBILEVTON_SIZE (1024|768|512 = height), MOBILEVTON_CFG, MOBILEVTON_SHIFT, MOBILEVTON_SEED,
     MOBILEVTON_FRAMING (torso|viton; torso is clearly better on the test photo), MOBILEVTON_WHITE_BG (1|0), MOBILEVTON_GARMENT (cutout|raw).
quality: fast = 8 steps at 768, balanced = 8 steps at 1024, high = 28 steps at 1024 (vendor default);
measured on M5 (busy machine): ~6 s / ~11 s / ~33-50 s of model time. 8 steps looked the same as 12 and 28
at 1024; 4 steps gave a wrong-colour sleeve; 768 degrades fine checks; 512 is unusable.
MOBILEVTON_STEPS / MOBILEVTON_SIZE override.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "mobile-vton"
CKPT = ROOT / "models" / "mobile-vton" / "checkpoint"
GARMENT_CACHE = ROOT / "data" / "tryon" / "mobilevton" / "garments"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"

QUALITY = {"fast": (8, 768), "balanced": (8, 1024), "high": (28, 1024)}
NEG_PROMPT = (
    "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, "
    "missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, "
    "mutated, ugly, disgusting, blurry, amputation, NSFW"
)  # verbatim from vendor inference.py

_lock = threading.Lock()
_pipe = None
_device = None
_dtype = None
_processor = None
last_debug: dict = {}


# ------------------------------------------------------------------ loading
def pick_device() -> str:
    import torch

    want = os.environ.get("TRYON_MOBILEVTON_DEVICE", "auto").strip().lower()
    if want in ("mps", "cpu"):
        return want
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _pick_dtype(device: str):
    import torch

    want = os.environ.get("MOBILEVTON_DTYPE", "fp16" if device == "mps" else "fp32").lower()
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[want]


def _import_vendor():
    import diffusers.pipelines.controlnet as _pc

    if not hasattr(_pc, "MultiControlNetModel"):
        from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
        _pc.MultiControlNetModel = MultiControlNetModel
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    import logging
    logging.getLogger("Mobile_VTON").setLevel(logging.ERROR)
    for name in ("Mobile_VTON.pipelines.tryon_pipeline_full_cat",):
        logging.getLogger(name).setLevel(logging.ERROR)
    from Mobile_VTON.models.autoencoders.vae import Decoder
    from Mobile_VTON.models.unets.unet_2d_condition_garment import UNet2DConditionModel as UnetGarment
    from Mobile_VTON.models.unets.unet_2d_condition_tryon import UNet2DConditionModel as UnetTryon
    from Mobile_VTON.pipelines.tryon_pipeline_full_cat import T2IMobilePipelineV1_3_NotLoadingT5_Decoder as Pipe
    logging.getLogger("Mobile_VTON.pipelines.tryon_pipeline_full_cat").setLevel(logging.ERROR)
    return Decoder, UnetGarment, UnetTryon, Pipe


def load():
    """Build the vendor pipeline exactly as vendor inference.py does (same classes, same subfolders)."""
    global _pipe, _device, _dtype, _processor
    with _lock:
        if _pipe is not None:
            return _pipe
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler
        from diffusers.models.autoencoders import AutoencoderKL
        from transformers import AutoImageProcessor, AutoModel, CLIPTextModelWithProjection, CLIPTokenizer

        if not (CKPT / "denoiser" / "diffusion_pytorch_model.safetensors").exists():
            raise FileNotFoundError(f"{CKPT} missing; run scripts/fetch_mobilevton.sh")
        Decoder, UnetGarment, UnetTryon, Pipe = _import_vendor()
        _device = pick_device()
        _dtype = _pick_dtype(_device)
        c = str(CKPT)
        sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000,
                                                shift=float(os.environ.get("MOBILEVTON_SHIFT", "3.0")))
        tok1 = CLIPTokenizer.from_pretrained(c, subfolder="tokenizer")
        tok2 = CLIPTokenizer.from_pretrained(c, subfolder="tokenizer_2")
        te1 = CLIPTextModelWithProjection.from_pretrained(c, subfolder="text_encoder")
        te2 = CLIPTextModelWithProjection.from_pretrained(c, subfolder="text_encoder_2")
        vae = AutoencoderKL.from_pretrained(c, subfolder="vae")
        with open(CKPT / "vae_decoder" / "decoder.json") as f:
            vae_decoder = Decoder(**json.load(f))
        # plain state dict of tensors: loads with weights_only=True
        vae_decoder.load_state_dict(torch.load(CKPT / "vae_decoder" / "decoder.pt", map_location="cpu",
                                               weights_only=True), strict=True)
        image_encoder = AutoModel.from_pretrained(c, subfolder="image_encoder")
        denoiser = UnetTryon.from_pretrained(c, subfolder="denoiser")
        denoiser_garment = UnetGarment.from_pretrained(c, subfolder="denoiser_garment")
        for m in (vae, te1, te2, denoiser, denoiser_garment, image_encoder):
            m.to(_device, dtype=_dtype).eval().requires_grad_(False)
        vae_decoder.to(_device, dtype=torch.float32).eval().requires_grad_(False)
        _pipe = Pipe(vae=vae, vae_decoder=vae_decoder, scheduler=sched, tokenizer=tok1, tokenizer_2=tok2,
                     text_encoder=te1, text_encoder_2=te2, image_encoder=image_encoder,
                     denoiser=denoiser, denoiser_garment=denoiser_garment)
        _pipe.set_progress_bar_config(disable=True)
        _processor = AutoImageProcessor.from_pretrained(c, subfolder="image_encoder")
        torch.set_grad_enabled(False)
        return _pipe


def unload() -> None:
    global _pipe
    with _lock:
        _pipe = None
        try:
            import gc

            import torch
            gc.collect()
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


# ------------------------------------------------------------------ captions (style of txt_files/image_descriptions.txt)
def _catalog_item(garment_path: str) -> dict | None:
    try:
        stem = int(Path(garment_path).stem)
        items = json.loads(CATALOG.read_text())
        return next((i for i in items if i.get("id") == stem), None)
    except Exception:
        return None


def describe(item: dict | None) -> str:
    """One VITON-HD-style description ("The clothing item is a ...") from catalog fields."""
    if not item:
        return "The clothing item is a casual shirt with a classic collar and a button-down front."
    colour = item.get("colour", "")
    sleeve = {"long": "long-sleeve", "short": "short-sleeve"}.get(item.get("sleeve", ""), "")
    kind = "T-shirt" if item.get("type") == "tshirt" else "shirt"
    fabric = item.get("fabric", "cotton")
    pattern = item.get("pattern", "solid")
    pat = {
        "solid": f"It has a plain {colour} colour with no pattern or embellishments.",
        "checked": f"It features a fine {colour} and white checked pattern all over.",
        "striped": f"It features a {colour} striped pattern.",
        "printed": f"It features an all-over {colour} print.",
    }.get(pattern, "")
    front = ("It has a classic pointed collar and a button-down front."
             if kind == "shirt" else "It has a crew neckline.")
    style = item.get("usage", "casual")
    return (f"The clothing item is a {colour} {sleeve} {kind} made from {fabric} fabric. {front} {pat} "
            f"The style is {style}, suitable for everyday wear.").replace("  ", " ")


def captions(item: dict | None) -> tuple[str, str]:
    """(try-on caption, garment caption) exactly as vendor inference.py builds them."""
    desc = describe(item)
    return "Replace the upper body with " + " ".join(desc.split()[1:]), desc


# ------------------------------------------------------------------ garment image
def garment_image(garment_path: str, size: tuple[int, int], mode: str = "cutout") -> Image.Image:
    """VITON-HD `cloth`: the garment alone on white, centred, filling most of a 3:4 frame.
    mode="raw": the catalog photo itself, padded to 3:4 on white."""
    from app import tryon_composite as tc

    W, H = size
    if mode == "raw":
        photo = _open(garment_path)
        k = min(W / photo.width, H / photo.height)
        im = photo.resize((round(photo.width * k), round(photo.height * k)), Image.LANCZOS)
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        canvas.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
        return canvas
    p = Path(garment_path)
    cache = GARMENT_CACHE / f"{p.stem}_{int(p.stat().st_mtime)}_{W}x{H}.png"
    if cache.exists():
        return Image.open(cache).convert("RGB")
    photo = _open(garment_path)
    seg = tc.segment(photo, photo.size)
    mask = np.isin(seg, tc.GARMENT)
    if mask.sum() < 0.02 * mask.size:
        raise RuntimeError("no garment found in the catalog photo")
    m = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    soft = m.filter(ImageFilter.GaussianBlur(1.0))
    ys, xs = np.nonzero(np.asarray(m) > 127)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    cloth = Image.composite(photo, Image.new("RGB", photo.size, (255, 255, 255)), soft).crop(box)
    k = min(W * 0.9 / cloth.width, H * 0.9 / cloth.height)
    cloth = cloth.resize((max(1, round(cloth.width * k)), max(1, round(cloth.height * k))), Image.LANCZOS)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(cloth, ((W - cloth.width) // 2, (H - cloth.height) // 2))
    GARMENT_CACHE.mkdir(parents=True, exist_ok=True)
    canvas.save(cache)
    return canvas


# ------------------------------------------------------------------ person crop
def person_box(seg: np.ndarray, full_size: tuple[int, int], framing: str = "viton") -> tuple[int, int, int, int]:
    """3:4 box in full-image pixels. viton: head to a bit below the shirt hem (VITON-HD framing);
    torso: top + arms only (tc.crop_box, like the DM-VTON backend)."""
    from app import tryon_composite as tc

    if framing == "torso":
        box, _ = tc.crop_box(seg, full_size, {"3:4": 0.75}, expand=0.15)
        return box
    W, H = full_size
    sh, sw = seg.shape
    sx, sy = W / sw, H / sh
    body = np.isin(seg, (tc.TOP, tc.ARMS, tc.HANDS, tc.FACE, tc.HAIR, tc.DRESS))
    ys, xs = np.nonzero(body)
    if len(xs) == 0:
        return tc.crop_box(seg, full_size, {"3:4": 0.75})[0]
    tys = np.nonzero(np.isin(seg, (tc.TOP,)).any(1))[0]
    y0 = ys.min() * sy
    y1 = ((tys.max() + 1) if len(tys) else ys.max() + 1) * sy
    y0, y1 = y0 - 0.04 * (y1 - y0), y1 + 0.12 * (y1 - y0)
    x0, x1 = xs.min() * sx, (xs.max() + 1) * sx
    x0, x1 = x0 - 0.08 * (x1 - x0), x1 + 0.08 * (x1 - x0)
    h = y1 - y0
    w = x1 - x0
    if w / h < 0.75:
        w = h * 0.75
    else:
        h = w / 0.75
    if h > H:
        h, w = H, H * 0.75
    if w > W:
        w, h = W, W / 0.75
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = int(round(min(max(0, cx - w / 2), W - w)))
    y0 = int(round(min(max(0, cy - h / 2), H - h)))
    return x0, y0, int(round(x0 + w)), int(round(y0 + h))


# ------------------------------------------------------------------ model call
def settings(quality: str = "balanced") -> dict:
    steps, height = QUALITY.get(quality, QUALITY["balanced"])
    steps = int(os.environ.get("MOBILEVTON_STEPS", steps))
    height = int(os.environ.get("MOBILEVTON_SIZE", height))
    width = height * 3 // 4
    return {"steps": steps, "height": height, "width": width,
            "cfg": float(os.environ.get("MOBILEVTON_CFG", "2.5")),
            "seed": int(os.environ.get("MOBILEVTON_SEED", "42"))}


_res = (1024, 768)


def _set_resolution(height: int, width: int) -> None:
    """Rotary position tables for a height x width input, in BOTH nets.

    The constructors build a square 128-grid RoPE and resize it with the vendor's set_sample_size_rsd
    to the config size: 2048x768 for the try-on net (person and garment latents stacked), 1024x768
    for the garment net. The pipeline only calls the square set_sample_size(height // 8) on the
    try-on net when the size changes, which gives a table that matches neither (and never touches the
    garment net). Here: rebuild the 128 grid, resize it the same way the constructor does, then set
    sample_size so the pipeline skips its square reset."""
    global _res
    if (height, width) == _res:
        return
    h8, w8 = height // 8, width // 8
    for net, hh in ((_pipe.denoiser, 2 * h8), (_pipe.denoiser_garment, h8)):
        net.set_sample_size(128)
        net.set_sample_size_rsd(hh, w8)
        net.sample_size = h8
    _res = (height, width)


def infer(person_im: Image.Image, cloth_im: Image.Image, item: dict | None, steps: int, height: int,
          width: int, cfg: float = 2.5, seed: int = 42, timings: dict | None = None) -> Image.Image:
    """Run the vendor pipeline on prepared PIL inputs (person and cloth resized to width x height)."""
    import torch

    pipe = load()
    _set_resolution(height, width)
    tm = timings if timings is not None else {}
    sync = (lambda: torch.mps.synchronize()) if _device == "mps" else (lambda: None)
    person_im = person_im.resize((width, height), Image.BICUBIC)
    cloth_im = cloth_im.resize((width, height), Image.BICUBIC)
    cap, cap_c = captions(item)
    t = time.perf_counter()
    to_t = lambda im: torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)[None]
    cloth_pure = to_t(cloth_im) * 2 - 1                       # [-1, 1], as transforms.Normalize([.5],[.5])
    ip = _processor(images=cloth_im, return_tensors="pt").pixel_values
    pe, npe, ppe, nppe = pipe.encode_prompt(prompt=[cap], prompt_2=[cap], num_images_per_prompt=1,
                                            do_classifier_free_guidance=True, negative_prompt=[NEG_PROMPT],
                                            device=_device)
    pec, npec, _, _ = pipe.encode_prompt(prompt=[cap_c], prompt_2=[cap_c], num_images_per_prompt=1,
                                         do_classifier_free_guidance=False, negative_prompt=[NEG_PROMPT],
                                         device=_device)
    sync()
    tm["encode"] = time.perf_counter() - t
    gen = torch.Generator(device="cpu").manual_seed(seed)
    t = time.perf_counter()
    out = pipe(prompt_embeds=pe, negative_prompt_embeds=npe, pooled_prompt_embeds=ppe,
               negative_pooled_prompt_embeds=nppe, num_inference_steps=steps, generator=gen,
               height=height, width=width, guidance_scale=cfg, text_embeds_cloth=pec,
               negative_text_embeds_cloth=npec, cloth=cloth_pure.to(_device), image=to_t(person_im).to(_device),
               ip_adapter_image=ip.to(_device), device=torch.device(_device))[0]
    sync()
    tm["denoise_decode"] = time.perf_counter() - t
    return out[0]


# ------------------------------------------------------------------ public entry
def run_tryon(person_path: str, garment_path: str, out_path: str, quality: str = "balanced",
              garment: dict | None = None, on_stage=None) -> dict:
    import torch
    from app import tryon_composite as tc

    stage = on_stage or (lambda _s: None)
    t0 = time.perf_counter()
    timings: dict = {}
    s = settings(quality)

    t = time.perf_counter()
    load()
    timings["load"] = time.perf_counter() - t

    stage("Reading your photo")
    t = time.perf_counter()
    person = _open(person_path)
    size = tc.seg_size(*person.size)
    seg_orig = tc.segment(person, size)
    timings["parse"] = time.perf_counter() - t

    stage("Preparing the shirt")
    t = time.perf_counter()
    framing = os.environ.get("MOBILEVTON_FRAMING", "torso")
    box = person_box(seg_orig, person.size, framing)
    crop = person.crop(box)
    if _flag("MOBILEVTON_WHITE_BG", "0"):
        seg_full = np.asarray(Image.fromarray(seg_orig).resize(person.size, Image.NEAREST))
        a = np.asarray(crop).copy()
        a[seg_full[box[1]:box[3], box[0]:box[2]] == tc.BACKGROUND] = 255
        crop = Image.fromarray(a)
    gmode = os.environ.get("MOBILEVTON_GARMENT", "cutout")
    cloth = garment_image(garment_path, (s["width"], s["height"]), gmode)
    item = garment if garment else _catalog_item(garment_path)
    timings["prepare"] = time.perf_counter() - t

    stage("Fitting the shirt to your photo")
    gen = infer(crop, cloth, item, s["steps"], s["height"], s["width"], s["cfg"], s["seed"], timings)
    peak = None
    if _device == "mps":
        try:
            peak = round(torch.mps.driver_allocated_memory() / 2**30, 2)
        except Exception:
            pass

    stage("Blending it into your photo")
    t = time.perf_counter()
    bw, bh = box[2] - box[0], box[3] - box[1]
    full = person.copy()
    full.paste(gen.resize((bw, bh), Image.LANCZOS), box[:2])
    seg_gen = tc.segment(full, size)
    # alignment only inside the crop (outside it the two images are identical)
    sx, sy = size[0] / person.width, size[1] / person.height
    cb = (int(box[0] * sx), int(box[1] * sy), int(np.ceil(box[2] * sx)), int(np.ceil(box[3] * sy)))
    iou = tc.person_iou(seg_orig[cb[1]:cb[3], cb[0]:cb[2]], seg_gen[cb[1]:cb[3], cb[0]:cb[2]])
    aligned = iou >= tc.ALIGN_MIN
    region, protected = tc.edit_mask(seg_orig, seg_gen, aligned=aligned,
                                     untucked=os.environ.get("TRYON_UNTUCKED", "1") != "0")
    inside = np.zeros_like(region)
    inside[cb[1]:cb[3], cb[0]:cb[2]] = True
    result, m = tc.blend(person, full, region & inside, protected)
    timings["composite"] = time.perf_counter() - t

    global last_debug
    last_debug = {"person_in": crop, "cloth": cloth, "raw": gen, "box": box, "captions": captions(item)}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    return {
        "seconds": round(time.perf_counter() - t0, 2),
        "device": _device,
        "model": "mobile-vton",
        "composited": True,
        "aligned": bool(aligned),
        "iou": round(float(iou), 3),
        "stages": {"extract": gmode == "cutout", "crop": True, "composite": True},
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "settings": {**s, "dtype": str(_dtype).replace("torch.", ""), "framing": framing, "garment": gmode},
        "box": list(box),
        "peak_gb": peak,
    }
