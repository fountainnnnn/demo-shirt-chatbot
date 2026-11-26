"""Virtual try-on with FASHN VTON v1.5, run locally.

Contract (docs/contracts.md):
    check_photo(path) -> {"ok": bool, "issues": [str]}
    run_tryon(person_path, garment_path, out_path, quality="balanced")
        -> {"seconds": float, "device": "mps" | "cuda" | "cpu"}
    unload()

Weights come from scripts/fetch_tryon_model.sh (models/fashn + models/hf).
Nothing here touches the network: the human parser is loaded from the local
Hugging Face cache by path.

Device: Apple GPU (MPS) when available, then CUDA, then CPU. Override with
TRYON_DEVICE=cpu|mps|cuda. MPS runs in float32 (FASHN only uses bf16 on CUDA).
"""

from __future__ import annotations

import gc
import os
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "models" / "fashn"
HF_HOME = ROOT / "models" / "hf"
PARSER_REPO = "models--fashn-ai--fashn-human-parser"

# Diffusion steps per quality level, picked from timings on an M-series Mac
# (see the try-on spike notes). FASHN's own guide: 20 fast / 30 balanced / 50 quality.
STEPS = {"fast": 12, "balanced": 20, "best": 30}

_lock = threading.Lock()
_pipe = None
_device: str | None = None
_pose = None  # light DWPose-only detector for check_photo when the full model isn't loaded


def _pick_device() -> str:
    import torch

    forced = os.environ.get("TRYON_DEVICE")
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _parser_dir() -> str:
    snaps = sorted((HF_HOME / "hub" / PARSER_REPO / "snapshots").glob("*/config.json"))
    if not snaps:
        raise FileNotFoundError(f"Human parser weights missing under {HF_HOME}; run scripts/fetch_tryon_model.sh")
    return str(snaps[-1].parent)


def _patch_rope_for_mps() -> None:
    """FASHN's rope() builds a float64 tensor on the model's device; MPS has no float64.
    Same maths in float32, swapped in at module level (no vendor edit)."""
    import torch
    from einops import rearrange
    from fashn_vton import tryon_mmdit

    if getattr(tryon_mmdit.rope, "_mps_safe", False):
        return

    def rope(pos, dim, theta):
        assert dim % 2 == 0
        scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
        omega = 1.0 / (theta**scale)
        out = torch.einsum("...n,d->...nd", pos.float(), omega)
        out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
        out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
        return out.float()

    rope._mps_safe = True
    tryon_mmdit.rope = rope


def _load():
    global _pipe, _device
    if _pipe is not None:
        return _pipe
    import torch
    from fashn_human_parser import FashnHumanParser
    from fashn_vton import TryOnPipeline

    device = _pick_device()
    if device == "mps":
        _patch_rope_for_mps()
    hp_device = "cpu" if device == "cpu" else device

    class _Pipeline(TryOnPipeline):
        def _setup_hp_model(self):
            # Load from the local cache by path so nothing is fetched at runtime.
            self.hp_model = FashnHumanParser(model_id=_parser_dir(), device=hp_device)

    _pipe = _Pipeline(weights_dir=str(WEIGHTS_DIR), device=device)
    _device = device
    torch.set_grad_enabled(False)
    return _pipe


def load() -> str:
    """Load the model now (otherwise the first run_tryon does it). Returns the device."""
    with _lock:
        _load()
        return _device


def _detector():
    if _pipe is not None:
        return _pipe.pose_model.pose_estimation
    global _pose
    if _pose is None:
        from fashn_vton.dwpose.wholebody import Wholebody

        _pose = Wholebody(checkpoints_dir=str(WEIGHTS_DIR / "dwpose"), device="cpu")
    return _pose


# OpenPose body indices after FASHN's reorder: 1 neck, 2 R shoulder, 5 L shoulder,
# 3/6 elbows, 8 R hip, 11 L hip.
_VIS = 0.3


def check_photo(path: str) -> dict:
    """Cheap pre-flight check with the DWPose models that ship with FASHN (CPU, ~1 s)."""
    import numpy as np
    from PIL import Image, ImageOps
    from fashn_vton.dwpose.onnxdet import inference_detector
    from fashn_vton.dwpose.onnxpose import inference_pose

    issues: list[str] = []
    try:
        with Image.open(path) as im:
            img = np.array(ImageOps.exif_transpose(im).convert("RGB"))
    except Exception:
        return {"ok": False, "issues": ["could not read the image"]}

    h, w = img.shape[:2]
    if min(h, w) < 256:
        issues.append("photo is too small (use at least 512 px on the short side)")

    det = _detector()
    bgr = np.ascontiguousarray(img[..., ::-1])
    # ONNX Runtime sessions are safe to call from several threads, so no lock here:
    # a photo check must not wait for a try-on that is running.
    boxes = inference_detector(det.session_det, bgr)
    boxes = np.asarray(boxes).reshape(-1, 4)
    if len(boxes) == 0:
        return {"ok": False, "issues": issues + ["no person found"]}

    # Ignore small background people: count boxes at least 30% of the biggest one's area.
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    main = int(np.argmax(areas))
    if int((areas >= 0.3 * areas[main]).sum()) > 1:
        issues.append("more than one person in the photo")

    _, scores = inference_pose(det.session_pose, boxes[main : main + 1], bgr)
    s = scores[0]
    # mmpose (COCO wholebody) order: 5/6 shoulders, 11/12 hips, 0 nose
    shoulders = s[5] > _VIS and s[6] > _VIS
    hips = s[11] > _VIS or s[12] > _VIS
    if not shoulders:
        issues.append("shoulders not visible (face the camera, upper body in frame)")
    elif not hips:
        issues.append("upper body not fully visible (show down to the waist)")

    box = boxes[main]
    if (box[3] - box[1]) < 0.3 * h:
        issues.append("person is too small in the frame (move closer)")

    return {"ok": not issues, "issues": issues}


def run_tryon(person_path: str, garment_path: str, out_path: str, quality: str = "balanced") -> dict:
    """Dress the person in the garment (category tops) and save a PNG/JPEG to out_path."""
    if quality not in STEPS:
        raise ValueError(f"quality must be one of {sorted(STEPS)}")
    from PIL import Image, ImageOps

    def _open(p):
        with Image.open(p) as im:
            return ImageOps.exif_transpose(im).convert("RGB")

    person, garment = _open(person_path), _open(garment_path)
    with _lock:
        t0 = time.perf_counter()
        pipe = _load()
        result = pipe(
            person_image=person,
            garment_image=garment,
            category="tops",
            garment_photo_type="model",  # catalog photos are on-model
            num_timesteps=STEPS[quality],
            seed=42,
        )
        if _device == "mps":
            import torch

            torch.mps.synchronize()
        seconds = time.perf_counter() - t0
        device = _device
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(out_path)
    return {"seconds": round(seconds, 2), "device": device}


def unload() -> None:
    """Free the try-on model (keeps the small CPU pose detector if check_photo built one)."""
    global _pipe, _device
    with _lock:
        _pipe = None
        _device = None
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
