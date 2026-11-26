"""Mobile-VTON backend. Fast tests: captions, crop box. Slow test (RUN_SLOW=1): a real run on a catalog
photo used as the person (no personal photo in tests), needs models/mobile-vton (scripts/fetch_mobilevton.sh)."""
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app import tryon_mobilevton as mv

ROOT = Path(__file__).resolve().parent.parent
CAT = ROOT / "data" / "catalog" / "images"


def test_captions_follow_vendor_format():
    item = {"colour": "black", "sleeve": "long", "type": "shirt", "fabric": "cotton flannel",
            "pattern": "checked", "usage": "casual"}
    tryon, cloth = mv.captions(item)
    assert cloth.startswith("The clothing item is a black long-sleeve shirt")
    assert "checked" in cloth
    # vendor: "Replace the upper body with " + desc without its first word
    assert tryon == "Replace the upper body with " + " ".join(cloth.split()[1:])


def test_person_box_is_3_by_4_and_inside_image():
    from app import tryon_composite as tc

    seg = np.zeros((400, 300), np.uint8)
    seg[60:100, 120:180] = tc.FACE
    seg[100:260, 90:210] = tc.TOP
    seg[110:240, 70:90] = tc.ARMS
    for framing in ("viton", "torso"):
        x0, y0, x1, y1 = mv.person_box(seg, (1500, 2000), framing)
        assert 0 <= x0 < x1 <= 1500 and 0 <= y0 < y1 <= 2000
        assert abs((x1 - x0) / (y1 - y0) - 0.75) < 0.01


def test_settings_quality_presets(monkeypatch):
    monkeypatch.delenv("MOBILEVTON_STEPS", raising=False)
    monkeypatch.delenv("MOBILEVTON_SIZE", raising=False)
    s = mv.settings("fast")
    assert (s["height"], s["width"]) == (768, 576) and s["height"] % 8 == 0
    assert mv.settings("balanced")["height"] == 1024
    assert mv.settings("high")["steps"] == 28


@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="slow: set RUN_SLOW=1")
def test_run_tryon_end_to_end(tmp_path):
    if not (mv.CKPT / "denoiser" / "diffusion_pytorch_model.safetensors").exists():
        pytest.skip("weights missing: scripts/fetch_mobilevton.sh")
    person, garment = CAT / "11371.jpg", CAT / "13884.jpg"
    out = tmp_path / "out.png"
    os.environ.setdefault("MOBILEVTON_STEPS", "4")
    os.environ.setdefault("MOBILEVTON_SIZE", "512")
    r = mv.run_tryon(str(person), str(garment), str(out))
    assert out.exists() and r["model"] == "mobile-vton" and r["composited"]
    a = np.asarray(Image.open(person).convert("RGB"), np.int16)
    b = np.asarray(Image.open(out).convert("RGB"), np.int16)
    assert a.shape == b.shape
    changed = np.abs(a - b).max(-1) > 0
    assert 0.02 < changed.mean() < 0.6        # the shirt changed, the rest did not
    # top-left corner (background) is the original, pixel for pixel
    assert not changed[:100, :100].any()
    mv.unload()
