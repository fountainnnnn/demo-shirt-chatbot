"""Slow try-on tests (load the real FASHN model). Run with: RUN_SLOW=1 .venv/bin/pytest tests/test_tryon.py (local FASHN backend, app/tryon_fashn.py)"""

import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="slow: set RUN_SLOW=1"),
]

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data" / "catalog" / "images"
PERSON = IMAGES / "14052.jpg"
GARMENT = IMAGES / "30802.jpg"


@pytest.fixture(scope="module")
def tryon():
    if not (ROOT / "models" / "fashn" / "model.safetensors").exists():
        pytest.skip("weights missing: run scripts/fetch_tryon_model.sh")
    from app import tryon_fashn as tryon

    yield tryon
    tryon.unload()


def test_check_photo_accepts_catalog_model(tryon):
    assert tryon.check_photo(str(PERSON)) == {"ok": True, "issues": []}


def test_check_photo_rejects_blank(tryon, tmp_path):
    from PIL import Image

    blank = tmp_path / "blank.jpg"
    Image.new("RGB", (768, 1024), "white").save(blank)
    res = tryon.check_photo(str(blank))
    assert res["ok"] is False
    assert "no person found" in res["issues"]


def test_run_tryon_fast(tryon, tmp_path):
    import numpy as np
    from PIL import Image

    out = tmp_path / "out.png"
    res = tryon.run_tryon(str(PERSON), str(GARMENT), str(out), quality="fast")
    assert res["device"] in {"mps", "cuda", "cpu"}
    assert res["seconds"] > 0
    arr = np.asarray(Image.open(out).convert("RGB"), dtype=np.float32)
    assert arr.shape[0] > 256 and arr.shape[1] > 256
    assert np.isfinite(arr).all()
    assert arr.std() > 10, "output looks blank"
    # The white shirt should turn light blue: mean blue > mean red over the torso.
    h, w = arr.shape[:2]
    torso = arr[int(h * 0.4) : int(h * 0.6), int(w * 0.4) : int(w * 0.6)]
    assert torso[..., 2].mean() > torso[..., 0].mean()
