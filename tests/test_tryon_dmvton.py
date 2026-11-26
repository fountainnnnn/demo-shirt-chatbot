"""DM-VTON backend (app/tryon_dmvton.py).

Fast: the pure-PyTorch correlation that replaces the cupy kernel, against a slow loop reference.
Slow (RUN_SLOW=1): one real try-on on a catalog photo (needs vendor/dm-vton and models/dmvton,
see scripts/fetch_dmvton.sh, plus the FASHN human parser).
"""
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app import tryon_dmvton  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _reference(a: np.ndarray, b: np.ndarray, md: int = 3) -> np.ndarray:
    """Direct transcription of kernel_Correlation_updateOutput (stride 1): channel c has
    x offset c % 7 - 3 and y offset c // 7 - 3; mean over channels; zero outside."""
    n, c, h, w = a.shape
    out = np.zeros((n, (2 * md + 1) ** 2, h, w))
    for ch in range((2 * md + 1) ** 2):
        dx, dy = ch % (2 * md + 1) - md, ch // (2 * md + 1) - md
        for y in range(h):
            for x in range(w):
                if 0 <= y + dy < h and 0 <= x + dx < w:
                    out[:, ch, y, x] = (a[:, :, y, x] * b[:, :, y + dy, x + dx]).sum(1) / c
    return out


def test_correlation_matches_cupy_kernel_semantics():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(2, 8, 7, 10, generator=g)
    b = torch.randn(2, 8, 7, 10, generator=g)
    got = tryon_dmvton.correlation(a, b).numpy()
    assert got.shape == (2, 49, 7, 10)
    np.testing.assert_allclose(got, _reference(a.double().numpy(), b.double().numpy()), atol=1e-5)


def test_correlation_shift_lands_in_expected_channel():
    # second = first shifted right by 2 and down by 1 -> peak at dx=+2, dy=+1
    a = torch.zeros(1, 1, 9, 9)
    a[0, 0, 4, 4] = 1.0
    b = torch.zeros(1, 1, 9, 9)
    b[0, 0, 5, 6] = 1.0
    out = tryon_dmvton.correlation(a, b)[0, :, 4, 4]
    assert int(out.argmax()) == (1 + 3) * 7 + (2 + 3)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="slow: set RUN_SLOW=1")
def test_run_tryon(tmp_path):
    if not (ROOT / "models" / "dmvton" / "dmvton_pf_warp.pt").exists():
        pytest.skip("weights missing: run scripts/fetch_dmvton.sh")
    person = ROOT / "data" / "catalog" / "images" / "14052.jpg"
    garment = ROOT / "data" / "catalog" / "images" / "11371.jpg"
    out = tmp_path / "out.png"
    try:
        r = tryon_dmvton.run_tryon(str(person), str(garment), str(out))
    finally:
        tryon_dmvton.unload()
    assert out.exists()
    assert r["model"] == "dm-vton" and r["composited"] is True
    assert r["device"] in ("mps", "cpu")
    assert r["seconds"] < 60
