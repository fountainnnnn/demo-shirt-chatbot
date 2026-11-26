"""Try-on backend dispatcher (contract: docs/contracts.md, app/tryon.py).

    check_photo(path) -> {"ok": bool, "issues": [str]}
    run_tryon(person_path, garment_path, out_path, quality="balanced") -> dict
        # {"seconds", "device", ...}; hosted adds "model" and "cost_usd"
    unload()
    provider() -> str

TRYON_PROVIDER:
    dmvton (default)      local DM-VTON, app/tryon_dmvton.py (small, MPS/CPU; CC BY-NC-SA 4.0, academic use only)
    openrouter            hosted image model, app/tryon_openrouter.py (paid; sends the photo to OpenRouter)
    mobilevton            local Mobile-VTON, app/tryon_mobilevton.py (sharper, about 11 s here, 3.5 GB; CC BY-NC-SA 4.0)
    fashn                 local FASHN VTON v1.5, app/tryon_fashn.py (heavy: torch + 2 GB weights)
    oxygen                planned strong-GPU backend, not built (see docs/architecture.md, "Try-on")
"""
from __future__ import annotations

import os

from app import models  # noqa: F401  (loads .env)

PROVIDERS = ("openrouter", "fashn", "dmvton", "mobilevton", "oxygen")


def provider() -> str:
    p = os.environ.get("TRYON_PROVIDER", "dmvton").strip().lower() or "dmvton"
    if p not in PROVIDERS:
        raise ValueError(f"TRYON_PROVIDER must be one of {PROVIDERS}")
    return p


def _impl():
    p = provider()
    if p == "openrouter":
        from app import tryon_openrouter as impl
        return impl
    if p == "dmvton":
        from app import tryon_dmvton as impl
        return impl
    if p == "mobilevton":
        from app import tryon_mobilevton as impl
        return impl
    if p == "fashn":
        from app import tryon_fashn as impl
        return impl
    raise NotImplementedError(
        f"TRYON_PROVIDER={p} is a documented option that is not built yet "
        "(oxygen: Oxygen-TryOn for a strong GPU). See docs/architecture.md, 'Try-on'. "
        "Use TRYON_PROVIDER=dmvton, mobilevton, openrouter or fashn."
    )


def check_photo(path: str) -> dict:
    return _impl().check_photo(path)


def run_tryon(person_path: str, garment_path: str, out_path: str, quality: str = "balanced",
              garment: dict | None = None, on_stage=None) -> dict:
    """`garment` (the catalog item) lets the hosted model know which garment in the photo is meant;
    `on_stage(text)` reports the real stage in plain words."""
    p = provider()
    if p in ("openrouter", "dmvton", "mobilevton"):
        return _impl().run_tryon(person_path, garment_path, out_path, quality=quality, garment=garment,
                                 on_stage=on_stage)
    return _impl().run_tryon(person_path, garment_path, out_path, quality=quality)


def unload() -> None:
    try:
        _impl().unload()
    except NotImplementedError:
        pass


def model_name() -> str:
    p = provider()
    if p == "openrouter":
        from app import tryon_openrouter
        return tryon_openrouter.model_name()
    return {"dmvton": "dm-vton", "mobilevton": "mobile-vton", "fashn": "fashn-vton-1.5"}.get(p, p)


def device() -> str:
    """Where try-on runs: 'openrouter', or the local torch device ('mps' / 'cpu'), without loading the model."""
    p = provider()
    if p == "openrouter":
        return "openrouter"
    try:
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        return "cpu"
