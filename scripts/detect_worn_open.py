"""Propose catalog items whose photo shows the shirt worn OPEN over another top.

Writes data/catalog/worn_open_candidates.json for a person to review; the reviewed list is
data/catalog/worn_open.json (what the app reads). The detector is a candidate generator only:
on this catalog it proposed 8 items, of which 1 (13884) is really worn open; the rest are large
checks whose light squares look like an under-layer band.

The try-on uses the photo's garment, and the human parser labels the shirt and the T-shirt under
it both as `top`, so for these items the picture may include the layer underneath. Detection: a
tall, light, smooth band down the middle of the garment (app.tryon_dmvton.under_layer). Local
only (FASHN human parser). Run after enrich_catalog.py:

    .venv/bin/python scripts/detect_worn_open.py
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app import catalog, tryon_composite as tc, tryon_dmvton as td  # noqa: E402


def main():
    items = catalog.load()
    flagged = []
    for it in items:
        if it["type"] != "shirt":
            continue
        photo = td._open(str(catalog.CATALOG_DIR / it["image"]))
        mask = np.isin(tc.segment(photo, photo.size), tc.GARMENT)
        if mask.sum() == 0:
            continue
        ys, xs = np.nonzero(mask)
        band = td.under_layer(np.asarray(photo), mask, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        if band.any():
            flagged.append(it["id"])
    (catalog.CATALOG_DIR / "worn_open_candidates.json").write_text(json.dumps(sorted(flagged)) + "\n")
    print(f"worn_open candidates: {len(flagged)} of {len(items)} items: {flagged}")
    print("review them and list the real ones in data/catalog/worn_open.json")


if __name__ == "__main__":
    main()
