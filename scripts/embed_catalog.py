"""Embed every catalog description into data/catalog/embeddings.npy (+ embeddings_ids.json,
embeddings_meta.json naming the model that produced the vectors).

Uses app.models.embed (the active provider: OpenRouter or local). Catalog search
ignores the vectors when the stored model differs from the active one. Run after
enrich_catalog.py, whenever descriptions change, and after switching embedding model:

    .venv/bin/python scripts/embed_catalog.py
"""
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app import catalog, models  # noqa: E402


def main(batch: int = 32):
    items = catalog.load()
    texts = [i["description"] for i in items]
    parts = [np.asarray(models.embed(texts[s:s + batch]), dtype="float32") for s in range(0, len(texts), batch)]
    mat = np.vstack(parts)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.where(norms == 0, 1, norms)
    np.save(catalog.CATALOG_DIR / "embeddings.npy", mat)
    (catalog.CATALOG_DIR / "embeddings_ids.json").write_text(json.dumps([i["id"] for i in items]))
    meta = {"model": models.embed_model_name(), "dim": int(mat.shape[1]), "count": int(mat.shape[0]),
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    models.EMBED_META.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"embedded {mat.shape[0]} descriptions, dim {mat.shape[1]}, model {meta['model']}")


if __name__ == "__main__":
    main()
