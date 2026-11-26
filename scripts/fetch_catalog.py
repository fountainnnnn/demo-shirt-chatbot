"""Download ~100 sample shirts (photo + metadata) into data/catalog/ (metadata in raw.json; scripts/enrich_catalog.py builds catalog.json).

Source: Hugging Face dataset benitomartin/fashion-product-images-small-900x1200
(a re-hosted, upscaled copy of the Kaggle "Fashion Product Images" set).
"""
import json, pathlib, subprocess, urllib.parse, urllib.request

DS = "benitomartin/fashion-product-images-small-900x1200"
API = "https://datasets-server.huggingface.co/filter"
OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "catalog"
WANT = [("Shirts", 60), ("Tshirts", 40)]


def get(url):
    # curl handles flaky connections better than urllib here
    return subprocess.run(
        ["curl", "-sSfL", "--retry", "6", "--retry-all-errors", "--max-time", "90",
         "-A", "catalog-fetch", url],
        check=True, capture_output=True,
    ).stdout


def main():
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    items = []
    for article, n in WANT:
        got, offset = 0, 0
        while got < n:
            q = urllib.parse.urlencode({
                "dataset": DS, "config": "default", "split": "train",
                "where": f"\"articleType\"='{article}'",
                "offset": offset, "length": 100,
            })
            rows = json.loads(get(f"{API}?{q}"))["rows"]
            if not rows:
                break
            offset += len(rows)
            for r in rows[:: max(1, len(rows) // n)]:
                if got >= n:
                    break
                row = r["row"]
                img = row.pop("image")
                path = OUT / "images" / f"{row['id']}.jpg"
                if not path.exists():
                    path.write_bytes(get(img["src"]))
                row["image"] = f"images/{row['id']}.jpg"
                items.append(row)
                got += 1
    (OUT / "raw.json").write_text(json.dumps(items, indent=1))
    print(len(items), "items")


if __name__ == "__main__":
    main()
