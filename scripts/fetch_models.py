"""Download the local chat and embedding models (GGUF) into models/.

Plain HTTPS from Hugging Face, no token needed. Resumes partial downloads,
retries on errors, and checks size + SHA-256 before accepting a file.

    .venv/bin/python scripts/fetch_models.py            # both models
    .venv/bin/python scripts/fetch_models.py --only embed
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "chat": {
        # Non-thinking instruct model; chosen over Qwen3.5-4B (slower hybrid arch, invents defaults).
        "repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        "file": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "size": 2497281120,
        "sha256": "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597",
        "dest": ROOT / "models" / "chat",
    },
    "embed": {
        "repo": "CompendiumLabs/bge-small-en-v1.5-gguf",
        "file": "bge-small-en-v1.5-f16.gguf",
        "size": 67308128,
        "sha256": "f0b2fef971e8366438bfd2d9aefea1b0115919389448806d290237f638bae999",
        "dest": ROOT / "models" / "embed",
    },
}

CHUNK = 1 << 20
RETRIES = 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK * 8), b""):
            h.update(block)
    return h.hexdigest()


def download(spec: dict) -> Path:
    spec["dest"].mkdir(parents=True, exist_ok=True)
    final = spec["dest"] / spec["file"]
    part = final.with_suffix(final.suffix + ".part")
    url = f"https://huggingface.co/{spec['repo']}/resolve/main/{spec['file']}"

    if final.exists() and final.stat().st_size == spec["size"]:
        print(f"{final.name}: present, verifying hash...")
        if sha256(final) == spec["sha256"]:
            print(f"{final.name}: ok")
            return final
        print(f"{final.name}: hash mismatch, downloading again")
        final.unlink()

    for attempt in range(1, RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > spec["size"]:
            part.unlink()
            have = 0
        if have == spec["size"]:
            break
        req = urllib.request.Request(url, headers={"User-Agent": "fetch-models/1.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and resp.status != 206:  # server ignored Range: start over
                    have = 0
                    part.unlink(missing_ok=True)
                mode = "ab" if have else "wb"
                done, last = have, 0.0
                with part.open(mode) as out:
                    while True:
                        block = resp.read(CHUNK)
                        if not block:
                            break
                        out.write(block)
                        done += len(block)
                        if time.time() - last > 2:
                            pct = 100 * done / spec["size"]
                            print(f"\r{final.name}: {done/1e6:,.0f}/{spec['size']/1e6:,.0f} MB ({pct:.1f}%)",
                                  end="", flush=True)
                            last = time.time()
            print()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            print(f"\n{final.name}: attempt {attempt} failed ({e}); retrying")
            time.sleep(min(2 * attempt, 15))
            continue
        if part.stat().st_size == spec["size"]:
            break
        print(f"{final.name}: short read ({part.stat().st_size} bytes), resuming")
    else:
        sys.exit(f"{final.name}: gave up after {RETRIES} attempts")

    print(f"{final.name}: verifying hash...")
    digest = sha256(part)
    if digest != spec["sha256"]:
        part.unlink()
        sys.exit(f"{final.name}: SHA-256 mismatch ({digest}); deleted, run again")
    part.rename(final)
    print(f"{final.name}: ok ({spec['size']/1e6:,.0f} MB)")
    return final


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=sorted(MODELS), help="fetch just one model")
    args = ap.parse_args()
    for name, spec in MODELS.items():
        if args.only and name != args.only:
            continue
        download(spec)


if __name__ == "__main__":
    main()
