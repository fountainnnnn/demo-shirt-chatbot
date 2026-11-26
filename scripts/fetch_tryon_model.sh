#!/usr/bin/env bash
# Install FASHN VTON v1.5 and download its weights (~2.2 GB) into the repo.
#   vendor/fashn-vton-1.5   source (Apache-2.0), pinned commit below
#   models/fashn/           model.safetensors + dwpose/*.onnx
#   models/hf/              Hugging Face cache (human parser, ~244 MB)
# Safe to re-run: skips what is already there, resumes partial downloads.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

FASHN_REF="${FASHN_REF:-main}"
VENDOR="$ROOT/vendor/fashn-vton-1.5"
export UV_CACHE_DIR="$ROOT/.uv/cache"
export HF_HOME="$ROOT/models/hf"
export HF_HUB_DISABLE_TELEMETRY=1
# Plain HTTPS downloads; the Xet transfer client fails behind some proxies.
export HF_HUB_DISABLE_XET=1
# Use certifi's CA bundle (the macOS keychain path fails inside sandboxes).
PY="$ROOT/.venv/bin/python"
export SSL_CERT_FILE="${SSL_CERT_FILE:-$("$PY" -c 'import certifi; print(certifi.where())')}"

# 1. Source
if [ ! -f "$VENDOR/pyproject.toml" ]; then
  mkdir -p "$ROOT/vendor"
  tmp="$(mktemp -d)"
  curl -fsSL --retry 5 -o "$tmp/fashn.tar.gz" \
    "https://codeload.github.com/fashn-AI/fashn-vton-1.5/tar.gz/$FASHN_REF"
  tar xzf "$tmp/fashn.tar.gz" -C "$tmp"
  mv "$tmp"/fashn-vton-1.5-* "$VENDOR"
  rm -rf "$tmp"
fi
for p in "$ROOT"/scripts/patches/*.diff; do
  [ -e "$p" ] || continue
  patch -d "$VENDOR" -p1 -N --silent < "$p" || true
done

# 2. Install. Upstream pins onnxruntime-gpu (CUDA only), so install deps by hand
#    with the CPU onnxruntime and the package itself without deps. Torch is left as is.
uv pip install -p "$ROOT/.venv" opencv-python einops onnxruntime matplotlib "fashn-human-parser>=0.1.1"
uv pip install -p "$ROOT/.venv" --no-deps -e "$VENDOR"

# 3. Weights (retry: large downloads sometimes drop)
mkdir -p "$ROOT/models/fashn" "$HF_HOME"
for i in 1 2 3 4 5; do
  if "$PY" "$VENDOR/scripts/download_weights.py" --weights-dir "$ROOT/models/fashn"; then
    break
  fi
  echo "download failed, retry $i" >&2
  sleep 5
done

# 4. Verify sizes (exact sizes on the Hub, checked 2026-11-26)
"$PY" - <<'EOF2'
import os, sys
from pathlib import Path
want = {
    "models/fashn/model.safetensors": 1943668048,
    "models/fashn/dwpose/yolox_l.onnx": 216746733,
    "models/fashn/dwpose/dw-ll_ucoco_384.onnx": 134399116,
}
parser = sorted(Path(os.environ["HF_HOME"], "hub", "models--fashn-ai--fashn-human-parser", "snapshots").glob("*/model.safetensors"))
if parser:
    want[str(parser[-1])] = 256146352
else:
    print("human parser: MISSING")
bad = []
for path, size in want.items():
    have = os.path.getsize(path) if os.path.exists(path) else 0
    print(f"{path}: {have} bytes {'ok' if have == size else f'EXPECTED {size}'}")
    if have != size:
        bad.append(path)
if bad or not parser:
    sys.exit("incomplete download; re-run this script")
print("FASHN VTON ready.")
EOF2
