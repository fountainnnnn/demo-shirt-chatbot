#!/usr/bin/env bash
# Fetch DM-VTON code (vendor/dm-vton) and the pretrained student checkpoints (models/dmvton).
# Code and weights: CC BY-NC-SA 4.0, "for academic purposes only" (see vendor/dm-vton/README.md).
# No CUDA/cupy needed: app/tryon_dmvton.py replaces the cupy correlation at import time.
set -euo pipefail
cd "$(dirname "$0")/.."

REF="${DMVTON_REF:-main}"
WARP_ID="16H16AhGJrqndDrAaHWZvTFzGSsaDecpS"   # dmvton_pf_warp.pt (README "pretrained checkpoints" Drive folder
GEN_ID="1i-Ple5L9__LV_Fbx1cZyxiOC-CRFxi0C"    # dmvton_pf_gen.pt   1wfWGsR0vWC5LrA26xhj92ec_GoCKV80A)
WARP_SHA="0e2fb5220f0e6829a17de8894c56b09463e08f279aa4ac410bf808bc6287b74e"
GEN_SHA="3147a3731e5fc18c473f563a4d896a5ddf620f0a7f293b8fbda3069a76e8e6f7"

tmp="$(mktemp -d "${TMPDIR:-/tmp}/dmvton.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

if [ ! -f vendor/dm-vton/pipelines/dmvton_pipeline.py ]; then
  curl -fsSL --retry 5 -o "$tmp/dmvton.tar.gz" "https://codeload.github.com/KiseKloset/DM-VTON/tar.gz/refs/heads/$REF"
  rm -rf vendor/dm-vton && mkdir -p vendor/dm-vton
  tar xzf "$tmp/dmvton.tar.gz" -C vendor/dm-vton --strip-components=1
fi

fetch() {  # id name sha
  local out="models/dmvton/$2"
  if [ -f "$out" ] && [ "$(shasum -a 256 "$out" | cut -d' ' -f1)" = "$3" ]; then return; fi
  curl -fsSL --retry 5 -C - -o "$tmp/$2" "https://drive.usercontent.google.com/download?id=$1&export=download&confirm=t"
  local got; got="$(shasum -a 256 "$tmp/$2" | cut -d' ' -f1)"
  if [ "$got" != "$3" ]; then
    echo "checksum mismatch for $2 ($got). If Drive returned an HTML page, open" >&2
    echo "  https://drive.google.com/drive/folders/1wfWGsR0vWC5LrA26xhj92ec_GoCKV80A" >&2
    echo "and save $2 to $out by hand." >&2
    exit 1
  fi
  mv "$tmp/$2" "$out"
}
mkdir -p models/dmvton
fetch "$WARP_ID" dmvton_pf_warp.pt "$WARP_SHA"
fetch "$GEN_ID" dmvton_pf_gen.pt "$GEN_SHA"
ls -la models/dmvton
