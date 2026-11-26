#!/usr/bin/env bash
# Fetch Mobile-VTON (CVPR 2026) code + checkpoint for app/tryon_mobilevton.py.
#   code    -> vendor/mobile-vton      (github.com/tmllab/2026_CVPR_Mobile-VTON, CC BY-NC-SA 4.0)
#   weights -> models/mobile-vton/checkpoint  (huggingface.co/FlashStight/Mobile-VTON, CC BY-NC-SA 4.0)
# Non-commercial, academic use only. ~3.76 GB.
# Sizes and sha256 below come from the HF API file listing (LFS oid = sha256).
# Usage: scripts/fetch_mobilevton.sh [--mirror]   (--mirror uses AndrewChoyCS/Mobile-VTON)
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="FlashStight/Mobile-VTON"
[ "${1:-}" = "--mirror" ] && REPO="AndrewChoyCS/Mobile-VTON"
REV="main"
DEST="models/mobile-vton"
CURL=(curl -fL -C - --retry 8 --retry-all-errors --retry-delay 3 -sS)

# ---------------------------------------------------------------- code
if [ ! -f vendor/mobile-vton/inference.py ]; then
  mkdir -p vendor/mobile-vton
  tmp="$(mktemp -d)"
  curl -fsSL --retry 5 -o "$tmp/src.tar.gz" \
    https://codeload.github.com/tmllab/2026_CVPR_Mobile-VTON/tar.gz/refs/heads/main
  tar -xzf "$tmp/src.tar.gz" -C vendor/mobile-vton --strip-components=1
  find vendor/mobile-vton -name __pycache__ -prune -exec rm -rf {} +
  rm -rf "$tmp"
fi

# ---------------------------------------------------------------- weights
# path  size  sha256 ("-" = small file, size check only)
FILES=$(cat <<'EOF'
checkpoint/denoiser/config.json 4095 -
checkpoint/denoiser/diffusion_pytorch_model.safetensors 840173776 34ee9331e414a8d89ebd8c4c485094947f1fa710ef34d40a7726a623a7f3eca2
checkpoint/denoiser_garment/config.json 3970 -
checkpoint/denoiser_garment/diffusion_pytorch_model.safetensors 759200976 902e5f6d47e9aef909497d5d7d400dd51cb39764fac63aca1e5b730927e53184
checkpoint/image_encoder/config.json 917 -
checkpoint/image_encoder/model.safetensors 173185040 a39e985dd10fd1ae72faa49c35ad211a17dc0457ebe7dcbe7b32b92da4007506
checkpoint/image_encoder/preprocessor_config.json 436 -
checkpoint/model_index.json 890 -
checkpoint/scheduler/scheduler_config.json 414 -
checkpoint/text_encoder/config.json 650 -
checkpoint/text_encoder/model.safetensors 247324096 b4ddf499861727e83713b525ba1e087202f248a7ebb2268b04d1de09de5b67c2
checkpoint/text_encoder_2/config.json 648 -
checkpoint/text_encoder_2/model.safetensors 1389382688 c4ba596d8c78b5509aafd523f619e13aa129ae1a886a1e92766d7cddd9706f3c
checkpoint/tokenizer/merges.txt 524619 -
checkpoint/tokenizer/special_tokens_map.json 588 -
checkpoint/tokenizer/tokenizer_config.json 705 -
checkpoint/tokenizer/vocab.json 1059962 -
checkpoint/tokenizer_2/merges.txt 524619 -
checkpoint/tokenizer_2/special_tokens_map.json 576 -
checkpoint/tokenizer_2/tokenizer_config.json 856 -
checkpoint/tokenizer_2/vocab.json 1059962 -
checkpoint/vae/config.json 864 -
checkpoint/vae/diffusion_pytorch_model.safetensors 335306212 965f71f229a5981c0a025db771d8868998f5e4e8412dc5585c687f638fa59679
checkpoint/vae_decoder/decoder.json 1200 -
checkpoint/vae_decoder/decoder.pt 7644444 13ab9dec8e181b6d0e2efa3ecf466018b009bf5251c798f7aafc0b258106eb17
EOF
)

fsize() { stat -f %z "$1" 2>/dev/null || stat -c %s "$1"; }
fail=0
while read -r path size sha; do
  [ -z "$path" ] && continue
  out="$DEST/$path"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ "$(fsize "$out")" = "$size" ]; then
    :
  else
    [ -f "$out" ] && [ "$(fsize "$out")" -gt "$size" ] && rm -f "$out"
    echo "fetch $path ($size bytes)"
    for _ in 1 2 3; do
      "${CURL[@]}" -o "$out" "https://huggingface.co/$REPO/resolve/$REV/$path" || true
      [ "$(fsize "$out")" = "$size" ] && break
    done
  fi
  got="$(fsize "$out")"
  if [ "$got" != "$size" ]; then echo "SIZE MISMATCH $path: $got != $size"; fail=1; continue; fi
  if [ "$sha" != "-" ]; then
    h="$(shasum -a 256 "$out" | cut -d' ' -f1)"
    if [ "$h" != "$sha" ]; then echo "SHA256 MISMATCH $path"; rm -f "$out"; fail=1; continue; fi
  fi
done <<< "$FILES"

[ "$fail" = 0 ] || { echo "some files failed; re-run to resume"; exit 1; }
du -sh "$DEST"
echo "ok: Mobile-VTON checkpoint in $DEST/checkpoint"
