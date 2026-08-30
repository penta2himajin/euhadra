#!/usr/bin/env bash
#
# Download the ONNX INT8 export of `nvidia/parakeet-tdt-0.6b-v3`
# (25 European languages including en/es) from
# `istupakov/parakeet-tdt-0.6b-v3-onnx`.
#
# Layout written to $PARAKEET_V3_DIR (default vendor/parakeet_v3):
#
#   encoder-model.int8.onnx   (~652 MB)
#   decoder_joint-model.int8.onnx
#   vocab.txt
#
# Matches what `parakeet-rs::ParakeetTDT::from_pretrained` looks for
# (INT8 preferred after FP32 filenames — so this script deliberately
# does *not* fetch the FP32 pair, which would otherwise win the search).
#
# Idempotent. Prefer an already-unpacked local copy when present.
#
# Usage:
#   scripts/setup_parakeet_v3.sh
#   PARAKEET_V3_DIR=/path scripts/setup_parakeet_v3.sh
#
# Licensing (informational — defer to upstream URLs for authoritative text;
# see docs/model-licenses.md for the consolidated table):
#   - nvidia/parakeet-tdt-0.6b-v3: CC-BY-4.0
#   - istupakov/parakeet-tdt-0.6b-v3-onnx: CC-BY-4.0 (inherited)
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${PARAKEET_V3_DIR:-$ROOT/vendor/parakeet_v3}"
HF="https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main"

mkdir -p "$DIR"

need() {
  local f="$1"
  [[ -s "$DIR/$f" ]] || return 1
  return 0
}

if need encoder-model.int8.onnx && need decoder_joint-model.int8.onnx && need vocab.txt; then
  echo "[skip] parakeet-v3 INT8 already present at $DIR"
  du -sh "$DIR"
  exit 0
fi

# Refuse a mixed FP32+INT8 tree: parakeet-rs prefers encoder-model.onnx
# over the INT8 file, which would silently load ~2.4 GB FP32.
if [[ -e "$DIR/encoder-model.onnx" || -e "$DIR/encoder-model.onnx.data" ]]; then
  echo "[error] $DIR contains FP32 encoder files; remove them so INT8 is selected" >&2
  exit 2
fi

require() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[error] required tool '$1' not on PATH" >&2
    exit 3
  fi
}
require curl

for f in vocab.txt decoder_joint-model.int8.onnx encoder-model.int8.onnx; do
  target="$DIR/$f"
  if [[ -s "$target" ]]; then
    echo "[skip] $f already present"
    continue
  fi
  echo "[get ] $f"
  curl -fL --retry 3 --retry-delay 2 --max-time 1800 \
    -o "$target.part" "$HF/$f"
  mv "$target.part" "$target"
done

echo "Done. Bundle at $DIR"
ls -lh "$DIR"
