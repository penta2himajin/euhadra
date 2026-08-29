#!/usr/bin/env bash
#
# Download the sherpa-onnx Zipformer export of ReazonSpeech k2 v2 (ja/en)
# used as the Japanese shipping ASR path.
#
# Layout written to $REAZON_JA_DIR (default vendor/reazon_ja):
#
#   encoder.int8.onnx   ← encoder-epoch-35-avg-1.int8.onnx
#   decoder.int8.onnx
#   joiner.int8.onnx
#   tokens.txt
#
# Source: k2-fsa/sherpa-onnx asr-models release
#   sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17.tar.bz2
# Upstream weights: reazon-research/reazonspeech-k2-v2 (Apache-2.0)
#
# Idempotent. Prefer an already-unpacked hayamimi vendor copy when present
# to avoid a second multi-hundred-MB download in this environment.
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${REAZON_JA_DIR:-$ROOT/vendor/reazon_ja}"
NAME="sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17"
GH="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${NAME}.tar.bz2"
HAYAMIMI_COPY="$ROOT/vendor/hayamimi/$NAME"

mkdir -p "$DIR"

need() {
  local f="$1"
  [[ -f "$DIR/$f" ]] || return 1
  return 0
}

if need encoder.int8.onnx && need decoder.int8.onnx && need joiner.int8.onnx && need tokens.txt; then
  echo "[skip] reazon-ja already present at $DIR"
  du -sh "$DIR"
  exit 0
fi

SRC=""
if [[ -d "$HAYAMIMI_COPY" ]]; then
  echo "[use ] local $HAYAMIMI_COPY"
  SRC="$HAYAMIMI_COPY"
else
  TMP="$DIR/.${NAME}.tar.bz2.part"
  echo "[get ] $GH"
  curl -fL --retry 3 --retry-delay 2 -o "$TMP" "$GH"
  echo "  extracting…"
  mkdir -p "$DIR/.extract"
  tar -xjf "$TMP" -C "$DIR/.extract"
  SRC="$(find "$DIR/.extract" -maxdepth 1 -type d -name 'sherpa-onnx-zipformer*' | head -1)"
  rm -f "$TMP"
fi

cp -f "$SRC/encoder-epoch-35-avg-1.int8.onnx" "$DIR/encoder.int8.onnx"
cp -f "$SRC/decoder-epoch-35-avg-1.int8.onnx" "$DIR/decoder.int8.onnx"
cp -f "$SRC/joiner-epoch-35-avg-1.int8.onnx" "$DIR/joiner.int8.onnx"
cp -f "$SRC/tokens.txt" "$DIR/tokens.txt"
rm -rf "$DIR/.extract"

echo "Done. Bundle at $DIR"
ls -lh "$DIR"
