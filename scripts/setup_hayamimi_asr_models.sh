#!/usr/bin/env bash
#
# Download the hayamimi (oboroge0/hayamimi) shipping ASR bundles used for
# the head-to-head speed comparison against euhadra's shipping models.
#
# Sources match hayamimi's `scripts/download_models.py` (k2-fsa/sherpa-onnx
# asr-models release). Dest: vendor/hayamimi/<name>/
#
# Usage:
#   scripts/setup_hayamimi_asr_models.sh
#   HAYAMIMI_DIR=/path scripts/setup_hayamimi_asr_models.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${HAYAMIMI_DIR:-$ROOT/vendor/hayamimi}"
GH="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
mkdir -p "$DIR"

fetch_tbz2() {
  local url="$1"
  local name="$2"
  local target="$DIR/$name"
  if [[ -d "$target" && -n "$(ls -A "$target" 2>/dev/null || true)" ]]; then
    echo "[skip] $name"
    return 0
  fi
  echo "[get ] $name"
  local tmp="$DIR/.${name}.tar.bz2.part"
  curl -fL --retry 3 --retry-delay 2 -o "$tmp" "$url"
  echo "  extracting -> $target"
  mkdir -p "$DIR"
  tar -xjf "$tmp" -C "$DIR"
  # tarball top-level dir should match $name; rename if upstream differs
  local top
  top="$(tar -tjf "$tmp" | head -1 | cut -d/ -f1)"
  if [[ "$top" != "$name" && -d "$DIR/$top" ]]; then
    rm -rf "$target"
    mv "$DIR/$top" "$target"
  fi
  rm -f "$tmp"
}

fetch_tbz2 \
  "$GH/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17.tar.bz2" \
  "sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17"

fetch_tbz2 \
  "$GH/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2" \
  "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"

fetch_tbz2 \
  "$GH/sherpa-onnx-paraformer-zh-int8-2025-10-07.tar.bz2" \
  "sherpa-onnx-paraformer-zh-int8-2025-10-07"

# IMPORTANT: hayamimi requires the 2024-07-17 SenseVoice export (2025-09-09 is broken).
fetch_tbz2 \
  "$GH/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2" \
  "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"

echo "Done. Models under $DIR"
du -sh "$DIR"/* 2>/dev/null || true
