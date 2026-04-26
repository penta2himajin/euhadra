#!/usr/bin/env bash
#
# Download the ONNX-exported `parakeet-tdt-0.6b-v3` model bundle
# (~2.4 GB FP32 + 622 MB int8 + 18 MB decoder) from the
# `istupakov/parakeet-tdt-0.6b-v3-onnx` HuggingFace mirror.
#
# Used as the en (and 25-language European multilingual) ASR backend
# in eval_l1_smoke when `--parakeet-en-dir` is given. The model uses
# the default 128-mel preprocessor — `ParakeetAdapter::load(dir)`
# (without feature_size override) loads it correctly.
#
# Idempotent: skips files that already exist. Override the install
# location with `PARAKEET_EN_DIR`. Pass `PARAKEET_EN_QUANTIZED=1` to
# skip the FP32 weights and pull the ~3× smaller int8 versions
# instead (some quality loss observed in practice — ~25% WER vs FP32's
# ~7.5% — so FP32 is the default).
#
# Usage:
#   scripts/setup_parakeet_en.sh
#   PARAKEET_EN_DIR=/path/to/dir scripts/setup_parakeet_en.sh

set -euo pipefail

DIR="${PARAKEET_EN_DIR:-vendor/parakeet_en}"
HF_REPO="https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main"

mkdir -p "$DIR"

require() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[error] required tool '$1' not on PATH" >&2
        exit 3
    fi
}
require curl

# Headers + small files first so missing downloads fail fast.
files=(vocab.txt config.json encoder-model.onnx decoder_joint-model.onnx)
if [[ "${PARAKEET_EN_QUANTIZED:-0}" == "1" ]]; then
    # int8 path: smaller, ~3× faster on CPU, but the int8 quantisation
    # noticeably hurts WER on this model — keep FP32 unless size is the
    # blocker.
    files+=(encoder-model.int8.onnx decoder_joint-model.int8.onnx)
else
    files+=(encoder-model.onnx.data)
fi

for f in "${files[@]}"; do
    target="$DIR/$f"
    if [[ -s "$target" ]]; then
        echo "[skip] $f already present"
        continue
    fi
    echo "[get] $f"
    curl -fL --retry 3 --retry-delay 2 --max-time 1200 \
        -o "$target" "$HF_REPO/$f"
done

# Sanity check: encoder-model.onnx and (when FP32) its external weights.
if [[ ! -s "$DIR/encoder-model.onnx" ]]; then
    echo "[error] $DIR/encoder-model.onnx missing or empty" >&2
    exit 4
fi
if [[ "${PARAKEET_EN_QUANTIZED:-0}" != "1" ]] && [[ ! -s "$DIR/encoder-model.onnx.data" ]]; then
    echo "[error] $DIR/encoder-model.onnx.data missing — FP32 encoder needs it" >&2
    exit 4
fi

echo "PARAKEET_EN_DIR=$DIR"
