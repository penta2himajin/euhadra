#!/usr/bin/env bash
#
# Download `nvidia/parakeet-tdt_ctc-0.6b-ja` as ONNX.
#
# Default: the INT8 build (~885 MB) from
# `penta2himajin/parakeet-tdt_ctc-0.6b-ja-onnx-int8`, verified against its
# SHA256SUMS. Only the encoder's MatMuls are quantised, which keeps the
# FP32 CER (FLEURS-ja test 0.0559 vs 0.0566) at ~1.9× the speed; a build
# that quantises every op loses 35 % (#173). Built reproducibly by
# `scripts/quantize_parakeet_ja.py`.
#
#   encoder-model.int8.onnx + .data      (INT8 MatMuls)
#   decoder_joint-model.onnx + .data     (FP32)
#   vocab.txt, config.json, SHA256SUMS
#
# `--fp32`: the FP32 export (~2.5 GB) from
# `sunilmahendrakar/parakeet-tdt-0.6b-ja-onnx` instead — the input of
# `quantize_parakeet_ja.py`. Default dir vendor/parakeet_ja_fp32.
#
# Both sources are pinned to a revision. Idempotent: skips files that
# already exist.
#
# Usage:
#   scripts/setup_parakeet_ja.sh
#   scripts/setup_parakeet_ja.sh --fp32
#   PARAKEET_JA_DIR=/path/to/dir scripts/setup_parakeet_ja.sh
#
# Why a separate ja model: NVIDIA's flagship `parakeet-tdt-0.6b-v3` is a
# multilingual European model — its vocabulary doesn't include Japanese
# (verified empirically: it outputs Latin-script romanisation when fed
# Japanese audio). The dedicated `parakeet-tdt_ctc-0.6b-ja` (Hybrid
# TDT-CTC, 80-mel preprocessor, ReazonSpeech-trained) achieves
# ~3–9% CER on FLEURS / JSUT vs whisper-tiny's ~42% on the same audio.
#
# Licensing (informational — defer to upstream URLs for authoritative text;
# see docs/model-licenses.md for the consolidated table):
#   - nvidia/parakeet-tdt_ctc-0.6b-ja: CC-BY-4.0
#     Declaration: https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja
#       (model-card YAML: `license: cc-by-4.0`)
#   - sunilmahendrakar/parakeet-tdt-0.6b-ja-onnx (FP32 ONNX export):
#     CC-BY-4.0 (inherited)
#   - penta2himajin/parakeet-tdt_ctc-0.6b-ja-onnx-int8 (INT8 build of the
#     above; modifications listed in its model card): CC-BY-4.0 (inherited)
#   - Canonical CC-BY-4.0 text:
#     https://creativecommons.org/licenses/by/4.0/legalcode.en
#   Attribution required: credit NVIDIA, provide a link to the
#   CC-BY-4.0 text, and indicate modifications if any.

set -euo pipefail

if [[ "${1:-}" == "--fp32" ]]; then
    DIR="${PARAKEET_JA_DIR:-vendor/parakeet_ja_fp32}"
    HF_REPO="https://huggingface.co/sunilmahendrakar/parakeet-tdt-0.6b-ja-onnx/resolve/90b7d30b38ba48adf897d44b4f2e39894751b112"
    FILES=(vocab.txt config.json encoder-model.onnx decoder_joint-model.onnx encoder-model.onnx.data decoder_joint-model.onnx.data)
    ENCODER=encoder-model.onnx
else
    DIR="${PARAKEET_JA_DIR:-vendor/parakeet_ja}"
    HF_REPO="https://huggingface.co/penta2himajin/parakeet-tdt_ctc-0.6b-ja-onnx-int8/resolve/6eab63685c4f2a879e93b2e5fb6e6accb479cbc7"
    FILES=(SHA256SUMS vocab.txt config.json encoder-model.int8.onnx decoder_joint-model.onnx encoder-model.int8.onnx.data decoder_joint-model.onnx.data)
    ENCODER=encoder-model.int8.onnx
    # parakeet-rs prefers encoder-model.onnx over the INT8 file, so a
    # leftover FP32 encoder would silently load ~2.5 GB of FP32.
    if [[ -e "$DIR/encoder-model.onnx" || -e "$DIR/encoder-model.onnx.data" ]]; then
        echo "[error] $DIR contains FP32 encoder files; remove them so INT8 is selected" >&2
        exit 2
    fi
fi

mkdir -p "$DIR"

require() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[error] required tool '$1' not on PATH" >&2
        exit 3
    fi
}
require curl

# Order matters: large `.data` files are most likely to fail on flaky
# networks; placing them later means small headers + vocab succeed
# first, leaving an obvious diagnostic if the big chunks fail.
for f in "${FILES[@]}"; do
    target="$DIR/$f"
    if [[ -s "$target" ]]; then
        echo "[skip] $f already present"
        continue
    fi
    echo "[get] $f"
    curl -fL --retry 3 --retry-delay 2 --max-time 1200 \
        -o "$target.part" "$HF_REPO/$f"
    mv "$target.part" "$target"
done

# Sanity check: the encoder + its external weights file must both
# exist. parakeet-rs will SIGSEGV if .data is missing.
if [[ ! -s "$DIR/$ENCODER" ]] || [[ ! -s "$DIR/$ENCODER.data" ]]; then
    echo "[error] $DIR is missing $ENCODER and/or $ENCODER.data" >&2
    exit 4
fi

if [[ -s "$DIR/SHA256SUMS" ]]; then
    require shasum
    (cd "$DIR" && shasum -a 256 -c SHA256SUMS)
fi

echo "PARAKEET_JA_DIR=$DIR"
