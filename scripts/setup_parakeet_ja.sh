#!/usr/bin/env bash
#
# Download the ONNX-exported `nvidia/parakeet-tdt_ctc-0.6b-ja` model
# bundle (~2.4 GB) from the `sunilmahendrakar/parakeet-tdt-0.6b-ja-onnx`
# HuggingFace mirror. The mirror's directory layout matches what
# `parakeet-rs::ParakeetTDT::from_pretrained_with_feature_size` expects
# (encoder-model.onnx + .data, decoder_joint-model.onnx + .data,
# vocab.txt, config.json).
#
# Idempotent: skips files that already exist. Pass `PARAKEET_JA_DIR`
# to override the default location.
#
# Usage:
#   scripts/setup_parakeet_ja.sh
#   PARAKEET_JA_DIR=/path/to/dir scripts/setup_parakeet_ja.sh
#
# Why a separate ja repo: NVIDIA's flagship `parakeet-tdt-0.6b-v3` is a
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
#   - sunilmahendrakar/parakeet-tdt-0.6b-ja-onnx (ONNX mirror, what this
#     script actually downloads): CC-BY-4.0 (inherited)
#     Declaration: https://huggingface.co/sunilmahendrakar/parakeet-tdt-0.6b-ja-onnx
#       (model-card YAML: `license: cc-by-4.0`)
#   - Canonical CC-BY-4.0 text:
#     https://creativecommons.org/licenses/by/4.0/legalcode.en
#   Attribution required: credit NVIDIA, provide a link to the
#   CC-BY-4.0 text, and indicate modifications if any.

set -euo pipefail

DIR="${PARAKEET_JA_DIR:-vendor/parakeet_ja}"
HF_REPO="https://huggingface.co/sunilmahendrakar/parakeet-tdt-0.6b-ja-onnx/resolve/main"

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
for f in vocab.txt config.json encoder-model.onnx decoder_joint-model.onnx encoder-model.onnx.data decoder_joint-model.onnx.data; do
    target="$DIR/$f"
    if [[ -s "$target" ]]; then
        echo "[skip] $f already present"
        continue
    fi
    echo "[get] $f"
    curl -fL --retry 3 --retry-delay 2 --max-time 1200 \
        -o "$target" "$HF_REPO/$f"
done

# Sanity check: the encoder + its external weights file must both
# exist. parakeet-rs will SIGSEGV if .data is missing.
if [[ ! -s "$DIR/encoder-model.onnx" ]] || [[ ! -s "$DIR/encoder-model.onnx.data" ]]; then
    echo "[error] $DIR is missing encoder-model.onnx and/or encoder-model.onnx.data" >&2
    exit 4
fi

echo "PARAKEET_JA_DIR=$DIR"
