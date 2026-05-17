#!/usr/bin/env bash
#
# Download the `onnx-community/whisper-large-v3-turbo` bundle into the
# directory layout that `WhisperOnnxAdapter::load` expects:
#
#   <DIR>/
#     tokenizer.json                                 ← loaded by WhisperTokenizer
#     tokenizer_config.json                          ← supporting metadata
#     vocab.json / merges.txt / added_tokens.json    ← BPE artefacts
#     special_tokens_map.json                        ← `<|...|>` resolution
#     preprocessor_config.json                       ← mel config (cross-check)
#     generation_config.json / config.json           ← architecture refs
#     normalizer.json                                ← Whisper text normaliser
#     onnx/encoder_model_q4.onnx                     ← ~250 MB
#     onnx/decoder_model_q4.onnx                     ← ~330 MB
#     onnx/decoder_with_past_model_q4.onnx           ← ~330 MB
#
# Idempotent: skips files that already exist. Pass `WHISPER_ONNX_DIR`
# to override the default location and `WHISPER_ONNX_QUANT` to choose
# a quantisation suffix (default `q4` — best CER+RTF on CPU per the
# #83 bench; `int8` decoder collapses into repeating tokens, `fp16`
# load fails on ORT's CPU EP).
#
# Usage:
#   scripts/setup_whisper_onnx_turbo.sh
#   WHISPER_ONNX_DIR=/path/to/dir scripts/setup_whisper_onnx_turbo.sh
#   WHISPER_ONNX_QUANT=q4f16 scripts/setup_whisper_onnx_turbo.sh
#
# Why this bundle: `onnx-community/whisper-large-v3-turbo` is the
# official HF community export of `openai/whisper-large-v3-turbo` —
# same encoder/decoder weights, packaged as ONNX with multiple
# precomputed quantisation variants. The Korean ASR bench (issue #83,
# `docs/korean-asr-alternatives.md` §G) measured `q4` at CER 1.09% /
# RTF 0.484 on FLEURS-ko (4-core Xeon @ 2.1 GHz), beating CT2 FP16
# and whisper-rs Q4_0 on both axes.
#
# Licensing (informational — defer to upstream URLs for authoritative text;
# see docs/model-licenses.md for the consolidated table):
#   - openai/whisper-large-v3-turbo: MIT
#     Declaration: https://huggingface.co/openai/whisper-large-v3-turbo
#       (model-card YAML: `license: mit`)
#   - onnx-community/whisper-large-v3-turbo: MIT (inherited)
#     Declaration: https://huggingface.co/onnx-community/whisper-large-v3-turbo
#       (model-card YAML: `license: mit`)
#   - Canonical MIT text: https://opensource.org/licenses/MIT
#   No attribution required by the license itself, but OpenAI's
#   model card requests citation: "Robust Speech Recognition via
#   Large-Scale Weak Supervision" (Radford et al. 2022).

set -euo pipefail

DIR="${WHISPER_ONNX_DIR:-vendor/whisper_onnx_turbo}"
QUANT="${WHISPER_ONNX_QUANT:-q4}"
HF_REPO="https://huggingface.co/onnx-community/whisper-large-v3-turbo/resolve/main"

mkdir -p "$DIR/onnx"

require() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[error] required tool '$1' not on PATH" >&2
        exit 3
    fi
}
require curl

# Order matters: small tokenizer/config files first so a flaky
# network leaves an obvious diagnostic before the ~900 MB ONNX
# downloads kick in.
TEXT_FILES=(
    tokenizer.json
    tokenizer_config.json
    vocab.json
    merges.txt
    added_tokens.json
    special_tokens_map.json
    preprocessor_config.json
    generation_config.json
    config.json
    normalizer.json
)

ONNX_FILES=(
    "onnx/encoder_model_${QUANT}.onnx"
    "onnx/decoder_model_${QUANT}.onnx"
    "onnx/decoder_with_past_model_${QUANT}.onnx"
)

for f in "${TEXT_FILES[@]}" "${ONNX_FILES[@]}"; do
    target="$DIR/$f"
    if [[ -s "$target" ]]; then
        echo "[skip] $f already present"
        continue
    fi
    echo "[get] $f"
    curl -fL --retry 3 --retry-delay 2 --max-time 1200 \
        --create-dirs -o "$target" "$HF_REPO/$f"
done

# Sanity: the three ONNX files the adapter loads must all exist.
for f in "${ONNX_FILES[@]}"; do
    if [[ ! -s "$DIR/$f" ]]; then
        echo "[error] $DIR/$f missing or empty" >&2
        exit 4
    fi
done
if [[ ! -s "$DIR/tokenizer.json" ]]; then
    echo "[error] $DIR/tokenizer.json missing or empty" >&2
    exit 4
fi

echo "WHISPER_ONNX_DIR=$DIR"
echo "WHISPER_ONNX_QUANT=$QUANT"
