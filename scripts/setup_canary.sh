#!/usr/bin/env bash
#
# Download the ONNX-exported `nvidia/canary-180m-flash` model bundle
# from the `istupakov/canary-180m-flash-onnx` HuggingFace mirror.
# Defaults to the INT8-quantised pair (encoder ~134 MB, decoder
# ~79.5 MB, total ~213 MB); pass `CANARY_FP32=1` to fetch the
# full-precision pair instead (encoder ~463 MB, decoder ~316 MB,
# total ~779 MB).
#
# `canary-180m-flash` is a single multilingual checkpoint covering
# en / de / fr / es — the same bundle services every language the
# adapter speaks, so this script is intentionally language-agnostic
# and the caller picks the install path via `CANARY_DIR`. The legacy
# `CANARY_ES_DIR` variable is honoured as a fallback for the period
# when only `es` was wired up.
#
# The directory layout the resulting bundle produces matches what
# `CanaryAdapter::load` expects:
#
#   <CANARY_DIR>/
#     encoder-model.onnx       (or encoder-model.int8.onnx by default)
#     decoder-model.onnx       (or decoder-model.int8.onnx by default)
#     vocab.txt
#
# Idempotent: skips files that already exist. Pass `CANARY_DIR`
# (or legacy `CANARY_ES_DIR`) to override the default location.
#
# Why istupakov over the upstream nvidia/canary-180m-flash repo:
# NeMo's `model.export()` doesn't support Canary (open issue
# NVIDIA-NeMo/NeMo#11004). istupakov's project ships a custom
# encoder+decoder export with vocab.txt under the same upstream
# CC-BY-4.0 license — see docs/canary-integration.md for the full
# decision log.
#
# Usage:
#   scripts/setup_canary.sh                          # INT8, default dir
#   CANARY_FP32=1 scripts/setup_canary.sh            # full-precision
#   CANARY_DIR=/path scripts/setup_canary.sh         # custom location
#
# Licensing (informational — defer to upstream URLs for authoritative text;
# see docs/model-licenses.md for the consolidated table):
#   - nvidia/canary-180m-flash (model card): CC-BY-4.0
#     https://huggingface.co/nvidia/canary-180m-flash
#   - istupakov/canary-180m-flash-onnx (ONNX mirror): CC-BY-4.0 (inherited)
#     https://huggingface.co/istupakov/canary-180m-flash-onnx
#   Attribution required: credit NVIDIA, provide a link to the
#   CC-BY-4.0 text, and indicate modifications if any.

set -euo pipefail

DIR="${CANARY_DIR:-${CANARY_ES_DIR:-models/canary-180m-flash-onnx}}"
HF_REPO="https://huggingface.co/istupakov/canary-180m-flash-onnx/resolve/main"

if [[ "${CANARY_FP32:-0}" == "1" ]]; then
    ENCODER_FILE="encoder-model.onnx"
    DECODER_FILE="decoder-model.onnx"
    VARIANT="full-precision (~779 MB total)"
else
    ENCODER_FILE="encoder-model.int8.onnx"
    DECODER_FILE="decoder-model.int8.onnx"
    VARIANT="INT8 (~213 MB total)"
fi

echo "[setup] Canary-180M-Flash $VARIANT → $DIR"

mkdir -p "$DIR"

require() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[error] required tool '$1' not on PATH" >&2
        exit 3
    fi
}
require curl

# Order: smallest files first so `vocab.txt` / `config.json` succeed
# even on flaky networks, leaving the large encoder/decoder downloads
# at the tail where partial-progress is recoverable on retry.
for f in vocab.txt config.json "$ENCODER_FILE" "$DECODER_FILE"; do
    target="$DIR/$f"
    if [[ -s "$target" ]]; then
        echo "[skip] $f already present"
        continue
    fi
    echo "[get] $f"
    curl -fL --retry 3 --retry-delay 2 --max-time 1200 \
        -o "$target" "$HF_REPO/$f"
done

# When the user picks INT8 weights, drop a symlink so
# `CanaryConfig::default()` (which expects the unquantised filenames)
# also works without an explicit `with_int8_weights()` call.
if [[ "${CANARY_FP32:-0}" != "1" ]]; then
    for pair in "encoder-model.onnx encoder-model.int8.onnx" "decoder-model.onnx decoder-model.int8.onnx"; do
        link="${pair% *}"
        target="${pair#* }"
        if [[ ! -e "$DIR/$link" && -s "$DIR/$target" ]]; then
            (cd "$DIR" && ln -s "$target" "$link")
        fi
    done
fi

# Sanity: encoder, decoder, and vocab must all be non-empty.
for required in "$ENCODER_FILE" "$DECODER_FILE" vocab.txt; do
    if [[ ! -s "$DIR/$required" ]]; then
        echo "[error] $DIR/$required is missing or empty" >&2
        exit 4
    fi
done

echo "CANARY_DIR=$DIR"
