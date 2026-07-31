#!/usr/bin/env bash
#
# Download the sentence-embedding ONNX bundles that
# `OnnxEmbeddingFilter::load` / `OnnxTextEmbedder::load` accept, into
# the directory layout those loaders expect:
#
#   <DIR>/
#     model.onnx        ← the graph (flattened from the repo's onnx/ dir)
#     tokenizer.json    ← HuggingFace fast tokenizer
#     config.json       ← architecture reference (input-signature probe)
#
# One bundle per candidate model. Selection via `EMBEDDER_MODEL`:
#
#   bge-small   BAAI/bge-small-en-v1.5              (current default, en only)
#   granite     ibm-granite/granite-embedding-97m-multilingual-r2
#   potion      minishlab/potion-multilingual-128M  (static embeddings)
#
# Idempotent: skips files that already exist. Pass `EMBEDDER_DIR` to
# override the install location, `EMBEDDER_QUANT=int8` to prefer a
# quantised graph where the upstream repo ships one (granite only).
#
# Usage:
#   scripts/setup_embedders.sh                          # bge-small (baseline)
#   EMBEDDER_MODEL=granite scripts/setup_embedders.sh
#   EMBEDDER_MODEL=granite EMBEDDER_QUANT=int8 scripts/setup_embedders.sh
#   EMBEDDER_MODEL=all scripts/setup_embedders.sh       # all three
#
# Why these three: `docs/model-upgrade-candidates.md` §2 records the
# comparison. bge-small-en-v1.5 is what Tier 1/2 embedding consumers
# use today and is English-only; granite-embedding-97m-multilingual-r2
# is the same 384 dimensions under Apache-2.0 with ja/zh/ko/es in its
# enhanced-language set; potion-multilingual-128M is a static
# (inference-free) embedding table that costs a table lookup instead
# of a transformer forward pass.
#
# Licensing (informational — defer to upstream URLs for authoritative text;
# see docs/model-licenses.md for the consolidated table):
#   - BAAI/bge-small-en-v1.5: MIT
#     Declaration: https://huggingface.co/BAAI/bge-small-en-v1.5
#       (model-card YAML: `license: mit`)
#   - ibm-granite/granite-embedding-97m-multilingual-r2: Apache-2.0
#     Declaration: https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2
#       (model-card YAML: `license: apache-2.0`)
#   - minishlab/potion-multilingual-128M: MIT
#     Declaration: https://huggingface.co/minishlab/potion-multilingual-128M
#       (model-card YAML: `license: mit`)
#   - Canonical MIT text: https://opensource.org/licenses/MIT
#   - Canonical Apache-2.0 text: https://www.apache.org/licenses/LICENSE-2.0.txt

set -euo pipefail

MODEL="${EMBEDDER_MODEL:-bge-small}"
QUANT="${EMBEDDER_QUANT:-fp32}"

require() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[error] required tool '$1' not on PATH" >&2
        exit 3
    fi
}
require curl

fetch() {
    local url="$1" target="$2"
    if [[ -s "$target" ]]; then
        echo "[skip] $(basename "$target") already present"
        return
    fi
    echo "[get] $url"
    curl -fL --retry 3 --retry-delay 2 --max-time 1200 \
        --create-dirs -o "$target" "$url"
}

install_one() {
    local name="$1" repo="$2" onnx_path="$3" dir="$4"
    local base="https://huggingface.co/$repo/resolve/main"

    echo "[setup] $name ($repo) → $dir"
    mkdir -p "$dir"

    # Small files first so a flaky network fails with an obvious
    # diagnostic before the ~100-400 MB graph download starts.
    fetch "$base/tokenizer.json" "$dir/tokenizer.json"
    fetch "$base/config.json" "$dir/config.json"
    # Not every repo ships these; absence is not fatal.
    curl -fsL --retry 2 -o "$dir/tokenizer_config.json" \
        "$base/tokenizer_config.json" 2>/dev/null || true
    curl -fsL --retry 2 -o "$dir/special_tokens_map.json" \
        "$base/special_tokens_map.json" 2>/dev/null || true

    fetch "$base/$onnx_path" "$dir/model.onnx"

    for f in model.onnx tokenizer.json; do
        if [[ ! -s "$dir/$f" ]]; then
            echo "[error] $dir/$f missing or empty" >&2
            exit 4
        fi
    done
    echo "EMBEDDER_DIR=$dir"
}

install_bge_small() {
    install_one "bge-small-en-v1.5" "BAAI/bge-small-en-v1.5" \
        "onnx/model.onnx" "${EMBEDDER_DIR:-vendor/embedder_bge_small}"
}

install_granite() {
    local onnx_path="onnx/model.onnx"
    if [[ "$QUANT" == "int8" ]]; then
        onnx_path="onnx/model_quint8_avx2.onnx"
    fi
    install_one "granite-embedding-97m-multilingual-r2" \
        "ibm-granite/granite-embedding-97m-multilingual-r2" \
        "$onnx_path" "${EMBEDDER_DIR:-vendor/embedder_granite_97m}"
}

install_potion() {
    install_one "potion-multilingual-128M" \
        "minishlab/potion-multilingual-128M" \
        "onnx/model.onnx" "${EMBEDDER_DIR:-vendor/embedder_potion}"
}

case "$MODEL" in
    bge-small) install_bge_small ;;
    granite) install_granite ;;
    potion) install_potion ;;
    all)
        # `EMBEDDER_DIR` would collapse all three into one directory,
        # so the `all` path always uses the per-model defaults.
        unset EMBEDDER_DIR
        install_bge_small
        install_granite
        install_potion
        ;;
    *)
        echo "[error] unknown EMBEDDER_MODEL='$MODEL'" >&2
        echo "        expected one of: bge-small, granite, potion, all" >&2
        exit 2
        ;;
esac
