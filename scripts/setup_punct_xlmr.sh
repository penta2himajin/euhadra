#!/usr/bin/env bash
#
# Download the multilingual punctuation / truecase / SBD ONNX bundle
# that `scripts/eval_punctuation.py` scores.
#
# Layout expected by the evaluator (and by `punctuators` with a local
# directory):
#
#   <DIR>/
#     model.onnx     ← ~1.1 GB graph (4 heads: pre/post punct, case, SBD)
#     sp.model       ← SentencePiece unigram (47-lang, lowercase)
#     config.yaml    ← label lists + max_length
#
# Model: 1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase
# (Apache-2.0). This is the Tier 2 punctuation candidate named in
# docs/model-upgrade-candidates.md §2.2. It does **not** plug into
# `OnnxPunctuationRestorer` (different tokenizer, I/O, and label
# scheme) — the bake-off harness calls it through the upstream
# `punctuators` package until a native Rust adapter exists.
#
# Idempotent: skips files that already exist. Pass `PUNCT_XLMR_DIR`
# to override the install location.
#
# Usage:
#   scripts/setup_punct_xlmr.sh
#   PUNCT_XLMR_DIR=vendor/my_punct scripts/setup_punct_xlmr.sh
#
# Licensing (informational — defer to the upstream URL for authoritative
# text; see docs/model-licenses.md for the consolidated table):
#   - 1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase: Apache-2.0
#     Declaration: https://huggingface.co/1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase
#       (model-card YAML: `license: apache-2.0`)

set -euo pipefail

DIR="${PUNCT_XLMR_DIR:-vendor/punct_xlmr}"
REPO="1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
BASE="https://huggingface.co/$REPO/resolve/main"

require() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "[error] $1 is required but not installed" >&2
        exit 2
    }
}
require curl

fetch() {
    local url="$1" target="$2"
    if [[ -s "$target" ]]; then
        echo "[skip] $(basename "$target") already present"
        return
    fi
    echo "[get] $url"
    curl -fL --retry 3 --retry-delay 2 --max-time 3600 \
        --create-dirs -o "$target" "$url"
}

echo "[setup] xlm-roberta punctuation ($REPO) → $DIR"
mkdir -p "$DIR"

# Small files first so a flaky network fails before the 1.1 GB graph.
fetch "$BASE/config.yaml" "$DIR/config.yaml"
fetch "$BASE/sp.model" "$DIR/sp.model"
fetch "$BASE/model.onnx" "$DIR/model.onnx"

for f in model.onnx sp.model config.yaml; do
    if [[ ! -s "$DIR/$f" ]]; then
        echo "[error] $DIR/$f missing or empty" >&2
        exit 4
    fi
done

# Sanity: the graph is ~1 GB. A truncated download is worse than a
# missing one — it loads and then crashes mid-eval.
size=$(wc -c <"$DIR/model.onnx")
if (( size < 500000000 )); then
    echo "[error] $DIR/model.onnx is only $size bytes; expected ~1.1 GB. Re-run after deleting it." >&2
    exit 5
fi

echo "PUNCT_XLMR_DIR=$DIR"
ls -lh "$DIR/model.onnx" "$DIR/sp.model" "$DIR/config.yaml"
