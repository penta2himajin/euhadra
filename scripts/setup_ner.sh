#!/usr/bin/env bash
#
# Download the NER ONNX bundle that `OnnxEntityRecognizer::load` accepts,
# into the layout that loader expects:
#
#   <DIR>/
#     model.onnx        ← token-classification graph
#     tokenizer.json    ← HuggingFace fast tokenizer
#     config.json       ← carries `id2label`; see the warning below
#
# Model: dslim/distilbert-NER — DistilBERT fine-tuned on CoNLL-2003,
# emitting the four classes euhadra's `EntityLabel` knows (PER / LOC /
# ORG / MISC). `docs/spec.md` §3.6 picks it for being the same BERT
# token-classification shape as the punctuation restorer, so both run on
# one inference path.
#
# Idempotent: skips files that already exist. Pass `NER_DIR` to override
# the install location.
#
# Upstream ships only `onnx/model.onnx` — there is no quantised graph in
# this repo, despite `docs/spec.md` §3.6 quoting a ~65 MB INT8 figure.
# Quantising is left to the consumer.
#
# Usage:
#   scripts/setup_ner.sh
#   NER_DIR=vendor/my_ner scripts/setup_ner.sh
#
# IMPORTANT — check `id2label` before trusting the output. The label
# order passed to `OnnxEntityRecognizer::load` must match the model's
# own. A mismatch does not error; it mislabels every entity, and PER
# becomes ORG without anything failing. `OnnxEntityRecognizer::
# default_labels()` matches dslim/distilbert-NER. For any other
# checkpoint, read `config.json`:
#
#   jq -r '.id2label | to_entries | sort_by(.key|tonumber) | .[].value' \
#       <DIR>/config.json
#
# Licensing (informational — defer to the upstream URL for authoritative
# text; see docs/model-licenses.md for the consolidated table):
#   - dslim/distilbert-NER: MIT
#     Declaration: https://huggingface.co/dslim/distilbert-NER
#       (model-card YAML: `license: mit`)
#   - Training data CoNLL-2003 carries its own terms, which apply to the
#     corpus rather than to these weights.

set -euo pipefail

DIR="${NER_DIR:-vendor/ner_distilbert}"
REPO="dslim/distilbert-NER"
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
    curl -fL --retry 3 --retry-delay 2 --max-time 1200 \
        --create-dirs -o "$target" "$url"
}

onnx_path="onnx/model.onnx"

echo "[setup] distilbert-NER ($REPO) → $DIR"
mkdir -p "$DIR"

# Small files first, so a flaky network fails with an obvious diagnostic
# before the graph download starts.
fetch "$BASE/tokenizer.json" "$DIR/tokenizer.json"
fetch "$BASE/config.json" "$DIR/config.json"
curl -fsL --retry 2 -o "$DIR/tokenizer_config.json" \
    "$BASE/tokenizer_config.json" 2>/dev/null || true
curl -fsL --retry 2 -o "$DIR/special_tokens_map.json" \
    "$BASE/special_tokens_map.json" 2>/dev/null || true

fetch "$BASE/$onnx_path" "$DIR/model.onnx"

for f in model.onnx tokenizer.json; do
    if [[ ! -s "$DIR/$f" ]]; then
        echo "[error] $DIR/$f missing or empty" >&2
        exit 4
    fi
done

# Surface the label order rather than leaving it to be discovered by a
# silently mislabelled entity later.
if command -v jq >/dev/null 2>&1 && [[ -s "$DIR/config.json" ]]; then
    echo "[labels] id2label order from config.json:"
    jq -r '.id2label | to_entries | sort_by(.key|tonumber) | .[].value' \
        "$DIR/config.json" 2>/dev/null | tr '\n' ' ' || true
    echo
    echo "[labels] compare against OnnxEntityRecognizer::default_labels()"
fi

echo "NER_DIR=$DIR"
