#!/usr/bin/env bash
#
# Self-export the official `FunAudioLLM/SenseVoiceSmall` checkpoint to
# the ONNX bundle layout that `SenseVoiceAdapter::load` expects:
#
#   <DIR>/
#     model.onnx          ← FP32 export
#     model.int8.onnx     ← INT8 quantised export
#     am.mvn              ← global CMVN statistics (Kaldi-NNet text format)
#     tokens.txt          ← one SentencePiece piece per line, line = id
#     metadata.json       ← lang2id / with_itn_id / blank_id / lfr_m / lfr_n
#     config.yaml         ← kept for reference / debugging
#
# Idempotent: skips outputs that already exist. Pass
# `SENSEVOICE_DIR` to override the default location.
#
# Usage:
#   scripts/setup_sensevoice.sh
#   SENSEVOICE_DIR=/path/to/dir scripts/setup_sensevoice.sh
#
# Why "self-export" rather than a pre-exported third-party bundle:
# the official `FunAudioLLM/SenseVoice` distribution ships only a
# PyTorch checkpoint + a SentencePiece BPE model. We want the ONNX
# graph that comes out of the upstream `FunASR.AutoModel.export()`
# code path, not a re-packaged variant whose graph might diverge from
# the maintained one.
#
# Python deps (install before running):
#   pip install funasr modelscope onnx onnxruntime sentencepiece
#
# Licensing:
#   - FunASR runtime: MIT
#     (https://github.com/modelscope/FunASR/blob/main/LICENSE)
#   - SenseVoice model weights: CC-BY-NC-4.0 / SenseVoice MODEL_LICENSE
#     (https://github.com/FunAudioLLM/SenseVoice/blob/main/MODEL_LICENSE)
#     — non-commercial use only without explicit upstream permission.

set -euo pipefail

DIR="${SENSEVOICE_DIR:-models/sensevoice-small-onnx}"
HF_REPO_ID="${SENSEVOICE_HF_REPO_ID:-iic/SenseVoiceSmall}"

mkdir -p "$DIR"

require() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[error] required tool '$1' not on PATH" >&2
        exit 3
    fi
}
require python3

# Refuse to start if every output is already in place — saves a 200MB
# checkpoint download + minutes of CPU on a re-run.
if [[ -s "$DIR/model.onnx" && -s "$DIR/model.int8.onnx" \
        && -s "$DIR/am.mvn" && -s "$DIR/tokens.txt" \
        && -s "$DIR/metadata.json" ]]; then
    echo "[skip] $DIR already populated"
    echo "SENSEVOICE_DIR=$DIR"
    exit 0
fi

echo "[setup] SenseVoice-Small → $DIR (repo=$HF_REPO_ID)"

# Run the official FunASR export. AutoModel downloads the checkpoint
# from ModelScope (or HuggingFace if MODELSCOPE_OFFLINE is unset and
# ModelScope is unreachable), then `export(type="onnx")` writes the
# ONNX files into the model's local cache directory and returns its
# path.
PY_OUT="$(python3 - "$HF_REPO_ID" <<'PY'
import json
import os
import sys
from pathlib import Path

repo_id = sys.argv[1]

try:
    from funasr import AutoModel
except ImportError as exc:
    sys.stderr.write(
        "[error] python package `funasr` is required: "
        "pip install funasr modelscope onnx onnxruntime sentencepiece\n"
    )
    sys.stderr.write(f"        underlying ImportError: {exc}\n")
    sys.exit(5)

# `disable_update=True` keeps modelscope from contacting its update
# server during init — flaky in CI and unnecessary for our use.
model = AutoModel(model=repo_id, disable_update=True)

# `quantize=True` emits both `model.onnx` (FP32) and `model_quant.onnx`
# (INT8) into the same directory. Returns the directory path.
export_dir = model.export(type="onnx", quantize=True)
if isinstance(export_dir, (list, tuple)):
    # Older funasr returns a list of file paths; take the parent dir.
    export_dir = os.path.dirname(export_dir[0])

print(json.dumps({"export_dir": str(Path(export_dir).resolve())}))
PY
)"

EXPORT_DIR="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["export_dir"])' <<<"$PY_OUT")"
echo "[ok] funasr export dir: $EXPORT_DIR"

# Copy the artefacts we need into $DIR. We rename `model_quant.onnx`
# to `model.int8.onnx` to match the convention used by the Canary
# adapter (`with_int8_weights()` swaps `model.onnx` → `model.int8.onnx`).
copy_if_present() {
    local src="$1" dst="$2"
    if [[ -s "$dst" ]]; then
        echo "[skip] $(basename "$dst") already present"
        return 0
    fi
    if [[ ! -s "$src" ]]; then
        echo "[error] expected upstream artefact missing: $src" >&2
        exit 6
    fi
    echo "[copy] $(basename "$dst")"
    cp "$src" "$dst"
}

copy_if_present "$EXPORT_DIR/model.onnx"        "$DIR/model.onnx"
copy_if_present "$EXPORT_DIR/model_quant.onnx"  "$DIR/model.int8.onnx"
copy_if_present "$EXPORT_DIR/am.mvn"            "$DIR/am.mvn"
copy_if_present "$EXPORT_DIR/config.yaml"       "$DIR/config.yaml"

# Locate the SentencePiece BPE model. The upstream filename has
# changed historically (`bpe.model`, `chn_jpn_yue_eng_ko_spectok.bpe.model`),
# so we fall back to the first `*.bpe.model` we find.
BPE_MODEL=""
for candidate in "$EXPORT_DIR/chn_jpn_yue_eng_ko_spectok.bpe.model" \
                 "$EXPORT_DIR/bpe.model"; do
    if [[ -s "$candidate" ]]; then
        BPE_MODEL="$candidate"
        break
    fi
done
if [[ -z "$BPE_MODEL" ]]; then
    BPE_MODEL="$(find "$EXPORT_DIR" -maxdepth 2 -name '*.bpe.model' -print -quit || true)"
fi
if [[ -z "$BPE_MODEL" || ! -s "$BPE_MODEL" ]]; then
    echo "[error] no SentencePiece *.bpe.model found under $EXPORT_DIR" >&2
    exit 7
fi
echo "[ok] BPE model: $BPE_MODEL"

# Dump tokens.txt (one piece per line, line index = id) from the
# SentencePiece BPE. The Rust adapter expects this exact layout.
if [[ ! -s "$DIR/tokens.txt" ]]; then
    echo "[gen] tokens.txt"
    python3 - "$BPE_MODEL" "$DIR/tokens.txt" <<'PY'
import sys

try:
    import sentencepiece as spm
except ImportError as exc:
    sys.stderr.write(
        "[error] python package `sentencepiece` is required: "
        "pip install sentencepiece\n"
    )
    sys.stderr.write(f"        underlying ImportError: {exc}\n")
    sys.exit(5)

bpe_path, out_path = sys.argv[1], sys.argv[2]
sp = spm.SentencePieceProcessor()
sp.Load(bpe_path)

n = sp.GetPieceSize()
if n < 1000:
    sys.stderr.write(
        f"[error] BPE size {n} is suspiciously small (expected ~25K)\n"
    )
    sys.exit(2)

with open(out_path, "w", encoding="utf-8") as fh:
    for i in range(n):
        # `id_to_piece` gives back the piece string. SenseVoice's
        # special markers (e.g. `<|en|>`, `<|HAPPY|>`) are stored as
        # user-defined pieces in the BPE and round-trip verbatim.
        fh.write(sp.id_to_piece(i))
        fh.write("\n")

print(f"[ok] wrote {n} tokens to {out_path}")
PY
fi

# Write metadata.json with the LID / textnorm / blank / LFR constants.
# These values are baked into the upstream `model.py`
# (https://github.com/FunAudioLLM/SenseVoice/blob/main/model.py):
#   self.lid_dict      = {"auto": 0, "zh": 3, "en": 4, "yue": 7,
#                          "ja": 11, "ko": 12, "nospeech": 13}
#   self.textnorm_dict = {"withitn": 14, "woitn": 15}
#   blank_id           = 0  (CTC default in the upstream config)
# LFR(7,6) is hard-coded in the FunASR frontend for SenseVoice.
if [[ ! -s "$DIR/metadata.json" ]]; then
    echo "[gen] metadata.json"
    python3 - "$DIR/metadata.json" <<'PY'
import json, sys

out_path = sys.argv[1]
metadata = {
    "lang2id": {
        "auto": 0,
        "zh": 3,
        "en": 4,
        "yue": 7,
        "ja": 11,
        "ko": 12,
        "nospeech": 13,
    },
    "with_itn_id": 14,
    "without_itn_id": 15,
    "blank_id": 0,
    "lfr_m": 7,
    "lfr_n": 6,
}
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(metadata, fh, ensure_ascii=False, indent=2)
print(f"[ok] wrote metadata to {out_path}")
PY
fi

# Sanity: every adapter-required file must be present.
for required in model.onnx model.int8.onnx am.mvn tokens.txt metadata.json; do
    if [[ ! -s "$DIR/$required" ]]; then
        echo "[error] $DIR is missing $required" >&2
        exit 4
    fi
done

echo "SENSEVOICE_DIR=$DIR"
