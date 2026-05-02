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
#   pip install funasr modelscope onnx onnxruntime sentencepiece torch torchaudio
#
# Licensing:
#   - FunASR runtime: MIT
#     (https://github.com/modelscope/FunASR/blob/main/LICENSE)
#   - SenseVoice model weights: CC-BY-NC-4.0 / SenseVoice MODEL_LICENSE
#     (https://github.com/FunAudioLLM/SenseVoice/blob/main/MODEL_LICENSE)
#     — non-commercial use only without explicit upstream permission.

# `inherit_errexit` propagates `set -e` into command substitutions so
# a Python failure inside `$(python3 - <<PY ... PY)` no longer leaves
# bash silently stepping over an empty capture. xtrace is on so the
# CI log records every command, not just stderr — without it a
# failure mode like "modelscope download timed out" looks identical
# to "model.export() raised" in the log summary.
set -euo pipefail
shopt -s inherit_errexit
set -x

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
df -h .

# Single end-to-end Python driver: does the FunASR export, locates the
# upstream artefacts inside the modelscope cache, copies them into $DIR
# under our naming convention, materialises tokens.txt from the
# SentencePiece BPE, and writes metadata.json. Doing it all in one
# `python3 -` invocation means any uncaught exception propagates as a
# non-zero exit code that bash's set -e respects, which the previous
# `$(python3 - ...)` capture pattern accidentally swallowed.
python3 - "$HF_REPO_ID" "$DIR" <<'PY'
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

repo_id = sys.argv[1]
out_dir = Path(sys.argv[2]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

print(f"[py] repo_id={repo_id} out_dir={out_dir}", flush=True)

try:
    from funasr import AutoModel
    print(f"[py] funasr={__import__('funasr').__version__}", flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(5)

try:
    import sentencepiece as spm
    print(f"[py] sentencepiece={spm.__version__}", flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(5)

# Match the parameter set used by FunAudioLLM/SenseVoice/export.py
# verbatim so we get the same ONNX graph the upstream demo exports
# (cpu device, opset 14, both FP32 + INT8 quantised model).
print("[py] AutoModel(...)", flush=True)
model = AutoModel(model=repo_id, device="cpu", disable_update=True)
print("[py] model.export(type='onnx', quantize=True, opset_version=14)", flush=True)
res = model.export(type="onnx", quantize=True, opset_version=14, device="cpu")
print(f"[py] export returned: {res!r}", flush=True)

# `model.export` returns either a single file path (older funasr), a
# list of file paths (newer funasr), or a dict whose values are file
# paths. We only need the directory those files live in — that's
# where am.mvn + the BPE model sit too, since funasr exports into the
# checkpoint's own cache dir.
def _resolve_export_dir(res):
    if isinstance(res, (list, tuple)) and res:
        return Path(res[0]).resolve().parent
    if isinstance(res, dict) and res:
        any_path = next(iter(res.values()))
        return Path(any_path).resolve().parent
    if isinstance(res, (str, os.PathLike)):
        p = Path(res).resolve()
        return p if p.is_dir() else p.parent
    raise RuntimeError(f"unexpected export() return type: {type(res).__name__}")

export_dir = _resolve_export_dir(res)
print(f"[py] export_dir={export_dir}", flush=True)
print(f"[py] export_dir contents: {sorted(p.name for p in export_dir.iterdir())}",
      flush=True)

# Copy ONNX + sidecars under our naming convention. We rename
# `model_quant.onnx` to `model.int8.onnx` to match the Canary adapter
# convention (`with_int8_weights()` swaps `model.onnx` → `model.int8.onnx`).
def _copy_required(src_name: str, dst_name: str):
    src = export_dir / src_name
    if not src.is_file() or src.stat().st_size == 0:
        raise FileNotFoundError(f"missing upstream artefact: {src}")
    dst = out_dir / dst_name
    if dst.is_file() and dst.stat().st_size > 0:
        print(f"[py] skip {dst_name} (already present)", flush=True)
        return
    print(f"[py] copy {src_name} → {dst_name}", flush=True)
    shutil.copy2(src, dst)

_copy_required("model.onnx", "model.onnx")
_copy_required("model_quant.onnx", "model.int8.onnx")
_copy_required("am.mvn", "am.mvn")

# config.yaml is a debugging aid (not consumed by the Rust adapter),
# so be lenient on absence — older / newer funasr exports may name it
# differently.
for cfg_candidate in ("config.yaml", "configuration.yaml"):
    src = export_dir / cfg_candidate
    if src.is_file():
        shutil.copy2(src, out_dir / "config.yaml")
        break

# Find the SentencePiece BPE model. The upstream filename has changed
# historically (`bpe.model`, `chn_jpn_yue_eng_ko_spectok.bpe.model`).
bpe_candidates = [
    export_dir / "chn_jpn_yue_eng_ko_spectok.bpe.model",
    export_dir / "bpe.model",
]
bpe_path = next((p for p in bpe_candidates if p.is_file()), None)
if bpe_path is None:
    matches = list(export_dir.glob("*.bpe.model"))
    if matches:
        bpe_path = matches[0]
if bpe_path is None or not bpe_path.is_file():
    raise FileNotFoundError(
        f"no SentencePiece *.bpe.model found under {export_dir}"
    )
print(f"[py] BPE model: {bpe_path}", flush=True)

# Dump tokens.txt (one piece per line, line index = id) so the Rust
# adapter can read the vocab without depending on the SentencePiece
# runtime. SenseVoice's special markers (e.g. `<|en|>`, `<|HAPPY|>`)
# are user-defined pieces in the BPE and round-trip verbatim.
tokens_path = out_dir / "tokens.txt"
if not (tokens_path.is_file() and tokens_path.stat().st_size > 0):
    sp = spm.SentencePieceProcessor()
    sp.Load(str(bpe_path))
    n = sp.GetPieceSize()
    if n < 1000:
        raise RuntimeError(
            f"BPE size {n} is suspiciously small (expected ~25K)"
        )
    with tokens_path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(sp.id_to_piece(i))
            fh.write("\n")
    print(f"[py] wrote {n} tokens to {tokens_path}", flush=True)

# Write metadata.json. Constants are baked into the upstream model.py
# (https://github.com/FunAudioLLM/SenseVoice/blob/main/model.py):
#   self.lid_dict      = {"auto": 0, "zh": 3, "en": 4, "yue": 7,
#                          "ja": 11, "ko": 12, "nospeech": 13}
#   self.textnorm_dict = {"withitn": 14, "woitn": 15}
#   blank_id           = 0  (CTC default in the upstream config)
# LFR(7,6) is hard-coded in the FunASR frontend for SenseVoice.
metadata_path = out_dir / "metadata.json"
if not (metadata_path.is_file() and metadata_path.stat().st_size > 0):
    metadata = {
        "lang2id": {
            "auto": 0, "zh": 3, "en": 4, "yue": 7,
            "ja": 11, "ko": 12, "nospeech": 13,
        },
        "with_itn_id": 14,
        "without_itn_id": 15,
        "blank_id": 0,
        "lfr_m": 7,
        "lfr_n": 6,
    }
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    print(f"[py] wrote metadata to {metadata_path}", flush=True)

print("[py] export complete.", flush=True)
PY

# Sanity: every adapter-required file must be present.
for required in model.onnx model.int8.onnx am.mvn tokens.txt metadata.json; do
    if [[ ! -s "$DIR/$required" ]]; then
        echo "[error] $DIR is missing $required" >&2
        exit 4
    fi
done

df -h .
echo "SENSEVOICE_DIR=$DIR"
