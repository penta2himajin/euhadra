#!/usr/bin/env bash
#
# Download the sherpa-onnx CTC export of DataoceanAI's Dolphin, the
# model selected for the Korean path (`docs/korean-asr-alternatives.md`
# §I). Layout:
#
#   <DIR>/
#     model.int8.onnx   ← the CTC graph
#     tokens.txt        ← symbol<space>id, one per line
#
# Idempotent: skips files that already exist. Pass `DOLPHIN_KO_DIR`
# to override the location and `DOLPHIN_KO_SIZE` to choose `small`
# (default) or `base`.
#
# Usage:
#   scripts/setup_dolphin_ko.sh
#   DOLPHIN_KO_SIZE=base scripts/setup_dolphin_ko.sh
#   DOLPHIN_KO_DIR=/path scripts/setup_dolphin_ko.sh
#
# Why HuggingFace and not the sherpa-onnx GitHub release: the release
# ships a `.tar.bz2` that has to be fetched, decompressed and flattened
# locally, so the two files we keep cost ~680 MB of peak disk (archive +
# extraction + copy) and a bzip2 pass. The upstream maintainer publishes
# the same export on HuggingFace with both files already at the repo
# root, which makes this two `curl`s and ~250 MB, and matches every
# other model script here.
#
# The artefacts are the same bytes, not merely the same build. Verified
# 2026-08-08 by fetching both sources and comparing SHA-256:
#
#   model.int8.onnx  c1afcb9265de0ebd853eb8f570b371f399a6f9b2b9af9a3cb17c2e509171e697
#   tokens.txt       c3788261a51df1899ea4b210b552cd42139204de72c0ad60f6cebb199078872e
#
# Why `small` and not `base`: measured on the same FLEURS-ko 30-utt set
# in the same container at one intra-op thread, small scores CER 0.0655
# at RTF 0.094 while base scores 0.1565 at RTF 0.044. Base buys 2.1x the
# speed for 2.4x the error, which is the wrong side of the trade when
# small is already 6x faster than the incumbent. The `medium` (0.9B) and
# `large` (1.7B) checkpoints are not public yet; §I records that we
# measure them when they are.
#
# Why Dolphin over the incumbent Whisper: whisper-large-v3-turbo pays a
# fixed 30-second encoder pass per utterance whatever the audio length
# (`docs/korean-asr-alternatives.md` §A.2), so a 4.8-second dictation
# utterance costs the same ~7 s as an 18.9-second one. Dolphin is CTC —
# non-autoregressive, cost proportional to audio — and 6x faster on one
# thread than Whisper q4 on four. It is also less accurate (CER 0.0655
# against 0.0269, or ~0.049 against ~0.025 once Korean ITN lands); §I
# states that trade plainly.
#
# One thread is not a typo. Above one intra-op thread this backend stops
# reproducing itself: five runs at four threads gave five different
# transcripts. `scripts/run_sherpa_ctc.py` defaults to 1 for that
# reason, and §I.1 records the measurement.
#
# Licensing (informational — defer to upstream URLs for authoritative text;
# see docs/model-licenses.md for the consolidated table):
#   - DataoceanAI/Dolphin (code and weights): Apache-2.0
#     Declaration: https://github.com/DataoceanAI/Dolphin
#   - k2-fsa/sherpa-onnx CTC conversion (what this script downloads):
#     redistribution of the above, Apache-2.0
#     Declaration: https://huggingface.co/csukuangfj/sherpa-onnx-dolphin-small-ctc-multi-lang-int8-2025-04-02
#       (model-card YAML: `license: apache-2.0`). The repo owner is the
#       sherpa-onnx maintainer, so this is the same publisher as the
#       GitHub release — a per-artefact licence declaration rather than
#       one buried in an archive.
#   - Canonical Apache-2.0 text: https://www.apache.org/licenses/LICENSE-2.0.txt

set -euo pipefail

SIZE="${DOLPHIN_KO_SIZE:-small}"
case "$SIZE" in
    small | base) ;;
    *)
        echo "[error] DOLPHIN_KO_SIZE must be 'small' or 'base', got '$SIZE'" >&2
        exit 2
        ;;
esac

DIR="${DOLPHIN_KO_DIR:-vendor/dolphin_ko}"
NAME="sherpa-onnx-dolphin-${SIZE}-ctc-multi-lang-int8-2025-04-02"
HF_REPO="https://huggingface.co/csukuangfj/${NAME}/resolve/main"

require() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[error] required tool '$1' not on PATH" >&2
        exit 3
    fi
}
require curl

mkdir -p "$DIR"

# Order: the small file first, so a broken link or a proxy serving HTML
# surfaces before 250 MB of bandwidth is spent on it.
for f in tokens.txt model.int8.onnx; do
    target="$DIR/$f"
    if [[ -s "$target" ]]; then
        echo "[skip] $f already present"
        continue
    fi
    echo "[get] $f"
    curl -fL --retry 3 --retry-delay 2 --max-time 1800 \
        -o "$target" "$HF_REPO/$f"
done

# Both files must be non-empty before the adapter is pointed at this
# directory: an interrupted `curl` leaves a zero-byte file behind, and
# the skip above would then read it as already fetched.
for f in tokens.txt model.int8.onnx; do
    if [[ ! -s "$DIR/$f" ]]; then
        echo "[error] $DIR/$f is missing or empty" >&2
        exit 4
    fi
done

echo "DOLPHIN_KO_DIR=$DIR"
echo "DOLPHIN_KO_SIZE=$SIZE"
