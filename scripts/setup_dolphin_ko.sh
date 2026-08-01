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
# Idempotent: skips an existing, complete bundle. Pass `DOLPHIN_KO_DIR`
# to override the location and `DOLPHIN_KO_SIZE` to choose `small`
# (default) or `base`.
#
# Usage:
#   scripts/setup_dolphin_ko.sh
#   DOLPHIN_KO_SIZE=base scripts/setup_dolphin_ko.sh
#   DOLPHIN_KO_DIR=/path scripts/setup_dolphin_ko.sh
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
#     redistribution of the above
#     Declaration: https://github.com/k2-fsa/sherpa-onnx
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
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${NAME}.tar.bz2"

require() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[error] required tool '$1' not on PATH" >&2
        exit 3
    fi
}
require curl
require tar

if [[ -s "$DIR/model.int8.onnx" && -s "$DIR/tokens.txt" ]]; then
    echo "[skip] $DIR already populated"
    echo "DOLPHIN_KO_DIR=$DIR"
    exit 0
fi

mkdir -p "$DIR"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[get] $NAME"
curl -fL --retry 3 --retry-delay 2 --max-time 1800 -o "$TMP/model.tar.bz2" "$URL"
tar xf "$TMP/model.tar.bz2" -C "$TMP"

# The archive unpacks into a directory named after the release; flatten
# the two files the adapter needs so the layout does not encode a date.
for f in model.int8.onnx tokens.txt; do
    src="$TMP/$NAME/$f"
    if [[ ! -s "$src" ]]; then
        echo "[error] $f missing from the archive" >&2
        exit 4
    fi
    cp "$src" "$DIR/$f"
done

echo "DOLPHIN_KO_DIR=$DIR"
echo "DOLPHIN_KO_SIZE=$SIZE"
