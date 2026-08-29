#!/usr/bin/env bash
#
# Run the euhadra vs hayamimi shipping-ASR speed comparison on this host.
# Requires models under vendor/ and FLEURS under data/fleurs_subset/.
#
# Usage:
#   scripts/run_hayamimi_speed_compare.sh
#   RESULTS=/tmp/speed_cmp scripts/run_hayamimi_speed_compare.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${RESULTS:-$ROOT/docs/benchmarks/hayamimi_speed_compare/raw}"
mkdir -p "$OUT_DIR"

THREADS="${THREADS:-4}"
AUDIO_ROOT="${AUDIO_ROOT:-data/fleurs_subset}"

need() {
  if [ ! -e "$1" ]; then
    echo "missing $1" >&2
    exit 1
  fi
}

need "$AUDIO_ROOT/en/manifest.tsv"
need "$AUDIO_ROOT/ja/manifest.tsv"
need "$AUDIO_ROOT/zh/manifest.tsv"
need "$AUDIO_ROOT/es/manifest.tsv"
need "$AUDIO_ROOT/ko/manifest.tsv"

need vendor/canary_en
need vendor/parakeet_ja
need vendor/paraformer_zh
need vendor/dolphin_ko
need vendor/hayamimi/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17
need vendor/hayamimi/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8
need vendor/hayamimi/sherpa-onnx-paraformer-zh-int8-2025-10-07
need vendor/hayamimi/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17

EUH=./target/release/examples/bench_shipping_asr
if [ ! -x "$EUH" ]; then
  echo "== building euhadra bench_shipping_asr =="
  export CC=gcc
  export CXX=g++
  export RUSTFLAGS="-C link-arg=-L/usr/lib/gcc/x86_64-linux-gnu/13"
  cargo build --release --features onnx --example bench_shipping_asr
fi

run_euh() {
  lang="$1"
  kind="$2"
  dir="$3"
  out="$4"
  echo "== euhadra ${lang} / ${kind} =="
  "$EUH" --kind "$kind" --model-dir "$dir" --language "$lang" \
    --manifest "$AUDIO_ROOT/$lang/manifest.tsv" \
    --audio-root "$AUDIO_ROOT" \
    --json-out "$out"
}

run_hya() {
  lang="$1"
  kind="$2"
  dir="$3"
  out="$4"
  shift 4
  echo "== hayamimi ${lang} / ${kind} =="
  python3 scripts/bench_hayamimi_asr.py --kind "$kind" --model-dir "$dir" \
    --manifest "$AUDIO_ROOT/$lang/manifest.tsv" \
    --audio-root "$AUDIO_ROOT" \
    --threads "$THREADS" \
    --json-out "$out" \
    "$@"
}

run_euh en canary vendor/canary_en "$OUT_DIR/euhadra_en.json"
run_hya en parakeet_v3 \
  "vendor/hayamimi/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8" \
  "$OUT_DIR/hayamimi_en.json"

run_euh es canary vendor/canary_en "$OUT_DIR/euhadra_es.json"
run_hya es parakeet_v3 \
  "vendor/hayamimi/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8" \
  "$OUT_DIR/hayamimi_es.json"

run_euh ja parakeet vendor/parakeet_ja "$OUT_DIR/euhadra_ja.json"
run_hya ja reazon \
  "vendor/hayamimi/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17" \
  "$OUT_DIR/hayamimi_ja.json"

run_euh zh paraformer vendor/paraformer_zh "$OUT_DIR/euhadra_zh.json"
run_hya zh paraformer_zh \
  "vendor/hayamimi/sherpa-onnx-paraformer-zh-int8-2025-10-07" \
  "$OUT_DIR/hayamimi_zh.json"

run_euh ko dolphin vendor/dolphin_ko "$OUT_DIR/euhadra_ko.json"
run_hya ko sensevoice \
  "vendor/hayamimi/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17" \
  "$OUT_DIR/hayamimi_ko.json" --language ko

python3 scripts/summarize_hayamimi_speed_compare.py \
  --raw-dir "$OUT_DIR" \
  --out docs/benchmarks/hayamimi_speed_compare.md \
  --json-out docs/benchmarks/hayamimi_speed_compare.json

echo "Done. See docs/benchmarks/hayamimi_speed_compare.md"
