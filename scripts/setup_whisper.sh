#!/usr/bin/env bash
#
# Build whisper.cpp and download the tiny GGML models used by the L1
# smoke evaluation. Idempotent: skips already-built binaries / present
# models so it is safe to run on a warm CI cache.
#
# Usage:
#   scripts/setup_whisper.sh            # builds into ./vendor/whisper.cpp
#   WHISPER_DIR=/tmp/whisper.cpp scripts/setup_whisper.sh
#
# After it runs, the eval binary expects:
#   $WHISPER_DIR/build/bin/whisper-cli
#   $WHISPER_DIR/models/ggml-tiny.en.bin
#   $WHISPER_DIR/models/ggml-tiny.bin
#
# Licensing (informational — defer to upstream URLs for authoritative text;
# see docs/model-licenses.md for the consolidated table):
#   - whisper.cpp runtime (ggerganov): MIT
#     Declaration + text: https://github.com/ggerganov/whisper.cpp/blob/master/LICENSE
#   - OpenAI Whisper model weights (incl. ggml-* conversions): Apache-2.0
#     Declaration: https://github.com/openai/whisper/blob/main/LICENSE
#     Canonical text: https://www.apache.org/licenses/LICENSE-2.0.txt

set -euo pipefail

WHISPER_DIR="${WHISPER_DIR:-vendor/whisper.cpp}"
WHISPER_REF="${WHISPER_REF:-v1.7.4}"

mkdir -p "$(dirname "$WHISPER_DIR")"

# Every other setup_*.sh passes `curl --retry 3 --retry-delay 2`; this
# one had no retry on either of its two network operations. A single
# TCP timeout reaching github.com therefore turned the whole ASR eval
# job red — observed on run 30685477340, where the connect failed after
# 136 s on a cache miss. `git clone` has no --retry of its own, so it
# gets a loop; a failed attempt can leave a partial directory behind,
# which has to go before retrying or the next clone refuses the target.
retry_clone() {
    local attempt
    for attempt in 1 2 3; do
        if git clone --depth 1 --branch "$WHISPER_REF" \
            https://github.com/ggerganov/whisper.cpp "$WHISPER_DIR"; then
            return 0
        fi
        echo "[setup_whisper] clone attempt $attempt failed" >&2
        rm -rf "$WHISPER_DIR"
        sleep $((attempt * 2))
    done
    echo "[error] could not clone whisper.cpp after 3 attempts" >&2
    return 1
}

if [[ ! -d "$WHISPER_DIR/.git" ]]; then
    echo "[setup_whisper] cloning whisper.cpp@$WHISPER_REF into $WHISPER_DIR" >&2
    retry_clone
else
    echo "[setup_whisper] reusing existing $WHISPER_DIR" >&2
fi

if [[ ! -x "$WHISPER_DIR/build/bin/whisper-cli" ]]; then
    echo "[setup_whisper] configuring + building whisper-cli" >&2
    # GGML_NATIVE=OFF + GGML_OPENMP=OFF: portable, deterministic CI build.
    cmake -B "$WHISPER_DIR/build" -S "$WHISPER_DIR" \
        -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
        -DCMAKE_BUILD_TYPE=Release > /dev/null
    cmake --build "$WHISPER_DIR/build" --config Release \
        --target whisper-cli -j > /dev/null
else
    echo "[setup_whisper] whisper-cli already built" >&2
fi

mkdir -p "$WHISPER_DIR/models"
for model in ggml-tiny.en.bin ggml-tiny.bin; do
    target="$WHISPER_DIR/models/$model"
    if [[ -f "$target" ]]; then
        echo "[setup_whisper] $model already present" >&2
        continue
    fi
    url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$model"
    echo "[setup_whisper] downloading $model from $url" >&2
    curl -sSL --fail --retry 3 --retry-delay 2 --max-time 600 "$url" -o "$target"
done

echo "[setup_whisper] done." >&2
echo "WHISPER_CLI_PATH=$WHISPER_DIR/build/bin/whisper-cli"
echo "WHISPER_MODEL_DIR=$WHISPER_DIR/models"
