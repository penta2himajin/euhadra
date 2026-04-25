#!/usr/bin/env bash
#
# Download + extract one of the L2 evaluation datasets and convert it
# into the unified `manifest.tsv` format that `examples/eval_l2.rs`
# consumes. Idempotent: skips already-present archives, extractions,
# and manifests.
#
# Usage:
#   scripts/download_l2_data.sh <target> [--out <dir>]
#
# Targets:
#   librispeech-test-clean   ~350 MB, OpenSLR / CC-BY 4.0
#   librispeech-test-other   ~330 MB, OpenSLR / CC-BY 4.0
#   aishell1-test            extracted from data_aishell.tgz (16 GB total,
#                            but you only need to keep the test subset)
#   musan-noise              ~11 GB, OpenSLR 17 / CC-BY 4.0
#   slr26-rir                ~2.1 GB, OpenSLR 26 / Apache-2.0-style
#
# After it runs, the eval binary expects:
#   <out>/<target>/manifest.tsv
#   <out>/<target>/audio/<id>.wav      (16 kHz, mono, 16-bit PCM)

set -euo pipefail

TARGET="${1:-}"
shift || true

OUT="data/l2"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$TARGET" ]]; then
    cat >&2 <<EOF
usage: $0 <target> [--out <dir>]
targets:
  librispeech-test-clean
  librispeech-test-other
  aishell1-test
  musan-noise
  slr26-rir
EOF
    exit 2
fi

mkdir -p "$OUT"

# ---- helpers ---------------------------------------------------------------

require() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[error] required tool '$1' not on PATH" >&2
        exit 3
    fi
}
require curl
require tar

# Convert FLAC → 16 kHz mono PCM WAV. Falls back gracefully when ffmpeg
# is unavailable.
flac_to_wav() {
    local src="$1"
    local dst="$2"
    if command -v ffmpeg >/dev/null 2>&1; then
        ffmpeg -loglevel error -y -i "$src" -ar 16000 -ac 1 -sample_fmt s16 "$dst"
    elif command -v sox >/dev/null 2>&1; then
        sox "$src" -r 16000 -c 1 -b 16 "$dst"
    else
        echo "[error] need ffmpeg or sox to convert FLAC → WAV" >&2
        exit 4
    fi
}

# ---- LibriSpeech -----------------------------------------------------------

setup_librispeech() {
    local subset="$1" # test-clean or test-other
    local out_dir="$OUT/librispeech-$subset"
    local manifest="$out_dir/manifest.tsv"
    if [[ -f "$manifest" ]]; then
        echo "[skip] $manifest already present" >&2
        return 0
    fi

    local cache="$OUT/.cache/librispeech"
    mkdir -p "$cache" "$out_dir/audio"
    local tarball="$cache/$subset.tar.gz"
    if [[ ! -f "$tarball" ]]; then
        local url="https://www.openslr.org/resources/12/$subset.tar.gz"
        echo "[get] $url" >&2
        curl -fL --retry 3 -o "$tarball" "$url"
    fi

    local extract="$cache/$subset"
    if [[ ! -d "$extract" ]]; then
        echo "[extract] $tarball" >&2
        mkdir -p "$extract"
        tar -xzf "$tarball" -C "$extract"
    fi

    # LibriSpeech layout: LibriSpeech/<split>/<reader>/<chapter>/{<id>.flac, <reader>-<chapter>.trans.txt}
    local root="$extract/LibriSpeech/$subset"
    if [[ ! -d "$root" ]]; then
        echo "[error] expected $root after extraction" >&2
        exit 5
    fi

    echo -e "id\taudio_path\treference" > "$manifest"
    while IFS= read -r -d '' trans; do
        local dir
        dir="$(dirname "$trans")"
        while IFS= read -r line; do
            local id="${line%% *}"
            local ref="${line#* }"
            local flac="$dir/$id.flac"
            local wav="$out_dir/audio/$id.wav"
            if [[ -f "$flac" ]]; then
                flac_to_wav "$flac" "$wav"
                printf "%s\taudio/%s.wav\t%s\n" "$id" "$id" "$ref" >> "$manifest"
            fi
        done < "$trans"
    done < <(find "$root" -name "*.trans.txt" -print0)
    echo "[done] $manifest" >&2
}

# ---- AISHELL-1 -------------------------------------------------------------

setup_aishell1_test() {
    local out_dir="$OUT/aishell1-test"
    local manifest="$out_dir/manifest.tsv"
    if [[ -f "$manifest" ]]; then
        echo "[skip] $manifest already present" >&2
        return 0
    fi

    local cache="$OUT/.cache/aishell1"
    mkdir -p "$cache" "$out_dir/audio"
    local tarball="$cache/data_aishell.tgz"
    if [[ ! -f "$tarball" ]]; then
        local url="https://www.openslr.org/resources/33/data_aishell.tgz"
        echo "[get] $url (≈16 GB; this takes a while)" >&2
        curl -fL --retry 3 -o "$tarball" "$url"
    fi

    local extract="$cache/data_aishell"
    if [[ ! -d "$extract" ]]; then
        echo "[extract] $tarball" >&2
        mkdir -p "$extract"
        tar -xzf "$tarball" -C "$extract"
        # Inner archives contain per-speaker tarballs that need a second pass.
        find "$extract" -name "*.tar.gz" -exec tar -xzf {} -C "$(dirname {})" \;
    fi

    # Transcripts file: data_aishell/transcript/aishell_transcript_v0.8.txt
    local trans
    trans="$(find "$extract" -name "aishell_transcript_v0.8.txt" -print -quit)"
    if [[ -z "$trans" ]]; then
        echo "[error] AISHELL transcript file not found" >&2
        exit 5
    fi

    echo -e "id\taudio_path\treference" > "$manifest"
    while IFS= read -r line; do
        local id="${line%% *}"
        local ref="${line#* }"
        # Test split files live under .../wav/test/<speaker>/<id>.wav
        local wav
        wav="$(find "$extract" -path "*test*/$id.wav" -print -quit)"
        if [[ -n "$wav" && -f "$wav" ]]; then
            cp "$wav" "$out_dir/audio/$id.wav"
            printf "%s\taudio/%s.wav\t%s\n" "$id" "$id" "$ref" >> "$manifest"
        fi
    done < "$trans"
    echo "[done] $manifest" >&2
}

# ---- MUSAN noise -----------------------------------------------------------

setup_musan_noise() {
    local out_dir="$OUT/musan-noise"
    if [[ -d "$out_dir" ]] && [[ -n "$(ls -A "$out_dir" 2>/dev/null)" ]]; then
        echo "[skip] $out_dir already populated" >&2
        return 0
    fi

    local cache="$OUT/.cache/musan"
    mkdir -p "$cache" "$out_dir"
    local tarball="$cache/musan.tar.gz"
    if [[ ! -f "$tarball" ]]; then
        local url="https://www.openslr.org/resources/17/musan.tar.gz"
        echo "[get] $url (≈11 GB)" >&2
        curl -fL --retry 3 -o "$tarball" "$url"
    fi

    local extract="$cache/musan"
    if [[ ! -d "$extract" ]]; then
        echo "[extract] $tarball" >&2
        mkdir -p "$extract"
        tar -xzf "$tarball" -C "$extract"
    fi

    # We only want the `noise` subset (not music / speech).
    local noise="$extract/musan/noise"
    if [[ ! -d "$noise" ]]; then
        echo "[error] expected $noise after extraction" >&2
        exit 5
    fi
    cp -r "$noise"/. "$out_dir"/
    echo "[done] $out_dir populated" >&2
}

# ---- OpenSLR SLR26 RIR -----------------------------------------------------

setup_slr26_rir() {
    local out_dir="$OUT/rir-slr26"
    if [[ -d "$out_dir" ]] && [[ -n "$(ls -A "$out_dir" 2>/dev/null)" ]]; then
        echo "[skip] $out_dir already populated" >&2
        return 0
    fi

    local cache="$OUT/.cache/slr26"
    mkdir -p "$cache" "$out_dir"
    local zip="$cache/rirs_noises.zip"
    if [[ ! -f "$zip" ]]; then
        local url="https://www.openslr.org/resources/28/rirs_noises.zip"
        echo "[get] $url (≈2.1 GB)" >&2
        curl -fL --retry 3 -o "$zip" "$url"
    fi

    require unzip
    local extract="$cache/rirs_noises"
    if [[ ! -d "$extract" ]]; then
        echo "[extract] $zip" >&2
        unzip -q "$zip" -d "$cache"
    fi

    local rir="$extract/RIRS_NOISES/simulated_rirs"
    if [[ ! -d "$rir" ]]; then
        rir="$extract/RIRS_NOISES/real_rirs_isotropic_noises"
    fi
    if [[ ! -d "$rir" ]]; then
        echo "[error] expected RIR subdir under $extract" >&2
        exit 5
    fi
    cp -r "$rir"/. "$out_dir"/
    echo "[done] $out_dir populated" >&2
}

case "$TARGET" in
    librispeech-test-clean) setup_librispeech "test-clean" ;;
    librispeech-test-other) setup_librispeech "test-other" ;;
    aishell1-test) setup_aishell1_test ;;
    musan-noise) setup_musan_noise ;;
    slr26-rir) setup_slr26_rir ;;
    *) echo "[error] unknown target: $TARGET" >&2; exit 2 ;;
esac
