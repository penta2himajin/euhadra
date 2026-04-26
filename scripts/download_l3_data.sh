#!/usr/bin/env bash
#
# Download / clone the L3 evaluation datasets that have a public,
# license-friendly source. Idempotent: skips already-present targets.
#
# Usage:
#   scripts/download_l3_data.sh <target> [--out <dir>]
#
# Targets:
#   cs2w           ~ small,  CC-BY-SA 4.0  (text-only zh spoken→written;
#                                           awaits a zh filler-filter
#                                           implementation before it can
#                                           drive eval_l3)
#   tedlium3-test  ~50 GB,   CC-BY-NC-ND 3.0 (en, OSS-eval only; useful
#                                            for paragraph-splitter F1 +
#                                            ablation)
#
# Other L3 sources require manual access:
#   buckeye        — registration at https://buckeyecorpus.osu.edu
#   cejc           — NINJAL Corpus Portal (free academic / paid commercial)
#   magicdata-ramc — OpenSLR 123 direct download (use the URL printed below)

set -euo pipefail

TARGET="${1:-}"
shift || true

OUT="data/l3"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$TARGET" ]]; then
    cat >&2 <<EOF
usage: $0 <target> [--out <dir>]
targets (publicly available):
  cs2w
  tedlium3-test
  magicdata-ramc-info        prints download instructions only
EOF
    exit 2
fi

mkdir -p "$OUT"

require() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[error] required tool '$1' not on PATH" >&2
        exit 3
    fi
}

# ---- CS2W (zh spoken→written, EMNLP 2023, CC-BY-SA 4.0) -------------------

setup_cs2w() {
    local out_dir="$OUT/cs2w"
    if [[ -d "$out_dir/.git" ]]; then
        echo "[skip] $out_dir already cloned" >&2
        return 0
    fi
    require git
    echo "[get] git clone https://github.com/guozishan/CS2W → $out_dir" >&2
    git clone --depth 1 https://github.com/guozishan/CS2W "$out_dir"
    echo "[done] $out_dir" >&2
    cat <<EOF
note: CS2W is a text-only spoken→written parallel corpus. Driving
      eval_l3 against it requires a zh filler-filter implementation,
      which is not yet present in this codebase. The clone is staged
      for the follow-up PR that adds that filter.
EOF
}

# ---- TED-LIUM 3 test split (en, CC-BY-NC-ND 3.0, OSS evaluation only) -----

setup_tedlium3_test() {
    local out_dir="$OUT/tedlium3-test"
    local manifest="$out_dir/manifest.tsv"
    if [[ -f "$manifest" ]]; then
        echo "[skip] $manifest already present" >&2
        return 0
    fi

    require curl
    require tar
    require sox

    local cache="$OUT/.cache/tedlium3"
    mkdir -p "$cache" "$out_dir/audio"
    local tarball="$cache/TEDLIUM_release-3.tgz"
    if [[ ! -f "$tarball" ]]; then
        echo "[get] OpenSLR 51 (≈50 GB)" >&2
        curl -fL --retry 3 -o "$tarball" \
            "https://www.openslr.org/resources/51/TEDLIUM_release-3.tgz"
    fi
    local extract="$cache/TEDLIUM_release-3"
    if [[ ! -d "$extract" ]]; then
        echo "[extract] $tarball" >&2
        mkdir -p "$extract"
        tar -xzf "$tarball" -C "$extract"
    fi

    local sph_root="$extract/TEDLIUM_release-3/legacy/test/sph"
    local stm_root="$extract/TEDLIUM_release-3/legacy/test/stm"
    if [[ ! -d "$sph_root" ]]; then
        echo "[error] expected $sph_root after extraction" >&2
        exit 5
    fi

    echo -e "id\taudio_path\treference" > "$manifest"
    while IFS= read -r -d '' stm; do
        local talk
        talk="$(basename "$stm" .stm)"
        local sph="$sph_root/$talk.sph"
        if [[ ! -f "$sph" ]]; then
            continue
        fi
        # Convert SPH → 16 kHz mono WAV once per talk.
        local wav_full="$out_dir/audio/${talk}_full.wav"
        if [[ ! -f "$wav_full" ]]; then
            sox -t sph "$sph" -r 16000 -c 1 -b 16 "$wav_full"
        fi
        # Each STM line: "<talk> <ch> <speaker> <start_s> <end_s> <attrs> <text>"
        local idx=0
        while IFS= read -r line; do
            [[ "$line" =~ ^\;\; ]] && continue
            [[ -z "$line" ]] && continue
            local start_s end_s text
            read -r _talk _ch _spk start_s end_s _attrs text <<< "$line"
            # Skip empty / non-speech rows.
            if [[ -z "$text" || "$text" == "ignore_time_segment_in_scoring" ]]; then
                continue
            fi
            local id="${talk}_${idx}"
            local clip="$out_dir/audio/${id}.wav"
            sox "$wav_full" "$clip" trim "$start_s" "=$end_s"
            printf "%s\taudio/%s.wav\t%s\n" "$id" "$id" "$text" >> "$manifest"
            idx=$((idx + 1))
        done < "$stm"
        # Drop the per-talk full WAV after slicing to save disk.
        rm -f "$wav_full"
    done < <(find "$stm_root" -name "*.stm" -print0)
    echo "[done] $manifest" >&2
}

# ---- MagicData-RAMC info ---------------------------------------------------

magicdata_ramc_info() {
    cat <<EOF
MagicData-RAMC (OpenSLR 123, ~180h Mandarin conversation):

  curl -fL -o data/l3/.cache/magicdata-ramc.tar.gz \\
      https://www.openslr.org/resources/123/MDT2021S003-RAMC.tar.gz

After extraction, transcripts live under \`Train\`/\`Dev\`/\`Test\` with
.txt files paired to .wav files. The transcripts contain surface
disfluency markers (partial words, repetitions). A loader for
MagicData-RAMC is not yet implemented in this codebase — file an issue
or PR to add it.

Total download is ~25 GB; we don't auto-fetch by default to save
bandwidth.
EOF
}

case "$TARGET" in
    cs2w) setup_cs2w ;;
    tedlium3-test) setup_tedlium3_test ;;
    magicdata-ramc-info) magicdata_ramc_info ;;
    *) echo "[error] unknown target: $TARGET" >&2; exit 2 ;;
esac
