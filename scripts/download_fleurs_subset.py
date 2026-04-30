#!/usr/bin/env python3
"""
Download a small FLEURS subset for L1 smoke evaluation.

For each language config (HuggingFace `google/fleurs`), we stream the
first N test-split utterances, decode the audio to 16-bit PCM mono WAV,
and emit a manifest TSV that the Rust eval binary consumes.

Output layout:
  <out>/<lang>/audio/<id>.wav       (16 kHz, mono, 16-bit PCM)
  <out>/<lang>/manifest.tsv         (id<TAB>audio_path<TAB>reference)

Why a Python helper: the FLEURS dataset on HF is gated behind a legacy
loader script (`fleurs.py`) and there is no parquet mirror. The
`datasets` library + `librosa` is the cleanest way to extract specific
samples. Pin `datasets==2.21.0` for `trust_remote_code` support.

The output is tiny (~10 MB / language @ 10 utterances) and is meant to
be cached in CI; the script is idempotent and skips already-present
WAVs.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

LANGS = {
    "en": "en_us",
    "ja": "ja_jp",
    "zh": "cmn_hans_cn",
    # FLEURS ships Latin-American Spanish (es_419) — that's the
    # variety the eval baselines elsewhere in the project (Common
    # Voice ES, MLS) draw from too. The es_es Iberian variant is
    # NOT in FLEURS upstream.
    "es": "es_419",
}


def _import_deps():
    try:
        from datasets import load_dataset  # type: ignore
        import soundfile as sf  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as e:
        print(
            f"missing dependency: {e}\n"
            "install with: pip install 'datasets==2.21.0' soundfile librosa",
            file=sys.stderr,
        )
        sys.exit(2)
    return load_dataset, sf, np


def download_one(lang: str, n: int, out_dir: Path) -> None:
    load_dataset, sf, np = _import_deps()

    config = LANGS.get(lang)
    if config is None:
        raise SystemExit(f"unknown lang {lang!r}; expected one of {list(LANGS)}")

    lang_dir = out_dir / lang
    audio_dir = lang_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = lang_dir / "manifest.tsv"

    print(f"[{lang}] streaming google/fleurs {config} test split, n={n}", file=sys.stderr)
    ds = load_dataset(
        "google/fleurs",
        config,
        split="test",
        streaming=True,
        trust_remote_code=True,
    )

    # FLEURS shares one `id` across multiple speakers reading the same
    # sentence. Saving by `id` would overwrite the WAV with the last
    # speaker and emit duplicate manifest rows that double-count those
    # utterances at evaluation time. Dedup by id while we stream so we
    # collect `n` distinct sentences regardless of speaker overlap.
    rows = []
    seen: set[str] = set()
    for sample in ds:
        if len(rows) >= n:
            break
        sid = str(sample["id"])
        if sid in seen:
            continue
        seen.add(sid)
        wav_path = audio_dir / f"{sid}.wav"
        if not wav_path.exists():
            arr = np.array(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"]["sampling_rate"]
            sf.write(str(wav_path), arr, sr, subtype="PCM_16")
        # Reference transcription
        ref = sample["transcription"].strip()
        rows.append((sid, str(wav_path.relative_to(out_dir)), ref))

    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("id\taudio_path\treference\n")
        for sid, p, ref in rows:
            f.write(f"{sid}\t{p}\t{ref}\n")
    print(f"[{lang}] wrote {len(rows)} entries → {manifest_path}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--lang",
        action="append",
        choices=list(LANGS),
        help="languages to download; repeat or omit for all (en, ja, zh)",
    )
    p.add_argument("--n", type=int, default=10, help="utterances per language")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("FLEURS_DATA_DIR", "data/fleurs_subset")),
        help="output root directory",
    )
    args = p.parse_args()

    langs = args.lang or list(LANGS)
    args.out.mkdir(parents=True, exist_ok=True)
    for lang in langs:
        download_one(lang, args.n, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
