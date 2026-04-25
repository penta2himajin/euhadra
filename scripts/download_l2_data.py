#!/usr/bin/env python3
"""
Download a HuggingFace-hosted L2 dataset and emit a unified
`manifest.tsv` for `examples/eval_l2.rs`.

Currently supports:
- reazonspeech-test  (reazon-research/reazonspeech, test split, ~1 GB)

For OpenSLR / direct-archive datasets, use `scripts/download_l2_data.sh`.

Usage:
  scripts/download_l2_data.py reazonspeech-test [--n N] [--out <dir>]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


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


def setup_reazonspeech_test(n: int | None, out_dir: Path) -> None:
    load_dataset, sf, np = _import_deps()
    name = "reazonspeech-test"
    target = out_dir / name
    audio_dir = target / "audio"
    manifest_path = target / "manifest.tsv"
    if manifest_path.exists():
        print(f"[skip] {manifest_path} already present", file=sys.stderr)
        return
    audio_dir.mkdir(parents=True, exist_ok=True)

    # ReazonSpeech is large; "tiny" config (~1 GB) is the closest to a
    # test split available on HF. The user can override with `--n` to
    # cap the count.
    print("[get] reazon-research/reazonspeech (config=tiny, split=train, streaming=True)",
          file=sys.stderr)
    ds = load_dataset(
        "reazon-research/reazonspeech",
        "tiny",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    rows = []
    for i, sample in enumerate(ds):
        if n is not None and i >= n:
            break
        sid = str(sample.get("name") or sample.get("id") or i)
        wav_path = audio_dir / f"{sid}.wav"
        if not wav_path.exists():
            arr = np.array(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"]["sampling_rate"]
            sf.write(str(wav_path), arr, sr, subtype="PCM_16")
        ref = (sample.get("transcription") or sample.get("text") or "").strip()
        rows.append((sid, f"audio/{sid}.wav", ref))

    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("id\taudio_path\treference\n")
        for sid, p, ref in rows:
            f.write(f"{sid}\t{p}\t{ref}\n")
    print(f"[done] wrote {len(rows)} entries → {manifest_path}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", choices=["reazonspeech-test"])
    p.add_argument("--n", type=int, default=None,
                   help="cap utterance count (default: streaming, all)")
    p.add_argument("--out", type=Path,
                   default=Path(os.environ.get("L2_DATA_DIR", "data/l2")),
                   help="output root directory")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.target == "reazonspeech-test":
        setup_reazonspeech_test(args.n, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
