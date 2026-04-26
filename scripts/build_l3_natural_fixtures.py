#!/usr/bin/env python3
"""
Build natural-speech ablation fixtures for Phase C-2.

Streams a small subset of a HuggingFace ASR dataset, runs whisper-tiny
on each clip, and emits a JSONL fixture file shaped like
`tests/evaluation/fixtures/<lang>.jsonl` so that
`cargo eval-l3 --task ablation` can replay it through the post-ASR
pipeline.

Why this is a separate script from the eval binary: building these
fixtures is a one-time job (or once-per-corpus-update). Once committed,
the fixtures are reused across every PR's L3 ablation run without
re-incurring the whisper inference cost.

Currently supported sources:

  reazonspeech-tiny  ja, ~1 GB streaming, §30-4 (ML use only).
                     Output: tests/evaluation/fixtures-natural/ja.jsonl

For en (TED-LIUM 3) and zh (WenetSpeech meeting), the tarball-based
download dwarfs the per-utterance whisper cost. Use the L2 download
scripts to stage the manifest, then point this script at the manifest
TSV. (Not yet implemented; tracked as a follow-up.)

Usage:
  scripts/build_l3_natural_fixtures.py reazonspeech-tiny --n 30 \\
      --whisper-cli vendor/whisper.cpp/build/bin/whisper-cli \\
      --model vendor/whisper.cpp/models/ggml-tiny.bin \\
      --out tests/evaluation/fixtures-natural
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
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


def transcribe_with_whisper(whisper_cli: Path, model: Path, wav: Path, language: str) -> str:
    cmd = [
        str(whisper_cli),
        "-m", str(model),
        "-f", str(wav),
        "--no-timestamps",
        "--no-prints",
        "-l", language,
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def build_reazonspeech(args) -> int:
    load_dataset, sf, np = _import_deps()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ja.jsonl"
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path} already exists (pass --force to regenerate)", file=sys.stderr)
        return 0

    print("[get] reazon-research/reazonspeech (config=tiny, split=train, streaming)",
          file=sys.stderr)
    ds = load_dataset(
        "reazon-research/reazonspeech",
        "tiny",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, sample in enumerate(itertools.islice(ds, args.n)):
            arr = np.array(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"]["sampling_rate"]
            if sr != 16000:
                # Quick resample via librosa (only loaded if needed).
                import librosa  # type: ignore
                arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
                sr = 16000
            wav_path = Path(tmp) / f"clip_{i:04d}.wav"
            sf.write(str(wav_path), arr, sr, subtype="PCM_16")
            ref = (sample.get("transcription") or sample.get("text") or "").strip()
            try:
                hyp = transcribe_with_whisper(args.whisper_cli, args.model, wav_path, "ja")
            except subprocess.CalledProcessError as e:
                print(f"[skip] {i}: whisper failed: {e.stderr}", file=sys.stderr)
                continue
            if not ref or not hyp:
                continue
            rows.append({
                "id": f"reazonspeech_natural_{i:04d}",
                "category": "natural_speech",
                "reference": ref,
                "asr_hypothesis": hyp,
            })

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] wrote {len(rows)} entries → {out_path}", file=sys.stderr)
    return 0


def build_from_manifest(args, lang: str) -> int:
    """Run whisper over an existing FLEURS-format manifest TSV
    (id\\taudio_path\\treference) and emit the fixture JSONL.
    Used as a stand-in for ReazonSpeech / TED-LIUM / WenetSpeech when
    those corpora aren't accessible from the runner."""
    manifest = args.manifest
    if not manifest or not manifest.exists():
        print(f"--manifest required for source=manifest (got {manifest})", file=sys.stderr)
        return 2
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{lang}.jsonl"
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path} already exists (pass --force to regenerate)", file=sys.stderr)
        return 0

    rows = []
    with manifest.open("r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t", 2)
            if len(cols) != 3:
                continue
            sid, audio_rel, ref = cols
            # FLEURS-style manifests reference paths relative to the
            # *parent* of the language dir (e.g. "en/audio/1675.wav"
            # from /tmp/fleurs_subset/en/manifest.tsv), L2-style ones
            # reference paths relative to the manifest's own dir.
            # Try both.
            candidates = [
                manifest.parent / audio_rel,
                manifest.parent.parent / audio_rel,
            ]
            wav = next((p for p in candidates if p.exists()), None)
            if wav is None:
                continue
            try:
                hyp = transcribe_with_whisper(args.whisper_cli, args.model, wav, lang)
            except subprocess.CalledProcessError as e:
                print(f"[skip] {sid}: whisper failed: {e.stderr}", file=sys.stderr)
                continue
            if not ref or not hyp:
                continue
            rows.append({
                "id": f"{lang}_natural_{sid}",
                "category": "natural_speech",
                "reference": ref.strip(),
                "asr_hypothesis": hyp,
            })
            if args.n is not None and len(rows) >= args.n:
                break

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] wrote {len(rows)} entries → {out_path}", file=sys.stderr)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", choices=["reazonspeech-tiny", "manifest"])
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--lang", default="ja",
                   help="language code passed to whisper (en/ja/zh)")
    p.add_argument("--manifest", type=Path,
                   help="for source=manifest: path to a FLEURS-format manifest TSV")
    p.add_argument("--whisper-cli", type=Path,
                   default=Path(os.environ.get("WHISPER_CLI_PATH", "vendor/whisper.cpp/build/bin/whisper-cli")))
    p.add_argument("--model", type=Path,
                   default=Path(os.environ.get("WHISPER_MODEL_MULTI", "vendor/whisper.cpp/models/ggml-tiny.bin")))
    p.add_argument("--out", type=Path, default=Path("tests/evaluation/fixtures-natural"))
    p.add_argument("--force", action="store_true",
                   help="overwrite existing fixture file")
    args = p.parse_args()

    if args.source == "reazonspeech-tiny":
        return build_reazonspeech(args)
    if args.source == "manifest":
        return build_from_manifest(args, args.lang)
    return 1


if __name__ == "__main__":
    sys.exit(main())
