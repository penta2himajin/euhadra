#!/usr/bin/env python3
"""
Build natural-speech ablation fixtures for Phase C-2.

Streams a small subset of a HuggingFace ASR dataset, runs an ASR model
on each clip, and emits a JSONL fixture file shaped like
`tests/evaluation/fixtures/<lang>.jsonl` so that
`cargo eval-l3 --task ablation` can replay it through the post-ASR
pipeline.

Why this is a separate script from the eval binary: building these
fixtures is a one-time job (or once-per-corpus-update). Once committed,
the fixtures are reused across every PR's L3 ablation run without
re-incurring the ASR inference cost.

Currently supported sources:

  reazonspeech-tiny  ja, ~1 GB streaming, §30-4 (ML use only).
                     Output: tests/evaluation/fixtures-natural/ja.jsonl
                     ASR: whisper-tiny (legacy fixtures match this).

  manifest           Run a FLEURS-format manifest TSV through the ASR
                     specified by `--asr`. Used for {en, ja, zh, es}
                     when streaming the dataset isn't appropriate.
                     ASR options: `whisper` (default; multi-lingual
                     whisper-tiny), `canary-es` (Spanish only; Canary
                     180M Flash via the Rust eval_l1_smoke binary).

For es specifically, the canary-es backend is the right choice
because Canary is the production ASR for Spanish in the pipeline —
using whisper-tiny would create a fixture whose hypothesis is from a
different model than the one being measured downstream.

Usage:
  # ja (legacy): stream ReazonSpeech, transcribe with whisper-tiny.
  scripts/build_l3_natural_fixtures.py reazonspeech-tiny --n 30 \\
      --whisper-cli vendor/whisper.cpp/build/bin/whisper-cli \\
      --model vendor/whisper.cpp/models/ggml-tiny.bin \\
      --out tests/evaluation/fixtures-natural

  # es: run FLEURS-es manifest through Canary.
  scripts/build_l3_natural_fixtures.py manifest --lang es \\
      --asr canary-es \\
      --manifest data/fleurs_subset/es/manifest.tsv \\
      --canary-es-dir models/canary-180m-flash-onnx \\
      --eval-binary ./target/release/examples/eval_l1_smoke \\
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


def transcribe_manifest_with_canary_es(
    eval_binary: Path,
    canary_es_dir: Path,
    data_dir: Path,
) -> dict[str, str]:
    """
    Run the eval_l1_smoke binary on the full es manifest under
    `data_dir/es/manifest.tsv` and parse its `--dump-utterances`
    output to recover one hypothesis per utterance ID.

    Returns a mapping `{utterance_id: hypothesis}`. Caller pairs that
    against the manifest's reference column to assemble fixture rows.

    Why one batch invocation rather than per-utt: model load is a
    multi-second ONNX session init we don't want to pay 10× over. The
    eval binary already handles batched manifest replay; we just
    extract the hypotheses out of its dump-utterances output.
    """
    cmd = [
        str(eval_binary),
        # eval_l1_smoke requires whisper paths even for an es-only run
        # (it builds the Cli even if those branches don't fire). Use
        # placeholder paths — they're never opened for `--langs es`.
        "--whisper-cli", "/dev/null",
        "--model-en", "/dev/null",
        "--model-multi", "/dev/null",
        "--data-dir", str(data_dir),
        "--canary-es-dir", str(canary_es_dir),
        "--langs", "es",
        "--dump-utterances",
    ]
    # eval_l1_smoke writes the per-utterance dump (`[es N] WER=... /
    # ref: ... / hyp: ...`) to stderr; the per-language summary +
    # baseline gates go to stdout. Merge so a single .stdout buffer
    # contains both. (`capture_output=True` is incompatible with an
    # explicit `stderr=` arg, so route both manually.)
    out = subprocess.run(
        cmd, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )

    # Per-utterance dump format from eval_l1_smoke.rs:
    #
    #     [es  N] WER=0.NNNN
    #          ref: <reference text>
    #          hyp: <hypothesis text>
    #
    # We index by the manifest's id (resolved by matching `ref` to
    # the manifest reference column, since the dump doesn't print
    # the manifest id directly).
    hyps: dict[str, str] = {}
    cur_ref: str | None = None
    for raw_line in out.stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("ref:"):
            cur_ref = line[len("ref:") :].strip()
        elif line.startswith("hyp:") and cur_ref is not None:
            hyp = line[len("hyp:") :].strip()
            hyps[cur_ref] = hyp
            cur_ref = None
    return hyps


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
    """Run an ASR backend over an existing FLEURS-format manifest TSV
    (id\\taudio_path\\treference) and emit the fixture JSONL.
    Used as a stand-in for ReazonSpeech / TED-LIUM / WenetSpeech when
    those corpora aren't accessible from the runner, and as the
    primary path for es (Canary)."""
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

    # Canary path: one batched call to eval_l1_smoke, then index by
    # reference text (the manifest order is deterministic so this is
    # safe — and the dump-utterances output is in manifest order too).
    canary_hyps: dict[str, str] | None = None
    if args.asr == "canary-es":
        if lang != "es":
            print(f"--asr canary-es is es-only (got --lang {lang})", file=sys.stderr)
            return 2
        if args.canary_es_dir is None:
            print("--canary-es-dir required for --asr canary-es", file=sys.stderr)
            return 2
        if not args.eval_binary.exists():
            print(
                f"--eval-binary {args.eval_binary} not found; "
                "build it with `cargo build --release --example eval_l1_smoke --features onnx`",
                file=sys.stderr,
            )
            return 2
        # eval_l1_smoke takes --data-dir not --manifest; the manifest
        # path is implicitly `<data-dir>/es/manifest.tsv`. The user
        # passes the manifest here for clarity, so derive data-dir
        # from it: manifest is .../es/manifest.tsv → data-dir = ../..
        data_dir = manifest.parent.parent
        print(f"[asr] running canary-es over {data_dir}/es/manifest.tsv", file=sys.stderr)
        try:
            canary_hyps = transcribe_manifest_with_canary_es(
                args.eval_binary, args.canary_es_dir, data_dir
            )
        except subprocess.CalledProcessError as e:
            print(f"[error] canary-es failed: {e.stderr}", file=sys.stderr)
            return 3
        print(f"[asr] canary-es returned {len(canary_hyps)} hypotheses", file=sys.stderr)

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
            ref = ref.strip()
            if not ref:
                continue

            if canary_hyps is not None:
                hyp = canary_hyps.get(ref)
                if hyp is None:
                    print(f"[skip] {sid}: no canary hypothesis matched ref", file=sys.stderr)
                    continue
            else:
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
                if not hyp:
                    continue

            rows.append({
                "id": f"{lang}_natural_{sid}",
                "category": "natural_speech",
                "reference": ref,
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
                   help="language code (en/ja/zh/es)")
    p.add_argument("--manifest", type=Path,
                   help="for source=manifest: path to a FLEURS-format manifest TSV")
    p.add_argument("--asr", choices=["whisper", "canary-es"], default="whisper",
                   help="ASR backend (default: whisper-tiny via --whisper-cli; "
                        "use canary-es for --lang es to match production ASR)")
    p.add_argument("--whisper-cli", type=Path,
                   default=Path(os.environ.get("WHISPER_CLI_PATH", "vendor/whisper.cpp/build/bin/whisper-cli")))
    p.add_argument("--model", type=Path,
                   default=Path(os.environ.get("WHISPER_MODEL_MULTI", "vendor/whisper.cpp/models/ggml-tiny.bin")))
    p.add_argument("--canary-es-dir", type=Path,
                   default=Path(os.environ["CANARY_ES_DIR"]) if "CANARY_ES_DIR" in os.environ else None,
                   help="directory containing canary-180m-flash ONNX bundle "
                        "(required for --asr canary-es)")
    p.add_argument("--eval-binary", type=Path,
                   default=Path("./target/release/examples/eval_l1_smoke"),
                   help="path to compiled eval_l1_smoke binary "
                        "(used as the canary-es ASR runner)")
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
