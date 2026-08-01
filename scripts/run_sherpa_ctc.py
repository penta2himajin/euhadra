#!/usr/bin/env python3
"""Transcribe a FLEURS-style manifest with a sherpa-onnx backend.

Candidate ASR backends get measured before anyone commits to porting
them into the Rust stack — `docs/korean-asr-alternatives.md` §A.1 did
that for transformers and whisper.cpp, §A.2 for the ONNX quantisation
variants, and §I for the CTC candidates this script drives.

It writes `id<TAB>hypothesis` on stdout and timings on stderr. Scoring
is deliberately *not* done here: pipe the output through

    cargo run --release --example score_hypotheses -- \\
        --manifest <manifest> --hypotheses <this output> --metric cer

so the CER comes from `eval::metrics::cer_lenient`, the same function
the committed baselines use. Re-implementing the normaliser in Python
would make the comparison against the incumbent meaningless.

Usage:
    scripts/run_sherpa_ctc.py dolphin \\
        vendor/dolphin_ko/model.int8.onnx vendor/dolphin_ko/tokens.txt \\
        data/fleurs_subset/ko/manifest.tsv data/fleurs_subset > hyp.tsv

Requires `pip install sherpa-onnx`. Not part of any CI job: it pulls a
third-party runtime and model weights, which belong in a release-time
measurement rather than a per-PR gate.
"""

import argparse
import sys
import time
import wave

import numpy as np
import sherpa_onnx

KINDS = ("dolphin", "omnilingual", "sensevoice")

# Why the default is 1 and not `min(4, nproc)` like the Whisper adapter:
# above one intra-op thread these backends stop being reproducible.
# Measured on FLEURS-ko 30 utterances, four cores, Dolphin small INT8 —
# five runs at `--threads 4` produced five *different* transcripts
# (CER 0.0818 / 0.0865 / 0.0895 / 0.0921 / 0.0926) from a byte-identical
# model file, while three runs at `--threads 1` were byte-identical to
# each other at CER 0.0655. Dolphin base disagreed with itself at
# `--threads 2` as well. A benchmark that reports one draw from that
# distribution as "the" number is not a measurement, so the default is
# the setting that reproduces. See §I.1.
DEFAULT_THREADS = 1


def build(kind, model, tokens, threads):
    if kind == "dolphin":
        return sherpa_onnx.OfflineRecognizer.from_dolphin_ctc(
            model=model, tokens=tokens, num_threads=threads
        )
    if kind == "omnilingual":
        return sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
            model=model, tokens=tokens, num_threads=threads
        )
    if kind == "sensevoice":
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model, tokens=tokens, num_threads=threads, use_itn=True
        )
    raise SystemExit(f"unknown backend {kind!r}; expected one of {KINDS}")


def read_manifest(path, root):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("id\t"):
                continue
            uid, audio, _ref = line.rstrip("\n").split("\t")
            rows.append((uid, f"{root}/{audio}"))
    return rows


def read_wav(path):
    with wave.open(path) as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16-bit mono PCM")
        rate = w.getframerate()
        frames = w.getnframes()
        pcm = np.frombuffer(w.readframes(frames), dtype=np.int16)
    return rate, frames / rate, pcm.astype(np.float32) / 32768.0


def parse_args(argv):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("kind", choices=KINDS)
    p.add_argument("model")
    p.add_argument("tokens")
    p.add_argument("manifest")
    p.add_argument("audio_root")
    p.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=(
            "intra-op threads (default 1). Values above 1 make these "
            "backends non-reproducible; raise it only to measure "
            "throughput, never to report accuracy."
        ),
    )
    return p.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])

    t0 = time.time()
    rec = build(args.kind, args.model, args.tokens, args.threads)
    print(
        f"loaded {args.kind} in {time.time() - t0:.1f}s (threads={args.threads})",
        file=sys.stderr,
    )
    if args.threads != 1:
        print(
            f"[warn] threads={args.threads}: output is not reproducible; "
            "accuracy figures from this run are one draw, not a measurement",
            file=sys.stderr,
        )

    total_audio = total_asr = 0.0
    print("id\thypothesis")
    for uid, path in read_manifest(args.manifest, args.audio_root):
        rate, duration, pcm = read_wav(path)
        t = time.time()
        stream = rec.create_stream()
        stream.accept_waveform(rate, pcm)
        rec.decode_stream(stream)
        hypothesis = stream.result.text.strip()
        elapsed = time.time() - t

        total_audio += duration
        total_asr += elapsed
        print(
            f"{uid}: audio={duration:.2f}s asr={elapsed * 1000:.0f}ms "
            f'hyp="{hypothesis[:70]}"',
            file=sys.stderr,
        )
        # Tabs would corrupt the two-column contract the scorer reads.
        print(f"{uid}\t{hypothesis.replace(chr(9), ' ')}")

    if total_audio > 0:
        print(
            f"\ntotal_audio_s : {total_audio:.2f}\n"
            f"total_asr_s   : {total_asr:.2f}\n"
            f"rtf           : {total_asr / total_audio:.4f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
