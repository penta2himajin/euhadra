#!/usr/bin/env python3
"""Offline RTF bench for hayamimi's shipping ASR models (sherpa-onnx).

Pairs with `examples/bench_shipping_asr.rs` (euhadra adapters) on the same
FLEURS manifests. Measures decode time only — no VAD / LID / punctuation.

Usage:
  python3 scripts/bench_hayamimi_asr.py \\
    --kind reazon --model-dir vendor/hayamimi/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17 \\
    --manifest data/fleurs_subset/ja/manifest.tsv \\
    --audio-root data/fleurs_subset \\
    --threads 4 --json-out /tmp/hayamimi_ja.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def read_wav(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise SystemExit(f"{path}: expected 16 kHz, got {sr}")
    return audio


def percentile(xs, p: float) -> float:
    if not xs:
        return 0.0
    arr = sorted(xs)
    idx = int(round((len(arr) - 1) * p))
    return arr[min(idx, len(arr) - 1)]


def _pick(model_dir: Path, *patterns: str) -> Path:
    """Return the first existing path matching any basename glob pattern."""
    for pat in patterns:
        hits = sorted(model_dir.glob(pat))
        if hits:
            return hits[0]
    raise SystemExit(
        f"no file matching {patterns} under {model_dir}; have: "
        + ", ".join(p.name for p in sorted(model_dir.iterdir())[:20])
    )


def build_recognizer(kind: str, model_dir: Path, threads: int, language: str):
    import sherpa_onnx

    d = model_dir
    tokens = str(_pick(d, "tokens.txt"))
    if kind == "reazon":
        # Zipformer transducer (ReazonSpeech ja/en). Prefer int8.
        enc = str(_pick(d, "encoder*-int8.onnx", "encoder*.onnx"))
        dec = str(_pick(d, "decoder*-int8.onnx", "decoder*.onnx"))
        joi = str(_pick(d, "joiner*-int8.onnx", "joiner*.onnx"))
        return (
            "hayamimi/reazonspeech-zipformer",
            sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=enc,
                decoder=dec,
                joiner=joi,
                tokens=tokens,
                num_threads=threads,
                provider="cpu",
                model_type="zipformer",
            ),
        )
    if kind == "parakeet_v3":
        enc = str(_pick(d, "encoder*.int8.onnx", "encoder*.onnx", "encoder.int8.onnx"))
        dec = str(_pick(d, "decoder*.int8.onnx", "decoder*.onnx", "decoder.int8.onnx"))
        joi = str(_pick(d, "joiner*.int8.onnx", "joiner*.onnx", "joiner.int8.onnx"))
        return (
            "hayamimi/parakeet-tdt-0.6b-v3-int8",
            sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=enc,
                decoder=dec,
                joiner=joi,
                tokens=tokens,
                num_threads=threads,
                provider="cpu",
                model_type="nemo_transducer",
            ),
        )
    if kind == "paraformer_zh":
        model = str(_pick(d, "model.int8.onnx", "model.onnx", "*.int8.onnx"))
        return (
            "hayamimi/paraformer-zh-int8",
            sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=model,
                tokens=tokens,
                num_threads=threads,
                provider="cpu",
            ),
        )
    if kind == "sensevoice":
        model = str(_pick(d, "model.int8.onnx", "model.onnx"))
        return (
            "hayamimi/sensevoice-small-int8",
            sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=model,
                tokens=tokens,
                num_threads=threads,
                provider="cpu",
                language=language or "auto",
                use_itn=False,
            ),
        )
    raise SystemExit(f"unknown kind {kind!r}")


def decode(recognizer, samples: np.ndarray) -> str:
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, samples)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--kind",
        required=True,
        choices=["reazon", "parakeet_v3", "paraformer_zh", "sensevoice"],
    )
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--audio-root", type=Path, required=True)
    ap.add_argument("--language", default="", help="SenseVoice language hint (ko/zh/…)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    print(f"loading {args.kind} from {args.model_dir}…", flush=True)
    t0 = time.perf_counter()
    model_name, recognizer = build_recognizer(
        args.kind, args.model_dir, args.threads, args.language
    )
    print(f"loaded {model_name} in {time.perf_counter() - t0:.1f}s", flush=True)

    rows_meta = []
    with args.manifest.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            uid, rel, *_ = line.split("\t")
            rows_meta.append((uid, args.audio_root / rel))

    warm = read_wav(rows_meta[0][1])
    for _ in range(args.warmup):
        decode(recognizer, warm)

    out_rows = []
    total_audio = 0.0
    total_asr = 0.0
    asr_ms = []
    for uid, path in rows_meta:
        samples = read_wav(path)
        dur = len(samples) / 16000.0
        t = time.perf_counter()
        hyp = decode(recognizer, samples)
        asr_s = time.perf_counter() - t
        total_audio += dur
        total_asr += asr_s
        asr_ms.append(asr_s * 1000.0)
        rtf = asr_s / max(dur, 1e-9)
        print(
            f"{uid}: audio={dur:.2f}s asr={asr_s*1000:.0f}ms rtf={rtf:.3f} hyp={hyp[:60]!r}",
            flush=True,
        )
        out_rows.append(
            {
                "id": uid,
                "audio_s": dur,
                "asr_s": asr_s,
                "rtf": rtf,
                "hyp": hyp,
            }
        )

    report = {
        "side": "hayamimi",
        "model": model_name,
        "kind": args.kind,
        "language": args.language,
        "threads": args.threads,
        "n": len(out_rows),
        "total_audio_s": total_audio,
        "total_asr_s": total_asr,
        "mean_rtf": total_asr / max(total_audio, 1e-9),
        "p50_asr_ms": percentile(asr_ms, 0.50),
        "p95_asr_ms": percentile(asr_ms, 0.95),
        "rows": out_rows,
    }
    print(
        f"SUMMARY model={report['model']} n={report['n']} "
        f"audio={report['total_audio_s']:.1f}s asr={report['total_asr_s']:.1f}s "
        f"mean_rtf={report['mean_rtf']:.3f} "
        f"p50={report['p50_asr_ms']:.0f}ms p95={report['p95_asr_ms']:.0f}ms",
        flush=True,
    )
    if args.json_out:
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
