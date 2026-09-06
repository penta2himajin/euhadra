#!/usr/bin/env python3
"""Project end-to-end latency impact of swapping in XLM-R punctuation.

CI today measures:

- `evaluate-asr` — ASR + post-ASR pipeline with `BasicPunctuationRestorer`
  (E2E ≈ ASR; rule punct is ~μs and invisible in ms totals)
- `evaluate-fast` — per-layer μ-benchmark; punctuation p50 ≈ 0.5 μs
- punctuation bake-off (local, not CI) — XLM-R alone ≈ 240–260 ms/utt

This script answers "how much slower is the whole utterance path if we
chain XLM-R after ASR?" by:

1. Loading ASR p50/p95 from `docs/benchmarks/ci_baseline.json`
2. Timing `basic` vs `xlmr` (+ lang normalizer) on two text corpora:
   - L1 fixtures (`tests/evaluation/fixtures/*.jsonl`) — short ASR hyps
   - punct bake-off inputs (`data/punct_eval/*_punct.jsonl`) — longer
3. Reporting projected E2E = ASR_p50 + punct_p50 and the relative lift

It does **not** re-run ASR. That keeps the number comparable to the
committed CI ASR baseline without downloading models. Wall-clock is
CPU-only ONNX Runtime, same as the bake-off.

Usage:
    scripts/setup_punct_xlmr.sh   # once
    scripts/eval_punct_e2e_impact.py \\
        --output docs/benchmarks/punctuation/e2e_impact.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_punctuation import (  # noqa: E402
    basic_restore,
    es_punct_normalize,
    ja_punct_normalize,
    make_xlmr_fn,
    zh_punct_normalize,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "docs/benchmarks/ci_baseline.json"
DEFAULT_FIXTURES = ROOT / "tests/evaluation/fixtures"
DEFAULT_PUNCT = ROOT / "data/punct_eval"
DEFAULT_XLMR = ROOT / "vendor/punct_xlmr"


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def load_asr_baseline(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    out = {}
    for lang, row in data.get("languages", {}).items():
        out[lang] = {
            "asr_p50_ms": row["asr_latency_ms"]["p50"],
            "asr_p95_ms": row["asr_latency_ms"]["p95"],
            "e2e_p50_ms": row["e2e_latency_ms"]["p50"],
            "e2e_p95_ms": row["e2e_latency_ms"]["p95"],
            "rtf": row.get("rtf"),
            "samples": row.get("samples"),
        }
    return out


def load_fixture_texts(fixtures_dir: Path, lang: str) -> list[str]:
    path = fixtures_dir / f"{lang}.jsonl"
    if not path.exists():
        return []
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        t = row.get("asr_hypothesis") or row.get("hypothesis") or ""
        if t.strip():
            texts.append(t.strip())
    return texts


def load_punct_texts(punct_dir: Path, lang: str) -> list[str]:
    path = punct_dir / f"{lang}_punct.jsonl"
    if not path.exists():
        return []
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        t = row.get("input") or ""
        if t.strip():
            texts.append(t.strip())
    return texts


def time_fn(fn, texts: list[str], warmup: int = 2) -> dict:
    if not texts:
        return {"n": 0, "p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "chars_p50": 0}
    for t in texts[:warmup]:
        fn(t)
    times: list[float] = []
    for t in texts:
        t0 = time.perf_counter()
        fn(t)
        times.append((time.perf_counter() - t0) * 1000.0)
    chars = sorted(len(t) for t in texts)
    return {
        "n": len(texts),
        "p50_ms": round(percentile(times, 0.50), 2),
        "p95_ms": round(percentile(times, 0.95), 2),
        "mean_ms": round(statistics.fmean(times), 2),
        "chars_p50": chars[len(chars) // 2],
        "chars_mean": round(statistics.fmean(chars), 1),
    }


def lang_post(lang: str):
    if lang == "zh":
        return zh_punct_normalize
    if lang == "ja":
        return ja_punct_normalize
    if lang == "es":
        return es_punct_normalize
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--fixtures-dir", type=Path, default=DEFAULT_FIXTURES)
    ap.add_argument("--punct-dir", type=Path, default=DEFAULT_PUNCT)
    ap.add_argument("--xlmr-dir", type=Path, default=DEFAULT_XLMR)
    ap.add_argument("--langs", default="en,ja,zh,es,ko")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    if not (args.xlmr_dir / "model.onnx").exists():
        print(
            f"[error] {args.xlmr_dir}/model.onnx missing; "
            "run scripts/setup_punct_xlmr.sh first",
            file=sys.stderr,
        )
        return 2

    asr = load_asr_baseline(args.baseline)
    print(f"[e2e] loading xlmr from {args.xlmr_dir}…")
    xlmr = make_xlmr_fn(args.xlmr_dir)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    report = {
        "schema_version": 1,
        "note": (
            "Projected E2E = ASR p50 from ci_baseline.json + punct p50. "
            "CI evaluate-asr currently uses BasicPunctuationRestorer only; "
            "XLM-R is not wired into CI. Post-ASR layers other than punct "
            "are ~μs and omitted."
        ),
        "asr_baseline": str(args.baseline),
        "languages": {},
    }

    print(
        f"{'lang':4} {'corpus':10} {'punct':8} "
        f"{'punct_p50':>9} {'asr_p50':>8} {'e2e_proj':>9} {'Δms':>7} {'×':>6}"
    )
    for lang in langs:
        base = asr.get(lang)
        if not base:
            print(f"[warn] no ASR baseline for {lang}", file=sys.stderr)
            continue
        post = lang_post(lang)

        def xlmr_only(t: str, _x=xlmr) -> str:
            return _x(t)

        def xlmr_fix(t: str, _x=xlmr, _p=post) -> str:
            out = _x(t)
            return _p(out) if _p else out

        backends = {
            "basic": basic_restore,
            "xlmr": xlmr_only,
        }
        if post is not None:
            backends["xlmr_fix"] = xlmr_fix

        corpora = {
            "l1_fixtures": load_fixture_texts(args.fixtures_dir, lang),
            "punct_wiki": load_punct_texts(args.punct_dir, lang),
        }

        lang_row: dict = {
            "asr_baseline_ms": {
                "p50": base["asr_p50_ms"],
                "p95": base["asr_p95_ms"],
            },
            "ci_e2e_ms": {
                "p50": base["e2e_p50_ms"],
                "p95": base["e2e_p95_ms"],
            },
            "corpora": {},
        }

        for corpus_name, texts in corpora.items():
            if not texts:
                continue
            corpus_row = {"texts": len(texts), "backends": {}}
            for bname, fn in backends.items():
                timed = time_fn(fn, texts)
                asr_p50 = base["asr_p50_ms"]
                proj = asr_p50 + timed["p50_ms"]
                delta = timed["p50_ms"]
                ci_e2e = base["e2e_p50_ms"] or asr_p50
                ratio = proj / ci_e2e if ci_e2e else 0.0
                entry = {
                    **timed,
                    "projected_e2e_p50_ms": round(proj, 1),
                    "delta_vs_ci_e2e_ms": round(delta, 1),
                    "projected_e2e_over_ci": round(ratio, 3),
                }
                corpus_row["backends"][bname] = entry
                print(
                    f"{lang:4} {corpus_name:10} {bname:8} "
                    f"{timed['p50_ms']:9.1f} {asr_p50:8.0f} "
                    f"{proj:9.1f} {delta:7.1f} {ratio:6.2f}×"
                )
            lang_row["corpora"][corpus_name] = corpus_row
        report["languages"][lang] = lang_row

    print("\n## Headline (L1 fixtures ≈ real ASR hyp length)")
    print(
        f"{'lang':4} {'ASR':>8} {'basic':>8} {'xlmr':>8} "
        f"{'+fix':>8} {'E2E+xlmr':>10} {'slowdown':>9}"
    )
    for lang, row in report["languages"].items():
        corp = row["corpora"].get("l1_fixtures", {})
        backs = corp.get("backends", {})
        asr_p50 = row["asr_baseline_ms"]["p50"]
        basic = backs.get("basic", {}).get("p50_ms", 0.0)
        xlmr_ms = backs.get("xlmr", {}).get("p50_ms", 0.0)
        fix = backs.get("xlmr_fix", backs.get("xlmr", {}))
        fix_ms = fix.get("p50_ms", 0.0)
        proj = fix.get("projected_e2e_p50_ms", asr_p50 + fix_ms)
        slow = fix.get("projected_e2e_over_ci", 0.0)
        print(
            f"{lang:4} {asr_p50:8.0f} {basic:8.2f} {xlmr_ms:8.1f} "
            f"{fix_ms:8.1f} {proj:10.0f} {slow:8.2f}×"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"\n[ok] wrote {args.output}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
