# Punctuation bake-off

Slot F1 / terminal accuracy / latency for Tier 2 punctuation restorers.

## Reproduce

```bash
scripts/setup_punct_xlmr.sh
scripts/build_punct_annotations.py --langs en,ja,zh,ko,es --per-lang 30
scripts/eval_punctuation.py \
  --annotations data/punct_eval \
  --backends basic,xlmr \
  --xlmr-dir vendor/punct_xlmr \
  --output docs/benchmarks/punctuation/bakeoff.json
```

Smoke (no model download):

```bash
scripts/eval_punctuation.py \
  --annotations tests/evaluation/annotations/punct_smoke.jsonl \
  --backends basic
```

## Latest result

See [`bakeoff.json`](./bakeoff.json). Summary (CPU, 4-core class host, 2026-09-06):

| Lang | basic F1 | xlmr F1 | basic term | xlmr term | xlmr p50 |
|---|---:|---:|---:|---:|---:|
| en | 0.35 | **0.78** | 0.97 | 1.00 | ~243 ms |
| ja | 0.35 | **0.63** | 1.00 | 1.00 | ~242 ms |
| ko | 0.40 | **0.70** | 1.00 | 1.00 | ~241 ms |
| zh | 0.22 | **0.35** | 1.00 | 1.00 | ~243 ms |
| es | 0.21 | **0.75** | 1.00 | 1.00 | ~253 ms |

`basic` is `BasicPunctuationRestorer` (terminal mark only). `xlmr` is
`1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase` via the
upstream `punctuators` package. Gold is synthetic (strip punct from
clean Wikipedia sentences) per `docs/spec.md` §11.4.

> **es note:** an earlier draft reported xlmr es ≈ 0.33. That was an
> annotation artefact (U+200B in Wikipedia extracts). Invisibles are
> now stripped at build/eval time.

Write-up: [`docs/model-upgrade-candidates.md`](../../model-upgrade-candidates.md) §7.

## zh follow-up (`ZhPunctNormalizer`)

XLM-R emits `，` where written Chinese wants `、`, and over-inserts `。`
mid-clause. The product post-pass covers both:

| Backend / scoring | zh F1 | Notes |
|---|---:|---|
| `basic` (strict) | 0.22 | terminal only |
| `xlmr` (strict) | 0.35 | glyph mismatch + over-seg |
| `xlmr_zh` = xlmr + `ZhPunctNormalizer` (strict) | **0.67** | enum `，`→`、` + demote mid `。`→`，` |
| `xlmr` with `--equiv-zh-commas` | 0.74 | upper bound if `、`≡`，` |
| `xlmr_zh` with `--equiv-zh-commas` | **0.88** | demote recovers most residual |

```bash
scripts/eval_punctuation.py \
  --annotations data/punct_eval --langs zh \
  --backends basic,xlmr,xlmr_zh \
  --output docs/benchmarks/punctuation/bakeoff_zh.json

scripts/eval_punctuation.py \
  --annotations data/punct_eval --langs zh \
  --backends xlmr,xlmr_zh --equiv-zh-commas \
  --output docs/benchmarks/punctuation/bakeoff_zh_equiv.json
```

## Over-segmentation post-passes (ja / es)

| Backend | Lang | F1 | Δ vs xlmr |
|---|---|---:|---:|
| `xlmr` | ja | 0.63 | — |
| `xlmr_ja` | ja | **0.69** | +0.06 |
| `xlmr` | es | 0.75 | — |
| `xlmr_es` | es | **0.79** | +0.05 |

```bash
scripts/eval_punctuation.py \
  --annotations data/punct_eval --langs ja,es \
  --backends xlmr,xlmr_ja,xlmr_es \
  --output docs/benchmarks/punctuation/bakeoff_overseg_ja_es.json
```

Chain each normalizer only on its language path (`ZhPunctNormalizer` on
Japanese would wrongly demote `。` → `，`).
