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
| en | 0.35 | **0.78** | 0.97 | 1.00 | ~246 ms |
| ja | 0.35 | **0.63** | 1.00 | 1.00 | ~246 ms |
| ko | 0.40 | **0.70** | 1.00 | 1.00 | ~245 ms |
| zh | 0.22 | **0.35** | 1.00 | 1.00 | ~247 ms |
| es | 0.20 | **0.33** | 0.95 | 0.95 | ~257 ms |

`basic` is `BasicPunctuationRestorer` (terminal mark only). `xlmr` is
`1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase` via the
upstream `punctuators` package. Gold is synthetic (strip punct from
clean Wikipedia sentences) per `docs/spec.md` §11.4.

Write-up: [`docs/model-upgrade-candidates.md`](../../model-upgrade-candidates.md) §7.

## zh follow-up (`ZhPunctNormalizer`)

XLM-R emits `，` where written Chinese wants the enumeration comma `、`.
Two measurements isolate that:

| Backend / scoring | zh F1 | Notes |
|---|---:|---|
| `basic` (strict) | 0.22 | terminal only |
| `xlmr` (strict) | 0.35 | glyph mismatch dominates |
| `xlmr_zh` = xlmr + `ZhPunctNormalizer` (strict) | **0.59** | converts short-run `，` → `、` |
| `xlmr` with `--equiv-zh-commas` | 0.74 | upper bound if `、`≡`，` |
| `xlmr_zh` with `--equiv-zh-commas` | 0.74 | normalizer does not hurt positions |

Reproduce:

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

Reports: [`bakeoff_zh.json`](./bakeoff_zh.json), [`bakeoff_zh_equiv.json`](./bakeoff_zh_equiv.json).

