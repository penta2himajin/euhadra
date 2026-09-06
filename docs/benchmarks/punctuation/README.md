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
