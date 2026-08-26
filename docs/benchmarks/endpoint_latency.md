# Endpoint latency baseline

**Issue**: #148 (parent #147)  
**Runner**: `examples/eval_endpoint.rs`  
**Gate file**: [`ci_baseline_endpoint.json`](./ci_baseline_endpoint.json)

## What this measures

| KPI | Definition |
|---|---|
| `endpoint_to_partial_ms` | Segmenter closes an utterance (processable) → that utterance's `Partial` is ready |
| `endpoint_to_final_ms` | Last segment close → `SessionResult` ready (includes `FinalPass`) |
| `segment_rtf` | Mean (partial ASR time / speech duration) over timed segments |

This is **not** the L1 ASR/E2E p50 in `ci_baseline.json` (full-file wall clock).

`min_silence` hangover is excluded from the KPI numerator.

## Configuration

- Corpus: FLEURS subset, en / ja / es, n=30 (same as `evaluate-vad`)
- VAD: `EarshotVad` + default `SegmenterConfig`
- Final pass: `SpeechOnly` (library default)
- Models: Canary-180M-Flash INT8 (en, es), Parakeet TDT-CTC 0.6B ja (ja)
- Audio fed in ~100 ms chunks to exercise the live segmenter path

`es` was added under #150. The first `es` baseline row was seeded by
scaling an agent measurement by the observed en (GHA / agent) ratio so
the relative gate matches `ubuntu-latest`; recalibrate from CI if the
gate trips on a warm run that is not a real regression.

## Seed measurement

Baseline numbers are from **GitHub Actions `ubuntu-latest`** (the gate
runner), not a developer workstation. Agent / laptop hosts are typically
~2× faster; do not rewrite the baseline from a fast host or CI will fail
on relative regression.

Re-measure on CI (or an equivalent runner) when updating intentionally:

```bash
cargo run --release --features onnx,vad,testing --example eval_endpoint -- \
  --canary-en-dir vendor/canary_en \
  --parakeet-ja-dir vendor/parakeet_ja \
  --langs en,ja,es \
  --write-baseline docs/benchmarks/ci_baseline_endpoint.json
```

CI gates with `--baseline docs/benchmarks/ci_baseline_endpoint.json`
(relative fail bands + absolute warn floors, same idea as L1). Absolute
warns at ≥ 1 s are expected on current GHA numbers for final (and often
partial) — they surface user-perceived slowness without failing the job.
