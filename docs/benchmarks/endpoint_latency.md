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

- Corpus: FLEURS subset, en / ja, n=30 (same as `evaluate-vad`)
- VAD: `EarshotVad` + default `SegmenterConfig`
- Final pass: `SpeechOnly` (library default)
- Models: Canary-180M-Flash INT8 (en), Parakeet TDT-CTC 0.6B ja (ja)
- Audio fed in ~100 ms chunks to exercise the live segmenter path

## Seed measurement (this agent host)

See `ci_baseline_endpoint.json` `generated` field. Re-measure on CI runners when updating the baseline intentionally:

```bash
cargo run --release --features onnx,vad,testing --example eval_endpoint -- \
  --canary-en-dir vendor/canary_en \
  --parakeet-ja-dir vendor/parakeet_ja \
  --langs en,ja \
  --write-baseline docs/benchmarks/ci_baseline_endpoint.json
```

CI gates with `--baseline docs/benchmarks/ci_baseline_endpoint.json` (relative + absolute warn floors, same idea as L1).
