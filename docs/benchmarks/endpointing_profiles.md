# Endpointing profiles — quality vs latency

**Parent**: #147  
**Related**: [`endpoint_latency.md`](./endpoint_latency.md), [`vad_delta_wer.md`](./vad_delta_wer.md), `docs/spec.md` §3.7

Library defaults favour **under-segmentation** (wait longer before closing an
utterance). That is intentional: cutting speech mid-sentence is destructive and
hard to recover from (#134), while waiting only costs latency.

This note documents the shipped default and a small set of **candidate**
knobs for latency-sensitive call sites. Candidates are starting points for
measurement, not alternate defaults.

## Knobs that matter

| Setting | Default | Effect on endpointing |
|---|---|---|
| `SegmenterConfig::min_silence` | **700 ms** | Continuous silence required to close an utterance. Primary hysteresis. |
| `SegmenterConfig::max_speech` | 30 s | Safety cut for speakers who never pause. Deliberate over-segmentation. |
| `SegmenterConfig::min_speech` | 120 ms | Ignores brief noise before opening speech. |
| `FinalPass` | `SpeechOnly` | Source for the session-end transcript — not a second-pass decode (#146). |

`min_silence` hangover is **not** part of the endpoint-latency KPI numerator
(`endpoint_to_partial_ms` / `endpoint_to_final_ms`). It is a user-facing wait
you chose; the KPI starts when the segmenter has already closed.

## Profiles

### Quality (shipped default)

```text
min_silence = 700 ms
max_speech  = 30 s
FinalPass   = SpeechOnly
```

Use for dictation and any path where a fluent wrong fragment is worse than a
few hundred milliseconds of silence wait. Matches `SegmenterConfig::default()`
and what `euhadra record` uses unless overridden.

Measured under this profile:

- VAD ΔWER (en / ja): [`vad_delta_wer.md`](./vad_delta_wer.md)
- Endpoint latency (en / ja): [`endpoint_latency.md`](./endpoint_latency.md) /
  [`ci_baseline_endpoint.json`](./ci_baseline_endpoint.json)

### Low-latency candidate (not default)

```text
min_silence = 300–400 ms
max_speech  = 15–20 s   # optional; only if long monologues must still emit
FinalPass   = SpeechOnly
```

Shorter `min_silence` closes earlier after a pause, so partials appear sooner.
It also raises the chance of cutting mid-sentence breaths. Prefer this only
after re-running:

```bash
# ΔWER / over-segmentation under the candidate silence
cargo run --release --features onnx,vad --example eval_vad -- \
  --canary-en-dir vendor/canary_en \
  --parakeet-ja-dir vendor/parakeet_ja \
  --langs en,ja --noise-db=-45 --detectors none,earshot \
  --policies speech-only

# Segment-close → partial / final under the same SegmenterConfig
# (wire the candidate into a local SegmenterConfig if measuring by hand)
cargo run --release --features onnx,vad,testing --example eval_endpoint -- \
  --canary-en-dir vendor/canary_en \
  --parakeet-ja-dir vendor/parakeet_ja \
  --langs en,ja
```

CLI override for live capture:

```bash
cargo run --features cli -- record \
  --whisper-cli /path/to/whisper-cli \
  --model /path/to/ggml-base.bin \
  --language en \
  --min-silence-ms 350
```

### Aggressive Silero-style (do not ship)

```text
min_silence ≈ 100 ms
FinalPass   = JoinSegments   # especially dangerous
```

Silero’s own short `min_silence` targets endpointing demos, not dictation
quality. #134 showed short prefixes producing fluent wrong English (“However,
due to the slow communication.”) and short Japanese hallucinations. Under
identical segmentation, `JoinSegments` on Canary + room noise scored ~4× worse
ΔWER than `SpeechOnly` ([`vad_delta_wer.md`](./vad_delta_wer.md)). Do not pair a
short silence with `JoinSegments` unless the product explicitly accepts
fragment transcripts.

## Open-interval draft (explicitly non-default)

Periodically re-transcribing the still-open buffer before the segmenter closes
was considered under #147 and rejected as a default by #134 (prefix churn and
worse-than-realtime warm RTF). If revisited, fix update interval, max buffer,
and display conditions **before** measuring — and keep it opt-in.

## How to try the utterance path

Library (already the supported API):

```rust
use euhadra::prelude::*;
use euhadra::vad::{EarshotVad, SegmenterConfig};

let pipeline = Pipeline::builder()
    .asr(/* ... */)
    .vad(EarshotVad::new())
    .segmenter_config(SegmenterConfig::default())
    .final_pass(FinalPass::SpeechOnly)
    .build()?;

let mut session = pipeline.session();
// take session.partials; feed session.audio; session.finish().await
```

CLI (`cli` implies `vad`):

```bash
cargo run --features cli -- record --model … --language en
# Partials on stderr as each utterance closes; Ctrl+C for FinalPass.

cargo run --features cli -- dictate --file speech.wav --model … --vad
```
