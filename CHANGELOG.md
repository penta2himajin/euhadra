# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project is `0.x`: minor versions may contain breaking changes, as
stated in the README's Stability section.

## [Unreleased]

### Added

- **Voice activity detection** (`euhadra::vad`). Until now the recording
  reached the ASR adapter exactly as captured, so 30 seconds of audio
  with 5 seconds of speech in it fed 25 seconds of silence to the model —
  which is where "Thanks for watching" and 「ご視聴ありがとうございました」
  come from. `PipelineBuilder::vad` puts a detector ahead of the adapter,
  and the silence stops arriving.

  It sits ahead of the adapter rather than inside `mic` capture so that
  WAV input gets the same treatment.

  - `VadBackend` / `VadStream` — score frames for speech.
  - `Segmenter` — turn those scores into utterance boundaries. Separate
    from the backend on purpose: swapping detectors changes which frames
    are speech and nothing about where the cuts land.
  - `EarshotVad` (feature `vad`) — the recommended backend. A 40 KiB
    network embedded in the pure-Rust [`earshot`] crate: no ONNX runtime,
    no model download, nothing to redistribute. 16 kHz only.
  - `VadBackend::default_threshold` — score calibration is a property of
    the backend, not of the segmentation policy, so
    `SegmenterConfig::threshold` is an `Option` that defers to it.
    `EarshotVad` calibrates to 0.2; applying `EnergyVad`'s 0.5 to it cost
    +0.05 WER and looked like a bad backend rather than a bad number.
  - `EnergyVad` — level against an adapting noise floor, no dependency at
    all, any sample rate. A stopgap: it answers "louder than the room?",
    so a keyboard passes and a quiet speaker eventually does not.

  `SegmenterConfig`'s defaults lean towards waiting — 700 ms of silence
  to close an utterance, well above Silero's 100 ms. The asymmetry is
  deliberate: under-segmenting costs latency, while over-segmenting hands
  the model a fragment, and a model given a fragment answers fluently and
  wrongly (#134 measured a 3-second prefix producing "However, due to the
  slow communication.").

- **Incremental output.** `Session::partials` delivers one transcript per
  utterance while the speaker is still talking. Lossy, so ignoring it
  cannot stall a session; dropping the receiver also skips the
  per-utterance ASR pass.

  This is not streaming ASR — no bundled adapter has a streaming API, and
  re-transcribing a growing prefix was measured and rejected in #134
  (182% / 350% churn, warm RTF 1.54).

- **`FinalPass`** — what the returned transcript is computed from.
  `SpeechOnly` (default) transcribes the detected speech joined as one
  utterance, so the silence is gone but the model still sees each
  utterance whole; `WholeUtterance` leaves the final text byte-identical
  to a pipeline with no detector; `JoinSegments` concatenates the
  partials in a single ASR pass and is the only policy that inherits
  segmentation errors in full.

- `Diagnostics::speech_segments` reports what the detector decided, and
  `Stage::Vad` reports a detector that could not run — a rate mismatch
  degrades to the unsegmented path rather than failing the session.

- `RecordingAsr` (feature `testing`) — a mock that records the audio it
  was handed, so a test can tell "the adapter saw the silence" from "the
  adapter saw only the speech".

### Measured

`docs/benchmarks/vad_delta_wer.md`, on the FLEURS en/ja subsets with 5 s
of silence added either side of each utterance, using the models euhadra
actually ships (`en` = canary-180m-flash INT8, `ja` =
parakeet-tdt_ctc-0.6b-ja). The clean numbers match `ci_baseline.json`
exactly.

| condition | en (WER) | Δ | ja (CER) | Δ |
|---|---|---|---|---|
| clean, no detector | 0.0762 | — | 0.0724 | — |
| padded, **no detector** | **0.1875** | **+0.1114** | 0.1211 | +0.0487 |
| padded, `SpeechOnly` | 0.0762 | **+0.0000** | 0.0759 | +0.0035 |
| padded, `JoinSegments` (−45 dBFS) | **0.3940** | **+0.3178** | 0.1376 | +0.0652 |

- **Silence does real damage**, and the amount depends on the decoder.
  Asked to transcribe 10 s of silence alone, Canary returns a runaway
  repetition (`".S. Sometimes it's a long way, …"` ×50) or a fluent
  invented paragraph; Parakeet returns 「心の声。」. An attention
  decoder chooses its own output length, a transducer's is bounded by
  acoustic frames.
- **The default configuration removes it** — Δ+0.0000 to +0.0150.
- **`SpeechOnly` vs `JoinSegments` is a 4.6× difference from the policy
  alone**, off identical segmentation. Dropping the silence does not
  require cutting anything, and now that is measured rather than argued.

Still unmeasured: zh / ko / es (no model bundles), and the ΔWER run is
not wired into CI — it needs model downloads, so it would belong with
`evaluate (ASR live smoke)` rather than the new `vad` job.

### Resolved by the measurement

- **No text-side hallucination removal.** #133's gate was whether
  hallucinated text survives a detector. It does not, so euhadra does
  not acquire a per-language blacklist — the class of work `docs/spec.md`
  §11.4 identifies as not centrally scalable.

[`earshot`]: https://crates.io/crates/earshot

## [0.2.0] — 2026-08-07

Breaking. `0.x`, so a minor bump carries removals — see Removed.

### Added

- **`OnnxEntityRecognizer`** (`onnx`) — named-entity recognition via ONNX
  token classification, on the same inference path as the punctuation
  restorer. Detects PER / LOC / ORG / MISC with codepoint offsets.
  `scripts/setup_ner.sh` fetches `dslim/distilbert-NER` (MIT, 249 MB
  fp32; upstream ships no quantised graph). As a `TextProcessor` it
  leaves the text untouched and reports findings as the new
  `CorrectionKind::EntityDetected`; `detect()` returns the structured
  form. Narrowing `PhonemeCorrector`'s candidates with it is not wired
  yet — processors cannot pass metadata to one another.
- **Spanish filler F1 is gated in CI.** Spanish is the one language whose
  gold cannot be committed — CIEMPIESS Test is CC-BY-SA-4.0 — so the job
  fetches transcripts, generates annotations into a gitignored cache, and
  keeps only the score. It is drift detection between the Rust filter and
  the Python generator rather than a quality measurement; `docs/spec.md`
  §11.4 says what it does and does not catch.
- **Release workflow.** A `v*.*.*` tag now publishes to crates.io via
  Trusted Publishing (OIDC), with no long-lived token stored.
- **`CONTRIBUTING.md`**, leading with the request this project most
  needs: native-speaker review of the evaluation gold sets.
- **`CHANGELOG.md`** — this file.

### Removed

- **`OnnxEmbeddingFilter`, and with it `Segmenter`, `Segment`,
  `segment()` and `FillerLexicon`.** Deprecated and unwired since the
  rule-based filters measured better in every language. The calibration
  bench (`examples/bench_embedder.rs`) went too. Use
  `FillerFilter::for_language`.

### Fixed

- **docs.rs builds again.** `feature(doc_auto_cfg)` was removed in Rust
  1.92.0 and merged into `doc_cfg`; since docs.rs builds on nightly, the
  old name was a hard error, which is why 0.1.0 has no rendered
  documentation. 0.1.0 stays broken — a rebuild of a published version
  hits the same error — so this takes effect from 0.2.0.

### Documentation

- **Phase 1 scoped to what euhadra actually ships.** The macOS OS Shell
  and the `LlmRefiner` / `ContextProvider` implementations moved out of
  Phase 1 (#122, #123). Two of the OS Shell's four responsibilities were
  already solved in cross-platform Rust, and deferring the other two left
  nothing to justify a UniFFI boundary.
- **Language support policy** (`docs/spec.md` §11.4): euhadra does not
  claim a language it cannot measure. Records what is measured versus
  what CI gates, that the in-tree gold is Claude-drafted and unreviewed,
  and that the tiers cannot all scale to 100 languages the same way.

## [0.1.0] — 2026-08-06

First release. Published manually; there is no `v0.1.0` tag, because
tagging it after the fact would fail the release workflow's publish step
with `crate version 0.1.0 already uploaded`.

### Added

- Async pipeline runtime (tokio) with cancellation and backpressure.
- Adapter traits: `AsrAdapter`, `TextFilter`, `TextProcessor`,
  `LlmRefiner`, `ContextProvider`, `OutputEmitter`.
- ASR adapters: WhisperLocal, and behind `onnx` — Parakeet, Paraformer,
  Canary, Dolphin, SenseVoice, Whisper-ONNX.
- Tier 1 filler removal for English, Japanese, Chinese, Korean and
  Spanish.
- Tier 2 processing: self-correction detection, punctuation restoration
  (rule-based and BERT/ONNX), inverse text normalisation, spoken-form
  expansion, phoneme-distance dictionary correction, paragraph splitting.
- `StdoutEmitter`; `ClipboardEmitter` behind `clipboard`; microphone
  capture behind `mic`; the `euhadra` binary behind `cli`.
- Evaluation harness and mock adapters behind `testing`.

[Unreleased]: https://github.com/penta2himajin/euhadra/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/penta2himajin/euhadra/releases/tag/v0.2.0
[0.1.0]: https://crates.io/crates/euhadra/0.1.0
