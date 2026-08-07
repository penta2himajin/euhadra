# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project is `0.x`: minor versions may contain breaking changes, as
stated in the README's Stability section.

## [Unreleased]

### Added

- **`OnnxEntityRecognizer`** (`onnx`) — named-entity recognition via ONNX
  token classification, on the same inference path as the punctuation
  restorer. Detects PER / LOC / ORG / MISC with codepoint offsets.
  `scripts/setup_ner.sh` fetches `dslim/distilbert-NER`. As a
  `TextProcessor` it leaves the text untouched and reports findings as
  the new `CorrectionKind::EntityDetected`; `detect()` returns the
  structured form. Narrowing `PhonemeCorrector`'s candidates with it is
  not wired yet.

### Removed

- **`OnnxEmbeddingFilter`, and with it `Segmenter`, `Segment`,
  `segment()` and `FillerLexicon`.** Deprecated and unwired since the
  rule-based filters measured better in every language. The calibration
  bench (`examples/bench_embedder.rs`) went too. Use
  `FillerFilter::for_language`.

### Fixed

- **docs.rs builds again.** `feature(doc_auto_cfg)` was removed in Rust
  1.92.0 and merged into `doc_cfg`; since docs.rs builds on nightly, the
  old name was a hard error and 0.1.0 has no rendered documentation.

### Added

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

[Unreleased]: https://github.com/penta2himajin/euhadra/compare/master...HEAD
[0.1.0]: https://crates.io/crates/euhadra/0.1.0
