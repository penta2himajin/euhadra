# euhadra

A programmable voice input framework — ASR, LLM refinement, and OS integration as composable adapters.

> **euhadra** is named after the Japanese land snail genus *Euhadra* (マイマイ属).
> ear → cochlea → snail → *Euhadra* — a chain from hearing to the framework's identity as a Japan-born OSS project.

## What it does

euhadra provides an async pipeline that transforms speech into clean, formatted text — **without requiring an LLM**:

```
Microphone / WAV
    → ASR (whisper.cpp local)
    → TextFilter (filler removal: um, uh, えーと...)
    → TextProcessor (self-correction, punctuation, capitalization)
    → [LlmRefiner] (seam only — no implementation ships)
    → Output (clipboard / stdout)
```

Each stage is a Rust trait. Swap any component without touching the rest.

## Install

### As a library

```toml
[dependencies]
euhadra = "0.1"
```

The default build is deliberately lean — the pipeline runtime plus the rule-based
Tier 1/2 text processing, seven direct dependencies, no ML runtime and no system
libraries. Opt into the rest:

| Feature | Adds | Cost |
|---------|------|------|
| *(default)* | Pipeline runtime, filler filters, self-correction, punctuation, ITN | pure Rust |
| `onnx` | ONNX ASR adapters, BERT punctuation, embeddings, G2P | ONNX Runtime; needs Rust 1.88 |
| `mic` | Microphone capture (`cpal`) | ALSA headers on Linux (`libasound2-dev`) |
| `clipboard` | `ClipboardEmitter` (`arboard`) | — |
| `cli` | The `euhadra` binary; implies `mic` + `clipboard` | needs Rust 1.85 |
| `testing` | Mock adapters and the WER/CER evaluation harness | — |

Microphone capture is behind a feature because `cpal` links ALSA on Linux, and a
consumer who only wants the text-processing tiers should not have to install
system packages to compile.

`testing` is for building test doubles against euhadra's traits, and for running
the evaluation harness. It is off by default because neither is library surface;
put it under `[dev-dependencies]` rather than `[dependencies]`.

### Stability

This is `0.x`. The adapter traits — `AsrAdapter`, `TextFilter`, `TextProcessor`,
`LlmRefiner`, `ContextProvider`, `OutputEmitter` — are the part meant to be
stable, because implementing one is the reason to depend on this crate. They
will still change if a real integration shows they are wrong, and Phase 2's
`Command` and `StructuredInput` output modes are expected to move `LlmRefiner`.

Everything around them is fluid: the builder, the concrete adapters, the
evaluation harness. Minor versions may break either group until `1.0`. Pin an
exact version if that matters to you.

### As a CLI

```bash
cargo install euhadra --features cli
```

## Getting Started

### Prerequisites

1. **Rust** (1.78+): https://rustup.rs
2. **whisper.cpp**: local ASR engine

Build whisper.cpp:

```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
cmake -B build && cmake --build build --config Release
bash models/download-ggml-model.sh base
```

### Build from source

```bash
git clone https://github.com/penta2himajin/euhadra
cd euhadra
cargo build --features cli
```

### Transcribe a WAV file

```bash
# Raw whisper transcription
cargo run --features cli -- transcribe \
  --file speech.wav \
  --whisper-cli /path/to/whisper.cpp/build/bin/whisper-cli \
  --model /path/to/whisper.cpp/models/ggml-base.bin \
  --language en
```

### Full pipeline (filter + process)

```bash
# English: filler removal + self-correction + punctuation
cargo run --features cli -- dictate \
  --file speech.wav \
  --whisper-cli /path/to/whisper-cli \
  --model /path/to/ggml-base.bin \
  --language en

# Japanese: filler removal (えーと, あの, etc.) + ASR artifact cleanup
cargo run --features cli -- dictate \
  --file speech.wav \
  --whisper-cli /path/to/whisper-cli \
  --model /path/to/ggml-base.bin \
  --language ja
```

### Record from microphone

```bash
# Record → transcribe → print to stdout
cargo run --features cli -- record \
  --whisper-cli /path/to/whisper-cli \
  --model /path/to/ggml-base.bin \
  --language en

# Record → transcribe → copy to clipboard
cargo run --features cli -- record \
  --whisper-cli /path/to/whisper-cli \
  --model /path/to/ggml-base.bin \
  --language en \
  --clipboard
```

Press Ctrl+C to stop recording.

### Use as a library

```rust
use euhadra::prelude::*;
use euhadra::whisper_local::WhisperLocal;

#[tokio::main]
async fn main() {
    // Minimal: ASR + filler filter + self-correction + punctuation.
    // Only .asr() is required; no LLM is involved anywhere below.
    let pipeline = Pipeline::builder()
        .asr(WhisperLocal::new("whisper-cli", "ggml-base.bin").with_language("en"))
        .filter(FillerFilter::for_language(Language::English))
        .processor(SelfCorrectionDetector::new())
        .processor(BasicPunctuationRestorer)
        .emitter(StdoutEmitter)
        .build()
        .unwrap();

    // Load audio and run it through every configured tier
    let audio = euhadra::whisper_local::read_wav("speech.wav".as_ref()).unwrap();
    let result = pipeline.transcribe(&[audio]).await.unwrap();

    // result.raw_text  — original ASR output
    // result.output    — filtered + processed text
}
```

For Japanese, change the ASR language and the filter language — nothing else:

```rust
let pipeline = Pipeline::builder()
    .asr(WhisperLocal::new("whisper-cli", "ggml-base.bin").with_language("ja"))
    .filter(FillerFilter::for_language(Language::Japanese))
    .processor(SelfCorrectionDetector::new())
    .processor(BasicPunctuationRestorer)
    .emitter(ClipboardEmitter::new())   // requires the `clipboard` feature
    .build()
    .unwrap();
```

`FillerFilter::for_language` picks the segmentation the script needs: whitespace
for English, Spanish and Korean, `、` for Japanese, `，` for Chinese. Pairing
these by hand is a real hazard — `SimpleFillerFilter` splits on whitespace, so a
Japanese utterance arrives as a single token and one that opens with a filler is
removed in full, leaving an empty transcript and no error. Prefer
`for_language`; reach for a concrete filter only when you need to customise its
lexicon, and then keep it matched to the language yourself.

## Three-tier text processing

euhadra processes ASR output through three independent layers, each optional:

| Tier | Component | What it does | LLM? | Size |
|------|-----------|-------------|------|------|
| 1 | **TextFilter** | Filler removal (um, uh, えーと) | No | 0 MB (rules) or 33 MB (embeddings) |
| 2 | **TextProcessor** | Punctuation, capitalization, self-correction | No | 0 MB (rules) or 5-50 MB (ONNX) |
| 3 | **LlmRefiner** | Tone adjustment, context-adaptive rewriting | Yes | Trait only — nothing ships |

Tier 1 + 2 alone produce clean, punctuated text without any LLM or network calls.

### Languages

Text processing is per-language work — filler lexicons are hand-written, and
punctuation and self-correction behave differently per script. euhadra therefore
only claims a language it can measure.

| Language | Filler filter | ASR baseline (CI) | Filler F1 gold |
|----------|---------------|-------------------|----------------|
| English  | ✅ | ✅ | ✅ in tree |
| Japanese | ✅ | ✅ | ✅ in tree |
| Chinese  | ✅ | ✅ | ✅ in tree |
| Korean   | ✅ | ✅ | ✅ in tree |
| Spanish  | ✅ | ✅ | ⚠️ generated in CI |

The gold sets themselves are provisional. Most were drafted by Claude and have
**not yet been reviewed by native speakers** — this covers the filler sets for
English, Japanese, Chinese and Korean, and every self-correction set. The one
language whose annotations come from human markup (Spanish, via CIEMPIESS) is
the one not currently measured. Native-speaker review of the existing five is
the single most useful contribution to this project right now; see
[CONTRIBUTING.md](CONTRIBUTING.md).

Everything else is unmeasured. The pipeline will still run on other languages —
ASR is a pluggable adapter and several backends are multilingual — but the Tier 1
and Tier 2 stages have no lexicon for them and no way to tell you when they are
wrong, so treat the output as unvalidated.

Spanish is the case worth understanding. Its gold set is not missing; it cannot
be shipped. The source corpus (CIEMPIESS Test) is CC-BY-SA-4.0, so committing
derived annotations would propagate ShareAlike into this MIT/Apache tree. The
generator is checked in and writes to a gitignored cache instead, and only the
resulting scores are committed. That posture is deliberate — but the CI wiring
that would run it never landed, so in practice Spanish went unverified, and a
defect that silently disabled filler removal for punctuated input survived until
every language was run by hand.

## CLI reference

```
euhadra dictate     Transcribe a WAV file through the full pipeline
  --file <path>       WAV file (16-bit PCM)
  --whisper-cli       Path to whisper-cli binary
  --model             Path to GGML model
  --language          Language hint (en, ja, etc.)
  --no-filter         Skip filler removal
  --no-process        Skip text processing (punctuation, self-correction)

euhadra record      Record from microphone through the full pipeline
  --whisper-cli       Path to whisper-cli binary
  --model             Path to GGML model
  --language          Language hint
  --clipboard         Output to clipboard instead of stdout
  --no-filter         Skip filler removal
  --no-process        Skip text processing

euhadra transcribe  Whisper-only transcription (no pipeline)
  --file <path>       WAV file
  --whisper-cli       Path to whisper-cli binary
  --model             Path to GGML model
  --language          Language hint
```

## ONNX feature (optional)

For higher-quality text processing with ML models (no Python required):

```bash
cargo build --features onnx
```

This enables:
- `OnnxPunctuationRestorer` — CNN-BiLSTM punctuation + capitalization model
- `WhisperOnnxAdapter` — Whisper-large-v3-turbo ASR via ONNX Runtime (encoder + KV-cached decoder loop). Best CER+RTF for Korean on CPU per the [#83 backend bench](docs/korean-asr-alternatives.md): 1.09% / 0.484 on FLEURS-ko with the `q4` quantisation.

Without the `onnx` feature, euhadra uses rule-based implementations with zero ML dependencies.

### Whisper-ONNX setup

```bash
# Downloads tokenizer + q4 ONNX bundle (~900 MB) into vendor/whisper_onnx_turbo
scripts/setup_whisper_onnx_turbo.sh

# Use as the ASR stage
cargo run --release --features onnx --example bench_whisper_onnx_ko -- \
    --model-dir vendor/whisper_onnx_turbo \
    --manifest data/fleurs_subset/ko/manifest.tsv \
    --audio-root data/fleurs_subset
```

From Rust:

```rust
use euhadra::whisper_onnx::WhisperOnnxAdapter;

let asr = WhisperOnnxAdapter::load("vendor/whisper_onnx_turbo")?
    .with_language("ko");
// ...then pass `asr` into PipelineBuilder::asr().
```

## Architecture

```
[euhadra core (Rust)]
    ├── Pipeline runtime (tokio async)
    ├── ASR adapter trait         → WhisperLocal (whisper.cpp), ParakeetAdapter, ParaformerAdapter (zh)
    ├── TextFilter trait          → FillerFilter::for_language → Simple / Japanese / Chinese / Spanish
    ├── TextProcessor trait       → SelfCorrectionDetector, BasicPunctuationRestorer,
    │                                SpokenFormNormalizer, InverseTextNormalizer,
    │                                PhonemeCorrector, ParagraphSplitter
    ├── LlmRefiner trait          → no implementation (see below)
    ├── ContextProvider trait     → no implementation (see below)
    ├── OutputEmitter trait       → StdoutEmitter, ClipboardEmitter [clipboard]
    ├── [mic] Microphone capture  → cpal, cross-platform
    └── [onnx] ONNX backends      → OnnxPunctuationRestorer, and the embedder / G2P
                                    that PhonemeCorrector and ParagraphSplitter
                                    use when available
```

euhadra is a library, not an application. It ships the traits; native OS
integration — accessibility APIs, global hotkeys, on-device LLM bridges — is
something a consuming app provides, not something euhadra links in. Microphone
capture and clipboard insertion are the exceptions, and only because they turned
out to be solvable in cross-platform Rust.

`LlmRefiner` and `ContextProvider` are therefore defined but unimplemented. That
is deliberate rather than unfinished: Tiers 1 and 2 have ground truth and are
gated on WER/CER/F1 in CI, whereas a free-form LLM rewrite has no test that can
assert it is correct. euhadra provides the seam; what you plug into it is your
opinion, not ours.

## Project structure

```
src/
  lib.rs               — module declarations
  types.rs             — domain types (AudioChunk, AsrResult, ContextSnapshot, etc.)
  traits.rs            — 4 core adapter traits
  filter.rs            — TextFilter trait + English/Japanese filler filters
  processor.rs         — TextProcessor trait + self-correction + punctuation
  pipeline.rs          — PipelineBuilder + async session runtime
  emitters.rs          — ClipboardEmitter (arboard)
  mic.rs               — Microphone capture (cpal)
  whisper_local.rs     — WhisperLocal ASR adapter (whisper.cpp subprocess)
  onnx_processing.rs   — [onnx feature] ONNX-based filters and processors
  mock.rs              — mock implementations for testing
  prelude.rs           — convenience re-exports
  main.rs              — CLI entry point
models/
  euhadra.als          — Alloy formal model
docs/
  spec.md              — full technical specification
  model-upgrade-candidates.md — model survey + backend calibration log
  model-licenses.md    — upstream license summary for bundled weights
```

## Development

```bash
cargo test                  # run unit + integration tests
cargo run --features cli -- --help         # CLI usage
cargo build --features onnx # with ONNX inference (requires ort)
```

## Evaluation

Quality is tracked across three layers (full policy in [`docs/evaluation.md`](docs/evaluation.md)):

| Layer | What it measures | How to run | Where it runs |
|---|---|---|---|
| **L1 ASR live smoke** | FLEURS WER/CER + RTF + ASR/E2E latency | `cargo eval-l1 -- ...` | Every PR (CI: `evaluate-asr`) |
| **L1 layer fast** | Tier 1+2 ablation ΔWER + per-layer μ-bench latency | `cargo eval-l1-fast` | Every PR (CI: `evaluate-fast`) |
| **L2 standard + Robust** | LibriSpeech / AISHELL-1 / ReazonSpeech WER + MUSAN/RIR SNR sweep | `cargo eval-l2 -- --dataset … --condition …` | Manual / release-time |
| **L3 direct F1 + ablation** | Layer-isolated F1 against annotated data; ΔWER on natural-speech fixtures | `cargo eval-l3 -- --task {filler,self-correction,phoneme-correction,ablation} …` | Manual / research |

Regression detection lives in `docs/benchmarks/ci_baseline*.json` — both the WER/CER + latency snapshot and the tolerance policy travel with the file. Two axes:

- **Relative**: `+regression%` against the committed baseline (catches drift)
- **Absolute**: hard floors tied to user-perceived dictation quality (RTF ≥ 1.0, latency p50 ≥ 1 s, etc.) that don't move with the baseline

Setup scripts (idempotent, skip-if-present):

```bash
scripts/setup_whisper.sh                   # whisper.cpp + ggml-tiny models (zh L1)
scripts/setup_canary.sh                    # canary-180m-flash-onnx (en + es L1, ~213 MB INT8)
scripts/setup_parakeet_ja.sh               # parakeet-tdt_ctc-0.6b-ja ONNX (ja L1, ~2.4 GB)
scripts/setup_paraformer_zh.sh             # FunASR Paraformer-large ONNX (zh L1, ~240 MB)
scripts/download_fleurs_subset.py          # L1 FLEURS subset
scripts/download_l2_data.sh <dataset>      # LibriSpeech / AISHELL-1 / MUSAN / RIR
scripts/download_l2_data.py reazonspeech-test
scripts/download_l3_data.sh <dataset>      # CS2W / TED-LIUM 3
scripts/build_l3_natural_fixtures.py manifest --manifest <path>
```

## License

Licensed under either of

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([`LICENSE-MIT`](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to work on euhadra. The most
useful contribution right now is native-speaker review of the evaluation gold
sets — see the Languages section above for why.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
