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
    → [LlmRefiner] (optional: tone adjustment)
    → Output (clipboard / stdout)
```

Each stage is a Rust trait. Swap any component without touching the rest.

## Getting Started

### Prerequisites

1. **Rust** (1.75+): https://rustup.rs
2. **whisper.cpp**: local ASR engine

Build whisper.cpp:

```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
cmake -B build && cmake --build build --config Release
bash models/download-ggml-model.sh base
```

### Install euhadra

```bash
git clone https://github.com/example/euhadra  # replace with actual URL
cd euhadra
cargo build
```

### Transcribe a WAV file

```bash
# Raw whisper transcription
cargo run -- transcribe \
  --file speech.wav \
  --whisper-cli /path/to/whisper.cpp/build/bin/whisper-cli \
  --model /path/to/whisper.cpp/models/ggml-base.bin \
  --language en
```

### Full pipeline (filter + process)

```bash
# English: filler removal + self-correction + punctuation
cargo run -- dictate \
  --file speech.wav \
  --whisper-cli /path/to/whisper-cli \
  --model /path/to/ggml-base.bin \
  --language en

# Japanese: filler removal (えーと, あの, etc.) + ASR artifact cleanup
cargo run -- dictate \
  --file speech.wav \
  --whisper-cli /path/to/whisper-cli \
  --model /path/to/ggml-base.bin \
  --language ja
```

### Record from microphone

```bash
# Record → transcribe → print to stdout
cargo run -- record \
  --whisper-cli /path/to/whisper-cli \
  --model /path/to/ggml-base.bin \
  --language en

# Record → transcribe → copy to clipboard
cargo run -- record \
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
    // Minimal: ASR + filler filter + self-correction + punctuation
    let pipeline = Pipeline::builder()
        .asr(WhisperLocal::new("whisper-cli", "ggml-base.bin").with_language("en"))
        .filter(SimpleFillerFilter::english())
        .processor(SelfCorrectionDetector::new())
        .processor(BasicPunctuationRestorer)
        .refiner(MockRefiner::passthrough())    // no LLM needed
        .context(MockContextProvider::new())
        .emitter(StdoutEmitter)
        .build()
        .unwrap();

    // Load audio and run
    let audio = euhadra::whisper_local::read_wav("speech.wav".as_ref()).unwrap();
    let (audio_tx, _cancel, handle) = pipeline.session();
    audio_tx.send(audio).await.unwrap();
    drop(audio_tx);

    let result = handle.await.unwrap().unwrap();
    // result.raw_text  — original ASR output
    // result.output    — filtered + processed text
}
```

For Japanese:

```rust
let pipeline = Pipeline::builder()
    .asr(WhisperLocal::new("whisper-cli", "ggml-base.bin").with_language("ja"))
    .filter(JapaneseFillerFilter::new())
    .processor(SelfCorrectionDetector::new())
    .processor(BasicPunctuationRestorer)
    .refiner(MockRefiner::passthrough())
    .context(MockContextProvider::new())
    .emitter(ClipboardEmitter::new())
    .build()
    .unwrap();
```

## Three-tier text processing

euhadra processes ASR output through three independent layers, each optional:

| Tier | Component | What it does | LLM? | Size |
|------|-----------|-------------|------|------|
| 1 | **TextFilter** | Filler removal (um, uh, えーと) | No | 0 MB (rules) or 33 MB (embeddings) |
| 2 | **TextProcessor** | Punctuation, capitalization, self-correction | No | 0 MB (rules) or 5-50 MB (ONNX) |
| 3 | **LlmRefiner** | Tone adjustment, context-adaptive rewriting | Yes | Optional |

Tier 1 + 2 alone produce clean, punctuated text without any LLM or network calls.

## CLI reference

```
euhadra dictate     Transcribe a WAV file through the full pipeline
  --file <path>       WAV file (16-bit PCM)
  --whisper-cli       Path to whisper-cli binary
  --model             Path to GGML model
  --language          Language hint (en, ja, etc.)
  --filler-script     Path to filler_filter.py (embedding mode)
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
- `OnnxEmbeddingFilter` — embedding-based filler detection (replaces Python script)
- `OnnxPunctuationRestorer` — CNN-BiLSTM punctuation + capitalization model

Without the `onnx` feature, euhadra uses rule-based implementations with zero ML dependencies.

## Architecture

```
[OS Shell (Swift/Kotlin/C++)]
    ↕ C ABI / UniFFI
[euhadra core (Rust)]
    ├── Pipeline runtime (tokio async)
    ├── ASR adapter trait         → WhisperLocal (whisper.cpp)
    ├── TextFilter trait          → SimpleFillerFilter, JapaneseFillerFilter, ChineseFillerFilter
    ├── TextProcessor trait       → SelfCorrectionDetector, BasicPunctuationRestorer
    ├── LLM refiner trait         → (optional, pluggable)
    ├── Context provider trait    → Manual, Accessibility API
    ├── Output emitter trait      → Clipboard, stdout
    └── [onnx] ONNX processing   → OnnxEmbeddingFilter, OnnxPunctuationRestorer
```

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
scripts/
  filler_filter.py     — legacy Python embedding filter (replaced by onnx feature)
models/
  euhadra.als          — Alloy formal model
docs/
  spec.md              — full technical specification
```

## Development

```bash
cargo test                  # run unit + integration tests
cargo run -- --help         # CLI usage
cargo build --features onnx # with ONNX inference (requires ort)
```

## Evaluation

Quality is tracked across three layers (full policy in [`docs/evaluation.md`](docs/evaluation.md)):

| Layer | What it measures | How to run | Where it runs |
|---|---|---|---|
| **L1 ASR live smoke** | FLEURS WER/CER + RTF + ASR/E2E latency | `cargo eval-l1 -- ...` | Every PR (CI: `evaluate-asr`) |
| **L1 layer fast** | Tier 1+2 ablation ΔWER + per-layer μ-bench latency | `cargo eval-l1-fast` | Every PR (CI: `evaluate-fast`) |
| **L2 standard + Robust** | LibriSpeech / AISHELL-1 / ReazonSpeech WER + MUSAN/RIR SNR sweep | `cargo eval-l2 -- --dataset … --condition …` | Manual / release-time |
| **L3 direct F1 + ablation** | Layer-isolated F1 against annotated data; ΔWER on natural-speech fixtures | `cargo eval-l3 -- --task {self-correction,ablation} …` | Manual / research |

Regression detection lives in `docs/benchmarks/ci_baseline*.json` — both the WER/CER + latency snapshot and the tolerance policy travel with the file. Two axes:

- **Relative**: `+regression%` against the committed baseline (catches drift)
- **Absolute**: hard floors tied to user-perceived dictation quality (RTF ≥ 1.0, latency p50 ≥ 1 s, etc.) that don't move with the baseline

Setup scripts (idempotent, skip-if-present):

```bash
scripts/setup_whisper.sh                   # whisper.cpp + ggml-tiny models (zh L1)
scripts/setup_parakeet_en.sh               # parakeet-tdt-0.6b-v3 ONNX (en L1, ~2.4 GB)
scripts/setup_parakeet_ja.sh               # parakeet-tdt_ctc-0.6b-ja ONNX (ja L1, ~2.4 GB)
scripts/download_fleurs_subset.py          # L1 FLEURS subset
scripts/download_l2_data.sh <dataset>      # LibriSpeech / AISHELL-1 / MUSAN / RIR
scripts/download_l2_data.py reazonspeech-test
scripts/download_l3_data.sh <dataset>      # CS2W / TED-LIUM 3
scripts/build_l3_natural_fixtures.py manifest --manifest <path>
```
