# euhadra vs FluidAudio — what differs, and where euhadra should differentiate

Research log, 2026-10-17. Compares euhadra (0.3.0 as published, and `master` at `39d00f0`) with FluidAudio (`FluidInference/FluidAudio` at `503b4bd`, v0.17.7) and asks which of euhadra's properties are worth strengthening and which weaknesses block its goal.

Paths: `src/…` is euhadra 0.3.0 unless marked `master:`. `FA:` is FluidAudio at `503b4bd`, relative to the package root. Claims marked **(inferred)** come from reading code, not from a statement in the source.

## 1. The goal this comparison is measured against

The maintainer states euhadra's goal as:

1. **Transcribe what was said, accurately.**
2. **Contain generative variance inside euhadra**, so that a caller gets high-quality text without having to know where a model can invent, drop or vary output.

That goal sets euhadra apart from libraries whose aim is "reasonable transcription anywhere, with quality that depends on the model". It is **not what `docs/spec.md` §1.3 says today**. The spec positions euhadra against end-user products (Aqua Voice, TYPELESS) and lists BYO-model, a programmable pipeline and self-hosting as the differentiators. §1.4 Phase 1 reads "ASR → LLM refinement → text insertion". This document uses the stated goal as the yardstick. Updating the spec is recommendation R1.

## 2. Summary

- **The two projects are not substitutes.** FluidAudio is a model runtime and model catalogue for the Apple Neural Engine. euhadra is a pipeline whose value sits around the model: a silence guard ahead of ASR, deterministic text cleanup, and measured regression gates. FluidAudio has no stage after the model except ITN, punctuation commit and vocabulary boosting [FA-1].
- **euhadra's goal-aligned strengths** are the silence guard (padded-silence en WER 0.1875 → 0.0762) [E-9], LLM-free Tier 1/2 cleanup costing microseconds [E-6], determinism treated as a correctness property (Dolphin pinned to one thread) [E-3], and CI gates on WER, RTF, latency and VAD ΔWER [E-9].
- **euhadra's shortfalls against its own goal**: the Japanese rules are thin (punctuation F1 0.35, fillers removed only between `、`) [E-4]; the gold data is small and unreviewed [E-9]; and `master` swapped the ja default model to one with worse CER (0.0724 → 0.1057) [E-2]. Separately, it is CPU only, with no unload in 0.3.0, which makes "always loaded" expensive [E-3][E-6].
- **FluidAudio is ahead on** hardware use (ANE), model breadth (including 2-bit and QAT Parakeet variants), maturity (73 releases, ~2,800 tests, 20+ dictation apps) and download control [FA-3][FA-6][FA-10].
- **Recommended direction:** keep euhadra's identity as the layer that turns any recogniser's output into faithful text, and stop competing on runtime. Make the ASR runtime a backend (CoreML via FluidAudio on Apple, OpenVINO/QNN on Windows), and focus effort on what FluidAudio structurally lacks: Japanese cleanup quality, guarantees a caller can rely on, and evidence for them (§7).

## 3. Side-by-side

| Axis | euhadra | FluidAudio |
|---|---|---|
| What it is | Rust async pipeline: ASR → VAD → Tier 1 filters → Tier 2 processors → (Tier 3 LLM seam) → emitter [E-1] | Swift SDK of per-model `actor` managers: ASR, VAD, diarization, speaker embedding, ITN/TN, TTS, echo cancellation [FA-1] |
| Post-ASR text cleanup | Filler removal, self-repair, punctuation, ITN, term dictionary, phoneme correction, paragraphing; 5 languages [E-1] | None beyond ITN, punctuation commit and vocabulary boosting; no filler or self-repair (`grep -i filler` empty) [FA-1] |
| Silence / no-speech guard | Built in: Earshot VAD ahead of the adapter, `NoSpeech` on empty capture [E-5][E-9] | Not built into `AsrManager`; the app composes `VadManager` itself; inputs < 0.3 s throw [FA-5][FA-12] |
| ASR backends | 7 adapters, pure-Rust front ends over ONNX Runtime, plus whisper.cpp subprocess; mostly INT8 [E-2] | 10+ CoreML model families (Parakeet v2/v3/Ultra/Redux/Phonon-2/ja, Nemotron, Canary, Cohere, SenseVoice, Paraformer); no Whisper [FA-2] |
| Hardware | CPU only; no execution provider configured anywhere [E-3] | CoreML `.cpuAndNeuralEngine` by default, GPU opt-in per model; Apple only [FA-3] |
| Japanese ASR | Parakeet TDT-CTC 0.6B ja ONNX, FLEURS-ja CER 0.0724 (n=30) [E-9]; `master`: ReazonSpeech Zipformer INT8, CER 0.1057 [E-2] | Parakeet-ja CoreML, JSUT CER 6.88 %, FLEURS-ja 10.29 % [FA-2] |
| Japanese text handling | Rule-based fillers (between `、` only), cue-word self-repair, terminal `。`, ITN, kana/width folding for dictionary matching [E-4] | Model output only; language hint, vocabulary boosting and ITN do not apply to `tdtJa`; text dropped at 15 s seams (#1031) [FA-4] |
| Streaming | Batch per VAD-closed utterance; prefix re-transcription measured and rejected [E-5] | Batch 15 s windows, sliding window, and true streaming (English, plus Nemotron multilingual at ja CER 13.79 %) [FA-5] |
| Model lifecycle | 0.3.0: eager load, no unload; `master`: `AsrLifecycle`, implemented by SenseVoice and WhisperLocal only [E-6] | Lazy download with mirror/proxy/revision pin/offline mode; `cleanup()` per manager [FA-6] |
| Extensibility | One small trait per stage (`AsrAdapter`, `TextFilter`, `TextProcessor`, `VadBackend`, …) passed to a builder [E-8] | Swap weights matching a built-in contract; no processor or filter protocol [FA-8] |
| Evaluation | L1 CI gates (FLEURS n=30/lang), layer ablation, VAD ΔWER, endpoint latency; gold sets drafted by Claude, unreviewed [E-9] | Benchmark CLI and macOS CI workflows; maintainer-run tables on LibriSpeech, FLEURS, JSUT, AMI [FA-9] |
| Maturity | 1 maintainer, ~half of commits by AI agents, 0 stars, 3 releases in Aug 2026, 637 tests, Linux-only CI [E-10] | 3.1k stars, 73 releases, 118 contributors (2 do ~85 %), ~2,800 tests, used by 20+ dictation apps [FA-10] |
| Licence | Code MIT OR Apache-2.0; no weights bundled; model licences documented, SenseVoice flagged [E-11] | Code Apache-2.0; weights a patchwork (CC-BY-4.0, NVIDIA OML, OpenMDW, OpenRAIL-M, FunASR terms); README's "MIT/Apache" claim is inaccurate [FA-11] |

## 4. Where euhadra already differs in the direction of its goal

Each item represents something a FluidAudio caller must account for and handle, whereas euhadra handles it itself.

1. **Silence never reaches the recogniser.** Parakeet-ja outputs 「心の声。」 for 10 s of silence [E-4]. euhadra runs Earshot ahead of the adapter and has measured the effect: padded-silence en WER 0.1875 → 0.0762, ja CER 0.1211 → 0.0759 [E-9]. In FluidAudio the app wires `VadManager` to `AsrManager` itself, and an input under 0.3 s throws instead of returning nothing [FA-5][FA-12].
2. **Cleanup without a generative model.** Tiers 1 and 2 consist of rules and small classifiers. They cost microseconds (ja filler p50 0.4 µs, self-correction 2.8 µs) [E-6], and the LLM tier is a deliberately unimplemented seam [E-1]. Nothing in the default path can produce text that was not in the transcript.
3. **Determinism treated as correctness.** Dolphin is pinned to one intra-op thread because four threads gave five different transcripts in five runs [E-3]. FluidAudio has no comparable statement (none found).
4. **Choices that rest on measurements.** Examples: `FinalPass::SpeechOnly` over `JoinSegments` (en WER 0.0855 vs 0.3940), the Earshot threshold of 0.2, and rejecting prefix re-transcription (182 % / 350 % churn) [E-5][E-9]. CI gates WER/CER, RTF, latency and VAD ΔWER against committed baselines [E-9].
5. **A stable place to put caller-specific fixes.** `TermDictionary` is ordered after punctuation and folds kana and width for matching [E-4]. FluidAudio's vocabulary boosting does not apply to Japanese at all, and in English it replaced common words in real dictation (#967) [FA-4][FA-12].
6. **Not tied to one vendor.** The pipeline is plain Rust, and the ASR runtime is a trait [E-8]. FluidAudio is CoreML only, and its Windows path is a separate, stalled project [FA-3].

## 5. Where euhadra falls short of its own goal

1. **Japanese text quality, the part FluidAudio does not cover, is thin.**
   - Fillers are removed only when a `、`-delimited segment equals a lexicon entry, so 「えーと今日は」 keeps its filler **(inferred)** [E-4].
   - Rule-based ja punctuation only appends a final `。`, scoring F1 0.3468 (recall 0.21) [E-4]. XLM-R with `JaPunctNormalizer` reaches 0.6882 on synthetic gold, but exists only in Python and adds ~243 ms p50 [E-4].
   - `SelfCorrectionDetector` ignores the language and tries ko → zh → ja → es → en on every input, applying at most one correction [E-12].
2. **The evidence for "accurate" is thin.** CI uses FLEURS n=30 per language. The ja filler (25 utterances) and self-correction (40) gold sets were drafted by Claude and have not been reviewed by a native speaker [E-9]. Such small, unreviewed sets cannot show that cleanup never removes real speech.
3. **`master` traded Japanese accuracy for size.** The ja default moved from Parakeet-ja (CER 0.0724) to ReazonSpeech Zipformer INT8 (CER 0.1057, ~72 MB vs ~2.4 GB, Apache-2.0) [E-2]. Under the stated goal, accuracy is the first criterion, and a 46 % relative CER increase needs an explicit decision. FluidAudio's quantised CoreML Parakeet-ja is 590 MB on disk with JSUT CER 6.88 % [FA-2], so size and accuracy do not have to be traded off against each other this way.
4. **Variance a caller still has to know about.**
   - VAD degrades silently to an unsegmented run on a sample-rate mismatch; this is reported only in `Diagnostics::failures` [E-12].
   - `max_speech` force-splits unbroken speech at 30 s [E-5]. Whether the split can cut a word, as FluidAudio's seam bug does, is unmeasured.
   - `WhisperLocal` writes audio to a temporary WAV file [E-6], which conflicts with callers that promise no audio will be written to disk.
5. **Records lag the code.** Since 0.3.0, the CHANGELOG `[Unreleased]` section is empty after 19 commits, and the latency baselines were not re-measured after the model swaps [E-10][E-9]. A project whose differentiator is measurement needs to keep its measurements current.
6. **Resource cost makes "always loaded" impractical.** Every ONNX session runs on the CPU EP [E-3]. 0.3.0 cannot unload a model [E-6]. Inference runs synchronously inside `async fn`, blocking a tokio worker during decoding **(inferred)** [E-3]. No RSS or load-time numbers are published [E-6].

## 6. Where FluidAudio is simply ahead

It is not euhadra's job to match these feature for feature. Where euhadra needs the capability, it can integrate it as a backend (§7).

- **Hardware.** The ANE is the default for ASR and VAD, and the GPU is opt-in per model [FA-3]. One Shrec run on an M1 with VAD, Parakeet-ja and a diarizer loaded measured a 119 MB average / 152 MB peak app footprint and a 47 MB `aned` peak, while 1.35 GB of weights sat on disk [FA-6]. That run did not separate ANE-resident memory, so it is a lower bound **(inferred)**.
- **Model catalogue.** It offers several sizes within the same model family, such as Parakeet Redux at 2-bit (183 MB encoder) and QAT Phonon-2 [FA-2]. It also offers streaming models with partial results [FA-5].
- **Distribution.** Models download lazily, with support for mirrors, proxies, revision pinning, offline mode, resumable downloads, and explicit `cleanup()` [FA-6].
- **Adoption and cadence.** Releases are published roughly weekly, with ~2,800 tests and macOS CI benchmarks [FA-9][FA-10].
- **Breadth outside dictation.** Diarization, TTS and echo cancellation [FA-1]. These are outside euhadra's goal.

## 7. Recommendations

Ordered by how directly each one serves the stated goal. Effort figures are rough guesses, not estimates from measured work.

### Define and defend the identity

- **R1. Restate the goal in `docs/spec.md` §1.3–1.4.** State that euhadra is the layer that turns a recogniser's output into faithful text and contains generative variance. Move "LLM refinement" out of the Phase 1 definition, since the LLM tier is a deliberate seam. Without this, contributors and users judge euhadra as "one more ASR library" and consider it lacking in breadth.
- **R2. Write the contract down as guarantees, and test each one.** Candidates:
  - no text for a capture with no speech;
  - identical output for identical input and configuration, gated by an N-run determinism check in CI;
  - Tier 1/2 never insert words absent from the transcript (only removals, punctuation and dictionary substitutions);
  - failures degrade to the raw transcript and are reported, never silently dropped.

  These are the properties a FluidAudio caller must build on its own (§4).

### Strengthen what only euhadra does

- **R3. Japanese cleanup quality.**
  - Filler removal that does not require `、` (prefix and inline fillers).
  - Internal punctuation: ship XLM-R + `JaPunctNormalizer` as a Rust adapter, or a smaller ja-native model, behind a latency gate.
  - Language-aware self-repair.
  - This is the gap FluidAudio leaves for Japanese (§3, Japanese text handling).
- **R4. Evidence.**
  - Get the ja gold sets reviewed by a native speaker (issue #125).
  - Expand them beyond 25/40 utterances, including dictation-style speech rather than read speech.
  - Add a precision gate for "cleanup removed real speech". The goal promises accuracy, so false removals matter more than missed fillers.
- **R5. Revisit the ja default model under the accuracy-first criterion.** Either keep Parakeet-ja as the default and offer Reazon as a small option, or record why a 46 % relative CER increase is acceptable. Measure a quantised Parakeet-ja first: FluidAudio's INT8-encoder CoreML build suggests model size can be reduced without that loss (§5.3). *Status: Parakeet-ja is the ja path again (#167). Reazon trails it even with sherpa-onnx's FP32 decode (FLEURS-ja CER 0.0885 vs 0.0724) and stays as an option; a MatMul-only INT8 build with 7-bit weights (#173) keeps the FP32 CER at ~885 MB (FLEURS-ja test 0.0549 vs 0.0566), while an all-op INT8 build loses 35 %.*

### Fix what blocks always-on use

- **R6. Make the runtime a backend, not a rewrite.**
  - Expose execution-provider configuration on the ONNX adapters; `parakeet-rs-featsize` already has CoreML/OpenVINO/DirectML providers that euhadra does not surface [E-3].
  - Add an `AsrAdapter` that delegates to a host-provided recogniser. On Apple the host is FluidAudio (CoreML), which also provides the `tdtJa` CTC encoder path. On Windows it would be OpenVINO or QNN; see Typwrtr's `docs/research/windows-npu-asr.md`.
  - The pipeline's guarantees (R2) then hold regardless of the underlying runtime. This is how euhadra benefits from FluidAudio rather than competing with it.
- **R7. Resource lifecycle.**
  - Implement `AsrLifecycle` on every adapter.
  - Move decoding to `spawn_blocking`.
  - Publish RSS and cold/warm load time per adapter alongside WER, and gate them in CI as latency already is.
- **R8. Keep the records current.** Populate the CHANGELOG for the 19 commits since 0.3.0, re-measure the latency baselines after the model swaps, and add a macOS runner that runs `cargo test --features onnx`, not just `check`.

### Leave alone

- Diarization, TTS and echo cancellation are outside the goal.
- Streaming partials are deferred for dictation in any case. Revisit only if a caller needs text before the utterance closes.

## 8. Not verified

- How much memory the ANE holds for FluidAudio models, beyond the app footprint and the `aned` process. The Shrec run did not measure system-wide wired memory before and after the run.
- The RSS of any euhadra adapter. None is published, and none was measured for this document.
- Whether euhadra's 30 s force-split loses words, as FluidAudio's 15 s seam does (#1031).
- Whether a quantised Parakeet-ja ONNX preserves the FP32 CER. FluidAudio's ja figures (JSUT 6.88 %, FLEURS 10.29 %) and euhadra's (FLEURS n=30, 0.0724) use different sets and sizes and are not comparable.
- That FluidAudio's ja ITN is unreachable from Swift is inferred from the API, not verified in a run.

## 9. Sources

euhadra (0.3.0 crate source unless marked; `master` = `39d00f0`):

- [E-1] Scope and tiers: `README.md:10-21`, `:193-203`; `src/pipeline.rs:41-53`; LLM seam `src/traits.rs:86-89`, `README.md:496-500`.
- [E-2] Adapters and models: `src/parakeet.rs:1-50`, `src/canary/adapter.rs:39-98`, `src/dolphin/adapter.rs:57-69`; `master:` `src/reazon/adapter.rs`, commits `850a79d` (ja → Reazon, CER 0.0724 → 0.1057) and `a8ee7f5` (en/es → Parakeet-v3 INT8).
- [E-3] Execution: every `ort::Session::builder()` in `src/` without `with_execution_providers`; Dolphin threads `src/dolphin/adapter.rs:44-56`, `:99-107`; Parakeet `ParakeetTDT::from_pretrained(dir, None)` `src/parakeet.rs:44-50`, and `parakeet-rs-featsize` 0.3.8 `src/execution.rs:12-30`.
- [E-4] Japanese: `src/filter.rs:470-600`; `src/processor.rs:141-153`, `:606-664`, `:670-800`; `src/dictionary.rs:136-160`; silence hallucination `README.md:376-381`; `master:` `docs/benchmarks/punctuation/bakeoff.json`, `bakeoff_overseg_ja_es.json`, `src/processor.rs:1010-1019`.
- [E-5] Streaming and segmentation: `src/traits.rs:9-28`; `src/pipeline.rs:198-236`, `:359-590`; `src/vad/segmenter.rs:16-60`; `README.md:358-362`, `:383-404`.
- [E-6] Resources: `src/whisper_local.rs:64-85`; `master:` `src/traits.rs` (`AsrLifecycle`), `src/sensevoice/adapter.rs:376-410`; per-layer cost `master:docs/benchmarks/ci_baseline_layers.json`; Canary INT8 sizes `master:docs/canary-integration.md:786-822`.
- [E-8] Traits: `src/traits.rs:20-28`, `src/filter.rs:13-18`, `src/processor.rs:15-22`, `src/vad/mod.rs:75-148`, `src/router.rs:83-90`.
- [E-9] Evaluation: `v0.3.0:docs/benchmarks/ci_baseline.json`; VAD ΔWER `CHANGELOG.md:141-166`; Earshot sweep `src/vad/earshot.rs:62-90`; gold-set status `master:docs/evaluation.md:1004-1027`, `README.md:211-225`.
- [E-10] Maturity: `gh repo view penta2himajin/euhadra`; `git log v0.3.0..master`; `master:CHANGELOG.md:9`; `.github/workflows/ci.yml` (`runs-on: ubuntu-latest`; `onnx` checked, not tested).
- [E-11] Licences: `Cargo.toml` `license`; `docs/model-licenses.md`.
- [E-12] Known issues: `CHANGELOG.md:125-127` (VAD rate mismatch); `src/processor.rs:612-628` (self-repair ordering).

FluidAudio (`503b4bd`, v0.17.7; GitHub and Hugging Face queried 2026-10-17):

- [FA-1] Scope: `Sources/FluidAudio/{ASR,VAD,Diarizer,Speaker,ITN,TTS,Enhancement,Decision}`; `ASR/Shared/PunctuationCommitLayer.swift:1-40`; no `filler` in `Sources/` or `Documentation/`.
- [FA-2] Models: `Sources/FluidAudio/ModelNames.swift:4-142`; `Documentation/Models.md:5-41`; ja `Documentation/Benchmarks.md:1064-1092` and the `FluidInference/parakeet-0.6b-ja-coreml` model card.
- [FA-3] Hardware: `Shared/MLModelConfigurationUtils.swift:11-30`; `ASR/Parakeet/SlidingWindow/TDT/AsrModels.swift:144-160`; `Package.swift:7-11`; `README.md:739` (Windows → fluid-server).
- [FA-4] Japanese: `TDT/AsrManager.swift:335-345`, `:373-496`; `ITN/TextNormalizer.swift:68-145`; issue #1031.
- [FA-5] Streaming and VAD: `TDT/ChunkProcessor.swift:27-28`; `Shared/ASRConstants.swift:12-15`; `Streaming/StreamingAsrManager.swift:20-60`; `VAD/VadTypes.swift:12-47`; `Benchmarks.md:615-637`.
- [FA-6] Resources: `Shared/Download/ModelHub.swift:23-140`; `ModelRegistry.swift:27-196`; `Streaming/StreamingAsrManager.swift:50-53`; Shrec measurement, 2026-10-17 (M1, seminar mode, 56 s of audio).
- [FA-8] Extensibility: `ASR/Parakeet/Streaming/StreamingAsrManager.swift:20-74`; `Diarizer/DiarizerProtocol.swift:6`; `Documentation/Orukeet.md:1-10`; `ITN/TextNormalizer.swift:184-215`.
- [FA-9] Evaluation: `Documentation/CLI.md`; `Documentation/Benchmarks.md`; `.github/workflows/*benchmark*.yml`.
- [FA-10] Maturity: `gh repo view FluidInference/FluidAudio`; releases and `stats/commit_activity` API; `Tests/FluidAudioTests/`; `README.md:25-126`.
- [FA-11] Licences: `LICENSE`; Hugging Face `cardData.license` per model repo; `README.md:434`.
- [FA-12] Dictation limitations: `TDT/AsrManager.swift:448-451`; issues #967, #899, #912, #954, #1022.
