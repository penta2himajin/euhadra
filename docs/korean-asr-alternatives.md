# Korean ASR — License-clean alternatives

Issue [#83] tracks the search for a commercially-clean replacement for
`FunAudioLLM/SenseVoiceSmall` on the Korean (`ko`) routing path. The
upstream FunASR Model License v1.1 is a custom Alibaba license whose
commercial-use posture has been raised in upstream issues without an
official maintainer answer (see `docs/model-licenses.md`).

This document records what each candidate looks like today and the
recommended next step. It complements `docs/model-licenses.md` — that
file is the canonical license summary for what we already ship;
this file is the **research log** for what we might switch to.

Current measured baseline (`docs/benchmarks/ci_baseline.json`):
**SenseVoice-Small INT8 on FLEURS-ko 10-utterance subset, CER ≈ 6.32%**
(this is the target we need to match or beat).

[#83]: https://github.com/penta2himajin/euhadra/issues/83

## Candidates

### A. OpenAI Whisper-large-v3-turbo (already integrated)

| Field | Value |
|---|---|
| HF id | `openai/whisper-large-v3-turbo` |
| License | **MIT** |
| Training data license | OpenAI-curated weak supervision corpus (≥5M h labelled audio); license posture inherits the MIT model release |
| Architecture | Encoder/decoder; 809M params; 4 decoder layers (vs 32 in large-v3) |
| Korean reported | OpenAI's turbo announcement evaluates Korean with **CER** rather than WER but does **not publish the numeric value** for FLEURS-ko or CommonVoice-ko ([discussion #2363](https://github.com/openai/whisper/discussions/2363)). large-v3 announcement places Korean in the "10–20% error-rate reduction vs large-v2" bucket ([discussion #1762](https://github.com/openai/whisper/discussions/1762)). We have to measure it ourselves. |
| Integration cost | **Zero**. Existing `WhisperLocal` already loads any GGML/whisper.cpp-compatible Whisper checkpoint; pointing `ko` at this model is one config line on the Menura side. |
| Verdict | **Recommended replacement for ko**. Measured FLEURS-ko CER 1.96% via transformers FP32 and **1.52% via whisper.cpp Q5_0** — better than SenseVoice's 6.64% baseline by ~4×. Q5_0 incurs no measurable accuracy loss vs FP32. RTF is the only trade-off: this session container's shared 4-core Xeon @ 2.1 GHz reaches RTF 2.0; typical user hardware (Apple M-series, modern Ryzen / Core) lands ~RTF 0.05–0.4 per community reports. The CER win + clean MIT licence justify the trade-off for dictation use cases where accuracy dominates. See A.1 below. |

#### A.1 Measured FLEURS-ko 10-utt CER + RTF (2026-05-16)

Same harness as Section B.1 (CPU, 4 threads, lenient normalisation
via `eval::metrics::cer_lenient`). Two runtimes for turbo: the
FP32 transformers path (sanity reference) and the **whisper.cpp Q5_0
GGML** path that matches what euhadra's `WhisperLocal` actually
shells out to in production.

| Model | CER (lenient) | RTF (this container) | p50 / p95 latency | Weights size | Runtime |
|---|---|---|---|---|---|
| `whisper-large-v3-turbo` Q5_0 (whisper.cpp, batched 10 files / single load) | **1.52%** | 2.033 | 24646 / 38433 ms | ~547 MB Q5_0 | `whisper-cli` subprocess (production `WhisperLocal` path) |
| `whisper-large-v3-turbo` Q5_0 (whisper.cpp, per-utt subprocess) | 1.74% | 2.036 | 22654 / 24839 ms | ~547 MB Q5_0 | `whisper-cli` per call |
| `whisper-large-v3-turbo` FP32 (transformers PyTorch) | 1.96% | 0.567 | 6294 / 7054 ms | ~1.6 GB FP32 | transformers + PyTorch |
| `FunAudioLLM/SenseVoiceSmall` (`ci_baseline.json`, CI runner) | 6.64% | **0.047** | 540 / 776 ms | ~234 MB INT8 | euhadra `SenseVoiceAdapter` |

Per-utt lenient CER for whisper.cpp Q5_0 turbo: **5 / 10 utts perfect
(0.0000)**, the rest below 0.06. **Q5_0 quantisation incurred no
measurable accuracy loss** vs FP32 transformers (and slightly improved
the per-utt CER aggregate, within sampling noise).

Caveats:

- **RTF is hardware-bound, not model-bound**: the bench ran on a
  shared 4-core Xeon @ 2.1 GHz inside a VM (no turbo boost, no
  AVX-512 wins beyond what's already enabled). Community reports for
  the same Q5_0 turbo build on Apple M2 / Ryzen 7000 / modern Core i7
  land between RTF 0.05 and 0.4. Production users on those targets
  will see a much smaller gap to SenseVoice than this container's
  RTF 2.0 suggests.
- **Subprocess overhead is NOT the bottleneck**: batched mode (one
  model load, 10 files) and per-utt mode (10 loads) take essentially
  the same total wall time (≈226 s), so we're CPU-bound during
  decode, not paying for repeated model load.
- **Whisper pads to 30 s frames**: each utt's decoder loop processes
  a 30-second mel spectrogram regardless of actual audio length,
  which is why short utts (4.8 s) and long utts (18.9 s) come out
  within 10 % of each other. This is a property of the architecture,
  not the runtime.
- **Cross-environment SenseVoice baseline**: the SenseVoice baseline
  row was measured on a CI runner; we did not re-run SenseVoice in
  this container because the funasr export pipeline failed to install
  (oss2/crcmod wheel build errors). The CER side of the comparison is
  hardware-independent regardless.

Bench scripts (kept under `/tmp/`, not committed):
- `/tmp/ko_bench_final.py` — kresnik + SenseVoice (via sherpa-onnx, unreliable)
- `/tmp/ko_bench_whisper.py` — turbo FP32 via transformers
- `/tmp/ko_bench_whispercpp.py` — turbo Q5_0 via whisper.cpp, per-utt
- `/tmp/ko_bench_whispercpp_batch.py` — turbo Q5_0 via whisper.cpp, batched
- Lenient rescore via a throwaway `examples/score_ko_bench.rs` (also not committed)

### B. `kresnik/wav2vec2-large-xlsr-korean`

| Field | Value |
|---|---|
| HF id | `kresnik/wav2vec2-large-xlsr-korean` |
| Model license | **Apache-2.0** |
| Training data | `kresnik/zeroth_korean` — **CC-BY-4.0** ([dataset card](https://huggingface.co/datasets/kresnik/zeroth_korean); upstream openslr.org Zeroth corpus) |
| Architecture | wav2vec2-XLSR CTC, ~0.3B params; PyTorch+Safetensors |
| Korean reported | **Zeroth-Korean test set: WER 4.74% / CER 1.78%**. No FLEURS-ko or CommonVoice-ko numbers in the model card. |
| Last update | 2024-10-31 (active) |
| Integration cost | **Needs the still-unbuilt `Wav2Vec2Adapter` (issue [#92] / #F)** — wav2vec2 CTC decode + tokenizer wiring. The factory shape we already merged in PR #101 stays unchanged; only a new runtime id `"wav2vec2"` is added. |
| Verdict | **Not competitive on FLEURS-ko under this measurement** — see below. License remains the cleanest of the candidates, but the accuracy and latency gap relative to SenseVoice is too wide to justify the adapter work in #92 on Korean-only grounds. (#92 may still be worth implementing for the other languages that wav2vec2 fine-tunes cover well — Thai, Javanese, Sundanese — see issue [#83] discussion.) |

[#92]: https://github.com/penta2himajin/euhadra/issues/92

#### B.1 Measured FLEURS-ko 10-utt CER + RTF (2026-05-16)

Bench run inside the standard cloud session container (CPU, 4 threads),
on the same 10 FLEURS-ko utterances that drive the L1 baseline. kresnik
was loaded via `transformers` (FP32, PyTorch) since `Wav2Vec2Adapter` is
not yet implemented. SenseVoice's number is the canonical baseline from
`docs/benchmarks/ci_baseline.json` (measured via euhadra's own
`SenseVoiceAdapter`, INT8 ONNX) — the cleanest reference available.
A direct sherpa-onnx Python wrapper of SenseVoice was attempted in the
same env, but it returned truncated / empty / hallucinated text on
several utts (a known reliability gap of the Python wrapper for
SenseVoice mode) so it is not used as the comparison number.

CER is computed with euhadra's `eval::metrics::cer_lenient` so both
sides see identical text normalisation (Korean numeral conversion,
punctuation stripping, whitespace collapsing).

| Model | CER (lenient) | RTF | p50 / p95 latency | Weights size | Runtime |
|---|---|---|---|---|---|
| `kresnik/wav2vec2-large-xlsr-korean` | **17.44%** | **0.118** | 1390 / 2185 ms | ~1.3 GB FP32 | transformers + PyTorch CPU |
| `FunAudioLLM/SenseVoiceSmall` (`ci_baseline.json`) | **6.64%** | **0.047** | 540 / 776 ms | ~234 MB INT8 | euhadra `SenseVoiceAdapter` |

Read: kresnik is **~2.6× worse on CER and ~2.5× slower on RTF** on this
test set. Caveats and what they would change:

- **Quantisation gap**: kresnik is FP32; SenseVoice is INT8 ONNX.
  Exporting kresnik to ONNX + INT8 would likely halve its RTF and drift
  CER by ≤0.5 pp. Even at the best end of that range, kresnik would
  still be ~1.2× slower and ~10pp worse than SenseVoice.
- **Cross-environment timings**: SenseVoice RTF was measured on a CI
  runner, kresnik in this container. The CER side of the comparison
  is independent of the host.
- **Domain shift**: kresnik's own model card reports Zeroth-Korean CER
  1.78% (very narrow, easy read-speech corpus); on FLEURS-ko (more
  varied read speech) the CER inflates to 17.44%. This is consistent
  with prior reports of XLSR transfer brittleness across Korean corpora.

The bench script and full per-utt transcripts are kept at
`/tmp/ko_bench_final.py` / `/tmp/ko_bench_result.json` in the session
container; they are not committed to the repo because they depend on
ephemeral model downloads.

### C. `facebook/w2v-bert-2.0` + community Korean fine-tunes

| Field | Value |
|---|---|
| Base | `facebook/w2v-bert-2.0` — MIT, 4.5M h unlabelled pre-training across 143 languages |
| Reference Korean fine-tune | `HERIUN/w2v-bert-2.0-korean-colab-CV16.0` — MIT, trained on CommonVoice 16.0 (CC0-1.0) |
| Korean reported | **None** — the HF card's "Training results" section is empty (`"More information needed"`). The artefact looks like a tutorial run, not a converged production fine-tune. |
| Integration cost | Needs `W2VBertGenericAdapter` (issue [#93] / #G). |
| Verdict | **Not production-ready today** because no public Korean fine-tune ships with measurable accuracy. Track #93 for the adapter; revisit when a better Korean checkpoint is published or when we run our own fine-tune. |

[#93]: https://github.com/penta2himajin/euhadra/issues/93

### D. Community Whisper fine-tunes (Korean-only)

Existing Korean fine-tunes of Whisper variants that could load through `WhisperLocal` (or the planned WhisperLocal fine-tune extension, issue [#97] / #K) without a new adapter family:

| Model | License | Base | Korean reported |
|---|---|---|---|
| `spow12/whisper-medium-zeroth_korean` | Apache-2.0 | whisper-medium | Zeroth: **WER 3.96 / CER 1.71** |
| `ghost613/whisper-large-v3-turbo-korean` | Not declared on the model card | whisper-large-v3-turbo | Zeroth: WER 4.89 / CER 2.06. Card notes **"Models did not converge, better results are possible"** |

Verdict: `spow12/whisper-medium-zeroth_korean` is small, Apache-2.0 clean, and has the strongest reported Zeroth numbers in this group — but it is a `whisper-medium` (0.8B) fine-tune, slower than turbo. `ghost613` is explicitly self-described as not converged and ships without a declared license on the card. Use cases for these are narrow given Whisper-large-v3-turbo as the base candidate; track as fallback options if turbo on FLEURS-ko under-performs.

[#97]: https://github.com/penta2himajin/euhadra/issues/97

### E. KsponSpeech-trained models (SpeechBrain / NeMo / ESPnet)

| Model | License | Korean reported |
|---|---|---|
| `speechbrain/asr-conformer-transformerlm-ksponspeech` | Apache-2.0 (model) | KsponSpeech eval clean CER 7.33% / other 7.99% (2022-07) |
| ESPnet `egs2/ksponspeech/asr1` recipe | Apache-2.0 (toolkit) | Per-recipe; checkpoint distribution varies |
| NeMo Korean Conformer-Transducer (community discussions in [NVIDIA/NeMo#3648](https://github.com/NVIDIA/NeMo/discussions/3648)) | NVIDIA does not ship an official Korean ASR checkpoint today; community work exists but is not curated | — |

**Critical caveat:** KsponSpeech is distributed through [AI Hub](https://aihub.or.kr/aidata/105), operated by the Korean National Information Society Agency (NIA). AI Hub's terms are government-set and require per-user application; **commercial redistribution of derivative model weights is not unambiguously permitted under publicly available terms**. Even when the model code is Apache-2.0, the training-data licence flows through to downstream model weights and is the binding constraint for euhadra's commercial posture.

Verdict: **avoid as primary candidate** until AI Hub's commercial terms are independently confirmed. The Conformer-LM accuracy (~7.3% CER) is also weaker than SenseVoice's 6.32% on a different read-speech test set, so the upside is limited.

### F. NVIDIA NeMo (Canary / Parakeet) — Korean coverage

The current NVIDIA Parakeet family (`parakeet-tdt-0.6b-v2` / `-v3` / `-ja`) and Canary (`canary-180m-flash`) lists do not include Korean. Canary covers 25 EU languages; Parakeet-v3 covers EU 25 + en + ru + uk; the `-ja` variant is Japanese-only. There is no announced 2026 NeMo Korean ASR checkpoint at the time of this writing.

Verdict: **not a 2026 candidate**. If NVIDIA releases a Korean Canary/Parakeet, revisit — the licence (CC-BY-4.0) and integration (existing `CanaryFactory` / `ParakeetFactory`) would both be straightforward.

### G. Whisper-large-v3-turbo runtime backends (CPU)

Once Section A.1 established turbo as the recommended ko model, the
next question was *which inference engine* to ship it under. We
benchmarked four candidate backends on the **same FLEURS-ko 10-utt
subset, same 4-core Xeon @ 2.1 GHz VM, same `eval::metrics::cer_lenient`
scorer** as Section A.1:

| Path | Engine / format | weighted CER | RTF | p50 / p95 | Bundle | New deps |
|---|---|---:|---:|---|---:|---|
| **ORT `q4` turbo** (`whisper-onnx`) | `ort` crate on `onnx-community/whisper-large-v3-turbo` q4 export | **1.09%** ✅ | **0.484** ✅ | 5.4 / 5.7 s | ~1.1 GB | none (reuses existing `onnx` feature) |
| CT2 FP16 turbo (`faster-whisper`) | `ct2rs` crate on `deepdml/faster-whisper-large-v3-turbo-ct2` FP16 (upcast to FP32 here) | 1.32% | 1.28 | 14.2 / 17.2 s | 1.5 GB | `ct2rs`, libctranslate2 built from source (`cmake` + C++17) |
| whisper-rs Q4_0 GGML | `whisper-rs` crate on `ggml-large-v3-turbo-q4_0.bin` (locally quantised) | 1.74% | 1.78 | 19.7 / 20.1 s | 452 MB | `whisper-rs`, whisper.cpp built from source (`cmake` + C++) |
| whisper-rs Q5_0 GGML | `whisper-rs` crate on `ggml-large-v3-turbo-q5_0.bin` | 1.74% | 1.99 | 22.1 / 22.4 s | 547 MB | `whisper-rs`, whisper.cpp built from source |
| whisper.cpp Q5_0 subprocess | existing `WhisperLocal` adapter, `whisper-cli` subprocess (A.1 baseline) | 1.52% | 2.03 | 22.7 / 24.8 s | 547 MB | external `whisper-cli` binary at runtime |

For context (Section A.1 reference rows):

| | CER | RTF |
|---|---:|---:|
| SenseVoice INT8 ONNX (`ci_baseline.json`) | 6.64% | 0.047 |
| transformers FP32 turbo (Python, sanity ref) | 1.96% | 0.567 |
| ORT INT8 turbo | **broken (260%+)** | 0.255 |

#### Findings

- **ORT `q4` wins both axes.** CER 0.23 pp better than CT2, RTF 2.6× faster. CER 0.65 pp better than whisper-rs Q4_0, RTF 3.7× faster.
- **No new system deps for ORT.** Reuses the existing `onnx` feature gate that already ships `ort`, `tokenizers`, `ndarray`, `rustfft`. CT2 and whisper-rs both pull in a `cmake` + C++17 build step.
- **Q4_0 quantisation matters more than process model.** GGML Q4_0 vs Q5_0 (same in-process whisper-rs path) shaved ~10% off RTF with no CER loss. In-process vs subprocess on Q5_0 (whisper-rs vs whisper-cli) was a wash — subprocess startup wasn't the bottleneck on this VM.
- **INT8 of turbo via ORT is broken.** The decoder collapses into repeating-token hallucinations after a few autoregressive steps (`"일 일 일 일 일 …"`). Use `q4` instead. (CT2 INT8 would likely be fine because CT2 internally upcasts where needed; we did not test CT2 INT8 in this session.)
- **Container hardware is pessimistic.** The shared 4-core Xeon @ 2.1 GHz VM has no FP16 acceleration and modest single-core perf. Apple M-series / modern Ryzen would run all four paths several times faster; the ranking between them is what we should generalise, not the absolute RTF.

#### Decision

Use the **`whisper-onnx` runtime (ORT q4 turbo)** as the production
Korean default. PR #105 ships the integration shape (factory + session
loading + KV-cache schema discovery + tests) under the existing `onnx`
feature; the autoregressive decode loop is a focused follow-up
(~400-600 LOC of `ndarray` + `ort` plumbing — see the `transcribe_samples`
module docs in `src/whisper_onnx.rs` for the loop sketch).

The two sibling-path PRs explored as part of the same investigation
were closed in favour of ORT q4:

- **PR #103 (`whisper-rs` Q4_0 GGML)** — closed. Reusable as a fallback
  if a future deployment target needs whisper.cpp specifically (e.g.
  GPU offload via whisper.cpp's CUDA/Metal backends).
- **PR #104 (CTranslate2 / `ct2rs`)** — closed. Reusable if INT8 quantised
  CT2 bundles become available and benchmark closer to / above ORT q4
  on production hardware.

[#105]: https://github.com/penta2himajin/euhadra/pull/105

## Verdict and recommended sequencing

License cleanliness (descending):

1. **Whisper-large-v3-turbo (MIT)** — already shipped via `WhisperLocal`.
2. **`kresnik/wav2vec2-large-xlsr-korean` (Apache-2.0 + Zeroth CC-BY-4.0)** — clean both layers.
3. `spow12/whisper-medium-zeroth_korean` (Apache-2.0 + Zeroth CC-BY-4.0) — clean, smaller upside.
4. `facebook/w2v-bert-2.0` (MIT) — base only; no production Korean fine-tune yet.
5. KsponSpeech-trained models — model licence clean but **training-data licence unconfirmed for commercial redistribution**.

### Step 1 (done — 2026-05-16)

Measured Whisper-large-v3-turbo on the canonical FLEURS-ko 10-utt
subset (see A.1). Result: **CER 1.96% vs SenseVoice 6.64%** —
turbo wins by ~3.4× on accuracy.

### Step 1.5 (done — 2026-05-17)

Compared four CPU inference backends for the chosen turbo model
(see G). Result: **ORT `q4` is both the most accurate (CER 1.09%)
and the fastest (RTF 0.484) Whisper backend on x86 CPU**, and reuses
the existing `onnx` feature gate with no new system deps.

**Recommendation:** switch the Menura `ko` default from `sensevoice`
to **`whisper-onnx`** (runtime id, PR #105) + the
`onnx-community/whisper-large-v3-turbo` q4 bundle.

Concrete follow-up work, with status as of this writing:

- **Done (PR #106).** Autoregressive decode loop in
  `src/whisper_onnx/adapter.rs`'s `transcribe_samples` (mel → encoder
  → first decoder pass → KV-cache loop → detokenise). End-to-end
  bench reproduced the POC numbers (CER 0.95 % via `cer_lenient`).
- **Done (PR #107).** `scripts/setup_whisper_onnx_turbo.sh` fetches
  the q4 bundle (~900 MB) from `onnx-community/whisper-large-v3-turbo`,
  README + `docs/model-licenses.md` updated.
- **Done (this PR).** L1 eval routes `ko` through `WhisperOnnxAdapter`
  via a new `--whisper-onnx-ko-dir` flag; `.github/workflows/ci.yml`
  caches the bundle and passes the flag; `ci_baseline.json` ko entry
  refreshed (CER 0.0664 → 0.0095, the SenseVoice → WhisperOnnx swap).
  `--sensevoice-dir` is still supported as a fallback but is no longer
  exercised by CI.
- **Pending.** Update Menura's `asr_models.toml` to point
  `ko.runtime = "whisper-onnx"`,
  `ko.model_source.path = "/models/whisper-onnx-turbo"`. Out of scope
  for this PR (separate repo).

### Step 2 (after #92 lands)

Once `Wav2Vec2Adapter` (issue #92) is implemented, **kresnik wav2vec2
is now de-prioritised as a Korean replacement**: the measured
CER 17.44% is too far below turbo's 1.96% to justify the switch on
Korean alone. #92 may still be worth pursuing for other languages
that wav2vec2 fine-tunes cover well (Thai, Javanese, Sundanese — see
issue #83 discussion), but Korean routing should not gate on it.

### Step 3 (deferred)

Revisit KsponSpeech-derived models only if AI Hub's commercial terms are
either clarified upstream or covered by a separate written permission.
Track `spow12/whisper-medium-zeroth_korean` as a backup if turbo's
production RTF turns out to be unacceptable. If a Korean Parakeet/Canary
ever ships from NVIDIA, revisit via the existing factories.

## What this PR does and doesn't do

This PR is **documentation only**:

- Records the candidates evaluated and their licence chains.
- Recommends the Step 1 measurement plan.
- Cross-links from `docs/model-licenses.md` so reviewers find this file from the canonical licence summary.

It does **not**:

- Change the default Korean routing — that switch waits on Step 1 measurement.
- Add a Wav2Vec2 or W2VBert adapter — those remain in their own issues (#92 / #93).
- Modify `setup_sensevoice.sh` or remove SenseVoice — see issue #83 conclusion notes.
