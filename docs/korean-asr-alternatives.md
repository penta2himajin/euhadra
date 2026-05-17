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
| Verdict | **Recommended replacement for ko**. Measured FLEURS-ko CER 1.96% — better than SenseVoice's 6.64% baseline by ~3.4×. RTF in this measurement is 0.567 (FP32 PyTorch Python) but production via whisper.cpp INT5 GGML should land near 0.1–0.2 on CPU; that is still 2–4× slower than SenseVoice, but the CER win and clean MIT licence justify the trade-off for dictation use cases where accuracy dominates. See A.1 below. |

#### A.1 Measured FLEURS-ko 10-utt CER + RTF (2026-05-16)

Same harness as Section B.1 (CPU, 4 threads, lenient normalisation
via `eval::metrics::cer_lenient`). turbo was loaded via `transformers`
(FP32 PyTorch) since we don't have whisper.cpp + GGML turbo set up
in the session container; the production `WhisperLocal` path would
use whisper.cpp INT5 weights and be substantially faster.

| Model | CER (lenient) | RTF | p50 / p95 latency | Weights size | Runtime |
|---|---|---|---|---|---|
| `openai/whisper-large-v3-turbo` | **1.96%** | 0.567 (FP32 Python; whisper.cpp INT5 expected ≈0.1–0.2) | 6294 / 7054 ms | ~1.6 GB FP32 | transformers + PyTorch CPU |
| `FunAudioLLM/SenseVoiceSmall` (`ci_baseline.json`) | 6.64% | **0.047** | 540 / 776 ms | ~234 MB INT8 | euhadra `SenseVoiceAdapter` |

Per-utt lenient CER for turbo: **5 / 10 utts perfect (0.0000)**,
3 / 10 < 0.05, 2 / 10 < 0.06. The remaining drift is mostly digit
normalisation (turbo emits `천구백사십년` for `1940년`) which the
lenient Korean-numeral converter handles partially. Even the worst
utt (1883) is only 5.66% CER.

Caveats:

- **RTF gap is real**: FP32 PyTorch turbo is ~12× slower than INT8 ONNX
  SenseVoice in this container. Production parity is much closer:
  whisper.cpp INT5 turbo on a recent x86 CPU benchmarks around RTF 0.1–0.2
  (community reports), so the realistic gap is ~2–4×, not 12×.
- **Cross-environment timings**: SenseVoice baseline was measured on a
  CI runner; turbo here in a cloud session container. The CER side of
  the comparison is hardware-independent.
- **No GGML turbo bundle yet**: euhadra's `WhisperLocal` already accepts
  any GGML Whisper checkpoint; switching `ko` to turbo just requires
  shipping the GGML conversion (≈800 MB for `ggml-large-v3-turbo-q5_0.bin`).
  No Rust code change.

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
turbo wins by ~3.4× on accuracy, with RTF that is acceptable for
production via whisper.cpp INT5 (estimated 0.1–0.2 vs SenseVoice 0.047).

**Recommendation:** switch the Menura `ko` default from `sensevoice`
to `whisper-local` + `ggml-large-v3-turbo-q5_0.bin`. Concrete
follow-up PR ([the "Step 1 enablement" PR][step1-enable]):

- Add `scripts/setup_whisper_turbo.sh` to fetch the GGML turbo bundle
  (or wire the existing `setup_whisper.sh` to accept the model id).
- Update Menura's `asr_models.toml` to point `ko.runtime = "whisper-local"`,
  `ko.model_source.path = ".../ggml-large-v3-turbo-q5_0.bin"`,
  `ko.options.cli_path = ".../whisper-cli"`.
- Optionally measure turbo via the actual `WhisperLocal` path in the
  CI runner to get the production RTF for the `ci_baseline.json` entry.
- Keep `SenseVoiceFactory` registered as a fallback runtime; flip
  back if a regression appears.

[step1-enable]: https://github.com/penta2himajin/euhadra/issues/83#step-1

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
