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
| Latency breakdown | §A.2: ~90% of per-utterance cost is a fixed 30-second encoder pass that runs identically for 0.5 s and 18.9 s of audio, and the q4 quantisation we default to is *slower* than no quantisation on the ORT CPU EP. |
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

#### A.2 Where the time actually goes (2026-07-31)

Section A.1 established that turbo is accurate on `ko` and slow. This
run decomposes the slowness, on the same 10-utterance FLEURS-ko subset
via `examples/bench_whisper_onnx_ko.rs`.

**This container is roughly 2× faster than the CI runner**: the same q4
configuration that `ci_baseline.json` records at RTF 1.339 measures
0.675 here. Absolute numbers below are therefore not comparable with
the committed baseline; every comparison in this section is
within-container and holds.

##### Cost is fixed per utterance, not per second of audio

The mel front-end pads every input to 30 seconds (`whisper_onnx::mel`,
`N_SAMPLES = 480_000`), so the encoder runs the same amount of work
whatever the audio length. Measured at q4, across audio spanning 3.9×
in duration:

| audio | ASR time | implied RTF |
|---|---|---|
| 4.80 s | 7048 ms | 1.47 |
| 7.02 s | 7004 ms | 1.00 |
| 12.48 s | 8470 ms | 0.68 |
| 18.90 s | 8363 ms | 0.44 |

Duration varies 3.9×, time varies 1.21×. Feeding **0.5 s of silence**
costs 6.5–9.5 s — the same as a twelve-second sentence. So better than
90% of per-utterance cost is the fixed encoder pass, and **RTF is a
misleading headline for this workload**: it improves with longer audio
purely because the constant is amortised. The number that matters for
dictation is the fixed ~7 s wait, and dictation utterances are short.

This bites turbo specifically. `large-v3-turbo` cut the decoder from 32
layers to 4, which leaves roughly 78% of its 809M parameters in the
encoder — the part that always runs on 30 seconds.

##### q4 buys no speed at all

Isolated on 0.5 s of silence, so the measurement is the encoder pass:

| encoder | fixed pass |
|---|---|
| **int8** | **~3.6 s** |
| fp32 | ~7.2 s |
| q4 (current default) | ~7.8 s |

**q4 is slower than not quantising.** ONNX Runtime's CPU EP has no
optimised 4-bit kernels, so dequantisation costs more than the memory
traffic it saves. q4 was chosen in the first place for accuracy and
loadability — the int8 decoder degenerates and fp16 will not load on
the CPU EP — never for speed. It is a cost being paid for nothing.

##### Full comparison, 10 utterances

| encoder | decoder | RTF | CER (lenient) |
|---|---|---|---|
| q4 | q4 (current default) | 0.675 | **0.0095** |
| fp32 | q4 | 0.737 | 0.0135 |
| **int8** | q4 | **0.367** | 0.0505 |
| int8 | int8 | 0.415 | 1.2671 |
| quantized | quantized | 0.420 | 1.5548 |

The int8 and `quantized` decoders collapse, confirming §A.1's note.

The int8 **encoder** is more interesting: 1.8× faster overall, and its
CER regression is not a uniform degradation but a single utterance
falling into a repetition loop —

```
1715 (18.90 s): cer=2.7763 asr=19700 ms
  hyp="지하철의 정규 안내 방송은 카탈로니아어로라잇라잇라잇라잇…"
```

The other nine ran in 4.2–4.8 s at CER 0.0000–0.0566. The loop also
destroys latency, since it runs to the token limit. A repetition
penalty or no-repeat-ngram constraint might contain it; untested.

##### The 30 s window cannot be shortened in this export

```
input : input_features ['batch_size', 128, 3000]
```

The mel axis is fixed, not dynamic, so shorter audio cannot simply be
fed as-is. Shortening it means re-exporting from PyTorch with the
positional embeddings sliced — the approach faster-whisper and
whisper.cpp take — which is a known technique but a real piece of work.

##### What this implies

The slowness decomposes into a fixed 30-second encoder pass (~90% of
the cost) multiplied by a quantisation that is slower than none. Both
follow from the model choice, and the headroom available without
changing it is limited:

- **int8 encoder** — 1.8× faster, contingent on containing the
  repetition collapse. Even if that works, RTF 0.37 is still an order
  of magnitude behind `paraformer-large`'s 0.035 on `zh`.
- **Shortening the window** — up to 2.7× on this utterance-length
  distribution, but requires a re-export and only pays off while we
  stay on Whisper.

Against that, `Qwen3-ASR-0.6B` carries a **180M** encoder against
turbo's 635M and has no fixed-window constraint
([`model-upgrade-candidates.md`](./model-upgrade-candidates.md) §2.1).
Replacing the backend looks like better value than tuning this one.


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

### H. `Qwen3-ASR-0.6B` (measured 2026-07-31)

| Field | Value |
|---|---|
| HF id | `Qwen/Qwen3-ASR-0.6B` (upstream), measured via the `cattle12/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25` INT8 export |
| License | Upstream **Apache-2.0**. The sherpa-onnx mirror measured here **declares no license** — resolve before adopting |
| Architecture | AuT encoder (180M) + Qwen3-0.6B autoregressive decoder, 0.9B total in bf16. Three graphs: `conv_frontend`, `encoder`, `decoder` (KV cache) |
| Reported ko | FLEURS-ko CER 3.72% (technical report, Table A.2) |
| Verdict | **Structurally right, currently unusable.** It removes Whisper's fixed-window problem outright, but the only published export degenerates on ~7% of utterances |

Measured on 30 FLEURS-ko utterances against the incumbent, same
container, same `eval::metrics::cer_lenient` via
`examples/score_hypotheses.rs`:

| | `whisper-large-v3-turbo` q4 | `Qwen3-ASR-0.6B` INT8 |
|---|---|---|
| RTF | 0.600 | **0.324** |
| CER (lenient) | **0.0269** | 0.1911 |
| Degenerate utterances | 0 / 30 | **2 / 30** |
| CER excluding those | — | 0.0838 (n=28) |

#### H.1 The structural claim holds

§A.2 established that Whisper's cost is fixed per utterance because the
mel front-end pads to 30 seconds. Qwen3-ASR has no such window, and the
measurement shows it:

| audio | Qwen3-ASR | whisper q4 |
|---|---|---|
| 4.80 s | **1723 ms** | 7048 ms |
| 18.90 s | 6803 ms | 8363 ms |

3.9× the audio, 3.9× the time — linear, where Whisper varied by 1.21×
over the same span. **On a 4.8-second utterance it is 4.1× faster**,
and dictation utterances are short. This is the property worth moving
for; the aggregate RTF understates it.

#### H.2 The INT8 export degenerates

```
1677: 嗯，嗯，嗯，嗯，嗯，嗯，嗯，…                      (CER 2.38)
1992: "，""，""，""，""，""，…                          (CER 1.00)
1705: 1940年8月15日，英纳布郡的弗朗斯南部被侵略…        (Chinese output for Korean audio)
```

Repetition loops, on roughly 7% of utterances. And this is the **third
independent occurrence of the same failure mode in this codebase**:

| Model | Component quantised to INT8 | Result |
|---|---|---|
| whisper-large-v3-turbo | decoder | CER 1.27 — collapses (§A.2) |
| whisper-large-v3-turbo | encoder | 1 / 10 utterances loops (§A.2) |
| Qwen3-ASR-0.6B | whole model | 2 / 30 utterances loop |

The common factor is **INT8 quantisation of an autoregressive decoder**.
It is worth treating as a standing expectation rather than a surprise:
non-autoregressive backends have not shown it — `paraformer-large` runs
INT8 on `zh` at RTF 0.035 without incident.

Even setting the degenerate utterances aside, CER 0.0838 against the
incumbent's 0.0269 is a threefold regression, so quantisation is
costing accuracy well before it costs coherence.

#### H.3 What is blocked

- **No language pin.** sherpa-onnx's Qwen3-ASR factory exposes no
  language parameter, though its Whisper factory does. The Chinese
  output above cannot be prevented at this layer.
- **No higher-precision export published** in that mirror — INT8 only.
- **1.7B not exported** ([sherpa-onnx#3535](https://github.com/k2-fsa/sherpa-onnx/issues/3535)
  is a feature request).
- **Mirror declares no licence**, as noted above.

#### H.4 Where to go next

The direction is validated and the artefact is not. In rough order of
expected value:

1. **Non-autoregressive candidates first.** sherpa-onnx 1.13 ships
   factories for `omnilingual_asr_ctc`, `dolphin_ctc`, `funasr_nano`,
   `fire_red_asr`, `moonshine_v2` and `cohere_transcribe`. The CTC ones
   cannot exhibit the failure mode in the table above, and
   `examples/score_hypotheses.rs` makes each a few minutes' work.
2. **A higher-precision Qwen3-ASR export**, if the CTC options do not
   pan out. If the degeneration is quantisation-induced, fp16 or fp32
   should resolve it and the structural advantage survives intact.
3. **Keeping Whisper and containing its int8 encoder collapse** with a
   repetition penalty. Even if that works it lands at RTF 0.37, so it
   is a stopgap rather than a destination.

### I. CTC candidates: Dolphin and Omnilingual-ASR (measured 2026-07-31)

§H.4 put non-autoregressive backends first. This section measures the
two sherpa-onnx CTC factories that cover Korean, and settles the
Korean routing question.

#### I.1 These backends do not reproduce above one thread

This has to come first, because it invalidates a number reported
earlier in this investigation.

Running Dolphin small INT8 over the same 30 FLEURS-ko utterances, from
a byte-identical model file, with `num_threads=4` on a four-core
container:

| run | output md5 | CER |
|---|---|---|
| 1 | `ad55c8da…` | 0.0921 |
| 2 | `35a8c813…` | 0.0895 |
| 3 | `ae4b1e05…` | 0.0865 |
| 4 | `462403ca…` | 0.0926 |
| 5 | `84cd4638…` | 0.0818 |

Five runs, five different transcripts. The variation is not a rounding
artefact — individual utterances come back truncated in one run and
complete in the next.

At one intra-op thread the same model is exactly reproducible
(Dolphin small throughout):

| threads | runs | distinct outputs | CER |
|---|---|---|---|
| 1 | 3 | 1 | **0.0655** |
| 2 | 3 | 1 | 0.0655 |
| 4 | 5 | 5 | 0.0818 – 0.0926 |

Dolphin **base** additionally disagreed with itself at two threads
(0.1565 / 0.1597 over two runs), so "≤ 2 is safe" is not the rule —
**one thread is the only setting that reproduces**, and it is also the
most accurate one. Omnilingual at one thread was likewise identical
across two runs.

The incumbent is not affected: `whisper-large-v3-turbo` q4 through
`src/whisper_onnx/adapter.rs`, which pins `intra = min(4, nproc) = 4`
and `inter = 1`, returned CER 0.0269 on two consecutive runs. So this
is specific to the sherpa-onnx CTC path rather than a general property
of ONNX Runtime in this stack — but it is a property of the runtime we
would be adopting, and §I.5 treats it as a porting requirement rather
than a footnote.

`scripts/run_sherpa_ctc.py` therefore defaults to `--threads 1` and
prints a warning above that. Every accuracy figure below is from a
single-threaded, reproduced run; the RTF figures are too, which
understates the speed advantage rather than inflating it.

**Correction.** Dolphin small was first reported at CER 0.0925, and an
earlier version of the §I.4 decomposition was computed from that run. That figure was a single draw from the four-thread
distribution above. The reproducible value is 0.0655, and §I.4 has been
recomputed against it.

#### I.2 Results

| Field | Dolphin | Omnilingual-ASR |
|---|---|---|
| Upstream | [`DataoceanAI/Dolphin`](https://github.com/DataoceanAI/Dolphin) | Meta Omnilingual ASR (1600 languages) |
| Export measured | `sherpa-onnx-dolphin-{small,base}-ctc-multi-lang-int8-2025-04-02` | `sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12` |
| License | **Apache-2.0** (model card YAML `license: apache-2.0`, upstream declaration on the repo above) | **Apache-2.0** (`LICENSE` shipped in the bundle, © Meta Platforms) |
| Architecture | CTC branch only of a hybrid model; base 0.1B / small 0.4B params (`medium` 0.9B and `large` 1.7B are announced but not published) | encoder + CTC head, 300M params |
| Korean | one of 40 supported Eastern languages (plus 22 Chinese dialects) | one of ~1600 |

Measured on the same 30 FLEURS-ko utterances, same container, same
`eval::metrics::cer_lenient` via `examples/score_hypotheses.rs`, all at
one intra-op thread except the incumbent (see §I.1):

| | model size | CER (lenient) | RTF | reproducible |
|---|---|---|---|---|
| `whisper-large-v3-turbo` q4 (incumbent, 4 threads) | ~900 MB | **0.0269** | 0.584 | yes |
| **Dolphin small INT8** | 239 MB | 0.0655 | **0.094** | yes |
| Dolphin base INT8 | 99 MB | 0.1565 | 0.044 | yes |
| Omnilingual 300M-CTC INT8 | 349 MB | 0.1655 | 0.377 | yes |
| `Qwen3-ASR-0.6B` INT8 (§H) | ~1 GB | 0.1911 | 0.324 | not tested |

**Omnilingual is out.** At 0.1655 it is the least accurate candidate
measured *and* costs 4× Dolphin small's RTF for it — 1600-language
coverage buys nothing here. Its characteristic failure is dropping or
mangling numeric content outright (`2011년 8월에` → `년 파월에`,
`1만 년 전에` → `년전에`), which is the one error class §I.4 shows is
otherwise recoverable, so it forecloses the only available remedy.

**Dolphin base is out.** It buys 2.1× the speed of small for 2.4× the
error, which is the wrong side of the trade given small is already 6×
faster than the incumbent.

**The §H.2 expectation held.** All three CTC backends ran INT8 with
zero degenerate utterances, against 2/30 for the INT8 autoregressive
Qwen3-ASR export and a full collapse for the INT8 Whisper decoder.
Non-autoregressive decoding does not exhibit the repetition failure
mode, as predicted.

#### I.3 Dolphin small against the incumbent

The comparison is deliberately unfair to Dolphin: it runs on **one**
thread, the incumbent on **four**.

| | whisper-turbo q4 (4 threads) | Dolphin small (1 thread) |
|---|---|---|
| CER (lenient) | 0.0269 | 0.0655 |
| RTF | 0.584 | **0.094** |
| 4.8 s utterance (`1907`) | 7297 ms | **405 ms** |
| 18.9 s utterance (`1715`) | 7743 ms | 1824 ms |
| cost model | fixed 30 s encoder pass (§A.2) | proportional to audio |

**6.2× faster in aggregate, 2.4× less accurate.** But the per-utterance
rows are the ones that matter for dictation: §A.2 established that
Whisper charges the same ~7 s for five seconds of speech as for
nineteen, because the mel front-end pads to 30 seconds. A CTC model has
no window to pad — 3.9× the audio costs 4.5× the time. On the 4.8-second
utterance the gap is not 6× but **18×**.

The aggregate RTF understates the advantage twice over — once through
the thread handicap, once through the utterance-length mix of a
read-speech corpus, which is longer-winded than dictation.

#### I.4 What euhadra's own layers can recover

**Corrected 2026-08-01.** This section previously claimed that Korean
ITN recovers about a quarter of Dolphin's error. That was wrong, and
the error was in the measurement rather than the arithmetic. It is
rewritten below; §I.4.1 records what happened, because the mistake is
the reusable part.

The honest answer: **Tier 1/2 post-processing recovers essentially
nothing of the measured gap.**

| error class | recoverable | why |
|---|---|---|
| numeral surface form | **not on this metric** | `cer_lenient` already normalises Sino-Korean numerals on both sides (§I.4.1). Real product value, zero measurable CER value |
| deletions (`주기율표 상에 원소가` → `주기`) | **no** | no layer can restore audio the ASR never emitted |
| general-vocabulary substitutions (`직후`→`찍고`, `봉쇄`→`봉세`) | no | needs a language model; that is Tier 3's job, not Tier 1/2's |
| proper nouns | in principle | `PhonemeCorrector` needs a Korean IPA lexicon and a Korean G2P; CMUdict and DeepPhonemizer are English-only. And it only fixes terms the *user* registered — `직후`→`찍고` is ordinary vocabulary nobody puts in a dictionary |

Word spacing costs nothing either: `cer_lenient` strips whitespace
before aligning, so Korean 띄어쓰기 differences are already free.
Punctuation likewise — FLEURS references carry none.

**So the trade in §I.3 stands as measured, with nothing to add.** 2.4×
the error for 6.2× the throughput, and no post-processing discount.

##### I.4.1 How the numeral claim went wrong

Dolphin spells numbers out where the FLEURS reference uses digits:

```
ref: 다리 밑 수직 간격은 15미터이며 공사는 2011년 8월에 …
hyp: 다리미   수직 간격은 십오 미터이며 공사는 이천십일년 팔월에 …
```

That looks like an obvious cost, and splitting the 30 utterances on
whether the reference contains a digit appears to confirm it:

| subset | n | Dolphin small | whisper-turbo q4 |
|---|---|---|---|
| reference contains digits | 8 | 0.1112 | 0.0332 |
| reference has no digits | 22 | 0.0489 | 0.0246 |
| all | 30 | 0.0655 | 0.0269 |

Dolphin's error doubles on the digit-bearing subset; Whisper's barely
moves. I read that as a numeral-form penalty worth ~0.017 CER and
concluded Korean ITN would recover it.

**It is not a numeral-form penalty.** `eval::metrics::cer_lenient`
normalises Sino-Korean numerals to Arabic digits *as part of its
lenient pass*, on the reference and the hypothesis alike — it is in the
function's own doc comment. So the metric was already blind to the
thing I claimed to be measuring:

```
reference: 공사는 2011년 8월에 마무리되었다
hypothesis 공사는 2011년 8월에 마무리되었다      → CER 0.0000
hypothesis 공사는 이천십일년 팔월에 마무리되었다  → CER 0.0000
```

Those 8 utterances are simply harder for unrelated reasons. The
correlation was real; the causal reading was invented.

Measured directly rather than inferred — running the Korean ITN module
over the adapter's own output on the same 30 utterances:

| | CER |
|---|---|
| Dolphin (Rust adapter) | **0.0618** |
| + Korean ITN | 0.0646 |

**Worse, with zero utterances improved.** The regression came from
homograph misfires the module had not been evaluated against
(`돌연변이만이` → `돌연변20,000이`, `제공되나` → `제0되나`,
`십일 위` → `10일 위`), on top of a guard that corrupted 15 of 15
common words tried (`만났다` → `10,000났다`, `조용히` →
`1,000,000,000,000용히`) because Korean's Sino-numeral scale units are
also the opening syllables of ordinary vocabulary.

Three things worth keeping from this:

1. **Check what the metric already does before attributing an error to
   a cause the metric normalises away.** A subset split shows
   correlation; it does not identify a mechanism.
2. **Korean ITN still has product value and no CER value.** A user
   reading dictated text wants `2011년 8월`, not `이천십일년 팔월`.
   That is a real reason to want it — it is just not this document's
   reason, and it cannot be justified with these numbers.
3. **The upstream patch is not ready.** `patches/text-processing-rs-ko-itn.patch`
   remains staged, but with the defects above; it was submitted as
   [FluidInference/text-processing-rs#86](https://github.com/FluidInference/text-processing-rs/pull/86)
   and withdrawn the same day. A Korean span scanner needs a
   unit/counter lexicon or a morphological analyser to reach acceptable
   precision, which is a larger piece of work than the patch assumed.

#### I.5 Decision

**Adopt Dolphin small for the Korean path.** The trade is 2.4× the
error for 6.2× the throughput — 18× on a short utterance — and euhadra
is a dictation framework, where a fixed ~7-second wait per utterance is
a product defect and a CER difference of 0.04 is a nuisance. §A.2
established that Whisper's latency is structural, not a tuning problem;
this is the first measured backend that removes it without the INT8
degeneration of §H.2.

Follow-up work, in order:

1. ~~**Land the Korean ITN patch upstream.**~~ *Attempted and withdrawn
   — see §I.4.1.* Submitted as
   [FluidInference/text-processing-rs#86](https://github.com/FluidInference/text-processing-rs/pull/86)
   and closed the same day. The patch applies cleanly to
   `FluidInference/text-processing-rs@8a043f1` and passes the suite
   (1183 tests, adding no clippy warnings), but it corrupts ordinary
   Korean prose, and the CER gain that justified it was an artefact of
   a metric that already normalises numerals. Reviving it means adding
   a unit/counter lexicon, not re-sending the patch.
2. **Implement `DolphinAdapter`** — *done, §I.6.* A Rust ONNX adapter in the shape of
   `src/sensevoice/adapter.rs`, reusing `paraformer::fbank` for the
   front-end, behind the `onnx` feature gate. This is what "adopt"
   actually costs; the measurements here were taken through
   `sherpa-onnx`'s Python bindings, which euhadra does not depend on.
3. **Pin one intra-op thread, or prove otherwise.** *Done, §I.6.* §I.1 is a
   correctness requirement for the adapter, not a benchmarking
   footnote: a dictation backend that returns a different transcript
   each time it sees the same audio is not acceptable. The port
   defaults to a single thread and reproduces itself across runs.
4. **Re-measure when `medium` (0.9B) and `large` (1.7B) publish.**
   Small already sits at 0.0655 with base at 0.1565, so the size curve
   is steep here; medium is the most likely candidate to close the
   remaining gap to Whisper while keeping the CTC cost model. Neither
   checkpoint has a public export today, so this is standby work rather
   than a blocker on the items above.

Not adopted: Omnilingual-ASR (least accurate and 4× the cost),
Dolphin base (2.4× the error of small), Qwen3-ASR INT8 (§H.2).

#### I.6 The Rust port (2026-08-01)

`src/dolphin/` implements items 2 and 3 above. Measured on the same 30
utterances, in the same container, through
`examples/bench_dolphin_ko.rs`:

| | sherpa-onnx (Python, 1 thread) | `DolphinAdapter` (Rust) |
|---|---|---|
| CER (lenient) | 0.0655 | **0.0618** |
| RTF | **0.094** | 0.147 |
| identical across 3 runs | yes | **yes** |

**It reproduces itself**, which was the requirement: three runs, one
output md5. `INTRA_THREADS` is pinned to 1 in the adapter with §I.1
cited at the constant, so raising it means re-running that experiment
rather than reading a benchmark.

**It does not reproduce sherpa byte-for-byte** — 26 of 30 transcripts
differ, almost entirely in word spacing (`북 발트해` against `붓발트해`,
`다리미 수직` against `다리 이 수직`). The front-end is not the cause:
`FbankOpts::dolphin_default` is pinned against kaldi-native-fbank by
`tests/fixtures/dolphin_fbank_golden.json` and agrees to **< 2e-3 per
bin** on both a two-tone and a wideband case, with matching frame counts
on real audio. What remains is f32 accumulation differing between
`rustfft` and Kaldi's real-FFT, landing on a graph whose argmax is
demonstrably knife-edge — §I.1 measured this same model moving across
CER 0.082–0.093 purely from thread scheduling. A 0.0618/0.0655
difference sits well inside that band, and bit-equality across two FFT
implementations was never available. The port is the more accurate of
the two draws, but that is luck, not an improvement.

Three things that bit, all silent when wrong, all now pinned by tests:

- The front-end is **not** Paraformer's. Povey window against Hamming,
  `snip_edges = false` against `true`, `high_freq = -400` (Kaldi's
  "Nyquist minus 400 Hz") against full band, and Kaldi's
  `log(max(e, FLT_EPSILON))` against FunASR's `log(e + 1e-10)`. Each
  produces correctly-shaped features and a plausible transcript.
- The CMVN lives in the graph's `metadata_props`, not a sidecar. Skipping
  it is not an error, just a worse transcript, so
  `dolphin::adapter` has a test asserting a shifted CMVN actually moves
  the features.
- `tokens.txt` is the two-column `symbol<space>id` form, not the
  one-piece-per-line list `sensevoice::vocab` reads. Reading position as
  id would silently shift the whole vocabulary.

**RTF 0.147 against sherpa's 0.094** — the Rust path is ~1.6× slower
than the C++ reference, and still 4× faster than the incumbent Whisper
q4's 0.584. The gap is not diagnosed; the front-end recomputes a
512-point FFT per frame with no reuse, which is the first place to look
if it matters. It has not been optimised because the adopted backend is
already comfortably ahead of what it replaces.

#### I.7 Routing `ko` to it (2026-08-01)

The adapter existed but nothing dispatched to it — no factory, so no
runtime id, so nothing Menura could name. Now wired:

- `DolphinFactory`, runtime id **`"dolphin"`**. That string is a
  published contract (`src/router.rs`: "Must be stable across releases
  — Menura's config…"). `AdapterRequest.language` is accepted and
  ignored: the CTC graph takes `(x, x_len)` and nothing else, and the
  `<ko>`/`<ja>` tags in its vocabulary are output symbols, not a
  selector. Same shape as `paraformer`.
- `examples/eval_l1_smoke.rs` gains `--dolphin-ko-dir`, which **takes
  precedence over `--whisper-onnx-ko-dir`**.
- CI fetches the bundle and passes that flag. The whisper-onnx bundle
  is still fetched — it stays the runtime for anyone who wants the
  accuracy, and dropping it would leave that path untested.

##### The baseline moves, deliberately

This is the accuracy regression §I.3 described, arriving where it is
visible. On the CI's own 10-utterance subset, measured in this
container:

| | whisper-onnx q4 | Dolphin small |
|---|---|---|
| CER | 0.0095 | **0.0349** (+267%) |
| ASR p50 | 14791 ms (CI runner) | **1518 ms** (this container) |
| RTF | 1.339 (CI runner) | 0.134 (this container) |

Two things worth stating plainly rather than leaving in the diff:

**The trade looks worse on 10 utterances than on 30.** §I.3 measured
2.3× the error for 4.0× the speed across 30; the CI subset gives 3.7×
for 3.5× — roughly break-even. That is sampling noise on n=10, not a
change in the models, but it is the number CI will show. If the ko slot
is going to carry a routing decision, ten utterances is a thin basis
for it.

**Only the CER transfers between environments.** whisper-onnx q4 scores
0.0095 on the CI runner and 0.0095 here — exact. Latency and RTF do
not: this container is ~2.04× faster on the same model. The baseline's
ko latency entries are therefore *scaled estimates*, not measurements,
and the first CI run is what settles them.

##### What ko still costs, after the change

Against the other languages in `ci_baseline.json`, all on the same
runner:

| lang | backend | RTF | ASR p50 |
|---|---|---|---|
| zh | paraformer-large | 0.035 | 464 ms |
| ja | parakeet-tdt_ctc-0.6b | 0.106 | 1194 ms |
| en | canary-180m-flash | 0.107 | 969 ms |
| es | canary-180m-flash | 0.123 | 1085 ms |
| ko *(was)* | whisper-large-v3-turbo-q4 | 1.339 | 14791 ms |
| ko *(now, estimated)* | dolphin-small-ctc-int8 | ~0.27 | ~3100 ms |

ko was **11–38× slower than every other language**, and the only one in
the matrix that was not interactive at all: about a second everywhere
else, fifteen seconds for Korean — while carrying the *best* CER in the
table. Dolphin closes most of that and not all of it: ko stays roughly
3× the others on this corpus. Note the corpus flatters Whisper here,
being 11-second read speech where its fixed 30-second window is most
amortised; on the 4.8-second utterance in §I.3 the gap was 18×.

Still open: the Menura-side `asr_models.toml` entry (`ko.runtime =
"dolphin"`, separate repo), and Korean ITN, which §I.4.1 withdrew.

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

### Step 4 (decided — 2026-07-31)

Steps 1/1.5 chose `whisper-large-v3-turbo` q4 on accuracy alone, before
§A.2 decomposed its latency. With that decomposition in hand — ~90% of
per-utterance cost is a fixed 30-second encoder pass — the choice does
not survive: turbo is the right model for transcribing files and the
wrong one for dictation.

§H measured the autoregressive alternative (`Qwen3-ASR-0.6B`, structurally
right, INT8 export degenerate) and §I the non-autoregressive ones.
**The `ko` path moves to Dolphin small CTC** (§I.5): 6.2× the throughput
for 2.4× the error, 18× on a short utterance, Apache-2.0 throughout.

Whisper-ONNX stays in the tree and stays wired into CI; this is a
routing change for `ko`, not a removal. The porting work is §I.5's
list, headed by the Korean ITN patch — which is worth landing whichever
backend wins.

## What this PR does and doesn't do

This PR is **documentation only**:

- Records the candidates evaluated and their licence chains.
- Recommends the Step 1 measurement plan.
- Cross-links from `docs/model-licenses.md` so reviewers find this file from the canonical licence summary.

It does **not**:

- Change the default Korean routing — that switch waits on Step 1 measurement.
- Add a Wav2Vec2 or W2VBert adapter — those remain in their own issues (#92 / #93).
- Modify `setup_sensevoice.sh` or remove SenseVoice — see issue #83 conclusion notes.
