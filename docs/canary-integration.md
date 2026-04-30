# Canary ONNX Integration — Decision Log & Fallback Plan

## Decision (2026-04-30)

Integrate **NVIDIA Canary-180M-Flash via the
[istupakov/canary-180m-flash-onnx](https://huggingface.co/istupakov/canary-180m-flash-onnx)
ONNX export** as the Spanish (and en/de/fr) ASR backend, replacing the
plan to wire Whisper-es subprocess as the MVP.

## Why not the original Canary plan

NeMo upstream `model.export()` blocks Canary export — see
[NVIDIA-NeMo/NeMo#11004](https://github.com/NVIDIA-NeMo/NeMo/issues/11004),
open since October 2024. Building the export pipeline ourselves was
estimated at 1–2 weeks before any inference code could land.

## Why istupakov over Parakeet-TDT-0.6B-v3-multi

| Axis | Canary-180M-Flash (istupakov) | Parakeet-TDT-0.6B-v3 |
|---|---|---|
| Size (INT8) | **~213 MB** (134 + 79.5) | ~671 MB |
| Spanish MLS WER | **3.17 %** | 4.39 % |
| Spanish FLEURS WER | (not reported in card; ~3.3 % expected) | 3.45 % |
| German MLS WER | **4.81 %** | (not reported per-lang) |
| French MLS WER | **4.75 %** | 4.97 % |
| AST (translation) | **yes** (en↔de/fr/es) | no |
| ONNX export | **available** | available |
| Existing Rust harness | none (this work) | `parakeet-rs` reusable |
| Languages | 4 (en/de/fr/es) | 25 (EU) |

Canary wins on accuracy-per-byte for our four target languages; the
size advantage matters for the on-device Cochlis distribution. The
loss is having to build a Rust inference loop from scratch (~500
lines), since `parakeet-rs` doesn't speak Canary's encoder-decoder
architecture.

## Architecture summary

The istupakov export ships **encoder + decoder ONNX graphs only** —
preprocessing and decoding are the caller's responsibility, mirrored
from the [`onnx-asr`](https://github.com/istupakov/onnx-asr) Python
reference.

| Component | Source | Effort | This repo |
|---|---|---|---|
| Mel preprocessor (Hann window, log-mel, CMVN) | `numpy_preprocessor.py` | ~150 lines Rust | `src/canary/frontend.rs` |
| SentencePiece vocab + special tokens | `vocab.txt` | ~80 lines Rust | `src/canary/vocab.rs` |
| Encoder ONNX call | `nemo.py` | ~50 lines, `ort` crate | `src/canary/encoder.rs` |
| Autoregressive decoder loop with KV cache | `nemo.py` `_decode` | ~120 lines | `src/canary/decoder.rs` |
| `CanaryAdapter` impl `AsrAdapter` trait | new | ~80 lines | `src/canary/adapter.rs` |
| Setup script (download ONNX) | new | ~40 lines bash | `scripts/setup_canary_es.sh` |

### Mel preprocessor parameters (verbatim from `onnx-asr`)
```
sample_rate     = 16_000
n_fft           = 512
win_length      = 400      # 25 ms
hop_length      = 160      # 10 ms
n_mels          = 128      # config.json features_size
preemph         = 0.97
window          = hann
log_guard       = 2**-24
norm            = per-feature mean/var across time (CMVN-like)
```

> **Note**: `onnx-asr`'s python source defaults `_features_size` to 80
> (the value Parakeet-TDT-0.6B-ja uses), but it actually reads
> `features_size` from `config.json` at runtime. The istupakov bundle
> ships `features_size=128`, and the encoder errors with
> `Got: 80 Expected: 128` if fed an 80-mel buffer. The Rust default
> matches the bundle.

### Decoder prefix layout
```
[<sot>, <ctx_lang>, <transcribe>, <pnc>, <lang>, ...]
                                          ^slot 4 = source language
```
Greedy decode with KV cache (`decoder_mems` shape `(layers, batch, T, hidden)`).

## Fallback triggers — when to abandon this plan

**Hard fallback** (switch to Parakeet-TDT-0.6B-v3-multi):
1. **Quality regression > 1 pp WER** vs. the istupakov model card on
   FLEURS-es / MLS-es when measured via our `eval_l1_smoke` harness.
   Indicates the export is lossy or our preprocessing/decoder differs
   from `onnx-asr`.
2. **Decoder loop > 2 weeks of work** beyond the initial frontend PR.
   We budgeted 1-2 weeks total; doubling that hits the opportunity
   cost where Parakeet-v3 (already runnable via `parakeet-rs`) becomes
   the better path despite its larger size.
3. **License clarification trouble** — if it turns out istupakov's
   export imposes additional constraints beyond the upstream
   CC-BY-4.0, we cannot ship it in Cochlis.

**Soft fallback** (keep Canary, augment with Parakeet for unsupported
langs):
4. **Need for languages outside en/de/fr/es** — Canary covers only
   four. If users request it/pt/nl/etc. we add Parakeet-v3-multi as a
   second adapter rather than dropping Canary.

## Migration plan if fallback triggered

The `AsrAdapter` trait already abstracts ASR backends. A fallback to
Parakeet-v3-multi is:
1. Drop `src/canary/` (or keep dormant under `--features canary`).
2. Add a `ParakeetMultiAdapter::load_es(...)` constructor in the
   existing `src/parakeet.rs`, downloading
   [`istupakov/parakeet-tdt-0.6b-v3-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx).
3. Update `scripts/setup_parakeet_es.sh` (analogous to
   `setup_parakeet_en.sh`).
4. Switch the CLI auto-select branch for `--language es` to the new
   adapter.

The Tier 1 SpanishFillerFilter and `eval_l3 --task filler --lang es`
work is independent of which ASR backend produces transcripts, so
none of that PR-19/20/21 work is at risk.

## Validation gates per PR

Each Canary PR must pass:
- Numerical sanity (frontend output non-NaN, finite, expected shape)
- Build clean on default + `--features onnx` features
- Clippy zero warnings
- Existing test suite green

Per-PR scope is intentionally small (frontend / vocab / encoder /
decoder / adapter / wiring) so any single PR failing review doesn't
strand the whole effort.

## End-to-end validation (2026-04-30)

### v1 — 10-utt smoke (initial integration check)

First live run against the istupakov bundle on a FLEURS-es 10-utterance
subset (`scripts/setup_canary_es.sh` + `--canary-es-dir`, INT8):

```
[es] n=10  CER=0.0975  RTF=0.091
asr p50=801ms p95=2098ms
```

8/10 utterances within model-card range (0–7 % CER). Two outliers
(utt 8 CER 24.75 %, utt 9 CER 56.64 %) dropped multi-word chunks —
same failure mode in both, suggesting an autoregressive coverage
issue rather than frontend/encoder.

### v2 — 100-utt run with WER as primary (2026-04-30)

Widened to 100 unique utterances (`download_fleurs_subset.py --lang
es --n 100` after the dedup-by-id fix that removes FLEURS multi-
speaker duplicates) and switched primary metric to WER (matches
Canary's published numbers).

**FP32 weights** (deterministic, identical results across 3 reruns):

```
[es] n=100  WER=0.3537  RTF=0.150
asr p50=1138ms p95=2958ms
median(WER)=0.083
```

WER bucket distribution (n=99 reportable, 1 hard failure
counted as 1.0 in the mean):

| WER range | count |
|---|---|
| 0–5 % | 41 |
| 5–10 % | 14 |
| 10–20 % | 19 |
| 20–50 % | 20 |
| 50–100 % | 3 |
| **> 100 % (repetition loop)** | **2** |
| Hard fail (no output) | 1 |

**INT8 weights** (non-deterministic across 3 reruns: 34.5 % / 36.4 %
/ 94.4 %, 7 hard failures vs FP32's 1):

```
[es] n=100  WER=~0.4 (mean of 3 runs, σ ≈ 0.27)  RTF=0.060
```

INT8 introduces additional non-determinism on top of the
greedy-decoding repetition issue, with the failure mode amplified
(7 hard failures + worst-case 94 % WER). FP32 is the right baseline
for evaluating model fidelity; INT8 quality is a separate question
once the underlying decoding is stabilised.

### Failure-mode inventory (FP32, n=100)

Three distinct decoder failure patterns observed:

1. **Hard fail (1 utt)** — decoder emits only the prefix, returns
   empty text. Pipeline raises "no speech detected".
2. **Chunk dropout (mid-WER)** — decoder skips a multi-word region
   in the middle of a long sentence, emitting only the head + tail.
   Same failure mode as v1 utterances 8 and 9.
3. **Repetition loop (utt 62, 63)** — decoder gets stuck repeating
   a single token until `max_sequence_length=1024` is hit. Examples:
   - `"Asus E E E E E E E …"` (~500 repeats of "E") for an audio
     containing the acronym "ASUS".
   - `"boda boda boda boda …"` (~500 repeats of "boda") for an
     audio containing "boda-boda mototaxi".

   These are the WER > 100 % outliers (12.6×, 8.6× the reference
   length).

All three are classic greedy-decode failure modes — production
autoregressive ASR systems mitigate them with **repetition penalty**,
**length penalty**, and/or **beam search**. None of those is
implemented in `src/canary/decoder.rs` yet.

### Fallback-trigger status

| Trigger | Status |
|---|---|
| > 1 pp WER regression vs model card | **TRIGGERED** — 35.37 % WER vs MLS model-card 3.17 % WER (Δ ≈ 32 pp) on FP32, and 99-utt mean drops to 16.6 % when the two repetition-loop outliers are removed (still Δ ≈ 13 pp). Note: MLS test ≠ FLEURS test, so the headline numbers aren't strictly apples-to-apples; FLEURS is generally harder for Spanish than MLS LibriVox audio. The Canary-1B-v2 (not 180M) FLEURS-es WER is 2.90 %; the 180M model's FLEURS WER isn't published, so a 13–32 pp gap is the right order-of-magnitude flag regardless. |
| Decoder loop > 2 weeks of work | **Cleared** — frontend → vocab → encoder → decoder → adapter delivered in 5 PRs over a single session. |
| License clarification trouble | **Cleared** — istupakov bundle is upstream CC-BY-4.0 unchanged. |

The first trigger fired but in a way that suggests **fixable
decoding problems rather than fundamental model unsuitability**:

- Median WER 8.3 % is within reasonable distance from the model card.
- 41/100 utterances achieve WER < 5 % (matching model-card quality).
- The catastrophic outliers all share the same root cause (greedy
  decode without repetition penalty / length control).

So the **soft-fallback decision is: invest in fixing the decoder
before triggering a hard fallback to Parakeet-v3-multi**.

### Next investigation steps (in order)

1. **Implement repetition penalty in `src/canary/decoder.rs`** —
   discount the logit of any token already emitted in the current
   utterance, with a tunable `repetition_penalty` parameter
   (typically 1.1–1.3 in the literature). Would fix utt 62 / 63.
2. **Implement minimum-length / EOS-confidence gate** — only accept
   `<|endoftext|>` if its logit dominates by some margin AND the
   total emitted length is plausible relative to encoder frames.
   Would address the hard-fail (1 utt) and chunk-dropout cases.
3. Re-run the 100-utt FP32 smoke and confirm WER drops below the
   1-pp-vs-model-card trigger (target: WER ≤ 5 % on FP32).
4. After (1)–(3), reassess INT8: with deterministic decoding, the
   INT8 quality question becomes "is INT8 within X pp of FP32?"
   rather than "is INT8 broken?".
5. If (1)–(3) don't move the needle below ~10 % WER, **execute the
   Parakeet-TDT-0.6B-v3-multi hard fallback** per the migration plan
   already documented in this file.
