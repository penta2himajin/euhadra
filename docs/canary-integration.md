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

First live run against the real istupakov bundle on a FLEURS-es
10-utterance subset (`scripts/setup_canary_es.sh` + `--canary-es-dir`):

```
[es] n=10  CER=0.0975  RTF=0.091
asr p50=801ms p95=2098ms
```

Per-utterance CER spread:

| # | CER | Notes |
|---|---|---|
| 1 | 0.0142 | minor article diff (`del clima` vs `de clima`) |
| 2 | 0.0672 | drops `«` punctuation, "Orabek dos mil dos" instead of "oravec 2002" |
| 3 | 0.0000 | perfect |
| 4 | 0.0149 | drops `¿` `?` |
| 5 | 0.0112 | "Carpaneo" vs "carpanedo" + extra commas |
| 6 | 0.0252 | "sacar puntas" vs "sacapuntas" |
| 7 | 0.0000 | perfect |
| 8 | **0.2475** | drops a chunk: `tiene dos hijos adultos no causó` |
| 9 | **0.5664** | drops a chunk: `fenómenos climáticos regionales y estacionales extremos encontramos` |
| 10 | 0.0286 | minor preposition diff (`que forma parte de` vs `que forma parte`) |

**Assessment**: 8/10 utterances are within model-card range (0–7 % CER).
The two outliers (8, 9) drop multi-word chunks of audio — same failure
mode in both, suggesting an autoregressive coverage issue rather than
a frontend/encoder problem (frontend output is finite and shape is
correct; the problem is the decoder skipping ahead).

The MLS-Spanish reference is **WER 3.17 %**, not CER, so the headline
9.75 % CER is not directly comparable. CER is roughly 0.7× of WER for
Spanish, so the equivalent WER would be ~14 % — above the model-card
number. Without the two outliers the 8-sample mean CER is **1.7 %**
(equiv. WER ~2.5 %), which IS within model-card range.

### Fallback-trigger status

| Trigger | Status |
|---|---|
| > 1 pp WER regression vs model card | **Inconclusive** — sample too small + outliers dominate. Need a wider FLEURS-es subset (≥100 utterances) and a WER metric to call. |
| Decoder loop > 2 weeks of work | **Cleared** — frontend → vocab → encoder → decoder → adapter delivered in 5 PRs over a single session. |
| License clarification trouble | **Cleared** — istupakov bundle is upstream CC-BY-4.0 unchanged. |

### Next investigation steps (in order)

1. Diff the two outliers' encoder outputs vs `onnx-asr`'s Python
   reference on the same WAV to isolate whether the regression comes
   from our Rust pipeline or is intrinsic to the istupakov export.
2. Widen the FLEURS-es subset to 100 utterances and switch metric to
   WER to compare with the model card directly.
3. If the outliers remain, profile the decoder's argmax stream to see
   whether `<|endoftext|>` is being emitted prematurely (the most
   likely root cause of mid-utterance dropouts).
