# euhadra vs hayamimi — shipping ASR speed (this host)

Offline decode RTF on the same FLEURS 10-utt subset per language.
No VAD / LID / punctuation — model+runtime only.

Pairs are each project's **shipping default** for that language
(not the same architecture).

| Lang | euhadra | hayamimi |
|---|---|---|
| en | Canary-180M-Flash INT8 | Parakeet TDT 0.6B v3 INT8 |
| es | Canary-180M-Flash INT8 | Parakeet TDT 0.6B v3 INT8 |
| ja | Parakeet TDT-CTC 0.6B ja | ReazonSpeech Zipformer INT8 |
| zh | Paraformer-large (quant) | Paraformer-zh INT8 (sherpa) |
| ko | Dolphin small CTC INT8 | SenseVoice Small INT8 |

## Host

- CPU: `Intel(R) Xeon(R) Processor` (4 cores / 4 threads visible)
- Platform: `Linux-6.12.94+-x86_64-with-glibc2.39`
- Generated: `2026-08-29T10:29:59.515073+00:00`

## Results

| Lang | euhadra model | mean RTF | p50 ms | hayamimi model | mean RTF | p50 ms | Faster |
|---|---|---:|---:|---|---:|---:|---|
| en | `euhadra/canary-180m-flash-int8` | 0.049 | 474 | `hayamimi/parakeet-tdt-0.6b-v3-int8` | 0.045 | 440 | **hayamimi (1.09× faster)** |
| es | `euhadra/canary-180m-flash-int8` | 0.052 | 466 | `hayamimi/parakeet-tdt-0.6b-v3-int8` | 0.044 | 447 | **hayamimi (1.19× faster)** |
| ja | `euhadra/parakeet-tdt_ctc-0.6b-ja` | 0.032 | 367 | `hayamimi/reazonspeech-zipformer` | 0.017 | 178 | **hayamimi (1.95× faster)** |
| zh | `euhadra/paraformer-large` | 0.014 | 190 | `hayamimi/paraformer-zh-int8` | 0.014 | 178 | **hayamimi (1.01× faster)** |
| ko | `euhadra/dolphin-small-ctc-int8` | 0.052 | 627 | `hayamimi/sensevoice-small-int8` | 0.019 | 211 | **hayamimi (2.76× faster)** |

## Notes

- RTF = total ASR wall time / total audio duration (warmup excluded).
- euhadra uses its Rust `ort` adapters; hayamimi models run via `sherpa-onnx` Python (CPU).
- Threading: hayamimi fixed to 4 threads; euhadra uses each adapter's ORT defaults.
- Absolute numbers are host-specific; relative ranking on this machine is the point.

## Raw JSON

Per-utterance dumps live under `docs/benchmarks/hayamimi_speed_compare/raw/`.

