# Model Licenses

euhadra integrates several third-party ASR / runtime components as optional
backends. This document is an informational summary of each component's
upstream license — **always defer to the upstream license URL** for the
authoritative text. None of the entries below constitute legal advice.

## ASR model weights

| Model | Languages | Adapter | License | Upstream license URL |
|---|---|---|---|---|
| OpenAI Whisper (large-v3 / base / tiny etc.) | 99 | `WhisperLocal` | Apache 2.0 | <https://github.com/openai/whisper/blob/main/LICENSE> |
| `nvidia/canary-180m-flash` | en / de / fr / es | `CanaryAdapter` | CC-BY-4.0 | <https://huggingface.co/nvidia/canary-180m-flash> (model card) / <https://creativecommons.org/licenses/by/4.0/legalcode> |
| `istupakov/canary-180m-flash-onnx` (ONNX mirror) | en / de / fr / es | `CanaryAdapter` | CC-BY-4.0 (inherited from upstream) | <https://huggingface.co/istupakov/canary-180m-flash-onnx> |
| `nvidia/parakeet-tdt_ctc-0.6b-ja` | ja | `ParakeetAdapter` | CC-BY-4.0 | <https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja> (model card) / <https://creativecommons.org/licenses/by/4.0/legalcode> |
| `sunilmahendrakar/parakeet-tdt-0.6b-ja-onnx` (ONNX mirror) | ja | `ParakeetAdapter` | CC-BY-4.0 (inherited from upstream) | <https://huggingface.co/sunilmahendrakar/parakeet-tdt-0.6b-ja-onnx> |
| `funasr/Paraformer-large` | zh | `ParaformerAdapter` | FunASR Model License v1.1 (custom Alibaba license) | <https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE> |
| `FunAudioLLM/SenseVoiceSmall` | ja / ko / zh / yue / en | `SenseVoiceAdapter` | FunASR Model License v1.1 (custom Alibaba license — SenseVoice's own `LICENSE` is a 1-line redirect to this file) | <https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE> |

## Runtime / framework licenses

| Component | License | Upstream license URL |
|---|---|---|
| whisper.cpp (ggerganov) | MIT | <https://github.com/ggerganov/whisper.cpp/blob/master/LICENSE> |
| FunASR runtime (modelscope) | MIT | <https://github.com/modelscope/FunASR/blob/main/LICENSE> |
| SenseVoice repo code (FunAudioLLM) | redirects to FunASR | <https://github.com/FunAudioLLM/SenseVoice/blob/main/LICENSE> |

## Attribution requirements (summary)

The bullets below paraphrase the upstream texts. They are **not** a
substitute for reading the licenses linked above.

### Apache 2.0 — Whisper
- Include a copy of the license and the original copyright notice.
- State significant modifications if you redistribute a derivative.

### MIT — whisper.cpp, FunASR runtime
- Retain the copyright notice + permission notice.

### CC-BY-4.0 — Canary-180M-Flash, parakeet-tdt_ctc-0.6b-ja
- Credit NVIDIA as the original author.
- Provide a link to the CC-BY-4.0 license or include its text.
- Indicate if you modified the model.

### FunASR Model License v1.1 — Paraformer-large, SenseVoice-Small
- Attribute source + author and retain the original model name in
  derivative work (§2.2).
- **Custom Alibaba license, not OSI-approved.** Commercial use is
  **not** explicitly prohibited (see §2.1: "You are free to use, copy,
  modify, and share"). The license also adds a §4.2 "no malicious
  smearing" community-conduct clause whose violation terminates the
  grant.
- Upstream maintainers have not publicly confirmed commercial use in
  the issues that explicitly raised the question
  ([FunAudioLLM/SenseVoice#277](https://github.com/FunAudioLLM/SenseVoice/issues/277),
  [FunAudioLLM/SenseVoice#279](https://github.com/FunAudioLLM/SenseVoice/issues/279)),
  so users planning commercial distribution should consult counsel.

## Errata

A prior version of `scripts/setup_sensevoice.sh` described
SenseVoice-Small as **CC-BY-NC-4.0 / non-commercial only**. That was
incorrect: the actual license is the same FunASR Model License v1.1
that covers Paraformer-large, and it does not include a
non-commercial clause. The wording was corrected together with the
introduction of this document; the upstream license URL
(<https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE>) is the
authoritative source.
