# Model Licenses

euhadra integrates several third-party ASR / runtime components as optional
backends. This document is an informational summary of each component's
upstream license. **Always defer to the upstream license URL** for the
authoritative text. None of the entries below constitute legal advice.

Each row in the tables below lists two URLs:

- **License declaration** — where the *specific distribution* we consume
  declares which license it uses (the HuggingFace model card or the
  upstream repository).
- **License text** — the canonical, immutable text of that license
  (SPDX-compatible URL or the upstream LICENSE file).

If a row's declaration URL ever stops resolving or its declared license
changes, the **License text** column still resolves to the canonical
license body.

## ASR model weights

| Model | SPDX / Label | License declaration | License text |
|---|---|---|---|
| OpenAI Whisper (`large-v3` / `base` / `tiny` etc.) | `Apache-2.0` | <https://github.com/openai/whisper/blob/main/LICENSE> | <https://www.apache.org/licenses/LICENSE-2.0.txt> |
| `openai/whisper-large-v3-turbo` (HF weights) | `MIT` | <https://huggingface.co/openai/whisper-large-v3-turbo> (model-card YAML: `license: mit`) | <https://opensource.org/licenses/MIT> |
| `onnx-community/whisper-large-v3-turbo` (ONNX mirror — what `scripts/setup_whisper_onnx_turbo.sh` downloads) | `MIT` (inherited) | <https://huggingface.co/onnx-community/whisper-large-v3-turbo> (model-card YAML: `license: mit`) | <https://opensource.org/licenses/MIT> |
| `nvidia/canary-180m-flash` | `CC-BY-4.0` | <https://huggingface.co/nvidia/canary-180m-flash> (model-card YAML: `license: cc-by-4.0`) | <https://creativecommons.org/licenses/by/4.0/legalcode.en> |
| `istupakov/canary-180m-flash-onnx` (ONNX mirror) | `CC-BY-4.0` (inherited) | <https://huggingface.co/istupakov/canary-180m-flash-onnx> (model-card YAML: `license: cc-by-4.0`) | <https://creativecommons.org/licenses/by/4.0/legalcode.en> |
| `nvidia/parakeet-tdt_ctc-0.6b-ja` | `CC-BY-4.0` | <https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja> (model-card YAML: `license: cc-by-4.0`) | <https://creativecommons.org/licenses/by/4.0/legalcode.en> |
| `sunilmahendrakar/parakeet-tdt-0.6b-ja-onnx` (ONNX mirror) | `CC-BY-4.0` (inherited) | <https://huggingface.co/sunilmahendrakar/parakeet-tdt-0.6b-ja-onnx> (model-card YAML: `license: cc-by-4.0`) | <https://creativecommons.org/licenses/by/4.0/legalcode.en> |
| `funasr/Paraformer-large` | `Apache-2.0` (per HF mirror — see note ¹) | <https://huggingface.co/funasr/Paraformer-large> (model-card YAML: `license: apache-2.0`) | <https://www.apache.org/licenses/LICENSE-2.0.txt> |
| `FunAudioLLM/SenseVoiceSmall` | `LicenseRef-FunASR-MODEL-LICENSE-1.1` (HF declares `license: other`, `license_link` points at the file below) | <https://huggingface.co/FunAudioLLM/SenseVoiceSmall> (model-card YAML: `license: other`, `license_name: model-license`, `license_link: …/FunASR/MODEL_LICENSE`) | <https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE> |

The SenseVoice licence posture is the subject of issue [#83] — a research
log of license-clean Korean ASR alternatives lives at
[`korean-asr-alternatives.md`](./korean-asr-alternatives.md).

[#83]: https://github.com/penta2himajin/euhadra/issues/83

**¹** Note on Paraformer-large: The FunASR project's repository README
states that "the use of pretraining model is subject to the
[model license](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)"
(custom Alibaba license). However, the HuggingFace mirror at
`funasr/Paraformer-large` — which is the actual distribution point our
`scripts/setup_paraformer_zh.sh` pulls weights from — declares
`license: apache-2.0` in its model-card YAML frontmatter. This document
treats the declaration on the consumed distribution as authoritative,
but the divergence is worth flagging if you rely on stricter compliance
with the FunASR project's umbrella statement.

## Runtime / framework licenses

| Component | SPDX | License URL |
|---|---|---|
| whisper.cpp (ggerganov) | `MIT` | <https://github.com/ggerganov/whisper.cpp/blob/master/LICENSE> |
| FunASR runtime (modelscope) | `MIT` | <https://github.com/modelscope/FunASR/blob/main/LICENSE> |
| SenseVoice repository code (FunAudioLLM) | (redirects to FunASR's license) | <https://github.com/FunAudioLLM/SenseVoice/blob/main/LICENSE> |

## License-specific attribution notes

Bullets below paraphrase the upstream license text — they are **not** a
substitute for reading the canonical text linked above.

### Apache-2.0
- Include a copy of the license and the original copyright notice.
- State significant modifications if you redistribute a derivative.
- Canonical text: <https://www.apache.org/licenses/LICENSE-2.0.txt>

### MIT
- Retain the copyright notice and permission notice.

### CC-BY-4.0
- Credit the licensor (NVIDIA / mirror author, as applicable).
- Provide a link to the CC-BY-4.0 legalcode or include its text.
- Indicate whether you modified the work.
- Canonical text: <https://creativecommons.org/licenses/by/4.0/legalcode.en>

### FunASR Model License v1.1 (`LicenseRef-FunASR-MODEL-LICENSE-1.1`)
- Attribute source + author and retain the original model name (§2.2).
- **Custom Alibaba license, not OSI-approved.** Commercial use is
  **not** explicitly prohibited (§2.1: "You are free to use, copy,
  modify, and share").
- Adds a §4.2 "no malicious smearing" community-conduct clause whose
  violation automatically terminates the grant (§5).
- Upstream maintainers have not publicly confirmed commercial use in
  the issues that explicitly raised the question
  ([FunAudioLLM/SenseVoice#277](https://github.com/FunAudioLLM/SenseVoice/issues/277),
  [FunAudioLLM/SenseVoice#279](https://github.com/FunAudioLLM/SenseVoice/issues/279)).
  Consult counsel before relying on it for commercial distribution.
- Canonical text:
  <https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE>

## Errata

A prior version of `scripts/setup_sensevoice.sh` described
SenseVoice-Small as **CC-BY-NC-4.0 / non-commercial only**. That was
incorrect: the actual license declared on the HuggingFace model card is
the FunASR Model License v1.1 (custom Alibaba license, no
non-commercial clause). The wording was corrected together with the
introduction of this document; the **License text** column above is the
authoritative source.
