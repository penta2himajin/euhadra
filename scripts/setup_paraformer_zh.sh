#!/usr/bin/env bash
#
# Download the FunASR Paraformer-large Chinese ONNX bundle from the
# `funasr/Paraformer-large` HuggingFace mirror and lay it out as
# `ParaformerAdapter::load` expects:
#
#   <DIR>/
#     model.onnx        ← renamed from model_quant.onnx (int8 quantized,
#                          identical graph schema, ~238 MB instead of ~880 MB)
#     am.mvn            ← global CMVN statistics (Kaldi NNet text format)
#     tokens.json       ← extracted from config.yaml's `token_list:`
#     config.yaml       ← kept for reference / debugging
#
# Idempotent: skips files that already exist. Pass `PARAFORMER_ZH_DIR`
# to override the default location.
#
# Usage:
#   scripts/setup_paraformer_zh.sh
#   PARAFORMER_ZH_DIR=/path/to/dir scripts/setup_paraformer_zh.sh
#
# Why the quantized variant: the upstream HF mirror only ships
# model_quant.onnx (int8). It produces nearly identical CER to the
# fp32 graph on FLEURS-zh (the FunASR project recommends quant for
# all CPU deployments). Swap in a non-quantised model.onnx by
# dropping it into $PARAFORMER_ZH_DIR — the script will skip the
# download.
#
# Licensing (informational — defer to upstream URLs for authoritative text;
# see docs/model-licenses.md for the consolidated table):
#   - FunASR runtime: MIT
#     Declaration + text: https://github.com/modelscope/FunASR/blob/main/LICENSE
#   - Paraformer-large model weights: Apache-2.0 (per HF mirror metadata)
#     Declaration: https://huggingface.co/funasr/Paraformer-large
#       (model-card YAML: `license: apache-2.0`)
#     Canonical text: https://www.apache.org/licenses/LICENSE-2.0.txt
#     Note: the FunASR project README claims pretrained models are
#     governed by https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE
#     (custom Alibaba license). The HuggingFace distribution we pull
#     from instead declares Apache-2.0, so this script treats that as
#     authoritative. See docs/model-licenses.md note ¹ for context.

set -euo pipefail

DIR="${PARAFORMER_ZH_DIR:-vendor/paraformer_zh}"
HF_REPO="https://huggingface.co/funasr/Paraformer-large/resolve/main"

mkdir -p "$DIR"

require() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[error] required tool '$1' not on PATH" >&2
        exit 3
    fi
}
require curl
require python3

# Order: small files first so a flaky link surfaces obvious errors
# before we eat 240 MB of bandwidth.
download() {
    local src="$1" dst="$2"
    if [[ -s "$dst" ]]; then
        echo "[skip] $(basename "$dst") already present"
        return 0
    fi
    echo "[get] $(basename "$dst")"
    curl -fL --retry 3 --retry-delay 2 --max-time 1200 \
        -o "$dst" "$HF_REPO/$src"
}

download "am.mvn"       "$DIR/am.mvn"
download "config.yaml"  "$DIR/config.yaml"

# Prefer a hand-placed model.onnx (fp32) if the user dropped one in.
if [[ ! -s "$DIR/model.onnx" ]]; then
    download "model_quant.onnx" "$DIR/model.onnx"
fi

# Derive tokens.json from config.yaml's `token_list:` block.
# `kaldi_native_fbank` and tokenizers' YAML parser are heavy deps;
# the canonical FunASR layout used to ship tokens.json directly
# but the public HF mirror only includes config.yaml — we extract
# it ourselves to keep the bundle self-contained.
if [[ ! -s "$DIR/tokens.json" ]]; then
    echo "[gen] tokens.json (from config.yaml)"
    python3 - "$DIR/config.yaml" "$DIR/tokens.json" <<'PY'
import json
import re
import sys

cfg_path, out_path = sys.argv[1], sys.argv[2]

# config.yaml uses block-style `token_list:` followed by `- "<blank>"`,
# `- "<s>"`, ... one token per line. We extract the first such block
# rather than depending on a YAML library.
with open(cfg_path, "r", encoding="utf-8") as fh:
    text = fh.read()

m = re.search(r"^token_list:\s*\n((?:\s*-\s.*\n)+)", text, re.MULTILINE)
if not m:
    sys.stderr.write("[error] no `token_list:` block found in config.yaml\n")
    sys.exit(2)

raw = m.group(1)
tokens = []
for line in raw.splitlines():
    line = line.strip()
    if not line.startswith("-"):
        continue
    tok = line[1:].strip()
    # Strip outer quotes (config.yaml uses both single and double quoted forms)
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ("'", '"'):
        tok = tok[1:-1]
    tokens.append(tok)

if len(tokens) < 100:
    sys.stderr.write(
        f"[error] token_list parse produced only {len(tokens)} entries — expected ~8404\n"
    )
    sys.exit(2)

with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(tokens, fh, ensure_ascii=False)

print(f"[ok] wrote {len(tokens)} tokens to {out_path}")
PY
fi

# Sanity: am.mvn + model + tokens must all be present.
for required in am.mvn model.onnx tokens.json; do
    if [[ ! -s "$DIR/$required" ]]; then
        echo "[error] $DIR is missing $required" >&2
        exit 4
    fi
done

echo "PARAFORMER_ZH_DIR=$DIR"
