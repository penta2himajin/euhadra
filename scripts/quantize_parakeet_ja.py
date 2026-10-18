# /// script
# requires-python = "==3.11.*"
# dependencies = ["onnxruntime==1.31.0", "onnx==1.23.2", "numpy==2.4.6"]
# ///
"""Build the INT8 Parakeet-ja bundle published as
`penta2himajin/parakeet-tdt_ctc-0.6b-ja-onnx-int8` (#173).

Only the encoder's constant-weight MatMuls are quantised (per-channel,
dynamic, QInt8 with 7-bit range). Quantising every op, as the public OpenVoiceOS INT8 build
does, costs FLEURS-ja CER 0.0566 → 0.0765; MatMul-only keeps it at 0.0549.
The decoder/joint stays FP32 (46 MB).

The output is byte-for-byte reproducible from the same input and pinned
versions, so the published files can be checked against `SHA256SUMS`.

    scripts/setup_parakeet_ja.sh --fp32        # FP32 source → vendor/parakeet_ja_fp32
    uv run scripts/quantize_parakeet_ja.py vendor/parakeet_ja_fp32 out/
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)

quantize_dynamic(
    src / "encoder-model.onnx",
    dst / "encoder-model.int8.onnx",
    op_types_to_quantize=["MatMul"],
    weight_type=QuantType.QInt8,
    per_channel=True,
    # 7-bit weights: x86 without VNNI multiplies U8S8 with VPMADDUBSW,
    # which saturates a 16-bit pair sum (255·127·2 > 32767). Per-channel
    # scales put every channel at ±127. Without this, the same files
    # scored ja CER 0.30 in one CI job and 0.07 in another (#175);
    # 255·63·2 stays below 32767. ARM and VNNI x86 are unaffected.
    reduce_range=True,
    use_external_data_format=True,
)
for f in ["decoder_joint-model.onnx", "decoder_joint-model.onnx.data", "vocab.txt"]:
    shutil.copyfile(src / f, dst / f)
config = json.loads((src / "config.json").read_text())
config["encoder_path"] = "encoder-model.int8.onnx"
(dst / "config.json").write_text(json.dumps(config, indent=4) + "\n")

with open(dst / "SHA256SUMS", "w") as out:
    for f in sorted(p.name for p in dst.iterdir() if p.name != "SHA256SUMS" and p.is_file()):
        out.write(f"{hashlib.sha256((dst / f).read_bytes()).hexdigest()}  {f}\n")
print((dst / "SHA256SUMS").read_text(), end="")
