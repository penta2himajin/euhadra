#!/usr/bin/env python3
"""Aggregate raw JSON from the euhadra/hayamimi speed compare into a report."""
from __future__ import annotations

import argparse
import json
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path


PAIRS = [
    ("en", "euhadra_en.json", "hayamimi_en.json"),
    ("es", "euhadra_es.json", "hayamimi_es.json"),
    ("ja", "euhadra_ja.json", "hayamimi_ja.json"),
    ("zh", "euhadra_zh.json", "hayamimi_zh.json"),
    ("ko", "euhadra_ko.json", "hayamimi_ko.json"),
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def row(lang: str, e: dict, h: dict) -> str:
    if h["mean_rtf"] < e["mean_rtf"]:
        ratio = e["mean_rtf"] / h["mean_rtf"]
        winner = f"hayamimi ({ratio:.2f}× faster)"
    elif e["mean_rtf"] < h["mean_rtf"]:
        ratio = h["mean_rtf"] / e["mean_rtf"]
        winner = f"euhadra ({ratio:.2f}× faster)"
    else:
        winner = "tie"
    return (
        f"| {lang} | `{e['model']}` | {e['mean_rtf']:.3f} | {e['p50_asr_ms']:.0f} | "
        f"`{h['model']}` | {h['mean_rtf']:.3f} | {h['p50_asr_ms']:.0f} | "
        f"**{winner}** |"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, required=True)
    args = ap.parse_args()

    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    pairs = []
    for lang, ef, hf in PAIRS:
        e = load(args.raw_dir / ef)
        h = load(args.raw_dir / hf)
        pairs.append({"lang": lang, "euhadra": e, "hayamimi": h})

    host = {
        "hostname": socket.gethostname(),
        "cpu": cpu,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }

    lines = [
        "# euhadra vs hayamimi — shipping ASR speed (this host)",
        "",
        "Offline decode RTF on the same FLEURS 10-utt subset per language.",
        "No VAD / LID / punctuation — model+runtime only.",
        "",
        "Pairs are each project's **shipping default** for that language",
        "(not the same architecture).",
        "",
        "| Lang | euhadra | hayamimi |",
        "|---|---|---|",
        "| en | Canary-180M-Flash INT8 | Parakeet TDT 0.6B v3 INT8 |",
        "| es | Canary-180M-Flash INT8 | Parakeet TDT 0.6B v3 INT8 |",
        "| ja | Parakeet TDT-CTC 0.6B ja | ReazonSpeech Zipformer INT8 |",
        "| zh | Paraformer-large (quant) | Paraformer-zh INT8 (sherpa) |",
        "| ko | Dolphin small CTC INT8 | SenseVoice Small INT8 |",
        "",
        "## Host",
        "",
        f"- CPU: `{host['cpu']}`",
        f"- Platform: `{host['platform']}`",
        f"- Generated: `{host['generated_utc']}`",
        "",
        "## Results",
        "",
        "| Lang | euhadra model | mean RTF | p50 ms | hayamimi model | mean RTF | p50 ms | Faster |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for p in pairs:
        lines.append(row(p["lang"], p["euhadra"], p["hayamimi"]))

    lines += [
        "",
        "## Notes",
        "",
        "- RTF = total ASR wall time / total audio duration (warmup excluded).",
        "- euhadra uses its Rust `ort` adapters; hayamimi models run via `sherpa-onnx` Python (CPU).",
        "- Threading: hayamimi fixed to 4 threads; euhadra uses each adapter's ORT defaults.",
        "- Absolute numbers are host-specific; relative ranking on this machine is the point.",
        "",
        "## Raw JSON",
        "",
        f"Per-utterance dumps live under `{args.raw_dir}/`.",
        "",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.json_out.write_text(
        json.dumps({"host": host, "pairs": pairs}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
