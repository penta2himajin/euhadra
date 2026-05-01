#!/usr/bin/env python3
"""
Compare numerical tensors emitted by

  scripts/dump_canary_python_tensors.py     (numpy `.npy`)
  examples/dump_canary_rust_tensors.rs       (raw `.bin`)

on the same WAV. Prints a max-abs / max-rel error per tensor and
the L2 norm of the difference. Used to confirm the Rust pipeline
matches `onnx-asr`'s Python reference.

The first-PR finding (recorded in `docs/canary-integration.md`):
on FLEURS-es utterance 2001 the first two greedy tokens match
exactly, and on the failing utterances 1725 / 1915 both
implementations produce the same `Asus E E E …` and `boda boda …`
loops — the catastrophic failure modes are upstream model / export
behaviour, not Rust-side bugs.

Usage:
  python3 scripts/compare_canary_tensors.py \\
      --python data/cache/canary_python_dump \\
      --rust data/cache/canary_rust_dump
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def load_rust_bin(path: Path) -> np.ndarray:
    """Load a Rust-side `.bin` tensor (header + raw f32 little-endian)."""
    raw = path.read_bytes()
    (ndim,) = struct.unpack("<I", raw[:4])
    shape = struct.unpack(f"<{ndim}I", raw[4 : 4 + 4 * ndim])
    body = raw[4 + 4 * ndim :]
    arr = np.frombuffer(body, dtype=np.float32).reshape(shape)
    return arr


def diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Returns (max_abs, max_rel, l2)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.shape != b.shape:
        return (float("nan"), float("nan"), float("nan"))
    delta = a - b
    max_abs = float(np.max(np.abs(delta)))
    denom = float(np.max(np.abs(b))) or 1.0
    max_rel = max_abs / denom
    l2 = float(np.linalg.norm(delta))
    return max_abs, max_rel, l2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--rust", type=Path, required=True)
    args = p.parse_args()

    pairs: list[tuple[str, str]] = [
        ("mel.npy", "mel.bin"),
        ("encoder_emb.npy", "encoder_emb.bin"),
        ("encoder_mask.npy", "encoder_mask.bin"),
        ("prefix.npy", "prefix.bin"),
        ("step0_logits.npy", "step0_logits.bin"),
        ("step0_hidden.npy", "step0_hidden.bin"),
        ("step1_logits.npy", "step1_logits.bin"),
        ("step1_hidden.npy", "step1_hidden.bin"),
        ("next_token_step0.npy", "next_token_step0.bin"),
        ("next_token_step1.npy", "next_token_step1.bin"),
    ]
    print(f"{'tensor':<26} {'py_shape':<22} {'rs_shape':<22} {'max_abs':>11} {'max_rel':>11} {'l2':>11}")
    for py_name, rs_name in pairs:
        py_path = args.python / py_name
        rs_path = args.rust / rs_name
        if not py_path.exists() or not rs_path.exists():
            print(f"{py_name:<26} (missing)")
            continue
        py = np.load(py_path).astype(np.float32)
        rs = load_rust_bin(rs_path)
        ma, mr, l2 = diff(py, rs)
        py_shape = "x".join(map(str, py.shape))
        rs_shape = "x".join(map(str, rs.shape))
        print(
            f"{py_name:<26} {py_shape:<22} {rs_shape:<22} "
            f"{ma:>11.4g} {mr:>11.4g} {l2:>11.4g}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
