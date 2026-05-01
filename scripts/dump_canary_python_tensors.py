#!/usr/bin/env python3
"""
Dump intermediate tensors from `onnx_asr`'s NemoConformerAED running
Canary-180M-Flash on a single WAV. We dump:

  mel.npy            log-mel features after CMVN  [T, n_mels=128]  f32
  encoder_emb.npy    encoder_embeddings           [1, T_sub, D]    f32
  encoder_mask.npy   encoder_mask                 [1, T_sub]       i64
  step0_logits.npy   logits at the first decode step [1, prefix_len, V] f32
  step1_logits.npy   logits at the second decode step [1, 1, V]    f32
  prefix.npy         the 10-token prefix as fed to the decoder  i64
  next_token_step0.npy   argmax of step0_logits[:, -1, :]   i64
  next_token_step1.npy   argmax of step1_logits[:, -1, :]   i64

Run it on a stable FLEURS-es utterance so we have ground truth to
compare the Rust pipeline against. Default: data/fleurs_subset/es/audio/2001.wav.

Usage:
  python3 scripts/dump_canary_python_tensors.py \\
      --wav data/fleurs_subset/es/audio/2001.wav \\
      --model-dir models/canary-180m-flash-onnx \\
      --out data/cache/canary_python_dump
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx_asr
import onnxruntime as ort


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--wav", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--use-int8",
        action="store_true",
        help="Use the *.int8.onnx pair instead of the FP32 pair.",
    )
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # `onnx_asr` doesn't expose intermediate tensors via its public
    # API, so we rebuild the pipeline by hand: load encoder/decoder
    # ONNX sessions directly + reuse the preprocessor and vocab
    # helpers from `onnx_asr` so we get exactly the same numerics.
    from onnx_asr.preprocessors.numpy_preprocessor import NemoPreprocessorNumpy
    import soundfile as sf

    encoder_filename = "encoder-model.int8.onnx" if args.use_int8 else "encoder-model.onnx"
    decoder_filename = "decoder-model.int8.onnx" if args.use_int8 else "decoder-model.onnx"

    encoder_path = args.model_dir / encoder_filename
    decoder_path = args.model_dir / decoder_filename
    vocab_path = args.model_dir / "vocab.txt"

    # Load vocab — onnx-asr replaces ▁ with space when building the dict.
    vocab: dict[int, str] = {}
    with vocab_path.open("rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            piece, idx = line.rsplit(" ", 1)
            vocab[int(idx)] = piece.replace("▁", " ")
    tokens = {token: idx for idx, token in vocab.items()}

    # Build the 10-token prefix exactly like NemoConformerAED.__init__.
    prefix = np.array(
        [
            [
                tokens[" "],
                tokens["<|startofcontext|>"],
                tokens["<|startoftranscript|>"],
                tokens["<|emo:undefined|>"],
                tokens["<|es|>"],   # source = es
                tokens["<|es|>"],   # target = es (ASR mode)
                tokens["<|pnc|>"],
                tokens["<|noitn|>"],
                tokens["<|notimestamp|>"],
                tokens["<|nodiarize|>"],
            ]
        ],
        dtype=np.int64,
    )
    np.save(args.out / "prefix.npy", prefix)

    eos_token_id = tokens["<|endoftext|>"]
    print(f"prefix={prefix.tolist()}", file=sys.stderr)
    print(f"eos_token_id={eos_token_id}", file=sys.stderr)

    # --- Frontend (mel + CMVN) ---
    audio, sr = sf.read(str(args.wav))
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if sr != 16000:
        raise SystemExit(f"expected 16 kHz audio, got {sr}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    pre = NemoPreprocessorNumpy("nemo128")
    waveforms = audio[None, :]
    waveforms_lens = np.array([audio.shape[0]], dtype=np.int64)
    # NemoPreprocessorNumpy.__call__ returns (features [B, n_mels, T], features_lens [B]).
    features, features_lens = pre(waveforms, waveforms_lens)
    print(
        f"features.shape={features.shape} dtype={features.dtype} "
        f"features_lens={features_lens.tolist()}",
        file=sys.stderr,
    )
    # Save in [T, n_mels] row-major layout to match Rust frontend output.
    mel_t_first = features[0]  # [n_mels, T]
    mel_t_major = mel_t_first.T.copy()  # [T, n_mels] row-major
    np.save(args.out / "mel.npy", mel_t_major.astype(np.float32))
    np.save(args.out / "features.npy", features.astype(np.float32))
    np.save(args.out / "features_lens.npy", features_lens.astype(np.int64))

    # --- Encoder ---
    enc = ort.InferenceSession(str(encoder_path))
    encoder_outputs = enc.run(
        ["encoder_embeddings", "encoder_mask"],
        {"audio_signal": features.astype(np.float32), "length": features_lens.astype(np.int64)},
    )
    encoder_embeddings, encoder_mask = encoder_outputs
    print(
        f"encoder_embeddings.shape={encoder_embeddings.shape} "
        f"encoder_mask.shape={encoder_mask.shape}",
        file=sys.stderr,
    )
    np.save(args.out / "encoder_emb.npy", encoder_embeddings)
    np.save(args.out / "encoder_mask.npy", encoder_mask)

    # --- Decoder: first two steps ---
    dec = ort.InferenceSession(str(decoder_path))
    shapes = {x.name: x.shape for x in dec.get_inputs()}
    L = shapes["decoder_mems"][0]
    H = shapes["decoder_mems"][3]
    print(f"decoder_mems shape spec: L={L} H={H}", file=sys.stderr)

    decoder_mems = np.empty((L, 1, 0, H), dtype=np.float32)

    # Step 0: send full prefix.
    logits0, hidden0 = dec.run(
        ["logits", "decoder_hidden_states"],
        {
            "input_ids": prefix,
            "encoder_embeddings": encoder_embeddings,
            "encoder_mask": encoder_mask,
            "decoder_mems": decoder_mems,
        },
    )
    print(
        f"step0 logits.shape={logits0.shape} hidden.shape={hidden0.shape}",
        file=sys.stderr,
    )
    np.save(args.out / "step0_logits.npy", logits0)
    np.save(args.out / "step0_hidden.npy", hidden0)
    next0 = int(np.argmax(logits0[:, -1], axis=-1)[0])
    np.save(args.out / "next_token_step0.npy", np.array([next0], dtype=np.int64))
    print(f"step0 next_token={next0} ({vocab.get(next0, '?')!r})", file=sys.stderr)

    # Step 1: feed back the next token with hidden as decoder_mems.
    decoder_mems = hidden0
    next_input = np.array([[next0]], dtype=np.int64)
    logits1, hidden1 = dec.run(
        ["logits", "decoder_hidden_states"],
        {
            "input_ids": next_input,
            "encoder_embeddings": encoder_embeddings,
            "encoder_mask": encoder_mask,
            "decoder_mems": decoder_mems,
        },
    )
    print(
        f"step1 logits.shape={logits1.shape} hidden.shape={hidden1.shape}",
        file=sys.stderr,
    )
    np.save(args.out / "step1_logits.npy", logits1)
    np.save(args.out / "step1_hidden.npy", hidden1)
    next1 = int(np.argmax(logits1[:, -1], axis=-1)[0])
    np.save(args.out / "next_token_step1.npy", np.array([next1], dtype=np.int64))
    print(f"step1 next_token={next1} ({vocab.get(next1, '?')!r})", file=sys.stderr)

    print(f"\ndump complete → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
