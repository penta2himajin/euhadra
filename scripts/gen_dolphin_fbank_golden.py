#!/usr/bin/env python3
"""Regenerate the Kaldi golden fixture for Dolphin's feature front-end.

`DolphinAdapter` has to reproduce what sherpa-onnx feeds the model, and
that front-end is *not* the one `paraformer_default()` describes:
sherpa's `FeatureExtractorConfig` defaults to `snip_edges = False`,
`high_freq = -400` (Kaldi's "Nyquist minus 400 Hz" convention) and
kaldi-native-fbank's Povey window, against Paraformer's snipped edges,
full-Nyquist band and Hamming window. Those three differences are
silent — the shapes still line up, the features are just wrong — so the
Rust implementation is pinned against the reference C++ one rather than
against a reading of the source.

The waveform is a fixed two-tone signal so the fixture is reproducible
without shipping audio, and 0.25 s is long enough that the reflected
padding `snip_edges = False` applies at both ends shows up in the first
and last frames.

Usage (only needed when the front-end config changes):

    pip install kaldi-native-fbank
    scripts/gen_dolphin_fbank_golden.py > tests/fixtures/dolphin_fbank_golden.json
"""

import json
import math
import sys

import kaldi_native_fbank as knf

SAMPLE_RATE = 16_000
NUM_SAMPLES = 4_000
# Two tones, one low and one high, so every mel band carries signal and
# a filterbank that is subtly misaligned cannot pass by accident.
TONES = ((220.0, 0.5), (1750.0, 0.25))


def tone_waveform():
    return [
        sum(amp * math.sin(2 * math.pi * hz * i / SAMPLE_RATE) for hz, amp in TONES)
        for i in range(NUM_SAMPLES)
    ]


# Two pure tones leave most of the filterbank near the log floor, where
# a misplaced filter edge costs nothing. This second case excites every
# band at once. The generator is a plain LCG over exact 32-bit integers
# so Rust reproduces the samples bit-for-bit rather than approximately —
# a waveform that only *nearly* matches would show up as a front-end
# error that isn't one.
LCG_SEED = 12345
LCG_MUL = 1103515245
LCG_ADD = 12345
LCG_MOD = 1 << 31


def noise_waveform():
    out = []
    state = LCG_SEED
    for _ in range(NUM_SAMPLES):
        state = (LCG_MUL * state + LCG_ADD) % LCG_MOD
        # Map to [-0.5, 0.5] through a power-of-two divisor, which is
        # exact in binary floating point.
        out.append(state / float(LCG_MOD) - 0.5)
    return out


CASES = {"tones": tone_waveform, "noise": noise_waveform}


def options():
    o = knf.FbankOptions()
    # These four lines are exactly what sherpa-onnx's
    # FeatureExtractorConfig sets for the Dolphin CTC recogniser; the
    # rest are kaldi-native-fbank defaults (Povey window, preemph 0.97,
    # remove_dc_offset, round_to_power_of_two → 512-pt FFT).
    o.frame_opts.dither = 0.0
    o.frame_opts.snip_edges = False
    o.mel_opts.num_bins = 80
    o.mel_opts.low_freq = 20.0
    o.mel_opts.high_freq = -400.0
    return o


def frames_for(o, samples):
    fbank = knf.OnlineFbank(o)
    fbank.accept_waveform(SAMPLE_RATE, samples)
    fbank.input_finished()
    return [
        [float(v) for v in fbank.get_frame(i)] for i in range(fbank.num_frames_ready)
    ]


def main():
    o = options()
    cases = {name: frames_for(o, make()) for name, make in CASES.items()}
    json.dump(
        {
            "generator": "scripts/gen_dolphin_fbank_golden.py",
            "reference": f"kaldi-native-fbank {knf.__version__}",
            "sample_rate": SAMPLE_RATE,
            "num_samples": NUM_SAMPLES,
            "tones": [list(t) for t in TONES],
            "config": {
                "num_bins": o.mel_opts.num_bins,
                "low_freq": o.mel_opts.low_freq,
                "high_freq": o.mel_opts.high_freq,
                "dither": o.frame_opts.dither,
                "snip_edges": o.frame_opts.snip_edges,
                "window_type": o.frame_opts.window_type,
                "preemph_coeff": o.frame_opts.preemph_coeff,
                "remove_dc_offset": bool(o.frame_opts.remove_dc_offset),
            },
            "lcg": {"seed": LCG_SEED, "mul": LCG_MUL, "add": LCG_ADD, "mod": LCG_MOD},
            "cases": {
                name: {"num_frames": len(f), "frames": f} for name, f in cases.items()
            },
        },
        sys.stdout,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
