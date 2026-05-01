//! Dump intermediate tensors from the euhadra Canary pipeline on a
//! single WAV — counterpart to `scripts/dump_canary_python_tensors.py`.
//!
//! Outputs (all f32 native-endian little-endian, with a 4-byte
//! header `[ndim:u32]` followed by `ndim * u32` shape and the raw
//! data; loadable from numpy via the helper `load_rust_tensor` in
//! `scripts/compare_canary_tensors.py`):
//!
//!   mel.bin              [T, 128] log-mel + CMVN
//!   encoder_emb.bin      [1, T_sub, 1024]
//!   encoder_mask.bin     [1, T_sub] (i64 stored as f32 for header
//!                                    simplicity — caller must cast)
//!   step0_logits.bin     [1, 10, V]
//!   step1_logits.bin     [1, 1, V]
//!   prefix.bin           [10] i64-as-f32
//!   next_token_step0.bin [1] i64-as-f32
//!   next_token_step1.bin [1] i64-as-f32
//!
//! The Python and Rust sides agree on shape order and element type
//! per the dump comparator.

use std::path::PathBuf;

use clap::Parser;
use euhadra::canary::{
    decoder::{
        argmax_last_position, build_decoder_prefix, CanaryDecoder, DecodeOptions,
        DECODER_INPUT_DECODER_MEMS, DECODER_INPUT_ENCODER_EMBEDDINGS,
        DECODER_INPUT_ENCODER_MASK, DECODER_INPUT_IDS, DECODER_OUTPUT_HIDDEN_STATES,
        DECODER_OUTPUT_LOGITS,
    },
    encoder::CanaryEncoder,
    frontend::{MelFrontend, MelOpts},
    vocab::Vocab,
};
use euhadra::whisper_local::read_wav;
use ndarray::{Array2, Array3, Array4};
use ort::session::Session;
use ort::value::Value;
use std::io::Write;

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long)]
    wav: PathBuf,
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long)]
    out: PathBuf,
}

fn write_tensor_f32(path: &std::path::Path, shape: &[usize], data: &[f32]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(&(shape.len() as u32).to_le_bytes())?;
    for d in shape {
        f.write_all(&(*d as u32).to_le_bytes())?;
    }
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    f.write_all(bytes)?;
    Ok(())
}

fn write_tensor_i64_as_f32(
    path: &std::path::Path,
    shape: &[usize],
    data: &[i64],
) -> std::io::Result<()> {
    let as_f32: Vec<f32> = data.iter().map(|x| *x as f32).collect();
    write_tensor_f32(path, shape, &as_f32)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    std::fs::create_dir_all(&cli.out)?;

    // --- Frontend ---
    let wav = read_wav(&cli.wav).map_err(|e| format!("read wav: {e}"))?;
    let samples = wav.samples;
    eprintln!(
        "audio: {} samples, sr={}, ch={}",
        samples.len(),
        wav.sample_rate,
        wav.channels
    );
    let opts = MelOpts::canary_default();
    let n_mels = opts.n_mels;
    let frontend = MelFrontend::new(opts);
    let (mel, n_frames) = frontend.compute(&samples);
    eprintln!("mel: {} frames × {} mels", n_frames, n_mels);
    write_tensor_f32(&cli.out.join("mel.bin"), &[n_frames, n_mels], &mel)?;

    // --- Encoder ---
    let encoder_path = cli.model_dir.join("encoder-model.onnx");
    let encoder = CanaryEncoder::load(&encoder_path)?;
    let enc_out = encoder.encode(&mel, n_mels, n_frames)?;
    let emb_shape = enc_out.embeddings.shape().to_vec();
    let mask_shape = enc_out.mask.shape().to_vec();
    eprintln!(
        "encoder_embeddings.shape={:?} encoder_mask.shape={:?}",
        emb_shape, mask_shape
    );
    let emb_data: Vec<f32> = enc_out.embeddings.iter().copied().collect();
    write_tensor_f32(&cli.out.join("encoder_emb.bin"), &emb_shape, &emb_data)?;
    let mask_data: Vec<i64> = enc_out.mask.iter().copied().collect();
    write_tensor_i64_as_f32(&cli.out.join("encoder_mask.bin"), &mask_shape, &mask_data)?;

    // --- Vocab + prefix ---
    let vocab = Vocab::from_file(&cli.model_dir.join("vocab.txt"))?;
    let mut decode_opts = DecodeOptions::for_asr("es");
    // Disable all gates so we can compare bare model behaviour.
    decode_opts.repetition_penalty = 1.0;
    decode_opts.min_token_to_frame_ratio = 0.0;
    decode_opts.eos_confidence_margin = 0.0;
    let prefix_i64 = build_decoder_prefix(&vocab, &decode_opts)?;
    write_tensor_i64_as_f32(&cli.out.join("prefix.bin"), &[prefix_i64.len()], &prefix_i64)?;
    eprintln!("prefix={:?}", prefix_i64);

    // --- Decoder: directly drive a session for tensor capture ---
    let decoder_path = cli.model_dir.join("decoder-model.onnx");
    // Open a raw ort session so we can dump tensors at each step
    // rather than going through CanaryDecoder::decode (which already
    // drives the full loop and doesn't expose intermediate state).
    let mut session = Session::builder()?.commit_from_file(&decoder_path)?;
    let mems_input = session
        .inputs()
        .iter()
        .find(|i| i.name() == DECODER_INPUT_DECODER_MEMS)
        .ok_or("decoder missing decoder_mems")?;
    let mems_shape = mems_input
        .dtype()
        .tensor_shape()
        .ok_or("decoder_mems is not a tensor")?;
    let layers = mems_shape[0] as usize;
    let hidden = mems_shape[3] as usize;
    eprintln!("decoder_mems: L={} H={}", layers, hidden);

    // Step 0: full prefix.
    let input_ids: Array2<i64> = Array2::from_shape_vec((1, prefix_i64.len()), prefix_i64.clone())?;
    let decoder_mems: Array4<f32> = Array4::zeros((layers, 1, 0, hidden));

    let (logits0, hidden0): (Array3<f32>, Array4<f32>) = {
        let outputs = session.run(vec![
            (
                DECODER_INPUT_IDS,
                Value::from_array(input_ids)?.into_dyn(),
            ),
            (
                DECODER_INPUT_ENCODER_EMBEDDINGS,
                Value::from_array(enc_out.embeddings.clone())?.into_dyn(),
            ),
            (
                DECODER_INPUT_ENCODER_MASK,
                Value::from_array(enc_out.mask.clone())?.into_dyn(),
            ),
            (
                DECODER_INPUT_DECODER_MEMS,
                Value::from_array(decoder_mems)?.into_dyn(),
            ),
        ])?;
        let logits0_idx = outputs
            .keys()
            .position(|k| k == DECODER_OUTPUT_LOGITS)
            .ok_or("missing logits")?;
        let hidden0_idx = outputs
            .keys()
            .position(|k| k == DECODER_OUTPUT_HIDDEN_STATES)
            .ok_or("missing decoder_hidden_states")?;
        let logits0: Array3<f32> = outputs[logits0_idx]
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality()?;
        let hidden0: Array4<f32> = outputs[hidden0_idx]
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality()?;
        (logits0, hidden0)
    };
    let l0_shape = logits0.shape().to_vec();
    let h0_shape = hidden0.shape().to_vec();
    eprintln!("step0 logits.shape={:?} hidden.shape={:?}", l0_shape, h0_shape);
    write_tensor_f32(
        &cli.out.join("step0_logits.bin"),
        &l0_shape,
        &logits0.iter().copied().collect::<Vec<_>>(),
    )?;
    write_tensor_f32(
        &cli.out.join("step0_hidden.bin"),
        &h0_shape,
        &hidden0.iter().copied().collect::<Vec<_>>(),
    )?;
    let next0 = argmax_last_position(&logits0)[0] as i64;
    write_tensor_i64_as_f32(&cli.out.join("next_token_step0.bin"), &[1], &[next0])?;
    eprintln!("step0 next_token={} ({:?})", next0, vocab.piece(next0 as u32));

    // Step 1: just the next token.
    let input_ids1: Array2<i64> = Array2::from_shape_vec((1, 1), vec![next0])?;
    let (logits1, hidden1): (Array3<f32>, Array4<f32>) = {
        let outputs = session.run(vec![
            (
                DECODER_INPUT_IDS,
                Value::from_array(input_ids1)?.into_dyn(),
            ),
            (
                DECODER_INPUT_ENCODER_EMBEDDINGS,
                Value::from_array(enc_out.embeddings.clone())?.into_dyn(),
            ),
            (
                DECODER_INPUT_ENCODER_MASK,
                Value::from_array(enc_out.mask.clone())?.into_dyn(),
            ),
            (
                DECODER_INPUT_DECODER_MEMS,
                Value::from_array(hidden0.clone())?.into_dyn(),
            ),
        ])?;
        let logits1_idx = outputs
            .keys()
            .position(|k| k == DECODER_OUTPUT_LOGITS)
            .ok_or("missing logits step1")?;
        let hidden1_idx = outputs
            .keys()
            .position(|k| k == DECODER_OUTPUT_HIDDEN_STATES)
            .ok_or("missing decoder_hidden_states step1")?;
        let logits1: Array3<f32> = outputs[logits1_idx]
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality()?;
        let hidden1: Array4<f32> = outputs[hidden1_idx]
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality()?;
        (logits1, hidden1)
    };
    let l1_shape = logits1.shape().to_vec();
    let h1_shape = hidden1.shape().to_vec();
    eprintln!("step1 logits.shape={:?} hidden.shape={:?}", l1_shape, h1_shape);
    write_tensor_f32(
        &cli.out.join("step1_logits.bin"),
        &l1_shape,
        &logits1.iter().copied().collect::<Vec<_>>(),
    )?;
    write_tensor_f32(
        &cli.out.join("step1_hidden.bin"),
        &h1_shape,
        &hidden1.iter().copied().collect::<Vec<_>>(),
    )?;
    let next1 = argmax_last_position(&logits1)[0] as i64;
    write_tensor_i64_as_f32(&cli.out.join("next_token_step1.bin"), &[1], &[next1])?;
    eprintln!("step1 next_token={} ({:?})", next1, vocab.piece(next1 as u32));

    // Stash the fact that CanaryDecoder is unused to silence dead-code
    // warnings on this dev binary.
    let _: Option<CanaryDecoder> = None;

    eprintln!("\ndump complete → {}", cli.out.display());
    Ok(())
}
