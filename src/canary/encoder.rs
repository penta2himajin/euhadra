//! ONNX session wrapper for the Canary encoder.
//!
//! Loads `encoder-model.onnx` from the
//! [`istupakov/canary-180m-flash-onnx`](https://huggingface.co/istupakov/canary-180m-flash-onnx)
//! bundle and exposes a single inference call:
//!
//! ```text
//! [B=1, n_mels=128, T] f32 audio_signal
//! [B=1] i64 length          ──▶  encoder_embeddings  [B=1, T_sub, D] f32
//!                                encoder_mask        [B=1, T_sub]    i64
//! ```
//!
//! Subsampling is handled internally by the ONNX graph — for
//! Canary-180M-Flash the FastConformer encoder produces ~T/8 frames
//! at 50 Hz hop down to ~6.25 Hz output.
//!
//! End-to-end fidelity verification against `onnx-asr` outputs lives
//! in the adapter PR; this module only validates load + I/O marshalling.

use ndarray::{Array1, Array2, Array3};
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::Mutex;

use crate::traits::AsrError;

/// Names of the encoder's I/O tensors. Match the upstream NeMo
/// export and the istupakov bundle exactly. Pinned as constants so a
/// stray re-export with renamed I/Os fails loudly at session-load
/// time rather than silently producing garbage.
pub const ENCODER_INPUT_AUDIO: &str = "audio_signal";
pub const ENCODER_INPUT_LENGTH: &str = "length";
pub const ENCODER_OUTPUT_EMBEDDINGS: &str = "encoder_embeddings";
pub const ENCODER_OUTPUT_MASK: &str = "encoder_mask";

/// Output of a single encoder pass. `embeddings.dim()` is `(1,
/// T_sub, hidden_dim)`; `mask.dim()` is `(1, T_sub)` with `1` for
/// valid frames and `0` for padding (the encoder may right-pad to a
/// power-of-two boundary depending on the export).
#[derive(Debug)]
pub struct EncoderOutput {
    pub embeddings: Array3<f32>,
    pub mask: Array2<i64>,
}

/// Pure helper: pack the row-major `[T, n_mels]` mel buffer
/// returned by `frontend::MelFrontend::compute` into the encoder's
/// expected `[1, n_mels, T]` shape, plus the matching `[1]` length
/// tensor.
///
/// Splitting this from `encode()` keeps the data-marshalling logic
/// testable without needing a real ONNX model checked in.
pub fn pack_mel_for_encoder(
    mel_row_major: &[f32],
    n_mels: usize,
    n_frames: usize,
) -> Result<(Array3<f32>, Array1<i64>), AsrError> {
    if n_mels == 0 || n_frames == 0 {
        return Err(AsrError {
            message: format!("empty mel features (n_mels={n_mels}, n_frames={n_frames})"),
        });
    }
    let expected = n_mels * n_frames;
    if mel_row_major.len() != expected {
        return Err(AsrError {
            message: format!(
                "mel buffer length {} does not match n_mels({}) * n_frames({}) = {}",
                mel_row_major.len(),
                n_mels,
                n_frames,
                expected
            ),
        });
    }

    let mut packed = Array3::<f32>::zeros((1, n_mels, n_frames));
    for t in 0..n_frames {
        for m in 0..n_mels {
            packed[[0, m, t]] = mel_row_major[t * n_mels + m];
        }
    }
    let length: Array1<i64> = Array1::from(vec![n_frames as i64]);
    Ok((packed, length))
}

/// `encoder-model.onnx` session wrapped in a Mutex so the adapter
/// can keep a shared encoder across pipeline sessions while still
/// borrowing it mutably for `run`.
pub struct CanaryEncoder {
    session: Mutex<Session>,
    /// True iff ORT profiling was enabled at load time via
    /// `CANARY_PROFILE_DIR`. When set, `Drop` calls `end_profiling`
    /// so the JSON trace gets flushed to disk before the session
    /// vanishes.
    profiling: bool,
}

impl CanaryEncoder {
    /// Load the encoder ONNX from `path`. The file is expected to
    /// expose I/O names matching `ENCODER_INPUT_*` / `ENCODER_OUTPUT_*`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, AsrError> {
        let path = path.as_ref();
        let builder = Session::builder().map_err(|e| AsrError {
            message: format!("Canary encoder builder {}: {e}", path.display()),
        })?;
        let (mut builder, profiling) = crate::canary::profiling::apply(builder, "encoder")
            .map_err(|e| AsrError {
                message: format!("Canary encoder profiling {}: {e}", path.display()),
            })?;
        let session = builder.commit_from_file(path).map_err(|e| AsrError {
            message: format!("load Canary encoder {}: {e}", path.display()),
        })?;
        validate_encoder_io(&session, path)?;
        Ok(Self {
            session: Mutex::new(session),
            profiling,
        })
    }

    /// Run the encoder on `(mel, n_frames)` produced by
    /// `frontend::MelFrontend::compute`. `n_mels` is 128 for
    /// Canary-180M-Flash (config.json features_size=128) but is
    /// taken as a parameter so a non-default preprocessor doesn't
    /// quietly produce wrong-shaped tensors.
    pub fn encode(
        &self,
        mel_row_major: &[f32],
        n_mels: usize,
        n_frames: usize,
    ) -> Result<EncoderOutput, AsrError> {
        let (audio_signal, length) = pack_mel_for_encoder(mel_row_major, n_mels, n_frames)?;

        let audio_value = Value::from_array(audio_signal).map_err(|e| AsrError {
            message: format!("audio_signal Value: {e}"),
        })?;
        let length_value = Value::from_array(length).map_err(|e| AsrError {
            message: format!("length Value: {e}"),
        })?;

        let mut session = self.session.lock().map_err(|e| AsrError {
            message: format!("encoder session lock poisoned: {e}"),
        })?;
        let outputs = session
            .run(vec![
                (ENCODER_INPUT_AUDIO, audio_value.into_dyn()),
                (ENCODER_INPUT_LENGTH, length_value.into_dyn()),
            ])
            .map_err(|e| AsrError {
                message: format!("Canary encoder run: {e}"),
            })?;

        // The order ort gives back outputs in matches the input order
        // declared in the ONNX graph. Look up by name to be robust to
        // re-exports that swap the two.
        let emb_idx = output_index(&outputs, ENCODER_OUTPUT_EMBEDDINGS).ok_or_else(|| AsrError {
            message: format!("encoder missing output {ENCODER_OUTPUT_EMBEDDINGS}"),
        })?;
        let mask_idx = output_index(&outputs, ENCODER_OUTPUT_MASK).ok_or_else(|| AsrError {
            message: format!("encoder missing output {ENCODER_OUTPUT_MASK}"),
        })?;

        let embeddings = outputs[emb_idx]
            .try_extract_array::<f32>()
            .map_err(|e| AsrError {
                message: format!("extract {ENCODER_OUTPUT_EMBEDDINGS}: {e}"),
            })?
            .to_owned()
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| AsrError {
                message: format!("{ENCODER_OUTPUT_EMBEDDINGS} rank: {e}"),
            })?;
        let mask = outputs[mask_idx]
            .try_extract_array::<i64>()
            .map_err(|e| AsrError {
                message: format!("extract {ENCODER_OUTPUT_MASK}: {e}"),
            })?
            .to_owned()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| AsrError {
                message: format!("{ENCODER_OUTPUT_MASK} rank: {e}"),
            })?;

        Ok(EncoderOutput { embeddings, mask })
    }
}

impl Drop for CanaryEncoder {
    fn drop(&mut self) {
        if self.profiling {
            crate::canary::profiling::flush(&self.session, "encoder");
        }
    }
}

fn validate_encoder_io(session: &Session, path: &Path) -> Result<(), AsrError> {
    let input_names: Vec<String> = session.inputs().iter().map(|i| i.name().to_string()).collect();
    if !input_names.iter().any(|n| n == ENCODER_INPUT_AUDIO) {
        return Err(AsrError {
            message: format!(
                "encoder {} missing input {ENCODER_INPUT_AUDIO} (have: {input_names:?})",
                path.display()
            ),
        });
    }
    if !input_names.iter().any(|n| n == ENCODER_INPUT_LENGTH) {
        return Err(AsrError {
            message: format!(
                "encoder {} missing input {ENCODER_INPUT_LENGTH} (have: {input_names:?})",
                path.display()
            ),
        });
    }
    let output_names: Vec<String> = session.outputs().iter().map(|o| o.name().to_string()).collect();
    if !output_names.iter().any(|n| n == ENCODER_OUTPUT_EMBEDDINGS) {
        return Err(AsrError {
            message: format!(
                "encoder {} missing output {ENCODER_OUTPUT_EMBEDDINGS} (have: {output_names:?})",
                path.display()
            ),
        });
    }
    if !output_names.iter().any(|n| n == ENCODER_OUTPUT_MASK) {
        return Err(AsrError {
            message: format!(
                "encoder {} missing output {ENCODER_OUTPUT_MASK} (have: {output_names:?})",
                path.display()
            ),
        });
    }
    Ok(())
}

fn output_index(outputs: &ort::session::SessionOutputs<'_>, name: &str) -> Option<usize> {
    outputs.keys().position(|k| k == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_mel_transposes_to_n_mels_t_layout() {
        // 4-frame, 3-mel input. Row-major layout: frame 0 mels [a,b,c],
        // frame 1 mels [d,e,f], etc.
        let mel = vec![
            1.0, 2.0, 3.0, // frame 0
            4.0, 5.0, 6.0, // frame 1
            7.0, 8.0, 9.0, // frame 2
            10.0, 11.0, 12.0, // frame 3
        ];
        let (packed, length) = pack_mel_for_encoder(&mel, 3, 4).unwrap();

        assert_eq!(packed.dim(), (1, 3, 4));
        // After transpose, [0, m, t] holds frame t's mel m.
        assert_eq!(packed[[0, 0, 0]], 1.0);
        assert_eq!(packed[[0, 1, 0]], 2.0);
        assert_eq!(packed[[0, 2, 0]], 3.0);
        assert_eq!(packed[[0, 0, 3]], 10.0);
        assert_eq!(packed[[0, 1, 3]], 11.0);
        assert_eq!(packed[[0, 2, 3]], 12.0);
        assert_eq!(packed[[0, 1, 2]], 8.0);

        assert_eq!(length.shape(), &[1]);
        assert_eq!(length[0], 4);
    }

    #[test]
    fn pack_mel_rejects_size_mismatch() {
        // 3 mels × 4 frames = 12, but only 11 values supplied.
        let mel = vec![0.0_f32; 11];
        let err = pack_mel_for_encoder(&mel, 3, 4).unwrap_err();
        assert!(err.message.contains("does not match"), "{}", err.message);
    }

    #[test]
    fn pack_mel_rejects_zero_dimensions() {
        assert!(pack_mel_for_encoder(&[1.0], 0, 1).is_err());
        assert!(pack_mel_for_encoder(&[1.0], 1, 0).is_err());
        assert!(pack_mel_for_encoder(&[], 0, 0).is_err());
    }

    #[test]
    fn pack_mel_canary_default_shape() {
        // Canary-180M-Flash's preprocessor produces 128 mels per
        // frame (config.json features_size=128). Pack a 1-second
        // utterance (~98 frames after framing) and verify the
        // encoder-input shape matches `audio_signal`.
        let n_frames = 98;
        let n_mels = 128;
        let mel = vec![0.0_f32; n_frames * n_mels];
        let (packed, length) = pack_mel_for_encoder(&mel, n_mels, n_frames).unwrap();
        assert_eq!(packed.dim(), (1, 128, 98));
        assert_eq!(length[0], 98);
    }

    #[test]
    fn load_nonexistent_model_returns_error() {
        // Don't use `unwrap_err` (would require Debug on the Ok variant).
        match CanaryEncoder::load("/nonexistent/path/to/encoder.onnx") {
            Ok(_) => panic!("expected error, got Ok"),
            Err(e) => assert!(
                e.message.contains("load Canary encoder"),
                "{}",
                e.message
            ),
        }
    }

    #[test]
    fn io_name_constants_match_onnx_asr_conventions() {
        // Pin the public I/O names so a stray edit can't desync this
        // crate from the upstream onnx-asr / NeMo Canary export.
        assert_eq!(ENCODER_INPUT_AUDIO, "audio_signal");
        assert_eq!(ENCODER_INPUT_LENGTH, "length");
        assert_eq!(ENCODER_OUTPUT_EMBEDDINGS, "encoder_embeddings");
        assert_eq!(ENCODER_OUTPUT_MASK, "encoder_mask");
    }
}
