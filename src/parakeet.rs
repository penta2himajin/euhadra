//! Parakeet TDT ASR adapter — powered by parakeet-rs.
//!
//! Wraps the `parakeet-rs` crate for ONNX inference of NVIDIA Parakeet TDT models.
//! Handles mel spectrogram, FastConformer encoding, and TDT greedy decoding internally.
//!
//! Requires `--features onnx` and a model directory containing:
//! encoder-model.onnx, decoder_joint-model.onnx, vocab.txt
//!
//! Supports both English (parakeet-tdt-0.6b-v2) and multilingual models
//! including Japanese (parakeet-tdt-0.6b-v3).

use async_trait::async_trait;
use parakeet_rs::{ParakeetTDT, Transcriber};
use std::path::Path;
use std::sync::Mutex;
use tokio::sync::mpsc;

use crate::traits::{AsrAdapter, AsrError};
use crate::types::{AsrResult, AudioChunk};

/// Parakeet TDT ASR adapter using parakeet-rs.
///
/// Runs entirely in Rust via ONNX Runtime. No Python required.
///
/// # Usage
/// ```no_run
/// use euhadra::parakeet::ParakeetAdapter;
///
/// let asr = ParakeetAdapter::load("/path/to/parakeet-tdt-0.6b-v3")
///     .expect("failed to load model");
/// ```
pub struct ParakeetAdapter {
    model: Mutex<ParakeetTDT>,
}

impl ParakeetAdapter {
    /// Load with the default 128-mel preprocessor — matches
    /// `parakeet-tdt-0.6b-v2` and the multilingual European
    /// `parakeet-tdt-0.6b-v3`.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, AsrError> {
        let model =
            ParakeetTDT::from_pretrained(model_dir.as_ref(), None).map_err(|e| AsrError {
                message: format!("failed to load ParakeetTDT model: {e}"),
            })?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    /// Load with an explicit mel-feature size. Use this for variants
    /// trained with a non-default preprocessor:
    ///
    /// - `nvidia/parakeet-tdt_ctc-0.6b-ja` (Japanese, Hybrid TDT-CTC) → **80**
    /// - `parakeet-tdt-0.6b-v2` / `parakeet-tdt-0.6b-v3` → **128** (same as `load`)
    ///
    /// Underlying support requires the fork at
    /// `penta2himajin/parakeet-rs@feature-size-injection`; see Cargo.toml.
    pub fn load_with_feature_size(
        model_dir: impl AsRef<Path>,
        feature_size: usize,
    ) -> Result<Self, AsrError> {
        let model =
            ParakeetTDT::from_pretrained_with_feature_size(model_dir.as_ref(), None, feature_size)
                .map_err(|e| AsrError {
                    message: format!(
                        "failed to load ParakeetTDT model (feature_size={feature_size}): {e}"
                    ),
                })?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }
}

#[async_trait]
impl AsrAdapter for ParakeetAdapter {
    async fn transcribe(
        &self,
        mut audio_rx: mpsc::Receiver<AudioChunk>,
        result_tx: mpsc::Sender<AsrResult>,
    ) -> Result<(), AsrError> {
        // Accumulate all audio chunks
        let mut all_samples: Vec<f32> = Vec::new();
        while let Some(chunk) = audio_rx.recv().await {
            all_samples.extend(&chunk.samples);
        }

        if all_samples.is_empty() {
            return Err(AsrError {
                message: "no audio received".into(),
            });
        }

        let n_samples = all_samples.len();
        tracing::info!(audio_samples = n_samples, "transcribing with parakeet-rs");

        // Run transcription (CPU-bound)
        let result = {
            let mut model = self.model.lock().unwrap();
            model
                .transcribe_samples(all_samples, 16000, 1, None)
                .map_err(|e| AsrError {
                    message: format!("transcription failed: {e}"),
                })?
        };

        let text = result.text.trim().to_string();
        if !text.is_empty() {
            result_tx
                .send(AsrResult {
                    text,
                    is_final: true,
                    confidence: 1.0,
                    timestamp: std::time::Duration::ZERO,
                })
                .await
                .map_err(|e| AsrError {
                    message: format!("send: {e}"),
                })?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[test]
    fn load_nonexistent_model_returns_error() {
        let result = ParakeetAdapter::load("/nonexistent/path/to/model");
        assert!(result.is_err(), "loading from nonexistent path should fail");
        let err = result.err().unwrap();
        assert!(
            err.message.contains("failed to load"),
            "error message should indicate load failure: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn transcribe_empty_audio_sends_empty_text() {
        // Verify the AsrAdapter contract: when audio channel closes immediately
        // with no chunks, the adapter drains and emits its result.
        let (tx, rx) = mpsc::channel::<AudioChunk>(1);
        let (result_tx, mut result_rx) = mpsc::channel::<AsrResult>(1);

        // Drop sender immediately → receiver gets None → no audio
        drop(tx);

        use crate::mock::MockAsr;
        use crate::traits::AsrAdapter;

        let mock = MockAsr::new("");
        let _ = mock.transcribe(rx, result_tx).await;
        // MockAsr with empty string still sends an AsrResult with empty text.
        // The real ParakeetAdapter would return an error for empty audio.
        let result = result_rx.try_recv();
        assert!(
            result.is_ok(),
            "MockAsr should send a result even with empty transcript"
        );
        assert!(result.unwrap().text.is_empty());
    }

    #[test]
    fn adapter_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ParakeetAdapter>();
    }
}
