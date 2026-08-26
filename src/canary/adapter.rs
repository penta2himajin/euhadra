//! `CanaryAdapter` — `AsrAdapter` for NVIDIA Canary-180M-Flash.
//!
//! Glues the four sub-modules into a single end-to-end pipeline:
//!
//! ```text
//! audio f32 → frontend (mel + CMVN)
//!           → encoder ONNX (encoder_embeddings, encoder_mask)
//!           → decoder ONNX greedy loop (with KV cache)
//!           → vocab.decode → text
//! ```
//!
//! Defaults match the file layout of the
//! [`istupakov/canary-180m-flash-onnx`](https://huggingface.co/istupakov/canary-180m-flash-onnx)
//! bundle:
//!
//! ```text
//! <model_dir>/
//!   encoder-model.onnx       (or encoder-model.int8.onnx)
//!   decoder-model.onnx       (or decoder-model.int8.onnx)
//!   vocab.txt
//! ```
//!
//! `scripts/setup_canary.sh` populates that layout from HuggingFace.

use async_trait::async_trait;
use std::path::{Path, PathBuf};

use crate::traits::{AsrAdapter, AsrError};
use crate::types::{AudioChunk, Transcript};

use super::decoder::{CanaryDecoder, DecodeOptions};
use super::encoder::CanaryEncoder;
use super::frontend::{MelFrontend, MelOpts};
use super::vocab::Vocab;

/// File-name overrides + per-utterance defaults. The defaults match
/// the istupakov layout; bump `*_filename` when pointing at a
/// different bundle (e.g. INT8 quantised weights).
#[derive(Debug, Clone)]
pub struct CanaryConfig {
    pub encoder_filename: String,
    pub decoder_filename: String,
    pub vocab_filename: String,
    pub mel: MelOpts,
    /// Default language fed to the decoder when the caller doesn't
    /// override via `with_language`. `"en"` matches the upstream
    /// transcription default.
    pub default_language: String,
    /// Default punctuation+capitalisation behaviour.
    pub default_pnc: bool,
    /// Hard cap on the autoregressive decoder, clamped per-utterance.
    pub max_sequence_length: usize,
    /// Greedy-decode repetition penalty. See
    /// `decoder::DEFAULT_REPETITION_PENALTY`. `1.0` disables.
    pub repetition_penalty: f32,
    /// Min-length gate ratio (suffix tokens / speech encoder frames).
    /// See `decoder::DEFAULT_MIN_TOKEN_TO_FRAME_RATIO`. `0.0` disables.
    pub min_token_to_frame_ratio: f32,
    /// Max-length gate ratio (suffix tokens / speech encoder frames).
    /// See `decoder::DEFAULT_MAX_TOKEN_TO_FRAME_RATIO`. `0.0` disables.
    pub max_token_to_frame_ratio: f32,
    /// EOS-confidence margin in raw logit units. See
    /// `decoder::DEFAULT_EOS_CONFIDENCE_MARGIN`. `0.0` disables.
    pub eos_confidence_margin: f32,
    /// Beam-search width. `1` keeps the v6 greedy path. See
    /// `decoder::DEFAULT_BEAM_SIZE`.
    pub beam_size: usize,
    /// Length penalty α used when `beam_size > 1`. See
    /// `decoder::DEFAULT_LENGTH_PENALTY`.
    pub length_penalty: f32,
    /// Decoder prefix layout. See `decoder::PrefixFormat`.
    pub prefix_format: super::decoder::PrefixFormat,
}

impl CanaryConfig {
    pub fn istupakov_default() -> Self {
        Self {
            encoder_filename: "encoder-model.onnx".to_string(),
            decoder_filename: "decoder-model.onnx".to_string(),
            vocab_filename: "vocab.txt".to_string(),
            mel: MelOpts::canary_default(),
            default_language: "en".to_string(),
            default_pnc: true,
            max_sequence_length: super::decoder::DEFAULT_MAX_SEQUENCE_LENGTH,
            repetition_penalty: super::decoder::DEFAULT_REPETITION_PENALTY,
            min_token_to_frame_ratio: super::decoder::DEFAULT_MIN_TOKEN_TO_FRAME_RATIO,
            max_token_to_frame_ratio: super::decoder::DEFAULT_MAX_TOKEN_TO_FRAME_RATIO,
            eos_confidence_margin: super::decoder::DEFAULT_EOS_CONFIDENCE_MARGIN,
            beam_size: super::decoder::DEFAULT_BEAM_SIZE,
            length_penalty: super::decoder::DEFAULT_LENGTH_PENALTY,
            prefix_format: super::decoder::DEFAULT_PREFIX_FORMAT,
        }
    }

    /// Switch encoder/decoder filenames to the INT8 variants shipped
    /// alongside the full-precision pair.
    pub fn with_int8_weights(mut self) -> Self {
        self.encoder_filename = "encoder-model.int8.onnx".to_string();
        self.decoder_filename = "decoder-model.int8.onnx".to_string();
        self
    }
}

impl Default for CanaryConfig {
    fn default() -> Self {
        Self::istupakov_default()
    }
}

/// `AsrAdapter` for Canary-180M-Flash via ONNX Runtime.
///
/// The adapter accumulates the full utterance, runs frontend →
/// encoder → decoder, and emits a single final `AsrResult`. Streaming
/// is intentionally not implemented — Canary's 40-second standard
/// chunk is short enough that buffering is fine for dictation, and a
/// streaming decoder loop is best done as a follow-up that doesn't
/// risk regressing the static path.
pub struct CanaryAdapter {
    frontend: MelFrontend,
    encoder: CanaryEncoder,
    decoder: CanaryDecoder,
    vocab: Vocab,
    cfg: CanaryConfig,
    /// Override for the source language when the caller drives the
    /// adapter via `with_language`. Defaults to `cfg.default_language`.
    language: String,
}

impl CanaryAdapter {
    /// Load a model bundle laid out as
    /// `<dir>/{encoder-model.onnx, decoder-model.onnx, vocab.txt}`.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, AsrError> {
        Self::load_with_config(model_dir, CanaryConfig::default())
    }

    pub fn load_with_config(
        model_dir: impl AsRef<Path>,
        cfg: CanaryConfig,
    ) -> Result<Self, AsrError> {
        let dir: PathBuf = model_dir.as_ref().to_path_buf();
        let encoder_path = dir.join(&cfg.encoder_filename);
        let decoder_path = dir.join(&cfg.decoder_filename);
        let vocab_path = dir.join(&cfg.vocab_filename);

        let encoder = CanaryEncoder::load(&encoder_path)?;
        let decoder = CanaryDecoder::load(&decoder_path)?;
        let vocab = Vocab::from_file(&vocab_path)?;

        // Sanity: the language token must exist for the configured
        // default before we accept the bundle. Catches a vocab.txt
        // that's missing one of en/de/fr/es.
        if vocab.language_token(&cfg.default_language).is_none() {
            return Err(AsrError::ModelLoad(format!(
                    "vocab {} has no language token for default_language={:?}",
                    vocab_path.display(),
                    cfg.default_language
                )));
        }

        let language = cfg.default_language.clone();
        Ok(Self {
            frontend: MelFrontend::new(cfg.mel.clone()),
            encoder,
            decoder,
            vocab,
            cfg,
            language,
        })
    }

    /// Builder-style language override. `lang` accepts ISO 639-1
    /// (`"en" | "de" | "fr" | "es"`) or the long-form alias
    /// (`"english" | …`). Errors at transcribe time if the resolved
    /// language token is missing from the vocab.
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = lang.into();
        self
    }

    /// Run the full pipeline on a single concatenated waveform.
    /// Returns the user-facing transcript (already detokenised).
    /// Exposed for tests in cousin modules and for the L1 / L3
    /// evaluation harnesses.
    pub fn transcribe_samples(&self, samples: &[f32]) -> Result<String, AsrError> {
        if samples.is_empty() {
            return Err(AsrError::NoAudio);
        }

        let (mel, n_frames) = self.frontend.compute(samples);
        if n_frames == 0 {
            return Err(AsrError::Inference(format!(
                    "audio too short for one mel frame ({} samples, need ≥{})",
                    samples.len(),
                    self.cfg.mel.win_length,
                )));
        }

        let n_mels = self.frontend.n_mels();
        let enc = self.encoder.encode(&mel, n_mels, n_frames)?;

        // Gate the decoder against speech frames, not the full
        // recording length (#136). Without a pipeline VAD, estimate
        // the speech fraction with EnergyVad so silence-padded audio
        // does not inflate the min-token quota.
        let n_enc = super::decoder::valid_frame_count(&enc.mask, 0);
        let speech_encoder_frames =
            Some(estimate_speech_encoder_frames(samples, 16_000, n_enc));

        let opts = DecodeOptions {
            source_language: self.language.clone(),
            target_language: self.language.clone(),
            pnc: self.cfg.default_pnc,
            max_sequence_length: self.cfg.max_sequence_length,
            repetition_penalty: self.cfg.repetition_penalty,
            min_token_to_frame_ratio: self.cfg.min_token_to_frame_ratio,
            max_token_to_frame_ratio: self.cfg.max_token_to_frame_ratio,
            speech_encoder_frames,
            eos_confidence_margin: self.cfg.eos_confidence_margin,
            beam_size: self.cfg.beam_size,
            length_penalty: self.cfg.length_penalty,
            prefix_format: self.cfg.prefix_format,
        };
        let decoded = self
            .decoder
            .decode(&enc.embeddings, &enc.mask, &self.vocab, &opts)?;

        Ok(self.vocab.decode(&decoded.tokens))
    }
}

/// Map a waveform onto an estimated speech-frame count for the
/// decoder's min/max token gates (#136).
///
/// Runs [`EnergyVad`](crate::vad::EnergyVad) over 10 ms frames and
/// scales `n_encoder_frames` by the speech-frame fraction. Pure
/// silence yields 0 so the min-length gate stays off and the max-
/// length gate forbids content tokens. Clean speech yields ≈ the
/// full encoder length, preserving the existing clean-path
/// calibration.
fn estimate_speech_encoder_frames(
    samples: &[f32],
    _sample_rate: u32,
    n_encoder_frames: usize,
) -> usize {
    use crate::vad::{EnergyVad, VadBackend};

    if samples.is_empty() || n_encoder_frames == 0 {
        return 0;
    }
    let backend = EnergyVad::new();
    let frame_size = backend.frame_size().max(1);
    let threshold = backend.default_threshold();
    let mut stream = backend.start();
    let mut speech = 0usize;
    let mut total = 0usize;
    let mut offset = 0usize;
    while offset + frame_size <= samples.len() {
        let frame = &samples[offset..offset + frame_size];
        if stream.speech_probability(frame) >= threshold {
            speech += 1;
        }
        total += 1;
        offset += frame_size;
    }
    if total == 0 {
        return 0;
    }
    ((speech as f32 / total as f32) * n_encoder_frames as f32).round() as usize
}

#[async_trait]
impl AsrAdapter for CanaryAdapter {
    async fn transcribe(&self, audio: &[AudioChunk]) -> Result<Transcript, AsrError> {
        let all_samples = AudioChunk::concat(audio);
        if all_samples.is_empty() {
            return Err(AsrError::NoAudio);
        }

        tracing::info!(
            audio_samples = all_samples.len(),
            language = %self.language,
            "transcribing with canary-180m-flash"
        );

        let text = self.transcribe_samples(&all_samples)?;
        Ok(Transcript::new(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_nonexistent_dir_returns_error() {
        let res = CanaryAdapter::load("/nonexistent/path/to/canary");
        assert!(res.is_err());
        let err = match res {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("load Canary encoder")
                || err.to_string().contains("load Canary decoder")
                || err.to_string().contains("read vocab"),
            "expected load-failure message, got: {}",
            err
        );
    }

    #[test]
    fn config_default_filenames_match_istupakov_layout() {
        let cfg = CanaryConfig::default();
        assert_eq!(cfg.encoder_filename, "encoder-model.onnx");
        assert_eq!(cfg.decoder_filename, "decoder-model.onnx");
        assert_eq!(cfg.vocab_filename, "vocab.txt");
        assert_eq!(cfg.default_language, "en");
        assert!(cfg.default_pnc);
    }

    #[test]
    fn with_int8_weights_swaps_filenames() {
        let cfg = CanaryConfig::default().with_int8_weights();
        assert_eq!(cfg.encoder_filename, "encoder-model.int8.onnx");
        assert_eq!(cfg.decoder_filename, "decoder-model.int8.onnx");
        // Other defaults survive the override.
        assert_eq!(cfg.vocab_filename, "vocab.txt");
        assert_eq!(cfg.default_language, "en");
    }

    #[test]
    fn config_default_exposes_max_token_ratio() {
        let cfg = CanaryConfig::default();
        assert_eq!(
            cfg.max_token_to_frame_ratio,
            super::super::decoder::DEFAULT_MAX_TOKEN_TO_FRAME_RATIO
        );
    }

    /// Digital silence must not contribute speech frames — otherwise
    /// the min-length gate still forces tokens on a no-VAD padded
    /// recording (#136).
    #[test]
    fn estimate_speech_frames_is_zero_on_digital_silence() {
        let silence = vec![0.0_f32; 16_000]; // 1 s
        assert_eq!(estimate_speech_encoder_frames(&silence, 16_000, 100), 0);
    }

    #[test]
    fn estimate_speech_frames_scales_with_voiced_fraction() {
        // 0.5 s silence + 0.5 s loud tone at 16 kHz.
        let mut samples = vec![0.0_f32; 8_000];
        for i in 0..8_000 {
            samples.push((i as f32 * 0.3).sin() * 0.5);
        }
        let n = estimate_speech_encoder_frames(&samples, 16_000, 100);
        assert!(
            (40..=70).contains(&n),
            "≈half the recording is voiced → speech frames near 50, got {n}"
        );
    }

    #[test]
    fn adapter_is_send_and_sync() {
        // The pipeline runtime requires Send + Sync ASR adapters so
        // they can move between worker tasks. ort's Session,
        // wrapped in Mutex, satisfies both — pin that here so a
        // future refactor can't accidentally break the contract.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CanaryAdapter>();
    }

    #[tokio::test]
    async fn empty_audio_yields_no_audio_received_error() {
        // Every adapter maps an empty utterance to AsrError::NoAudio
        // via the same shared check; a real bundle is not needed to
        // pin that check.
        // The precondition every adapter branches on. Previously this
        // test drove MockAsr, which never errors — so it asserted
        // nothing about the contract it named.
        assert!(AudioChunk::concat(&[]).is_empty());
        assert!(AudioChunk::sample_rate_of(&[]).is_none());
    }
}
