use async_trait::async_trait;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::router::{AdapterRequest, AsrRuntimeFactory, ModelSource, RouterError};
use crate::traits::{AsrAdapter, AsrError};
use crate::types::{AsrResult, AudioChunk};

/// Local ASR adapter backed by whisper.cpp.
///
/// Accumulates audio chunks into a temporary WAV file, then invokes
/// `whisper-cli` to transcribe.  This is the "final-only" strategy:
/// one invocation per session, one final result.
pub struct WhisperLocal {
    /// Path to the whisper-cli binary.
    cli_path: PathBuf,
    /// Path to the GGML model file.
    model_path: PathBuf,
    /// Language hint (e.g. "en", "ja").  None = auto-detect.
    language: Option<String>,
}

impl WhisperLocal {
    pub fn new(cli_path: impl Into<PathBuf>, model_path: impl Into<PathBuf>) -> Self {
        Self {
            cli_path: cli_path.into(),
            model_path: model_path.into(),
            language: None,
        }
    }

    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Transcribe a WAV file and return the raw text.
    async fn transcribe_file(&self, wav_path: &Path) -> Result<String, AsrError> {
        let mut cmd = tokio::process::Command::new(&self.cli_path);
        cmd.arg("-m").arg(&self.model_path);
        cmd.arg("-f").arg(wav_path);
        cmd.arg("--no-timestamps");
        cmd.arg("--no-prints");

        if let Some(ref lang) = self.language {
            cmd.arg("-l").arg(lang);
        }

        let output = cmd.output().await.map_err(|e| AsrError {
            message: format!("failed to run whisper-cli: {e}"),
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(AsrError {
                message: format!("whisper-cli exited with {}: {stderr}", output.status),
            });
        }

        let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(text)
    }
}

#[async_trait]
impl AsrAdapter for WhisperLocal {
    async fn transcribe(
        &self,
        mut audio_rx: mpsc::Receiver<AudioChunk>,
        result_tx: mpsc::Sender<AsrResult>,
    ) -> Result<(), AsrError> {
        // Accumulate all audio chunks
        let mut all_samples: Vec<f32> = Vec::new();
        let mut sample_rate = 16000u32;
        let mut channels = 1u16;

        while let Some(chunk) = audio_rx.recv().await {
            sample_rate = chunk.sample_rate;
            channels = chunk.channels;
            all_samples.extend(&chunk.samples);
        }

        if all_samples.is_empty() {
            return Err(AsrError {
                message: "no audio received".into(),
            });
        }

        // Write to a temporary WAV file
        let tmp_path = std::env::temp_dir().join(format!("euhadra_{}.wav", std::process::id()));
        write_wav(&tmp_path, &all_samples, sample_rate, channels).map_err(|e| AsrError {
            message: format!("failed to write WAV: {e}"),
        })?;

        // Transcribe
        let text = self.transcribe_file(&tmp_path).await?;

        // Clean up
        let _ = std::fs::remove_file(&tmp_path);

        if !text.is_empty() {
            let result = AsrResult {
                text,
                is_final: true,
                confidence: 1.0,
                timestamp: std::time::Duration::ZERO,
            };
            result_tx.send(result).await.map_err(|e| AsrError {
                message: format!("failed to send result: {e}"),
            })?;
        }

        Ok(())
    }
}

/// Transcribe a WAV file directly (convenience method, no channel needed).
pub async fn transcribe_file(
    cli_path: impl Into<PathBuf>,
    model_path: impl Into<PathBuf>,
    wav_path: &Path,
    language: Option<&str>,
) -> Result<String, AsrError> {
    let mut adapter = WhisperLocal::new(cli_path, model_path);
    if let Some(lang) = language {
        adapter = adapter.with_language(lang);
    }
    adapter.transcribe_file(wav_path).await
}

// ---------------------------------------------------------------------------
// Minimal WAV writer (16-bit PCM)
// ---------------------------------------------------------------------------

fn write_wav(path: &Path, samples: &[f32], sample_rate: u32, channels: u16) -> std::io::Result<()> {
    use std::io::Write;

    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * channels as u32 * bits_per_sample as u32 / 8;
    let block_align = channels * bits_per_sample / 8;
    let data_size = samples.len() as u32 * 2; // 16-bit = 2 bytes per sample

    let mut f = std::fs::File::create(path)?;

    // RIFF header
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_size).to_le_bytes())?;
    f.write_all(b"WAVE")?;

    // fmt chunk
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?; // chunk size
    f.write_all(&1u16.to_le_bytes())?; // PCM format
    f.write_all(&channels.to_le_bytes())?;
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&byte_rate.to_le_bytes())?;
    f.write_all(&block_align.to_le_bytes())?;
    f.write_all(&bits_per_sample.to_le_bytes())?;

    // data chunk
    f.write_all(b"data")?;
    f.write_all(&data_size.to_le_bytes())?;

    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let i16_val = (clamped * 32767.0) as i16;
        f.write_all(&i16_val.to_le_bytes())?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Minimal WAV reader (for loading test files)
// ---------------------------------------------------------------------------

/// Read a 16-bit PCM WAV file into f32 samples.
pub fn read_wav(path: &Path) -> Result<AudioChunk, String> {
    use std::io::Read;

    let mut f = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).map_err(|e| format!("read: {e}"))?;

    if buf.len() < 12 || &buf[0..4] != b"RIFF" || &buf[8..12] != b"WAVE" {
        return Err("not a valid WAV file".into());
    }

    let mut channels: u16 = 1;
    let mut sample_rate: u32 = 16000;
    let mut bits_per_sample: u16 = 16;
    let mut data_samples: Option<Vec<f32>> = None;

    // Walk chunks starting after "RIFF" + size + "WAVE" (byte 12)
    let mut pos = 12;
    while pos + 8 <= buf.len() {
        let chunk_id = &buf[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([buf[pos + 4], buf[pos + 5], buf[pos + 6], buf[pos + 7]]) as usize;
        let chunk_data_start = pos + 8;
        let chunk_data_end = (chunk_data_start + chunk_size).min(buf.len());

        if chunk_id == b"fmt " && chunk_size >= 16 {
            let d = &buf[chunk_data_start..chunk_data_end];
            // audio_format = u16 at offset 0 (1 = PCM)
            channels = u16::from_le_bytes([d[2], d[3]]);
            sample_rate = u32::from_le_bytes([d[4], d[5], d[6], d[7]]);
            // byte_rate at 8..12, block_align at 12..14
            bits_per_sample = u16::from_le_bytes([d[14], d[15]]);
        } else if chunk_id == b"data" {
            let data = &buf[chunk_data_start..chunk_data_end];
            let samples: Vec<f32> = match bits_per_sample {
                16 => data
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                    .collect(),
                _ => return Err(format!("unsupported bits_per_sample: {bits_per_sample}")),
            };
            data_samples = Some(samples);
        }

        // Advance to next chunk (chunks are word-aligned)
        pos = chunk_data_start + ((chunk_size + 1) & !1);
    }

    let samples = data_samples.ok_or("no data chunk found")?;

    Ok(AudioChunk {
        samples,
        sample_rate,
        channels,
    })
}

// ---------------------------------------------------------------------------
// Router factory
// ---------------------------------------------------------------------------

/// Options accepted by `WhisperLocalFactory` via `AdapterRequest.options`.
///
/// The model path comes from `AdapterRequest.model_source` (the per-language
/// artifact, e.g. `ggml-base.bin`). `cli_path` is the deployment-specific
/// `whisper-cli` binary location, which Menura supplies once per install.
#[derive(Debug, Deserialize)]
struct WhisperLocalOptions {
    cli_path: PathBuf,
}

/// Router factory that builds `WhisperLocal` adapters via `AsrRouter`.
///
/// Registered under the runtime id `"whisper-local"`. Menura's
/// `asr_models.toml` should set `runtime = "whisper-local"` for any
/// language that should be served by whisper.cpp.
pub struct WhisperLocalFactory;

impl WhisperLocalFactory {
    pub const ID: &'static str = "whisper-local";
}

#[async_trait]
impl AsrRuntimeFactory for WhisperLocalFactory {
    fn id(&self) -> &'static str {
        Self::ID
    }

    async fn instantiate(&self, req: &AdapterRequest) -> Result<Arc<dyn AsrAdapter>, RouterError> {
        let ModelSource::LocalPath(model_path) = &req.model_source;

        let opts: WhisperLocalOptions =
            serde_json::from_value(req.options.clone()).map_err(|e| {
                RouterError::InvalidRequest(format!(
                    "whisper-local options must include `cli_path`: {e}"
                ))
            })?;

        let mut adapter = WhisperLocal::new(opts.cli_path, model_path.clone());
        if !req.language.is_empty() {
            adapter = adapter.with_language(req.language.clone());
        }
        Ok(Arc::new(adapter))
    }
}

#[cfg(test)]
mod factory_tests {
    use super::*;
    use crate::router::{AdapterRequest, AsrRouter, ModelSource, RouterError};
    use serde_json::json;
    use std::path::PathBuf;

    fn req(language: &str, options: serde_json::Value) -> AdapterRequest {
        AdapterRequest {
            language: language.into(),
            runtime: WhisperLocalFactory::ID.into(),
            model_source: ModelSource::LocalPath(PathBuf::from("/tmp/ggml-base.bin")),
            options,
        }
    }

    #[tokio::test]
    async fn factory_id_matches_constant() {
        let f = WhisperLocalFactory;
        assert_eq!(f.id(), "whisper-local");
    }

    #[tokio::test]
    async fn dispatch_with_cli_path_returns_adapter() {
        let router = AsrRouter::new().register(WhisperLocalFactory);
        let result = router
            .dispatch(req("en", json!({ "cli_path": "/usr/bin/whisper-cli" })))
            .await;
        assert!(result.is_ok(), "expected Ok, got error");
    }

    #[tokio::test]
    async fn dispatch_without_cli_path_returns_invalid_request() {
        let router = AsrRouter::new().register(WhisperLocalFactory);
        match router.dispatch(req("en", json!({}))).await {
            Err(RouterError::InvalidRequest(msg)) => {
                assert!(
                    msg.contains("cli_path"),
                    "error message should mention cli_path, got: {msg}"
                );
            }
            Err(other) => panic!("expected InvalidRequest, got {other:?}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[tokio::test]
    async fn empty_language_does_not_pass_language_hint() {
        // We can't inspect WhisperLocal internals, but we can at least confirm
        // that instantiation succeeds with an empty language field.
        let router = AsrRouter::new().register(WhisperLocalFactory);
        let result = router
            .dispatch(req("", json!({ "cli_path": "/usr/bin/whisper-cli" })))
            .await;
        assert!(result.is_ok());
    }
}
