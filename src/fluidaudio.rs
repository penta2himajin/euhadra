//! FluidAudio ASR adapter: Parakeet on CoreML and the Apple Neural Engine.
//!
//! macOS only, behind the `fluidaudio` feature. Recognition runs in
//! FluidAudio (Swift) through `fluidaudio-rs`; everything around it — the
//! voice activity detector, Tier 1/2 text processing, the evaluation gates
//! — stays in euhadra, so swapping this in for the ONNX adapters changes
//! only where the model runs (#166).
//!
//! Measured on FLEURS-ja test (321 utterances, `cer_lenient`, Apple
//! Silicon), against the INT8 ONNX Parakeet-ja this replaces on Apple:
//!
//! | | CER, ≤ 15 s | CER, > 15 s | RTF | peak RSS |
//! |---|---|---|---|---|
//! | this adapter | 0.0523 | 0.1205 | 0.0095 | 102 MB |
//! | ONNX INT8 (`ParakeetAdapter`) | 0.0492 | 0.0688 | 0.029 | 1 080 MB |
//!
//! The > 15 s column is why [`max_input_duration`](AsrAdapter::max_input_duration)
//! is declared: FluidAudio splits longer Japanese input itself and drops
//! text at the seams (FluidAudio#1031), so the pipeline packs utterances
//! into ≤ 15 s calls instead. ANE-resident memory lives in the `aned`
//! process and is not in the RSS figure.
//!
//! Binaries that link this need `/usr/lib/swift` on their rpath for the
//! Swift concurrency runtime; euhadra's `build.rs` adds it for its own
//! tests and examples.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use fluidaudio_rs::{AsrModelVersion, FluidAudio};

use crate::traits::{AdapterLoadState, AsrAdapter, AsrError, AsrLifecycle};
use crate::types::{AudioChunk, Transcript};

/// FluidAudio's Parakeet input rate.
const SAMPLE_RATE: u32 = 16_000;

/// FluidAudio refuses shorter input with an error; below this there is no
/// utterance to recognise, so it is answered with an empty transcript,
/// which the pipeline reports as no speech.
const MIN_INPUT: Duration = Duration::from_millis(300);

/// One window of FluidAudio's Parakeet. Past it FluidAudio stitches
/// windows itself, which drops Japanese text at the seams (#1031).
const MAX_INPUT: Duration = Duration::from_secs(15);

/// Where the CoreML models come from.
#[derive(Debug, Clone)]
enum ModelSource {
    /// Downloaded on first use into `<dir>/<repo folder>` (FluidAudio's
    /// default cache when `None`), reused afterwards.
    Cache(Option<PathBuf>),
    /// Compiled models already in exactly this directory; never touches
    /// the network.
    Local(PathBuf),
}

/// Parakeet TDT on CoreML via FluidAudio.
pub struct FluidAudioAdapter {
    /// `None` once [`unload`](AsrLifecycle::unload)ed. The mutex also
    /// serialises calls: one model instance decodes one utterance at a
    /// time.
    engine: Arc<Mutex<Option<FluidAudio>>>,
    version: AsrModelVersion,
    source: ModelSource,
}

impl FluidAudioAdapter {
    /// Japanese Parakeet TDT-CTC 0.6B (`FluidInference/parakeet-0.6b-ja-coreml`,
    /// CC-BY-4.0, ~600 MB), downloaded into `<cache_dir>/parakeet-ja` on
    /// first use. The first load also compiles the models for the Neural
    /// Engine, which takes about a minute.
    pub fn japanese(cache_dir: Option<PathBuf>) -> Result<Self, AsrError> {
        Self::open(AsrModelVersion::TdtJa, ModelSource::Cache(cache_dir))
    }

    /// Japanese Parakeet from compiled models already in `dir`, with no
    /// network access — for a host that ships or fetches the models
    /// itself.
    pub fn japanese_local(dir: impl AsRef<Path>) -> Result<Self, AsrError> {
        Self::open(
            AsrModelVersion::TdtJa,
            ModelSource::Local(dir.as_ref().to_path_buf()),
        )
    }

    fn open(version: AsrModelVersion, source: ModelSource) -> Result<Self, AsrError> {
        let engine = Self::load(version, &source)?;
        Ok(Self {
            engine: Arc::new(Mutex::new(Some(engine))),
            version,
            source,
        })
    }

    fn load(version: AsrModelVersion, source: &ModelSource) -> Result<FluidAudio, AsrError> {
        let engine = FluidAudio::new().map_err(|e| AsrError::ModelLoad(e.to_string()))?;
        match source {
            ModelSource::Cache(dir) => engine.init_asr_with(version, dir.as_deref()),
            ModelSource::Local(dir) => engine.init_asr_local(version, dir),
        }
        .map_err(|e| AsrError::ModelLoad(e.to_string()))?;
        Ok(engine)
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Option<FluidAudio>>, AsrError> {
        self.engine
            .lock()
            .map_err(|e| AsrError::Inference(format!("FluidAudio lock poisoned: {e}")))
    }
}

#[async_trait]
impl AsrAdapter for FluidAudioAdapter {
    async fn transcribe(&self, audio: &[AudioChunk]) -> Result<Transcript, AsrError> {
        if let Some(chunk) = audio
            .iter()
            .find(|c| c.sample_rate != SAMPLE_RATE || c.channels != 1)
        {
            return Err(AsrError::Config(format!(
                "FluidAudio takes {SAMPLE_RATE} Hz mono, got {} Hz × {} channels",
                chunk.sample_rate, chunk.channels
            )));
        }
        let samples = AudioChunk::concat(audio);
        if samples.is_empty() {
            return Err(AsrError::NoAudio);
        }
        if (samples.len() as f64) < MIN_INPUT.as_secs_f64() * SAMPLE_RATE as f64 {
            return Ok(Transcript::new(String::new()));
        }

        // FluidAudio blocks on its own async runtime, so keep it off the
        // async executor. There is no cancel hook in FluidAudio: if the
        // pipeline cancels, the in-flight utterance finishes and its
        // result is discarded.
        let engine = Arc::clone(&self.engine);
        let result = tokio::task::spawn_blocking(move || {
            let guard = engine
                .lock()
                .map_err(|e| AsrError::Inference(format!("FluidAudio lock poisoned: {e}")))?;
            let engine = guard.as_ref().ok_or(AsrError::NotLoaded)?;
            engine
                .transcribe_samples(&samples)
                .map_err(|e| AsrError::Inference(e.to_string()))
        })
        .await
        .map_err(|e| AsrError::Inference(format!("FluidAudio task failed: {e}")))??;

        Ok(Transcript {
            text: result.text,
            confidence: result.confidence.clamp(0.0, 1.0),
            duration: Some(Duration::from_secs_f64(result.duration.max(0.0))),
        })
    }

    fn max_input_duration(&self) -> Option<Duration> {
        Some(MAX_INPUT)
    }
}

impl AsrLifecycle for FluidAudioAdapter {
    fn load_state(&self) -> AdapterLoadState {
        match self.engine.lock() {
            Ok(guard) if guard.is_some() => AdapterLoadState::Loaded,
            _ => AdapterLoadState::Unloaded,
        }
    }

    fn unload(&self) -> Result<(), AsrError> {
        if let Some(engine) = self.lock()?.take() {
            engine.cleanup();
        }
        Ok(())
    }

    fn reload(&self) -> Result<(), AsrError> {
        let mut guard = self.lock()?;
        if guard.is_none() {
            *guard = Some(Self::load(self.version, &self.source)?);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An adapter with nothing loaded, so the input checks can be tested
    /// without downloading 600 MB of models.
    fn unloaded() -> FluidAudioAdapter {
        FluidAudioAdapter {
            engine: Arc::new(Mutex::new(None)),
            version: AsrModelVersion::TdtJa,
            source: ModelSource::Cache(None),
        }
    }

    fn chunk(seconds: f32, sample_rate: u32, channels: u16) -> AudioChunk {
        AudioChunk {
            samples: vec![0.0; (seconds * sample_rate as f32) as usize * channels as usize],
            sample_rate,
            channels,
        }
    }

    #[tokio::test]
    async fn declares_the_fifteen_second_window() {
        assert_eq!(unloaded().max_input_duration(), Some(MAX_INPUT));
    }

    #[tokio::test]
    async fn rejects_audio_that_is_not_16_khz_mono() {
        let adapter = unloaded();
        for bad in [chunk(1.0, 44_100, 1), chunk(1.0, SAMPLE_RATE, 2)] {
            let err = adapter.transcribe(&[bad]).await.unwrap_err();
            assert!(matches!(err, AsrError::Config(_)), "got {err:?}");
        }
    }

    #[tokio::test]
    async fn empty_audio_is_no_audio() {
        let err = unloaded().transcribe(&[]).await.unwrap_err();
        assert!(matches!(err, AsrError::NoAudio), "got {err:?}");
    }

    /// FluidAudio would throw; an empty transcript is what the pipeline
    /// turns into `NoSpeech`.
    #[tokio::test]
    async fn input_shorter_than_fluidaudio_accepts_is_an_empty_transcript() {
        let transcript = unloaded()
            .transcribe(&[chunk(0.2, SAMPLE_RATE, 1)])
            .await
            .unwrap();
        assert!(transcript.text.is_empty());
    }

    #[tokio::test]
    async fn an_unloaded_adapter_says_so() {
        let adapter = unloaded();
        assert_eq!(adapter.load_state(), AdapterLoadState::Unloaded);
        let err = adapter
            .transcribe(&[chunk(1.0, SAMPLE_RATE, 1)])
            .await
            .unwrap_err();
        assert!(matches!(err, AsrError::NotLoaded), "got {err:?}");
        adapter.unload().expect("unloading twice is fine");
    }
}
