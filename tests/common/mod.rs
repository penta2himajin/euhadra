//! Shared helpers for integration tests.
//!
//! Each integration test file is its own crate, so individual helpers may
//! appear unused from the perspective of any single test binary. The blanket
//! `dead_code` allow keeps Cargo from warning when only a subset is consumed.

#![allow(dead_code)]

use async_trait::async_trait;
use euhadra::prelude::*;
use std::time::Duration;
use tokio::sync::mpsc;

/// A 10-ms chunk of silence at 16 kHz, mono. The pipeline never inspects the
/// samples — they exist only to drive the audio channel through the ASR stage.
pub fn silence_chunk() -> AudioChunk {
    AudioChunk {
        samples: vec![0.0; 160],
        sample_rate: 16000,
        channels: 1,
    }
}

/// Send a single silence chunk into a session.
///
/// Closing the stream is `Session::finish`'s job, so this only sends.
pub async fn send_one(audio: &mpsc::Sender<AudioChunk>) {
    audio.send(silence_chunk()).await.unwrap();
}

// ---------------------------------------------------------------------------
// Custom adapters used by integration tests
// ---------------------------------------------------------------------------

/// Drains the audio channel without ever emitting a recognition result.
/// Used to exercise the pipeline's "no speech detected" error path.
pub struct SilentAsr;

#[async_trait]
impl AsrAdapter for SilentAsr {
    async fn transcribe(&self, _audio: &[AudioChunk]) -> Result<Transcript, AsrError> {
        // Silence: a transcript with no text, which the pipeline turns
        // into PipelineError::NoSpeech.
        Ok(Transcript::default())
    }
}

/// Sleeps for `delay` before returning a passthrough refinement. Used to
/// keep the pipeline in the refinement phase long enough for cancellation to
/// land.
pub struct SlowRefiner {
    pub delay: Duration,
}

#[async_trait]
impl LlmRefiner for SlowRefiner {
    async fn refine(&self, input: RefinementInput) -> Result<RefinementOutput, RefineError> {
        tokio::time::sleep(self.delay).await;
        Ok(RefinementOutput::TextInsertion {
            text: input.raw_text,
            formatting: None,
        })
    }
}

/// Sleeps for `delay` before returning the default snapshot. Used to keep the
/// pipeline in the context-fetch phase long enough for cancellation to land.
pub struct SlowContextProvider {
    pub delay: Duration,
}

#[async_trait]
impl ContextProvider for SlowContextProvider {
    async fn get_context(&self) -> ContextSnapshot {
        tokio::time::sleep(self.delay).await;
        ContextSnapshot::default()
    }
}
