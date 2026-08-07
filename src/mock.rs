use async_trait::async_trait;

use crate::traits::*;
use crate::types::*;

// ---------------------------------------------------------------------------
// MockAsr — returns a fixed transcript
// ---------------------------------------------------------------------------

pub struct MockAsr {
    pub transcript: String,
}

impl MockAsr {
    pub fn new(transcript: impl Into<String>) -> Self {
        Self {
            transcript: transcript.into(),
        }
    }
}

#[async_trait]
impl AsrAdapter for MockAsr {
    async fn transcribe(&self, _audio: &[AudioChunk]) -> Result<Transcript, AsrError> {
        Ok(Transcript::new(self.transcript.clone()))
    }
}

// ---------------------------------------------------------------------------
// RecordingAsr — records what it was given, answers differently each call
// ---------------------------------------------------------------------------

/// What one call to [`RecordingAsr::transcribe`] was handed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsrCall {
    /// Total samples across every chunk in the call.
    pub samples: usize,
    /// The rate the chunks declared, or 0 when there were none.
    pub sample_rate: u32,
}

/// An ASR adapter that remembers the audio it received.
///
/// [`MockAsr`] answers the same way whatever it is given, which is enough
/// for text-stage tests but cannot distinguish "the adapter saw the
/// silence" from "the adapter saw only the speech" — the thing voice
/// activity detection exists to change. This one records each call and
/// answers with the next transcript in its list, so both the audio
/// reaching the model and the number of passes over it are observable.
pub struct RecordingAsr {
    calls: std::sync::Arc<std::sync::Mutex<Vec<AsrCall>>>,
    transcripts: Vec<String>,
}

impl RecordingAsr {
    /// Answers with `transcripts` in order, repeating the last one once
    /// they run out.
    pub fn new<S: Into<String>>(transcripts: impl IntoIterator<Item = S>) -> Self {
        let transcripts: Vec<String> = transcripts.into_iter().map(Into::into).collect();
        assert!(
            !transcripts.is_empty(),
            "RecordingAsr needs at least one transcript"
        );
        Self {
            calls: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            transcripts,
        }
    }

    /// A handle to the call log, cloneable before the adapter is moved
    /// into a pipeline.
    pub fn calls(&self) -> std::sync::Arc<std::sync::Mutex<Vec<AsrCall>>> {
        std::sync::Arc::clone(&self.calls)
    }
}

#[async_trait]
impl AsrAdapter for RecordingAsr {
    async fn transcribe(&self, audio: &[AudioChunk]) -> Result<Transcript, AsrError> {
        let mut calls = self.calls.lock().expect("call log poisoned");
        let index = calls.len();
        calls.push(AsrCall {
            samples: audio.iter().map(|c| c.samples.len()).sum(),
            sample_rate: AudioChunk::sample_rate_of(audio).unwrap_or(0),
        });
        let text = self
            .transcripts
            .get(index)
            .or_else(|| self.transcripts.last())
            .expect("checked non-empty in new");
        Ok(Transcript::new(text.clone()))
    }
}

// ---------------------------------------------------------------------------
// MockContextProvider — returns a fixed context
// ---------------------------------------------------------------------------

pub struct MockContextProvider {
    pub snapshot: ContextSnapshot,
}

impl MockContextProvider {
    pub fn new() -> Self {
        Self {
            snapshot: ContextSnapshot::default(),
        }
    }

    pub fn with_app(mut self, name: impl Into<String>, field_type: FieldType) -> Self {
        self.snapshot.app_name = Some(name.into());
        self.snapshot.field_type = Some(field_type);
        self
    }
}

impl Default for MockContextProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ContextProvider for MockContextProvider {
    async fn get_context(&self) -> ContextSnapshot {
        self.snapshot.clone()
    }
}

// ---------------------------------------------------------------------------
// MockRefiner — passes text through or applies a fixed transformation
// ---------------------------------------------------------------------------

pub enum MockRefinerBehavior {
    /// Return the raw text unchanged.
    Passthrough,
    /// Uppercase the raw text (simple transformation for testing).
    Uppercase,
    /// Return a fixed string regardless of input.
    Fixed(String),
    /// Simulate an error.
    Fail(String),
}

pub struct MockRefiner {
    pub behavior: MockRefinerBehavior,
}

impl MockRefiner {
    pub fn passthrough() -> Self {
        Self {
            behavior: MockRefinerBehavior::Passthrough,
        }
    }

    pub fn uppercase() -> Self {
        Self {
            behavior: MockRefinerBehavior::Uppercase,
        }
    }

    pub fn fixed(text: impl Into<String>) -> Self {
        Self {
            behavior: MockRefinerBehavior::Fixed(text.into()),
        }
    }

    pub fn failing(msg: impl Into<String>) -> Self {
        Self {
            behavior: MockRefinerBehavior::Fail(msg.into()),
        }
    }
}

#[async_trait]
impl LlmRefiner for MockRefiner {
    async fn refine(&self, input: RefinementInput) -> Result<RefinementOutput, RefineError> {
        match &self.behavior {
            MockRefinerBehavior::Passthrough => Ok(RefinementOutput::TextInsertion {
                text: input.raw_text,
                formatting: None,
            }),
            MockRefinerBehavior::Uppercase => Ok(RefinementOutput::TextInsertion {
                text: input.raw_text.to_uppercase(),
                formatting: None,
            }),
            MockRefinerBehavior::Fixed(s) => Ok(RefinementOutput::TextInsertion {
                text: s.clone(),
                formatting: None,
            }),
            MockRefinerBehavior::Fail(msg) => Err(RefineError::Failed(msg.clone())),
        }
    }
}

// ---------------------------------------------------------------------------
// MockEmitter — collects output in a shared buffer
// ---------------------------------------------------------------------------

pub struct MockEmitter {
    pub outputs: std::sync::Arc<tokio::sync::Mutex<Vec<RefinementOutput>>>,
}

impl MockEmitter {
    pub fn new() -> Self {
        Self {
            outputs: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
        }
    }

    pub fn outputs(&self) -> std::sync::Arc<tokio::sync::Mutex<Vec<RefinementOutput>>> {
        self.outputs.clone()
    }
}

impl Default for MockEmitter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl OutputEmitter for MockEmitter {
    async fn emit(&self, output: RefinementOutput) -> EmitResult {
        self.outputs.lock().await.push(output);
        EmitResult::ok()
    }

    async fn undo(&self) -> EmitResult {
        let mut buf = self.outputs.lock().await;
        if buf.pop().is_some() {
            EmitResult::ok()
        } else {
            EmitResult::fail("nothing to undo")
        }
    }
}

