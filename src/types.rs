use std::collections::HashMap;
use std::time::Duration;

// ---------------------------------------------------------------------------
// ASR layer
// ---------------------------------------------------------------------------

/// A chunk of raw audio data flowing from the microphone.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioChunk {
    /// Flatten a captured utterance into one sample buffer.
    ///
    /// Every backend wants a contiguous buffer, and the chunk
    /// boundaries only record how the audio happened to arrive, so this
    /// lives here rather than being rewritten in each adapter.
    pub fn concat(chunks: &[AudioChunk]) -> Vec<f32> {
        let total = chunks.iter().map(|c| c.samples.len()).sum();
        let mut out = Vec::with_capacity(total);
        for chunk in chunks {
            out.extend_from_slice(&chunk.samples);
        }
        out
    }

    /// The sample rate of the first chunk, if there is one. Capture does
    /// not change rate mid-utterance, so the first chunk speaks for all.
    pub fn sample_rate_of(chunks: &[AudioChunk]) -> Option<u32> {
        chunks.first().map(|c| c.sample_rate)
    }
}

/// What an [`AsrAdapter`](crate::traits::AsrAdapter) produces for one
/// utterance.
///
/// `#[non_exhaustive]`: word timings and per-token confidences are the
/// obvious future additions, and callers only ever read this, so
/// reserving the right to grow it costs them nothing.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct Transcript {
    /// The recognised text.
    pub text: String,
    /// How confident the model is, in `0.0..=1.0`. Backends that do not
    /// report a confidence leave this at `1.0` rather than inventing a
    /// number.
    pub confidence: f32,
    /// Length of the audio this came from, when the adapter knows it.
    pub duration: Option<Duration>,
}

impl Transcript {
    /// A transcript carrying only text, with no confidence claim.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            confidence: 1.0,
            duration: None,
        }
    }
}

/// A half-open character range `[start, end)` within a piece of text.
///
/// Filters report what they detected as spans so a caller can highlight,
/// undo, or score them; the evaluation harness scores gold spans against
/// detected ones. It lives here rather than in either because both need
/// it and neither owns it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    /// Character-level intersection-over-union with another span.
    /// Returns 0.0 when either span is empty.
    pub fn iou(&self, other: &Span) -> f64 {
        if self.is_empty() || other.is_empty() {
            return 0.0;
        }
        let inter_start = self.start.max(other.start);
        let inter_end = self.end.min(other.end);
        if inter_start >= inter_end {
            return 0.0;
        }
        let inter = (inter_end - inter_start) as f64;
        let union = (self.len() + other.len()) as f64 - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }
}

// ---------------------------------------------------------------------------
// Context layer
// ---------------------------------------------------------------------------

/// The kind of text field that currently has focus.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FieldType {
    CodeEditor,
    EmailCompose,
    ChatMessage,
    Terminal,
    Document,
    SearchBar,
    Generic,
}

/// A snapshot of the OS / application context at the moment of dictation.
#[derive(Debug, Clone, Default)]
pub struct ContextSnapshot {
    pub app_name: Option<String>,
    pub app_bundle_id: Option<String>,
    pub field_content: Option<String>,
    pub field_type: Option<FieldType>,
    pub custom_dictionary: Vec<String>,
    pub instructions: Option<String>,
    pub locale: Option<String>,
}

// ---------------------------------------------------------------------------
// LLM refinement layer
// ---------------------------------------------------------------------------

/// Which processing mode the LLM refiner should use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefinementMode {
    /// Normal text formatting (Phase 1).
    Dictation,
    /// Interpret as a command (Phase 2).
    Command,
    /// Produce structured output (Phase 2).
    Structured,
}

/// The input bundle sent to the LLM refiner.
#[derive(Debug, Clone)]
pub struct RefinementInput {
    pub raw_text: String,
    pub context: ContextSnapshot,
    pub mode: RefinementMode,
}

/// Optional formatting hints attached to a text insertion.
#[derive(Debug, Clone, Default)]
pub struct FormattingHint {
    pub language: Option<String>,
    pub style: Option<String>,
}

/// The output produced by the LLM refiner — deliberately extensible via enum
/// variants so that Phase 2+ additions do not break existing code.
#[derive(Debug, Clone)]
pub enum RefinementOutput {
    /// Phase 1: insert formatted text into the active application.
    TextInsertion {
        text: String,
        formatting: Option<FormattingHint>,
    },
    /// Phase 2: interpret as a command to execute.
    Command {
        action: String,
        parameters: HashMap<String, String>,
    },
    /// Phase 2-3: intent + optional text + metadata.
    StructuredInput {
        intent: String,
        text: Option<String>,
        metadata: HashMap<String, String>,
    },
}

// ---------------------------------------------------------------------------
// Output layer
// ---------------------------------------------------------------------------

/// Why an [`OutputEmitter`](crate::traits::OutputEmitter) could not
/// deliver its output.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum EmitError {
    /// The target refused the write — clipboard unavailable, no focused
    /// window, permission denied.
    #[error("output target unavailable: {0}")]
    Unavailable(String),

    /// The emitter cannot represent this output kind, e.g. a
    /// `Command` handed to a text-insertion emitter.
    #[error("unsupported output: {0}")]
    Unsupported(String),

    /// There was nothing to undo.
    #[error("nothing to undo")]
    NothingToUndo,
}

/// The result of an output emission attempt.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EmitResult {
    pub success: bool,
    pub error: Option<EmitError>,
}

impl EmitResult {
    /// The emission succeeded.
    pub fn ok() -> Self {
        Self {
            success: true,
            error: None,
        }
    }

    /// The emission failed for the given reason.
    pub fn failed(error: EmitError) -> Self {
        Self {
            success: false,
            error: Some(error),
        }
    }

    /// The emission failed with an unavailable target. Shorthand for
    /// the common case; use [`failed`](Self::failed) to pick a variant.
    pub fn fail(msg: impl Into<String>) -> Self {
        Self::failed(EmitError::Unavailable(msg.into()))
    }
}

// ---------------------------------------------------------------------------
// Activation layer
// ---------------------------------------------------------------------------

/// How a dictation session is started / stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationMethod {
    /// Global hotkey — hold to record, release to finish.
    Hotkey(String),
    /// Explicit push-to-talk toggle.
    PushToTalk,
    /// Voice Activity Detection — automatic start / stop.
    Vad,
}

// ---------------------------------------------------------------------------
// Pipeline state machine
// ---------------------------------------------------------------------------

/// The lifecycle states of a single dictation session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineState {
    Idle,
    Activating,
    Recording,
    Processing,
    Emitting,
    Cancelling,
}
