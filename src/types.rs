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

/// The result produced by an ASR adapter — either a partial hypothesis or a
/// final, committed transcription.
#[derive(Debug, Clone)]
pub struct AsrResult {
    pub text: String,
    pub is_final: bool,
    pub confidence: f32,
    pub timestamp: Duration,
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

/// Error information when emission fails.
#[derive(Debug, Clone)]
pub struct EmitError {
    pub message: String,
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for EmitError {}

/// The result of an output emission attempt.
#[derive(Debug, Clone)]
pub struct EmitResult {
    pub success: bool,
    pub error: Option<EmitError>,
}

impl EmitResult {
    pub fn ok() -> Self {
        Self {
            success: true,
            error: None,
        }
    }

    pub fn fail(msg: impl Into<String>) -> Self {
        Self {
            success: false,
            error: Some(EmitError {
                message: msg.into(),
            }),
        }
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
