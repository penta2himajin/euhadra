use async_trait::async_trait;

use crate::types::*;

// ---------------------------------------------------------------------------
// ASR Adapter
// ---------------------------------------------------------------------------

/// Turns audio into text.
///
/// Implementors may be local (Whisper.cpp, Parakeet via ONNX) or
/// cloud-based (OpenAI Whisper API, Deepgram). The pipeline treats them
/// identically.
///
/// The whole utterance is passed at once and one transcript comes back.
/// An adapter that can also emit partial hypotheses as audio arrives
/// will implement an additional streaming trait alongside this one —
/// that capability is not part of 0.1.0, and the shape here is
/// deliberately the one every backend can satisfy.
#[async_trait]
pub trait AsrAdapter: Send + Sync {
    /// Transcribe a complete utterance.
    ///
    /// `audio` holds the chunks in capture order. Adapters that need a
    /// contiguous buffer should concatenate; the split between chunks
    /// carries no meaning beyond how the audio happened to arrive.
    async fn transcribe(&self, audio: &[AudioChunk]) -> Result<Transcript, AsrError>;
}

/// Why an [`AsrAdapter`] could not produce a transcript.
///
/// The variants are the distinctions a caller can act on: a missing
/// model needs a different response than a runtime failure, and neither
/// is the same as the user simply not having spoken. Marked
/// `#[non_exhaustive]` so that finer distinctions can be added later
/// without breaking a `match`.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum AsrError {
    /// The model bundle could not be loaded — a missing file, a
    /// malformed vocabulary, unreadable normalisation statistics.
    #[error("failed to load model: {0}")]
    ModelLoad(String),

    /// The adapter was asked for something it cannot do, e.g. a beam
    /// width below the minimum it supports.
    #[error("invalid ASR configuration: {0}")]
    Config(String),

    /// No audio reached the adapter.
    #[error("no audio received")]
    NoAudio,

    /// The model loaded but inference failed.
    #[error("inference failed: {0}")]
    Inference(String),

    /// Aborted before completion, usually via the session's
    /// `CancellationToken`.
    #[error("cancelled")]
    Cancelled,
}

// ---------------------------------------------------------------------------
// Context Provider
// ---------------------------------------------------------------------------

/// Captures a snapshot of the current OS / application context.
///
/// Implementors call into platform-specific APIs: macOS Accessibility
/// (AXUIElement), Windows UI Automation, Linux AT-SPI, or a manual
/// provider for testing.
#[async_trait]
pub trait ContextProvider: Send + Sync {
    async fn get_context(&self) -> ContextSnapshot;
}

// ---------------------------------------------------------------------------
// LLM Refiner
// ---------------------------------------------------------------------------

/// Takes processed ASR text + application context and produces refined
/// output (Tier 3).
///
/// This is **text-stage** refinement only: it does not re-run ASR.
/// Do not confuse it with [`crate::pipeline::FinalPass`] (which picks
/// the audio/text source for the session-end transcript) or with a
/// delayed second-pass decode over grouped audio (#146). Concrete
/// implementations are gated behind the reserved `llm` feature (#122).
///
/// Implementors may call cloud LLMs (Cerebras, Groq, OpenAI) or on-device
/// models (Apple Foundation Models, Gemini Nano, Ollama).
#[async_trait]
pub trait LlmRefiner: Send + Sync {
    async fn refine(&self, input: RefinementInput) -> Result<RefinementOutput, RefineError>;
}

/// Why an [`LlmRefiner`] could not refine its input.
///
/// The pipeline degrades gracefully on any of these — it emits the
/// unrefined text rather than failing the session — so the distinction
/// exists for logging and for callers driving a refiner directly.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum RefineError {
    /// The model or endpoint could not be reached or initialised.
    #[error("refiner unavailable: {0}")]
    Unavailable(String),

    /// The refiner ran but its output could not be used.
    #[error("refinement failed: {0}")]
    Failed(String),

    /// Aborted before completion.
    #[error("cancelled")]
    Cancelled,
}

// ---------------------------------------------------------------------------
// Output Emitter
// ---------------------------------------------------------------------------

/// Delivers the final pipeline output to the OS / application.
///
/// Implementors handle clipboard insertion, key emulation, stdout, callbacks,
/// or any other output mechanism.
#[async_trait]
pub trait OutputEmitter: Send + Sync {
    /// Emit the refined output to the target.
    async fn emit(&self, output: RefinementOutput) -> EmitResult;

    /// Undo the most recent emission, if possible.
    async fn undo(&self) -> EmitResult;
}
