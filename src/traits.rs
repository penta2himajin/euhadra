use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::types::*;

// ---------------------------------------------------------------------------
// ASR Adapter
// ---------------------------------------------------------------------------

/// Converts a stream of audio chunks into a stream of recognition results.
///
/// Implementors may be local (Whisper.cpp, Apple Speech) or cloud-based
/// (OpenAI Whisper API, Deepgram, ElevenLabs Scribe).  The pipeline treats
/// them identically.
#[async_trait]
pub trait AsrAdapter: Send + Sync {
    /// Begin transcribing.  Audio chunks arrive on `audio_rx`; recognition
    /// results should be sent to `result_tx`.  The method returns when the
    /// audio stream is closed or the adapter encounters a fatal error.
    async fn transcribe(
        &self,
        audio_rx: mpsc::Receiver<AudioChunk>,
        result_tx: mpsc::Sender<AsrResult>,
    ) -> Result<(), AsrError>;
}

#[derive(Debug, Clone)]
pub struct AsrError {
    pub message: String,
}

impl std::fmt::Display for AsrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ASR error: {}", self.message)
    }
}

impl std::error::Error for AsrError {}

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

/// Takes raw ASR text + application context and produces refined output.
///
/// Implementors may call cloud LLMs (Cerebras, Groq, OpenAI) or on-device
/// models (Apple Foundation Models, Gemini Nano, Ollama).
#[async_trait]
pub trait LlmRefiner: Send + Sync {
    async fn refine(&self, input: RefinementInput) -> Result<RefinementOutput, RefineError>;
}

#[derive(Debug, Clone)]
pub struct RefineError {
    pub message: String,
}

impl std::fmt::Display for RefineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Refinement error: {}", self.message)
    }
}

impl std::error::Error for RefineError {}

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
