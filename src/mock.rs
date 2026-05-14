use async_trait::async_trait;
use tokio::sync::mpsc;

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
    async fn transcribe(
        &self,
        mut audio_rx: mpsc::Receiver<AudioChunk>,
        result_tx: mpsc::Sender<AsrResult>,
    ) -> Result<(), AsrError> {
        // Drain all audio (simulate listening)
        while audio_rx.recv().await.is_some() {}

        // Emit a single final result
        let result = AsrResult {
            text: self.transcript.clone(),
            is_final: true,
            confidence: 1.0,
            timestamp: std::time::Duration::ZERO,
        };
        result_tx.send(result).await.map_err(|e| AsrError {
            message: format!("send failed: {e}"),
        })?;

        Ok(())
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
            MockRefinerBehavior::Fail(msg) => Err(RefineError {
                message: msg.clone(),
            }),
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

// ---------------------------------------------------------------------------
// StdoutEmitter — prints to stdout (for CLI use)
// ---------------------------------------------------------------------------

pub struct StdoutEmitter;

#[async_trait]
impl OutputEmitter for StdoutEmitter {
    async fn emit(&self, output: RefinementOutput) -> EmitResult {
        match &output {
            RefinementOutput::TextInsertion { text, .. } => {
                println!("{text}");
                EmitResult::ok()
            }
            _ => EmitResult::fail("StdoutEmitter only supports TextInsertion"),
        }
    }

    async fn undo(&self) -> EmitResult {
        EmitResult::fail("StdoutEmitter does not support undo")
    }
}
