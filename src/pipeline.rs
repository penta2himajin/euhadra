use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::filter::TextFilter;
use crate::processor::TextProcessor;
use crate::state::StateMachine;
use crate::traits::*;
use crate::types::*;

// ---------------------------------------------------------------------------
// Pipeline configuration
// ---------------------------------------------------------------------------

/// Builds a configured pipeline from adapter implementations.
pub struct PipelineBuilder {
    asr: Option<Arc<dyn AsrAdapter>>,
    filters: Vec<Arc<dyn TextFilter>>,
    processors: Vec<Arc<dyn TextProcessor>>,
    refiner: Option<Arc<dyn LlmRefiner>>,
    context: Option<Arc<dyn ContextProvider>>,
    emitter: Option<Arc<dyn OutputEmitter>>,
    audio_channel_size: usize,
    asr_channel_size: usize,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            asr: None,
            filters: Vec::new(),
            processors: Vec::new(),
            refiner: None,
            context: None,
            emitter: None,
            audio_channel_size: 32,
            asr_channel_size: 8,
        }
    }

    pub fn asr(mut self, asr: impl AsrAdapter + 'static) -> Self {
        self.asr = Some(Arc::new(asr));
        self
    }

    /// Add a text filter applied between ASR and LLM refinement.
    /// Filters run in the order they are added.
    pub fn filter(mut self, filter: impl TextFilter + 'static) -> Self {
        self.filters.push(Arc::new(filter));
        self
    }

    pub fn refiner(mut self, refiner: impl LlmRefiner + 'static) -> Self {
        self.refiner = Some(Arc::new(refiner));
        self
    }

    /// Add a text processor applied between TextFilter and LLM refinement.
    /// Processors run in the order they are added.
    pub fn processor(mut self, proc: impl TextProcessor + 'static) -> Self {
        self.processors.push(Arc::new(proc));
        self
    }

    pub fn context(mut self, ctx: impl ContextProvider + 'static) -> Self {
        self.context = Some(Arc::new(ctx));
        self
    }

    pub fn emitter(mut self, emitter: impl OutputEmitter + 'static) -> Self {
        self.emitter = Some(Arc::new(emitter));
        self
    }

    pub fn audio_channel_size(mut self, size: usize) -> Self {
        self.audio_channel_size = size;
        self
    }

    pub fn asr_channel_size(mut self, size: usize) -> Self {
        self.asr_channel_size = size;
        self
    }

    pub fn build(self) -> Result<Pipeline, PipelineError> {
        Ok(Pipeline {
            asr: self.asr.ok_or(PipelineError::missing("asr"))?,
            filters: self.filters,
            processors: self.processors,
            refiner: self.refiner.ok_or(PipelineError::missing("refiner"))?,
            context: self.context.ok_or(PipelineError::missing("context"))?,
            emitter: self.emitter.ok_or(PipelineError::missing("emitter"))?,
            audio_channel_size: self.audio_channel_size,
            asr_channel_size: self.asr_channel_size,
        })
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// A fully-configured dictation pipeline ready to process voice input.
pub struct Pipeline {
    asr: Arc<dyn AsrAdapter>,
    filters: Vec<Arc<dyn TextFilter>>,
    processors: Vec<Arc<dyn TextProcessor>>,
    refiner: Arc<dyn LlmRefiner>,
    context: Arc<dyn ContextProvider>,
    emitter: Arc<dyn OutputEmitter>,
    audio_channel_size: usize,
    asr_channel_size: usize,
}

impl Pipeline {
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::new()
    }

    /// Run one dictation session: accept audio chunks via the returned sender,
    /// and the pipeline drives them through ASR → filter → process → context → refinement → emit.
    ///
    /// Returns an `AudioChunk` sender, a `CancellationToken` for aborting, and
    /// a join handle that resolves to the session result.
    pub fn session(
        &self,
    ) -> (
        mpsc::Sender<AudioChunk>,
        CancellationToken,
        tokio::task::JoinHandle<Result<SessionResult, PipelineError>>,
    ) {
        let (audio_tx, audio_rx) = mpsc::channel(self.audio_channel_size);
        let cancel = CancellationToken::new();

        let asr = Arc::clone(&self.asr);
        let filters: Vec<Arc<dyn TextFilter>> = self.filters.iter().map(Arc::clone).collect();
        let processors: Vec<Arc<dyn TextProcessor>> = self.processors.iter().map(Arc::clone).collect();
        let refiner = Arc::clone(&self.refiner);
        let context = Arc::clone(&self.context);
        let emitter = Arc::clone(&self.emitter);
        let asr_ch = self.asr_channel_size;
        let cancel_clone = cancel.clone();

        let handle = tokio::spawn(async move {
            run_session(asr, filters, processors, refiner, context, emitter, audio_rx, asr_ch, cancel_clone).await
        });

        (audio_tx, cancel, handle)
    }
}

/// The outcome of a completed dictation session.
#[derive(Debug, Clone)]
pub struct SessionResult {
    pub raw_text: String,
    pub output: RefinementOutput,
    pub emit_result: EmitResult,
}

// ---------------------------------------------------------------------------
// Pipeline errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PipelineError {
    pub message: String,
}

impl PipelineError {
    pub fn missing(component: &str) -> Self {
        Self {
            message: format!("missing required component: {component}"),
        }
    }
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "pipeline error: {}", self.message)
    }
}

impl std::error::Error for PipelineError {}

// ---------------------------------------------------------------------------
// Session execution
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn run_session(
    asr: Arc<dyn AsrAdapter>,
    filters: Vec<Arc<dyn TextFilter>>,
    processors: Vec<Arc<dyn TextProcessor>>,
    refiner: Arc<dyn LlmRefiner>,
    context: Arc<dyn ContextProvider>,
    emitter: Arc<dyn OutputEmitter>,
    audio_rx: mpsc::Receiver<AudioChunk>,
    asr_channel_size: usize,
    cancel: CancellationToken,
) -> Result<SessionResult, PipelineError> {
    let mut sm = StateMachine::new();

    // ── Activating ──────────────────────────────────────────────────────
    sm.transition(PipelineState::Activating)
        .map_err(|e| PipelineError { message: e.to_string() })?;

    // ── Recording + ASR ─────────────────────────────────────────────────
    sm.transition(PipelineState::Recording)
        .map_err(|e| PipelineError { message: e.to_string() })?;

    let (asr_tx, mut asr_rx) = mpsc::channel::<AsrResult>(asr_channel_size);

    let asr_handle = {
        let cancel = cancel.clone();
        tokio::spawn(async move {
            tokio::select! {
                result = asr.transcribe(audio_rx, asr_tx) => result,
                _ = cancel.cancelled() => Err(AsrError { message: "cancelled".into() }),
            }
        })
    };

    // Collect the final ASR result (discard partials, keep the last final).
    let mut final_text = String::new();
    loop {
        tokio::select! {
            maybe = asr_rx.recv() => {
                match maybe {
                    Some(result) if result.is_final => {
                        final_text = result.text;
                    }
                    Some(_partial) => {
                        // Phase 1: ignore partials
                    }
                    None => break, // ASR channel closed
                }
            }
            _ = cancel.cancelled() => {
                sm.cancel().ok();
                sm.reset();
                return Err(PipelineError { message: "cancelled during recording".into() });
            }
        }
    }

    // Wait for ASR task to finish
    let _ = asr_handle.await;

    if final_text.is_empty() {
        sm.reset();
        return Err(PipelineError { message: "no speech detected".into() });
    }

    // ── Processing (filter → context → refinement) ──────────────────────
    sm.transition(PipelineState::Processing)
        .map_err(|e| PipelineError { message: e.to_string() })?;

    // Apply text filters (filler removal, etc.)
    let mut filtered_text = final_text.clone();
    let mut all_removed: Vec<String> = Vec::new();
    for f in &filters {
        match f.filter(&filtered_text).await {
            Ok(result) => {
                tracing::debug!(
                    before = %filtered_text,
                    after = %result.text,
                    removed = ?result.removed,
                    "filter applied"
                );
                filtered_text = result.text;
                all_removed.extend(result.removed);
            }
            Err(e) => {
                tracing::warn!(error = %e, "filter failed, continuing with unfiltered text");
            }
        }
    }

    // Apply text processors (punctuation, self-correction, etc.)
    let mut processed_text = filtered_text.clone();
    let ctx_for_proc = ContextSnapshot::default(); // processors get a basic context
    for p in &processors {
        match p.process(&processed_text, &ctx_for_proc).await {
            Ok(result) => {
                tracing::debug!(
                    before = %processed_text,
                    after = %result.text,
                    corrections = ?result.corrections,
                    "processor applied"
                );
                processed_text = result.text;
            }
            Err(e) => {
                tracing::warn!(error = %e, "processor failed, continuing with unprocessed text");
            }
        }
    }

    let ctx = {
        let cancel = cancel.clone();
        tokio::select! {
            snapshot = context.get_context() => snapshot,
            _ = cancel.cancelled() => {
                sm.cancel().ok();
                sm.reset();
                return Err(PipelineError { message: "cancelled during context".into() });
            }
        }
    };

    let input = RefinementInput {
        raw_text: processed_text,
        context: ctx,
        mode: RefinementMode::Dictation,
    };

    let output = {
        let cancel = cancel.clone();
        tokio::select! {
            result = refiner.refine(input) => {
                result.unwrap_or_else(|_| {
                    // Graceful degradation: emit raw text on LLM failure
                    tracing::warn!("LLM refinement failed, falling back to raw text");
                    RefinementOutput::TextInsertion {
                        text: final_text.clone(),
                        formatting: None,
                    }
                })
            }
            _ = cancel.cancelled() => {
                sm.cancel().ok();
                sm.reset();
                return Err(PipelineError { message: "cancelled during refinement".into() });
            }
        }
    };

    // ── Emitting ────────────────────────────────────────────────────────
    sm.transition(PipelineState::Emitting)
        .map_err(|e| PipelineError { message: e.to_string() })?;

    let emit_result = emitter.emit(output.clone()).await;

    // ── Back to Idle ────────────────────────────────────────────────────
    sm.transition(PipelineState::Idle)
        .map_err(|e| PipelineError { message: e.to_string() })?;

    Ok(SessionResult {
        raw_text: final_text,
        output,
        emit_result,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::*;

    #[tokio::test]
    async fn full_pipeline_happy_path() {
        let emitter = MockEmitter::new();
        let outputs = emitter.outputs();

        let pipeline = Pipeline::builder()
            .asr(MockAsr::new("hello world"))
            .refiner(MockRefiner::uppercase())
            .context(MockContextProvider::new())
            .emitter(emitter)
            .build()
            .unwrap();

        let (audio_tx, _cancel, handle) = pipeline.session();

        // Send one audio chunk then close the channel
        audio_tx
            .send(AudioChunk {
                samples: vec![0.0; 160],
                sample_rate: 16000,
                channels: 1,
            })
            .await
            .unwrap();
        drop(audio_tx); // signal end of audio

        let result = handle.await.unwrap().unwrap();
        assert_eq!(result.raw_text, "hello world");
        assert!(result.emit_result.success);

        let buf = outputs.lock().await;
        assert_eq!(buf.len(), 1);
        match &buf[0] {
            RefinementOutput::TextInsertion { text, .. } => {
                assert_eq!(text, "HELLO WORLD");
            }
            _ => panic!("expected TextInsertion"),
        }
    }

    #[tokio::test]
    async fn graceful_degradation_on_llm_failure() {
        let emitter = MockEmitter::new();
        let outputs = emitter.outputs();

        let pipeline = Pipeline::builder()
            .asr(MockAsr::new("raw dictation text"))
            .refiner(MockRefiner::failing("API timeout"))
            .context(MockContextProvider::new())
            .emitter(emitter)
            .build()
            .unwrap();

        let (audio_tx, _cancel, handle) = pipeline.session();
        audio_tx
            .send(AudioChunk {
                samples: vec![0.0; 160],
                sample_rate: 16000,
                channels: 1,
            })
            .await
            .unwrap();
        drop(audio_tx);

        let result = handle.await.unwrap().unwrap();
        // Should fall back to raw text
        let buf = outputs.lock().await;
        match &buf[0] {
            RefinementOutput::TextInsertion { text, .. } => {
                assert_eq!(text, "raw dictation text");
            }
            _ => panic!("expected TextInsertion fallback"),
        }
        assert!(result.emit_result.success);
    }

    #[tokio::test]
    async fn cancellation_during_recording() {
        let pipeline = Pipeline::builder()
            .asr(MockAsr::new("will be cancelled"))
            .refiner(MockRefiner::passthrough())
            .context(MockContextProvider::new())
            .emitter(MockEmitter::new())
            .build()
            .unwrap();

        let (audio_tx, cancel, handle) = pipeline.session();

        // Send audio but cancel before closing the channel
        audio_tx
            .send(AudioChunk {
                samples: vec![0.0; 160],
                sample_rate: 16000,
                channels: 1,
            })
            .await
            .unwrap();

        // Cancel while ASR is still waiting for more audio
        cancel.cancel();

        let result = handle.await.unwrap();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn missing_component_fails_build() {
        let result = Pipeline::builder()
            .asr(MockAsr::new("test"))
            // missing refiner, context, emitter
            .build();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn pipeline_with_filler_filter() {
        use crate::filter::SimpleFillerFilter;

        let emitter = MockEmitter::new();
        let outputs = emitter.outputs();

        let pipeline = Pipeline::builder()
            .asr(MockAsr::new("um I think uh we should deploy"))
            .filter(SimpleFillerFilter::english())
            .refiner(MockRefiner::passthrough())
            .context(MockContextProvider::new())
            .emitter(emitter)
            .build()
            .unwrap();

        let (audio_tx, _cancel, handle) = pipeline.session();
        audio_tx
            .send(AudioChunk {
                samples: vec![0.0; 160],
                sample_rate: 16000,
                channels: 1,
            })
            .await
            .unwrap();
        drop(audio_tx);

        let result = handle.await.unwrap().unwrap();
        // Raw text still has fillers
        assert_eq!(result.raw_text, "um I think uh we should deploy");
        // Emitted output should be filtered
        let buf = outputs.lock().await;
        match &buf[0] {
            RefinementOutput::TextInsertion { text, .. } => {
                assert_eq!(text, "I think we should deploy");
            }
            _ => panic!("expected TextInsertion"),
        }
    }

    #[tokio::test]
    async fn pipeline_with_filter_and_processor() {
        use crate::filter::SimpleFillerFilter;
        use crate::processor::{BasicPunctuationRestorer, SelfCorrectionDetector};

        let emitter = MockEmitter::new();
        let outputs = emitter.outputs();

        // Input with fillers AND self-correction
        let pipeline = Pipeline::builder()
            .asr(MockAsr::new(
                "um I want to go to Boston no wait to Denver",
            ))
            .filter(SimpleFillerFilter::english())
            .processor(SelfCorrectionDetector::new())
            .processor(BasicPunctuationRestorer)
            .refiner(MockRefiner::passthrough())
            .context(MockContextProvider::new())
            .emitter(emitter)
            .build()
            .unwrap();

        let (audio_tx, _cancel, handle) = pipeline.session();
        audio_tx
            .send(AudioChunk {
                samples: vec![0.0; 160],
                sample_rate: 16000,
                channels: 1,
            })
            .await
            .unwrap();
        drop(audio_tx);

        let _result = handle.await.unwrap().unwrap();
        let buf = outputs.lock().await;
        match &buf[0] {
            RefinementOutput::TextInsertion { text, .. } => {
                // Fillers removed, self-correction resolved, capitalized, period added
                assert!(
                    !text.contains("um"),
                    "filler should be removed: {text}"
                );
                assert!(
                    !text.contains("Boston"),
                    "reparandum should be removed: {text}"
                );
                assert!(
                    text.contains("Denver"),
                    "repair should be kept: {text}"
                );
                assert!(
                    text.starts_with(|c: char| c.is_uppercase()),
                    "should be capitalized: {text}"
                );
                assert!(
                    text.ends_with('.'),
                    "should have terminal period: {text}"
                );
            }
            _ => panic!("expected TextInsertion"),
        }
    }
}
