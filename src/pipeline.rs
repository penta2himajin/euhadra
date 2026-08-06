use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::filter::TextFilter;
use crate::processor::{Correction, TextProcessor};
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

    /// Assemble the pipeline.
    ///
    /// Only an ASR adapter is required. Without a refiner the processed
    /// text passes through untouched; without a context provider the
    /// stages see an empty [`ContextSnapshot`]; without an emitter the
    /// output is returned in the [`SessionResult`] and nothing is
    /// written anywhere. Those three defaults are what makes the
    /// LLM-free path — the one this crate exists for — expressible.
    pub fn build(self) -> Result<Pipeline, PipelineError> {
        Ok(Pipeline {
            asr: self.asr.ok_or(PipelineError::MissingComponent("asr"))?,
            filters: self.filters,
            processors: self.processors,
            refiner: self.refiner,
            context: self.context,
            emitter: self.emitter,
            audio_channel_size: self.audio_channel_size,
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
    refiner: Option<Arc<dyn LlmRefiner>>,
    context: Option<Arc<dyn ContextProvider>>,
    emitter: Option<Arc<dyn OutputEmitter>>,
    audio_channel_size: usize,
}

impl Pipeline {
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::new()
    }

    /// Run a complete utterance through the pipeline.
    ///
    /// Use this when the audio is already in hand — a WAV file, a
    /// recording that has finished. For live capture, where the ASR
    /// should start working before the speaker stops, use
    /// [`session`](Self::session).
    ///
    /// ```no_run
    /// # use euhadra::prelude::*;
    /// # async fn f(pipeline: Pipeline, samples: Vec<AudioChunk>) -> Result<(), PipelineError> {
    /// let result = pipeline.transcribe(&samples).await?;
    /// println!("{}", result.text());
    /// # Ok(()) }
    /// ```
    pub async fn transcribe(&self, audio: &[AudioChunk]) -> Result<SessionResult, PipelineError> {
        run_session(
            &self.asr,
            &self.filters,
            &self.processors,
            self.refiner.as_deref(),
            self.context.as_deref(),
            self.emitter.as_deref(),
            audio,
            &CancellationToken::new(),
        )
        .await
    }

    /// Start a live session that accepts audio as it is captured.
    ///
    /// Feed chunks to [`Session::audio`], then call
    /// [`Session::finish`] to close the stream and await the result.
    ///
    /// ```no_run
    /// # use euhadra::prelude::*;
    /// # async fn f(pipeline: Pipeline, chunk: AudioChunk) -> Result<(), PipelineError> {
    /// let session = pipeline.session();
    /// session.audio.send(chunk).await.ok();
    /// let result = session.finish().await?;
    /// # Ok(()) }
    /// ```
    pub fn session(&self) -> Session {
        let (audio_tx, mut audio_rx) = mpsc::channel::<AudioChunk>(self.audio_channel_size);
        let cancel = CancellationToken::new();

        let asr = Arc::clone(&self.asr);
        let filters: Vec<Arc<dyn TextFilter>> = self.filters.iter().map(Arc::clone).collect();
        let processors: Vec<Arc<dyn TextProcessor>> =
            self.processors.iter().map(Arc::clone).collect();
        let refiner = self.refiner.clone();
        let context = self.context.clone();
        let emitter = self.emitter.clone();
        let cancel_inner = cancel.clone();

        let handle = tokio::spawn(async move {
            // Collect the utterance as it is captured, so the caller can
            // keep sending while this task is already running.
            let mut chunks: Vec<AudioChunk> = Vec::new();
            loop {
                tokio::select! {
                    maybe = audio_rx.recv() => match maybe {
                        Some(chunk) => chunks.push(chunk),
                        None => break,
                    },
                    _ = cancel_inner.cancelled() => return Err(PipelineError::Cancelled { during: "recording" }),
                }
            }

            run_session(
                &asr,
                &filters,
                &processors,
                refiner.as_deref(),
                context.as_deref(),
                emitter.as_deref(),
                &chunks,
                &cancel_inner,
            )
            .await
        });

        Session {
            audio: audio_tx,
            cancel,
            handle,
        }
    }
}

/// A live dictation session.
///
/// The audio stream is closed by [`finish`](Self::finish), which is also
/// what awaits the result — so there is no way to await a session whose
/// input is still open, and no sender to remember to drop.
pub struct Session {
    /// Send captured audio here.
    pub audio: mpsc::Sender<AudioChunk>,
    /// Cancel the session. Aborts whatever stage is in flight.
    pub cancel: CancellationToken,
    handle: tokio::task::JoinHandle<Result<SessionResult, PipelineError>>,
}

impl Session {
    /// Close the audio stream and wait for the result.
    pub async fn finish(self) -> Result<SessionResult, PipelineError> {
        let Session { audio, handle, .. } = self;
        drop(audio);
        match handle.await {
            Ok(result) => result,
            Err(e) => Err(PipelineError::TaskFailed(e.to_string())),
        }
    }

    /// Cancel the session and discard whatever it had produced.
    pub async fn abort(self) {
        self.cancel.cancel();
        let _ = self.handle.await;
    }
}

/// The outcome of a completed dictation session.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SessionResult {
    /// What the ASR adapter produced, before any text processing.
    pub raw_text: String,
    /// The final output after every enabled tier.
    pub output: RefinementOutput,
    /// How the output was delivered, or `None` when the pipeline has no
    /// emitter and the caller is taking the result by value.
    pub emit_result: Option<EmitResult>,
    /// What the text tiers did on the way through.
    pub diagnostics: Diagnostics,
}

impl SessionResult {
    /// The final text, whatever output shape the refiner produced.
    pub fn text(&self) -> &str {
        match &self.output {
            RefinementOutput::TextInsertion { text, .. } => text,
            RefinementOutput::StructuredInput { text, .. } => text.as_deref().unwrap_or_default(),
            RefinementOutput::Command { .. } => "",
        }
    }
}

/// What the Tier 1 and Tier 2 stages did to the text.
///
/// A stage that fails does not fail the session — the pipeline carries
/// on with the text it has — which is the right default but leaves the
/// caller unable to tell a clean run from a degraded one. [`failures`]
/// is how they tell.
///
/// [`failures`]: Diagnostics::failures
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct Diagnostics {
    /// Segments removed by the [`TextFilter`] stages, in text order.
    pub removed: Vec<String>,
    /// Corrections applied by the [`TextProcessor`] stages.
    pub corrections: Vec<Correction>,
    /// Stages that failed and were skipped. Empty on a clean run.
    pub failures: Vec<StageFailure>,
}

/// A stage that failed and was skipped.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StageFailure {
    /// Which tier the failing stage belongs to.
    pub stage: Stage,
    /// Its position among the stages of that tier, as configured.
    pub index: usize,
    /// What went wrong.
    pub reason: String,
}

/// Which tier a [`StageFailure`] came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Stage {
    /// Tier 1 — [`TextFilter`].
    Filter,
    /// Tier 2 — [`TextProcessor`].
    Processor,
    /// Tier 3 — [`LlmRefiner`](crate::traits::LlmRefiner).
    Refiner,
}

// ---------------------------------------------------------------------------
// Pipeline errors
// ---------------------------------------------------------------------------

/// Why a session could not produce a result.
///
/// These are the outcomes that end a session. A failing text stage is
/// not among them — that is reported through
/// [`Diagnostics::failures`] while the session continues.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum PipelineError {
    /// [`PipelineBuilder::build`] was called without a required
    /// component.
    #[error("pipeline is missing a required component: {0}")]
    MissingComponent(&'static str),

    /// The ASR adapter returned nothing usable.
    #[error("no speech detected")]
    NoSpeech,

    /// The ASR adapter failed.
    #[error(transparent)]
    Asr(#[from] AsrError),

    /// The session was cancelled through its `CancellationToken`.
    ///
    /// `during` names the stage that was in flight — `"recording"`,
    /// `"context"`, `"refinement"`. Cancellation is expected rather than
    /// exceptional, but knowing where it landed is what tells you
    /// whether propagation reaches every stage.
    #[error("cancelled during {during}")]
    Cancelled { during: &'static str },

    /// The session task itself failed — a panic in a stage, or the
    /// runtime shutting down under it.
    #[error("session task failed: {0}")]
    TaskFailed(String),

    /// The state machine refused a transition. A bug in the pipeline
    /// rather than in the caller's configuration.
    #[error("invalid state transition: {0}")]
    InvalidTransition(String),
}

// ---------------------------------------------------------------------------
// Session execution
// ---------------------------------------------------------------------------

/// Drive one utterance through every configured stage.
///
/// Text stages degrade rather than abort: a filter or processor that
/// fails is skipped, its reason recorded in [`Diagnostics::failures`],
/// and the text carries on unchanged. Only ASR failing, cancellation,
/// or a refused state transition ends the session.
#[allow(clippy::too_many_arguments)]
async fn run_session(
    asr: &Arc<dyn AsrAdapter>,
    filters: &[Arc<dyn TextFilter>],
    processors: &[Arc<dyn TextProcessor>],
    refiner: Option<&dyn LlmRefiner>,
    context: Option<&dyn ContextProvider>,
    emitter: Option<&dyn OutputEmitter>,
    audio: &[AudioChunk],
    cancel: &CancellationToken,
) -> Result<SessionResult, PipelineError> {
    let mut sm = StateMachine::new();
    let transition = |sm: &mut StateMachine, to| {
        sm.transition(to)
            .map(|_| ())
            .map_err(|e| PipelineError::InvalidTransition(e.to_string()))
    };

    transition(&mut sm, PipelineState::Activating)?;
    transition(&mut sm, PipelineState::Recording)?;

    // ── ASR ─────────────────────────────────────────────────────────────
    let transcript = tokio::select! {
        result = asr.transcribe(audio) => result?,
        _ = cancel.cancelled() => {
            sm.cancel().ok();
            sm.reset();
            return Err(PipelineError::Cancelled { during: "recording" });
        }
    };

    let raw_text = transcript.text.trim().to_string();
    if raw_text.is_empty() {
        sm.reset();
        return Err(PipelineError::NoSpeech);
    }

    transition(&mut sm, PipelineState::Processing)?;

    // ── Context ─────────────────────────────────────────────────────────
    // Fetched before the text stages, not after: TextProcessor::process
    // takes a ContextSnapshot because processors such as PhonemeCorrector
    // need the custom dictionary in it. Fetching it later — as this used
    // to — meant they were always handed an empty one.
    let ctx = match context {
        Some(provider) => tokio::select! {
            snapshot = provider.get_context() => snapshot,
            _ = cancel.cancelled() => {
                sm.cancel().ok();
                sm.reset();
                return Err(PipelineError::Cancelled { during: "context" });
            }
        },
        None => ContextSnapshot::default(),
    };

    let mut diagnostics = Diagnostics::default();

    // ── Tier 1: filters ─────────────────────────────────────────────────
    let mut text = raw_text.clone();
    for (index, f) in filters.iter().enumerate() {
        match f.filter(&text).await {
            Ok(result) => {
                tracing::debug!(before = %text, after = %result.text, removed = ?result.removed, "filter applied");
                text = result.text;
                diagnostics.removed.extend(result.removed);
            }
            Err(e) => {
                tracing::warn!(error = %e, index, "filter failed, continuing with unfiltered text");
                diagnostics.failures.push(StageFailure {
                    stage: Stage::Filter,
                    index,
                    reason: e.to_string(),
                });
            }
        }
    }

    // ── Tier 2: processors ──────────────────────────────────────────────
    for (index, p) in processors.iter().enumerate() {
        match p.process(&text, &ctx).await {
            Ok(result) => {
                tracing::debug!(before = %text, after = %result.text, corrections = ?result.corrections, "processor applied");
                text = result.text;
                diagnostics.corrections.extend(result.corrections);
            }
            Err(e) => {
                tracing::warn!(error = %e, index, "processor failed, continuing with unprocessed text");
                diagnostics.failures.push(StageFailure {
                    stage: Stage::Processor,
                    index,
                    reason: e.to_string(),
                });
            }
        }
    }

    // ── Tier 3: refinement (optional) ───────────────────────────────────
    let output = match refiner {
        None => RefinementOutput::TextInsertion {
            text: text.clone(),
            formatting: None,
        },
        Some(refiner) => {
            let input = RefinementInput {
                raw_text: text.clone(),
                context: ctx,
                mode: RefinementMode::Dictation,
            };
            tokio::select! {
                result = refiner.refine(input) => match result {
                    Ok(output) => output,
                    Err(e) => {
                        // Graceful degradation: the Tier 1+2 text is
                        // already useful on its own, which is the whole
                        // premise of the LLM being optional.
                        tracing::warn!(error = %e, "refinement failed, falling back to processed text");
                        diagnostics.failures.push(StageFailure {
                            stage: Stage::Refiner,
                            index: 0,
                            reason: e.to_string(),
                        });
                        RefinementOutput::TextInsertion { text: text.clone(), formatting: None }
                    }
                },
                _ = cancel.cancelled() => {
                    sm.cancel().ok();
                    sm.reset();
                    return Err(PipelineError::Cancelled { during: "refinement" });
                }
            }
        }
    };

    // ── Emit (optional) ─────────────────────────────────────────────────
    transition(&mut sm, PipelineState::Emitting)?;
    let emit_result = match emitter {
        Some(emitter) => Some(emitter.emit(output.clone()).await),
        None => None,
    };

    transition(&mut sm, PipelineState::Idle)?;

    Ok(SessionResult {
        raw_text,
        output,
        emit_result,
        diagnostics,
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

        let session = pipeline.session();

        // Send one audio chunk then close the channel
        session
            .audio
            .send(AudioChunk {
                samples: vec![0.0; 160],
                sample_rate: 16000,
                channels: 1,
            })
            .await
            .unwrap();
        let result = session.finish().await.unwrap();
        assert_eq!(result.raw_text, "hello world");
        assert!(result.emit_result.as_ref().unwrap().success);

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

        let session = pipeline.session();
        session
            .audio
            .send(AudioChunk {
                samples: vec![0.0; 160],
                sample_rate: 16000,
                channels: 1,
            })
            .await
            .unwrap();
        let result = session.finish().await.unwrap();
        // Should fall back to raw text
        let buf = outputs.lock().await;
        match &buf[0] {
            RefinementOutput::TextInsertion { text, .. } => {
                assert_eq!(text, "raw dictation text");
            }
            _ => panic!("expected TextInsertion fallback"),
        }
        assert!(result.emit_result.as_ref().unwrap().success);
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

        let session = pipeline.session();

        // Send audio but cancel before closing the channel
        session
            .audio
            .send(AudioChunk {
                samples: vec![0.0; 160],
                sample_rate: 16000,
                channels: 1,
            })
            .await
            .unwrap();

        // Cancel while the session is still collecting audio.
        session.cancel.cancel();

        let err = session.finish().await.unwrap_err();
        assert!(
            matches!(err, PipelineError::Cancelled { .. }),
            "expected a cancellation, got: {err}"
        );
    }

    #[tokio::test]
    async fn build_without_asr_fails() {
        let Err(err) = Pipeline::builder().build() else {
            panic!("a pipeline with no ASR adapter must not build");
        };
        assert!(matches!(err, PipelineError::MissingComponent("asr")));
    }

    /// An ASR adapter is the only thing a pipeline cannot do without.
    /// Refiner, context provider and emitter are all optional, which is
    /// what makes the LLM-free configuration expressible at all.
    #[tokio::test]
    async fn build_requires_only_asr() {
        let pipeline = Pipeline::builder()
            .asr(MockAsr::new("hello world"))
            .build()
            .expect("an ASR adapter alone must be enough");

        let result = pipeline
            .transcribe(&[AudioChunk {
                samples: vec![0.0; 160],
                sample_rate: 16000,
                channels: 1,
            }])
            .await
            .expect("session must run without refiner, context or emitter");

        assert_eq!(result.text(), "hello world");
        assert!(
            result.emit_result.is_none(),
            "no emitter configured, so nothing should have been emitted"
        );
    }

    /// The configuration `docs/spec.md` §9.4 advertises as the minimal
    /// LLM-free setup. It did not compile before — `build()` demanded a
    /// refiner — so the headline example of the crate was unbuildable.
    /// This pins it.
    #[tokio::test]
    async fn spec_minimal_llm_free_pipeline_runs() {
        use crate::filter::SimpleFillerFilter;
        use crate::processor::{BasicPunctuationRestorer, SelfCorrectionDetector};

        let pipeline = Pipeline::builder()
            .asr(MockAsr::new("um so i think it works"))
            .filter(SimpleFillerFilter::english())
            .processor(SelfCorrectionDetector::new())
            .processor(BasicPunctuationRestorer)
            .build()
            .expect("the spec's minimal pipeline must build");

        let result = pipeline
            .transcribe(&[AudioChunk {
                samples: vec![0.0; 160],
                sample_rate: 16000,
                channels: 1,
            }])
            .await
            .expect("the spec's minimal pipeline must run");

        assert!(
            !result.text().is_empty(),
            "expected text out of the Tier 1+2 path"
        );
        assert!(
            result.diagnostics.removed.iter().any(|r| r.contains("um")),
            "the filler filter should have reported what it removed, got {:?}",
            result.diagnostics.removed
        );
        assert!(
            result.diagnostics.failures.is_empty(),
            "no stage should have failed: {:?}",
            result.diagnostics.failures
        );
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

        let session = pipeline.session();
        session
            .audio
            .send(AudioChunk {
                samples: vec![0.0; 160],
                sample_rate: 16000,
                channels: 1,
            })
            .await
            .unwrap();
        let result = session.finish().await.unwrap();
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
            .asr(MockAsr::new("um I want to go to Boston no wait to Denver"))
            .filter(SimpleFillerFilter::english())
            .processor(SelfCorrectionDetector::new())
            .processor(BasicPunctuationRestorer)
            .refiner(MockRefiner::passthrough())
            .context(MockContextProvider::new())
            .emitter(emitter)
            .build()
            .unwrap();

        let session = pipeline.session();
        session
            .audio
            .send(AudioChunk {
                samples: vec![0.0; 160],
                sample_rate: 16000,
                channels: 1,
            })
            .await
            .unwrap();
        let _result = session.finish().await.unwrap();
        let buf = outputs.lock().await;
        match &buf[0] {
            RefinementOutput::TextInsertion { text, .. } => {
                // Fillers removed, self-correction resolved, capitalized, period added
                assert!(!text.contains("um"), "filler should be removed: {text}");
                assert!(
                    !text.contains("Boston"),
                    "reparandum should be removed: {text}"
                );
                assert!(text.contains("Denver"), "repair should be kept: {text}");
                assert!(
                    text.starts_with(|c: char| c.is_uppercase()),
                    "should be capitalized: {text}"
                );
                assert!(text.ends_with('.'), "should have terminal period: {text}");
            }
            _ => panic!("expected TextInsertion"),
        }
    }
}
