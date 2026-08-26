use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::filter::TextFilter;
use crate::processor::{Correction, TextProcessor};
use crate::state::StateMachine;
use crate::traits::*;
use crate::types::*;
use crate::vad::{Segmenter, SegmenterConfig, SpeechSegment, VadBackend, VadStream};

// ---------------------------------------------------------------------------
// Pipeline configuration
// ---------------------------------------------------------------------------

/// Builds a configured pipeline from adapter implementations.
///
/// Only [`asr`](PipelineBuilder::asr) is required. Filters and processors
/// run in the order they are added; a refiner, context provider and emitter
/// are optional.
///
/// ```
/// use euhadra::prelude::*;
/// use euhadra::whisper_local::WhisperLocal;
///
/// # fn f() -> Result<(), PipelineError> {
/// let pipeline = PipelineBuilder::new()
///     .asr(WhisperLocal::new("whisper-cli", "ggml-base.bin"))
///     .filter(FillerFilter::for_language(Language::English))
///     .processor(SelfCorrectionDetector::new())
///     .processor(BasicPunctuationRestorer)
///     .emitter(StdoutEmitter)
///     .build()?;
/// # let _ = pipeline;
/// # Ok(())
/// # }
/// ```
///
/// This example is compiled as a doctest, so it cannot drift from the API
/// the way a snippet in a design document can.
pub struct PipelineBuilder {
    asr: Option<Arc<dyn AsrAdapter>>,
    filters: Vec<Arc<dyn TextFilter>>,
    processors: Vec<Arc<dyn TextProcessor>>,
    refiner: Option<Arc<dyn LlmRefiner>>,
    context: Option<Arc<dyn ContextProvider>>,
    emitter: Option<Arc<dyn OutputEmitter>>,
    vad: Option<Arc<dyn VadBackend>>,
    segmenter_config: SegmenterConfig,
    final_pass: FinalPass,
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
            vad: None,
            segmenter_config: SegmenterConfig::default(),
            final_pass: FinalPass::SpeechOnly,
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

    /// Detect speech before the audio reaches the ASR adapter.
    ///
    /// This buys two things. The recording's silences stop being fed to
    /// the model, which is where hallucinated text comes from; and the
    /// utterances found on the way become incremental output, delivered
    /// on [`Session::partials`] while the speaker is still talking.
    ///
    /// Placed here rather than inside microphone capture so that file
    /// input gets the same treatment — a WAV with 25 seconds of silence
    /// in it has the same problem as a live capture.
    ///
    /// ```
    /// use euhadra::prelude::*;
    /// use euhadra::vad::EnergyVad;
    ///
    /// # fn f() -> Result<(), PipelineError> {
    /// let pipeline = PipelineBuilder::new()
    ///     .asr(euhadra::whisper_local::WhisperLocal::new("whisper-cli", "m.bin"))
    ///     .vad(EnergyVad::new())
    ///     .build()?;
    /// # let _ = pipeline;
    /// # Ok(())
    /// # }
    /// ```
    pub fn vad(mut self, vad: impl VadBackend + 'static) -> Self {
        self.vad = Some(Arc::new(vad));
        self
    }

    /// Tune how per-frame detections become utterance boundaries.
    ///
    /// Has no effect without [`vad`](Self::vad). See
    /// [`SegmenterConfig`] for why the defaults lean towards waiting.
    pub fn segmenter_config(mut self, config: SegmenterConfig) -> Self {
        self.segmenter_config = config;
        self
    }

    /// Choose what the final transcript is computed from. Defaults to
    /// [`FinalPass::SpeechOnly`]. Has no effect without
    /// [`vad`](Self::vad).
    pub fn final_pass(mut self, policy: FinalPass) -> Self {
        self.final_pass = policy;
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
            vad: self.vad,
            segmenter_config: self.segmenter_config,
            final_pass: self.final_pass,
            audio_channel_size: self.audio_channel_size,
        })
    }
}

/// What the final transcript is computed from once voice activity
/// detection has found the utterances.
///
/// The two failure modes of segmentation are separable, and these
/// variants separate them. Feeding silence to the model produces
/// hallucinated text; cutting an utterance in half produces a fluent,
/// confident, wrong transcript of a fragment (#134 measured a 3-second
/// prefix yielding "However, due to the slow communication."). Dropping
/// the silence does not require cutting anything, so the default does the
/// first and not the second.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum FinalPass {
    /// Transcribe the detected speech, joined, as one utterance.
    ///
    /// The silences are gone, so the hallucination exposure is gone with
    /// them, but the ASR still sees each utterance whole and in context —
    /// a boundary placed slightly wrong costs a little padding rather
    /// than a fragment. The default.
    #[default]
    SpeechOnly,

    /// Transcribe the recording exactly as captured, silences included.
    ///
    /// Segmentation then affects only what arrives on
    /// [`Session::partials`]; the final text is bit-for-bit what a
    /// pipeline with no VAD would have produced. Use this when
    /// incremental output is wanted and no change to the result is
    /// acceptable — measuring one against the other, for instance.
    WholeUtterance,

    /// Join the per-utterance transcripts.
    ///
    /// One ASR pass over the audio instead of two, so it is the cheapest
    /// option, and the final text is exactly the concatenation of what
    /// the caller already saw. It is also the only one that inherits
    /// segmentation errors in full: a mis-cut utterance is transcribed as
    /// a fragment and that fragment is the answer.
    JoinSegments,
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
    vad: Option<Arc<dyn VadBackend>>,
    segmenter_config: SegmenterConfig,
    final_pass: FinalPass,
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
        let cancel = CancellationToken::new();

        let Some(vad) = self.vad.as_deref() else {
            return self.run(AsrSource::Audio(audio), Vec::new(), &cancel).await;
        };

        let samples = AudioChunk::concat(audio);
        let sample_rate = AudioChunk::sample_rate_of(audio).unwrap_or(0);
        let channels = audio.first().map(|c| c.channels).unwrap_or(1);
        let segments =
            match crate::vad::segment_buffer(vad, &samples, sample_rate, &self.segmenter_config) {
                Ok(segments) => segments,
                Err(e) => {
                    // Degrade rather than fail. Transcribing the
                    // recording with its silence in it is what a pipeline
                    // without a VAD does, and that is still a result; the
                    // failure is reported through
                    // [`Diagnostics::failures`] the same way a filter's
                    // is.
                    tracing::warn!(error = %e, "voice activity detection disabled");
                    return self
                        .run(AsrSource::Audio(audio), Vec::new(), &cancel)
                        .await
                        .map(|mut result| {
                            result.diagnostics.failures.push(StageFailure {
                                stage: Stage::Vad,
                                index: 0,
                                reason: e.to_string(),
                            });
                            result
                        });
                }
            };

        let source = match self.final_pass {
            FinalPass::WholeUtterance => AsrSource::Audio(audio),
            FinalPass::SpeechOnly => {
                AsrSource::Owned(speech_only(&samples, &segments, sample_rate, channels)?)
            }
            FinalPass::JoinSegments => {
                let mut texts = Vec::new();
                for segment in &segments {
                    let chunk = slice_chunk(&samples, *segment, sample_rate, channels);
                    match transcribe_segment(&self.asr, &chunk, &cancel).await? {
                        Some(text) => texts.push(text),
                        None => continue,
                    }
                }
                AsrSource::Text(texts.join(" "))
            }
        };
        self.run(source, segments, &cancel).await
    }

    #[allow(clippy::needless_lifetimes)]
    async fn run<'a>(
        &self,
        source: AsrSource<'a>,
        segments: Vec<SpeechSegment>,
        cancel: &CancellationToken,
    ) -> Result<SessionResult, PipelineError> {
        run_session(
            &self.asr,
            &self.filters,
            &self.processors,
            self.refiner.as_deref(),
            self.context.as_deref(),
            self.emitter.as_deref(),
            source,
            segments,
            cancel,
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
        let (partial_tx, partial_rx) = mpsc::channel::<Partial>(PARTIAL_CHANNEL_SIZE);
        // `(index, segment, pcm, closed_at)` — `closed_at` is when the
        // segmenter decided the utterance was processable (KPI t0).
        let (segment_tx, mut segment_rx) = mpsc::channel::<(
            usize,
            SpeechSegment,
            AudioChunk,
            std::time::Instant,
        )>(SEGMENT_CHANNEL_SIZE);
        let cancel = CancellationToken::new();

        let asr = Arc::clone(&self.asr);
        let filters: Vec<Arc<dyn TextFilter>> = self.filters.iter().map(Arc::clone).collect();
        let processors: Vec<Arc<dyn TextProcessor>> =
            self.processors.iter().map(Arc::clone).collect();
        let refiner = self.refiner.clone();
        let context = self.context.clone();
        let emitter = self.emitter.clone();
        let vad = self.vad.clone();
        let segmenter_config = self.segmenter_config.clone();
        let final_pass = self.final_pass;
        let cancel_inner = cancel.clone();

        // Kept behind so the collector can ask whether the caller still
        // wants partials. A clone tracks the *receiver*, so holding one
        // here does not keep the channel open.
        let partial_probe = partial_tx.clone();

        // Utterances are transcribed on their own task so that a slow
        // model cannot stall audio ingestion. Stalling it would push
        // backpressure onto the capture device, which has nowhere to put
        // the samples it is still producing.
        let asr_partials = Arc::clone(&asr);
        let cancel_partials = cancel.clone();
        let asr_task = tokio::spawn(async move {
            let mut texts: Vec<String> = Vec::new();
            let mut partial_latencies: Vec<std::time::Duration> = Vec::new();
            let mut last_closed_at: Option<std::time::Instant> = None;
            while let Some((index, segment, chunk, closed_at)) = segment_rx.recv().await {
                if cancel_partials.is_cancelled() {
                    break;
                }
                // t0 for endpoint_to_final is the last segment that became
                // processable, even if this utterance's ASR call fails.
                last_closed_at = Some(closed_at);
                let rate = chunk.sample_rate;
                match transcribe_segment(&asr_partials, &chunk, &cancel_partials).await {
                    Ok(Some(text)) => {
                        let start = segment.offset(rate);
                        // Lossy on purpose: a caller who is not draining
                        // partials must not be able to stall the session.
                        // Spec §4.3 makes the same choice for the ASR
                        // channel.
                        let endpoint_latency = closed_at.elapsed();
                        let _ = partial_tx.try_send(Partial {
                            index,
                            text: text.clone(),
                            start,
                            end: start + segment.duration(rate),
                            endpoint_latency: Some(endpoint_latency),
                        });
                        partial_latencies.push(endpoint_latency);
                        texts.push(text);
                    }
                    Ok(None) => {}
                    Err(e) => {
                        // A failed utterance is not a failed session: the
                        // final pass re-reads the audio. Under
                        // `JoinSegments` there is no final pass, so this
                        // is a hole in the transcript — hence the warning
                        // rather than a silent skip.
                        tracing::warn!(error = %e, index, "utterance transcription failed");
                    }
                }
            }
            (texts, partial_latencies, last_closed_at)
        });

        let handle = tokio::spawn(async move {
            // Collect the utterance as it is captured, so the caller can
            // keep sending while this task is already running.
            let mut chunks: Vec<AudioChunk> = Vec::new();
            let mut samples: Vec<f32> = Vec::new();
            let mut sample_rate = 0u32;
            let mut channels = 1u16;
            let mut live: Option<LiveSegmenter> = None;
            let mut segments: Vec<SpeechSegment> = Vec::new();
            let mut vad_active = vad.is_some();
            let mut vad_failure: Option<String> = None;

            // Per-utterance transcription costs a whole extra ASR pass.
            // Under `JoinSegments` it *is* the transcript so it always
            // runs; otherwise it only earns its keep if someone is
            // listening, and a dropped receiver says nobody is.
            let wanted = |probe: &mpsc::Sender<Partial>| {
                final_pass == FinalPass::JoinSegments || !probe.is_closed()
            };
            let mut cancelled = false;

            loop {
                tokio::select! {
                    // Biased so cancellation wins a tie. Without it, a
                    // token tripped just as the audio stream closes is a
                    // coin flip between "cancelled" and "here is your
                    // result" — and a caller who cancelled should never
                    // receive a result.
                    biased;
                    _ = cancel_inner.cancelled() => { cancelled = true; break; }
                    maybe = audio_rx.recv() => match maybe {
                        Some(chunk) => {
                            if sample_rate == 0 {
                                sample_rate = chunk.sample_rate;
                                channels = chunk.channels;
                            }
                            samples.extend_from_slice(&chunk.samples);
                            chunks.push(chunk);

                            if vad_active {
                                if live.is_none() {
                                    let backend = vad.as_deref().expect("vad_active implies a backend");
                                    match LiveSegmenter::new(backend, sample_rate, segmenter_config.clone()) {
                                        Ok(segmenter) => live = Some(segmenter),
                                        Err(e) => {
                                            // Degrade rather than fail: the
                                            // whole-recording path still
                                            // works, it just keeps the
                                            // silence.
                                            tracing::warn!(error = %e, "voice activity detection disabled");
                                            vad_failure = Some(e.to_string());
                                            vad_active = false;
                                        }
                                    }
                                }
                                if let Some(segmenter) = live.as_mut() {
                                    for segment in segmenter.advance(&samples) {
                                        let index = segments.len();
                                        let closed_at = std::time::Instant::now();
                                        segments.push(segment);
                                        if wanted(&partial_probe) {
                                            let chunk = slice_chunk(
                                                &samples,
                                                segment,
                                                sample_rate,
                                                channels,
                                            );
                                            let _ = segment_tx
                                                .send((index, segment, chunk, closed_at))
                                                .await;
                                        }
                                    }
                                }
                            }
                        }
                        None => break,
                    },
                }
            }

            if !cancelled {
                if let Some(segmenter) = live.as_mut() {
                    for segment in segmenter.flush(&samples) {
                        let index = segments.len();
                        let closed_at = std::time::Instant::now();
                        segments.push(segment);
                        if wanted(&partial_probe) {
                            let chunk =
                                slice_chunk(&samples, segment, sample_rate, channels);
                            let _ = segment_tx
                                .send((index, segment, chunk, closed_at))
                                .await;
                        }
                    }
                }
            }

            // Closing the channel is what ends the utterance task.
            drop(segment_tx);
            drop(partial_probe);
            let (segment_texts, partial_latencies, last_closed_at) =
                asr_task.await.unwrap_or_default();

            if cancelled {
                return Err(PipelineError::Cancelled { during: "recording" });
            }

            let source = if !vad_active {
                AsrSource::Owned(chunks)
            } else {
                match final_pass {
                    FinalPass::WholeUtterance => AsrSource::Owned(chunks),
                    FinalPass::SpeechOnly => {
                        AsrSource::Owned(speech_only(&samples, &segments, sample_rate, channels)?)
                    }
                    FinalPass::JoinSegments => AsrSource::Text(segment_texts.join(" ")),
                }
            };

            run_session(
                &asr,
                &filters,
                &processors,
                refiner.as_deref(),
                context.as_deref(),
                emitter.as_deref(),
                source,
                segments,
                &cancel_inner,
            )
            .await
            .map(|mut result| {
                result.diagnostics.endpoint_to_partial = partial_latencies;
                if let Some(t0) = last_closed_at {
                    result.diagnostics.endpoint_to_final = Some(t0.elapsed());
                }
                if let Some(reason) = vad_failure {
                    result.diagnostics.failures.push(StageFailure {
                        stage: Stage::Vad,
                        index: 0,
                        reason,
                    });
                }
                result
            })
        });

        Session {
            audio: audio_tx,
            partials: partial_rx,
            cancel,
            handle,
        }
    }
}

/// How many utterance transcripts are buffered for a caller that is not
/// keeping up. Small on purpose — the point of a partial is that it is
/// current, and an old one is not worth holding a newer one back for.
const PARTIAL_CHANNEL_SIZE: usize = 8;

/// Utterances queued for transcription. Bounded so that a model slower
/// than real time applies backpressure to segmentation rather than
/// growing a queue without limit.
const SEGMENT_CHANNEL_SIZE: usize = 4;

/// Where a session's transcript comes from.
enum AsrSource<'a> {
    /// Borrowed audio — the caller already owns the buffer.
    Audio(&'a [AudioChunk]),
    /// Audio assembled by the pipeline, e.g. the speech cut out of a
    /// recording.
    Owned(Vec<AudioChunk>),
    /// Text already transcribed, utterance by utterance.
    Text(String),
}

/// A transcript for one utterance, delivered while the session is still
/// running.
///
/// Advisory unless the pipeline is set to [`FinalPass::JoinSegments`]:
/// the value in [`SessionResult`] is computed separately and is what the
/// session actually produced. Use these to show the speaker what has been
/// heard so far, not to build the final text.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Partial {
    /// Position in capture order, starting at zero.
    pub index: usize,
    /// What the ASR adapter made of this utterance alone.
    pub text: String,
    /// Where the utterance starts within the recording.
    pub start: std::time::Duration,
    /// Where it ends.
    pub end: std::time::Duration,
    /// Wall time from when the segment became processable (the
    /// segmenter closed it) until this partial was ready to send.
    ///
    /// This is the per-utterance half of the endpoint-latency KPI in
    /// #148 — not the L1 full-file ASR wall clock. `None` only if the
    /// timing path was skipped (should not happen on the live VAD
    /// path that emits partials).
    pub endpoint_latency: Option<std::time::Duration>,
}

/// Runs a [`Segmenter`] over audio that is still arriving.
struct LiveSegmenter {
    segmenter: Segmenter,
    stream: Box<dyn VadStream>,
    frame_size: usize,
    /// Samples already scored. Frames are whole and never rescored, so
    /// this only moves forward.
    consumed: usize,
}

impl LiveSegmenter {
    fn new(
        backend: &dyn VadBackend,
        sample_rate: u32,
        config: SegmenterConfig,
    ) -> Result<Self, crate::vad::VadError> {
        Ok(Self {
            segmenter: Segmenter::new(backend, sample_rate, config)?,
            stream: backend.start(),
            frame_size: backend.frame_size().max(1),
            consumed: 0,
        })
    }

    /// Score every whole frame that has arrived since the last call.
    fn advance(&mut self, samples: &[f32]) -> Vec<SpeechSegment> {
        let mut closed = Vec::new();
        while samples.len() - self.consumed >= self.frame_size {
            let frame = &samples[self.consumed..self.consumed + self.frame_size];
            let probability = self.stream.speech_probability(frame);
            self.consumed += self.frame_size;
            if let Some(segment) = self.segmenter.push(probability) {
                closed.push(segment);
            }
        }
        closed
    }

    /// Score the trailing partial frame and close anything still open.
    fn flush(&mut self, samples: &[f32]) -> Vec<SpeechSegment> {
        let mut closed = self.advance(samples);
        if samples.len() > self.consumed {
            let mut frame = samples[self.consumed..].to_vec();
            frame.resize(self.frame_size, 0.0);
            let probability = self.stream.speech_probability(&frame);
            self.consumed = samples.len();
            if let Some(segment) = self.segmenter.push(probability) {
                closed.push(segment);
            }
        }
        closed.extend(self.segmenter.flush());
        closed
    }
}

/// Cut one segment out of a recording.
///
/// Bounds are clamped: left-side [`SegmenterConfig::preroll`] /
/// [`SegmenterConfig::speech_pad`] can ask for samples before what has
/// been captured, and the trailing pad can reach past the end when it is
/// wider than the silence that closed the utterance. The segmenter
/// already refuses to start before the previous segment's end; this
/// clamp covers the buffer edges.
fn slice_chunk(
    samples: &[f32],
    segment: SpeechSegment,
    sample_rate: u32,
    channels: u16,
) -> AudioChunk {
    let start = segment.start.min(samples.len());
    let end = segment.end.clamp(start, samples.len());
    AudioChunk {
        samples: samples[start..end].to_vec(),
        sample_rate,
        channels,
    }
}

/// Join the detected speech into one recording, dropping the silence.
///
/// Overlapping segments are trimmed rather than duplicated — padding
/// wider than half the minimum silence can make two neighbours meet.
fn speech_only(
    samples: &[f32],
    segments: &[SpeechSegment],
    sample_rate: u32,
    channels: u16,
) -> Result<Vec<AudioChunk>, PipelineError> {
    let mut speech: Vec<f32> = Vec::new();
    let mut cursor = 0usize;
    for segment in segments {
        let start = segment.start.max(cursor).min(samples.len());
        let end = segment.end.clamp(start, samples.len());
        speech.extend_from_slice(&samples[start..end]);
        cursor = end;
    }
    if speech.is_empty() {
        return Err(PipelineError::NoSpeech);
    }
    Ok(vec![AudioChunk {
        samples: speech,
        sample_rate,
        channels,
    }])
}

/// Transcribe one utterance, returning `None` when the adapter found
/// nothing in it.
async fn transcribe_segment(
    asr: &Arc<dyn AsrAdapter>,
    chunk: &AudioChunk,
    cancel: &CancellationToken,
) -> Result<Option<String>, PipelineError> {
    if chunk.samples.is_empty() {
        return Ok(None);
    }
    let transcript = tokio::select! {
        biased;
        _ = cancel.cancelled() => return Err(PipelineError::Cancelled { during: "recording" }),
        result = asr.transcribe(std::slice::from_ref(chunk)) => result?,
    };
    let text = transcript.text.trim().to_string();
    Ok(if text.is_empty() { None } else { Some(text) })
}

/// A live dictation session.
///
/// The audio stream is closed by [`finish`](Self::finish), which is also
/// what awaits the result — so there is no way to await a session whose
/// input is still open, and no sender to remember to drop.
pub struct Session {
    /// Send captured audio here.
    pub audio: mpsc::Sender<AudioChunk>,
    /// Transcripts for utterances the detector has already closed,
    /// delivered while the speaker is still talking.
    ///
    /// Empty unless the pipeline was given a
    /// [`vad`](PipelineBuilder::vad). Lossy: if the queue fills, the
    /// newest partial is dropped rather than held, so ignoring this
    /// receiver cannot stall a session. Dropping it entirely also tells
    /// the pipeline to stop transcribing utterances separately, which
    /// saves an ASR pass under every [`FinalPass`] except
    /// [`JoinSegments`](FinalPass::JoinSegments).
    ///
    /// ```no_run
    /// # use euhadra::prelude::*;
    /// # async fn f(pipeline: Pipeline) -> Result<(), PipelineError> {
    /// let mut session = pipeline.session();
    /// let mut partials = std::mem::replace(
    ///     &mut session.partials,
    ///     tokio::sync::mpsc::channel(1).1,
    /// );
    /// tokio::spawn(async move {
    ///     while let Some(p) = partials.recv().await {
    ///         println!("[{:?}] {}", p.start, p.text);
    ///     }
    /// });
    /// let result = session.finish().await?;
    /// # let _ = result;
    /// # Ok(()) }
    /// ```
    pub partials: mpsc::Receiver<Partial>,
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
    /// The utterances voice activity detection found, in capture order.
    /// Empty when the pipeline has no [`vad`](PipelineBuilder::vad).
    ///
    /// Segmentation is the part of this pipeline most likely to be wrong
    /// in a way the text does not reveal, so what it decided is reported
    /// rather than left to be inferred from the transcript.
    pub speech_segments: Vec<SpeechSegment>,
    /// Per closed segment that produced a partial: time from segment
    /// close until that partial was ready. Same order as successful
    /// partials (not necessarily 1:1 with [`speech_segments`] when an
    /// utterance ASR call fails).
    pub endpoint_to_partial: Vec<std::time::Duration>,
    /// Time from the **last** successful segment close until the
    /// [`SessionResult`] was ready (FinalPass included). `None` when no
    /// segment closed, or when partials were not requested so per-
    /// utterance ASR never ran (timing then has no t0).
    pub endpoint_to_final: Option<std::time::Duration>,
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
    /// Voice activity detection, ahead of the ASR adapter. A failure
    /// here means the recording was transcribed with its silences
    /// intact.
    Vad,
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
    source: AsrSource<'_>,
    speech_segments: Vec<SpeechSegment>,
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
    // Already-transcribed text arrives here under
    // `FinalPass::JoinSegments`, where the utterances were transcribed
    // one at a time as they closed. Everything downstream is identical
    // either way.
    let raw_text = match &source {
        AsrSource::Text(text) => text.trim().to_string(),
        AsrSource::Audio(_) | AsrSource::Owned(_) => {
            let audio: &[AudioChunk] = match &source {
                AsrSource::Audio(audio) => audio,
                AsrSource::Owned(audio) => audio,
                AsrSource::Text(_) => unreachable!("matched above"),
            };
            let transcript = tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    sm.cancel().ok();
                    sm.reset();
                    return Err(PipelineError::Cancelled { during: "recording" });
                }
                result = asr.transcribe(audio) => result?,
            };
            transcript.text.trim().to_string()
        }
    };

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
            biased;
            _ = cancel.cancelled() => {
                sm.cancel().ok();
                sm.reset();
                return Err(PipelineError::Cancelled { during: "context" });
            }
            snapshot = provider.get_context() => snapshot,
        },
        None => ContextSnapshot::default(),
    };

    let mut diagnostics = Diagnostics {
        speech_segments,
        ..Diagnostics::default()
    };

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
                biased;
                _ = cancel.cancelled() => {
                    sm.cancel().ok();
                    sm.reset();
                    return Err(PipelineError::Cancelled { during: "refinement" });
                }
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
