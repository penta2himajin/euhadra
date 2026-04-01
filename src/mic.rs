use tokio::sync::mpsc;

use crate::types::AudioChunk;

/// Configuration for microphone capture.
#[derive(Debug, Clone)]
pub struct MicConfig {
    /// Target sample rate (whisper expects 16000).
    pub sample_rate: u32,
    /// Number of channels (whisper expects 1 = mono).
    pub channels: u16,
    /// Samples per chunk sent to the pipeline.
    pub chunk_size: usize,
}

impl Default for MicConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            chunk_size: 1600, // 100ms at 16kHz
        }
    }
}

/// Record audio from the default input device until the guard is dropped.
///
/// Returns an `mpsc::Receiver<AudioChunk>` that feeds into the pipeline,
/// and a guard whose drop stops recording.
///
/// # Example
///
/// ```no_run
/// use euhadra::mic::{MicConfig, record};
/// use euhadra::prelude::*;
///
/// # async fn example() {
/// let (audio_rx, stop) = record(MicConfig::default())
///     .expect("failed to open microphone");
///
/// // Feed into pipeline session
/// let pipeline = Pipeline::builder()
///     .asr(MockAsr::new("test"))
///     .refiner(MockRefiner::passthrough())
///     .context(MockContextProvider::new())
///     .emitter(StdoutEmitter)
///     .build()
///     .unwrap();
///
/// let (audio_tx, _cancel, handle) = pipeline.session();
///
/// // Bridge mic → pipeline
/// tokio::spawn(async move {
///     let mut rx = audio_rx;
///     while let Some(chunk) = rx.recv().await {
///         if audio_tx.send(chunk).await.is_err() { break; }
///     }
/// });
///
/// // When done, drop the guard to stop recording:
/// drop(stop);
/// # }
/// ```
pub fn record(
    config: MicConfig,
) -> Result<(mpsc::Receiver<AudioChunk>, MicStopGuard), MicError> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| MicError("no input device found".into()))?;

    tracing::info!(device = ?device.name().unwrap_or_default(), "using input device");

    let supported = device
        .supported_input_configs()
        .map_err(|e| MicError(format!("query configs: {e}")))?
        .find(|c| {
            c.channels() >= config.channels
                && c.min_sample_rate().0 <= config.sample_rate
                && c.max_sample_rate().0 >= config.sample_rate
        })
        .ok_or_else(|| MicError("no compatible audio config found".into()))?;

    let stream_config: cpal::StreamConfig = supported
        .with_sample_rate(cpal::SampleRate(config.sample_rate))
        .into();

    let (pipeline_tx, pipeline_rx) = mpsc::channel::<AudioChunk>(32);

    let (sync_tx, sync_rx) = std::sync::mpsc::channel::<Vec<f32>>();
    let chunk_size = config.chunk_size;
    let sample_rate = config.sample_rate;
    let channels = config.channels;
    let input_channels = stream_config.channels as usize;

    let stream = device
        .build_input_stream(
            &stream_config,
            move |data: &[f32], _info: &cpal::InputCallbackInfo| {
                let mono: Vec<f32> = if input_channels > 1 {
                    data.chunks(input_channels)
                        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
                        .collect()
                } else {
                    data.to_vec()
                };
                let _ = sync_tx.send(mono);
            },
            |err| {
                tracing::error!(error = %err, "audio input error");
            },
            None,
        )
        .map_err(|e| MicError(format!("build stream: {e}")))?;

    stream.play().map_err(|e| MicError(format!("start stream: {e}")))?;

    let _bridge = std::thread::spawn(move || {
        let mut buffer: Vec<f32> = Vec::with_capacity(chunk_size * 2);

        while let Ok(samples) = sync_rx.recv() {
            buffer.extend(samples);

            while buffer.len() >= chunk_size {
                let chunk_data: Vec<f32> = buffer.drain(..chunk_size).collect();
                let chunk = AudioChunk {
                    samples: chunk_data,
                    sample_rate,
                    channels,
                };
                if pipeline_tx.blocking_send(chunk).is_err() {
                    return;
                }
            }
        }

        if !buffer.is_empty() {
            let chunk = AudioChunk {
                samples: buffer,
                sample_rate,
                channels,
            };
            let _ = pipeline_tx.blocking_send(chunk);
        }
    });

    let guard = MicStopGuard {
        _stream: stream,
        _bridge: Some(_bridge),
    };

    Ok((pipeline_rx, guard))
}

// ---------------------------------------------------------------------------
// MicStopGuard — dropping stops the recording
// ---------------------------------------------------------------------------

/// Holds the audio stream alive.  Drop to stop recording.
pub struct MicStopGuard {
    _stream: cpal::Stream,
    _bridge: Option<std::thread::JoinHandle<()>>,
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MicError(pub String);

impl std::fmt::Display for MicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mic error: {}", self.0)
    }
}

impl std::error::Error for MicError {}
