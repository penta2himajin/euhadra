use super::{VadBackend, VadStream};

/// Frame length in samples. 10 ms at 16 kHz, which is the rate every
/// bundled ASR adapter wants; at other rates the frame is a different
/// duration, but [`Segmenter`](super::Segmenter) converts its durations
/// using the real sample rate, so the timing of a boundary does not move.
const FRAME_SIZE: usize = 160;

/// Level below which a frame is treated as digital silence, in dBFS.
/// Keeps `log10(0)` out of the arithmetic.
const FLOOR_DB: f32 = -100.0;

/// Detects speech by level against an adapting noise floor.
///
/// **A stopgap until a neural detector is wired in.** It answers "is this
/// louder than the room?", which is not the same question as "is this
/// speech": a keyboard, a door, or music all pass, and a whisper may not.
/// That is the accepted cost of a detector with no model, no download and
/// no `onnx` feature — the alternative was shipping nothing in the default
/// build and leaving every consumer to feed silence to their ASR.
///
/// The segmentation policy lives in [`Segmenter`](super::Segmenter), not
/// here, so replacing this with Silero changes which frames score as
/// speech and nothing else.
///
/// # How it decides
///
/// Two independent tests, and a frame passing either scores as speech:
///
/// - **Relative** — is it [`with_margin_db`](Self::with_margin_db) above
///   an adapting estimate of the room? This is what works in a quiet
///   room with a soft speaker.
/// - **Absolute** — is it above
///   [`with_speech_floor_db`](Self::with_speech_floor_db)? This is what
///   keeps a long unbroken utterance alive.
///
/// The absolute test is not redundant. The noise floor has to rise to
/// meet sustained noise or a noisy room reads as one continuous
/// utterance — but a rising floor also chases sustained *speech* and
/// eventually mutes it. Letting a loud frame bypass the floor entirely is
/// what separates the two cases, and level is the only thing available to
/// separate them with: to a level detector, steady room tone and steady
/// speech are the same signal at different volumes.
///
/// The consequence is the known limit of this backend: speech quieter
/// than the absolute floor, sustained for longer than the floor's few
/// seconds of memory, fades out. A neural detector does not have this
/// failure because it is not deciding on level.
#[derive(Debug, Clone)]
pub struct EnergyVad {
    margin_db: f32,
    ramp_db: f32,
    speech_floor_db: f32,
}

impl EnergyVad {
    /// A detector with the default 10 dB margin over the noise floor.
    pub fn new() -> Self {
        Self {
            margin_db: 10.0,
            ramp_db: 6.0,
            speech_floor_db: -38.0,
        }
    }

    /// How far above the noise floor a frame must sit to score 0.5.
    ///
    /// Raise it in a noisy room to stop the fan opening utterances; lower
    /// it for a quiet room and a soft speaker. 10 dB is roughly the gap
    /// between conversational speech and typical room tone.
    pub fn with_margin_db(mut self, margin_db: f32) -> Self {
        self.margin_db = margin_db;
        self
    }

    /// The width of the ramp around either threshold, in dB. A frame
    /// `ramp_db` above scores 1.0, `ramp_db` below scores 0.0.
    pub fn with_ramp_db(mut self, ramp_db: f32) -> Self {
        self.ramp_db = ramp_db.max(f32::EPSILON);
        self
    }

    /// The level, in dBFS, at which a frame scores as speech regardless
    /// of the noise floor.
    ///
    /// Lower it for a quiet microphone; raise it in a loud environment
    /// where the noise itself clears the default. Set it very low and a
    /// long utterance survives but loud noise never settles; set it very
    /// high and only the relative test is left, which fades out sustained
    /// speech.
    pub fn with_speech_floor_db(mut self, speech_floor_db: f32) -> Self {
        self.speech_floor_db = speech_floor_db;
        self
    }
}

impl Default for EnergyVad {
    fn default() -> Self {
        Self::new()
    }
}

impl VadBackend for EnergyVad {
    fn frame_size(&self) -> usize {
        FRAME_SIZE
    }

    fn required_sample_rate(&self) -> Option<u32> {
        // Level thresholding is rate-agnostic; nothing here was trained
        // at a particular rate.
        None
    }

    fn start(&self) -> Box<dyn VadStream> {
        Box::new(EnergyStream {
            margin_db: self.margin_db,
            ramp_db: self.ramp_db,
            speech_floor_db: self.speech_floor_db,
            // Starts pessimistic — a floor guessed too high would miss
            // the first words while it settled, and it falls fast.
            noise_floor_db: -60.0,
        })
    }
}

struct EnergyStream {
    margin_db: f32,
    ramp_db: f32,
    speech_floor_db: f32,
    noise_floor_db: f32,
}

/// Map "how many dB above the threshold" onto `0.0..=1.0`, hitting 0.5
/// exactly at the threshold so that a caller's
/// [`SegmenterConfig::threshold`](super::SegmenterConfig::threshold) of
/// 0.5 means precisely the threshold this backend computed.
fn ramp(excess_db: f32, ramp_db: f32) -> f32 {
    (0.5 + 0.5 * excess_db / ramp_db).clamp(0.0, 1.0)
}

impl VadStream for EnergyStream {
    fn speech_probability(&mut self, frame: &[f32]) -> f32 {
        let db = rms_db(frame);

        let relative = ramp(db - (self.noise_floor_db + self.margin_db), self.ramp_db);
        let absolute = ramp(db - self.speech_floor_db, self.ramp_db);

        // Asymmetric tracking. Down fast, so a recording that opens on a
        // door slam settles within a few frames. Up slowly — but
        // unconditionally, not only on frames judged quiet: gating the
        // rise on the verdict is circular, and noise that starts above
        // the initial guess would hold itself above it forever.
        if db < self.noise_floor_db {
            self.noise_floor_db += (db - self.noise_floor_db) * 0.5;
        } else {
            self.noise_floor_db += (db - self.noise_floor_db) * 0.005;
        }

        relative.max(absolute)
    }
}

fn rms_db(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return FLOOR_DB;
    }
    let sum_sq: f32 = frame.iter().map(|s| s * s).sum();
    let rms = (sum_sq / frame.len() as f32).sqrt();
    if rms <= 0.0 {
        return FLOOR_DB;
    }
    (20.0 * rms.log10()).max(FLOOR_DB)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tone at `amplitude`. RMS is `amplitude / sqrt(2)`, so 0.4 is
    /// about −11 dBFS (speech) and 0.005 about −49 dBFS (room tone).
    fn tone(n: usize, amplitude: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.3).sin() * amplitude).collect()
    }

    fn drive(backend: &EnergyVad, frames: &[Vec<f32>]) -> Vec<f32> {
        let mut stream = backend.start();
        frames.iter().map(|f| stream.speech_probability(f)).collect()
    }

    #[test]
    fn digital_silence_never_scores_as_speech() {
        let frames = vec![vec![0.0f32; FRAME_SIZE]; 200];
        let probs = drive(&EnergyVad::new(), &frames);
        assert!(
            probs.iter().all(|p| *p < 0.5),
            "silence scored as speech: max {:?}",
            probs.iter().cloned().fold(0.0f32, f32::max)
        );
    }

    #[test]
    fn a_loud_tone_scores_as_speech() {
        let frames = vec![tone(FRAME_SIZE, 0.4); 50];
        let probs = drive(&EnergyVad::new(), &frames);
        assert!(
            probs.iter().all(|p| *p >= 0.5),
            "a tone 90 dB over digital silence should score as speech, got {probs:?}"
        );
    }

    /// The floor has to adapt, or a recording made in a noisy room is one
    /// continuous utterance. 10 s of room tone is well past the floor's
    /// time constant.
    #[test]
    fn steady_background_noise_stops_scoring_as_speech() {
        // −49 dBFS: above the initial −60 dB guess, so the first frames
        // do read as speech until the floor catches up. That the floor
        // rises at all is the property under test — an earlier version
        // only raised it on frames already judged quiet, so noise that
        // started above the guess held itself above it indefinitely.
        let frames = vec![tone(FRAME_SIZE, 0.005); 1000];
        let probs = drive(&EnergyVad::new(), &frames);
        assert!(
            probs.last().copied().unwrap() < 0.5,
            "the floor should have risen to meet steady noise, ended at {:?}",
            probs.last()
        );
    }

    /// And having adapted, it must still hear speech over that noise.
    #[test]
    fn speech_is_detected_over_adapted_background_noise() {
        let mut frames = vec![tone(FRAME_SIZE, 0.005); 1000];
        frames.extend(vec![tone(FRAME_SIZE, 0.4); 20]);
        let probs = drive(&EnergyVad::new(), &frames);
        assert!(
            probs[1000..].iter().all(|p| *p >= 0.5),
            "speech 38 dB over the settled noise floor must score as speech, got {:?}",
            &probs[1000..]
        );
    }

    /// Sustained speech must not drag the floor up until it silences
    /// itself. This is what the absolute floor is for; without it the
    /// unconditional rise above would mute a long utterance.
    #[test]
    fn sustained_speech_does_not_mute_itself() {
        let frames = vec![tone(FRAME_SIZE, 0.4); 3000]; // 30 s at 16 kHz
        let probs = drive(&EnergyVad::new(), &frames);
        assert!(
            probs.last().copied().unwrap() >= 0.5,
            "30 s of continuous speech ended at {:?}",
            probs.last()
        );
    }

    /// The documented limit, pinned so it is a known property rather than
    /// a surprise: speech below the absolute floor fades once the noise
    /// floor climbs to meet it.
    #[test]
    fn speech_below_the_absolute_floor_eventually_fades() {
        let frames = vec![tone(FRAME_SIZE, 0.005); 2000];
        let probs = drive(&EnergyVad::new().with_speech_floor_db(-20.0), &frames);
        assert!(
            probs[0] >= 0.5 && probs.last().copied().unwrap() < 0.5,
            "expected a quiet steady signal to start as speech and fade; \
             got {:?} → {:?}",
            probs[0],
            probs.last()
        );
    }

    #[test]
    fn a_wider_margin_demands_a_louder_frame() {
        let frames = vec![tone(FRAME_SIZE, 0.005); 200];
        let strict = drive(&EnergyVad::new().with_margin_db(40.0), &frames);
        let lenient = drive(&EnergyVad::new().with_margin_db(3.0), &frames);
        assert!(
            strict.iter().sum::<f32>() < lenient.iter().sum::<f32>(),
            "a 40 dB margin should score fewer frames as speech than a 3 dB one"
        );
    }

    #[test]
    fn rms_db_handles_empty_and_zero_frames() {
        assert_eq!(rms_db(&[]), FLOOR_DB);
        assert_eq!(rms_db(&[0.0; 8]), FLOOR_DB);
        assert!(rms_db(&[1.0; 8]) > -0.001, "full scale should read ~0 dBFS");
    }
}
