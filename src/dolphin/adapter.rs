//! DataoceanAI Dolphin CTC ASR adapter.
//!
//! ```text
//! audio f32  →  Kaldi FBANK (80 mel, Povey window, snip_edges=false)
//!            →  per-bin CMVN from the graph's own metadata_props
//!            →  ONNX (x [1,T,80], x_len [1])  →  log_probs [1,T',V]
//!            →  argmax → unique_consecutive → drop blank
//!            →  tokens.txt lookup → strip <...> control tags
//!            →  text
//! ```
//!
//! Selected for the Korean routing path in
//! `docs/korean-asr-alternatives.md` §I: CTC is non-autoregressive, so
//! cost is proportional to audio rather than fixed at Whisper's padded
//! 30-second encoder pass, and it does not exhibit the INT8 repetition
//! collapse §H.2 found in every autoregressive decoder measured here.
//!
//! Dolphin's code and weights are Apache-2.0
//! (<https://github.com/DataoceanAI/Dolphin>); this is a re-implementation
//! of the inference path and ships no weights —
//! `scripts/setup_dolphin_ko.sh` fetches the k2-fsa CTC export.

use async_trait::async_trait;
use ndarray::{Array1, Array3};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::traits::{AsrAdapter, AsrError};
use crate::types::{AudioChunk, Transcript};

use crate::paraformer::fbank::{Fbank, FbankOpts};
use crate::sensevoice::vocab::ctc_collapse;

use super::metadata::Cmvn;
use super::vocab::{decode, load_tokens, BLANK_ID};

const ONNX_INPUT_X: &str = "x";
const ONNX_INPUT_X_LEN: &str = "x_len";
const META_MEAN: &str = "mean";
const META_INVSTD: &str = "invstd";
const META_MODEL_TYPE: &str = "model_type";

/// Intra-op threads.
///
/// **One, deliberately.** `docs/korean-asr-alternatives.md` §I.1
/// measured five runs of this model at four threads producing five
/// different transcripts from a byte-identical file, while one thread
/// reproduced exactly — and scored better, CER 0.0655 against
/// 0.0818–0.0926. A dictation backend that returns a different
/// transcript each time it hears the same audio is not usable, so this
/// is a correctness setting rather than a performance one. Raising it
/// requires re-running that experiment, not just a benchmark.
const INTRA_THREADS: usize = 1;

#[derive(Debug, Clone)]
pub struct DolphinConfig {
    /// Model file relative to the bundle directory.
    pub model_file: String,
}

impl Default for DolphinConfig {
    fn default() -> Self {
        Self {
            model_file: "model.int8.onnx".into(),
        }
    }
}

pub struct DolphinAdapter {
    session: Mutex<Session>,
    fbank: Fbank,
    cmvn: Cmvn,
    vocab: Vec<String>,
    input_x: String,
    input_x_len: String,
    model_path: PathBuf,
}

impl DolphinAdapter {
    /// Load a bundle laid out by `scripts/setup_dolphin_ko.sh`:
    /// `model.int8.onnx` + `tokens.txt`.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, AsrError> {
        Self::load_with_config(model_dir, DolphinConfig::default())
    }

    pub fn load_with_config(
        model_dir: impl AsRef<Path>,
        cfg: DolphinConfig,
    ) -> Result<Self, AsrError> {
        let dir = model_dir.as_ref();
        let model_path = dir.join(&cfg.model_file);
        let tokens_path = dir.join("tokens.txt");

        let vocab = load_tokens(&tokens_path)?;

        let session = Session::builder()
            .map_err(|e| AsrError::ModelLoad(format!("dolphin session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| AsrError::Inference(format!("dolphin optimization level: {e}")))?
            .with_intra_threads(INTRA_THREADS)
            .map_err(|e| AsrError::Inference(format!("dolphin with_intra_threads({INTRA_THREADS}): {e}")))?
            .with_inter_threads(1)
            .map_err(|e| AsrError::Inference(format!("dolphin with_inter_threads(1): {e}")))?
            .commit_from_file(&model_path)
            .map_err(|e| AsrError::ModelLoad(format!("dolphin load failed at {}: {e}", model_path.display())))?;

        let (input_x, input_x_len) = resolve_input_names(&session, &model_path)?;
        let cmvn = read_cmvn(&session, &model_path)?;

        let fbank = Fbank::new(FbankOpts::dolphin_default());

        // Fail at load rather than at the first utterance. Sizing the
        // front-end *from* the CMVN instead would make this check
        // vacuous and quietly feed the model the wrong band layout.
        if cmvn.dim() != fbank.n_mels() {
            return Err(AsrError::ModelLoad(format!(
                    "dolphin {}: CMVN covers {} bins but the front-end produces {}",
                    model_path.display(),
                    cmvn.dim(),
                    fbank.n_mels()
                )));
        }

        Ok(Self {
            session: Mutex::new(session),
            fbank,
            cmvn,
            vocab,
            input_x,
            input_x_len,
            model_path,
        })
    }

    /// Number of vocabulary entries, for diagnostics.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn transcribe_samples(&self, samples: &[f32]) -> Result<String, AsrError> {
        let (feats, n_frames) = prepare_features(&self.fbank, &self.cmvn, samples)?;

        let dim = self.cmvn.dim();
        let x: Array3<f32> =
            Array3::from_shape_vec((1, n_frames, dim), feats).map_err(|e| AsrError::Inference(format!("dolphin x tensor shape: {e}")))?;
        // x_len is int64 here, not the int32 SenseVoice takes.
        let x_len: Array1<i64> = Array1::from(vec![n_frames as i64]);

        let x_val = Value::from_array(x).map_err(|e| AsrError::Inference(format!("dolphin x Value: {e}")))?;
        let x_len_val = Value::from_array(x_len).map_err(|e| AsrError::Inference(format!("dolphin x_len Value: {e}")))?;

        let mut session = self.session.lock().map_err(|e| AsrError::Inference(format!("dolphin session lock poisoned: {e}")))?;
        let outputs = session
            .run(vec![
                (self.input_x.as_str(), x_val.into_dyn()),
                (self.input_x_len.as_str(), x_len_val.into_dyn()),
            ])
            .map_err(|e| AsrError::Inference(format!("dolphin ONNX run: {e}")))?;

        let log_probs = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| AsrError::Inference(format!("dolphin extract log_probs: {e}")))?;
        let view = log_probs.view();
        let shape = view.shape().to_vec();
        if shape.len() != 3 {
            return Err(AsrError::Inference(format!(
                    "dolphin {}: unexpected log_probs rank {} (shape {shape:?})",
                    self.model_path.display(),
                    shape.len()
                )));
        }
        let (t, v) = (shape[1], shape[2]);
        if v != self.vocab.len() {
            return Err(AsrError::ModelLoad(format!(
                    "dolphin {}: model emits {v} classes but tokens.txt has {}",
                    self.model_path.display(),
                    self.vocab.len()
                )));
        }

        let mut ids = Vec::with_capacity(t);
        for ti in 0..t {
            let mut best = 0u32;
            let mut best_v = f32::NEG_INFINITY;
            for vi in 0..v {
                let val = view[[0, ti, vi]];
                if val > best_v {
                    best_v = val;
                    best = vi as u32;
                }
            }
            ids.push(best);
        }

        let collapsed = ctc_collapse(&ids, BLANK_ID);
        let tokens: Vec<String> = collapsed
            .iter()
            .filter_map(|id| self.vocab.get(*id as usize).cloned())
            .collect();
        Ok(decode(&tokens))
    }
}

/// Everything between the samples and the ONNX call.
///
/// Split out so the front-end can be exercised without the weights —
/// the session is the only part of this adapter that needs a 239 MB
/// download, and the preconditions here are the ones a caller trips.
fn prepare_features(
    fbank: &Fbank,
    cmvn: &Cmvn,
    samples: &[f32],
) -> Result<(Vec<f32>, usize), AsrError> {
    if samples.is_empty() {
        return Err(AsrError::NoAudio);
    }

    let (mut feats, n_frames) = fbank.compute(samples);
    if n_frames == 0 {
        return Err(AsrError::Inference(format!(
                "audio too short for one FBANK frame ({} samples)",
                samples.len()
            )));
    }
    // 2000 features is a whole number of 40-wide frames as well as of
    // 80-wide ones, so `Cmvn::apply`'s divisibility check cannot catch
    // a half-width CMVN on its own.
    if cmvn.dim() != fbank.n_mels() {
        return Err(AsrError::ModelLoad(format!(
                "dolphin CMVN covers {} bins but the front-end produces {}",
                cmvn.dim(),
                fbank.n_mels()
            )));
    }
    cmvn.apply(&mut feats)?;
    Ok((feats, n_frames))
}

fn resolve_input_names(session: &Session, path: &Path) -> Result<(String, String), AsrError> {
    let names: Vec<String> = session
        .inputs()
        .iter()
        .map(|i| i.name().to_string())
        .collect();
    let find = |needle: &str| -> Result<String, AsrError> {
        names
            .iter()
            .find(|n| n.as_str() == needle)
            .cloned()
            .ok_or_else(|| AsrError::Inference(format!(
                    "dolphin ONNX {} missing input {needle:?} (have: {names:?})",
                    path.display()
                )))
    };
    Ok((find(ONNX_INPUT_X)?, find(ONNX_INPUT_X_LEN)?))
}

fn read_cmvn(session: &Session, path: &Path) -> Result<Cmvn, AsrError> {
    let meta = session.metadata().map_err(|e| AsrError::Inference(format!("dolphin metadata {}: {e}", path.display())))?;

    // Not fatal, but a bundle that does not say "dolphin-ctc" is very
    // likely the wrong file for this adapter.
    match meta.custom(META_MODEL_TYPE) {
        Some(t) if t == "dolphin-ctc" => {}
        other => tracing::warn!(
            model = %path.display(),
            model_type = ?other,
            "expected model_type=dolphin-ctc"
        ),
    }

    let get = |key: &str| -> Result<String, AsrError> {
        meta.custom(key).ok_or_else(|| AsrError::ModelLoad(format!(
                "dolphin {}: graph metadata has no {key:?}; this adapter reads \
                 the CMVN from the model rather than a sidecar",
                path.display()
            )))
    };
    Cmvn::parse(&get(META_MEAN)?, &get(META_INVSTD)?)
}

#[async_trait]
impl AsrAdapter for DolphinAdapter {
    async fn transcribe(&self, audio: &[AudioChunk]) -> Result<Transcript, AsrError> {
        let all_samples = AudioChunk::concat(audio);
        if all_samples.is_empty() {
            return Err(AsrError::NoAudio);
        }

        tracing::info!(
            audio_samples = all_samples.len(),
            "transcribing with dolphin-ctc"
        );

        let text = self.transcribe_samples(&all_samples)?;
        Ok(Transcript::new(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_points_at_the_int8_bundle() {
        assert_eq!(DolphinConfig::default().model_file, "model.int8.onnx");
    }

    #[test]
    fn intra_threads_is_pinned_to_one() {
        // §I.1: above one thread this backend does not reproduce.
        // Changing this constant means re-running that experiment.
        const { assert!(INTRA_THREADS == 1) };
    }

    #[test]
    fn load_reports_a_missing_bundle_rather_than_panicking() {
        let Err(err) = DolphinAdapter::load("/nonexistent/dolphin") else {
            panic!("loading a nonexistent bundle should fail");
        };
        assert!(
            err.to_string().contains("tokens.txt"),
            "unexpected error: {}",
            err
        );
    }

    /// An 80-bin identity CMVN, so `prepare_features` can be exercised
    /// without the bundle.
    fn identity_cmvn(dim: usize) -> Cmvn {
        let mean = vec!["0.0"; dim].join(",");
        let inv_std = vec!["1.0"; dim].join(",");
        Cmvn::parse(&mean, &inv_std).unwrap()
    }

    fn test_fbank() -> Fbank {
        Fbank::new(FbankOpts::dolphin_default())
    }

    #[test]
    fn empty_audio_is_rejected_before_any_inference() {
        let err = prepare_features(&test_fbank(), &identity_cmvn(80), &[]).unwrap_err();
        assert_eq!(err.to_string(), "no audio received");
    }

    #[test]
    fn features_are_one_row_per_frame() {
        // 4000 samples at snip_edges=false → 25 frames of 80 bins.
        let (feats, n) =
            prepare_features(&test_fbank(), &identity_cmvn(80), &[0.1; 4_000]).unwrap();
        assert_eq!(n, 25);
        assert_eq!(feats.len(), 25 * 80);
    }

    #[test]
    fn cmvn_actually_reaches_the_features() {
        // An identity CMVN and a shifted one must disagree, otherwise
        // the normalisation is silently being skipped — which produces
        // a plausible-looking transcript rather than an error.
        let fbank = test_fbank();
        let shifted = Cmvn::parse(&vec!["5.0"; 80].join(","), &vec!["1.0"; 80].join(",")).unwrap();
        let (plain, _) = prepare_features(&fbank, &identity_cmvn(80), &[0.1; 4_000]).unwrap();
        let (moved, _) = prepare_features(&fbank, &shifted, &[0.1; 4_000]).unwrap();
        for (a, b) in plain.iter().zip(&moved) {
            assert!((a - b - 5.0).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn a_cmvn_of_the_wrong_width_is_rejected() {
        // 25 x 80 = 2000 features divide evenly into 40-wide frames,
        // so this has to be caught by comparing widths, not by
        // divisibility.
        let err = prepare_features(&test_fbank(), &identity_cmvn(40), &[0.1; 4_000]).unwrap_err();
        assert!(err.to_string().contains("covers 40 bins"), "{}", err);
    }
}
