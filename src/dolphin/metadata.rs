//! The per-bin normalisation Dolphin carries inside its ONNX graph.
//!
//! Unlike SenseVoice — where `scripts/setup_sensevoice.sh` writes a
//! JSON sidecar — the k2-fsa Dolphin export puts everything the runtime
//! needs in the graph's own `metadata_props`: `mean` and `invstd` as
//! comma-separated lists of `feature_dim` floats, plus `model_type`,
//! `vocab_size` and provenance. Reading them from the model means the
//! bundle cannot drift out of sync with a sidecar we forgot to update.

use crate::traits::AsrError;

/// Mean/inverse-standard-deviation normalisation, one pair per mel bin.
#[derive(Debug, Clone)]
pub struct Cmvn {
    pub mean: Vec<f32>,
    pub inv_std: Vec<f32>,
}

impl Cmvn {
    /// Parse the two comma-separated lists.
    pub fn parse(mean: &str, inv_std: &str) -> Result<Self, AsrError> {
        let mean = parse_list(mean, "mean")?;
        let inv_std = parse_list(inv_std, "invstd")?;
        if mean.len() != inv_std.len() {
            return Err(AsrError {
                message: format!(
                    "dolphin CMVN length mismatch: mean has {}, invstd has {}",
                    mean.len(),
                    inv_std.len()
                ),
            });
        }
        if mean.is_empty() {
            return Err(AsrError {
                message: "dolphin CMVN is empty".into(),
            });
        }
        Ok(Self { mean, inv_std })
    }

    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    /// Apply `(x - mean) * inv_std` in place over a row-major
    /// `[frames, dim]` buffer.
    pub fn apply(&self, feats: &mut [f32]) -> Result<(), AsrError> {
        let dim = self.dim();
        if !feats.len().is_multiple_of(dim) {
            return Err(AsrError {
                message: format!(
                    "dolphin CMVN: {} features is not a whole number of {dim}-wide frames",
                    feats.len()
                ),
            });
        }
        for frame in feats.chunks_exact_mut(dim) {
            for ((v, m), s) in frame.iter_mut().zip(&self.mean).zip(&self.inv_std) {
                *v = (*v - m) * s;
            }
        }
        Ok(())
    }
}

fn parse_list(raw: &str, what: &str) -> Result<Vec<f32>, AsrError> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<f32>().map_err(|e| AsrError {
                message: format!("dolphin {what}: {s:?} is not a float: {e}"),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_comma_separated_floats() {
        let c = Cmvn::parse("-8.2,-7.25", "0.28,0.26").unwrap();
        assert_eq!(c.dim(), 2);
        assert!((c.mean[0] + 8.2).abs() < 1e-6);
        assert!((c.inv_std[1] - 0.26).abs() < 1e-6);
    }

    #[test]
    fn tolerates_whitespace_and_a_trailing_comma() {
        let c = Cmvn::parse(" 1.0 , 2.0 ,", "1.0,1.0").unwrap();
        assert_eq!(c.dim(), 2);
    }

    #[test]
    fn rejects_mismatched_or_empty_lists() {
        assert!(Cmvn::parse("1.0,2.0", "1.0").is_err());
        assert!(Cmvn::parse("", "").is_err());
        assert!(Cmvn::parse("1.0,x", "1.0,1.0").is_err());
    }

    #[test]
    fn applies_the_affine_map_per_bin() {
        let c = Cmvn::parse("1.0,10.0", "2.0,0.5").unwrap();
        let mut feats = vec![1.0, 10.0, 2.0, 12.0];
        c.apply(&mut feats).unwrap();
        // Bin 0: (1-1)*2 = 0, (2-1)*2 = 2. Bin 1: (10-10)*.5 = 0, (12-10)*.5 = 1.
        assert_eq!(feats, vec![0.0, 0.0, 2.0, 1.0]);
    }

    #[test]
    fn rejects_a_ragged_final_frame() {
        // Silently normalising a partial frame would misalign every bin
        // in it against the wrong mean.
        let c = Cmvn::parse("1.0,2.0", "1.0,1.0").unwrap();
        assert!(c.apply(&mut [1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn an_empty_feature_buffer_is_a_no_op() {
        let c = Cmvn::parse("1.0", "1.0").unwrap();
        assert!(c.apply(&mut []).is_ok());
    }
}
