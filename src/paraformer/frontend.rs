//! LFR (Low Frame Rate) splicing + CMVN normalisation, ported from
//! FunASR's `funasr_onnx.utils.frontend.WavFrontend`.
//!
//! The Paraformer-large family stacks `lfr_m=7` consecutive 80-mel
//! FBANK frames every `lfr_n=6` frames, then applies a global CMVN
//! whose statistics are stored in Kaldi-NNet text format (`am.mvn`)
//! alongside the model checkpoint.
//!
//! CMVN formula (matches the reference implementation):
//!   `out = (in + means) * vars`
//! where `means` are the negative per-dim means and `vars` the inverse
//! standard deviations exposed by `<AddShift>` / `<Rescale>`.

use std::path::Path;

use crate::traits::AsrError;

/// Loaded global CMVN statistics. Length equals the post-LFR feature
/// dimension (`n_mels * lfr_m`, i.e. 560 for Paraformer-large).
#[derive(Debug, Clone)]
pub struct Cmvn {
    pub means: Vec<f32>,
    pub vars: Vec<f32>,
}

impl Cmvn {
    pub fn dim(&self) -> usize {
        self.means.len()
    }
}

/// Apply the LFR window described in `WavFrontend.apply_lfr`. The
/// input is a row-major `[t_in, feat_dim]` matrix; the output has
/// shape `[t_out, feat_dim * lfr_m]` where `t_out = ceil(t_in / lfr_n)`.
///
/// Boundary handling mirrors the reference:
/// - The first frame is replicated `(lfr_m - 1) / 2` times on the left.
/// - The last frame is replicated on the right whenever the window
///   would otherwise read past the end of the input.
pub fn apply_lfr(
    feats: &[f32],
    t_in: usize,
    feat_dim: usize,
    lfr_m: usize,
    lfr_n: usize,
) -> (Vec<f32>, usize) {
    assert!(lfr_m > 0 && lfr_n > 0, "lfr_m/lfr_n must be positive");
    if t_in == 0 {
        return (Vec::new(), 0);
    }

    let t_out = t_in.div_ceil(lfr_n);
    let out_dim = feat_dim * lfr_m;
    let mut out = Vec::with_capacity(t_out * out_dim);

    let left_pad = (lfr_m - 1) / 2;
    // Build a virtual frame index list so the boundary cases stay
    // simple to reason about.
    let frame = |raw: i64| -> &[f32] {
        let clamped = raw.clamp(0, (t_in - 1) as i64) as usize;
        &feats[clamped * feat_dim..(clamped + 1) * feat_dim]
    };

    for i in 0..t_out {
        // The reference centres the window at lfr_n*i + (lfr_m-1)/2,
        // padding with the first frame on the left and with the last
        // frame on the right.
        let centre = (i * lfr_n) as i64;
        for k in 0..lfr_m {
            let idx = centre + k as i64 - left_pad as i64;
            out.extend_from_slice(frame(idx));
        }
    }

    debug_assert_eq!(out.len(), t_out * out_dim);
    (out, t_out)
}

/// Apply `(feat + means) * vars` in place.  The CMVN dim must match
/// `feat_dim`.
pub fn apply_cmvn(feats: &mut [f32], feat_dim: usize, cmvn: &Cmvn) {
    assert_eq!(
        cmvn.means.len(),
        feat_dim,
        "cmvn dim {} != feat dim {}",
        cmvn.means.len(),
        feat_dim
    );
    assert_eq!(cmvn.vars.len(), feat_dim);
    for chunk in feats.chunks_exact_mut(feat_dim) {
        for ((x, m), v) in chunk
            .iter_mut()
            .zip(cmvn.means.iter())
            .zip(cmvn.vars.iter())
        {
            *x = (*x + *m) * *v;
        }
    }
}

/// Parse a Kaldi-NNet text-format `am.mvn` file. The relevant blocks
/// look like:
///
/// ```text
/// <Nnet>
/// <AddShift> <LearnRateCoef> 0 [ -8.31 -8.42 ... ]
/// <Rescale> <LearnRateCoef> 0 [ 0.139 0.142 ... ]
/// </Nnet>
/// ```
///
/// Bracketed value lists may span multiple lines. We extract every
/// numeric token between the matching `[` and `]`.
pub fn load_cmvn(path: &Path) -> Result<Cmvn, AsrError> {
    let text = std::fs::read_to_string(path).map_err(|e| AsrError {
        message: format!("read am.mvn {}: {e}", path.display()),
    })?;

    let means = extract_block(&text, "<AddShift>").ok_or_else(|| AsrError {
        message: format!("am.mvn {}: missing <AddShift> block", path.display()),
    })?;
    let vars = extract_block(&text, "<Rescale>").ok_or_else(|| AsrError {
        message: format!("am.mvn {}: missing <Rescale> block", path.display()),
    })?;

    if means.len() != vars.len() {
        return Err(AsrError {
            message: format!(
                "am.mvn {}: mean dim {} != var dim {}",
                path.display(),
                means.len(),
                vars.len()
            ),
        });
    }
    if means.is_empty() {
        return Err(AsrError {
            message: format!("am.mvn {}: empty CMVN vectors", path.display()),
        });
    }

    Ok(Cmvn { means, vars })
}

fn extract_block(text: &str, tag: &str) -> Option<Vec<f32>> {
    let tag_pos = text.find(tag)?;
    let after_tag = &text[tag_pos + tag.len()..];
    let open = after_tag.find('[')?;
    let close = after_tag[open..].find(']')?;
    let body = &after_tag[open + 1..open + close];
    let vals: Vec<f32> = body
        .split_whitespace()
        .filter_map(|t| t.parse::<f32>().ok())
        .collect();
    if vals.is_empty() {
        None
    } else {
        Some(vals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flatten(rows: Vec<Vec<f32>>) -> (Vec<f32>, usize, usize) {
        let t = rows.len();
        let d = rows.first().map_or(0, |r| r.len());
        let mut out = Vec::with_capacity(t * d);
        for r in rows {
            out.extend(r);
        }
        (out, t, d)
    }

    #[test]
    fn apply_lfr_paraformer_dimensions() {
        // 16 input frames, 80 mels, lfr_m=7, lfr_n=6 → ceil(16/6)=3 frames × 560 dims.
        let (feats, t, d) = flatten(vec![vec![0.0_f32; 80]; 16]);
        let (out, t_out) = apply_lfr(&feats, t, d, 7, 6);
        assert_eq!(t_out, 3);
        assert_eq!(out.len(), 3 * 7 * 80);
    }

    #[test]
    fn apply_lfr_short_input_pads_with_replication() {
        // Single frame, distinguishable values per dim, so we can verify
        // that left and right padding both reuse it.
        let row: Vec<f32> = (0..4).map(|x| x as f32).collect();
        let (feats, t, d) = flatten(vec![row.clone()]);
        let (out, t_out) = apply_lfr(&feats, t, d, 7, 6);
        assert_eq!(t_out, 1);
        assert_eq!(out.len(), 7 * 4);
        // Every 4-wide chunk must equal the source row.
        for chunk in out.chunks_exact(4) {
            assert_eq!(chunk, row.as_slice());
        }
    }

    #[test]
    fn apply_lfr_centres_window_with_left_padding() {
        // 7 frames, lfr_m=7, lfr_n=6 → 2 output frames. left_pad=3 means
        // the first output is [f0,f0,f0,f0,f1,f2,f3].
        let mut rows = Vec::new();
        for i in 0..7 {
            rows.push(vec![i as f32; 2]);
        }
        let (feats, t, d) = flatten(rows);
        let (out, t_out) = apply_lfr(&feats, t, d, 7, 6);
        assert_eq!(t_out, 2);
        // First spliced frame
        let first: Vec<f32> = out[..7 * 2].to_vec();
        let expected_first: Vec<f32> = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
            .iter()
            .flat_map(|v| [*v, *v])
            .collect();
        assert_eq!(first, expected_first);
    }

    #[test]
    fn apply_cmvn_matches_reference_formula() {
        // (feat + mean) * var
        let mut feats = vec![1.0_f32, 2.0, 3.0, 4.0]; // two frames of dim 2
        let cmvn = Cmvn {
            means: vec![-1.0, -2.0],
            vars: vec![2.0, 0.5],
        };
        apply_cmvn(&mut feats, 2, &cmvn);
        // frame 0: (1-1)*2=0, (2-2)*0.5=0
        // frame 1: (3-1)*2=4, (4-2)*0.5=1
        assert_eq!(feats, vec![0.0, 0.0, 4.0, 1.0]);
    }

    #[test]
    fn load_cmvn_parses_addshift_and_rescale() {
        let dir =
            std::env::temp_dir().join(format!("euhadra_paraformer_mvn_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("am.mvn");
        let body = "<Nnet>\n\
            <AddShift> <LearnRateCoef> 0 [ -8.31 -8.42 -8.55 ]\n\
            <Rescale> <LearnRateCoef> 0 [ 0.139 0.142 0.145 ]\n\
            </Nnet>\n";
        std::fs::write(&path, body).unwrap();
        let cmvn = load_cmvn(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(cmvn.dim(), 3);
        assert!((cmvn.means[0] - -8.31).abs() < 1e-5);
        assert!((cmvn.vars[2] - 0.145).abs() < 1e-5);
    }

    #[test]
    fn load_cmvn_rejects_dim_mismatch() {
        let dir =
            std::env::temp_dir().join(format!("euhadra_paraformer_mvn_bad_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("am.mvn");
        std::fs::write(&path, "<AddShift> [ -1 -2 -3 ]\n<Rescale> [ 1 2 ]\n").unwrap();
        let res = load_cmvn(&path);
        std::fs::remove_dir_all(&dir).ok();
        assert!(res.is_err());
    }

    #[test]
    fn load_cmvn_handles_multiline_brackets() {
        let dir = std::env::temp_dir().join(format!(
            "euhadra_paraformer_mvn_multi_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("am.mvn");
        let body = "<Nnet>\n\
            <AddShift> <LearnRateCoef> 0 [\n  -1.0 -2.0\n  -3.0 ]\n\
            <Rescale> <LearnRateCoef> 0 [ 1.0\n  2.0\n  3.0 ]\n";
        std::fs::write(&path, body).unwrap();
        let cmvn = load_cmvn(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(cmvn.dim(), 3);
        assert!((cmvn.means[1] - -2.0).abs() < 1e-5);
        assert!((cmvn.vars[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn load_cmvn_missing_file_errors() {
        let res = load_cmvn(Path::new("/nonexistent/euhadra/am.mvn"));
        assert!(res.is_err());
    }
}
