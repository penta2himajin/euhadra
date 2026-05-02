//! `metadata.json` produced alongside the ONNX export.
//!
//! The official `FunAudioLLM/SenseVoice` export pipeline does not
//! embed the constants we need (LID / textnorm / blank id / LFR
//! configuration) into the ONNX `metadata_props` table. Rather than
//! hardcode them on the Rust side and risk silent drift if upstream
//! re-trains with a different vocab, `scripts/setup_sensevoice.sh`
//! dumps a small JSON sidecar at export time and we read it here.
//!
//! Schema:
//!
//! ```json
//! {
//!   "lang2id":      { "auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13 },
//!   "with_itn_id":  14,
//!   "without_itn_id": 15,
//!   "blank_id":     0,
//!   "lfr_m":        7,
//!   "lfr_n":        6
//! }
//! ```

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use crate::traits::AsrError;

#[derive(Debug, Clone, Deserialize)]
pub struct SenseVoiceMetadata {
    pub lang2id: HashMap<String, i32>,
    pub with_itn_id: i32,
    pub without_itn_id: i32,
    pub blank_id: u32,
    pub lfr_m: usize,
    pub lfr_n: usize,
}

impl SenseVoiceMetadata {
    pub fn load(path: &Path) -> Result<Self, AsrError> {
        let bytes = std::fs::read(path).map_err(|e| AsrError {
            message: format!("read metadata.json {}: {e}", path.display()),
        })?;
        let meta: SenseVoiceMetadata = serde_json::from_slice(&bytes).map_err(|e| AsrError {
            message: format!("parse metadata.json {}: {e}", path.display()),
        })?;
        meta.validate(path)?;
        Ok(meta)
    }

    /// Resolve a language hint to the integer the encoder expects.
    /// Accepts ISO 639-1 codes ("ko", "ja"), the long-form aliases
    /// the upstream demo scripts use ("korean"), and "auto".
    pub fn language_id(&self, lang: &str) -> Option<i32> {
        let key = match lang.to_lowercase().as_str() {
            "korean" => "ko",
            "japanese" => "ja",
            "chinese" | "mandarin" => "zh",
            "english" => "en",
            "cantonese" => "yue",
            other => return self.lang2id.get(other).copied(),
        };
        self.lang2id.get(key).copied()
    }

    fn validate(&self, path: &Path) -> Result<(), AsrError> {
        if self.lfr_m == 0 || self.lfr_n == 0 {
            return Err(AsrError {
                message: format!(
                    "metadata.json {}: lfr_m and lfr_n must be positive (got {} / {})",
                    path.display(),
                    self.lfr_m,
                    self.lfr_n
                ),
            });
        }
        // The upstream model ships with these four keys at minimum.
        // Catch a malformed sidecar early instead of failing at
        // transcribe time.
        for key in ["auto", "zh", "en", "ko"] {
            if !self.lang2id.contains_key(key) {
                return Err(AsrError {
                    message: format!(
                        "metadata.json {}: lang2id missing required key {:?}",
                        path.display(),
                        key
                    ),
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_json(name: &str, body: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "euhadra_sensevoice_meta_{}_{}",
            name,
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("metadata.json");
        std::fs::write(&path, body).unwrap();
        path
    }

    fn cleanup(path: &Path) {
        if let Some(dir) = path.parent() {
            std::fs::remove_dir_all(dir).ok();
        }
    }

    fn ok_body() -> &'static str {
        r#"{
          "lang2id": {
            "auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13
          },
          "with_itn_id": 14,
          "without_itn_id": 15,
          "blank_id": 0,
          "lfr_m": 7,
          "lfr_n": 6
        }"#
    }

    #[test]
    fn load_parses_canonical_layout() {
        let p = write_json("ok", ok_body());
        let m = SenseVoiceMetadata::load(&p).unwrap();
        cleanup(&p);
        assert_eq!(m.blank_id, 0);
        assert_eq!(m.with_itn_id, 14);
        assert_eq!(m.without_itn_id, 15);
        assert_eq!(m.lfr_m, 7);
        assert_eq!(m.lfr_n, 6);
        assert_eq!(m.lang2id.get("ko"), Some(&12));
        assert_eq!(m.lang2id.get("yue"), Some(&7));
    }

    #[test]
    fn language_id_accepts_iso_and_long_form() {
        let p = write_json("aliases", ok_body());
        let m = SenseVoiceMetadata::load(&p).unwrap();
        cleanup(&p);
        assert_eq!(m.language_id("ko"), Some(12));
        assert_eq!(m.language_id("KO"), Some(12));
        assert_eq!(m.language_id("korean"), Some(12));
        assert_eq!(m.language_id("Japanese"), Some(11));
        assert_eq!(m.language_id("mandarin"), Some(3));
        assert_eq!(m.language_id("xx"), None);
    }

    #[test]
    fn validate_rejects_zero_lfr() {
        let p = write_json(
            "bad_lfr",
            r#"{"lang2id":{"auto":0,"zh":3,"en":4,"ko":12},"with_itn_id":14,"without_itn_id":15,"blank_id":0,"lfr_m":0,"lfr_n":6}"#,
        );
        let res = SenseVoiceMetadata::load(&p);
        cleanup(&p);
        assert!(res.is_err());
    }

    #[test]
    fn validate_rejects_missing_required_lang_key() {
        // Missing "ko" — the whole point of the integration.
        let p = write_json(
            "no_ko",
            r#"{"lang2id":{"auto":0,"zh":3,"en":4},"with_itn_id":14,"without_itn_id":15,"blank_id":0,"lfr_m":7,"lfr_n":6}"#,
        );
        let res = SenseVoiceMetadata::load(&p);
        cleanup(&p);
        assert!(res.is_err());
    }

    #[test]
    fn load_missing_file_errors() {
        let res = SenseVoiceMetadata::load(Path::new("/nonexistent/euhadra/metadata.json"));
        assert!(res.is_err());
    }
}
