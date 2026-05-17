//! Whisper-large-v3 tokenizer loader and prompt builder.
//!
//! Loads the HF `tokenizer.json` bundle that ships in every
//! `onnx-community/whisper-large-v3*` snapshot, resolves the special
//! tokens needed to steer decoding (SOT / language / task / no-ts /
//! EOT), and builds the 4-token prompt the decoder consumes.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::traits::AsrError;

/// Whisper task — `transcribe` keeps source-language text;
/// `translate` would re-render into English. `transcribe` is the
/// only mode this adapter exposes today.
pub const TASK_TRANSCRIBE: &str = "<|transcribe|>";
pub const SOT: &str = "<|startoftranscript|>";
pub const NO_TIMESTAMPS: &str = "<|notimestamps|>";
pub const END_OF_TEXT: &str = "<|endoftext|>";

/// Wrapper around the HF tokenizer that caches the special-token ids
/// we look up every step.
pub struct WhisperTokenizer {
    inner: Tokenizer,
    sot: i64,
    transcribe: i64,
    no_timestamps: i64,
    eot: i64,
}

impl WhisperTokenizer {
    /// Load `<dir>/tokenizer.json`.
    pub fn load(model_dir: &Path) -> Result<Self, AsrError> {
        let tok_path = model_dir.join("tokenizer.json");
        let inner = Tokenizer::from_file(&tok_path).map_err(|e| AsrError {
            message: format!(
                "whisper-onnx tokenizer load failed at {}: {e}",
                tok_path.display()
            ),
        })?;
        let sot = lookup_id(&inner, SOT)?;
        let transcribe = lookup_id(&inner, TASK_TRANSCRIBE)?;
        let no_timestamps = lookup_id(&inner, NO_TIMESTAMPS)?;
        let eot = lookup_id(&inner, END_OF_TEXT)?;
        Ok(Self {
            inner,
            sot,
            transcribe,
            no_timestamps,
            eot,
        })
    }

    /// Build the 4-token decoder prompt for a transcription request.
    /// `lang` is a BCP-47 short code (`"ko"`, `"en"`, ...). Unknown
    /// codes fall back to `None`, letting the model run its own LID
    /// (Whisper does this when no language token is forced).
    pub fn build_prompt(&self, lang: Option<&str>) -> Result<Vec<i64>, AsrError> {
        let mut prompt = vec![self.sot];
        if let Some(code) = lang {
            let tok = format!("<|{code}|>");
            let id = lookup_id(&self.inner, &tok)?;
            prompt.push(id);
        }
        prompt.push(self.transcribe);
        prompt.push(self.no_timestamps);
        Ok(prompt)
    }

    pub fn eot(&self) -> i64 {
        self.eot
    }

    /// `tokenizer.decode(ids, skip_special_tokens=True)`. Returns
    /// the trimmed plain-text transcript.
    pub fn decode(&self, ids: &[i64]) -> Result<String, AsrError> {
        let u32_ids: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
        let text = self.inner.decode(&u32_ids, true).map_err(|e| AsrError {
            message: format!("whisper-onnx tokenizer decode: {e}"),
        })?;
        Ok(text.trim().to_string())
    }
}

fn lookup_id(tok: &Tokenizer, piece: &str) -> Result<i64, AsrError> {
    tok.token_to_id(piece)
        .map(|id| id as i64)
        .ok_or_else(|| AsrError {
            message: format!("whisper-onnx tokenizer missing special token {piece}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_dir() -> Option<PathBuf> {
        // Bench scratch dir from the Korean ASR session. Tests that
        // require the real tokenizer skip cleanly when it's absent so
        // CI without the model bundle still passes.
        let p = PathBuf::from("/tmp/whisper-onnx-turbo-int8");
        p.join("tokenizer.json").exists().then_some(p)
    }

    #[test]
    fn loads_special_token_ids_when_bundle_present() {
        let Some(dir) = fixture_dir() else {
            eprintln!("skip: /tmp/whisper-onnx-turbo-int8/tokenizer.json missing");
            return;
        };
        let tok = WhisperTokenizer::load(&dir).expect("load");
        // Anchors from added_tokens.json in the upstream bundle.
        assert_eq!(tok.sot, 50258);
        assert_eq!(tok.transcribe, 50360);
        assert_eq!(tok.no_timestamps, 50364);
        assert_eq!(tok.eot, 50257);
    }

    #[test]
    fn ko_prompt_matches_known_id_sequence() {
        let Some(dir) = fixture_dir() else {
            eprintln!("skip: tokenizer fixture missing");
            return;
        };
        let tok = WhisperTokenizer::load(&dir).unwrap();
        let p = tok.build_prompt(Some("ko")).unwrap();
        assert_eq!(p, vec![50258, 50264, 50360, 50364]);
    }

    #[test]
    fn auto_lid_prompt_omits_language_token() {
        let Some(dir) = fixture_dir() else {
            eprintln!("skip: tokenizer fixture missing");
            return;
        };
        let tok = WhisperTokenizer::load(&dir).unwrap();
        let p = tok.build_prompt(None).unwrap();
        assert_eq!(p, vec![50258, 50360, 50364]);
    }
}
