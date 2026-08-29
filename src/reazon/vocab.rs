//! Token table + icefall byte-level BPE decode for ReazonSpeech Zipformer.
//!
//! ReazonSpeech k2 v2 tokens are icefall byte-level BPE: each vocab entry
//! is a short string of "byte chars" drawn from
//! `icefall.byte_utils.PRINTABLE_BASE_CHARS`. After greedy search we strip
//! SentencePiece `▁` markers, concatenate the surfaces, and run
//! `byte_decode` to recover UTF-8 (including Japanese).

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::traits::AsrError;

/// icefall `PRINTABLE_BASE_CHARS` — 256 Unicode code points, one per byte.
/// Copied from <https://github.com/k2-fsa/icefall/blob/master/icefall/byte_utils.py>.
const PRINTABLE_BASE_CHARS: [u32; 256] = [
    256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274,
    275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
    88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
    109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 288,
    289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 308, 309,
    310, 311, 312, 313, 314, 315, 316, 317, 318, 321, 322, 323, 324, 325, 326, 327, 328, 330, 331,
    332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350,
    351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369,
    370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389,
    390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408,
    409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422,
];

/// BPE `<unk>` codepoint (`chr(8263)`); icefall maps it to space.
const BPE_UNK: char = '\u{2047}';

/// icefall `byte_decode`: map the BPE surface string back to UTF-8 text.
pub fn byte_decode(text: &str) -> String {
    let mut out = Vec::with_capacity(text.len());
    for ch in text.chars() {
        let b = if ch == BPE_UNK {
            Some(32u8)
        } else {
            // Linear scan is fine — vocab decode is tiny vs ONNX.
            PRINTABLE_BASE_CHARS
                .iter()
                .position(|&c| c == ch as u32)
                .map(|i| i as u8)
        };
        match b {
            Some(byte) => out.push(byte),
            None => return String::new(),
        }
    }
    match String::from_utf8(out) {
        Ok(s) => s,
        Err(_) => String::new(),
    }
}

/// `tokens.txt` lines: `<symbol> <id>`.
#[derive(Debug, Clone)]
pub struct Vocab {
    tokens: Vec<String>,
}

impl Vocab {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, AsrError> {
        let file = File::open(path.as_ref()).map_err(|e| {
            AsrError::ModelLoad(format!("reazon tokens open {}: {e}", path.as_ref().display()))
        })?;
        let reader = BufReader::new(file);
        let mut tokens = Vec::new();
        for (i, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| AsrError::ModelLoad(format!("reazon tokens read: {e}")))?;
            let mut parts = line.split_whitespace();
            let Some(sym) = parts.next() else {
                continue;
            };
            let id = parts
                .next()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(i);
            if id >= tokens.len() {
                tokens.resize(id + 1, String::new());
            }
            tokens[id] = sym.to_string();
        }
        if tokens.is_empty() {
            return Err(AsrError::ModelLoad("reazon tokens.txt is empty".into()));
        }
        Ok(Self { tokens })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn get(&self, id: i64) -> Option<&str> {
        if id < 0 {
            return None;
        }
        self.tokens.get(id as usize).map(|s| s.as_str())
    }

    /// Join non-blank token strings (strip SentencePiece `▁`), then byte-decode.
    pub fn decode_hyp(&self, ids: &[i64], blank_id: i64) -> String {
        let mut surface = String::new();
        for &id in ids {
            if id == blank_id {
                continue;
            }
            if let Some(tok) = self.get(id) {
                if tok.starts_with('<') && tok.ends_with('>') {
                    continue;
                }
                surface.push_str(&tok.replace('\u{2581}', ""));
            }
        }
        byte_decode(&surface)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_decode_maps_space_and_ascii_letter() {
        // Byte 32 → chr(PRINTABLE_BASE_CHARS[32]) = chr(32) = ' '
        assert_eq!(byte_decode(" "), " ");
        // Byte 65 ('A') → chr(65) = 'A' (ASCII range is identity in the table)
        assert_eq!(byte_decode("A"), "A");
    }

    #[test]
    fn byte_decode_japanese_from_reazon_token_surface() {
        // tokens.txt id 3 is `▁ƊĢĥ`; after stripping ▁ the surface byte-decodes to い.
        let surface: String = ['\u{018a}', '\u{0122}', '\u{0125}'].iter().collect();
        assert_eq!(byte_decode(&surface), "い");
    }

    #[test]
    fn printable_base_len_is_256() {
        assert_eq!(PRINTABLE_BASE_CHARS.len(), 256);
    }
}
