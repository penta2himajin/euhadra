use async_trait::async_trait;

use crate::types::ContextSnapshot;

// ---------------------------------------------------------------------------
// TextProcessor trait
// ---------------------------------------------------------------------------

/// Structural text correction applied between TextFilter and LlmRefiner.
///
/// Handles punctuation insertion, capitalization, self-correction detection,
/// and list formatting using lightweight models or rules — no LLM required.
///
/// Multiple processors can be chained.
#[async_trait]
pub trait TextProcessor: Send + Sync {
    async fn process(
        &self,
        text: &str,
        context: &ContextSnapshot,
    ) -> Result<ProcessResult, ProcessError>;
}

#[derive(Debug, Clone)]
pub struct ProcessResult {
    pub text: String,
    pub corrections: Vec<Correction>,
}

#[derive(Debug, Clone)]
pub struct Correction {
    pub kind: CorrectionKind,
    pub original: String,
    pub replacement: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorrectionKind {
    PunctuationInserted,
    Capitalized,
    SelfCorrectionRemoved,
    ListFormatted,
    DictionaryMatch,
}

#[derive(Debug, Clone)]
pub struct ProcessError {
    pub message: String,
}

impl std::fmt::Display for ProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "process error: {}", self.message)
    }
}

impl std::error::Error for ProcessError {}

// ---------------------------------------------------------------------------
// SelfCorrectionDetector
// ---------------------------------------------------------------------------

/// Detects and removes self-corrections (disfluency repairs) in ASR output.
///
/// Pattern: "I want to go to Boston, to Denver" → "I want to go to Denver"
///
/// Uses a sliding window to find repeated structural patterns (reparandum →
/// repair), where the repair "overwrites" the reparandum.  Detection is based
/// on the observation that self-corrections typically produce a "rough copy":
/// the repair reuses the same or similar words in roughly the same order.
pub struct SelfCorrectionDetector {
    /// Minimum number of shared words between reparandum and repair to trigger.
    min_shared_words: usize,
    /// Correction cue words that signal a self-correction boundary.
    correction_cues_en: Vec<String>,
    correction_cues_ja: Vec<String>,
}

impl SelfCorrectionDetector {
    pub fn new() -> Self {
        // Cues are sorted longest-first because the detector uses
        // `str::find(cue)` and a shorter cue that is a substring of a
        // longer one would otherwise win. Concretely: without this
        // sort, "ていうか" (4 chars) matches before "っていうか" (5 chars)
        // inside "鈴木課長、っていうか佐藤課長です", causing the detector
        // to treat just "っ" as the reparandum instead of the full
        // "鈴木課長" preceding the cue. Same hazard exists for en
        // ("no" / "no wait", "rather" / "or rather").
        let mut correction_cues_en: Vec<String> = vec![
            "no", "wait", "sorry", "i mean", "actually", "rather",
            "no wait", "or rather",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        correction_cues_en.sort_by_key(|c| std::cmp::Reverse(c.chars().count()));

        let mut correction_cues_ja: Vec<String> = vec![
            "いや", "じゃなくて", "じゃなく", "ではなく", "ていうか",
            "っていうか", "じゃない",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        correction_cues_ja.sort_by_key(|c| std::cmp::Reverse(c.chars().count()));

        Self {
            min_shared_words: 1,
            correction_cues_en,
            correction_cues_ja,
        }
    }

    /// Find self-correction patterns in English text.
    ///
    /// Looks for correction cues and removes the reparandum (the part before
    /// the cue that the speaker intended to discard).
    fn detect_english(&self, text: &str) -> Option<(String, Correction)> {
        let lower = text.to_lowercase();

        for cue in &self.correction_cues_en {
            if let Some(cue_pos) = lower.find(cue.as_str()) {
                let before_cue = &text[..cue_pos].trim_end();
                let after_cue = &text[cue_pos + cue.len()..].trim_start();

                if after_cue.is_empty() {
                    continue;
                }

                // Find the overlap: walk backwards from the cue to find where
                // the reparandum starts.  We look for the point where the
                // words before the cue start repeating structure found after it.
                let before_words: Vec<&str> = before_cue.split_whitespace().collect();
                let after_words: Vec<&str> = after_cue.split_whitespace().collect();

                if after_words.is_empty() {
                    continue;
                }

                // Find how many trailing words of `before` share a common
                // prefix with `after` (the "rough copy" pattern).
                let shared = Self::count_shared_prefix_from_end(&before_words, &after_words);

                if shared >= self.min_shared_words {
                    // Remove the reparandum: keep words before the shared
                    // prefix, then append the repair (after cue).
                    let keep_count = before_words.len() - shared;
                    let kept: Vec<&str> = before_words[..keep_count].to_vec();
                    let result = if kept.is_empty() {
                        after_cue.to_string()
                    } else {
                        format!("{} {}", kept.join(" "), after_cue)
                    };

                    let original_rm = before_words[keep_count..].join(" ");
                    return Some((
                        result,
                        Correction {
                            kind: CorrectionKind::SelfCorrectionRemoved,
                            original: format!("{} {}", original_rm, cue),
                            replacement: String::new(),
                        },
                    ));
                }
            }
        }
        None
    }

    /// Find self-correction patterns in Japanese text.
    fn detect_japanese(&self, text: &str) -> Option<(String, Correction)> {
        for cue in &self.correction_cues_ja {
            if let Some(cue_pos) = text.find(cue.as_str()) {
                let before = text[..cue_pos].trim_end_matches('、').trim_end();
                let after = text[cue_pos + cue.len()..].trim_start_matches('、').trim_start();

                if after.is_empty() || before.is_empty() {
                    continue;
                }

                // In Japanese, self-corrections often have the form:
                // "AAAいやBBB" where BBB replaces AAA.
                // We find the last comma-delimited segment before the cue
                // and remove it, keeping everything before and the repair after.
                let segments: Vec<&str> = before.split('、').collect();
                if segments.is_empty() {
                    continue;
                }

                let reparandum = segments.last().unwrap().trim();
                let kept_before = if segments.len() > 1 {
                    segments[..segments.len() - 1].join("、")
                } else {
                    String::new()
                };

                let result = if kept_before.is_empty() {
                    after.to_string()
                } else {
                    format!("{}、{}", kept_before, after)
                };

                return Some((
                    result,
                    Correction {
                        kind: CorrectionKind::SelfCorrectionRemoved,
                        original: format!("{}{}",  reparandum, cue),
                        replacement: String::new(),
                    },
                ));
            }
        }
        None
    }

    /// Count how many words at the end of `before` match the beginning of
    /// `after` when lowercased.
    fn count_shared_prefix_from_end(before: &[&str], after: &[&str]) -> usize {
        let mut count = 0;
        let max_check = before.len().min(after.len()).min(5); // limit search window

        for offset in 1..=max_check {
            let b_idx = before.len() - offset;
            if before[b_idx].to_lowercase() == after[0].to_lowercase() {
                // Check if subsequent words also match
                let mut matches = 1;
                for j in 1..offset.min(after.len()) {
                    if before[b_idx + j].to_lowercase() == after[j].to_lowercase() {
                        matches += 1;
                    } else {
                        break;
                    }
                }
                if matches >= 1 {
                    count = count.max(offset);
                }
            }
        }
        count
    }
}

impl Default for SelfCorrectionDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TextProcessor for SelfCorrectionDetector {
    async fn process(
        &self,
        text: &str,
        _context: &ContextSnapshot,
    ) -> Result<ProcessResult, ProcessError> {
        // Try English first, then Japanese
        if let Some((result, correction)) = self.detect_english(text) {
            return Ok(ProcessResult {
                text: result,
                corrections: vec![correction],
            });
        }
        if let Some((result, correction)) = self.detect_japanese(text) {
            return Ok(ProcessResult {
                text: result,
                corrections: vec![correction],
            });
        }

        Ok(ProcessResult {
            text: text.to_string(),
            corrections: vec![],
        })
    }
}

// ---------------------------------------------------------------------------
// BasicPunctuationRestorer — rule-based heuristic
// ---------------------------------------------------------------------------

/// Lightweight rule-based punctuation insertion.
///
/// This is a stopgap until a proper CNN-BiLSTM ONNX model is integrated.
/// It handles the most common cases:
/// - Capitalize first word of text
/// - Ensure text ends with a period (if no terminal punctuation)
/// - Capitalize after sentence-ending punctuation
pub struct BasicPunctuationRestorer;

#[async_trait]
impl TextProcessor for BasicPunctuationRestorer {
    async fn process(
        &self,
        text: &str,
        _context: &ContextSnapshot,
    ) -> Result<ProcessResult, ProcessError> {
        if text.is_empty() {
            return Ok(ProcessResult {
                text: String::new(),
                corrections: vec![],
            });
        }

        let mut result = String::with_capacity(text.len() + 8);
        let mut corrections = Vec::new();
        let mut capitalize_next = true;

        for ch in text.chars() {
            if capitalize_next && ch.is_alphabetic() {
                let upper: String = ch.to_uppercase().collect();
                if upper != ch.to_string() {
                    corrections.push(Correction {
                        kind: CorrectionKind::Capitalized,
                        original: ch.to_string(),
                        replacement: upper.clone(),
                    });
                }
                result.push_str(&upper);
                capitalize_next = false;
            } else {
                result.push(ch);
                if ch == '.' || ch == '!' || ch == '?' {
                    capitalize_next = true;
                } else if ch == '。' {
                    // Japanese sentence-ending — next sentence starts
                    capitalize_next = false; // no capitalization in Japanese
                }
            }
        }

        // Ensure terminal punctuation (English only — check if text is Latin-based)
        let trimmed = result.trim_end();
        let last_char = trimmed.chars().last();
        let is_latin = trimmed.chars().any(|c| c.is_ascii_alphabetic());
        if is_latin {
            if let Some(last) = last_char {
                if !matches!(last, '.' | '!' | '?' | ')' | '"' | '\'') {
                    result = format!("{}.", trimmed);
                    corrections.push(Correction {
                        kind: CorrectionKind::PunctuationInserted,
                        original: String::new(),
                        replacement: ".".to_string(),
                    });
                }
            }
        }

        Ok(ProcessResult {
            text: result,
            corrections,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_context() -> ContextSnapshot {
        ContextSnapshot::default()
    }

    // --- SelfCorrectionDetector tests ---

    #[tokio::test]
    async fn self_correction_with_no_wait() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("I want to go to Boston no wait to Denver", &empty_context())
            .await
            .unwrap();
        assert!(
            result.text.contains("Denver"),
            "repair should be kept: {}",
            result.text
        );
        assert!(
            !result.text.contains("Boston"),
            "reparandum should be removed: {}",
            result.text
        );
        assert_eq!(result.corrections.len(), 1);
        assert_eq!(
            result.corrections[0].kind,
            CorrectionKind::SelfCorrectionRemoved
        );
    }

    #[tokio::test]
    async fn self_correction_with_i_mean() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process(
                "the meeting is tomorrow i mean the meeting is on Friday",
                &empty_context(),
            )
            .await
            .unwrap();
        assert!(
            result.text.contains("Friday"),
            "repair should be kept: {}",
            result.text
        );
    }

    #[tokio::test]
    async fn no_self_correction_in_clean_text() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("I want to go to Denver for the conference", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "I want to go to Denver for the conference");
        assert!(result.corrections.is_empty());
    }

    #[tokio::test]
    async fn japanese_self_correction() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("明日、いや明後日に会議があります", &empty_context())
            .await
            .unwrap();
        assert!(
            result.text.contains("明後日"),
            "repair should be kept: {}",
            result.text
        );
        assert!(
            !result.text.contains("明日"),
            "reparandum should be removed: {}",
            result.text
        );
    }

    #[tokio::test]
    async fn japanese_self_correction_with_janakute() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process(
                "東京駅、じゃなくて品川駅で待ち合わせ",
                &empty_context(),
            )
            .await
            .unwrap();
        assert!(result.text.contains("品川駅"), "repair: {}", result.text);
        assert!(!result.text.contains("東京駅"), "reparandum: {}", result.text);
    }

    // --- BasicPunctuationRestorer tests ---

    #[tokio::test]
    async fn capitalize_first_word() {
        let proc = BasicPunctuationRestorer;
        let result = proc
            .process("hello world", &empty_context())
            .await
            .unwrap();
        assert!(result.text.starts_with('H'));
    }

    #[tokio::test]
    async fn add_terminal_period() {
        let proc = BasicPunctuationRestorer;
        let result = proc
            .process("hello world", &empty_context())
            .await
            .unwrap();
        assert!(result.text.ends_with('.'));
    }

    #[tokio::test]
    async fn preserve_existing_punctuation() {
        let proc = BasicPunctuationRestorer;
        let result = proc
            .process("Hello world!", &empty_context())
            .await
            .unwrap();
        assert!(result.text.ends_with('!'));
        assert!(!result.text.ends_with("!."));
    }

    #[tokio::test]
    async fn capitalize_after_period() {
        let proc = BasicPunctuationRestorer;
        let result = proc
            .process("hello. world", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "Hello. World.");
    }

    #[tokio::test]
    async fn japanese_no_terminal_period_added() {
        let proc = BasicPunctuationRestorer;
        // Pure Japanese text should not get an English period appended
        let result = proc
            .process("お江戸を発って二十里上方", &empty_context())
            .await
            .unwrap();
        assert!(
            !result.text.ends_with('.'),
            "Japanese should not get English period: {}",
            result.text
        );
    }

    #[tokio::test]
    async fn empty_text() {
        let proc = BasicPunctuationRestorer;
        let result = proc.process("", &empty_context()).await.unwrap();
        assert_eq!(result.text, "");
        assert!(result.corrections.is_empty());
    }

    /// Regression: when the input contains `っていうか`, the detector
    /// used to find the shorter `ていうか` substring first (because
    /// cues were iterated in declared order and `ていうか` precedes
    /// `っていうか` in the list). That treated just `っ` as the
    /// reparandum and left the real reparandum (e.g. `鈴木課長`) in
    /// the output. Sorting cues longest-first at construction time
    /// fixes this, surfaced by the L3 self-correction direct F1
    /// evaluation against the committed ja annotations.
    #[tokio::test]
    async fn ja_self_correction_with_tte_iu_ka() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("鈴木課長、っていうか佐藤課長です", &empty_context())
            .await
            .unwrap();
        assert!(
            !result.text.contains("鈴木課長"),
            "reparandum should be removed: {}",
            result.text,
        );
        assert!(
            result.text.contains("佐藤課長"),
            "repair should be kept: {}",
            result.text,
        );
        assert_eq!(result.text, "佐藤課長です");
    }

    /// Regression: same shape as `っていうか`/`ていうか` but for English
    /// (`no wait` containing `no` as a prefix). Without longest-first
    /// cue sorting, `no` could match first inside `no wait` and the
    /// detector would skip the full reparandum.
    #[tokio::test]
    async fn en_self_correction_no_wait_prefers_long_cue() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process(
                "I want to go to Boston no wait to Denver",
                &empty_context(),
            )
            .await
            .unwrap();
        assert!(!result.text.contains("Boston"), "{}", result.text);
        assert!(result.text.contains("Denver"), "{}", result.text);
    }
}
