use async_trait::async_trait;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// TextFilter trait
// ---------------------------------------------------------------------------

/// A lightweight text transformation applied between ASR and LLM refinement.
///
/// Filters are cheaper than full LLM calls and handle mechanical
/// transformations: filler removal, profanity filtering, normalization, etc.
/// Multiple filters can be chained.
#[async_trait]
pub trait TextFilter: Send + Sync {
    /// Transform text, returning the filtered version and a list of removed
    /// segments (for diagnostics / undo).
    async fn filter(&self, text: &str) -> Result<FilterResult, FilterError>;
}

#[derive(Debug, Clone)]
pub struct FilterResult {
    pub text: String,
    pub removed: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FilterError {
    pub message: String,
}

impl std::fmt::Display for FilterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "filter error: {}", self.message)
    }
}

impl std::error::Error for FilterError {}

// ---------------------------------------------------------------------------
// EmbeddingFillerFilter — calls the Python filler_filter.py subprocess
// ---------------------------------------------------------------------------

/// Filler-word removal using sentence embeddings.
///
/// Measures cosine similarity between each word and a known filler lexicon.
/// Words exceeding the threshold are removed.  Multi-word fillers ("you know",
/// "I mean") are handled by exact substring match before embedding check.
///
/// This is dramatically faster and cheaper than an LLM call — no network
/// round-trip, no token cost, runs entirely on-device.
pub struct EmbeddingFillerFilter {
    script_path: PathBuf,
    python_path: PathBuf,
}

impl EmbeddingFillerFilter {
    pub fn new(script_path: impl Into<PathBuf>) -> Self {
        Self {
            script_path: script_path.into(),
            python_path: "python3".into(),
        }
    }

    pub fn with_python(mut self, python_path: impl Into<PathBuf>) -> Self {
        self.python_path = python_path.into();
        self
    }
}

#[async_trait]
impl TextFilter for EmbeddingFillerFilter {
    async fn filter(&self, text: &str) -> Result<FilterResult, FilterError> {
        use tokio::io::AsyncWriteExt;
        use tokio::process::Command;

        let req = serde_json::json!({ "text": text });
        let req_str = serde_json::to_string(&req).map_err(|e| FilterError {
            message: format!("json serialize: {e}"),
        })?;

        let mut child = Command::new(&self.python_path)
            .arg(&self.script_path)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| FilterError {
                message: format!("spawn filler_filter.py: {e}"),
            })?;

        // Write request to stdin
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(format!("{req_str}\n").as_bytes())
                .await
                .map_err(|e| FilterError {
                    message: format!("write stdin: {e}"),
                })?;
            // Close stdin to signal EOF
        }

        let output = child.wait_with_output().await.map_err(|e| FilterError {
            message: format!("wait: {e}"),
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(FilterError {
                message: format!("filler_filter.py failed: {stderr}"),
            });
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let resp: serde_json::Value = serde_json::from_str(stdout.trim()).map_err(|e| FilterError {
            message: format!("json parse response: {e} (raw: {stdout})"),
        })?;

        if let Some(err) = resp.get("error") {
            return Err(FilterError {
                message: format!("filter script error: {err}"),
            });
        }

        let filtered = resp["filtered"].as_str().unwrap_or(text).to_string();
        let removed: Vec<String> = resp["removed"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        Ok(FilterResult {
            text: filtered,
            removed,
        })
    }
}

// ---------------------------------------------------------------------------
// SimpleFillerFilter — tiered, position-aware, zero-dependency
// ---------------------------------------------------------------------------

/// Filler removal using tiered word matching and position heuristics.
///
/// Fillers are split into two tiers:
/// - **Pure fillers** (um, uh, er, ah) — removed unconditionally.
/// - **Contextual fillers** (so, well, basically, actually) — removed only at
///   sentence-initial position, since they serve as discourse markers there
///   but act as real content words mid-sentence ("it went well", "do so").
///
/// Multi-word fillers ("you know", "I mean") are matched as bigrams.
pub struct SimpleFillerFilter {
    /// Always removed regardless of position.
    pure_fillers: Vec<String>,
    /// Removed only when sentence-initial (first word, or after punctuation).
    contextual_fillers: Vec<String>,
    /// Multi-word fillers matched as consecutive word pairs.
    multi_fillers: Vec<Vec<String>>,
}

impl SimpleFillerFilter {
    pub fn english() -> Self {
        Self {
            pure_fillers: vec![
                "um", "uh", "uhm", "umm", "hmm", "er", "ah", "eh",
            ]
            .into_iter()
            .map(String::from)
            .collect(),

            contextual_fillers: vec![
                "so", "well", "basically", "actually", "literally", "right",
            ]
            .into_iter()
            .map(String::from)
            .collect(),

            multi_fillers: vec![
                vec!["you".into(), "know".into()],
                vec!["i".into(), "mean".into()],
                vec!["you".into(), "see".into()],
                vec!["sort".into(), "of".into()],
                vec!["kind".into(), "of".into()],
            ],
        }
    }

    pub fn with_pure_fillers(mut self, fillers: Vec<String>) -> Self {
        self.pure_fillers = fillers;
        self
    }

    pub fn with_contextual_fillers(mut self, fillers: Vec<String>) -> Self {
        self.contextual_fillers = fillers;
        self
    }

    /// Japanese filler filter.
    ///
    /// Japanese fillers appear as comma-delimited segments in whisper output
    /// (e.g. "あの、" or "えーと、"), so matching is done per-segment rather
    /// than per-word.
    ///
    /// Also handles common ASR artifacts where whisper converts fillers to
    /// kanji (えーと → 映像, えー → 映映).
    pub fn japanese() -> Self {
        Self {
            pure_fillers: vec![
                // Hesitation markers
                "えーと", "えっと", "えー", "あー", "うーん", "うん",
                "ああ", "ええ",
                // Common ASR misrecognitions of fillers
                "映像", "映映",
            ]
            .into_iter()
            .map(String::from)
            .collect(),

            contextual_fillers: vec![
                // Discourse markers — filler at sentence start, content word otherwise
                "あの", "まあ", "その", "なんか", "ほら", "やっぱり",
                "まあまあ", "ちょっと",
            ]
            .into_iter()
            .map(String::from)
            .collect(),

            multi_fillers: vec![
                vec!["なんて".into(), "いうか".into()],
                vec!["何て".into(), "いうか".into()],
            ],
        }
    }

    /// Returns true if position `i` is sentence-initial: either the first word,
    /// or preceded only by removed words / punctuation.
    fn is_sentence_initial(words: &[&str], idx: usize, removed_indices: &[bool]) -> bool {
        if idx == 0 {
            return true;
        }
        // Walk backwards — if every preceding word was removed or ends with
        // sentence-ending punctuation, this position is sentence-initial.
        for j in (0..idx).rev() {
            if removed_indices[j] {
                continue;
            }
            // Previous kept word ends with sentence punctuation → we're at a boundary
            let prev = words[j];
            if prev.ends_with('.') || prev.ends_with('!') || prev.ends_with('?') || prev.ends_with(',') {
                return true;
            }
            return false;
        }
        true
    }
}

#[async_trait]
impl TextFilter for SimpleFillerFilter {
    async fn filter(&self, text: &str) -> Result<FilterResult, FilterError> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let n = words.len();
        let mut removed_indices = vec![false; n];
        let mut removed_labels: Vec<String> = Vec::new();

        // Pass 1: remove multi-word fillers
        let mut i = 0;
        'outer: while i + 1 < n {
            for mf in &self.multi_fillers {
                if mf.len() == 2
                    && !removed_indices[i]
                    && !removed_indices[i + 1]
                {
                    let w0 = words[i].to_lowercase();
                    let w1 = words[i + 1].to_lowercase();
                    let clean0: String = w0.chars().take_while(|c| c.is_alphanumeric()).collect();
                    let clean1: String = w1.chars().take_while(|c| c.is_alphanumeric()).collect();
                    if clean0 == mf[0] && clean1 == mf[1] {
                        removed_indices[i] = true;
                        removed_indices[i + 1] = true;
                        removed_labels.push(format!("{} {}", words[i], words[i + 1]));
                        i += 2;
                        // Skip the trailing `i += 1` and re-evaluate the
                        // while bound, otherwise the next iteration of
                        // `for mf` would index past the end when `i` has
                        // just been advanced to `n`.
                        continue 'outer;
                    }
                }
            }
            i += 1;
        }

        // Pass 2: remove pure fillers (unconditional)
        for i in 0..n {
            if removed_indices[i] {
                continue;
            }
            let lower = words[i].to_lowercase();
            let clean: String = lower.chars().take_while(|c| c.is_alphanumeric()).collect();
            if self.pure_fillers.contains(&clean) {
                removed_indices[i] = true;
                removed_labels.push(words[i].to_string());
            }
        }

        // Pass 3: remove contextual fillers only at sentence-initial position
        for i in 0..n {
            if removed_indices[i] {
                continue;
            }
            let lower = words[i].to_lowercase();
            let clean: String = lower.chars().take_while(|c| c.is_alphanumeric()).collect();
            if self.contextual_fillers.contains(&clean)
                && Self::is_sentence_initial(&words, i, &removed_indices)
            {
                removed_indices[i] = true;
                removed_labels.push(words[i].to_string());
            }
        }

        let kept: Vec<&str> = words
            .iter()
            .enumerate()
            .filter(|(i, _)| !removed_indices[*i])
            .map(|(_, w)| *w)
            .collect();

        Ok(FilterResult {
            text: kept.join(" "),
            removed: removed_labels,
        })
    }
}

// ---------------------------------------------------------------------------
// JapaneseFillerFilter — comma-segment based filler removal for Japanese
// ---------------------------------------------------------------------------

/// Filler removal for Japanese text from ASR output.
///
/// Japanese ASR output uses 、 (ideographic comma) as a natural delimiter.
/// Fillers appear as short segments between commas: "えーと、" "あの、" "まあ、".
/// This filter splits on 、, checks each segment against filler lexicons,
/// and removes matching segments.
///
/// Also handles ASR artifacts where whisper converts Japanese fillers into
/// kanji (e.g. えーと → 映像, えー → 映映).
pub struct JapaneseFillerFilter {
    /// Always removed regardless of position.
    pure_fillers: Vec<String>,
    /// Removed only at sentence-initial position.
    contextual_fillers: Vec<String>,
}

impl JapaneseFillerFilter {
    pub fn new() -> Self {
        Self {
            pure_fillers: vec![
                "えーと", "えっと", "えー", "あー", "うーん", "うん",
                "ああ", "ええ",
                // Common ASR misrecognitions
                "映像", "映映",
            ]
            .into_iter()
            .map(String::from)
            .collect(),

            contextual_fillers: vec![
                "あの", "まあ", "その", "なんか", "ほら", "やっぱり",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        }
    }
}

impl Default for JapaneseFillerFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TextFilter for JapaneseFillerFilter {
    async fn filter(&self, text: &str) -> Result<FilterResult, FilterError> {
        // Split by Japanese comma (、) keeping track of segments
        let segments: Vec<&str> = text.split('、').collect();
        let n = segments.len();
        let mut removed_indices = vec![false; n];
        let mut removed_labels: Vec<String> = Vec::new();

        // Pass 1: pure fillers — remove unconditionally
        for i in 0..n {
            let trimmed = segments[i].trim();
            if trimmed.is_empty() {
                continue;
            }
            if self.pure_fillers.iter().any(|f| trimmed == f) {
                removed_indices[i] = true;
                removed_labels.push(trimmed.to_string());
            }
        }

        // Pass 2: contextual fillers — remove only at sentence-initial position
        for i in 0..n {
            if removed_indices[i] {
                continue;
            }
            let trimmed = segments[i].trim();
            if trimmed.is_empty() {
                continue;
            }
            // Check if this is sentence-initial
            let is_initial = (0..i).all(|j| {
                removed_indices[j] || segments[j].trim().is_empty()
            });
            // Also check after sentence-ending punctuation (。)
            let after_period = i > 0 && !removed_indices[i - 1]
                && segments[i - 1].trim().ends_with('。');

            if self.contextual_fillers.iter().any(|f| trimmed == f)
                && (is_initial || after_period)
            {
                removed_indices[i] = true;
                removed_labels.push(trimmed.to_string());
            }
        }

        // Pass 3: standalone contextual fillers — Japanese-specific heuristic.
        //
        // In Japanese ASR output, fillers appear as independent comma-delimited
        // segments: "あの、お立ち合いの中に" → segments ["あの", "お立ち合いの中に"].
        //
        // A contextual filler word that IS the entire segment (not part of a
        // longer phrase) is almost certainly a filler, not a content word.
        // Compare: "あの" (standalone → filler) vs "あの人が来た" (part of
        // phrase → demonstrative "that person").
        for i in 0..n {
            if removed_indices[i] {
                continue;
            }
            let trimmed = segments[i].trim();
            if trimmed.is_empty() {
                continue;
            }
            // If the entire segment is exactly a contextual filler word,
            // it's a standalone filler — remove it.
            if self.contextual_fillers.iter().any(|f| trimmed == f) {
                removed_indices[i] = true;
                removed_labels.push(trimmed.to_string());
            }
        }

        let kept: Vec<&str> = segments
            .iter()
            .enumerate()
            .filter(|(i, _)| !removed_indices[*i])
            .map(|(_, s)| *s)
            .collect();

        // Rejoin with 、, then clean up leading/trailing commas and double commas
        let mut result = kept.join("、");
        // Remove leading comma
        while result.starts_with('、') {
            result = result[3..].to_string(); // 、 is 3 bytes in UTF-8
        }
        // Remove double commas left by removed segments
        while result.contains("、、") {
            result = result.replace("、、", "、");
        }

        Ok(FilterResult {
            text: result.trim().to_string(),
            removed: removed_labels,
        })
    }
}

// ---------------------------------------------------------------------------
// ChineseFillerFilter — Mandarin Chinese rule-based filter
// ---------------------------------------------------------------------------

/// Rule-based filter for Mandarin Chinese filler tokens.
///
/// Closely mirrors `JapaneseFillerFilter`: splits the input by Chinese
/// commas (`，`), then applies a 3-pass rule set per segment:
///
/// 1. **Pure fillers** — `嗯`, `呃`, `哦`, etc. Always removed.
/// 2. **Contextual fillers** — `那个`, `这个`, `就是`, `然后`, `怎么说`.
///    These are also content words in their non-filler use (`那个`
///    "that one", `这个` "this one", `然后` "and then") so we only
///    remove them when they appear at sentence-initial position or
///    just after a `。` sentence boundary.
/// 3. **Standalone contextual fillers** — when a contextual filler
///    forms the entire `，`-delimited segment (e.g. "那个，我们需要…"
///    → segment 0 is the literal string "那个"), it's almost certainly
///    a filler, not a demonstrative.
///
/// The closed-set lexicon is informed by the Mandarin disfluency
/// literature surveyed in `docs/evaluation.md` §5.6.2 — well-attested
/// surface forms with high inter-annotator agreement (κ 0.80–0.95).
/// `啊` is intentionally **not** included as a pure filler because it
/// frequently appears as a sentence-final particle that's part of the
/// content; downgrading that judgement to false-positive territory
/// is worse than leaving it in.
pub struct ChineseFillerFilter {
    /// Always removed regardless of position.
    pure_fillers: Vec<String>,
    /// Removed only at sentence-initial position or as a standalone segment.
    contextual_fillers: Vec<String>,
}

impl ChineseFillerFilter {
    pub fn new() -> Self {
        Self {
            pure_fillers: vec!["嗯", "呃", "哦", "唉", "呀"]
                .into_iter()
                .map(String::from)
                .collect(),
            contextual_fillers: vec!["那个", "这个", "就是", "然后", "怎么说"]
                .into_iter()
                .map(String::from)
                .collect(),
        }
    }
}

impl Default for ChineseFillerFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TextFilter for ChineseFillerFilter {
    async fn filter(&self, text: &str) -> Result<FilterResult, FilterError> {
        // Split by Chinese full-width comma. Half-width "," is rare in
        // Whisper's zh output but if it appears, we treat it the same.
        let normalized = text.replace(',', "，");
        let segments: Vec<&str> = normalized.split('，').collect();
        let n = segments.len();
        let mut removed_indices = vec![false; n];
        let mut removed_labels: Vec<String> = Vec::new();

        // Pass 1: pure fillers — remove unconditionally.
        for i in 0..n {
            let trimmed = segments[i].trim();
            if trimmed.is_empty() {
                continue;
            }
            if self.pure_fillers.iter().any(|f| trimmed == f) {
                removed_indices[i] = true;
                removed_labels.push(trimmed.to_string());
            }
        }

        // Pass 2: contextual fillers at sentence-initial position.
        for i in 0..n {
            if removed_indices[i] {
                continue;
            }
            let trimmed = segments[i].trim();
            if trimmed.is_empty() {
                continue;
            }
            let is_initial = (0..i).all(|j| {
                removed_indices[j] || segments[j].trim().is_empty()
            });
            let after_period = i > 0
                && !removed_indices[i - 1]
                && segments[i - 1].trim().ends_with('。');

            if self.contextual_fillers.iter().any(|f| trimmed == f)
                && (is_initial || after_period)
            {
                removed_indices[i] = true;
                removed_labels.push(trimmed.to_string());
            }
        }

        // Pass 3: standalone contextual fillers (the whole segment IS
        // the filler word). 「那个、我们需要…」 vs. 「那个人」.
        for i in 0..n {
            if removed_indices[i] {
                continue;
            }
            let trimmed = segments[i].trim();
            if trimmed.is_empty() {
                continue;
            }
            if self.contextual_fillers.iter().any(|f| trimmed == f) {
                removed_indices[i] = true;
                removed_labels.push(trimmed.to_string());
            }
        }

        let kept: Vec<&str> = segments
            .iter()
            .enumerate()
            .filter(|(i, _)| !removed_indices[*i])
            .map(|(_, s)| *s)
            .collect();

        // Rejoin with 「，」, then clean up. Both `、` and `，` are
        // 3 bytes in UTF-8 so the byte-skip from JapaneseFillerFilter
        // applies directly.
        let mut result = kept.join("，");
        while result.starts_with('，') {
            result = result[3..].to_string();
        }
        while result.contains("，，") {
            result = result.replace("，，", "，");
        }

        Ok(FilterResult {
            text: result.trim().to_string(),
            removed: removed_labels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn simple_filler_filter_removes_pure_fillers() {
        let filter = SimpleFillerFilter::english();
        let result = filter.filter("um I think uh we should deploy").await.unwrap();
        assert_eq!(result.text, "I think we should deploy");
        assert!(result.removed.contains(&"um".to_string()));
        assert!(result.removed.contains(&"uh".to_string()));
    }

    #[tokio::test]
    async fn simple_filler_filter_preserves_clean_text() {
        let filter = SimpleFillerFilter::english();
        let result = filter.filter("deploy the server now").await.unwrap();
        assert_eq!(result.text, "deploy the server now");
        assert!(result.removed.is_empty());
    }

    #[tokio::test]
    async fn contextual_filler_removed_at_sentence_start() {
        let filter = SimpleFillerFilter::english();
        let result = filter.filter("So I think we should deploy").await.unwrap();
        assert_eq!(result.text, "I think we should deploy");
        assert!(result.removed.contains(&"So".to_string()));
    }

    #[tokio::test]
    async fn contextual_word_preserved_mid_sentence() {
        let filter = SimpleFillerFilter::english();
        // "so" as conjunction mid-sentence should be kept
        let result = filter.filter("I think so we should deploy").await.unwrap();
        assert!(result.text.contains("so"));
    }

    #[tokio::test]
    async fn well_preserved_as_content_word() {
        let filter = SimpleFillerFilter::english();
        // "well" as adverb mid-sentence
        let result = filter.filter("it went well today").await.unwrap();
        assert_eq!(result.text, "it went well today");
    }

    #[tokio::test]
    async fn well_removed_at_sentence_start() {
        let filter = SimpleFillerFilter::english();
        let result = filter.filter("Well I think we should deploy").await.unwrap();
        assert_eq!(result.text, "I think we should deploy");
    }

    #[tokio::test]
    async fn multi_word_filler_removed() {
        let filter = SimpleFillerFilter::english();
        let result = filter
            .filter("I think you know we should deploy")
            .await
            .unwrap();
        assert_eq!(result.text, "I think we should deploy");
        assert!(result.removed.contains(&"you know".to_string()));
    }

    /// Regression: a multi-word filler whose match leaves `i` exactly at
    /// the end of the word list used to fall through to the next mf in
    /// the inner `for` loop and index `removed_indices[i + 1]`
    /// out-of-bounds.
    #[tokio::test]
    async fn multi_word_filler_at_end_of_input() {
        let filter = SimpleFillerFilter::english();
        let result = filter.filter("deploy now you know").await.unwrap();
        assert!(!result.text.contains("you know"), "{}", result.text);
        assert!(result.removed.contains(&"you know".to_string()));
    }

    #[tokio::test]
    async fn combined_pure_contextual_and_multi() {
        let filter = SimpleFillerFilter::english();
        let result = filter
            .filter("Well um I think you know we should uh deploy")
            .await
            .unwrap();
        assert_eq!(result.text, "I think we should deploy");
    }

    #[tokio::test]
    async fn jfk_speech_contextual_so_preserved() {
        let filter = SimpleFillerFilter::english();
        // "And so" — "so" follows "And" which is a kept word mid-sentence,
        // so it should be preserved
        let result = filter
            .filter("And so my fellow Americans ask not what your country can do for you")
            .await
            .unwrap();
        assert!(result.text.contains("so"), "mid-sentence 'so' should be kept: {}", result.text);
    }

    // --- Japanese filler filter tests ---

    #[tokio::test]
    async fn japanese_pure_filler_removed() {
        let filter = JapaneseFillerFilter::new();
        let result = filter.filter("えーと、拙者親方と申すは").await.unwrap();
        assert_eq!(result.text, "拙者親方と申すは");
        assert!(result.removed.contains(&"えーと".to_string()));
    }

    #[tokio::test]
    async fn japanese_contextual_filler_at_start() {
        let filter = JapaneseFillerFilter::new();
        let result = filter.filter("あの、お立ち合いの中に").await.unwrap();
        assert_eq!(result.text, "お立ち合いの中に");
        assert!(result.removed.contains(&"あの".to_string()));
    }

    #[tokio::test]
    async fn japanese_contextual_filler_standalone_removed() {
        let filter = JapaneseFillerFilter::new();
        // "あの" as standalone segment between commas → filler, removed
        let result = filter.filter("拙者親方と申すは、あの、お立ち合いの中に").await.unwrap();
        assert!(!result.text.contains("あの"), "standalone あの should be removed: {}", result.text);
        assert!(result.removed.contains(&"あの".to_string()));
    }

    #[tokio::test]
    async fn japanese_contextual_word_in_phrase_preserved() {
        let filter = JapaneseFillerFilter::new();
        // "あの" as part of a longer phrase → demonstrative, preserved
        let result = filter.filter("拙者親方と申すは、あの人が来た").await.unwrap();
        assert!(result.text.contains("あの人が来た"), "あの in phrase should be kept: {}", result.text);
    }

    #[tokio::test]
    async fn japanese_multiple_fillers() {
        let filter = JapaneseFillerFilter::new();
        let result = filter
            .filter("えーと、あの、お江戸を発って、えー、二十里上方")
            .await
            .unwrap();
        assert!(result.removed.contains(&"えーと".to_string()));
        assert!(result.removed.contains(&"えー".to_string()));
        assert!(result.text.contains("お江戸を発って"));
        assert!(result.text.contains("二十里上方"));
    }

    #[tokio::test]
    async fn japanese_asr_artifact_removed() {
        let filter = JapaneseFillerFilter::new();
        let result = filter.filter("映像、拙者親方と申すは").await.unwrap();
        assert_eq!(result.text, "拙者親方と申すは");
        assert!(result.removed.contains(&"映像".to_string()));
    }

    #[tokio::test]
    async fn japanese_clean_text_preserved() {
        let filter = JapaneseFillerFilter::new();
        let result = filter
            .filter("拙者親方と申すは、お立ち合いの中に、御存じのお方もございましょうが")
            .await
            .unwrap();
        assert_eq!(
            result.text,
            "拙者親方と申すは、お立ち合いの中に、御存じのお方もございましょうが"
        );
        assert!(result.removed.is_empty());
    }

    #[tokio::test]
    async fn japanese_uiro_with_fillers_full() {
        let filter = JapaneseFillerFilter::new();
        let result = filter
            .filter("映像、せっしゃおやかたとモースは、あの、おたち合いの中に、まあ、ご存知のおかたもございましょうが、その、お江戸をたってにじゅうりかみがた")
            .await
            .unwrap();
        assert!(!result.text.contains("映像"), "ASR artifact should be removed");
        assert!(result.text.contains("せっしゃおやかたとモースは"));
        assert!(result.text.contains("ご存知のおかたもございましょうが"));
    }

    // -----------------------------------------------------------------
    // ChineseFillerFilter
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn chinese_pure_filler_removed() {
        let filter = ChineseFillerFilter::new();
        let result = filter.filter("嗯，我们需要更新数据库").await.unwrap();
        assert_eq!(result.text, "我们需要更新数据库");
        assert!(result.removed.contains(&"嗯".to_string()));
    }

    #[tokio::test]
    async fn chinese_two_pure_fillers() {
        let filter = ChineseFillerFilter::new();
        let result = filter.filter("呃，明天发布新版本，哦").await.unwrap();
        assert!(!result.text.contains("呃"), "{}", result.text);
        assert!(!result.text.contains("哦"), "{}", result.text);
        assert!(result.text.contains("明天发布新版本"));
    }

    #[tokio::test]
    async fn chinese_contextual_filler_at_sentence_start() {
        let filter = ChineseFillerFilter::new();
        let result = filter.filter("那个，测试全部通过了").await.unwrap();
        assert_eq!(result.text, "测试全部通过了");
        assert!(result.removed.contains(&"那个".to_string()));
    }

    #[tokio::test]
    async fn chinese_contextual_word_in_phrase_preserved() {
        // "那个" inside a noun phrase ("那个人" — "that person") must
        // NOT be removed; only the standalone-segment form is a filler.
        let filter = ChineseFillerFilter::new();
        let result = filter.filter("那个人是谁").await.unwrap();
        assert_eq!(result.text, "那个人是谁");
        assert!(result.removed.is_empty());
    }

    #[tokio::test]
    async fn chinese_standalone_contextual_filler_mid_sentence() {
        // 「那个」 as its own 「，」-delimited segment — even mid-sentence
        // — is the filler usage (Pass 3).
        let filter = ChineseFillerFilter::new();
        let result = filter
            .filter("我觉得，那个，这个事情应该这样处理")
            .await
            .unwrap();
        assert!(!result.text.contains("，那个，"), "{}", result.text);
        assert!(result.text.contains("我觉得"));
        assert!(result.text.contains("这个事情应该这样处理"));
    }

    #[tokio::test]
    async fn chinese_clean_text_preserved() {
        let filter = ChineseFillerFilter::new();
        let result = filter
            .filter("配置文件已经修改，缓存命中率不错")
            .await
            .unwrap();
        assert_eq!(result.text, "配置文件已经修改，缓存命中率不错");
        assert!(result.removed.is_empty());
    }

    #[tokio::test]
    async fn chinese_halfwidth_comma_normalised() {
        // Whisper for zh almost always emits 「，」 but if a half-width
        // comma slips through we should still segment correctly.
        let filter = ChineseFillerFilter::new();
        let result = filter.filter("嗯,我们需要更新数据库").await.unwrap();
        assert!(!result.text.contains("嗯"));
        assert!(result.text.contains("我们需要更新数据库"));
    }

    #[tokio::test]
    async fn chinese_contextual_after_sentence_end() {
        // After a 。 sentence break the contextual filler appears as
        // its own 「，」-delimited segment — Pass 2's after_period
        // branch picks it up.
        let filter = ChineseFillerFilter::new();
        let result = filter
            .filter("第一个任务完成。，然后，开始第二个")
            .await
            .unwrap();
        assert!(!result.text.contains("然后"), "{}", result.text);
        assert!(result.text.contains("开始第二个"));
    }
}
