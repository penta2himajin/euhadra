use async_trait::async_trait;

use crate::traits::OutputEmitter;
use crate::types::{EmitResult, RefinementOutput};

// ---------------------------------------------------------------------------
// ClipboardEmitter — paste via system clipboard
// ---------------------------------------------------------------------------

/// Emits text by writing to the system clipboard.
///
/// On supported platforms, this performs:
/// 1. Save current clipboard contents
/// 2. Write refined text to clipboard
/// 3. (Caller is responsible for triggering Cmd+V / Ctrl+V if needed)
/// 4. Restore previous clipboard contents (via `undo`)
///
/// Uses the `arboard` crate for cross-platform clipboard access
/// (macOS, Windows, Linux/X11/Wayland).
pub struct ClipboardEmitter {
    /// If true, save and restore the previous clipboard contents on emit/undo.
    preserve_clipboard: bool,
    /// Stashed clipboard content for restoration.
    previous: std::sync::Arc<tokio::sync::Mutex<Option<String>>>,
}

impl ClipboardEmitter {
    pub fn new() -> Self {
        Self {
            preserve_clipboard: true,
            previous: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
        }
    }

    /// Create an emitter that does NOT preserve the previous clipboard contents.
    pub fn without_preservation() -> Self {
        Self {
            preserve_clipboard: false,
            previous: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
        }
    }
}

impl Default for ClipboardEmitter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl OutputEmitter for ClipboardEmitter {
    async fn emit(&self, output: RefinementOutput) -> EmitResult {
        let text = match &output {
            RefinementOutput::TextInsertion { text, .. } => text.clone(),
            RefinementOutput::Command { action, .. } => action.clone(),
            RefinementOutput::StructuredInput { text, .. } => {
                text.clone().unwrap_or_default()
            }
        };

        // Run clipboard operations on a blocking thread (arboard is sync)
        let preserve = self.preserve_clipboard;
        let previous = self.previous.clone();

        let result = tokio::task::spawn_blocking(move || {
            let mut clipboard = match arboard::Clipboard::new() {
                Ok(c) => c,
                Err(e) => return EmitResult::fail(format!("clipboard init: {e}")),
            };

            // Save previous contents
            if preserve {
                let prev = clipboard.get_text().ok();
                // Store for later undo — we do this synchronously since we're
                // already in a blocking context, but need to use the Arc<Mutex>
                // We'll use try_lock since we're in a sync context
                if let Ok(mut guard) = previous.try_lock() {
                    *guard = prev;
                }
            }

            // Write new text
            match clipboard.set_text(&text) {
                Ok(()) => {
                    tracing::info!(len = text.len(), "text written to clipboard");
                    EmitResult::ok()
                }
                Err(e) => EmitResult::fail(format!("clipboard write: {e}")),
            }
        })
        .await;

        match result {
            Ok(r) => r,
            Err(e) => EmitResult::fail(format!("clipboard task: {e}")),
        }
    }

    async fn undo(&self) -> EmitResult {
        if !self.preserve_clipboard {
            return EmitResult::fail("clipboard preservation disabled");
        }

        let previous = self.previous.clone();

        let result = tokio::task::spawn_blocking(move || {
            let prev = match previous.try_lock() {
                Ok(guard) => guard.clone(),
                Err(_) => return EmitResult::fail("could not access previous clipboard"),
            };

            match prev {
                Some(text) => {
                    let mut clipboard = match arboard::Clipboard::new() {
                        Ok(c) => c,
                        Err(e) => return EmitResult::fail(format!("clipboard init: {e}")),
                    };
                    match clipboard.set_text(&text) {
                        Ok(()) => EmitResult::ok(),
                        Err(e) => EmitResult::fail(format!("clipboard restore: {e}")),
                    }
                }
                None => {
                    // Nothing to restore — previous clipboard was empty
                    EmitResult::ok()
                }
            }
        })
        .await;

        match result {
            Ok(r) => r,
            Err(e) => EmitResult::fail(format!("clipboard undo task: {e}")),
        }
    }
}
