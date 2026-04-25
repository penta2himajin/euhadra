//! Fixture loader for the layer-ablation runner (Phase A-2).
//!
//! Fixtures are tiny JSON-Lines files of `{reference, asr_hypothesis}`
//! pairs hand-crafted to exercise specific layer behaviours (filler
//! removal, self-correction, clean baseline). Audio is deliberately not
//! involved — we replay the `asr_hypothesis` through `MockAsr` and
//! measure what each post-ASR layer contributes.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fixture {
    pub id: String,
    pub category: String,
    pub reference: String,
    pub asr_hypothesis: String,
}

pub fn load_jsonl(path: &Path) -> std::io::Result<Vec<Fixture>> {
    let raw = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let f: Fixture = serde_json::from_str(line).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}:{}: {e}", path.display(), i + 1),
            )
        })?;
        out.push(f);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn jsonl_round_trip() {
        let dir = tempdir();
        let path = dir.join("fix.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"id":"a","category":"clean","reference":"hello","asr_hypothesis":"hello"}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"id":"b","category":"pure_filler","reference":"hi","asr_hypothesis":"um hi"}}"#
        )
        .unwrap();
        let fixtures = load_jsonl(&path).unwrap();
        assert_eq!(fixtures.len(), 2);
        assert_eq!(fixtures[0].id, "a");
        assert_eq!(fixtures[1].asr_hypothesis, "um hi");
    }

    fn tempdir() -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("euhadra-eval-fixtures-{}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
