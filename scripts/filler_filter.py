#!/usr/bin/env python3
"""Embedding-based filler word detector for euhadra.

Reads JSON from stdin: {"text": "um I think we should uh deploy"}
Writes JSON to stdout: {"filtered": "I think we should deploy", "removed": ["um", "uh"]}

Two-tier detection:
  - Pure fillers (um, uh, er): removed unconditionally via embedding similarity
  - Contextual fillers (so, well, basically): removed only at sentence-initial
    position, since they're content words mid-sentence ("it went well")

Uses fastembed (ONNX-based) for lightweight, fast inference.
"""

import json
import sys
import numpy as np

# Lazy init
_model = None
_pure_filler_embeddings = None

# -------------------------------------------------------------------
# Filler lexicons — tiered
# -------------------------------------------------------------------

# Tier 1: Always fillers regardless of position
PURE_FILLERS = [
    "um", "uh", "uhm", "umm", "hmm", "er", "ah", "eh",
]

# Tier 2: Fillers only at sentence-initial position
# Mid-sentence these are real words: "it went well", "I think so", "that's actually true"
CONTEXTUAL_FILLERS = [
    "so", "well", "basically", "actually", "literally", "right",
]

# Multi-word fillers — matched as exact substrings before embedding check
MULTI_FILLERS = [
    "you know", "i mean", "you see", "sort of", "kind of",
    "or something", "or whatever",
]

PURE_THRESHOLD = 0.82
CONTEXTUAL_THRESHOLD = 0.88  # higher bar — only remove when very filler-like

def get_model():
    global _model, _pure_filler_embeddings
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding('BAAI/bge-small-en-v1.5')
        _pure_filler_embeddings = list(_model.embed(PURE_FILLERS))
    return _model, _pure_filler_embeddings

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def max_filler_sim(word_emb, filler_embs):
    return max(cosine_sim(word_emb, f) for f in filler_embs)

def is_sentence_initial(words, idx, removed):
    """Check if position idx is sentence-initial (first word or after punctuation/removed words)."""
    if idx == 0:
        return True
    for j in range(idx - 1, -1, -1):
        if removed[j]:
            continue
        prev = words[j]
        if prev.endswith(('.', '!', '?', ',')):
            return True
        return False
    return True

def remove_multi_fillers(text):
    """Remove known multi-word filler phrases (case-insensitive)."""
    removed = []
    lower = text.lower()
    for phrase in MULTI_FILLERS:
        while phrase in lower:
            idx = lower.index(phrase)
            original = text[idx:idx+len(phrase)]
            removed.append(original)
            text = text[:idx] + text[idx+len(phrase):]
            lower = text.lower()
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip(), removed

def filter_fillers(text):
    """Remove filler words using tiered embedding similarity + position heuristics."""
    model, pure_embs = get_model()
    
    # Phase 1: remove multi-word fillers by exact match
    text, multi_removed = remove_multi_fillers(text)
    
    # Phase 2: embedding-based single word filtering
    words = text.split()
    if not words:
        return "", multi_removed
    
    word_embs = list(model.embed(words))
    n = len(words)
    removed_flags = [False] * n
    single_removed = []
    
    # Pass A: pure fillers — remove unconditionally
    for i in range(n):
        w_lower = words[i].lower().strip(".,!?;:")
        sim = max_filler_sim(word_embs[i], pure_embs)
        if sim >= PURE_THRESHOLD and w_lower in PURE_FILLERS:
            removed_flags[i] = True
            single_removed.append(words[i])
    
    # Pass B: contextual fillers — remove only at sentence-initial position
    for i in range(n):
        if removed_flags[i]:
            continue
        w_lower = words[i].lower().strip(".,!?;:")
        if w_lower in CONTEXTUAL_FILLERS and is_sentence_initial(words, i, removed_flags):
            removed_flags[i] = True
            single_removed.append(words[i])
    
    kept = [w for i, w in enumerate(words) if not removed_flags[i]]
    filtered = " ".join(kept)
    return filtered, multi_removed + single_removed

def main():
    """Process JSON lines from stdin."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            text = req.get("text", "")
            filtered, removed = filter_fillers(text)
            resp = {"filtered": filtered, "removed": removed}
            print(json.dumps(resp), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    main()
