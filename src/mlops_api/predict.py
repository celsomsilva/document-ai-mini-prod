"""
Inference logic (Document AI MVP).

Loads the document index lazily and answers questions using retrieval.
Returns grounded snippets with page-level citations (traceability).

Keeps the same interface style as the original project:
- load artifacts from models/
- predict(features: dict) -> dict
"""

from __future__ import annotations

import json
from pathlib import Path
import logging
import math
import re
import hashlib
from typing import Any


logger = logging.getLogger(__name__)

INDEX_PATH = Path("models/index.json")
METADATA_PATH = Path("models/metadata.json")

_index: dict | None = None
_metadata: dict | None = None



def _embed(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    if not text:
        return vec

    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    if not tokens:
        return vec

    for tok in tokens:
        h = hashlib.sha256(tok.encode("utf-8")).digest()
        bucket = (h[0] << 8 | h[1]) % dim
        vec[bucket] += 1.0

    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    # vectors are already normalized, so dot product is cosine
    return sum(x * y for x, y in zip(a, b))



"""
Preserves the same function name used by the ML version.
Now it loads the document index + metadata instead of a scikit model.
"""
def load_model():

    global _index, _metadata

    if _index is None:
        try:
            _index = json.loads(INDEX_PATH.read_text())
            _metadata = json.loads(METADATA_PATH.read_text())
            logger.info("Index loaded into memory")
        except FileNotFoundError as e:
            raise RuntimeError(
                "Index artifacts not found. "
                "Ensure the offline step has run (python -m mlops_api.train)."
            ) from e
        except Exception as e:
            raise RuntimeError("Failed to load index artifacts.") from e


def _retrieve(question: str, top_k: int = 3, doc_id: str | None = None) -> list[dict[str, Any]]:
    load_model()
    assert _index is not None
    assert _metadata is not None

    dim = int(_metadata.get("embedding_dim", 64))
    q_emb = _embed(question, dim)

    scored = []
    for it in _index.get("items", []):
        if doc_id and it.get("doc_id") != doc_id:
            continue
        score = _cosine(q_emb, it["embedding"])
        scored.append((score, it))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": s, **it} for s, it in scored[:top_k]]


"""
features expected:
  - question: str
  - top_k: int (optional)
  - doc_id: str (optional)
"""
def predict(features: dict):

    question = (features.get("question") or "").strip()
    if not question:
        raise ValueError("question must be a non-empty string")

    top_k = int(features.get("top_k") or 3)
    doc_id = features.get("doc_id")

    hits = _retrieve(question=question, top_k=top_k, doc_id=doc_id)

    if not hits:
        answer = "No relevant content found in the indexed documents."
        best = 0.0
        citations = []
    else:
        # MVP grounded answer: concatenate the most relevant snippets
        citations = [
            {
                "doc_id": h["doc_id"],
                "page": h["page"],
                "chunk_id": h["chunk_id"],
                "snippet": h["text"][:260] + ("..." if len(h["text"]) > 260 else ""),
                "score": round(float(h["score"]), 6),
            }
            for h in hits
        ]
        answer = _beautify_answer(question, hits)
        best = float(hits[0]["score"])
    return {
    "prediction": answer,
    "model_version": _metadata.get("model_version"),
    "rmse": round(best, 6),
    "citations": citations,
    }
        
        
        
        
"""
Convert retrieved chunks into a human-friendly sentence.

This is a lightweight generation layer that formats the answer
without using an LLM, preserving grounded content.
"""        
def _beautify_answer(question: str, hits: list[dict]) -> str:
    if not hits:
        return "I could not find the answer in the documents."

    best = hits[0]["text"]

    q = question.lower()

    if "deductible" in q:
        import re
        m = re.search(r"\$[\d,]+", best)
        if m:
            return f"The deductible is {m.group(0)}."

    if "policy number" in q:
        import re
        m = re.search(r"P-\d+", best)
        if m:
            return f"The policy number is {m.group(0)}."

    # fallback elegante
    return best

    # Keep the same response keys as your original contract
    return {
        "prediction": answer,
        "model_version": _metadata.get("indexed_at"),
        "rmse": best,  # In this MVP, we reuse 'rmse' as a retrieval score (0..1)
        "citations": citations,
    }

