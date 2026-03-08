"""
Offline indexing script.

Creates a lightweight document index and saves artifacts to models/.
This keeps the same lifecycle pattern as the original mlops-mini-prod:
offline step -> saved artifacts -> API loads artifacts -> inference.

Artifacts:
- models/index.json
- models/metadata.json
"""

from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone
import logging
import math
import re
import hashlib

from pypdf import PdfReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

INDEX_PATH = MODEL_DIR / "index.json"
METADATA_PATH = MODEL_DIR / "metadata.json"

# ---- MVP knobs (keep simple, deterministic, fast) ----
EMBED_DIM = 64
CHUNK_CHARS = 900
CHUNK_OVERLAP = 120

# Built-in sample docs so Docker build never fails even without PDFs.
SAMPLE_DOCS = [
    {
        "doc_id": "sample-policy-001",
        "title": "Sample Policy (Synthetic)",
        "pages": [
            "Policy Number: P-001. Coverage includes fire and theft. Exclusions include flood.",
            "Endorsement A1: Adds coverage for business interruption. Schedule: deductible is $5,000.",
        ],
    }
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _chunk_text(text: str, chunk_chars: int, overlap: int) -> list[str]:
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks



"""
Deterministic hashing embedding (fast, no external models).
This is a placeholder for Azure/OpenAI embeddings later.

It is intentionally simple to keep CI fast and offline.
"""
def _embed(text: str, dim: int = EMBED_DIM) -> list[float]:

    vec = [0.0] * dim
    if not text:
        return vec

    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    if not tokens:
        return vec

    for tok in tokens:
        h = hashlib.sha256(tok.encode("utf-8")).digest()
        # Use first 2 bytes to pick a bucket
        bucket = (h[0] << 8 | h[1]) % dim
        vec[bucket] += 1.0

    # L2 normalize
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _read_pdf_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        pages.append(_normalize_text(txt))
    return pages


def build_index() -> dict:
    """
    Returns:
      {
        "items": [
          {
            "doc_id": str,
            "page": int,
            "chunk_id": str,
            "text": str,
            "embedding": [float...]
          }, ...
        ]
      }
    """
    items: list[dict] = []

    # 1) sample docs (always present)
    for doc in SAMPLE_DOCS:
        doc_id = doc["doc_id"]
        for i, page_text in enumerate(doc["pages"], start=1):
            page_text = _normalize_text(page_text)
            for j, chunk in enumerate(_chunk_text(page_text, CHUNK_CHARS, CHUNK_OVERLAP), start=1):
                items.append(
                    {
                        "doc_id": doc_id,
                        "page": i,
                        "chunk_id": f"{doc_id}-p{i}-c{j}",
                        "text": chunk,
                        "embedding": _embed(chunk),
                    }
                )

    # 2) optional PDFs in ./data (so you can add real PDFs later without code changes)
    data_dir = Path("data")
    if data_dir.exists():
        pdfs = sorted(data_dir.glob("*.pdf"))
        if pdfs:
            logger.info("Found %d PDFs in ./data. Adding them to the index.", len(pdfs))
        for pdf_path in pdfs:
            doc_id = pdf_path.stem
            pages = _read_pdf_pages(pdf_path)
            for i, page_text in enumerate(pages, start=1):
                if not page_text:
                    continue
                for j, chunk in enumerate(_chunk_text(page_text, CHUNK_CHARS, CHUNK_OVERLAP), start=1):
                    items.append(
                        {
                            "doc_id": doc_id,
                            "page": i,
                            "chunk_id": f"{doc_id}-p{i}-c{j}",
                            "text": chunk,
                            "embedding": _embed(chunk),
                        }
                    )

    return {"items": items}


def main():
    logger.info("Starting document index build")

    index = build_index()
    INDEX_PATH.write_text(json.dumps(index, indent=2))

    metadata = {
        "indexed_at": _utc_now_iso(),
        "index_type": "json+hash-embeddings",
        "embedding_dim": EMBED_DIM,
        "chunk_chars": CHUNK_CHARS,
        "chunk_overlap": CHUNK_OVERLAP,
        "items": len(index["items"]),
        "docs": sorted({it["doc_id"] for it in index["items"]}),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    logger.info("Index artifacts saved to models/")
    print("Index built successfully")
    print(f"Items: {metadata['items']}")
    print(f"Docs: {len(metadata['docs'])}")


if __name__ == "__main__":
    main()

