"""
rag/runbook_rag.py — Runbook RAG using FAISS + all-MiniLM-L6-v2.

Completely separate from the memory service (Postgres). This is static,
baked into the Docker image at build time, read-only at runtime. Uses the
same embedding model as the memory service for consistency.

build_index() is called once at Docker build time.
query(text, top_k) is called during the reasoning loop when the agent
needs procedural runbook guidance.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import structlog

log = structlog.get_logger(__name__)

RUNBOOKS_DIR = Path(__file__).parent.parent / "knowledge_base" / "runbooks"
INDEX_PATH = Path(__file__).parent / "runbook_index.pkl"

# Module-level lazy-loaded FAISS index
_index = None
_documents: List[dict] = []
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def build_index() -> None:
    """
    Build and persist the FAISS index from runbook markdown files.
    Called once at Docker build time (Dockerfile RUN layer).
    """
    try:
        import faiss
        import numpy as np
    except ImportError:
        log.warning("rag.build_index.faiss_not_available")
        return

    if not RUNBOOKS_DIR.exists():
        log.warning("rag.build_index.runbooks_dir_missing", path=str(RUNBOOKS_DIR))
        return

    documents = []
    runbook_files = list(RUNBOOKS_DIR.glob("*.md"))
    if not runbook_files:
        log.warning("rag.build_index.no_runbooks_found")
        return

    for path in runbook_files:
        content = path.read_text(encoding="utf-8")
        # Chunk by H2 sections
        sections = content.split("\n## ")
        for i, section in enumerate(sections):
            chunk_text = section.strip()
            if len(chunk_text) < 50:
                continue
            documents.append({
                "source": path.name,
                "section": i,
                "text": chunk_text[:2000],  # truncate long sections
            })

    if not documents:
        log.warning("rag.build_index.no_chunks_extracted")
        return

    model = _get_embed_model()
    texts = [d["text"] for d in documents]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine similarity (after normalization)
    index.add(embeddings)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"index": index, "documents": documents, "dim": dim}, f)

    log.info("rag.build_index.complete", n_chunks=len(documents), dim=dim)


def _load_index() -> bool:
    """Load the persisted FAISS index into memory."""
    global _index, _documents
    if _index is not None:
        return True
    if not INDEX_PATH.exists():
        log.warning("rag.load_index.not_found", path=str(INDEX_PATH))
        return False
    try:
        with open(INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        _index = data["index"]
        _documents = data["documents"]
        log.info("rag.load_index.complete", n_chunks=len(_documents))
        return True
    except Exception as exc:
        log.error("rag.load_index.error", error=str(exc))
        return False


def query(text: str, top_k: int = 3) -> List[dict]:
    """
    Query the runbook index for relevant procedures.
    Returns list of {source, text, score} dicts.
    """
    try:
        import faiss
        import numpy as np
    except ImportError:
        return []

    if not _load_index():
        return []

    model = _get_embed_model()
    embedding = model.encode([text], normalize_embeddings=True)
    embedding = np.array(embedding, dtype=np.float32)

    distances, indices = _index.search(embedding, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_documents):
            continue
        doc = _documents[idx]
        results.append({
            "source": doc["source"],
            "text": doc["text"],
            "score": round(float(dist), 4),
        })
    return results
