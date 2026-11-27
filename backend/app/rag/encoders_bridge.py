"""Temporary stand-ins for A's embed() and reranker (owner: B, to be replaced).

A owns app/encoders/. Until A's versions land, retrieval and ingestion use these
local implementations of the same models named in the spec. At integration,
replace the bodies of embed() and rerank() with calls to A's functions; nothing
else in rag/ needs to change.

IMPORTANT: ingestion and querying must use the same embedding model. If the
model changes, re-run scripts/ingest.py.
"""

import os
from functools import lru_cache

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = os.getenv("RERANKER_MODEL") or "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Same signature as A's embed() in spec 6.2. Returns unit-length vectors."""
    vectors = _embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def rerank(query: str, texts: list[str]) -> list[float]:
    """Relevance score for each (query, text) pair; higher is more relevant."""
    if not texts:
        return []
    scores = _reranker().predict([(query, t) for t in texts], show_progress_bar=False)
    return [float(s) for s in scores]
