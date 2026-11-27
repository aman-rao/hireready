"""Semantic search over the knowledge base, with reranking (spec 6.2, owner: B).

retrieve() embeds the query, pulls the top candidates from Chroma using
metadata filters, then reorders them with the cross-encoder reranker.
"""

from app.rag.chunking import skill_key
from app.rag.encoders_bridge import embed, rerank
from app.rag.store import get_collection
from app.schemas import Chunk

CANDIDATES = 20  # retrieved before reranking (spec 5: top-20 -> top-5)
FILTER_KEYS = {"kind", "skill", "interview_type", "difficulty", "source"}


def _where(filters: dict) -> dict | None:
    """Convert {"skill": "Python", "kind": "question"} into a Chroma where clause.

    Skill matches case-insensitively. A list value matches any of its items.
    Unknown keys raise, so a typo doesn't silently disable a filter.
    """
    unknown = set(filters) - FILTER_KEYS
    if unknown:
        raise ValueError(f"unknown filter keys: {sorted(unknown)}; allowed: {sorted(FILTER_KEYS)}")

    clauses = []
    for key, value in filters.items():
        if value is None:
            continue
        if key == "skill":
            key = "skill_key"
            value = [skill_key(v) for v in value] if isinstance(value, list) else skill_key(value)
        clauses.append({key: {"$in": value}} if isinstance(value, list) else {key: value})

    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def retrieve(query: str, filters: dict | None = None, k: int = 5, use_reranker: bool = True) -> list[Chunk]:
    """Return the k most relevant chunks for query, filtered by metadata.

    filters keys: kind ("question"/"doc"), skill, interview_type
    ("technical"/"behavioral"), difficulty, source. Values may be lists.
    Set use_reranker=False to get raw embedding order (for the reranker eval).
    """
    collection = get_collection()
    result = collection.query(
        query_embeddings=embed([query]),
        n_results=max(k, CANDIDATES) if use_reranker else k,
        where=_where(filters or {}),
        include=["documents", "metadatas", "distances"],
    )
    ids = result["ids"][0]
    if not ids:
        return []
    docs = result["documents"][0]
    metas = result["metadatas"][0]

    if use_reranker:
        scores = rerank(query, docs)
    else:
        scores = [1.0 - d for d in result["distances"][0]]  # cosine similarity

    ranked = sorted(zip(ids, docs, metas, scores), key=lambda r: r[3], reverse=True)[:k]
    return [
        Chunk(id=cid, text=text, source=meta.get("source", ""), metadata=meta, score=score)
        for cid, text, meta, score in ranked
    ]
