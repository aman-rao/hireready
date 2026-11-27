"""ChromaDB access (owner: B)."""

import os
from functools import lru_cache
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = Path(os.getenv("CHROMA_DIR") or BACKEND_DIR / "chroma")
COLLECTION_NAME = "interview_kb"


@lru_cache(maxsize=1)
def get_client():
    import chromadb

    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    """The knowledge-base collection. Raises if ingest.py has not been run."""
    return get_client().get_collection(COLLECTION_NAME)


def recreate_collection():
    """Drop and recreate the collection; used by ingest.py for an idempotent rebuild."""
    client = get_client()
    if COLLECTION_NAME in [c if isinstance(c, str) else c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    # Cosine distance, since embeddings are normalized sentence vectors.
    return client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
