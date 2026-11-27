"""Rebuild the Chroma knowledge base from backend/data/kb/*.jsonl (spec 7.4, owner: B).

Run from the backend/ folder:
    python scripts/ingest.py

Idempotent: it drops and recreates the collection every run, and record IDs are
derived from content, so running it twice gives the same result.
Invalid lines are reported and skipped; run scripts/validate_kb.py for details.
"""

import json
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from pydantic import ValidationError  # noqa: E402

from app.rag.chunking import to_records  # noqa: E402
from app.rag.encoders_bridge import EMBEDDING_MODEL, embed  # noqa: E402
from app.rag.kb_schema import KBEntry  # noqa: E402
from app.rag.store import CHROMA_DIR, recreate_collection  # noqa: E402

KB_DIR = BACKEND_DIR / "data" / "kb"
BATCH = 128


def load_entries() -> list[KBEntry]:
    entries: dict[str, KBEntry] = {}
    skipped = 0
    for path in sorted(KB_DIR.glob("*.jsonl")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entry = KBEntry.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValidationError):
                print(f"  skipping invalid line {path.name}:{lineno}")
                skipped += 1
                continue
            entries.setdefault(entry.id, entry)  # first copy of a duplicate wins
    if skipped:
        print(f"Skipped {skipped} invalid line(s); run scripts/validate_kb.py for details.")
    return list(entries.values())


def main() -> None:
    start = time.time()
    entries = load_entries()
    records = [r for e in entries for r in to_records(e)]
    print(f"Loaded {len(entries)} entries -> {len(records)} chunks")
    print(f"Embedding with {EMBEDDING_MODEL} ...")

    collection = recreate_collection()
    for i in range(0, len(records), BATCH):
        batch = records[i : i + BATCH]
        collection.add(
            ids=[r.id for r in batch],
            documents=[r.text for r in batch],
            metadatas=[r.metadata for r in batch],
            embeddings=embed([r.text for r in batch]),
        )

    print(f"Ingested {collection.count()} chunks into {CHROMA_DIR} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
