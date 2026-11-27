"""Turn KB entries into Chroma records (spec 7.4, owner: B).

Questions are stored one per chunk. Docs are split into ~400-token chunks with
~50-token overlap. Tokens are approximated as 0.75 words each, which is close
enough for English prose and avoids depending on a specific tokenizer.
"""

from dataclasses import dataclass

from app.rag.kb_schema import KBEntry

CHUNK_WORDS = 300  # ~400 tokens
OVERLAP_WORDS = 38  # ~50 tokens


@dataclass
class Record:
    id: str
    text: str
    metadata: dict


def skill_key(skill: str) -> str:
    """Case- and whitespace-insensitive form of a skill name, used for filtering."""
    return " ".join(skill.lower().split())


def split_words(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def to_records(entry: KBEntry) -> list[Record]:
    metadata = {
        "kind": entry.kind,
        "source": entry.source,
        "skill": entry.skill,
        "skill_key": skill_key(entry.skill),
        "interview_type": entry.interview_type,
    }
    # Chroma metadata values cannot be None.
    if entry.difficulty:
        metadata["difficulty"] = entry.difficulty
    if entry.title:
        metadata["title"] = entry.title

    if entry.kind == "question":
        return [Record(entry.id, entry.text, metadata)]

    parts = split_words(entry.text)
    return [
        Record(f"{entry.id}-{i}", f"{entry.title}\n\n{part}", {**metadata, "chunk_index": i})
        for i, part in enumerate(parts)
    ]
