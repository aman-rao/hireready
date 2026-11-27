import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.rag.chunking import split_words, to_records
from app.rag.kb_schema import KBEntry
from app.rag.retriever import _where

KB_DIR = Path(__file__).resolve().parents[1] / "data" / "kb"


def kb_lines():
    for path in sorted(KB_DIR.glob("*.jsonl")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                yield f"{path.name}:{lineno}", line


@pytest.mark.parametrize("where,line", list(kb_lines()))
def test_kb_line_is_valid(where, line):
    KBEntry.model_validate(json.loads(line))


def test_kb_has_no_duplicates():
    ids = [KBEntry.model_validate(json.loads(line)).id for _, line in kb_lines()]
    assert len(ids) == len(set(ids))


def test_question_requires_difficulty():
    with pytest.raises(ValidationError):
        KBEntry(kind="question", text="What is a list in Python?", source="x", skill="Python", interview_type="technical")


def test_id_is_stable_and_ignores_case_and_whitespace():
    a = KBEntry(kind="question", text="What is SQL? ", source="a", skill="SQL", interview_type="technical", difficulty="easy")
    b = KBEntry(kind="question", text="what is sql?", source="b", skill="SQL", interview_type="technical", difficulty="hard")
    assert a.id == b.id


def test_split_words_overlaps_and_covers_everything():
    words = [f"w{i}" for i in range(700)]
    chunks = split_words(" ".join(words), size=300, overlap=50)
    assert len(chunks) == 3
    assert chunks[0].split()[-50:] == chunks[1].split()[:50]
    assert chunks[-1].split()[-1] == "w699"


def test_doc_records_have_metadata_without_none():
    doc = KBEntry(kind="doc", title="T", text="word " * 700, source="s", skill="Machine Learning", interview_type="technical")
    records = to_records(doc)
    assert len(records) > 1
    assert all(None not in r.metadata.values() for r in records)
    assert records[0].metadata["skill_key"] == "machine learning"
    assert len({r.id for r in records}) == len(records)


def test_where_clause():
    assert _where({}) is None
    assert _where({"skill": "Machine  Learning"}) == {"skill_key": "machine learning"}
    assert _where({"kind": "question", "difficulty": ["easy", "medium"]}) == {
        "$and": [{"kind": "question"}, {"difficulty": {"$in": ["easy", "medium"]}}]
    }
    with pytest.raises(ValueError):
        _where({"skil": "Python"})
