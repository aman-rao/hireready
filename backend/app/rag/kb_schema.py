"""Format of knowledge-base entries in backend/data/kb/*.jsonl (owner: B).

One JSON object per line. Two kinds of entry:

- "question": an interview question, stored as one chunk.
- "doc": a short technical explainer, split into ~400-token chunks at ingestion.

These fields become Chroma metadata, which the retriever filters on.
"""

import hashlib
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas import Difficulty

KBKind = Literal["question", "doc"]
# A single question is either technical or behavioral; "mixed" is a session setting.
KBInterviewType = Literal["technical", "behavioral"]


class KBEntry(BaseModel):
    kind: KBKind
    text: str = Field(min_length=10)
    source: str = Field(min_length=1)  # e.g. "seed-handwritten", "llm-generated-reviewed", a URL
    skill: str = Field(min_length=1)  # canonical skill name, e.g. "Python"; behavioral uses a theme, e.g. "Teamwork"
    interview_type: KBInterviewType
    difficulty: Difficulty | None = None  # required for questions, optional for docs
    title: str | None = None  # docs only

    @model_validator(mode="after")
    def _check_kind_fields(self) -> "KBEntry":
        if self.kind == "question" and self.difficulty is None:
            raise ValueError("questions must have a difficulty")
        if self.kind == "doc" and not self.title:
            raise ValueError("docs must have a title")
        return self

    @property
    def id(self) -> str:
        """Stable ID from content, so re-running ingestion never duplicates entries."""
        key = f"{self.kind}|{self.skill}|{self.text.strip().lower()}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
