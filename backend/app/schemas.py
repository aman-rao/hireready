"""Shared interface contracts (spec section 6.1). Changes need agreement from affected owners."""

from typing import Literal

from pydantic import BaseModel

InterviewType = Literal["technical", "behavioral", "mixed"]
Difficulty = Literal["easy", "medium", "hard"]


class Experience(BaseModel):
    title: str
    company: str
    duration: str | None = None
    highlights: list[str] = []


class CandidateProfile(BaseModel):
    name: str | None = None
    skills: list[str]
    experience: list[Experience]
    education: list[str] = []
    raw_text: str


class JobProfile(BaseModel):
    title: str
    required_skills: list[str]
    nice_to_have: list[str] = []
    raw_text: str


class FitPrediction(BaseModel):
    predicted_role: str
    confidence: float  # 0-1
    top_roles: list[tuple[str, float]]


class SkillGap(BaseModel):
    matched: list[str]
    missing: list[str]
    extra: list[str]  # on resume, not in JD


class Chunk(BaseModel):
    id: str
    text: str
    source: str
    metadata: dict
    score: float  # post-rerank score


class Question(BaseModel):
    id: str
    text: str
    type: InterviewType
    skill: str | None = None
    difficulty: Difficulty
    is_follow_up: bool = False
    context_chunk_ids: list[str] = []  # for showing RAG grounding in the UI


class Feedback(BaseModel):
    question_id: str
    score: int  # 1-5
    strengths: list[str]
    improvements: list[str]
    model_answer_outline: str


class InterviewReport(BaseModel):
    session_id: str
    overall_score: float
    per_question: list[Feedback]
    skill_gap: SkillGap
    fit: FitPrediction
    prep_plan: list[str]
