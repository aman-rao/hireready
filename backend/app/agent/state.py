"""LangGraph interview state (spec 7.5, owner: B).

Kept here rather than in schemas.py because only the agent uses it.
Fields beyond spec 7.5 are marked "added".
"""

from typing import Literal, TypedDict

from app.schemas import (
    CandidateProfile,
    Chunk,
    Difficulty,
    Feedback,
    FitPrediction,
    InterviewType,
    JobProfile,
    Question,
    SkillGap,
)

QuestionType = Literal["technical", "behavioral"]


class Topic(TypedDict):
    skill: str
    type: QuestionType


class InterviewState(TypedDict):
    session_id: str
    candidate: CandidateProfile
    job: JobProfile
    skill_gap: SkillGap
    fit: FitPrediction  # added: build_report needs it for InterviewReport.fit
    interview_type: InterviewType
    asked: list[Question]
    answers: list[str]
    feedback: list[Feedback]
    current_difficulty: Difficulty
    follow_ups_used: int
    max_questions: int
    context: dict[str, list[Chunk]]  # added: question_id -> retrieved chunks, for the UI and evaluation
    topic: Topic | None  # added: the topic chosen for the question being generated
    retrieved: list[Chunk]  # added: chunks retrieved for the question being generated


def new_interview_state(
    session_id: str,
    candidate: CandidateProfile,
    job: JobProfile,
    skill_gap: SkillGap,
    fit: FitPrediction,
    interview_type: InterviewType = "mixed",
    max_questions: int = 5,
    starting_difficulty: Difficulty = "medium",
) -> InterviewState:
    """A fresh state for a new session."""
    return InterviewState(
        session_id=session_id,
        candidate=candidate,
        job=job,
        skill_gap=skill_gap,
        fit=fit,
        interview_type=interview_type,
        asked=[],
        answers=[],
        feedback=[],
        current_difficulty=starting_difficulty,
        follow_ups_used=0,
        max_questions=max_questions,
        context={},
        topic=None,
        retrieved=[],
    )
