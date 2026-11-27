"""Answer evaluation (spec 6.2 / 7.5, owner: B)."""

from pydantic import BaseModel, Field, field_validator

from app.agent import prompts
from app.agent.llm import complete_json
from app.agent.rubric import format_rubric
from app.agent.state import InterviewState
from app.schemas import Chunk, Feedback, Question


class _EvalOut(BaseModel):
    score: int = Field(ge=1, le=5)
    strengths: list[str] = []
    improvements: list[str] = []
    model_answer_outline: str

    @field_validator("model_answer_outline", mode="before")
    @classmethod
    def _join_list(cls, v):
        # Models sometimes return the outline as a list of bullets.
        return "\n".join(f"- {x}" for x in v) if isinstance(v, list) else v


def _fake_eval(question: Question, answer: str) -> _EvalOut:
    """Offline stand-in: scores by answer length so follow-ups can be demoed without an API key."""
    words = len(answer.split())
    score = 2 if words < 15 else 3 if words < 60 else 4
    return _EvalOut(
        score=score,
        strengths=["Answered the question directly."] if words >= 15 else [],
        improvements=["Add a concrete example from your own experience."]
        + (["Explain the reasoning in more depth."] if words < 60 else []),
        model_answer_outline=f"- Define the key idea behind {question.skill or 'the topic'}\n"
        "- Explain how it works and why\n- Give a concrete example\n- Mention a trade-off or pitfall",
    )


def evaluate(question: Question, answer: str, context: list[Chunk]) -> Feedback:
    """Spec 6.2. Score one answer against the rubric."""
    if not answer.strip():
        return Feedback(
            question_id=question.id,
            score=1,
            strengths=[],
            improvements=["No answer was given. Even a partial answer or your reasoning earns credit."],
            model_answer_outline="",
        )

    out = complete_json(
        prompts.EVALUATE_SYSTEM,
        prompts.EVALUATE_USER.format(
            question_type=question.type,
            difficulty=question.difficulty,
            question=question.text,
            answer=answer,
            retrieved_chunks=prompts.format_chunks(context),
            rubric=format_rubric(question.type),
        ),
        _EvalOut,
        fake=lambda: _fake_eval(question, answer),
        effort="medium",  # grading needs more care than question writing
    )
    return Feedback(question_id=question.id, **out.model_dump())


def submit_answer(state: InterviewState, question_id: str, answer: str) -> Feedback:
    """Convenience for the API layer: evaluate the current question and record the result in state.

    Raises ValueError if question_id is not the question currently waiting for an answer.
    """
    if len(state["asked"]) <= len(state["answers"]):
        raise ValueError("no question is waiting for an answer")
    current = state["asked"][-1]
    if current.id != question_id:
        raise ValueError(f"question {question_id} is not the current question ({current.id})")

    feedback = evaluate(current, answer, state["context"].get(current.id, []))
    state["answers"].append(answer)
    state["feedback"].append(feedback)
    return feedback
