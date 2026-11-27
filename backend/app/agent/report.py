"""Final interview report (spec 6.2, owner: B)."""

import logging

from pydantic import BaseModel

from app.agent import prompts
from app.agent.llm import LLMError, complete_json
from app.agent.state import InterviewState
from app.schemas import InterviewReport

log = logging.getLogger(__name__)


class _ReportOut(BaseModel):
    prep_plan: list[str]


def _fallback_plan(state: InterviewState) -> _ReportOut:
    """Rule-based plan, used offline and if the LLM call fails."""
    steps = [f"Study {s}: it is required for the role but missing from your resume." for s in state["skill_gap"].missing[:3]]
    revisited = set()
    for q, fb in zip(state["asked"], state["feedback"]):
        topic = q.skill or q.type
        if fb.score <= 3 and fb.improvements and topic not in revisited:
            revisited.add(topic)
            steps.append(f"Revisit {topic}: {fb.improvements[0]}")
    steps.append("Practise answering out loud, using the STAR structure for behavioral questions.")
    return _ReportOut(prep_plan=steps[:6])


def build_report(state: InterviewState) -> InterviewReport:
    """Spec 6.2. overall_score is the mean per-question score on the same 1-5 scale."""
    scores = [fb.score for fb in state["feedback"]]
    overall = round(sum(scores) / len(scores), 2) if scores else 0.0

    per_question_summary = "\n".join(
        f"- [{q.skill or q.type}] {q.text}\n  score {fb.score}; gaps: {'; '.join(fb.improvements) or 'none'}"
        for q, fb in zip(state["asked"], state["feedback"])
    )
    try:
        plan = complete_json(
            prompts.REPORT_SYSTEM,
            prompts.REPORT_USER.format(
                job_title=state["job"].title,
                missing_skills=", ".join(state["skill_gap"].missing) or "(none)",
                per_question_summary=per_question_summary or "(no answers)",
            ),
            _ReportOut,
            fake=lambda: _fallback_plan(state),
        )
    except LLMError:
        log.exception("prep plan generation failed; using rule-based plan")
        plan = _fallback_plan(state)

    return InterviewReport(
        session_id=state["session_id"],
        overall_score=overall,
        per_question=list(state["feedback"]),
        skill_gap=state["skill_gap"],
        fit=state["fit"],
        prep_plan=plan.prep_plan,
    )
