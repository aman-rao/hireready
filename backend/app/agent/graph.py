"""LangGraph question flow and next_question() (spec 7.5, owner: B).

Each call to next_question() runs the graph once:

    START -> route --"new"-------> select_topic -> retrieve_context -> generate_question -> END
                   --"follow_up"-> generate_follow_up -> END
                   --"pending"---> END   (current question not answered yet)
                   --"done"------> END   (max_questions reached)

Routing rules (spec 7.5):
- last score <= 2 and that question was not a follow-up -> one follow-up
- otherwise a new topic; difficulty goes up a level after a score >= 4
  (and, as an addition, down a level after a score <= 2)
- follow-ups count toward max_questions
"""

import logging
import uuid

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.agent import prompts
from app.agent.llm import complete_json
from app.agent.state import InterviewState
from app.agent.topics import select_topic as choose_topic
from app.rag.retriever import retrieve
from app.schemas import Chunk, Difficulty, Question

log = logging.getLogger(__name__)

LEVELS: list[Difficulty] = ["easy", "medium", "hard"]
QUESTION_CHUNKS = 3
DOC_CHUNKS = 2


# --- LLM output formats -----------------------------------------------------


class _QuestionOut(BaseModel):
    text: str
    context_chunk_ids: list[str] = []


class _FollowUpOut(BaseModel):
    text: str


# --- Helpers ----------------------------------------------------------------


def _shift(level: Difficulty, step: int) -> Difficulty:
    i = min(max(LEVELS.index(level) + step, 0), len(LEVELS) - 1)
    return LEVELS[i]


def _candidate_summary(state: InterviewState) -> str:
    c = state["candidate"]
    roles = "; ".join(f"{e.title} at {e.company}" for e in c.experience[:3]) or "none listed"
    return f"Skills: {', '.join(c.skills) or 'none listed'}\nExperience: {roles}"


def _decide(state: InterviewState) -> str:
    if len(state["asked"]) > len(state["answers"]):
        return "pending"
    if len(state["asked"]) >= state["max_questions"]:
        return "done"
    if state["feedback"]:
        last_q, last_fb = state["asked"][-1], state["feedback"][-1]
        if last_fb.score <= 2 and not last_q.is_follow_up:
            return "follow_up"
    return "new"


def _safe_retrieve(query: str, filters: dict, k: int) -> list[Chunk]:
    """Retrieval failures shouldn't stop the interview; the LLM can still ask a question."""
    try:
        return retrieve(query, filters, k=k)
    except Exception:
        log.exception("retrieval failed for %r with %s", query, filters)
        return []


# --- Nodes ------------------------------------------------------------------


def select_topic(state: InterviewState) -> dict:
    difficulty = state["current_difficulty"]
    if state["feedback"]:
        score = state["feedback"][-1].score
        if score >= 4:
            difficulty = _shift(difficulty, +1)
        elif score <= 2:
            difficulty = _shift(difficulty, -1)
    return {"topic": choose_topic(state), "current_difficulty": difficulty}


def retrieve_context(state: InterviewState) -> dict:
    topic = state["topic"]
    query = f"{topic['skill']} interview question for a {state['job'].title}"
    base = {"kind": "question", "interview_type": topic["type"]}

    # Progressively relax filters until we have enough questions.
    questions: list[Chunk] = []
    for filters in (
        {**base, "skill": topic["skill"], "difficulty": state["current_difficulty"]},
        {**base, "skill": topic["skill"]},
        base,
    ):
        found = _safe_retrieve(query, filters, QUESTION_CHUNKS)
        questions += [c for c in found if c.id not in {q.id for q in questions}]
        if len(questions) >= QUESTION_CHUNKS:
            break

    docs = _safe_retrieve(topic["skill"], {"kind": "doc", "skill": topic["skill"]}, DOC_CHUNKS)
    return {"retrieved": questions[:QUESTION_CHUNKS] + docs}


def generate_question(state: InterviewState) -> dict:
    topic, chunks = state["topic"], state["retrieved"]
    difficulty = state["current_difficulty"]
    asked_texts = [q.text for q in state["asked"]]

    def fake() -> _QuestionOut:
        unused = [c for c in chunks if c.metadata.get("kind") == "question" and c.text not in asked_texts]
        if unused:
            return _QuestionOut(text=unused[0].text, context_chunk_ids=[unused[0].id])
        return _QuestionOut(text=f"Tell me about a time you used {topic['skill']} in a project. What was the hardest part?")

    out = complete_json(
        prompts.QUESTION_SYSTEM,
        prompts.QUESTION_USER.format(
            job_title=state["job"].title,
            question_type=topic["type"],
            skill=topic["skill"],
            difficulty=difficulty,
            candidate_summary=_candidate_summary(state),
            asked_questions="\n".join(f"- {t}" for t in asked_texts) or "(none)",
            retrieved_chunks=prompts.format_chunks(chunks),
        ),
        _QuestionOut,
        fake=fake,
    )

    valid_ids = {c.id for c in chunks}
    cited = [i for i in out.context_chunk_ids if i in valid_ids] or [c.id for c in chunks]
    question = Question(
        id=uuid.uuid4().hex[:12],
        text=out.text,
        type=topic["type"],
        skill=topic["skill"],
        difficulty=difficulty,
        context_chunk_ids=cited,
    )
    return {
        "asked": state["asked"] + [question],
        "context": {**state["context"], question.id: chunks},
        "retrieved": [],
    }


def generate_follow_up(state: InterviewState) -> dict:
    parent, answer, fb = state["asked"][-1], state["answers"][-1], state["feedback"][-1]
    gaps = fb.improvements or ["the answer lacked detail"]

    out = complete_json(
        prompts.FOLLOW_UP_SYSTEM,
        prompts.FOLLOW_UP_USER.format(
            question=parent.text,
            answer=answer,
            improvements="\n".join(f"- {g}" for g in gaps),
        ),
        _FollowUpOut,
        fake=lambda: _FollowUpOut(text=f"Let's dig a little deeper. Could you expand on this: {gaps[0]}"),
    )

    question = Question(
        id=uuid.uuid4().hex[:12],
        text=out.text,
        type=parent.type,
        skill=parent.skill,
        difficulty=parent.difficulty,
        is_follow_up=True,
        context_chunk_ids=parent.context_chunk_ids,
    )
    return {
        "asked": state["asked"] + [question],
        "context": {**state["context"], question.id: state["context"].get(parent.id, [])},
        "follow_ups_used": state["follow_ups_used"] + 1,
    }


# --- Graph ------------------------------------------------------------------


def build_graph():
    g = StateGraph(InterviewState)
    g.add_node("select_topic", select_topic)
    g.add_node("retrieve_context", retrieve_context)
    g.add_node("generate_question", generate_question)
    g.add_node("generate_follow_up", generate_follow_up)

    g.add_conditional_edges(
        START,
        _decide,
        {"new": "select_topic", "follow_up": "generate_follow_up", "pending": END, "done": END},
    )
    g.add_edge("select_topic", "retrieve_context")
    g.add_edge("retrieve_context", "generate_question")
    g.add_edge("generate_question", END)
    g.add_edge("generate_follow_up", END)
    return g.compile()


question_graph = build_graph()


def next_question(state: InterviewState) -> Question | None:
    """Spec 6.2. Returns the question to show now, or None when the interview is done.

    Updates state in place (asked, context, difficulty, ...). Calling it again
    before the answer is submitted returns the same question.
    """
    state.update(question_graph.invoke(state))
    if len(state["asked"]) > len(state["answers"]):
        return state["asked"][-1]
    return None
