import pytest

import app.agent.graph as graph
from app.agent import build_report, new_interview_state, next_question, submit_answer
from app.agent.topics import select_topic
from app.schemas import CandidateProfile, Chunk, Experience, FitPrediction, JobProfile, SkillGap

STRONG = "word " * 80  # the fake evaluator scores 4 for 60+ words
WEAK = "not sure"  # and 2 for under 15 words


def fake_retrieve(query, filters, k=5):
    skill = filters.get("skill", "General")
    kind = filters.get("kind", "question")
    return [
        Chunk(
            id=f"{kind}-{skill}-{i}",
            text=f"Sample {kind} about {skill} #{i} for {query}",
            source="test",
            metadata={"kind": kind, "skill": skill},
            score=1.0 - i / 10,
        )
        for i in range(k)
    ]


@pytest.fixture(autouse=True)
def no_real_retrieval(monkeypatch):
    monkeypatch.setattr(graph, "retrieve", fake_retrieve)


def make_state(interview_type="mixed", max_questions=5):
    return new_interview_state(
        session_id="s1",
        candidate=CandidateProfile(
            skills=["Python", "SQL"],
            experience=[Experience(title="Data Analyst", company="Acme")],
            raw_text="...",
        ),
        job=JobProfile(title="ML Engineer", required_skills=["Python", "PyTorch", "Docker"], raw_text="..."),
        skill_gap=SkillGap(matched=["Python"], missing=["PyTorch", "Docker"], extra=["SQL"]),
        fit=FitPrediction(predicted_role="Data Science", confidence=0.7, top_roles=[("Data Science", 0.7)]),
        interview_type=interview_type,
        max_questions=max_questions,
    )


def test_topics_alternate_in_mixed_and_put_missing_skills_first():
    state = make_state()
    first = select_topic(state)
    assert first == {"skill": "PyTorch", "type": "technical"}


def test_next_question_is_idempotent_until_answered():
    state = make_state()
    q1 = next_question(state)
    assert next_question(state) is q1
    assert len(state["asked"]) == 1
    assert state["context"][q1.id]  # chunks stored for the UI


def test_weak_answer_gets_one_follow_up_then_moves_on():
    state = make_state()
    q1 = next_question(state)
    assert submit_answer(state, q1.id, WEAK).score <= 2

    follow = next_question(state)
    assert follow.is_follow_up and follow.skill == q1.skill
    assert state["follow_ups_used"] == 1

    submit_answer(state, follow.id, WEAK)
    q3 = next_question(state)
    assert not q3.is_follow_up  # never a follow-up to a follow-up
    assert q3.difficulty == "easy"  # lowered after a weak answer


def test_strong_answer_raises_difficulty():
    state = make_state()
    q1 = next_question(state)
    assert q1.difficulty == "medium"
    submit_answer(state, q1.id, STRONG)
    q2 = next_question(state)
    assert q2.difficulty == "hard"
    assert q2.type == "behavioral"


def test_full_interview_ends_with_report():
    state = make_state(max_questions=5)
    answers = [STRONG, WEAK, STRONG, STRONG, STRONG, STRONG]
    count = 0
    while (q := next_question(state)) is not None:
        submit_answer(state, q.id, answers[count])
        count += 1
    assert count == 5  # follow-ups count toward max_questions

    report = build_report(state)
    assert report.session_id == "s1"
    assert len(report.per_question) == 5
    assert 1 <= report.overall_score <= 5
    assert report.prep_plan
    assert report.fit.predicted_role == "Data Science"


def test_submit_answer_rejects_wrong_question():
    state = make_state()
    next_question(state)
    with pytest.raises(ValueError):
        submit_answer(state, "not-the-id", STRONG)


def test_empty_answer_scores_one_without_llm():
    state = make_state()
    q = next_question(state)
    assert submit_answer(state, q.id, "   ").score == 1


def test_retrieval_failure_does_not_stop_interview(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("chroma not built")

    monkeypatch.setattr(graph, "retrieve", broken)
    state = make_state()
    q = next_question(state)
    assert q is not None and q.text
