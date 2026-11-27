"""Run a mock interview in the terminal, using real retrieval (owner: B).

Run from the backend/ folder, after scripts/ingest.py:
    python scripts/try_interview.py            # you type the answers
    python scripts/try_interview.py --auto     # scripted answers (one weak, to show a follow-up)
    python scripts/try_interview.py --type technical --questions 3

Uses the LLM_PROVIDER from .env; set LLM_PROVIDER=fake to run without an API key.
The candidate and job are a built-in sample until A's parsing is ready.
"""

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.agent import build_report, new_interview_state, next_question, submit_answer  # noqa: E402
from app.agent.llm import provider  # noqa: E402
from app.schemas import CandidateProfile, Experience, FitPrediction, JobProfile, SkillGap  # noqa: E402

SAMPLE_CANDIDATE = CandidateProfile(
    name="Sample Candidate",
    skills=["Python", "SQL", "Machine Learning", "Statistics"],
    experience=[
        Experience(
            title="Data Analyst",
            company="Acme Corp",
            duration="2 years",
            highlights=["Built churn model in scikit-learn", "Automated SQL reporting"],
        )
    ],
    education=["BSc Computer Science"],
    raw_text="(sample)",
)
SAMPLE_JOB = JobProfile(
    title="Machine Learning Engineer",
    required_skills=["Python", "Machine Learning", "Deep Learning", "Docker"],
    nice_to_have=["System Design"],
    raw_text="(sample)",
)
SAMPLE_GAP = SkillGap(matched=["Python", "Machine Learning"], missing=["Deep Learning", "Docker"], extra=["SQL", "Statistics"])
SAMPLE_FIT = FitPrediction(predicted_role="Data Science", confidence=0.72, top_roles=[("Data Science", 0.72), ("ML Engineer", 0.18)])

AUTO_ANSWERS = [
    "Overfitting is when a model memorises the training data. " * 8,
    "I don't really know.",
    "Vanishing gradients happen when gradients shrink through many layers, so early layers stop learning. "
    "ReLU keeps gradients from saturating, residual connections give gradients a direct path, and "
    "normalization keeps activations in a stable range. For example, ResNets train 100+ layers this way.",
    "In my last team we disagreed on a deadline. I set up a meeting, we listed the risks, and agreed on a smaller first release. "
    "We shipped on time and the client was happy. I learned to raise concerns early. " * 2,
    "Containers package the app with its dependencies so it runs the same everywhere. " * 6,
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="use scripted answers")
    parser.add_argument("--type", default="mixed", choices=["technical", "behavioral", "mixed"])
    parser.add_argument("--questions", type=int, default=5)
    args = parser.parse_args()

    print(f"LLM provider: {provider()}\n")
    state = new_interview_state(
        "cli", SAMPLE_CANDIDATE, SAMPLE_JOB, SAMPLE_GAP, SAMPLE_FIT,
        interview_type=args.type, max_questions=args.questions,
    )

    n = 0
    while (q := next_question(state)) is not None:
        n += 1
        tag = "follow-up" if q.is_follow_up else f"{q.type}, {q.skill}, {q.difficulty}"
        print(f"Q{n} ({tag}):\n  {q.text}")
        for c in state["context"].get(q.id, []):
            print(f"    why: [{c.metadata.get('kind')}/{c.metadata.get('skill')}] {c.text[:70]}...")

        if args.auto:
            answer = AUTO_ANSWERS[(n - 1) % len(AUTO_ANSWERS)]
            print(f"A{n}: {answer[:100]}{'...' if len(answer) > 100 else ''}")
        else:
            answer = input(f"A{n}: ")

        fb = submit_answer(state, q.id, answer)
        print(f"  score {fb.score}/5")
        for s in fb.strengths:
            print(f"  + {s}")
        for s in fb.improvements:
            print(f"  - {s}")
        print()

    report = build_report(state)
    print(f"Overall score: {report.overall_score}/5")
    print("Prep plan:")
    for step in report.prep_plan:
        print(f"  * {step}")


if __name__ == "__main__":
    main()
