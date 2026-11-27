"""Answer-evaluation rubric (spec 7.5, owner: B).

Each answer is scored 1-5 overall. The criteria guide the LLM's judgement and
appear in the prompt; the per-criterion notes are not returned separately.
"""

TECHNICAL_CRITERIA = {
    "correctness": "Are the facts and reasoning right? Any errors or misconceptions?",
    "depth": "Does it go beyond a definition: trade-offs, edge cases, why it works?",
    "clarity": "Is it structured and easy to follow?",
    "examples": "Does it use a concrete example, code idea or real use case?",
}

BEHAVIORAL_CRITERIA = {
    "star_completeness": "Does it cover Situation, Task, Action and Result?",
    "specificity": "Is it one specific real event, not a generality ('I always...')?",
    "impact": "Is the result concrete, ideally measurable, and is the candidate's own role clear?",
    "clarity": "Is it concise and in a logical order?",
}

SCORE_SCALE = {
    1: "Missing or wrong: does not answer the question, or is mostly incorrect.",
    2: "Weak: partly relevant but with major gaps or errors.",
    3: "Adequate: correct and relevant, but shallow or missing an important piece.",
    4: "Strong: correct, well structured, with good depth or a concrete example.",
    5: "Excellent: what a top candidate would say; thorough, precise, insightful.",
}


def format_rubric(question_type: str) -> str:
    """Render the rubric for a question type ("technical" or "behavioral") as prompt text."""
    criteria = BEHAVIORAL_CRITERIA if question_type == "behavioral" else TECHNICAL_CRITERIA
    lines = ["Criteria:"]
    lines += [f"- {name}: {desc}" for name, desc in criteria.items()]
    lines.append("")
    lines.append("Score scale (overall, 1-5):")
    lines += [f"- {score}: {desc}" for score, desc in SCORE_SCALE.items()]
    return "\n".join(lines)
