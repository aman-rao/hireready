"""Choose what each question is about (owner: B).

Technical topics come from the skill gap: missing skills and matched skills are
interleaved, missing first, so the interview probes both the gaps and the
skills the candidate claims. Behavioral topics cycle through common themes.
"mixed" interviews alternate, starting with technical.
"""

from app.agent.state import InterviewState, Topic

BEHAVIORAL_THEMES = [
    "Teamwork",
    "Conflict Resolution",
    "Handling Failure",
    "Leadership",
    "Communication",
    "Learning Agility",
]


def _interleave(a: list[str], b: list[str]) -> list[str]:
    out = []
    for i in range(max(len(a), len(b))):
        out += [x[i] for x in (a, b) if i < len(x)]
    return out


def technical_skills(state: InterviewState) -> list[str]:
    gap = state["skill_gap"]
    ordered = _interleave(list(gap.missing), list(gap.matched))
    if not ordered:  # empty gap: fall back to the JD, then the resume
        ordered = list(state["job"].required_skills) or list(state["candidate"].skills)
    seen, unique = set(), []
    for skill in ordered:
        if skill.lower() not in seen:
            seen.add(skill.lower())
            unique.append(skill)
    return unique or ["General"]


def select_topic(state: InterviewState) -> Topic:
    """The topic for the next new (non-follow-up) question."""
    main_questions = [q for q in state["asked"] if not q.is_follow_up]
    n = len(main_questions)

    itype = state["interview_type"]
    if itype == "mixed":
        qtype = "technical" if n % 2 == 0 else "behavioral"
        index = n // 2  # nth question of that type
    else:
        qtype, index = itype, n

    pool = technical_skills(state) if qtype == "technical" else BEHAVIORAL_THEMES
    return Topic(skill=pool[index % len(pool)], type=qtype)
