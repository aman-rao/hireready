"""Check every backend/data/kb/*.jsonl file and report coverage (owner: B).

Run from the backend/ folder:
    python scripts/validate_kb.py

Exits with code 1 if any line is invalid or duplicated, so it can gate CI later.
"""

import json
import sys
from collections import Counter
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from pydantic import ValidationError  # noqa: E402

from app.rag.kb_schema import KBEntry  # noqa: E402

KB_DIR = BACKEND_DIR / "data" / "kb"
TARGET_QUESTIONS = 300
TARGET_SKILLS = 20


def main() -> int:
    errors: list[str] = []
    entries: list[KBEntry] = []
    seen: dict[str, str] = {}

    files = sorted(KB_DIR.glob("*.jsonl"))
    if not files:
        print(f"No .jsonl files found in {KB_DIR}")
        return 1

    for path in files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            where = f"{path.name}:{lineno}"
            try:
                entry = KBEntry.model_validate(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append(f"{where}: invalid JSON ({e.msg})")
                continue
            except ValidationError as e:
                msgs = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'entry'}: {err['msg']}" for err in e.errors())
                errors.append(f"{where}: {msgs}")
                continue
            if entry.id in seen:
                errors.append(f"{where}: duplicate of {seen[entry.id]}")
                continue
            seen[entry.id] = where
            entries.append(entry)

    questions = [e for e in entries if e.kind == "question"]
    docs = [e for e in entries if e.kind == "doc"]
    skills = Counter(e.skill for e in questions)

    print(f"Files: {', '.join(p.name for p in files)}")
    print(f"Questions: {len(questions)} / {TARGET_QUESTIONS} target")
    print(f"Question skills: {len(skills)} / {TARGET_SKILLS} target")
    print(f"Docs: {len(docs)} (skills: {', '.join(sorted({d.skill for d in docs})) or 'none'})")

    print("\nQuestions by type and difficulty:")
    by_type = Counter((e.interview_type, e.difficulty) for e in questions)
    for (itype, diff), n in sorted(by_type.items()):
        print(f"  {itype:<10} {diff:<7} {n}")

    print("\nQuestions per skill:")
    for skill, n in sorted(skills.items()):
        print(f"  {skill:<22} {n}")

    if errors:
        print(f"\n{len(errors)} problem(s):")
        for err in errors:
            print(f"  {err}")
        return 1

    print("\nAll entries valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
