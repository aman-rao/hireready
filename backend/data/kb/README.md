# Knowledge base

Interview questions and short explainer docs that the interview agent retrieves from.
Each `.jsonl` file holds one JSON object per line. The format is defined in
[app/rag/kb_schema.py](../../app/rag/kb_schema.py).

## Entry format

Question (stored as one chunk):

```json
{"kind": "question", "text": "What is overfitting?", "source": "seed-handwritten", "skill": "Machine Learning", "interview_type": "technical", "difficulty": "easy"}
```

Doc (split into ~400-token chunks):

```json
{"kind": "doc", "title": "SQL joins", "text": "A join combines rows...", "source": "seed-handwritten", "skill": "SQL", "interview_type": "technical"}
```

| Field | Values |
|---|---|
| `kind` | `question` or `doc` |
| `text` | the question, or the doc body |
| `source` | where it came from: `seed-handwritten`, `llm-generated`, `llm-generated-reviewed`, or a URL |
| `skill` | canonical skill name, e.g. `Python`, `Machine Learning`. Behavioral entries use a theme, e.g. `Teamwork` |
| `interview_type` | `technical` or `behavioral` |
| `difficulty` | `easy`, `medium` or `hard` (required for questions) |
| `title` | required for docs |

## Files

- `questions.jsonl`, `docs.jsonl`: hand-written seed entries.
- `generated_questions.jsonl`, `generated_docs.jsonl`: LLM-generated entries (Claude Haiku), deduplicated with embeddings.

To add entries, append lines to any `.jsonl` file here, or add a new `.jsonl` file.

## Workflow (run from `backend/`)

```
python scripts/validate_kb.py     # check format, duplicates and coverage
python scripts/ingest.py          # rebuild the Chroma index
```

Re-run `ingest.py` after any change here, and after changing the embedding model.
