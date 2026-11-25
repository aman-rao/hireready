# hireready
Personalized mock interviews from your resume and a job description, powered by LLMs, RAG, and encoder models.

## Repo structure

```
frontend/                 # C: Next.js (app/, components/, lib/api.ts)
backend/
  app/
    main.py               # C: FastAPI app + routes
    schemas.py            # shared contracts (spec 6.1)
    mocks.py              # shared mock implementations
    parsing/              # A
    encoders/             # A: embeddings, reranker
    fnn/                  # A: model def + inference
    rag/                  # B: retriever
    agent/                # B: LangGraph graph, prompts, rubric
  data/kb/                # B: JSONL knowledge-base sources
  models/                 # A: fnn.pt, label_map.json
  scripts/ingest.py       # B
  notebooks/              # A: FNN training, reranker eval
docs/                     # wireframes, report, slides
```

Branching: `feat/<owner>-<thing>`, PRs to `main`, one reviewer per PR.
