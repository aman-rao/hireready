"""Knowledge base and retrieval (owner: B)."""

from pathlib import Path

from dotenv import load_dotenv

# Load the repo-root .env before submodules read their settings.
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from app.rag.retriever import retrieve  # noqa: E402

__all__ = ["retrieve"]
