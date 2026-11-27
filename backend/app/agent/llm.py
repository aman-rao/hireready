"""LLM client (owner: B).

One function, complete_json(), returns a validated Pydantic object from Claude,
using the Anthropic SDK's structured outputs (messages.parse).

Settings in .env:
- ANTHROPIC_API_KEY: your key from console.anthropic.com
- LLM_PROVIDER: "anthropic" (default) or "fake". "fake" makes no network calls;
  each call site supplies a canned response. Used for tests, frontend
  development and as an offline demo fallback.
- LLM_MODEL: optional; defaults to claude-haiku-5-5 (cheap and fast).
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "claude-haiku-5-5"
MAX_TOKENS = 16000  # room for thinking plus the JSON answer; you pay only for tokens used
Effort = Literal["low", "medium", "high"]


class LLMError(RuntimeError):
    pass


def provider() -> str:
    return (os.getenv("LLM_PROVIDER") or "anthropic").lower()


def model() -> str:
    return os.getenv("LLM_MODEL") or DEFAULT_MODEL


@lru_cache(maxsize=1)
def _client():
    import anthropic

    if provider() != "anthropic":
        raise LLMError(f"Unknown LLM_PROVIDER '{provider()}'. Use 'anthropic' or 'fake'.")
    # Reads ANTHROPIC_API_KEY (or an `ant auth login` profile) from the environment.
    return anthropic.Anthropic()


def complete_json(
    system: str,
    user: str,
    schema: type[T],
    *,
    fake: Callable[[], T],
    effort: Effort = "low",
) -> T:
    """Ask Claude for output matching schema. Retries once if validation fails.

    effort controls how much Claude thinks before answering: "low" for simple
    generation, "medium" where judgement matters (e.g. grading answers).
    """
    if provider() == "fake":
        return fake()

    import anthropic

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = _client().messages.parse(
                model=model(),
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
                output_config={"effort": effort},
            )
        except anthropic.AuthenticationError as e:
            raise LLMError("Anthropic rejected the API key. Check ANTHROPIC_API_KEY in .env.") from e
        except TypeError as e:
            # The SDK raises TypeError when it finds no credentials at all.
            if "authentication" not in str(e).lower():
                raise
            raise LLMError("No Anthropic API key found. Set ANTHROPIC_API_KEY in .env, or LLM_PROVIDER=fake.") from e
        except anthropic.NotFoundError as e:
            raise LLMError(f"Model '{model()}' not found. Check LLM_MODEL in .env.") from e
        except anthropic.BadRequestError as e:
            raise LLMError(f"Bad request to Claude: {e.message}") from e
        except anthropic.RateLimitError as e:
            raise LLMError("Rate limited by Anthropic, even after retries. Wait a minute and try again.") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Could not reach the Anthropic API. Check your internet connection.") from e
        except ValidationError as e:
            last_error = e
            log.warning("Claude output failed validation (attempt %d): %s", attempt + 1, e)
            continue

        if response.stop_reason == "refusal":
            raise LLMError("Claude declined this request.")
        if response.stop_reason == "max_tokens":
            last_error = LLMError("response was cut off at max_tokens")
            log.warning("Claude hit max_tokens (attempt %d)", attempt + 1)
            continue
        if response.parsed_output is None:
            last_error = LLMError("no parsed output in response")
            continue
        return response.parsed_output

    raise LLMError(f"Claude output failed twice: {last_error}")
