"""Shared helpers for invoking an agent with structured output and a graceful fallback.

The Portfolio Manager, Trader, and Research Manager all follow the same
canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output (rare; mostly older Ollama models), the
   wrap is skipped and the agent uses free-text generation instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call itself fails for any reason
   (malformed JSON from a weak model, transient provider issue), attempt
   a manual JSON extraction pass on a plain-text response before giving
   up and returning prose.

Centralising the pattern here keeps the agent factories small and ensures
all three agents log the same warnings when fallback fires.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_JSON_HINT_BASE = (
    "\n\nRespond with a valid JSON object only — "
    "no markdown fences, no explanation, no text outside the JSON."
)


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Optional[Any]:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None


def _build_json_hint(schema: Optional[type[T]] = None) -> str:
    """Build a JSON hint string, optionally including required field names from the schema."""
    if schema is None:
        return _JSON_HINT_BASE
    required = [n for n, f in schema.model_fields.items() if f.is_required()]
    optional = [n for n, f in schema.model_fields.items() if not f.is_required()]
    fields = f"Required fields: {', '.join(required)}."
    if optional:
        fields += f" Optional fields: {', '.join(optional)}."
    return f"\n\n{fields}{_JSON_HINT_BASE}"


def _append_json_hint(prompt: Any, schema: Optional[type[T]] = None) -> Any:
    """Append the JSON hint to a string prompt or to the last message in a list."""
    hint = _build_json_hint(schema)
    if isinstance(prompt, str):
        return prompt + hint
    if isinstance(prompt, list) and prompt:
        last = prompt[-1]
        if isinstance(last, dict) and "content" in last:
            return prompt[:-1] + [{**last, "content": last["content"] + hint}]
    return prompt


def _extract_json(text: str, schema: type[T]) -> T:
    """Extract the first JSON object from a noisy LLM response and parse it.

    Handles:
    - Thinking tags (``<think>...</think>``) before the JSON
    - Markdown code fences (````json ... ````  or ````` ... `````)
    - JSON objects embedded in surrounding prose
    """
    # Strip thinking tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try markdown code block first (most common when model ignores JSON-only instruction)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return schema(**json.loads(m.group(1)))

    # Find outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return schema(**json.loads(text[start:end + 1]))

    raise ValueError("no JSON object found in response")


def invoke_structured_or_freetext(
    structured_llm: Optional[Any],
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    schema: type[T],
    agent_name: str,
) -> str:
    """Run the structured call and render to markdown; fall back gracefully on failure.

    Execution order:
    1. ``structured_llm.invoke(prompt)`` — provider's native structured output
    2. ``plain_llm.invoke(prompt + json_hint)`` + manual JSON extraction —
       handles cases where the model returns text instead of a tool call
    3. Return the raw prose from step 2 as a last resort

    Step 2 reuses the response already fetched in the fallback call, so the
    total number of LLM calls is at most 2 regardless of which path fires.
    """
    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            return render(result)
        except Exception as exc:
            logger.warning(
                "%s: structured-output invocation failed (%s); "
                "attempting JSON extraction from plain-text response",
                agent_name, exc,
            )

    # Plain-text call with a JSON nudge appended
    hinted_prompt = _append_json_hint(prompt, schema)
    response = plain_llm.invoke(hinted_prompt)
    raw = response.content

    try:
        result = _extract_json(raw, schema)
        logger.info("%s: JSON extraction succeeded on fallback response", agent_name)
        return render(result)
    except Exception as exc:
        logger.warning(
            "%s: JSON extraction failed (%s); returning free-text response",
            agent_name, exc,
        )

    return raw
