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


def _is_local_provider() -> bool:
    from tradingagents.dataflows.config import get_config
    return get_config().get("llm_provider", "").lower() in ("ollama", "lmstudio")


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Optional[Any]:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        # Ollama 0.30+ with the llama.cpp backend requires think=False at the
        # API level; the /no_think prompt prefix alone is no longer sufficient.
        if _is_local_provider():
            llm = llm.bind(extra_body={"think": False})
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


def _remap_fields_generic(data: dict, model_fields: dict) -> dict:
    """Remap unknown LLM output keys to canonical Pydantic field names.

    Two-phase strategy so callers never need hand-coded alias lists:

    Phase 1 — word-overlap: tokenize the incoming key against each canonical
    field name *and* its description text.  Best match above a 0.25 threshold
    wins (e.g. "debate_evaluation" → "rationale" because "debate" appears in
    rationale's description).

    Phase 2 — length heuristic: remaining unmatched *required* string fields
    are paired with alien string values sorted by length — longer values go to
    the field whose description is also longer (prose fields get prose values).
    """
    def _words(s: str) -> set[str]:
        return {w for w in re.split(r"\W+", s.lower()) if len(w) > 2}

    canonical = set(model_fields.keys())
    alien_keys = [k for k in data if k not in canonical]
    free_fields = list(canonical - (canonical & set(data.keys())))

    if not alien_keys or not free_fields:
        return data

    result = dict(data)
    remapped: set[str] = set()
    taken: set[str] = set()

    def _vocab(fname: str) -> set[str]:
        return _words(fname) | _words(model_fields[fname].description or "")

    # Phase 1: word-overlap scoring
    for key in alien_keys:
        key_words = _words(key)
        if not key_words:
            continue
        best_field, best_score = None, 0.0
        for f in free_fields:
            if f in taken:
                continue
            common = key_words & _vocab(f)
            if not common:
                continue
            score = len(common) / max(len(key_words), len(_words(f)) or 1)
            if score > best_score:
                best_score, best_field = score, f
        if best_field and best_score >= 0.25:
            result[best_field] = result.pop(key)
            remapped.add(key)
            taken.add(best_field)

    # Phase 2: length-based assignment for still-unmatched required fields
    # Handles any value type — dicts/lists land on string fields where
    # field_validators (_coerce_str) will convert them.
    unremapped = [k for k in alien_keys if k not in remapped and k in result]
    req_free = [
        f for f in free_fields
        if f not in taken and model_fields[f].is_required()
    ]
    if not unremapped or not req_free:
        return result

    # Sort: longer string values first; non-strings go last (they can't use length heuristic)
    def _value_len(k: str) -> int:
        v = result.get(k)
        return len(str(v)) if isinstance(v, str) else -1

    unremapped.sort(key=_value_len, reverse=True)
    req_free.sort(key=lambda f: len(model_fields[f].description or ""), reverse=True)
    for key, field in zip(unremapped, req_free):
        result[field] = result.pop(key)

    return result


def _extract_json(text: str, schema: type[T]) -> T:
    """Extract the first JSON object from a noisy LLM response and parse it.

    Handles:
    - Thinking tags (``<think>...</think>``) before the JSON
    - Markdown code fences (````json ... ````  or ````` ... `````)
    - JSON objects embedded in surrounding prose
    """
    # Strip thinking tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _parse(raw_json: str) -> T:
        data = json.loads(raw_json)
        data = _remap_fields_generic(data, schema.model_fields)
        return schema(**data)

    # Try markdown code block first (most common when model ignores JSON-only instruction)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return _parse(m.group(1))

    # Find outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return _parse(text[start:end + 1])

    raise ValueError("no JSON object found in response")


def _force_no_think(prompt: Any) -> Any:
    """Prepend /no_think to the first user turn for Ollama/LM Studio.

    Structured output (response_format JSON schema) + Qwen3 thinking mode
    causes the model to spend all tokens on reasoning and produce empty
    content. Suppress thinking unconditionally for structured calls
    regardless of the thinking_agents config.
    """
    from tradingagents.dataflows.config import get_config
    if get_config().get("llm_provider", "").lower() not in ("ollama", "lmstudio"):
        return prompt
    prefix = "/no_think\n"
    if isinstance(prompt, str):
        return prompt if prompt.startswith(prefix) else prefix + prompt
    if isinstance(prompt, list):
        for i, msg in enumerate(prompt):
            if isinstance(msg, dict) and msg.get("role") in ("user", "human"):
                content = msg.get("content", "")
                if isinstance(content, str) and not content.startswith(prefix):
                    return prompt[:i] + [{**msg, "content": prefix + content}] + prompt[i + 1:]
                return prompt
    return prompt


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
    no_think_prompt = _force_no_think(prompt)

    # Bind think=False at the API level for local providers so the fallback
    # plain-text call also suppresses thinking (prompt prefix alone is not
    # reliable on Ollama 0.30+ with the llama.cpp backend).
    if _is_local_provider():
        plain_llm = plain_llm.bind(extra_body={"think": False})

    if structured_llm is not None:
        try:
            result = structured_llm.invoke(no_think_prompt)
            return render(result)
        except Exception as exc:
            logger.warning(
                "%s: structured-output invocation failed (%s); "
                "attempting JSON extraction from plain-text response",
                agent_name, exc,
            )

    # Plain-text call with a JSON nudge appended
    hinted_prompt = _append_json_hint(no_think_prompt, schema)
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
