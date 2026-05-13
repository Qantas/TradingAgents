"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from tradingagents.agents.utils.structured import _remap_fields_generic


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# LLM output coercion helpers
# ---------------------------------------------------------------------------

_NONE_STRINGS = frozenset({
    "none", "null", "n/a", "na", "nan", "undefined", "unknown",
    "not applicable", "not available", "n.a.", "n.a", "-", "",
})

# Financial synonym → canonical enum value for PortfolioRating
_PORTFOLIO_SYNONYMS: dict[str, str] = {
    "accumulate": "Buy",
    "outperform": "Buy",
    "strong buy": "Buy",
    "long": "Buy",
    "positive": "Overweight",
    "add": "Overweight",
    "increase": "Overweight",
    "market perform": "Hold",
    "market weight": "Hold",
    "in line": "Hold",
    "neutral": "Hold",
    "reduce": "Underweight",
    "trim": "Underweight",
    "underperform": "Underweight",
    "avoid": "Sell",
    "exit": "Sell",
    "short": "Sell",
    "strong sell": "Sell",
}

# Financial synonym → canonical enum value for TraderAction (3-tier)
_TRADER_SYNONYMS: dict[str, str] = {
    "accumulate": "Buy",
    "long": "Buy",
    "purchase": "Buy",
    "acquire": "Buy",
    "outperform": "Buy",
    "strong buy": "Buy",
    "overweight": "Buy",
    "short": "Sell",
    "exit": "Sell",
    "divest": "Sell",
    "reduce": "Sell",
    "trim": "Sell",
    "avoid": "Sell",
    "underweight": "Sell",
    "neutral": "Hold",
    "maintain": "Hold",
    "keep": "Hold",
    "market perform": "Hold",
}


def _clean_str(v: str) -> str:
    """Strip markdown formatting, surrounding quotes, and trailing punctuation."""
    v = re.sub(r"[*_`]", "", v)       # bold/italic/code markers
    v = v.strip("\"'")                 # surrounding quotes
    v = v.rstrip(".,!?;:")             # trailing punctuation
    return v.strip()


def _coerce_enum(v, enum_class, synonyms: Optional[dict] = None):
    """Coerce a noisy LLM string to a canonical enum value.

    Tries in order:
    1. Exact case-insensitive match after stripping markdown/punctuation
    2. First token match (handles "Buy recommendation", "Buy.")
    3. Whole-word substring match (handles "Recommendation: Buy", "I recommend Overweight")
    4. Synonym map lookup (handles financial jargon like "Accumulate" → "Buy")
    Falls through to the original value so Pydantic raises a clear error.
    """
    if not isinstance(v, str):
        return v

    cleaned = _clean_str(v)
    lower = cleaned.lower()

    # 1. Exact case-insensitive
    for member in enum_class:
        if lower == member.value.lower():
            return member.value

    # 2. First token
    first = lower.split()[0] if lower.split() else lower
    for member in enum_class:
        if first == member.value.lower():
            return member.value

    # 3. Whole-word substring (e.g. "Recommendation: Overweight" or "strongly Overweight")
    for member in enum_class:
        pattern = r"\b" + re.escape(member.value.lower()) + r"\b"
        if re.search(pattern, lower):
            return member.value

    # 4. Synonym map — financial jargon and model-specific phrasing
    if synonyms:
        if lower in synonyms:
            return synonyms[lower]
        if first in synonyms:
            return synonyms[first]
        for syn, canonical in synonyms.items():
            if syn in lower:
                return canonical

    return v  # let Pydantic raise with a meaningful error


def _coerce_str(v) -> str:
    """Coerce LLM-generated value to a plain string.

    Handles lists (bullet-joined), dicts (JSON-dumped), and anything else
    that a model might return instead of a plain prose string.
    """
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        parts = []
        for item in v:
            parts.append(str(item) if not isinstance(item, dict) else json.dumps(item, indent=2))
        return "\n".join(parts)
    if isinstance(v, dict):
        return json.dumps(v, indent=2)
    return str(v)


def _coerce_float(v):
    """Coerce LLM-generated strings to float or None.

    Handles: null-ish strings, currency prefixes ($€£¥), thousands commas,
    approximation prefixes (~≈><), ranges (605-610 → 605), trailing
    annotations ("605 (current price)"), and markdown formatting.
    Returns None for Optional fields when parsing is impossible.
    """
    if v is None or isinstance(v, (int, float)):
        return v
    if not isinstance(v, str):
        return v

    stripped = v.strip()
    if stripped.lower() in _NONE_STRINGS:
        return None

    cleaned = re.sub(r"[*_`\"']", "", stripped).strip()   # markdown / quotes
    cleaned = re.sub(r"[$€£¥₩₹]", "", cleaned).strip()    # currency symbols
    cleaned = re.sub(r"^[~≈><≤≥+\s]+", "", cleaned).strip()  # approx prefixes
    cleaned = cleaned.replace(",", "")                      # thousands separator

    # Take first numeric token; covers ranges ("605-610"), annotations ("605 USD")
    m = re.match(r"^(-?\d+(?:\.\d+)?)", cleaned)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    return None  # fallback for Optional[float] fields


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def remap_fields(cls, data):
        if not isinstance(data, dict):
            return data
        return _remap_fields_generic(data, cls.model_fields)

    @field_validator("rationale", "strategic_actions", mode="before")
    @classmethod
    def coerce_prose(cls, v):
        return _coerce_str(v)

    @field_validator("recommendation", mode="before")
    @classmethod
    def coerce_recommendation(cls, v):
        return _coerce_enum(v, PortfolioRating, _PORTFOLIO_SYNONYMS)


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    position_sizing: Optional[str] = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )

    @model_validator(mode="before")
    @classmethod
    def remap_fields(cls, data):
        if not isinstance(data, dict):
            return data
        return _remap_fields_generic(data, cls.model_fields)

    @field_validator("reasoning", "position_sizing", mode="before")
    @classmethod
    def coerce_prose(cls, v):
        return _coerce_str(v) if v is not None else v

    @field_validator("action", mode="before")
    @classmethod
    def coerce_action(cls, v):
        return _coerce_enum(v, TraderAction, _TRADER_SYNONYMS)

    @field_validator("entry_price", "stop_loss", mode="before")
    @classmethod
    def coerce_price(cls, v):
        return _coerce_float(v)


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: Optional[float] = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )

    @model_validator(mode="before")
    @classmethod
    def remap_fields(cls, data):
        if not isinstance(data, dict):
            return data
        return _remap_fields_generic(data, cls.model_fields)

    @field_validator("executive_summary", "investment_thesis", "time_horizon", mode="before")
    @classmethod
    def coerce_prose(cls, v):
        return _coerce_str(v) if v is not None else v

    @field_validator("rating", mode="before")
    @classmethod
    def coerce_rating(cls, v):
        return _coerce_enum(v, PortfolioRating, _PORTFOLIO_SYNONYMS)

    @field_validator("price_target", mode="before")
    @classmethod
    def coerce_price_target(cls, v):
        return _coerce_float(v)


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    return "\n".join(parts)
