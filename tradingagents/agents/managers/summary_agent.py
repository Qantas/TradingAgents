import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.prompts import ChatPromptTemplate
from tradingagents.agents.utils.agent_utils import get_language_instruction, no_think_prefix

_TAIL_CHARS = 1500
_EXTRACT_MAX_TOKENS = 1000
_TOTAL_BUDGET = 30_000


def _trim_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return "...\n" + text[-max_chars:]


def _extract_agent_verdict(llm, name: str, text: str) -> tuple[str, str]:
    tail = text[-_TAIL_CHARS:] if len(text) > _TAIL_CHARS else text
    extract_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Extract the final trading verdict from this agent report. "
         "Output ONLY these five fields, one per line, nothing else:\n"
         "Action: BUY / SELL / SHORT / HOLD / WAIT (pick the single most specific action)\n"
         "Entry: exact price or price zone stated in the report, or —\n"
         "Stop: stop-loss level stated in the report, or —\n"
         "Target: price target stated in the report, or —\n"
         "Rationale: one sentence, max 15 words, capturing the core reason\n"
         "Use exact prices where stated. Do not add commentary or explanation."),
        ("human", no_think_prefix("summary") + "Agent: {name}\n\n{text}"),
    ])
    result = (extract_prompt | llm.bind(max_tokens=_EXTRACT_MAX_TOKENS)).invoke({"name": name, "text": tail})
    return name, result.content


def create_summary_agent(llm):
    def summary_agent_node(state):
        from tradingagents.dataflows.config import get_config
        debate = state.get("investment_debate_state", {})
        risk = state.get("risk_debate_state", {})

        sections = [
            ("Market Analyst", state.get("market_report", "")),
            ("Social Analyst", state.get("sentiment_report", "")),
            ("News Analyst", state.get("news_report", "")),
            ("Fundamentals Analyst", state.get("fundamentals_report", "")),
            ("Bull Researcher", debate.get("bull_history", "")),
            ("Bear Researcher", debate.get("bear_history", "")),
            ("Research Manager", debate.get("judge_decision", "")),
            ("Trader", state.get("trader_investment_plan", "")),
            ("Aggressive Risk Analyst", risk.get("aggressive_history", "")),
            ("Conservative Risk Analyst", risk.get("conservative_history", "")),
            ("Neutral Risk Analyst", risk.get("neutral_history", "")),
            ("Portfolio Manager", risk.get("judge_decision", "")),
        ]

        raw = [(name, text) for name, text in sections if text]

        provider = get_config().get("llm_provider", "").lower()
        # Ollama and LM Studio serialize inference server-side; N threads just queues
        # requests and risks connection pool exhaustion. Cloud providers run in parallel.
        max_workers = 1 if provider in ("lmstudio", "ollama") else min(len(raw), 4)

        extracted: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_extract_agent_verdict, llm, name, text): name
                for name, text in raw
            }
            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    _, verdict = future.result()
                    if verdict.strip():
                        extracted[agent_name] = verdict
                except Exception as exc:
                    logging.warning("summary: extraction failed for %s: %s", agent_name, exc)

        if extracted:
            content = "\n\n".join(
                f"## {name}\n{extracted[name]}"
                for name, _ in raw
                if name in extracted
            )
        else:
            # All extractions failed — fall back to trimmed raw tails so the format
            # LLM receives real prices rather than generating a hypothetical example.
            logging.warning("summary: all extractions failed, falling back to raw tails")
            total_chars = sum(len(t) for _, t in raw)
            if total_chars > _TOTAL_BUDGET:
                per_section = _TOTAL_BUDGET // len(raw)
                pairs = [(n, _trim_tail(t, per_section)) for n, t in raw]
            else:
                pairs = raw
            content = "\n\n".join(f"## {name}\n{text}" for name, text in pairs)

        system_message = (
            "You are a financial report summarizer. Given structured verdicts from multiple trading agents,"
            " produce a concise action summary organized by decision-making hierarchy"
            " (most authoritative agents first).\n\n"
            "**Output format (markdown):**\n\n"
            "**Section 1 — Agent Action Tables**\n"
            "Produce four separate subsection tables ordered from final decision-makers down to data gatherers.\n"
            "Each table has columns: | Agent | Action | Entry | Stop | Target | Rationale |\n"
            "- Action: BUY / SELL / SHORT / HOLD / WAIT / OVERWEIGHT / UNDERWEIGHT\n"
            "- Entry, Stop, Target: price or zone; use — if not specified\n"
            "- Rationale: one sentence, max 15 words\n\n"
            "Subsections (in this order — most important first):\n\n"
            "### Risk Management Team\n"
            "Rows in order: Portfolio Manager (final decision), Aggressive Risk Analyst, Conservative Risk Analyst, Neutral Risk Analyst\n\n"
            "### Trading Team\n"
            "Rows: Trader\n\n"
            "### Research Team\n"
            "Rows in order: Research Manager (synthesizer), Bull Researcher, Bear Researcher\n\n"
            "### Analyst Team\n"
            "Rows: Market Analyst, Social Analyst, News Analyst, Fundamentals Analyst\n\n"
            "**Section 2 — ## Consensus**\n"
            "- **Direction**: overall bias across agents\n"
            "- **Entry zone**: price range where the majority agree to enter\n"
            "- **Stop**: level where the bull thesis is invalidated\n"
            "- **Target**: primary price target\n"
            "- **Key risk**: single biggest risk to the thesis\n\n"
            "**Section 3 — ## Personal Investor Recommendation**\n"
            "Written in plain language for an individual investor (no jargon, no portfolio % sizing).\n"
            "Include:\n"
            "- **Verdict**: BUY / HOLD / AVOID — one clear call with a one-sentence reason\n"
            "- **How to enter**: lump sum vs. scale in; suggested price zone to start buying\n"
            "- **What to watch**: 2–3 concrete catalysts or signals that would confirm or invalidate the thesis\n"
            "- **Downside scenario**: how far the stock could fall and what would cause it\n"
            "- **Time horizon**: realistic holding period for the thesis to play out\n\n"
            "Be precise. Use exact prices where agents stated them."
            + get_language_instruction()
        )

        format_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", no_think_prefix("summary") + "{content}"),
        ])

        result = (format_prompt | llm).invoke({"content": content})
        return {"action_summary": result.content}

    return summary_agent_node
