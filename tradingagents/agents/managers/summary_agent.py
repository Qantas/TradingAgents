from langchain_core.prompts import ChatPromptTemplate
from tradingagents.agents.utils.agent_utils import get_language_instruction, no_think_prefix


def create_summary_agent(llm):
    def summary_agent_node(state):
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

        content = "\n\n".join(
            f"## {name}\n{text}" for name, text in sections if text
        )

        system_message = (
            no_think_prefix() + "You are a financial report summarizer. Given outputs from multiple trading agents,"
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

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{content}"),
        ])

        result = (prompt | llm).invoke({"content": content})
        return {"action_summary": result.content}

    return summary_agent_node
