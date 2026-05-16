from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)


def ensure_user_message(messages: list, agent_name: str = "") -> list:
    """Ensure the first user turn carries the /no_think prefix for local Qwen3 providers.

    LM Studio requires at least one user turn at the start of every call due to
    its jinja template. Ollama additionally needs the /no_think prefix in that
    first turn on every call — including mid-ReAct iterations after create_msg_delete
    replaces state with [HumanMessage("Continue")], which has no prefix.
    Cloud providers are unaffected.
    """
    from tradingagents.dataflows.config import get_config
    provider = get_config().get("llm_provider", "").lower()
    if provider not in ("ollama", "lmstudio"):
        return messages
    prefix = no_think_prefix(agent_name)
    if not messages or not isinstance(messages[0], HumanMessage):
        return [HumanMessage(content=f"{prefix}Begin your analysis.")] + list(messages)
    # For Ollama: ensure the prefix is in the first user turn on every call.
    # For LM Studio: having any HumanMessage first satisfies the template.
    if provider == "ollama" and prefix:
        content = messages[0].content
        if isinstance(content, str) and not content.startswith(prefix):
            return [HumanMessage(content=prefix + content)] + list(messages[1:])
    return messages


def no_think_prefix(agent_name: str = "") -> str:
    """Return '/no_think\\n' for local providers unless agent is in thinking_agents config.

    Cloud providers are unaffected (no prefix needed).
    Set thinking_agents in config to a set/list of agent names that should use thinking.
    Agent names: market, social, news, fundamentals, bull, bear, research_manager,
                 trader, aggressive, conservative, neutral, portfolio_manager, summary.
    """
    from tradingagents.dataflows.config import get_config
    cfg = get_config()
    thinking_agents = cfg.get("thinking_agents", set())
    if agent_name and agent_name in thinking_agents:
        return ""
    if cfg.get("llm_provider", "").lower() in ("ollama", "lmstudio"):
        return "/no_think\n"
    return ""


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Only applied to user-facing agents (analysts, portfolio manager).
    Internal debate agents stay in English for reasoning quality.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )

def create_msg_delete(messages_key="messages"):
    def delete_messages(state):
        messages = state.get(messages_key, [])
        removal_operations = [RemoveMessage(id=m.id) for m in messages]
        placeholder = HumanMessage(content="Continue")
        return {messages_key: removal_operations + [placeholder]}
    return delete_messages


        
