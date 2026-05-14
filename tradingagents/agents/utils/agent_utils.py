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


def ensure_user_message(messages: list) -> list:
    """Ensure messages start with a HumanMessage when provider is lmstudio.

    LM Studio's Qwen3.6 jinja template requires at least one user turn at the
    start of every call. The seed is not stored in state so we must prepend it
    on every invoke — including mid-ReAct iterations where state starts with
    an AIMessage (tool call) followed by ToolMessages.
    Ollama and cloud providers are unaffected.
    """
    from tradingagents.dataflows.config import get_config
    if get_config().get("llm_provider", "").lower() != "lmstudio":
        return messages
    if not messages or not isinstance(messages[0], HumanMessage):
        return [HumanMessage(content=f"{no_think_prefix()}Begin your analysis.")] + list(messages)
    return messages


def no_think_prefix() -> str:
    """Return '/no_think\\n' for local providers (ollama, lmstudio) to suppress Qwen3 thinking.

    Cloud providers are unaffected.
    """
    from tradingagents.dataflows.config import get_config
    if get_config().get("llm_provider", "").lower() in ("ollama", "lmstudio"):
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


        
