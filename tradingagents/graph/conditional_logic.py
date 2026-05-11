# TradingAgents/graph/conditional_logic.py

from tradingagents.agents.utils.agent_states import AgentState


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def should_continue_market(self, state: AgentState):
        messages = state.get("market_messages", [])
        if messages and messages[-1].tool_calls:
            return "tools_market"
        return "Msg Clear Market"

    def should_continue_social(self, state: AgentState):
        messages = state.get("social_messages", [])
        if messages and messages[-1].tool_calls:
            return "tools_social"
        return "Msg Clear Social"

    def should_continue_news(self, state: AgentState):
        messages = state.get("news_messages", [])
        if messages and messages[-1].tool_calls:
            return "tools_news"
        return "Msg Clear News"

    def should_continue_fundamentals(self, state: AgentState):
        messages = state.get("fundamentals_messages", [])
        if messages and messages[-1].tool_calls:
            return "tools_fundamentals"
        return "Msg Clear Fundamentals"

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue."""
        if state["investment_debate_state"]["count"] >= self.max_debate_rounds:
            return "Research Manager"
        return "Research Round Router"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        if state["risk_debate_state"]["count"] >= self.max_risk_discuss_rounds:
            return "Portfolio Manager"
        return "Risk Round Router"
