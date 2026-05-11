def create_research_debate_sync():
    def sync_node(state) -> dict:
        debate = state["investment_debate_state"]
        bull_resp = debate.get("current_bull_response", "")
        bear_resp = debate.get("current_bear_response", "")
        history = debate.get("history", "")
        count = debate.get("count", 0)

        new_history = history
        if bull_resp:
            new_history += "\n" + bull_resp
        if bear_resp:
            new_history += "\n" + bear_resp

        return {"investment_debate_state": {"history": new_history, "count": count + 1}}

    return sync_node


def create_risk_debate_sync():
    def sync_node(state) -> dict:
        risk = state["risk_debate_state"]
        agg_resp = risk.get("current_aggressive_response", "")
        con_resp = risk.get("current_conservative_response", "")
        neu_resp = risk.get("current_neutral_response", "")
        history = risk.get("history", "")
        count = risk.get("count", 0)

        new_history = history
        if agg_resp:
            new_history += "\n" + agg_resp
        if con_resp:
            new_history += "\n" + con_resp
        if neu_resp:
            new_history += "\n" + neu_resp

        return {"risk_debate_state": {"history": new_history, "count": count + 1}}

    return sync_node
