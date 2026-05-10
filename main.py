import sys
import time
from datetime import date
from pathlib import Path
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from cli.main import save_report_to_disk

from dotenv import load_dotenv

load_dotenv()

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "ollama"
config["deep_think_llm"] = "qwen3.6:27b-coding-nvfp4"
config["quick_think_llm"] = "qwen3.6:27b-coding-nvfp4"
config["max_debate_rounds"] = 1

config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
}

TICKERS = ["NVDA", "AAPL", "MSFT"]
TRADE_DATE = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()

meta = {
    "llm_provider": config["llm_provider"],
    "deep_thinker": config["deep_think_llm"],
    "shallow_thinker": config["quick_think_llm"],
    "research_depth": config["max_debate_rounds"],
    "research_depth_label": {1: "Shallow", 3: "Medium", 5: "Deep"}.get(config["max_debate_rounds"], str(config["max_debate_rounds"])),
}

ta = TradingAgentsGraph(debug=True, config=config)

for ticker in TICKERS:
    print(f"\n{'='*60}")
    print(f"Analyzing {ticker} on {TRADE_DATE}")
    print(f"{'='*60}\n")

    start = time.time()
    final_state, decision = ta.propagate(ticker, TRADE_DATE)
    elapsed = time.time() - start

    print(f"{ticker} decision: {decision}")

    save_path = Path("reports") / f"{ticker}_{TRADE_DATE.replace('-', '')}"
    timing = {"_total_seconds": elapsed}
    save_report_to_disk(final_state, ticker, save_path, timing=timing, meta=meta)
    print(f"Report saved to: {save_path}")
