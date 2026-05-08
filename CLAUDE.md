# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TradingAgents is a multi-agent LLM financial trading framework (v0.2.4). It deploys specialized agents — analysts, researchers, a trader, risk debators, and a portfolio manager — in a LangGraph state machine that produces a buy/hold/sell decision for a given ticker and date.

## Commands

**Install**: `pip install .` (requires Python >=3.10, uses setuptools)
**Run CLI**: `tradingagents` or `python -m cli.main`
**Run tests**: `pytest` (discover under `tests/`; markers: `unit`, `integration`, `smoke`)
**Run a single test**: `pytest tests/test_foo.py -v`
**Docker build**: `docker compose build && docker compose run --rm tradingagents`

## Architecture

### Top-level layout

```
TradingAgents/
├── tradingagents/          # Core library (installable package)
│   ├── agents/             # Agent node definitions + schemas
│   ├── dataflows/          # Data vendor abstractions (yfinance, alpha_vantage)
│   ├── graph/              # LangGraph workflow: setup, propagation, conditional logic
│   └── llm_clients/        # Multi-provider LLM wrappers (OpenAI, Anthropic, Google, etc.)
├── cli/                    # CLI entry point (Typer + Rich TUI)
├── tests/                  # Pytest suite
├── main.py                 # Example script (programmatic usage)
└── test.py                 # Quick smoke test for yfinance data flows
```

### Core graph pipeline

1. **Entry point**: `TradingAgentsGraph` in `tradingagents/graph/trading_graph.py` orchestrates everything. Call `.propagate(ticker, date)` to run the full pipeline.
2. **Graph setup**: `GraphSetup.setup_graph()` (in `graph/setup.py`) builds a LangGraph `StateGraph` with `AgentState`. Nodes run in sequence: analysts → bull/bear researchers → research manager → trader → risk debators → portfolio manager.
3. **Conditional edges**: `ConditionalLogic` (in `graph/conditional_logic.py`) decides when to continue debates vs move to the next node.
4. **State**: `AgentState` (in `agents/utils/agent_states.py`) carries reports, debate histories, and final decisions through the graph. `InvestDebateState` and `RiskDebateState` are sub-states for the debate loops.
5. **Propagation**: `Propagator` (in `graph/propagation.py`) creates the initial state and graph args.

### Agents

All agents live in `tradingagents/agents/`. Each agent is a LangChain agent wrapped into a callable node.

- **Analysts** (`agents/analysts/`): market, social_media, news, fundamentals. Each fetches data via tool nodes and produces a report.
- **Researchers** (`agents/researchers/`): bull_researcher, bear_researcher. Debate each other's analysis.
- **Risk debators** (`agents/risk_mgmt/`): aggressive_debator, neutral_debator, conservative_debator. Evaluate risk from different perspectives.
- **Managers** (`agents/managers/`): research_manager (synthesizes researcher debate into an investment plan), portfolio_manager (final buy/hold/sell decision).
- **Trader** (`agents/trader/trader.py`): translates research plan into a concrete transaction proposal.
- **Schemas** (`agents/schemas.py`): Pydantic models for structured output (ResearchPlan, TraderProposal, PortfolioDecision).

### Data layer

`tradingagents/dataflows/` provides a vendor-abstracted data API:

- `interface.py`: routing layer. `route_to_vendor(method, *args)` dispatches to the configured vendor (yfinance or alpha_vantage). Supports category-level and tool-level vendor config, plus fallback chains.
- `config.py`: global config store. `set_config()` writes, `get_config()` reads.
- `y_finance.py`: yfinance-based implementations.
- `alpha_vantage*.py`: Alpha Vantage implementations (stock, fundamentals, indicators, news).
- `stockstats_utils.py`: technical indicator computation using stockstats.

### LLM clients

`tradingagents/llm_clients/`:

- `factory.py`: `create_llm_client(provider, model, base_url)` — lazy-imports provider modules.
- Providers: openai, anthropic, google, xai, deepseek, qwen, glm, ollama, openrouter (all use `OpenAIClient`), azure (separate client), plus provider-specific clients.
- `base_client.py`: abstract base class.
- `model_catalog.py`: model name mappings per provider.
- `validators.py`: input validation utilities.

### CLI

`cli/main.py`: Typer CLI with a Rich-based live TUI. `tradingagents` command triggers `run_analysis()`, which builds a `TradingAgentsGraph`, streams graph output, and updates a live dashboard showing agent progress, messages, and report sections. Supports `--checkpoint` and `--clear-checkpoints` flags.

### Persistence

- **Decision log**: `~/.tradingagents/memory/trading_memory.md` — appends decisions, resolves returns on subsequent runs, injects reflections into prompts.
- **Checkpoints**: `~/.tradingagents/cache/checkpoints/<TICKER>.db` — SQLite-based LangGraph checkpoints for resume.

## Key conventions

- The graph is the source of truth. All agent flow goes through `TradingAgentsGraph.propagate()`.
- Data vendors are pluggable via `data_vendors` / `tool_vendors` in config. Default is yfinance (no API key needed).
- LLM providers are selected via `llm_provider` + API key env var. Multiple providers can be configured simultaneously.
- Structured output (Pydantic schemas) is layered on top of the prose-based agents for the three decision-making agents only.
- Tests use `conftest.py` autouse fixtures that dummy all API keys. Mark tests with `@pytest.mark.unit` etc.
