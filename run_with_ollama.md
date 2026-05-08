# Run TradingAgents with Ollama (Qwen3.6 27B)

## Prerequisites

1. Install [Ollama](https://ollama.ai)
2. Pull the model:

```bash
ollama pull qwen3.6:27b-q4_K_M
```

3. Start Ollama (if not already running as a background service):

```bash
ollama serve
```

4. Activate the conda environment and install dependencies:

```bash
conda activate tradingagents
pip install .
```

5. Verify the model is available:

```bash
ollama list
```

## CLI Mode

```bash
conda activate tradingagents

# Launch the interactive CLI
tradingagents
```

When prompted, select `ollama` as the provider and `qwen3.6:27b-q4_K_M` for both deep and shallow thinker.

## Programmatic Mode

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "ollama"
config["deep_think_llm"] = "qwen3.6:27b-q4_K_M"
config["quick_think_llm"] = "qwen3.6:27b-q4_K_M"

ta = TradingAgentsGraph(config=config)
_, decision = ta.propagate("NVDA", "2026-01-15")
print(decision)
```

Run it with:

```bash
conda activate tradingagents
python your_script.py
```

No API key is required — Ollama runs locally without authentication.
