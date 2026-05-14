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

---

## LM Studio vs Ollama: Debugging & Conclusion

### Background

During testing with `qwen3.6:27b-coding-nvfp4`, a switch was made to LM Studio under the (incorrect) belief that Ollama was not using the MLX backend. This section documents the investigation and conclusion.

### What the logs actually show

**Ollama server log:**
```
starting mlx runner subprocess    model=qwen3.6:27b-coding-nvfp4
MLX engine initialized             MLX version=0.31.2-7-ge8ebdeb  device=gpu
mlx runner is ready
```

**LM Studio** uses `llm_engine_mlx_amphibian.node` — also MLX.

Both run the same model via the same MLX framework on Apple Silicon. The `library=Metal` line in Ollama's GPU detection log refers to the hardware interface, not the inference path — the actual runner is MLX.

### Comparison

| | Ollama | LM Studio |
|---|---|---|
| MLX backend | ✅ Automatic (detected from model format) | ✅ |
| Parallel requests | ✅ `OLLAMA_NUM_PARALLEL` | ❌ Not yet supported for MLX backend |
| Generation speed | ~5–8 tok/s | ~5–8 tok/s (identical) |
| Stability | Stable | Crashed (500 error) during testing |
| Setup | CLI, scriptable | Requires GUI app running |

### Key correction

Ollama has always been running `qwen3.6:27b-coding-nvfp4` via the MLX runner. The earlier claim that "Ollama MLX was not available, using Apple Metal" was simply wrong. The `library=Metal` line in the log is just GPU hardware detection, not the inference path. The actual runner is MLX, as shown by `starting mlx runner subprocess` and `MLX engine initialized`.

### Conclusion

There is no speed advantage to LM Studio — both backends are MLX at identical throughput. Ollama is the better choice for TradingAgents: more stable, supports parallel request queuing, and doesn't require a GUI. The switch to LM Studio was unnecessary.
