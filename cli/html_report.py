import base64
import datetime
import io
import re
from pathlib import Path
from typing import Dict, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    _MATPLOTLIB = True
except ImportError:
    _MATPLOTLIB = False

try:
    import markdown as _md_lib

    _MARKDOWN = True
except ImportError:
    _MARKDOWN = False

_AGENT_KEYS = [
    "Analyst Phase",
    "Bull Researcher",
    "Bear Researcher",
    "Research Manager",
    "Trader",
    "Aggressive Analyst",
    "Conservative Analyst",
    "Neutral Analyst",
    "Portfolio Manager",
    "Summary",
]

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #1a1a2e;
  color: #cdd6f4;
  line-height: 1.65;
  font-size: 14px;
}
.container { max-width: 1200px; margin: 0 auto; padding: 36px 24px; }
h1 { font-size: 1.75rem; color: #89b4fa; margin-bottom: 8px; }
h2 { font-size: 1.2rem; color: #89b4fa; margin: 36px 0 12px;
     border-bottom: 1px solid #313244; padding-bottom: 6px; }
h3 { font-size: 1rem; color: #a6e3a1; margin: 18px 0 6px; }
h4 { font-size: 0.93rem; color: #f9e2af; margin: 12px 0 5px; }
p { margin: 7px 0; }
ul, ol { margin: 7px 0 7px 22px; }
li { margin: 3px 0; }
a { color: #89b4fa; }
strong { color: #f9e2af; }
em { color: #cba6f7; }
code {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  background: #11111b;
  border: 1px solid #313244;
  border-radius: 3px;
  padding: 1px 5px;
  font-size: 0.87em;
}
pre { background: #11111b; border: 1px solid #313244; border-radius: 6px;
      padding: 12px; overflow-x: auto; margin: 10px 0; }
pre code { border: none; padding: 0; background: none; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.9em; }
th { background: #181825; color: #89b4fa; padding: 7px 12px;
     text-align: left; border: 1px solid #313244; font-weight: 600; }
td { padding: 6px 12px; border: 1px solid #313244; }
tr:nth-child(even) { background: #181825; }
tr:hover { background: #1e1e2e; }
.meta {
  background: #181825; border: 1px solid #313244; border-radius: 8px;
  padding: 14px 20px; margin: 14px 0 24px;
  display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 7px 24px; font-size: 0.88em;
}
.meta-item { display: flex; gap: 7px; }
.meta-label { color: #6c7086; white-space: nowrap; }
.meta-value { color: #cdd6f4; font-weight: 500; }
.chart-wrap { margin: 16px 0; }
.chart-wrap img { width: 100%; border-radius: 8px; border: 1px solid #313244; }
.section-card {
  background: #181825; border: 1px solid #313244; border-radius: 8px;
  padding: 18px 22px; margin: 14px 0;
}
.section-card h1, .section-card h2 { color: #cba6f7; font-size: 1.05rem; margin-top: 12px; border: none; }
.section-card h3 { color: #a6e3a1; }
.section-card h4 { color: #f9e2af; }
.agent-label { font-size: 0.78rem; color: #6c7086; text-transform: uppercase;
               letter-spacing: 0.1em; margin-bottom: 4px; }
"""

# Maps heading keyword patterns → chart key.
# Charts are injected once per key (first matching heading wins).
_HEADING_CHART_MAP = [
    (r"moving.averag|sma|ema|death.cross|golden.cross|trend", "price"),
    (r"bollinger", "price"),
    (r"\bmacd\b|momentum", "macd"),
    (r"\brsi\b", "rsi"),
    (r"\batr\b|volatility", "atr"),
    (r"volume|vwma", "volume"),
]


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _ax_style(ax, bg="#0d0d1a", grid="#1e1e2e", tick="#6c7086"):
    ax.set_facecolor(bg)
    ax.tick_params(colors=tick, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#333344")
    ax.grid(color=grid, linewidth=0.5)


def _xaxis(ax, tick="#6c7086"):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", color=tick, fontsize=8)


def _generate_charts(ticker: str, trade_date: str) -> Dict[str, str]:
    """Return {chart_key: base64_png} for each indicator group, or {} on failure."""
    if not _MATPLOTLIB:
        return {}
    try:
        import numpy as np
        import pandas as pd
        import yfinance as yf

        BG, TICK = "#1a1a2e", "#6c7086"

        end = pd.Timestamp(trade_date)
        df = yf.download(
            ticker,
            start=(end - pd.DateOffset(months=5)).strftime("%Y-%m-%d"),
            end=(end + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            return {}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"]
        volume = df["Volume"]

        ema10 = close.ewm(span=10, adjust=False).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        vwma20 = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_sig
        delta = close.diff()
        avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.where(avg_loss != 0, np.nan)
        rsi = (100 - 100 / (1 + rs)).fillna(100)
        # ATR
        prev_close = close.shift(1)
        tr = pd.concat([
            (df["High"] - df["Low"]),
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(com=13, adjust=False).mean()

        # Trim to 3-month display window
        mask = close.index >= (end - pd.DateOffset(months=3))
        close, volume = close[mask], volume[mask]
        ema10, sma50, sma200, vwma20 = ema10[mask], sma50[mask], sma200[mask], vwma20[mask]
        bb_upper, bb_lower = bb_upper[mask], bb_lower[mask]
        macd, macd_sig, macd_hist = macd[mask], macd_sig[mask], macd_hist[mask]
        rsi, atr = rsi[mask], atr[mask]

        charts = {}

        # ── Price + MAs + Bollinger Bands ──────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 4), facecolor=BG)
        _ax_style(ax)
        ax.plot(close.index, close, color="#e0e0f0", linewidth=1.3, label="Close", zorder=5)
        ax.plot(ema10.index, ema10, color="#89b4fa", linewidth=0.85, label="EMA 10")
        ax.plot(sma50.index, sma50, color="#a6e3a1", linewidth=0.85, label="SMA 50")
        ax.plot(sma200.index, sma200, color="#f38ba8", linewidth=0.85, label="SMA 200")
        ax.fill_between(close.index, bb_upper, bb_lower, alpha=0.08, color="#cba6f7")
        ax.plot(bb_upper.index, bb_upper, color="#cba6f7", linewidth=0.7, linestyle=":", label="BB ±2σ")
        ax.plot(bb_lower.index, bb_lower, color="#cba6f7", linewidth=0.7, linestyle=":")
        ax.set_ylabel("Price (USD)", color=TICK, fontsize=9)
        ax.set_title(f"{ticker} — Price / Moving Averages / Bollinger Bands", color="#cdd6f4", fontsize=10, pad=8)
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.4, facecolor=BG, labelcolor="#cdd6f4", edgecolor="#333344")
        _xaxis(ax)
        fig.tight_layout()
        charts["price"] = _fig_to_b64(fig)

        # ── MACD ───────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 3), facecolor=BG)
        _ax_style(ax)
        ax.bar(macd_hist.index, macd_hist.clip(lower=0), color="#a6e3a1", width=0.8, alpha=0.85)
        ax.bar(macd_hist.index, macd_hist.clip(upper=0), color="#f38ba8", width=0.8, alpha=0.85)
        ax.plot(macd.index, macd, color="#89b4fa", linewidth=1.1, label="MACD")
        ax.plot(macd_sig.index, macd_sig, color="#f9e2af", linewidth=1.1, label="Signal")
        ax.axhline(0, color="#444455", linewidth=0.8)
        ax.set_ylabel("MACD", color=TICK, fontsize=9)
        ax.set_title(f"{ticker} — MACD (12, 26, 9)", color="#cdd6f4", fontsize=10, pad=8)
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.4, facecolor=BG, labelcolor="#cdd6f4", edgecolor="#333344")
        _xaxis(ax)
        fig.tight_layout()
        charts["macd"] = _fig_to_b64(fig)

        # ── RSI ────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 2.5), facecolor=BG)
        _ax_style(ax)
        ax.plot(rsi.index, rsi, color="#89dceb", linewidth=1.1)
        ax.axhline(70, color="#f38ba8", linewidth=0.8, linestyle="--", alpha=0.8, label="Overbought (70)")
        ax.axhline(50, color="#444455", linewidth=0.6, alpha=0.7)
        ax.axhline(30, color="#a6e3a1", linewidth=0.8, linestyle="--", alpha=0.8, label="Oversold (30)")
        ax.fill_between(rsi.index, rsi, 70, where=(rsi >= 70), alpha=0.15, color="#f38ba8")
        ax.fill_between(rsi.index, rsi, 30, where=(rsi <= 30), alpha=0.15, color="#a6e3a1")
        ax.set_ylim(0, 100)
        ax.set_ylabel("RSI (14)", color=TICK, fontsize=9)
        ax.set_title(f"{ticker} — RSI (14)", color="#cdd6f4", fontsize=10, pad=8)
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.4, facecolor=BG, labelcolor="#cdd6f4", edgecolor="#333344")
        ax.yaxis.set_ticks([30, 50, 70])
        _xaxis(ax)
        fig.tight_layout()
        charts["rsi"] = _fig_to_b64(fig)

        # ── ATR ────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 2.5), facecolor=BG)
        _ax_style(ax)
        ax.plot(atr.index, atr, color="#cba6f7", linewidth=1.1)
        ax.set_ylabel("ATR (14)", color=TICK, fontsize=9)
        ax.set_title(f"{ticker} — Average True Range (14)", color="#cdd6f4", fontsize=10, pad=8)
        _xaxis(ax)
        fig.tight_layout()
        charts["atr"] = _fig_to_b64(fig)

        # ── Volume + VWMA ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 2.5), facecolor=BG)
        _ax_style(ax)
        bar_colors = [
            "#a6e3a1" if i == 0 or close.iloc[i] >= close.iloc[i - 1] else "#f38ba8"
            for i in range(len(close))
        ]
        ax.bar(volume.index, volume, color=bar_colors, width=0.8, alpha=0.75)
        ax2 = ax.twinx()
        ax2.set_facecolor("#0d0d1a")
        ax2.plot(vwma20.index, vwma20, color="#fab387", linewidth=1.1, label="VWMA 20")
        ax2.tick_params(colors=TICK, labelsize=8)
        ax2.set_ylabel("VWMA (USD)", color=TICK, fontsize=9)
        ax2.legend(loc="upper right", fontsize=7.5, framealpha=0.4, facecolor=BG, labelcolor="#cdd6f4", edgecolor="#333344")
        ax.set_ylabel("Volume", color=TICK, fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
        ax.set_title(f"{ticker} — Volume / VWMA", color="#cdd6f4", fontsize=10, pad=8)
        _xaxis(ax)
        fig.tight_layout()
        charts["volume"] = _fig_to_b64(fig)

        return charts

    except Exception:
        return {}


def _inject_charts(market_html: str, charts: Dict[str, str], ticker: str) -> str:
    """Insert chart images after the first heading whose text matches each indicator keyword."""
    injected = set()

    def img(key: str) -> str:
        return (
            f'<div class="chart-wrap">'
            f'<img src="data:image/png;base64,{charts[key]}" alt="{ticker} {key} chart">'
            f"</div>"
        )

    def replacer(m: re.Match) -> str:
        tag = m.group(0)
        text = re.sub(r"<[^>]+>", "", tag).lower()
        for pattern, key in _HEADING_CHART_MAP:
            if key in charts and key not in injected and re.search(pattern, text):
                injected.add(key)
                return tag + img(key)
        return tag

    return re.sub(r"<h[34][^>]*>.*?</h[34]>", replacer, market_html, flags=re.DOTALL)


def _to_html(text: str) -> str:
    if _MARKDOWN:
        return _md_lib.markdown(text, extensions=["tables", "fenced_code"])
    import html
    return f"<pre>{html.escape(text)}</pre>"


def _card(label: str, content: str, charts: Dict[str, str] = None, ticker: str = "") -> str:
    html_body = _to_html(content)
    if charts:
        html_body = _inject_charts(html_body, charts, ticker)
    return (
        f'<div class="section-card">'
        f'<p class="agent-label">{label}</p>'
        f"{html_body}"
        f"</div>"
    )


def save_html_report(
    final_state: dict,
    ticker: str,
    save_path,
    timing: dict = None,
    meta: dict = None,
) -> Path:
    save_path = Path(save_path)
    trade_date = final_state.get("trade_date", "")
    charts = _generate_charts(ticker, trade_date)

    # Metadata block
    if meta:
        depth_rounds = meta.get("research_depth", "")
        depth_label = meta.get("research_depth_label", "")
        provider = meta.get("llm_provider", "")
        deep = meta.get("deep_thinker", "")
        quick = meta.get("shallow_thinker", "")
        analysis_date = meta.get("analysis_date", "")
        analysts_list = meta.get("analysts", [])
        analysts_str = ", ".join(
            a.value if hasattr(a, "value") else str(a) for a in analysts_list
        ) if analysts_list else "all"
        backend_url = meta.get("backend_url", "")
        output_language = meta.get("output_language", "")
        google_thinking = meta.get("google_thinking_level", "")
        openai_effort = meta.get("openai_reasoning_effort", "")
        anthropic_effort = meta.get("anthropic_effort", "")
        thinking_agents_cfg = meta.get("thinking_agents", [])
        if provider.lower() in ("ollama", "lmstudio"):
            thinking_status = f"Selective ({', '.join(sorted(thinking_agents_cfg))})" if thinking_agents_cfg else "OFF (/no_think)"
        else:
            thinking_status = "ON (provider default)"
    else:
        depth_rounds = depth_label = provider = deep = quick = analysis_date = ""
        analysts_str = backend_url = output_language = google_thinking = ""
        openai_effort = anthropic_effort = thinking_status = ""
        thinking_agents_cfg = []

    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_items = [
        ("Generated", generated),
        ("Analysis Date", analysis_date),
        ("Provider", provider),
        ("Deep Thinker", deep),
        ("Quick Thinker", quick),
        ("Research Depth", f"{depth_label} ({depth_rounds} rounds)" if depth_rounds else depth_label),
        ("Trade Date", trade_date),
        ("Analysts", analysts_str),
        ("Thinking", thinking_status),
        ("Backend URL", backend_url),
        ("Google Thinking Level", google_thinking),
        ("OpenAI Reasoning Effort", openai_effort),
        ("Anthropic Effort", anthropic_effort),
        ("Output Language", output_language),
    ]
    meta_html = '<div class="meta">' + "".join(
        f'<div class="meta-item">'
        f'<span class="meta-label">{k}:</span>'
        f'<span class="meta-value">{v}</span>'
        f"</div>"
        for k, v in meta_items
        if v
    ) + "</div>"

    # Timing table
    timing_html = ""
    if timing:
        total = timing.get("_total_seconds", 0)
        hh, rem = divmod(int(total), 3600)
        mm, ss = divmod(rem, 60)
        total_str = f"{hh:02d}:{mm:02d}:{ss:02d}"
        total_llm = timing.get("_total_llm_seconds", 0)
        has_llm = total_llm > 0

        _local = provider.lower() in ("ollama", "lmstudio")
        _ANALYST_NAMES = {"market", "social", "news", "fundamentals"}
        _KEY_TO_AGENT = {
            "Analyst Phase": None,
            "Bull Researcher": "bull",
            "Bear Researcher": "bear",
            "Research Manager": "research_manager",
            "Trader": "trader",
            "Aggressive Analyst": "aggressive",
            "Conservative Analyst": "conservative",
            "Neutral Analyst": "neutral",
            "Portfolio Manager": "portfolio_manager",
            "Summary": "summary",
        }

        def _thinking_html(key):
            if not provider:
                return "—"
            if not _local:
                return "ON"
            agent_name = _KEY_TO_AGENT.get(key)
            if agent_name is None:
                return "ON" if _ANALYST_NAMES & set(thinking_agents_cfg) else "OFF"
            return "ON" if agent_name in thinking_agents_cfg else "OFF"

        rows = []
        for key in _AGENT_KEYS:
            if key not in timing:
                continue
            secs = timing[key]
            m, s = divmod(int(secs), 60)
            pct = (secs / total * 100) if total else 0
            think_cell = _thinking_html(key)
            if has_llm:
                llm_secs = timing.get(f"llm_{key}")
                if llm_secs:
                    ls = max(1, round(llm_secs))
                    llm_cell = f"{ls // 60:02d}:{ls % 60:02d}"
                else:
                    llm_cell = "—"
                rows.append(f"<tr><td>{key}</td><td>{m:02d}:{s:02d}</td><td>{pct:.1f}%</td><td>{llm_cell}</td><td>{think_cell}</td></tr>")
            else:
                rows.append(f"<tr><td>{key}</td><td>{m:02d}:{s:02d}</td><td>{pct:.1f}%</td><td>{think_cell}</td></tr>")

        if has_llm:
            lh, lr = divmod(int(total_llm), 3600)
            lm, ls = divmod(lr, 60)
            llm_str = f"{lh:02d}:{lm:02d}:{ls:02d}"
            summary = f"Total elapsed: <strong>{total_str}</strong> &nbsp;|&nbsp; LLM generation: <strong>{llm_str}</strong>"
            head_row = "<tr><th>Agent</th><th>Wall Clock</th><th>% of Total</th><th>LLM Time</th><th>Thinking</th></tr>"
        else:
            summary = f"Total elapsed: <strong>{total_str}</strong>"
            head_row = "<tr><th>Agent</th><th>Duration</th><th>% of Total</th><th>Thinking</th></tr>"

        timing_html = (
            "<h2>Run Timing</h2>"
            f"<p style='margin-bottom:10px'>{summary}</p>"
            f"<table>{head_row}{''.join(rows)}</table>"
        )

    # Content sections
    body = []

    if final_state.get("action_summary"):
        body.append(
            "<h2>Action Summary</h2>"
            f'<div class="section-card">{_to_html(final_state["action_summary"])}</div>'
        )

    analyst_cards = []
    for name, key in [
        ("Market Analyst", "market_report"),
        ("Social Analyst", "sentiment_report"),
        ("News Analyst", "news_report"),
        ("Fundamentals Analyst", "fundamentals_report"),
    ]:
        text = final_state.get(key)
        if not text:
            continue
        # inject charts into market analyst only; charts dict is consumed per-card
        embed = charts if key == "market_report" else None
        analyst_cards.append(_card(name, text, charts=embed, ticker=ticker))
    if analyst_cards:
        body.append("<h2>I. Analyst Team Reports</h2>" + "".join(analyst_cards))

    debate = final_state.get("investment_debate_state", {})
    research_parts = [
        (name, debate.get(key))
        for name, key in [
            ("Bull Researcher", "bull_history"),
            ("Bear Researcher", "bear_history"),
            ("Research Manager", "judge_decision"),
        ]
        if debate.get(key)
    ]
    if research_parts:
        body.append(
            "<h2>II. Research Team Decision</h2>"
            + "".join(_card(n, t) for n, t in research_parts)
        )

    if final_state.get("trader_investment_plan"):
        body.append(
            "<h2>III. Trading Team Plan</h2>"
            + _card("Trader", final_state["trader_investment_plan"])
        )

    risk = final_state.get("risk_debate_state", {})
    risk_parts = [
        (name, risk.get(key))
        for name, key in [
            ("Aggressive Analyst", "aggressive_history"),
            ("Conservative Analyst", "conservative_history"),
            ("Neutral Analyst", "neutral_history"),
        ]
        if risk.get(key)
    ]
    if risk_parts:
        body.append(
            "<h2>IV. Risk Management Team Decision</h2>"
            + "".join(_card(n, t) for n, t in risk_parts)
        )

    if risk.get("judge_decision"):
        body.append(
            "<h2>V. Portfolio Manager Decision</h2>"
            + _card("Portfolio Manager", risk["judge_decision"])
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Analysis: {ticker}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
<h1>Trading Analysis Report: {ticker}</h1>
{meta_html}
{timing_html}
{"".join(body)}
</div>
</body>
</html>"""

    out = save_path / f"{save_path.name}_complete_report.html"
    out.write_text(html, encoding="utf-8")
    return out
