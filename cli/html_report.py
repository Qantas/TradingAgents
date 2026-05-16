import datetime
import json
import re
from pathlib import Path
from typing import Dict, Optional

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

_LC_CDN = "https://unpkg.com/lightweight-charts@4/dist/lightweight-charts.standalone.production.js"

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
.section-card {
  background: #181825; border: 1px solid #313244; border-radius: 8px;
  padding: 18px 22px; margin: 14px 0;
}
.section-card h1, .section-card h2 { color: #cba6f7; font-size: 1.05rem; margin-top: 12px; border: none; }
.section-card h3 { color: #a6e3a1; }
.section-card h4 { color: #f9e2af; }
.agent-label { font-size: 0.78rem; color: #6c7086; text-transform: uppercase;
               letter-spacing: 0.1em; margin-bottom: 4px; }
.dur-btns { display: flex; gap: 6px; margin: 12px 0 8px; flex-wrap: wrap; }
.dur-btn {
  background: #313244; border: 1px solid #45475a; color: #cdd6f4;
  padding: 3px 11px; border-radius: 4px; cursor: pointer;
  font-size: 0.78rem; font-family: inherit; transition: background 0.15s;
}
.dur-btn:hover { background: #45475a; }
.dur-btn.active { background: #89b4fa; color: #1e1e2e; border-color: #89b4fa; font-weight: 600; }
.lc-pane-label {
  font-size: 0.72rem; color: #6c7086; text-transform: uppercase;
  letter-spacing: 0.08em; margin: 10px 0 2px;
}
.lc-pane { width: 100%; border-radius: 6px; overflow: hidden; }
"""


def _prepare_chart_data(ticker: str, trade_date: str) -> dict:
    """Fetch 2 years of OHLCV and compute all indicators. Returns JSON-serialisable dict."""
    try:
        import numpy as np
        import pandas as pd
        import yfinance as yf

        end = pd.Timestamp(trade_date)
        df = yf.download(
            ticker,
            start=(end - pd.DateOffset(years=2)).strftime("%Y-%m-%d"),
            end=(end + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            return {}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close  = df["Close"]
        volume = df["Volume"]

        ema10    = close.ewm(span=10, adjust=False).mean()
        sma50    = close.rolling(50).mean()
        sma200   = close.rolling(200).mean()
        vwma20   = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
        sma20    = close.rolling(20).mean()
        bb_upper = sma20 + 2 * close.rolling(20).std()
        bb_lower = sma20 - 2 * close.rolling(20).std()

        ema12    = close.ewm(span=12, adjust=False).mean()
        ema26    = close.ewm(span=26, adjust=False).mean()
        macd_l   = ema12 - ema26
        macd_sig = macd_l.ewm(span=9, adjust=False).mean()
        macd_h   = macd_l - macd_sig

        delta    = close.diff()
        avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rs       = avg_gain / avg_loss.where(avg_loss != 0, np.nan)
        rsi      = (100 - 100 / (1 + rs)).fillna(50)

        prev_c = close.shift(1)
        tr     = pd.concat([(df["High"] - df["Low"]),
                             (df["High"] - prev_c).abs(),
                             (df["Low"]  - prev_c).abs()], axis=1).max(axis=1)
        atr    = tr.ewm(com=13, adjust=False).mean()

        def to_list(s):
            return [{"time": d.strftime("%Y-%m-%d"), "value": round(float(v), 4)}
                    for d, v in s.dropna().items()]

        ohlcv = [
            {
                "time":   row.Index.strftime("%Y-%m-%d"),
                "open":   round(float(row.Open),   4),
                "high":   round(float(row.High),   4),
                "low":    round(float(row.Low),    4),
                "close":  round(float(row.Close),  4),
                "volume": int(row.Volume),
            }
            for row in df.itertuples()
        ]

        return {
            "ohlcv":       ohlcv,
            "ema10":       to_list(ema10),
            "sma50":       to_list(sma50),
            "sma200":      to_list(sma200),
            "vwma20":      to_list(vwma20),
            "bb_upper":    to_list(bb_upper),
            "bb_lower":    to_list(bb_lower),
            "macd":        to_list(macd_l),
            "macd_signal": to_list(macd_sig),
            "macd_hist":   to_list(macd_h),
            "rsi":         to_list(rsi),
            "atr":         to_list(atr),
        }
    except Exception:
        return {}


def _interactive_charts_html(data: dict, ticker: str) -> str:
    """Return HTML+JS for Lightweight Charts interactive panes."""
    if not data:
        return ""

    cid = re.sub(r"[^A-Za-z0-9]", "_", ticker)
    data_json = json.dumps(data, separators=(",", ":"))

    btns = "".join(
        f'<button class="dur-btn" data-cid="{cid}" data-days="{days}"'
        f' onclick="taSetRange(\'{cid}\',{days})">{label}</button>'
        for label, days in [("5D", 5), ("1M", 30), ("3M", 90), ("6M", 180), ("1Y", 365), ("2Y", 730)]
    )

    panes = "".join(
        f'<div class="lc-pane-label">{title}</div>'
        f'<div id="lc-{cid}-{key}" class="lc-pane" style="height:{h}px"></div>'
        for key, title, h in [
            ("candle", "Candlestick · EMA10 · SMA50 · SMA200 · BB · VWMA20", 300),
            ("vol",    "Volume",                                               100),
            ("macd",   "MACD (12, 26, 9)",                                    110),
            ("rsi",    "RSI (14)",                                             100),
            ("atr",    "Average True Range (14)",                              80),
        ]
    )

    js = f"""
<script>
(function(){{
  window.__taData  = window.__taData  || {{}};
  window.__taCharts= window.__taCharts|| {{}};
  window.__taData['{cid}'] = {data_json};

  function init(){{
    if(typeof LightweightCharts==='undefined'){{ setTimeout(init,80); return; }}
    var d  = window.__taData['{cid}'];
    var LC = LightweightCharts;
    var BG='#1a1a2e',GRID='#1e1e2e',TXT='#cdd6f4',BDR='#333344';

    function mkChart(id,h){{
      var el=document.getElementById(id); if(!el) return null;
      return LC.createChart(el,{{
        width:el.clientWidth, height:h,
        layout:{{background:{{color:BG}},textColor:TXT}},
        grid:{{vertLines:{{color:GRID}},horzLines:{{color:GRID}}}},
        rightPriceScale:{{borderColor:BDR}},
        timeScale:{{borderColor:BDR,timeVisible:false}},
        crosshair:{{mode:LC.CrosshairMode.Normal}},
        handleScale:true, handleScroll:true,
      }});
    }}

    var c1=mkChart('lc-{cid}-candle',300);
    var c2=mkChart('lc-{cid}-vol',   100);
    var c3=mkChart('lc-{cid}-macd',  110);
    var c4=mkChart('lc-{cid}-rsi',   100);
    var c5=mkChart('lc-{cid}-atr',    80);
    var charts=[c1,c2,c3,c4,c5].filter(Boolean);
    window.__taCharts['{cid}']=charts;

    // ── Candle pane ──────────────────────────────────────────────────────
    var cs=c1.addCandlestickSeries({{
      upColor:'#a6e3a1',downColor:'#f38ba8',
      borderUpColor:'#a6e3a1',borderDownColor:'#f38ba8',
      wickUpColor:'#a6e3a1',wickDownColor:'#f38ba8',
    }});
    cs.setData(d.ohlcv);
    function line(chart,color,data,title){{
      chart.addLineSeries({{color:color,lineWidth:1,title:title,lastValueVisible:false,priceLineVisible:false}}).setData(data);
    }}
    line(c1,'#89b4fa',d.ema10,  'EMA10');
    line(c1,'#a6e3a1',d.sma50,  'SMA50');
    line(c1,'#f38ba8',d.sma200, 'SMA200');
    line(c1,'#fab387',d.vwma20, 'VWMA20');
    line(c1,'#cba6f7',d.bb_upper,'BB+');
    line(c1,'#cba6f7',d.bb_lower,'BB-');

    // ── Volume pane ──────────────────────────────────────────────────────
    if(c2){{
      var vs=c2.addHistogramSeries({{priceFormat:{{type:'volume'}}}});
      vs.setData(d.ohlcv.map(function(x){{
        return {{time:x.time,value:x.volume,color:x.close>=x.open?'#a6e3a180':'#f38ba880'}};
      }}));
    }}

    // ── MACD pane ────────────────────────────────────────────────────────
    if(c3){{
      var mh=c3.addHistogramSeries({{}});
      mh.setData(d.macd_hist.map(function(x){{
        return {{time:x.time,value:x.value,color:x.value>=0?'#a6e3a1':'#f38ba8'}};
      }}));
      line(c3,'#89b4fa',d.macd,       'MACD');
      line(c3,'#f9e2af',d.macd_signal,'Signal');
    }}

    // ── RSI pane ─────────────────────────────────────────────────────────
    if(c4){{
      line(c4,'#89dceb',d.rsi,'RSI');
      c4.priceScale('right').applyOptions({{autoScale:false,minimum:0,maximum:100}});
    }}

    // ── ATR pane ─────────────────────────────────────────────────────────
    if(c5) line(c5,'#cba6f7',d.atr,'ATR');

    // ── Sync all panes ───────────────────────────────────────────────────
    var syncing=false;
    charts.forEach(function(src){{
      src.timeScale().subscribeVisibleLogicalRangeChange(function(range){{
        if(syncing||!range) return;
        syncing=true;
        charts.forEach(function(c){{if(c!==src) c.timeScale().setVisibleLogicalRange(range);}});
        syncing=false;
      }});
    }});

    setTimeout(function(){{taSetRange('{cid}',90);}},100);

    window.addEventListener('resize',function(){{
      var ids=['lc-{cid}-candle','lc-{cid}-vol','lc-{cid}-macd','lc-{cid}-rsi','lc-{cid}-atr'];
      charts.forEach(function(c,i){{
        var el=document.getElementById(ids[i]);
        if(el) c.applyOptions({{width:el.clientWidth}});
      }});
    }});
  }}

  init();
}})();

window.taSetRange = window.taSetRange || function(cid,days){{
  var d=(window.__taData||{{}})[cid];
  if(!d||!d.ohlcv.length) return;
  var last=d.ohlcv[d.ohlcv.length-1].time;
  var from=new Date(last+'T00:00:00Z');
  from.setUTCDate(from.getUTCDate()-days);
  var fromStr=from.toISOString().slice(0,10);
  ((window.__taCharts||{{}})[cid]||[]).forEach(function(c){{
    c.timeScale().setVisibleRange({{from:fromStr,to:last}});
  }});
  document.querySelectorAll('.dur-btn[data-cid="'+cid+'"]').forEach(function(b){{
    b.classList.toggle('active', parseInt(b.dataset.days)===days);
  }});
}};
</script>
"""

    return f'<div class="dur-btns">{btns}</div>{panes}{js}'


def _to_html(text: str) -> str:
    if _MARKDOWN:
        return _md_lib.markdown(text, extensions=["tables", "fenced_code"])
    import html
    return f"<pre>{html.escape(text)}</pre>"


def _card(label: str, content: str, charts_html: str = "") -> str:
    return (
        f'<div class="section-card">'
        f'<p class="agent-label">{label}</p>'
        f"{charts_html}"
        f"{_to_html(content)}"
        f"</div>"
    )


def save_html_report(
    final_state: dict,
    ticker: str,
    save_path,
    timing: dict = None,
    meta: dict = None,
) -> "Path":
    save_path = Path(save_path)
    trade_date = final_state.get("trade_date", "")

    chart_data = _prepare_chart_data(ticker, trade_date)
    charts_html = _interactive_charts_html(chart_data, ticker)

    # Metadata block
    if meta:
        depth_rounds = meta.get("research_depth", "")
        depth_label  = meta.get("research_depth_label", "")
        provider     = meta.get("llm_provider", "")
        deep         = meta.get("deep_thinker", "")
        quick        = meta.get("shallow_thinker", "")
        analysis_date    = meta.get("analysis_date", "")
        analysts_list    = meta.get("analysts", [])
        analysts_str     = ", ".join(
            a.value if hasattr(a, "value") else str(a) for a in analysts_list
        ) if analysts_list else "all"
        backend_url      = meta.get("backend_url", "")
        output_language  = meta.get("output_language", "")
        google_thinking  = meta.get("google_thinking_level", "")
        openai_effort    = meta.get("openai_reasoning_effort", "")
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

    generated  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_items = [
        ("Generated",             generated),
        ("Analysis Date",         analysis_date),
        ("Provider",              provider),
        ("Deep Thinker",          deep),
        ("Quick Thinker",         quick),
        ("Research Depth",        f"{depth_label} ({depth_rounds} rounds)" if depth_rounds else depth_label),
        ("Trade Date",            trade_date),
        ("Analysts",              analysts_str),
        ("Thinking",              thinking_status),
        ("Backend URL",           backend_url),
        ("Google Thinking Level", google_thinking),
        ("OpenAI Reasoning Effort", openai_effort),
        ("Anthropic Effort",      anthropic_effort),
        ("Output Language",       output_language),
    ]
    meta_html = '<div class="meta">' + "".join(
        f'<div class="meta-item">'
        f'<span class="meta-label">{k}:</span>'
        f'<span class="meta-value">{v}</span>'
        f"</div>"
        for k, v in meta_items if v
    ) + "</div>"

    # Timing table
    timing_html = ""
    if timing:
        total = timing.get("_total_seconds", 0)
        hh, rem = divmod(int(total), 3600)
        mm, ss  = divmod(rem, 60)
        total_str = f"{hh:02d}:{mm:02d}:{ss:02d}"
        total_llm = timing.get("_total_llm_seconds", 0)
        has_llm   = total_llm > 0

        _local = provider.lower() in ("ollama", "lmstudio")
        _ANALYST_NAMES = {"market", "social", "news", "fundamentals"}
        _KEY_TO_AGENT  = {
            "Analyst Phase":      None,
            "Bull Researcher":    "bull",
            "Bear Researcher":    "bear",
            "Research Manager":   "research_manager",
            "Trader":             "trader",
            "Aggressive Analyst": "aggressive",
            "Conservative Analyst": "conservative",
            "Neutral Analyst":    "neutral",
            "Portfolio Manager":  "portfolio_manager",
            "Summary":            "summary",
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
            pct  = (secs / total * 100) if total else 0
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
            llm_str  = f"{lh:02d}:{lm:02d}:{ls:02d}"
            summary  = f"Total elapsed: <strong>{total_str}</strong> &nbsp;|&nbsp; LLM generation: <strong>{llm_str}</strong>"
            head_row = "<tr><th>Agent</th><th>Wall Clock</th><th>% of Total</th><th>LLM Time</th><th>Thinking</th></tr>"
        else:
            summary  = f"Total elapsed: <strong>{total_str}</strong>"
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
        ("Market Analyst",       "market_report"),
        ("Social Analyst",       "sentiment_report"),
        ("News Analyst",         "news_report"),
        ("Fundamentals Analyst", "fundamentals_report"),
    ]:
        text = final_state.get(key)
        if not text:
            continue
        embed = charts_html if key == "market_report" else ""
        analyst_cards.append(_card(name, text, charts_html=embed))
    if analyst_cards:
        body.append("<h2>I. Analyst Team Reports</h2>" + "".join(analyst_cards))

    debate = final_state.get("investment_debate_state", {})
    research_parts = [
        (name, debate.get(k))
        for name, k in [
            ("Bull Researcher",  "bull_history"),
            ("Bear Researcher",  "bear_history"),
            ("Research Manager", "judge_decision"),
        ]
        if debate.get(k)
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
        (name, risk.get(k))
        for name, k in [
            ("Aggressive Analyst",   "aggressive_history"),
            ("Conservative Analyst", "conservative_history"),
            ("Neutral Analyst",      "neutral_history"),
        ]
        if risk.get(k)
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

    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Analysis: {ticker}</title>
<style>{_CSS}</style>
<script src="{_LC_CDN}"></script>
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
    out.write_text(html_out, encoding="utf-8")
    return out
