"""
dashboard.py - Streamlit web dashboard for the crypto trading system.

Run:
    streamlit run dashboard.py

Opens in browser at http://localhost:8501
Auto-refreshes every 30 seconds.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Crypto Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Auto-refresh every 30 seconds
st.markdown(
    '<meta http-equiv="refresh" content="30">',
    unsafe_allow_html=True,
)

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=15)
def load_portfolio():
    path = "data/crypto_portfolio.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=15)
def load_jsonl(path, n=500):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows[-n:]


@st.cache_data(ttl=15)
def load_signal_weights():
    path = "data/signal_weights.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    # File is {"weights": {...}, "reason": ..., "updated_at": ...}
    if "weights" in d:
        return d["weights"]
    return d


@st.cache_data(ttl=15)
def load_optimizer_history():
    return load_jsonl("data/optimizer_history.jsonl", 20)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pnl_color(val):
    if val is None:
        return "gray"
    return "green" if val >= 0 else "red"


def _fmt_pct(val, with_sign=True):
    if val is None:
        return "N/A"
    s = f"{val:+.2f}%" if with_sign else f"{val:.2f}%"
    return s


# ── Header ────────────────────────────────────────────────────────────────────

st.title("📈 Crypto Trading Dashboard")
now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
st.caption(f"Live paper trading  •  Last loaded: {now_str}  •  Auto-refreshes every 30s")

with st.expander("Simulation assumptions (click to verify accuracy)"):
    st.markdown("""
**What is real:**
- Price data fetched live from **Binance public API** (same candles as live trading)
- All fills use the **actual Binance close price** of the 1-minute bar

**What is simulated (conservatively):**
- **Slippage:** buys fill at close + 0.05%, sells fill at close - 0.05% (configurable via `slippage_pct`)
- **Trading fee:** 0.1% per trade deducted from proceeds (Binance taker rate, configurable via `fee_pct`)
- **Benchmark:** orange dotted line = what $10k in BTC alone would be worth (buy & hold from session start)

**Known limitations vs live trading:**
- No order book depth — real large orders move the market more than 0.05%
- 1-min bar close is used; a real order placed mid-bar would get a different price
- No funding rates (relevant for perpetual futures, not spot)

**What "Alpha" means:** Strategy return minus BTC benchmark. Positive = beating buy-and-hold.
    """)
st.divider()

# ── Portfolio summary ─────────────────────────────────────────────────────────

portfolio = load_portfolio()
equity_rows = load_jsonl("data/crypto_equity.jsonl", 1440)  # last 24h

if portfolio:
    cash       = portfolio.get("cash", 0)
    equity     = portfolio.get("equity", cash)
    positions  = portfolio.get("positions", {})
    updated    = portfolio.get("updated_at", "")[:19].replace("T", " ")

    # Compute total P&L and benchmark comparison
    initial_equity = 10_000.0
    if equity_rows:
        initial_equity = equity_rows[0].get("equity", 10_000.0)
    total_pnl_pct = (equity / initial_equity - 1) * 100 if initial_equity else 0

    # BTC buy-and-hold benchmark from equity log
    btc_bench_pct = None
    if equity_rows:
        last_bench = next((r.get("btc_benchmark") for r in reversed(equity_rows)
                           if r.get("btc_benchmark")), None)
        if last_bench:
            btc_bench_pct = (last_bench / initial_equity - 1) * 100
    alpha = (total_pnl_pct - btc_bench_pct) if btc_bench_pct is not None else None

    # Total fees paid
    trades_all = load_jsonl("data/crypto_trades.jsonl", 5000)
    total_fees = sum(t.get("fee", 0) or 0 for t in trades_all)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Equity",   f"${equity:,.2f}",
                delta=f"{total_pnl_pct:+.2f}% from start")
    col2.metric("BTC Benchmark",
                f"{btc_bench_pct:+.2f}%" if btc_bench_pct is not None else "N/A",
                delta=f"Alpha: {alpha:+.2f}%" if alpha is not None else None)
    col3.metric("Cash",           f"${cash:,.2f}")
    col4.metric("Invested",       f"${equity - cash:,.2f}")
    col5.metric("Open Positions", len(positions))
    col6.metric("Fees Paid",      f"${total_fees:.2f}", delta="real drag")
else:
    st.warning("No portfolio data yet. Start the trader with: `python daily_runner.py`")
    st.stop()

st.divider()

# ── Equity curve ──────────────────────────────────────────────────────────────

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Equity Curve")

    if len(equity_rows) >= 2:
        df_eq = pd.DataFrame(equity_rows)
        df_eq["ts"] = pd.to_datetime(df_eq["ts"])
        df_eq = df_eq.sort_values("ts")

        # Resample to 5-min buckets to avoid too many points
        df_eq = df_eq.set_index("ts").resample("5min").last().dropna().reset_index()

        start_eq = df_eq["equity"].iloc[0]
        color     = "green" if df_eq["equity"].iloc[-1] >= start_eq else "red"

        fig = go.Figure()
        # Strategy equity
        fig.add_trace(go.Scatter(
            x=df_eq["ts"], y=df_eq["equity"],
            mode="lines", name="Strategy",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=f"rgba({'0,180,0' if color == 'green' else '200,0,0'},0.07)",
        ))
        # BTC buy-and-hold benchmark
        if "btc_benchmark" in df_eq.columns and df_eq["btc_benchmark"].notna().any():
            fig.add_trace(go.Scatter(
                x=df_eq["ts"], y=df_eq["btc_benchmark"],
                mode="lines", name="BTC Buy&Hold",
                line=dict(color="orange", width=1.5, dash="dot"),
            ))
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title=None,
            yaxis_title="USD",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.05),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Equity history building up — check back after a few minutes.")


# ── Open positions ─────────────────────────────────────────────────────────────

with col_right:
    st.subheader("Open Positions")

    if positions:
        rows = []
        for sym, pos in positions.items():
            entry  = pos.get("entry_price", 0)
            last   = pos.get("last_price", entry)
            qty    = pos.get("qty", 0)
            cost   = qty * entry
            value  = qty * last
            pnl    = value - cost
            pnl_pct = (last / entry - 1) * 100 if entry else 0
            hold_m  = 0
            try:
                et = datetime.fromisoformat(pos.get("entry_time", "")).replace(tzinfo=timezone.utc)
                hold_m = int((datetime.now(timezone.utc) - et).total_seconds() / 60)
            except Exception:
                pass
            rows.append({
                "Symbol":    sym,
                "Entry $":   f"${entry:,.4f}",
                "Current $": f"${last:,.4f}",
                "P&L":       f"{pnl_pct:+.2f}%",
                "Value":     f"${value:,.2f}",
                "Hold":      f"{hold_m}m",
            })
        df_pos = pd.DataFrame(rows)

        # Color P&L column
        def color_pnl(val):
            color = "color: green" if val.startswith("+") else "color: red"
            return color

        st.dataframe(
            df_pos.style.map(color_pnl, subset=["P&L"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No open positions right now.")

st.divider()

# ── Performance stats ─────────────────────────────────────────────────────────

st.subheader("Performance Stats")

trades = load_jsonl("data/crypto_trades.jsonl", 1000)
sells  = [t for t in trades if t.get("action") == "SELL" and t.get("pnl_pct") is not None]

col1, col2, col3, col4, col5 = st.columns(5)

if sells:
    pnls      = [t["pnl_pct"] for t in sells]
    wins      = [p for p in pnls if p >= 0]
    losses    = [p for p in pnls if p < 0]
    win_rate  = len(wins) / len(pnls) * 100
    avg_win   = sum(wins) / len(wins) if wins else 0
    avg_loss  = sum(losses) / len(losses) if losses else 0
    total_pnl = sum(t.get("pnl", 0) for t in sells)

    col1.metric("Closed Trades",  len(sells))
    col2.metric("Win Rate",        f"{win_rate:.1f}%")
    col3.metric("Avg Win",         f"{avg_win:+.2f}%")
    col4.metric("Avg Loss",        f"{avg_loss:+.2f}%")
    col5.metric("Total Realised",  f"${total_pnl:+.2f}")
else:
    col1.metric("Closed Trades", 0)
    col2.info("No closed trades yet")

# ── Recent trades table ────────────────────────────────────────────────────────

st.subheader("Recent Trades")

if trades:
    recent = list(reversed(trades[-30:]))
    rows = []
    for t in recent:
        action  = t.get("action", "")
        pnl_pct = t.get("pnl_pct")
        pnl_str = f"{pnl_pct:+.2f}%" if pnl_pct is not None else "-"
        rows.append({
            "Time":    t.get("ts", "")[:19].replace("T", " "),
            "Action":  action,
            "Symbol":  t.get("symbol", ""),
            "Price":   f"${t.get('price', 0):,.4f}",
            "Value":   f"${t.get('value', 0):,.2f}",
            "P&L":     pnl_str,
            "Reason":  t.get("reason", ""),
        })
    df_trades = pd.DataFrame(rows)

    def color_action(val):
        return "color: green; font-weight: bold" if val == "BUY" else "color: red; font-weight: bold"

    def color_pnl2(val):
        if val == "-": return ""
        return "color: green" if val.startswith("+") else "color: red"

    st.dataframe(
        df_trades.style
            .map(color_action, subset=["Action"])
            .map(color_pnl2, subset=["P&L"]),
        use_container_width=True,
        hide_index=True,
        height=300,
    )
else:
    st.info("No trades recorded yet.")

st.divider()

# ── P&L distribution ──────────────────────────────────────────────────────────

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("P&L Distribution")
    if len(sells) >= 3:
        pnls = [t["pnl_pct"] for t in sells]
        fig = px.histogram(
            x=pnls, nbins=20,
            color_discrete_sequence=["steelblue"],
            labels={"x": "Trade P&L (%)"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need more closed trades for distribution chart.")


# ── Signal weights ────────────────────────────────────────────────────────────

with col_b:
    st.subheader("Current Signal Weights")

    weights = load_signal_weights()
    if not weights:
        # Fall back to default weights from alpha_lab
        try:
            from alpha_lab import DEFAULT_WEIGHTS
            weights = DEFAULT_WEIGHTS
            st.caption("Showing default weights (optimizer hasn't run yet)")
        except Exception:
            pass

    if weights:
        df_w = pd.DataFrame(
            sorted(weights.items(), key=lambda x: -x[1]),
            columns=["Signal", "Weight"],
        )
        fig = px.bar(
            df_w, x="Weight", y="Signal",
            orientation="h",
            color="Weight",
            color_continuous_scale=["#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560"],
        )
        fig.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No signal weights yet.")

st.divider()

# ── Optimizer history ─────────────────────────────────────────────────────────

st.subheader("Optimizer Runs")

opt_history = load_optimizer_history()
if opt_history:
    rows = []
    for r in reversed(opt_history):
        rows.append({
            "Time":       r.get("ts", "")[:19].replace("T", " "),
            "Resolved":   r.get("n_resolved", 0),
            "Reason":     (r.get("reason") or r.get("commentary", ""))[:80],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("Optimizer hasn't run yet. First run is 1 hour after startup.")

# ── Footer ────────────────────────────────────────────────────────────────────

st.caption(
    "Paper trading only — no real money at risk.  "
    "Trader: `python daily_runner.py`  |  "
    "Dashboard: `streamlit run dashboard.py`"
)
