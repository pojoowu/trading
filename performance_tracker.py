"""
Performance tracker: records every trade decision, signal values, and
portfolio P&L after each daily run so the optimizer can learn from history.

Stored in data/performance_history.jsonl  (one JSON record per day)
Stored in data/strategy_params.json       (current live strategy parameters)
"""
import json
import logging
import os
from datetime import datetime, date
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HISTORY_FILE  = "data/performance_history.jsonl"
PARAMS_FILE   = "data/strategy_params.json"
SIGNAL_LOG    = "data/signal_log.jsonl"   # per-stock signal accuracy log


# ── Default strategy parameters (overridden by optimizer) ────────────────────

DEFAULT_PARAMS = {
    # Screener
    "min_market_cap_b":        5.0,
    "min_avg_volume":          500_000,
    "max_stocks_after_screen": 20,

    # Alpha signal weights (must sum to 1.0)
    "signal_weights": {
        "cross_momentum":  0.30,
        "sharpe_momentum": 0.20,
        "trend_following": 0.15,
        "analyst_upside":  0.15,
        "value_quality":   0.10,
        "volume_surge":    0.05,
        "short_reversal":  0.05,
    },

    # Technical thresholds
    "rsi_overbought":    75,
    "rsi_oversold":      30,
    "momentum_lookback": 252,
    "short_momentum":    21,

    # Portfolio
    "max_positions":      10,
    "max_position_pct":   0.15,
    "cash_reserve":       0.05,
    "stop_loss_pct":      0.07,
    "take_profit_pct":    0.20,

    # Backtest acceptance thresholds (stocks below these are rejected)
    "min_sharpe":         0.5,
    "max_drawdown_floor": -0.35,   # -35%

    # Meta
    "version":    1,
    "updated_at": "",
    "update_reason": "initial defaults",
}


def load_params() -> dict:
    """Load current strategy parameters (or return defaults if none saved)."""
    if os.path.exists(PARAMS_FILE):
        try:
            with open(PARAMS_FILE) as f:
                saved = json.load(f)
            # Merge with defaults so new keys are always present
            merged = {**DEFAULT_PARAMS, **saved}
            merged["signal_weights"] = {
                **DEFAULT_PARAMS["signal_weights"],
                **saved.get("signal_weights", {}),
            }
            return merged
        except Exception as exc:
            logger.warning("Could not load params (%s). Using defaults.", exc)
    return dict(DEFAULT_PARAMS)


def save_params(params: dict) -> None:
    """Persist updated strategy parameters."""
    os.makedirs("data", exist_ok=True)
    params["updated_at"] = datetime.utcnow().isoformat()
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)
    logger.info("Strategy params saved (version %s)", params.get("version"))


# ── Daily performance record ──────────────────────────────────────────────────

def record_daily_run(
    date_str: str,
    portfolio_equity: float,
    cash: float,
    positions: dict,          # {ticker: {shares, avg_cost, last_price, pnl_pct}}
    trades_executed: list,    # list of {action, ticker, shares, price}
    agent_picks: list[str],   # final tickers the agent chose
    signal_scores: dict,      # {ticker: {alpha_score, cross_momentum, …}}
    params_version: int,
    notes: str = "",
) -> None:
    """Append a daily performance record to the history file."""
    os.makedirs("data", exist_ok=True)
    record = {
        "date":             date_str,
        "portfolio_equity": round(portfolio_equity, 2),
        "cash":             round(cash, 2),
        "positions":        positions,
        "trades":           trades_executed,
        "agent_picks":      agent_picks,
        "signal_scores":    signal_scores,
        "params_version":   params_version,
        "notes":            notes,
        "recorded_at":      datetime.utcnow().isoformat(),
    }
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def record_signal_outcome(
    ticker: str,
    signal_date: str,
    signals: dict,         # signal values at decision time
    forward_return_1w: Optional[float] = None,
    forward_return_1m: Optional[float] = None,
) -> None:
    """
    Log signal values alongside actual forward returns.
    Used by the optimizer to measure which signals predicted returns best.
    """
    os.makedirs("data", exist_ok=True)
    record = {
        "ticker":              ticker,
        "signal_date":         signal_date,
        "signals":             signals,
        "forward_return_1w":   forward_return_1w,
        "forward_return_1m":   forward_return_1m,
        "logged_at":           datetime.utcnow().isoformat(),
    }
    with open(SIGNAL_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── History loading ───────────────────────────────────────────────────────────

def load_history(n_days: int = 90) -> list[dict]:
    """Load the last n_days of daily performance records."""
    if not os.path.exists(HISTORY_FILE):
        return []
    records = []
    with open(HISTORY_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records[-n_days:]


def load_signal_log(n_records: int = 500) -> list[dict]:
    """Load signal outcome records for signal accuracy analysis."""
    if not os.path.exists(SIGNAL_LOG):
        return []
    records = []
    with open(SIGNAL_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records[-n_records:]


# ── Equity curve metrics ──────────────────────────────────────────────────────

def compute_recent_metrics(history: list[dict]) -> dict:
    """
    Compute rolling performance metrics from history records.
    Returns dict with CAGR, Sharpe, max drawdown, win rate, etc.
    """
    if len(history) < 5:
        return {"error": "insufficient history", "n_days": len(history)}

    equities = [r["portfolio_equity"] for r in history if "portfolio_equity" in r]
    if len(equities) < 2:
        return {"error": "no equity data"}

    eq = pd.Series(equities)
    rets = eq.pct_change().dropna()

    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    n_years   = len(eq) / 252
    cagr      = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    sharpe    = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    peak      = eq.expanding().max()
    max_dd    = ((eq - peak) / peak).min()

    # Trade win rate from history
    all_trades = []
    for r in history:
        all_trades.extend(r.get("trades", []))
    wins = sum(1 for t in all_trades if t.get("pnl", 0) > 0)
    win_rate = wins / len(all_trades) if all_trades else None

    return {
        "n_days":         len(history),
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct":       round(cagr * 100, 2),
        "sharpe":         round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_pct":   round(win_rate * 100, 2) if win_rate is not None else None,
        "start_equity":   round(eq.iloc[0], 2),
        "end_equity":     round(eq.iloc[-1], 2),
        "latest_date":    history[-1].get("date", ""),
    }


# ── Signal accuracy analysis ──────────────────────────────────────────────────

def compute_signal_accuracy(signal_log: list[dict]) -> dict:
    """
    For each signal, compute its correlation with forward 1-month returns.
    Higher correlation = more predictive signal.
    """
    if not signal_log:
        return {}

    records = [r for r in signal_log if r.get("forward_return_1m") is not None]
    if len(records) < 10:
        return {"note": "insufficient signal outcomes to analyse"}

    signal_names = list(DEFAULT_PARAMS["signal_weights"].keys())
    correlations = {}

    for sig in signal_names:
        xs, ys = [], []
        for r in records:
            val = r.get("signals", {}).get(sig)
            ret = r.get("forward_return_1m")
            if val is not None and ret is not None:
                xs.append(float(val))
                ys.append(float(ret))
        if len(xs) >= 10:
            corr = float(np.corrcoef(xs, ys)[0, 1])
            correlations[sig] = round(corr, 4)

    return correlations
