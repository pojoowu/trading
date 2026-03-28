"""
Intraday backtester: simulates all three intraday strategies on hourly bars,
computes full metrics, and runs parameter grid search to find the best config.

Key difference from the daily backtester:
- Bars are hourly (not daily)
- Positions are opened and closed within a single trading day
- Stop-loss and take-profit are checked on every bar
- Overnight holding is NOT allowed (all positions closed at 3:45pm ET)
- Slippage is simulated as half the bar's spread
"""
import logging
import itertools
import copy
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
import pandas as pd

from intraday_strategy import (
    get_entry_signal,
    calc_atr_intraday,
    VWAPReversionParams,
    ORBParams,
    HourlyMomentumParams,
)

logger = logging.getLogger(__name__)


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class IntradayTrade:
    ticker:       str
    entry_time:   str
    exit_time:    str
    entry_price:  float
    exit_price:   float
    shares:       float
    pnl:          float
    pnl_pct:      float
    exit_reason:  str   # "take_profit" | "stop_loss" | "eod_close" | "signal_exit"


# ── Core simulation ───────────────────────────────────────────────────────────

def _simulate(
    ticker: str,
    df: pd.DataFrame,
    signal: pd.Series,
    stop_loss_pct:   float,
    take_profit_pct: float,
    capital_per_trade: float = 10_000.0,
    commission:        float = 0.001,
    slippage_pct:      float = 0.0005,
) -> tuple[list[IntradayTrade], pd.Series]:
    """
    Simulate trades from a signal Series on hourly bars.
    One trade at a time (no pyramiding).
    Returns (trades, equity_series).
    """
    trades    = []
    cash      = capital_per_trade
    shares    = 0.0
    entry_px  = 0.0
    entry_ts  = None
    equity    = []

    dates     = df.index
    closes    = df["Close"].values
    highs     = df["High"].values
    lows      = df["Low"].values
    sigs      = signal.values

    for i in range(len(dates)):
        ts    = dates[i]
        close = closes[i]
        high  = highs[i]
        low   = lows[i]
        sig   = sigs[i]
        hour  = ts.hour

        # Mark-to-market
        equity.append(cash + shares * close)

        # Manage open position
        if shares > 0 and entry_px > 0:
            # Check stop and take-profit on the bar's range
            stop_px   = entry_px * (1 - stop_loss_pct)
            target_px = entry_px * (1 + take_profit_pct)
            eod       = hour >= 15   # force close at / after 3pm

            exit_px     = None
            exit_reason = ""

            if low <= stop_px:
                exit_px     = stop_px * (1 - slippage_pct)
                exit_reason = "stop_loss"
            elif high >= target_px:
                exit_px     = target_px * (1 - slippage_pct)
                exit_reason = "take_profit"
            elif eod:
                exit_px     = close * (1 - slippage_pct)
                exit_reason = "eod_close"

            if exit_px:
                proceeds   = shares * exit_px * (1 - commission)
                pnl        = proceeds - shares * entry_px
                pnl_pct    = exit_px / entry_px - 1
                trades.append(IntradayTrade(
                    ticker=ticker,
                    entry_time=str(entry_ts),
                    exit_time=str(ts),
                    entry_price=round(entry_px, 4),
                    exit_price=round(exit_px, 4),
                    shares=round(shares, 4),
                    pnl=round(pnl, 2),
                    pnl_pct=round(pnl_pct, 5),
                    exit_reason=exit_reason,
                ))
                cash    = proceeds
                shares  = 0.0
                entry_px = 0.0
                entry_ts = None

        # Enter new position
        if shares == 0 and sig == 1 and hour < 15:
            enter_px = close * (1 + slippage_pct)
            cost     = capital_per_trade * (1 + commission)
            if cash >= cost:
                shares   = capital_per_trade / enter_px
                cash    -= shares * enter_px * (1 + commission)
                entry_px = enter_px
                entry_ts = ts

    return trades, pd.Series(equity, index=dates)


# ── Metrics ───────────────────────────────────────────────────────────────────

def _metrics(trades: list[IntradayTrade], equity: pd.Series) -> dict:
    if not trades:
        return {
            "total_trades": 0, "win_rate_pct": 0, "profit_factor": 0,
            "avg_trade_pct": 0, "sharpe": 0, "max_drawdown_pct": 0,
            "total_return_pct": 0, "expectancy": 0,
        }

    winners = [t for t in trades if t.pnl > 0]
    losers  = [t for t in trades if t.pnl <= 0]

    win_rate      = len(winners) / len(trades)
    gross_profit  = sum(t.pnl for t in winners)
    gross_loss    = abs(sum(t.pnl for t in losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_win       = gross_profit / len(winners) if winners else 0
    avg_loss      = gross_loss / len(losers)    if losers  else 0
    expectancy    = win_rate * avg_win - (1 - win_rate) * avg_loss

    # Daily equity for Sharpe
    daily_eq  = equity.resample("D").last().dropna()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe    = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0

    # Max drawdown
    peak   = equity.expanding().max()
    max_dd = ((equity - peak) / peak).min()

    total_ret = equity.iloc[-1] / equity.iloc[0] - 1 if len(equity) > 1 else 0

    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    return {
        "total_trades":     len(trades),
        "win_rate_pct":     round(win_rate * 100, 2),
        "profit_factor":    round(profit_factor, 3),
        "avg_trade_pct":    round(np.mean([t.pnl_pct for t in trades]) * 100, 3),
        "expectancy":       round(expectancy, 2),
        "sharpe":           round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_return_pct": round(total_ret * 100, 2),
        "exit_reasons":     exit_reasons,
    }


# ── Single strategy backtest ──────────────────────────────────────────────────

def backtest_intraday(
    ticker:    str,
    df:        pd.DataFrame,
    strategy:  str  = "vwap_reversion",
    params:    Optional[dict] = None,
    stop_loss_pct:   float = 0.015,
    take_profit_pct: float = 0.030,
) -> dict:
    """
    Backtest a single intraday strategy on hourly bars.

    Parameters
    ----------
    ticker          : ticker symbol (for labelling)
    df              : hourly OHLCV DataFrame from intraday_data.get_intraday_bars
    strategy        : "vwap_reversion" | "orb" | "hourly_momentum"
    params          : strategy parameter overrides
    stop_loss_pct   : hard stop loss per trade
    take_profit_pct : take profit per trade

    Returns
    -------
    dict with metrics, equity_curve, trades, best_params
    """
    if df is None or len(df) < 50:
        return {"ticker": ticker, "strategy": strategy, "error": "insufficient data", "metrics": {}}

    signal = get_entry_signal(df, strategy, params)
    trades, equity = _simulate(
        ticker, df, signal,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )
    m = _metrics(trades, equity)
    m["ticker"]   = ticker
    m["strategy"] = strategy

    return {
        "ticker":       ticker,
        "strategy":     strategy,
        "metrics":      m,
        "equity_curve": equity,
        "trades":       trades,
        "params_used":  params or {},
    }


# ── Parameter grid search ─────────────────────────────────────────────────────

# Search grids per strategy
PARAM_GRIDS = {
    "vwap_reversion": {
        "rsi_oversold":    [30, 35, 40],
        "vwap_entry_pct":  [0.001, 0.002, 0.003],
        "stop_loss_pct":   [0.010, 0.015, 0.020],
        "take_profit_pct": [0.020, 0.030, 0.040],
    },
    "orb": {
        "range_hours":     [1, 2],
        "volume_mult":     [1.2, 1.5, 2.0],
        "stop_loss_pct":   [0.015, 0.020, 0.030],
        "take_profit_pct": [0.030, 0.040, 0.060],
    },
    "hourly_momentum": {
        "momentum_threshold": [0.003, 0.005, 0.008],
        "rsi_min":            [40, 45, 50],
        "trailing_stop_pct":  [0.015, 0.020, 0.025],
        "take_profit_pct":    [0.030, 0.040, 0.050],
        "volume_mult":        [1.1, 1.3, 1.5],
    },
}


def _composite_score(m: dict) -> float:
    """Rank parameter sets: favour Sharpe + profit factor, penalise drawdown."""
    sharpe = m.get("sharpe", 0) or 0
    pf     = min(m.get("profit_factor", 0) or 0, 5)   # cap at 5
    wr     = (m.get("win_rate_pct", 0) or 0) / 100
    dd     = abs(m.get("max_drawdown_pct", 0) or 0)
    trades = m.get("total_trades", 0) or 0

    if trades < 5:   # need enough trades to be statistically meaningful
        return -999

    dd_penalty = max(0, dd - 10) * 0.05
    return sharpe * 0.4 + pf * 0.3 + wr * 0.2 - dd_penalty


def optimize_intraday(
    ticker:   str,
    df:       pd.DataFrame,
    strategy: str = "vwap_reversion",
    max_combinations: int = 50,
) -> dict:
    """
    Run a parameter grid search for an intraday strategy.
    Evaluates up to max_combinations parameter sets and returns the best.

    Returns
    -------
    dict with best_params, best_metrics, all_results (top 5)
    """
    if df is None or len(df) < 50:
        return {"error": "insufficient data"}

    grid   = PARAM_GRIDS.get(strategy, {})
    keys   = list(grid.keys())
    values = list(grid.values())

    all_combos = list(itertools.product(*values))
    # Randomly sample if too many
    if len(all_combos) > max_combinations:
        import random
        all_combos = random.sample(all_combos, max_combinations)

    results = []
    logger.info("Optimising %s on %s: testing %d parameter sets…", strategy, ticker, len(all_combos))

    for combo in all_combos:
        params = dict(zip(keys, combo))

        # Extract stop/take-profit from params (may be in strategy params)
        sl  = params.pop("stop_loss_pct",   0.015)
        tp  = params.pop("take_profit_pct", 0.030)

        result = backtest_intraday(ticker, df, strategy, params, sl, tp)
        m      = result.get("metrics", {})
        score  = _composite_score(m)

        results.append({
            "params":            {**params, "stop_loss_pct": sl, "take_profit_pct": tp},
            "metrics":           m,
            "score":             score,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    top5 = results[:5]

    best = top5[0] if top5 else {}
    logger.info(
        "Best %s params for %s: score=%.3f  sharpe=%.2f  win_rate=%.1f%%  trades=%d",
        strategy, ticker,
        best.get("score", 0),
        best.get("metrics", {}).get("sharpe", 0),
        best.get("metrics", {}).get("win_rate_pct", 0),
        best.get("metrics", {}).get("total_trades", 0),
    )

    return {
        "ticker":       ticker,
        "strategy":     strategy,
        "best_params":  best.get("params", {}),
        "best_metrics": best.get("metrics", {}),
        "best_score":   best.get("score", 0),
        "top5":         top5,
    }


# ── Cross-strategy comparison ─────────────────────────────────────────────────

def compare_strategies(
    ticker: str,
    df:     pd.DataFrame,
) -> dict:
    """
    Run and compare all three intraday strategies on the same data.
    Returns the winner and ranked results.
    """
    strategies = ["vwap_reversion", "orb", "hourly_momentum"]
    results    = []

    for strat in strategies:
        opt = optimize_intraday(ticker, df, strat)
        results.append({
            "strategy":    strat,
            "best_params": opt.get("best_params", {}),
            "metrics":     opt.get("best_metrics", {}),
            "score":       opt.get("best_score", -999),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    winner = results[0]

    return {
        "ticker":   ticker,
        "winner":   winner["strategy"],
        "best_params": winner["best_params"],
        "best_metrics": winner["metrics"],
        "ranking":  results,
    }


# ── Human-readable report ─────────────────────────────────────────────────────

def format_intraday_report(result: dict) -> str:
    m      = result.get("best_metrics") or result.get("metrics", {})
    strat  = result.get("winner") or result.get("strategy", "?")
    ticker = result.get("ticker", "?")
    params = result.get("best_params") or result.get("params_used", {})

    lines = [
        f"{'=' * 55}",
        f"  INTRADAY BACKTEST: {ticker}  [{strat}]",
        f"{'=' * 55}",
        f"  Total trades     : {m.get('total_trades', 'N/A')}",
        f"  Win rate         : {m.get('win_rate_pct', 'N/A')}%",
        f"  Profit factor    : {m.get('profit_factor', 'N/A')}",
        f"  Avg trade        : {m.get('avg_trade_pct', 'N/A')}%",
        f"  Expectancy       : ${m.get('expectancy', 'N/A')}",
        f"  Sharpe           : {m.get('sharpe', 'N/A')}",
        f"  Max drawdown     : {m.get('max_drawdown_pct', 'N/A')}%",
        f"  Total return     : {m.get('total_return_pct', 'N/A')}%",
        f"",
        f"  Exit reasons     : {m.get('exit_reasons', {})}",
        f"",
        f"  Best params      : {params}",
        f"{'=' * 55}",
    ]
    return "\n".join(lines)
