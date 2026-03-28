"""
Backtesting engine: vectorised event-driven backtester.

Strategy: buy/hold/sell based on composite alpha signals
- Enter: alpha Z-score > threshold
- Exit:  alpha Z-score < exit threshold, stop-loss hit, or take-profit hit
- Position sizing: equal-weight across top-N signals

Metrics returned: CAGR, Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor,
Calmar, monthly returns heatmap, trade log.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from config import (
    BACKTEST_YEARS,
    BACKTEST_INITIAL_CAPITAL,
    BACKTEST_COMMISSION,
    MAX_POSITIONS,
    MAX_POSITION_PCT,
)
from analyzer import calc_rsi, calc_macd, _ema

logger = logging.getLogger(__name__)


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    ticker:     str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date:  Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    shares:     float = 0.0
    pnl:        float = 0.0
    pnl_pct:    float = 0.0
    exit_reason: str = ""


# ── Metrics ───────────────────────────────────────────────────────────────────

def _calc_metrics(equity: pd.Series, trades: list[Trade], rf: float = 0.04) -> dict:
    """Compute performance metrics from daily equity curve and trade list."""
    if len(equity) < 2:
        return {}

    rets = equity.pct_change().dropna()
    n_days = len(rets)
    ann = 252

    # CAGR
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = n_days / ann
    cagr = (1 + total_return) ** (1 / max(years, 0.01)) - 1

    # Sharpe
    excess = rets - rf / ann
    sharpe = excess.mean() / excess.std() * np.sqrt(ann) if excess.std() > 0 else 0

    # Sortino (downside deviation)
    downside = rets[rets < 0]
    sortino = (rets.mean() - rf / ann) / downside.std() * np.sqrt(ann) if len(downside) > 1 else 0

    # Max drawdown
    peak = equity.expanding().max()
    dd   = (equity - peak) / peak
    max_dd = dd.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate / profit factor
    completed = [t for t in trades if t.exit_date is not None]
    winners = [t for t in completed if t.pnl > 0]
    losers  = [t for t in completed if t.pnl <= 0]
    win_rate = len(winners) / len(completed) if completed else 0
    gross_profit = sum(t.pnl for t in winners)
    gross_loss   = abs(sum(t.pnl for t in losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Monthly returns
    monthly = equity.resample("ME").last().pct_change().dropna()

    return {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct":         round(cagr * 100, 2),
        "sharpe":           round(sharpe, 3),
        "sortino":          round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar":           round(calmar, 3),
        "win_rate_pct":     round(win_rate * 100, 2),
        "profit_factor":    round(profit_factor, 3),
        "total_trades":     len(completed),
        "avg_trade_pct":    round(np.mean([t.pnl_pct for t in completed]) * 100, 2) if completed else 0,
        "monthly_returns":  monthly.round(4).to_dict(),
        "best_month_pct":   round(monthly.max() * 100, 2) if not monthly.empty else 0,
        "worst_month_pct":  round(monthly.min() * 100, 2) if not monthly.empty else 0,
    }


# ── Vectorised signal generator ───────────────────────────────────────────────

def _daily_signal(closes: pd.Series) -> pd.Series:
    """
    Fast vectorised signal: +1 (long), 0 (flat), -1 (short).
    Based on: momentum trend + RSI + MACD agreement.
    """
    n = len(closes)
    if n < 220:
        return pd.Series(0, index=closes.index)

    ma50  = closes.rolling(50).mean()
    ma200 = closes.rolling(200).mean()
    rsi   = calc_rsi(closes, 14)
    macd, sig, _ = calc_macd(closes)

    # Conditions for LONG
    trend_up = (ma50 > ma200).astype(int)
    rsi_ok   = ((rsi > 40) & (rsi < 75)).astype(int)
    macd_up  = (macd > sig).astype(int)

    # Combined: all 3 must agree for a trade
    signal = ((trend_up + rsi_ok + macd_up) >= 2).astype(int)
    return signal


# ── Single-stock backtest ─────────────────────────────────────────────────────

def backtest_single(
    ticker: str,
    df: pd.DataFrame,
    initial_capital: float = BACKTEST_INITIAL_CAPITAL,
    stop_loss_pct: float = 0.07,    # -7% stop
    take_profit_pct: float = 0.20,  # +20% take profit
    commission: float = BACKTEST_COMMISSION,
) -> dict:
    """
    Run a single-stock momentum strategy backtest.
    Returns a dict with equity curve, trade log, and performance metrics.
    """
    if df is None or len(df) < 220:
        return {"ticker": ticker, "error": "insufficient history", "metrics": {}}

    closes = df["AdjClose"].astype(float)
    signal = _daily_signal(closes)

    cash    = initial_capital
    shares  = 0.0
    entry   = None
    trades  = []
    equity  = []

    dates  = df.index.tolist()
    prices = closes.values
    sigs   = signal.values

    for i, (date, price, sig) in enumerate(zip(dates, prices, sigs)):
        # Mark-to-market equity
        equity.append(cash + shares * price)

        if shares == 0 and sig == 1 and i < len(sigs) - 1:
            # Enter long
            cost    = price * (1 + commission)
            shares  = (cash * (1 - commission)) / cost
            cash    = 0.0
            entry   = Trade(
                ticker=ticker,
                entry_date=date,
                entry_price=price,
                shares=shares,
            )

        elif shares > 0 and entry is not None:
            ret = price / entry.entry_price - 1
            exit_reason = ""

            if ret <= -stop_loss_pct:
                exit_reason = "stop_loss"
            elif ret >= take_profit_pct:
                exit_reason = "take_profit"
            elif sig == 0 and i > 0:
                exit_reason = "signal_exit"

            if exit_reason:
                proceeds = shares * price * (1 - commission)
                cash     = proceeds
                entry.exit_date  = date
                entry.exit_price = price
                entry.pnl        = proceeds - entry.shares * entry.entry_price
                entry.pnl_pct    = ret
                entry.exit_reason = exit_reason
                trades.append(entry)
                shares = 0.0
                entry  = None

    # Close any open position at last price
    if shares > 0 and entry is not None:
        price = prices[-1]
        proceeds = shares * price * (1 - commission)
        entry.exit_date  = dates[-1]
        entry.exit_price = price
        entry.pnl        = proceeds - entry.shares * entry.entry_price
        entry.pnl_pct    = price / entry.entry_price - 1
        entry.exit_reason = "end_of_backtest"
        trades.append(entry)
        cash = proceeds

    equity_series = pd.Series(equity, index=dates)
    metrics = _calc_metrics(equity_series, trades)
    metrics["ticker"] = ticker

    # Benchmark: buy-and-hold
    bh_return = (prices[-1] / prices[0] - 1) * 100
    metrics["buy_hold_return_pct"] = round(bh_return, 2)
    metrics["alpha_vs_bh_pct"] = round(
        metrics.get("total_return_pct", 0) - bh_return, 2
    )

    return {
        "ticker":        ticker,
        "equity_curve":  equity_series,
        "trades":        trades,
        "metrics":       metrics,
        "final_capital": round(equity[-1], 2) if equity else initial_capital,
    }


# ── Portfolio backtest ────────────────────────────────────────────────────────

def backtest_portfolio(
    price_data: dict[str, pd.DataFrame],
    top_n: int = MAX_POSITIONS,
    rebalance_freq: int = 21,   # rebalance every ~1 month
    initial_capital: float = BACKTEST_INITIAL_CAPITAL,
    commission: float = BACKTEST_COMMISSION,
) -> dict:
    """
    Run a long-only portfolio backtest that rebalances into top-N momentum stocks.

    Parameters
    ----------
    price_data      : {ticker: OHLCV DataFrame}
    top_n           : max number of simultaneous positions
    rebalance_freq  : trading days between rebalances
    initial_capital : starting capital in $
    commission      : one-way commission rate

    Returns
    -------
    dict with equity_curve, metrics, and allocation history
    """
    if not price_data:
        return {"error": "no price data"}

    # Align on common dates
    all_closes = {t: df["AdjClose"].astype(float) for t, df in price_data.items()}
    combined   = pd.DataFrame(all_closes).dropna(axis=0, how="all")
    combined   = combined.fillna(method="ffill").fillna(method="bfill")

    if len(combined) < 220:
        return {"error": "insufficient shared history"}

    tickers = combined.columns.tolist()
    dates   = combined.index.tolist()

    # Portfolio state
    portfolio_value = initial_capital
    holdings = {}          # {ticker: shares}
    cash     = initial_capital
    equity   = []
    alloc_history = []

    for day_idx, date in enumerate(dates):
        prices = combined.loc[date]

        # Current portfolio value
        pv = cash + sum(holdings.get(t, 0) * prices.get(t, 0) for t in holdings)
        equity.append(pv)

        # Rebalance
        if day_idx % rebalance_freq == 0 or day_idx == 0:
            # Compute signal for each ticker using data up to today
            scores = {}
            for ticker in tickers:
                hist = combined.iloc[: day_idx + 1][ticker].dropna()
                if len(hist) >= 220:
                    sig = _daily_signal(hist)
                    if sig.iloc[-1] == 1:
                        # Score = 6m return / 6m vol (Sharpe-like)
                        ret6m = hist.pct_change().dropna().iloc[-126:]
                        sh = ret6m.mean() / ret6m.std() if ret6m.std() > 0 else 0
                        scores[ticker] = sh

            # Top N
            top_tickers = sorted(scores, key=scores.get, reverse=True)[:top_n]

            # Sell positions not in top list
            for t in list(holdings.keys()):
                if t not in top_tickers and holdings[t] > 0:
                    proceeds = holdings[t] * prices.get(t, 0) * (1 - commission)
                    cash += proceeds
                    holdings[t] = 0

            # Buy new positions (equal weight)
            if top_tickers:
                invest_per_stock = (cash * (1 - 0.05)) / len(top_tickers)  # 5% cash reserve
                for t in top_tickers:
                    if holdings.get(t, 0) == 0 and prices.get(t, 0) > 0:
                        shares = invest_per_stock / (prices[t] * (1 + commission))
                        holdings[t] = shares
                        cash -= shares * prices[t] * (1 + commission)

            alloc_history.append({
                "date": str(date.date()),
                "positions": top_tickers,
                "portfolio_value": round(pv, 2),
            })

    equity_series = pd.Series(equity, index=dates)
    metrics = _calc_metrics(equity_series, [], rf=0.04)

    # Benchmark (equal-weight buy-hold from start)
    n_tickers = len(tickers)
    bh_shares = {t: (initial_capital / n_tickers) / combined[t].iloc[0]
                 for t in tickers if combined[t].iloc[0] > 0}
    bh_equity = sum(bh_shares.get(t, 0) * combined[t] for t in tickers)
    bh_return  = (bh_equity.iloc[-1] / bh_equity.iloc[0] - 1) * 100
    metrics["benchmark_bh_return_pct"] = round(bh_return, 2)
    metrics["alpha_vs_benchmark_pct"]  = round(
        metrics.get("total_return_pct", 0) - bh_return, 2
    )

    return {
        "equity_curve":    equity_series,
        "metrics":         metrics,
        "alloc_history":   alloc_history[-10:],   # last 10 rebalances
        "final_capital":   round(equity[-1], 2),
    }


# ── Human-readable summary ────────────────────────────────────────────────────

def format_backtest_report(result: dict) -> str:
    m = result.get("metrics", {})
    ticker = result.get("ticker", "Portfolio")
    lines = [
        f"{'=' * 55}",
        f"  BACKTEST RESULTS: {ticker}",
        f"{'=' * 55}",
        f"  Total Return     : {m.get('total_return_pct', 'N/A')}%",
        f"  CAGR             : {m.get('cagr_pct', 'N/A')}%",
        f"  Buy & Hold Ret   : {m.get('buy_hold_return_pct', m.get('benchmark_bh_return_pct', 'N/A'))}%",
        f"  Alpha vs B&H     : {m.get('alpha_vs_bh_pct', m.get('alpha_vs_benchmark_pct', 'N/A'))}%",
        f"",
        f"  Sharpe           : {m.get('sharpe', 'N/A')}",
        f"  Sortino          : {m.get('sortino', 'N/A')}",
        f"  Calmar           : {m.get('calmar', 'N/A')}",
        f"  Max Drawdown     : {m.get('max_drawdown_pct', 'N/A')}%",
        f"",
        f"  Total Trades     : {m.get('total_trades', 'N/A')}",
        f"  Win Rate         : {m.get('win_rate_pct', 'N/A')}%",
        f"  Profit Factor    : {m.get('profit_factor', 'N/A')}",
        f"  Avg Trade        : {m.get('avg_trade_pct', 'N/A')}%",
        f"  Best Month       : {m.get('best_month_pct', 'N/A')}%",
        f"  Worst Month      : {m.get('worst_month_pct', 'N/A')}%",
        f"{'=' * 55}",
    ]
    return "\n".join(lines)
