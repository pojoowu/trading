"""
Stock screener: filters the universe down to the most promising candidates
based on liquidity, market cap, and multi-factor momentum / quality scores.
"""
import logging
from typing import Optional
import pandas as pd
import numpy as np

from config import (
    UNIVERSE_TICKERS,
    MIN_MARKET_CAP_B,
    MIN_AVG_VOLUME,
    MAX_STOCKS_AFTER_SCREEN,
)
from data_fetcher import get_price_history, get_fundamentals

logger = logging.getLogger(__name__)


# ── Technical helpers ─────────────────────────────────────────────────────────

def _momentum(closes: pd.Series, lookback: int) -> float:
    """Simple price-return momentum over `lookback` trading days."""
    if len(closes) < lookback + 1:
        return float("nan")
    return closes.iloc[-1] / closes.iloc[-lookback] - 1


def _volatility(closes: pd.Series, window: int = 21) -> float:
    """Annualised 21-day rolling volatility."""
    returns = closes.pct_change().dropna()
    if len(returns) < window:
        return float("nan")
    return returns.iloc[-window:].std() * np.sqrt(252)


def _avg_volume(volumes: pd.Series, window: int = 20) -> float:
    if len(volumes) < window:
        return float("nan")
    return volumes.iloc[-window:].mean()


def _trend_strength(closes: pd.Series, short: int = 50, long: int = 200) -> float:
    """Return ratio of 50-day MA / 200-day MA. > 1 means uptrend."""
    if len(closes) < long:
        return float("nan")
    ma50 = closes.iloc[-short:].mean()
    ma200 = closes.iloc[-long:].mean()
    return ma50 / ma200 if ma200 else float("nan")


# ── Per-ticker scoring ────────────────────────────────────────────────────────

def score_ticker(
    ticker: str,
    df: pd.DataFrame,
    fundamentals: dict,
) -> dict:
    """
    Compute a composite screening score (0-100) for a ticker.
    Higher score = more attractive candidate.
    Returns a dict with score and individual signal values.
    """
    closes = df["AdjClose"].astype(float)
    volumes = df["Volume"].astype(float)

    # --- Liquidity filter ---
    avg_vol = _avg_volume(volumes)
    mkt_cap = fundamentals.get("marketCap") or 0
    mkt_cap_b = mkt_cap / 1e9

    if avg_vol < MIN_AVG_VOLUME or mkt_cap_b < MIN_MARKET_CAP_B:
        return {"ticker": ticker, "passed": False, "score": 0.0, "reason": "liquidity/market-cap filter"}

    # --- Signals ---
    mom_12m = _momentum(closes, 252)   # 12-month momentum (skip last month)
    mom_1m = _momentum(closes, 21)     # 1-month momentum
    mom_6m = _momentum(closes, 126)    # 6-month momentum
    vol_21d = _volatility(closes, 21)
    trend = _trend_strength(closes)

    # 52-week position (0=at low, 1=at high)
    hi52 = fundamentals.get("fiftyTwoWeekHigh") or closes.max()
    lo52 = fundamentals.get("fiftyTwoWeekLow") or closes.min()
    price_pos = (closes.iloc[-1] - lo52) / (hi52 - lo52) if hi52 != lo52 else 0.5

    # Fundamental quality signals (normalised, higher = better)
    roe = fundamentals.get("returnOnEquity") or 0
    rev_growth = fundamentals.get("revenueGrowth") or 0
    op_margin = fundamentals.get("operatingMargins") or 0

    # Analyst target upside
    cur_price = fundamentals.get("currentPrice") or closes.iloc[-1]
    target = fundamentals.get("targetMeanPrice") or cur_price
    analyst_upside = (target / cur_price - 1) if cur_price else 0

    # --- Composite score (weighted sum, capped to [0, 100]) ---
    def _clamp(x, lo=-1, hi=1):
        return max(lo, min(hi, x)) if not (x != x) else 0  # NaN -> 0

    score = 0.0
    score += 25 * _clamp((mom_12m or 0) / 0.60)          # 12m momentum
    score += 15 * _clamp((mom_6m or 0) / 0.40)           # 6m momentum
    score += 10 * (1 - _clamp((vol_21d or 0.30) / 0.80)) # low vol bonus
    score += 10 * _clamp((trend or 1) - 1, -0.3, 0.3) / 0.3  # MA trend
    score += 10 * price_pos                               # price near high
    score += 10 * _clamp((roe or 0) / 0.30)              # ROE
    score += 10 * _clamp((rev_growth or 0) / 0.30)       # revenue growth
    score += 5  * _clamp((op_margin or 0) / 0.30)        # operating margin
    score += 5  * _clamp((analyst_upside or 0) / 0.30)   # analyst upside

    score = max(0.0, min(100.0, score))

    return {
        "ticker": ticker,
        "passed": True,
        "score": round(score, 2),
        "mom_12m": round(mom_12m * 100, 2) if mom_12m == mom_12m else None,
        "mom_6m": round(mom_6m * 100, 2) if mom_6m == mom_6m else None,
        "mom_1m": round(mom_1m * 100, 2) if mom_1m == mom_1m else None,
        "vol_21d": round(vol_21d * 100, 2) if vol_21d == vol_21d else None,
        "trend_ma": round(trend, 4) if trend == trend else None,
        "price_pos_52w": round(price_pos * 100, 1),
        "mkt_cap_b": round(mkt_cap_b, 2),
        "avg_volume": int(avg_vol),
        "roe_pct": round((roe or 0) * 100, 2),
        "rev_growth_pct": round((rev_growth or 0) * 100, 2),
        "analyst_upside_pct": round(analyst_upside * 100, 2),
        "sector": fundamentals.get("sector", "Unknown"),
        "industry": fundamentals.get("industry", "Unknown"),
    }


# ── Main screening function ───────────────────────────────────────────────────

def run_screen(
    tickers: Optional[list[str]] = None,
    price_data: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Screen the universe and return a DataFrame of passing stocks sorted by score.

    Parameters
    ----------
    tickers : list of ticker strings (default: UNIVERSE_TICKERS from config)
    price_data : pre-fetched price DataFrames {ticker: df} (optional, to avoid re-fetching)

    Returns
    -------
    pd.DataFrame with columns: ticker, score, signals…, sector, industry
    Top MAX_STOCKS_AFTER_SCREEN rows only.
    """
    if tickers is None:
        tickers = UNIVERSE_TICKERS

    logger.info("Screening %d tickers…", len(tickers))
    rows = []

    for ticker in tickers:
        try:
            # Price history
            if price_data and ticker in price_data:
                df = price_data[ticker]
            else:
                df = get_price_history(ticker, years=2)
            if df is None or len(df) < 60:
                logger.debug("Skipping %s – insufficient price data", ticker)
                continue

            # Fundamentals
            fund = get_fundamentals(ticker)

            row = score_ticker(ticker, df, fund)
            rows.append(row)
        except Exception as exc:
            logger.warning("Error screening %s: %s", ticker, exc)

    if not rows:
        return pd.DataFrame()

    df_result = pd.DataFrame(rows)
    passed = df_result[df_result["passed"]].copy()
    passed = passed.sort_values("score", ascending=False).head(MAX_STOCKS_AFTER_SCREEN)
    passed = passed.reset_index(drop=True)

    logger.info(
        "Screening complete: %d passed, top %d selected",
        len(df_result[df_result["passed"]]),
        len(passed),
    )
    return passed
