"""
Alpha research: computes multiple independent return-predictive signals
for a list of candidate stocks and combines them into a composite alpha score.

Signals
-------
1. Cross-sectional momentum (12-1 month)
2. Short-term reversal (1-week)
3. Value / mean-reversion (Z-score of trailing PE vs sector median)
4. Trend following (price vs 200-day MA)
5. Earnings momentum (analyst revision proxy via targetMeanPrice gap)
6. Volatility-adjusted momentum (Sharpe-like over 6 months)
7. Volume surge (unusual volume as a breakout signal)
"""
import logging
from typing import Optional
import numpy as np
import pandas as pd

from config import MOMENTUM_LOOKBACK_DAYS, SHORT_MOMENTUM_DAYS

logger = logging.getLogger(__name__)


# ── Individual signal calculators ─────────────────────────────────────────────

def signal_cross_momentum(closes: pd.Series) -> float:
    """12-month minus 1-month momentum (standard Jegadeesh & Titman)."""
    n = len(closes)
    if n < MOMENTUM_LOOKBACK_DAYS:
        return np.nan
    ret_12m = closes.iloc[-1] / closes.iloc[-MOMENTUM_LOOKBACK_DAYS] - 1
    ret_1m  = closes.iloc[-1] / closes.iloc[-SHORT_MOMENTUM_DAYS] - 1
    return ret_12m - ret_1m   # exclude last month to avoid short-term reversal


def signal_short_reversal(closes: pd.Series) -> float:
    """1-week return (negative → expect reversion). We negate so higher = better."""
    if len(closes) < 6:
        return np.nan
    return -(closes.iloc[-1] / closes.iloc[-5] - 1)


def signal_trend_following(closes: pd.Series) -> float:
    """Price above 200-day MA → +1, below → -1, scaled by distance."""
    if len(closes) < 200:
        return np.nan
    ma200 = closes.iloc[-200:].mean()
    return (closes.iloc[-1] - ma200) / ma200


def signal_sharpe_momentum(closes: pd.Series, window: int = 126) -> float:
    """6-month Sharpe ratio (momentum / volatility)."""
    if len(closes) < window + 1:
        return np.nan
    rets = closes.pct_change().dropna().iloc[-window:]
    if rets.std() == 0:
        return np.nan
    return rets.mean() / rets.std() * np.sqrt(252)


def signal_volume_surge(volumes: pd.Series, window: int = 20) -> float:
    """Ratio of today's volume to 20-day average. Surge > 2× is a positive sign."""
    if len(volumes) < window + 1:
        return np.nan
    avg_vol = volumes.iloc[-window - 1 : -1].mean()
    if avg_vol == 0:
        return np.nan
    return volumes.iloc[-1] / avg_vol - 1   # 0 = average, >0 = above average


def signal_analyst_upside(fund: dict) -> float:
    """Analyst consensus price target upside vs current price."""
    cur = fund.get("currentPrice") or 0
    tgt = fund.get("targetMeanPrice") or 0
    if cur <= 0 or tgt <= 0:
        return np.nan
    return tgt / cur - 1


def signal_value_quality(fund: dict) -> float:
    """
    Blend of quality (ROE, gross margin) and moderate valuation (lower P/E preferred).
    Returns a score: higher = better quality-at-reasonable-price.
    """
    roe = fund.get("returnOnEquity") or np.nan
    gm  = fund.get("grossMargins") or np.nan
    pe  = fund.get("forwardPE") or fund.get("trailingPE") or np.nan

    quality = np.nanmean([roe, gm])   # avg of available quality metrics

    if np.isnan(pe) or pe <= 0:
        valuation = 0.0
    elif pe < 10:
        valuation = 1.0
    elif pe < 25:
        valuation = 1 - (pe - 10) / 30
    else:
        valuation = max(-1.0, -(pe - 25) / 30)

    if np.isnan(quality):
        return valuation
    return 0.6 * quality + 0.4 * valuation


# ── Z-score normalisation ─────────────────────────────────────────────────────

def _zscore(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score (demean, divide by std)."""
    mean, std = series.mean(), series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def _winsorise(series: pd.Series, z: float = 3.0) -> pd.Series:
    """Cap extreme values at ±z standard deviations."""
    return series.clip(lower=-z, upper=z)


# ── Composite alpha ───────────────────────────────────────────────────────────

# Signal weights (must sum to 1)
SIGNAL_WEIGHTS = {
    "cross_momentum":   0.30,
    "sharpe_momentum":  0.20,
    "trend_following":  0.15,
    "analyst_upside":   0.15,
    "value_quality":    0.10,
    "volume_surge":     0.05,
    "short_reversal":   0.05,
}


def compute_alpha(
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
) -> pd.DataFrame:
    """
    Compute normalised alpha scores for all candidate tickers.

    Parameters
    ----------
    tickers       : list of ticker symbols
    price_data    : {ticker: OHLCV DataFrame from data_fetcher}
    fundamentals  : {ticker: fundamentals dict from data_fetcher}

    Returns
    -------
    DataFrame sorted by alpha_score descending with columns:
    ticker, alpha_score, cross_momentum, sharpe_momentum, trend_following,
    analyst_upside, value_quality, volume_surge, short_reversal
    """
    rows = []
    for ticker in tickers:
        df  = price_data.get(ticker)
        fund = fundamentals.get(ticker, {})
        if df is None or df.empty:
            continue

        closes  = df["AdjClose"].astype(float)
        volumes = df["Volume"].astype(float)

        row = {
            "ticker":          ticker,
            "cross_momentum":  signal_cross_momentum(closes),
            "sharpe_momentum": signal_sharpe_momentum(closes),
            "trend_following": signal_trend_following(closes),
            "analyst_upside":  signal_analyst_upside(fund),
            "value_quality":   signal_value_quality(fund),
            "volume_surge":    signal_volume_surge(volumes),
            "short_reversal":  signal_short_reversal(closes),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).set_index("ticker")

    # Cross-sectional normalisation per signal
    signal_cols = list(SIGNAL_WEIGHTS.keys())
    for col in signal_cols:
        result[col] = _winsorise(_zscore(result[col].fillna(0)))

    # Composite alpha
    result["alpha_score"] = sum(
        result[col] * w for col, w in SIGNAL_WEIGHTS.items()
    )
    result["alpha_score"] = _zscore(result["alpha_score"])

    result = result.reset_index().sort_values("alpha_score", ascending=False)
    result = result.round(4)

    logger.info("Alpha computed for %d tickers", len(result))
    return result


# ── Convenience: summarise signal breakdown for a single ticker ───────────────

def alpha_breakdown(ticker: str, alpha_df: pd.DataFrame) -> dict:
    """Return the per-signal breakdown for a single ticker from the alpha DataFrame."""
    row = alpha_df[alpha_df["ticker"] == ticker]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()
