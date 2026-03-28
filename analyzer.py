"""
Deep analyzer: computes a full suite of technical indicators and
assembles a human-readable research report for a single stock.

Technical indicators (all implemented from scratch – no external TA lib):
  RSI, MACD, Bollinger Bands, ATR, Stochastic, OBV, EMA crossovers,
  support/resistance levels, recent candlestick patterns.
"""
import logging
from typing import Optional
import numpy as np
import pandas as pd

from config import RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, BOLLINGER_PERIOD, BOLLINGER_STD

logger = logging.getLogger(__name__)


# ── Technical indicator library ───────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(closes: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = closes.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_macd(closes: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    fast = _ema(closes, MACD_FAST)
    slow = _ema(closes, MACD_SLOW)
    macd = fast - slow
    signal = _ema(macd, MACD_SIGNAL)
    hist = macd - signal
    return macd, signal, hist


def calc_bollinger(closes: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, mid, lower) Bollinger Bands."""
    mid   = closes.rolling(BOLLINGER_PERIOD).mean()
    std   = closes.rolling(BOLLINGER_PERIOD).std()
    upper = mid + BOLLINGER_STD * std
    lower = mid - BOLLINGER_STD * std
    return upper, mid, lower


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def calc_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Returns (%K, %D)."""
    lo_min = low.rolling(k_period).min()
    hi_max = high.rolling(k_period).max()
    k = 100 * (close - lo_min) / (hi_max - lo_min).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def calc_obv(closes: pd.Series, volumes: pd.Series) -> pd.Series:
    direction = np.sign(closes.diff()).fillna(0)
    return (direction * volumes).cumsum()


def find_support_resistance(closes: pd.Series, window: int = 20, n_levels: int = 3) -> dict:
    """Identify key support and resistance levels using local min/max."""
    price = closes.values
    supports    = []
    resistances = []
    for i in range(window, len(price) - window):
        segment = price[i - window : i + window + 1]
        if price[i] == segment.min():
            supports.append(float(price[i]))
        if price[i] == segment.max():
            resistances.append(float(price[i]))

    def _cluster(levels, n):
        if not levels:
            return []
        arr = np.array(sorted(set(levels)))
        # simple mean-shift clustering
        out = []
        used = np.zeros(len(arr), bool)
        for _ in range(n):
            remaining = arr[~used]
            if not remaining.size:
                break
            best = remaining[-1] if _ == 0 else remaining[0]  # top resistance first
            cluster = arr[np.abs(arr - best) < 0.03 * best]
            out.append(float(cluster.mean()))
            used |= np.abs(arr - cluster.mean()) < 0.03 * best
        return sorted(out)

    return {
        "supports":    _cluster(supports, n_levels),
        "resistances": _cluster(resistances, n_levels),
    }


# ── Pattern detection ─────────────────────────────────────────────────────────

def detect_patterns(df: pd.DataFrame) -> list[str]:
    """
    Detect common candlestick / chart patterns over the last 5 bars.
    Returns a list of pattern names found.
    """
    patterns = []
    if len(df) < 5:
        return patterns

    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values

    # Doji (open ≈ close)
    if abs(c[-1] - o[-1]) < 0.001 * c[-1]:
        patterns.append("Doji")

    # Hammer (long lower shadow, small body at top)
    body = abs(c[-1] - o[-1])
    lower_shadow = min(c[-1], o[-1]) - l[-1]
    upper_shadow = h[-1] - max(c[-1], o[-1])
    if lower_shadow > 2 * body and upper_shadow < body and body > 0:
        patterns.append("Hammer")

    # Shooting Star (inverse hammer at top)
    if upper_shadow > 2 * body and lower_shadow < body and body > 0:
        patterns.append("Shooting Star")

    # Engulfing patterns
    if len(c) >= 2:
        prev_body = abs(c[-2] - o[-2])
        curr_body = abs(c[-1] - o[-1])
        if (
            c[-2] < o[-2] and c[-1] > o[-1]        # prev red, curr green
            and o[-1] <= c[-2] and c[-1] >= o[-2]  # current engulfs previous
            and curr_body > prev_body
        ):
            patterns.append("Bullish Engulfing")
        if (
            c[-2] > o[-2] and c[-1] < o[-1]
            and o[-1] >= c[-2] and c[-1] <= o[-2]
            and curr_body > prev_body
        ):
            patterns.append("Bearish Engulfing")

    # Golden / Death cross (short-term MA crossovers)
    close_series = pd.Series(df["Close"].values)
    ma20 = close_series.rolling(20).mean()
    ma50 = close_series.rolling(50).mean()
    if len(ma20.dropna()) > 2 and len(ma50.dropna()) > 2:
        if ma20.iloc[-2] < ma50.iloc[-2] and ma20.iloc[-1] > ma50.iloc[-1]:
            patterns.append("Golden Cross (20/50)")
        if ma20.iloc[-2] > ma50.iloc[-2] and ma20.iloc[-1] < ma50.iloc[-1]:
            patterns.append("Death Cross (20/50)")

    return patterns


# ── Full analysis for a single ticker ────────────────────────────────────────

def analyze_stock(
    ticker: str,
    df: pd.DataFrame,
    fund: Optional[dict] = None,
) -> dict:
    """
    Compute all technical indicators and return a comprehensive analysis dict.

    Parameters
    ----------
    ticker : str
    df     : OHLCV DataFrame from data_fetcher.get_price_history
    fund   : fundamentals dict from data_fetcher.get_fundamentals (optional)

    Returns
    -------
    dict with technical signals, trend assessment, risk metrics, and recommendation
    """
    if df is None or len(df) < 30:
        return {"ticker": ticker, "error": "insufficient data"}

    fund = fund or {}
    closes  = df["AdjClose"].astype(float)
    highs   = df["High"].astype(float)
    lows    = df["Low"].astype(float)
    volumes = df["Volume"].astype(float)
    opens   = df["Open"].astype(float)

    # --- Technical indicators ---
    rsi      = calc_rsi(closes)
    macd, macd_sig, macd_hist = calc_macd(closes)
    bb_up, bb_mid, bb_lo      = calc_bollinger(closes)
    atr      = calc_atr(highs, lows, closes)
    stoch_k, stoch_d          = calc_stochastic(highs, lows, closes)
    obv      = calc_obv(closes, volumes)

    # Moving averages
    ma20  = closes.rolling(20).mean()
    ma50  = closes.rolling(50).mean()
    ma200 = closes.rolling(200).mean()

    # Current values
    price   = closes.iloc[-1]
    rsi_cur = rsi.iloc[-1]
    macd_cur = macd.iloc[-1]
    macd_sig_cur = macd_sig.iloc[-1]
    macd_hist_cur = macd_hist.iloc[-1]
    bb_pct  = (price - bb_lo.iloc[-1]) / (bb_up.iloc[-1] - bb_lo.iloc[-1])  # 0-1
    stoch_k_cur = stoch_k.iloc[-1]
    stoch_d_cur = stoch_d.iloc[-1]
    atr_cur = atr.iloc[-1]
    atr_pct = atr_cur / price * 100    # ATR as % of price (volatility)

    obv_trend = "rising" if obv.iloc[-1] > obv.iloc[-20] else "falling"

    # MA relationship
    above_ma20  = price > ma20.iloc[-1] if not np.isnan(ma20.iloc[-1]) else None
    above_ma50  = price > ma50.iloc[-1] if not np.isnan(ma50.iloc[-1]) else None
    above_ma200 = price > ma200.iloc[-1] if not np.isnan(ma200.iloc[-1]) else None

    # Support / resistance
    sr = find_support_resistance(closes)

    # Patterns
    patterns = detect_patterns(df.iloc[-20:].copy())

    # --- Trend assessment ---
    bullish_signals = sum([
        rsi_cur > 50,
        macd_cur > macd_sig_cur,
        macd_hist_cur > 0,
        bb_pct > 0.5,
        stoch_k_cur > stoch_d_cur,
        stoch_k_cur > 50,
        above_ma20 is True,
        above_ma50 is True,
        above_ma200 is True,
        obv_trend == "rising",
    ])
    trend_score = bullish_signals / 10   # 0-1
    if trend_score >= 0.7:
        trend = "Strong Bullish"
    elif trend_score >= 0.55:
        trend = "Bullish"
    elif trend_score >= 0.45:
        trend = "Neutral"
    elif trend_score >= 0.30:
        trend = "Bearish"
    else:
        trend = "Strong Bearish"

    # --- Risk / reward ---
    # Simple ATR-based stop loss suggestion
    stop_loss  = round(price - 2 * atr_cur, 2)
    take_profit = round(price + 3 * atr_cur, 2)
    risk_reward = round((take_profit - price) / (price - stop_loss), 2) if price > stop_loss else None

    # --- Fundamental summary ---
    pe = fund.get("forwardPE") or fund.get("trailingPE")
    pb = fund.get("priceToBook")
    roe = fund.get("returnOnEquity")
    rev_growth = fund.get("revenueGrowth")

    # --- Overall recommendation ---
    rec_score = trend_score
    if pe and pe < 20:
        rec_score += 0.05
    if roe and roe > 0.15:
        rec_score += 0.05
    if rev_growth and rev_growth > 0.10:
        rec_score += 0.05
    if rsi_cur > 70:
        rec_score -= 0.10   # overbought penalty
    if rsi_cur < 30:
        rec_score += 0.05   # oversold bonus (mean reversion)

    if rec_score >= 0.70:
        recommendation = "BUY"
    elif rec_score >= 0.55:
        recommendation = "WEAK BUY"
    elif rec_score >= 0.40:
        recommendation = "HOLD"
    elif rec_score >= 0.25:
        recommendation = "WEAK SELL"
    else:
        recommendation = "SELL"

    return {
        "ticker": ticker,
        "price": round(price, 2),
        "trend": trend,
        "recommendation": recommendation,
        "rec_score": round(rec_score, 3),

        # Oscillators
        "rsi": round(rsi_cur, 2),
        "macd": round(macd_cur, 4),
        "macd_signal": round(macd_sig_cur, 4),
        "macd_histogram": round(macd_hist_cur, 4),
        "stochastic_k": round(stoch_k_cur, 2),
        "stochastic_d": round(stoch_d_cur, 2),

        # Volatility
        "atr": round(atr_cur, 2),
        "atr_pct": round(atr_pct, 2),
        "bb_position_pct": round(bb_pct * 100, 1),

        # Trend
        "above_ma20": above_ma20,
        "above_ma50": above_ma50,
        "above_ma200": above_ma200,
        "obv_trend": obv_trend,

        # Support / resistance
        "supports": sr["supports"],
        "resistances": sr["resistances"],

        # Patterns
        "patterns": patterns,

        # Risk / reward
        "suggested_stop_loss": stop_loss,
        "suggested_take_profit": take_profit,
        "risk_reward_ratio": risk_reward,

        # Fundamentals snapshot
        "pe_ratio": pe,
        "price_to_book": pb,
        "roe_pct": round(roe * 100, 2) if roe else None,
        "revenue_growth_pct": round(rev_growth * 100, 2) if rev_growth else None,
        "sector": fund.get("sector"),
        "analyst_recommendation": fund.get("recommendationKey"),
        "analyst_target": fund.get("targetMeanPrice"),
        "analyst_count": fund.get("numberOfAnalystOpinions"),
    }


def format_analysis_report(analysis: dict) -> str:
    """Return a human-readable report string from an analysis dict."""
    t = analysis.get("ticker", "?")
    lines = [
        f"{'=' * 60}",
        f"  ANALYSIS REPORT: {t}  ({analysis.get('trend', 'N/A')})",
        f"{'=' * 60}",
        f"  Price          : ${analysis.get('price', 'N/A')}",
        f"  Recommendation : {analysis.get('recommendation', 'N/A')} (score {analysis.get('rec_score', 'N/A')})",
        f"  Sector         : {analysis.get('sector', 'N/A')}",
        "",
        "  [ Technical Indicators ]",
        f"  RSI ({RSI_PERIOD})        : {analysis.get('rsi', 'N/A')}",
        f"  MACD            : {analysis.get('macd', 'N/A')}  Signal: {analysis.get('macd_signal', 'N/A')}",
        f"  Stochastic K/D  : {analysis.get('stochastic_k', 'N/A')} / {analysis.get('stochastic_d', 'N/A')}",
        f"  BB position     : {analysis.get('bb_position_pct', 'N/A')}%",
        f"  ATR (% of price): {analysis.get('atr_pct', 'N/A')}%",
        f"  OBV trend       : {analysis.get('obv_trend', 'N/A')}",
        f"  Above MA20/50/200: {analysis.get('above_ma20')} / {analysis.get('above_ma50')} / {analysis.get('above_ma200')}",
        "",
        "  [ Support & Resistance ]",
        f"  Support levels  : {analysis.get('supports', [])}",
        f"  Resistance lvls : {analysis.get('resistances', [])}",
        "",
        "  [ Candlestick Patterns ]",
        f"  {', '.join(analysis.get('patterns', [])) or 'None detected'}",
        "",
        "  [ Risk / Reward ]",
        f"  Stop loss       : ${analysis.get('suggested_stop_loss', 'N/A')}",
        f"  Take profit     : ${analysis.get('suggested_take_profit', 'N/A')}",
        f"  R/R ratio       : {analysis.get('risk_reward_ratio', 'N/A')}",
        "",
        "  [ Fundamentals ]",
        f"  PE ratio        : {analysis.get('pe_ratio', 'N/A')}",
        f"  Price/Book      : {analysis.get('price_to_book', 'N/A')}",
        f"  ROE             : {analysis.get('roe_pct', 'N/A')}%",
        f"  Revenue growth  : {analysis.get('revenue_growth_pct', 'N/A')}%",
        f"  Analyst rec     : {analysis.get('analyst_recommendation', 'N/A')}  target ${analysis.get('analyst_target', 'N/A')}  ({analysis.get('analyst_count', 'N/A')} analysts)",
        f"{'=' * 60}",
    ]
    return "\n".join(lines)
