"""
Alpha lab: 20+ intraday crypto alpha signals + IC computation.

Every signal takes a DataFrame of OHLCV bars and returns a float in [-1, +1].
Positive = bullish, negative = bearish, 0 = neutral.

IC (Information Coefficient) = Spearman correlation between signal and
forward return. Updated after every window using logged outcomes.

Signals
-------
Price-based:
  1.  momentum_1m        — 1-bar return
  2.  momentum_5m        — 5-bar return
  3.  momentum_15m       — 15-bar return
  4.  momentum_1h        — 60-bar return
  5.  mean_reversion     — Z-score of price vs 20-bar mean (inverted)
  6.  price_acceleration — Change in momentum (2nd derivative)
  7.  hl_position        — (Close - Low) / (High - Low), today's bar

Volatility:
  8.  volatility_ratio   — Short vol / long vol (low = calm = entry opp)
  9.  atr_pct            — ATR as % of price (higher = more volatile)
  10. bb_position        — Position within Bollinger Bands

Volume:
  11. volume_surge       — Volume vs 20-bar avg
  12. obv_momentum       — OBV slope over 5 bars
  13. vwap_deviation     — Price vs VWAP (negative = below = buy dip)

Oscillators:
  14. rsi_5             — RSI(5), scaled to [-1, +1]
  15. rsi_14            — RSI(14), scaled
  16. stoch_k           — Stochastic %K
  17. macd_signal       — MACD histogram sign & magnitude
  18. williams_r        — Williams %R

Structure:
  19. ema_cross         — EMA(5) vs EMA(20) gap
  20. support_proximity — Distance to nearest support level
  21. breakout_strength — Close vs recent N-bar high
  22. gap_fade          — Fade large gaps (mean reversion on open)
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(val):
    """Return 0 if NaN/inf."""
    if val is None:
        return 0.0
    try:
        v = float(val)
        return 0.0 if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return 0.0


def _clip(val, lo=-1.0, hi=1.0):
    return max(lo, min(hi, _safe(val)))


def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def _rsi(closes: pd.Series, n: int = 14) -> pd.Series:
    d  = closes.diff()
    g  = d.clip(lower=0).ewm(com=n-1, min_periods=n).mean()
    l  = (-d.clip(upper=0)).ewm(com=n-1, min_periods=n).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=n-1, min_periods=n).mean()


def _vwap(df: pd.DataFrame) -> pd.Series:
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    dv  = tp * df["Volume"]
    return dv.cumsum() / df["Volume"].cumsum().replace(0, np.nan)


# ── Individual signal functions ───────────────────────────────────────────────
# Each returns a single float for the LAST bar.

def sig_momentum_1m(df: pd.DataFrame) -> float:
    if len(df) < 2: return 0.0
    r = df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1
    return _clip(r / 0.01)   # normalise: 1% move → signal=1

def sig_momentum_5m(df: pd.DataFrame) -> float:
    if len(df) < 6: return 0.0
    r = df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1
    return _clip(r / 0.025)

def sig_momentum_15m(df: pd.DataFrame) -> float:
    if len(df) < 16: return 0.0
    r = df["Close"].iloc[-1] / df["Close"].iloc[-16] - 1
    return _clip(r / 0.05)

def sig_momentum_1h(df: pd.DataFrame) -> float:
    if len(df) < 61: return 0.0
    r = df["Close"].iloc[-1] / df["Close"].iloc[-61] - 1
    return _clip(r / 0.08)

def sig_mean_reversion(df: pd.DataFrame) -> float:
    if len(df) < 21: return 0.0
    c    = df["Close"]
    mean = c.iloc[-20:].mean()
    std  = c.iloc[-20:].std()
    if std == 0: return 0.0
    z = (c.iloc[-1] - mean) / std
    return _clip(-z / 2)   # inverted: below mean → positive signal

def sig_price_acceleration(df: pd.DataFrame) -> float:
    if len(df) < 4: return 0.0
    r1 = df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1
    r2 = df["Close"].iloc[-2] / df["Close"].iloc[-3] - 1
    return _clip((r1 - r2) / 0.005)

def sig_hl_position(df: pd.DataFrame) -> float:
    bar = df.iloc[-1]
    rng = bar["High"] - bar["Low"]
    if rng == 0: return 0.0
    pos = (bar["Close"] - bar["Low"]) / rng
    return _clip(pos * 2 - 1)   # 0→-1, 0.5→0, 1→+1

def sig_volatility_ratio(df: pd.DataFrame) -> float:
    if len(df) < 21: return 0.0
    rets = df["Close"].pct_change().dropna()
    v5   = rets.iloc[-5:].std()
    v20  = rets.iloc[-20:].std()
    if v20 == 0: return 0.0
    ratio = v5 / v20
    # Low short-vol = calmer market = slight buy bias
    return _clip(1 - ratio)

def sig_atr_pct(df: pd.DataFrame) -> float:
    if len(df) < 15: return 0.0
    atr = _atr(df).iloc[-1]
    pct = atr / df["Close"].iloc[-1]
    # Higher ATR = more volatile = higher signal (crypto likes volatility)
    return _clip(pct / 0.02)

def sig_bb_position(df: pd.DataFrame) -> float:
    if len(df) < 21: return 0.0
    c    = df["Close"]
    mid  = c.rolling(20).mean().iloc[-1]
    std  = c.rolling(20).std().iloc[-1]
    if std == 0: return 0.0
    bb   = (c.iloc[-1] - mid) / (2 * std)   # -1 at lower band, +1 at upper
    return _clip(bb)

def sig_volume_surge(df: pd.DataFrame) -> float:
    if len(df) < 21: return 0.0
    avg = df["Volume"].iloc[-20:].mean()
    if avg == 0: return 0.0
    ratio = df["Volume"].iloc[-1] / avg - 1
    return _clip(ratio / 3)   # 3× average → signal = 1

def sig_obv_momentum(df: pd.DataFrame) -> float:
    if len(df) < 7: return 0.0
    direction = np.sign(df["Close"].diff()).fillna(0)
    obv       = (direction * df["Volume"]).cumsum()
    slope     = (obv.iloc[-1] - obv.iloc[-6]) / (df["Volume"].iloc[-6:].mean() + 1e-9)
    return _clip(slope / 5)

def sig_vwap_deviation(df: pd.DataFrame) -> float:
    if len(df) < 5: return 0.0
    vwap  = _vwap(df).iloc[-1]
    price = df["Close"].iloc[-1]
    if vwap == 0: return 0.0
    dev   = (price - vwap) / vwap
    return _clip(-dev / 0.02)   # below VWAP → positive (buy dip)

def sig_rsi_5(df: pd.DataFrame) -> float:
    if len(df) < 7: return 0.0
    rsi = _rsi(df["Close"], 5).iloc[-1]
    return _clip((50 - rsi) / 50)   # oversold → positive

def sig_rsi_14(df: pd.DataFrame) -> float:
    if len(df) < 16: return 0.0
    rsi = _rsi(df["Close"], 14).iloc[-1]
    return _clip((50 - rsi) / 50)

def sig_stoch_k(df: pd.DataFrame) -> float:
    if len(df) < 15: return 0.0
    lo14 = df["Low"].rolling(14).min().iloc[-1]
    hi14 = df["High"].rolling(14).max().iloc[-1]
    if hi14 == lo14: return 0.0
    k = (df["Close"].iloc[-1] - lo14) / (hi14 - lo14) * 100
    return _clip((50 - k) / 50)

def sig_macd_signal(df: pd.DataFrame) -> float:
    if len(df) < 30: return 0.0
    c    = df["Close"]
    macd = _ema(c, 12) - _ema(c, 26)
    sig  = _ema(macd, 9)
    hist = macd.iloc[-1] - sig.iloc[-1]
    return _clip(hist / (df["Close"].iloc[-1] * 0.002 + 1e-9))

def sig_williams_r(df: pd.DataFrame) -> float:
    if len(df) < 15: return 0.0
    hi14 = df["High"].rolling(14).max().iloc[-1]
    lo14 = df["Low"].rolling(14).min().iloc[-1]
    if hi14 == lo14: return 0.0
    wr = (hi14 - df["Close"].iloc[-1]) / (hi14 - lo14) * -100
    return _clip((wr + 50) / 50)

def sig_ema_cross(df: pd.DataFrame) -> float:
    if len(df) < 22: return 0.0
    c    = df["Close"]
    e5   = _ema(c, 5).iloc[-1]
    e20  = _ema(c, 20).iloc[-1]
    gap  = (e5 - e20) / e20 if e20 != 0 else 0
    return _clip(gap / 0.01)

def sig_breakout_strength(df: pd.DataFrame) -> float:
    if len(df) < 21: return 0.0
    hi20  = df["High"].iloc[-20:].max()
    lo20  = df["Low"].iloc[-20:].min()
    price = df["Close"].iloc[-1]
    rng   = hi20 - lo20
    if rng == 0: return 0.0
    pos   = (price - lo20) / rng
    return _clip(pos * 2 - 1)

def sig_gap_fade(df: pd.DataFrame) -> float:
    """Fade large gaps: if price gapped up big, fade it (short bias)."""
    if len(df) < 2: return 0.0
    gap = (df["Open"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2]
    return _clip(-gap / 0.02)   # large up gap → negative signal (fade)

def sig_support_proximity(df: pd.DataFrame) -> float:
    """Buy when close to recent support, sell near resistance."""
    if len(df) < 21: return 0.0
    recent = df["Low"].iloc[-20:]
    support = recent.min()
    price   = df["Close"].iloc[-1]
    rng     = df["Close"].iloc[-20:].std()
    if rng == 0: return 0.0
    dist = (price - support) / rng
    return _clip(1 - dist / 2)   # closer to support → more bullish


# ── Signal registry ───────────────────────────────────────────────────────────

ALL_SIGNALS: dict[str, callable] = {
    "momentum_1m":       sig_momentum_1m,
    "momentum_5m":       sig_momentum_5m,
    "momentum_15m":      sig_momentum_15m,
    "momentum_1h":       sig_momentum_1h,
    "mean_reversion":    sig_mean_reversion,
    "price_accel":       sig_price_acceleration,
    "hl_position":       sig_hl_position,
    "volatility_ratio":  sig_volatility_ratio,
    "atr_pct":           sig_atr_pct,
    "bb_position":       sig_bb_position,
    "volume_surge":      sig_volume_surge,
    "obv_momentum":      sig_obv_momentum,
    "vwap_deviation":    sig_vwap_deviation,
    "rsi_5":             sig_rsi_5,
    "rsi_14":            sig_rsi_14,
    "stoch_k":           sig_stoch_k,
    "macd_signal":       sig_macd_signal,
    "williams_r":        sig_williams_r,
    "ema_cross":         sig_ema_cross,
    "breakout_strength": sig_breakout_strength,
    "gap_fade":          sig_gap_fade,
    "support_proximity": sig_support_proximity,
}

# Load any signals discovered by the optimizer at runtime
def _load_learned_signals():
    path = "data/learned_signals.py"
    try:
        import importlib.util, os
        if not os.path.exists(path):
            return
        spec = importlib.util.spec_from_file_location("learned_signals", path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        added = 0
        for attr in dir(mod):
            if attr.startswith("sig_") and callable(getattr(mod, attr)):
                name = attr[4:]   # strip "sig_" prefix
                if name not in ALL_SIGNALS:
                    ALL_SIGNALS[name] = getattr(mod, attr)
                    added += 1
        if added:
            logger.info("Loaded %d learned signal(s) from %s", added, path)
    except Exception as exc:
        logger.warning("Could not load learned signals: %s", exc)

_load_learned_signals()

# Momentum-biased defaults: crypto trends strongly, so weight momentum/breakout
# higher out of the box. The optimizer will refine these after ~1h of data.
DEFAULT_WEIGHTS = {
    "momentum_1m":       0.04,
    "momentum_5m":       0.09,
    "momentum_15m":      0.11,
    "momentum_1h":       0.09,
    "mean_reversion":    0.02,
    "price_accel":       0.06,
    "hl_position":       0.03,
    "volatility_ratio":  0.02,
    "atr_pct":           0.03,
    "bb_position":       0.02,
    "volume_surge":      0.07,
    "obv_momentum":      0.06,
    "vwap_deviation":    0.03,
    "rsi_5":             0.03,
    "rsi_14":            0.02,
    "stoch_k":           0.02,
    "macd_signal":       0.06,
    "williams_r":        0.02,
    "ema_cross":         0.07,
    "breakout_strength": 0.08,
    "gap_fade":          0.01,
    "support_proximity": 0.02,
}


# ── Compute all signals for one symbol at latest bar ─────────────────────────

def compute_signals(df: pd.DataFrame) -> dict[str, float]:
    """
    Compute all 22 signals for a symbol at the latest bar.
    Returns {signal_name: value} dict, all values in [-1, +1].
    """
    return {name: _safe(fn(df)) for name, fn in ALL_SIGNALS.items()}


def composite_score(signals: dict[str, float], weights: Optional[dict] = None) -> float:
    """
    Weighted average of signals → composite alpha score in [-1, +1].
    Positive = buy, negative = sell.
    """
    w = weights or DEFAULT_WEIGHTS
    total_w = sum(abs(w.get(s, 0)) for s in signals)
    if total_w == 0:
        return 0.0
    score = sum(signals.get(s, 0) * w.get(s, 0) for s in signals) / total_w
    return _clip(score)


# ── IC computation ─────────────────────────────────────────────────────────────

def compute_ic(signal_log: list[dict], horizon: str = "fwd_15m") -> dict[str, float]:
    """
    Compute Spearman IC for each signal vs forward returns.

    Parameters
    ----------
    signal_log : list of dicts with keys: signals, fwd_5m, fwd_15m, fwd_1h
    horizon    : "fwd_5m" | "fwd_15m" | "fwd_1h"

    Returns
    -------
    {signal_name: IC}  — Spearman rank correlation, range [-1, +1]
    """
    resolved = [r for r in signal_log if r.get(horizon) is not None]
    if len(resolved) < 10:
        return {}

    ic = {}
    for sig in ALL_SIGNALS:
        xs = [r["signals"].get(sig, 0) for r in resolved]
        ys = [r[horizon] for r in resolved]
        if len(xs) >= 10:
            try:
                corr, _ = scipy_stats.spearmanr(xs, ys)
                ic[sig] = round(float(corr), 4) if not np.isnan(corr) else 0.0
            except Exception:
                ic[sig] = 0.0
    return ic


def ic_to_weights(ic: dict[str, float], floor: float = 0.0) -> dict[str, float]:
    """
    Convert IC values to signal weights.
    Signals with negative IC get weight 0 (ignore them).
    Signals with positive IC get weight proportional to IC².
    """
    positive = {s: max(v - floor, 0) ** 2 for s, v in ic.items()}
    total    = sum(positive.values())
    if total == 0:
        return DEFAULT_WEIGHTS.copy()
    return {s: round(v / total, 6) for s, v in positive.items()}


def rank_symbols(
    bars: dict[str, pd.DataFrame],
    weights: Optional[dict] = None,
) -> list[tuple[str, float, dict]]:
    """
    Compute composite alpha for each symbol and return ranked list.

    Returns
    -------
    [(symbol, score, signals_dict), ...] sorted best-first
    """
    w   = weights or DEFAULT_WEIGHTS
    out = []
    for sym, df in bars.items():
        if len(df) < 30:
            continue
        sigs  = compute_signals(df)
        score = composite_score(sigs, w)
        out.append((sym, score, sigs))
    out.sort(key=lambda x: x[1], reverse=True)
    return out
