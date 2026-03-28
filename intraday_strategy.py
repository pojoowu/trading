"""
Intraday strategies for entry/exit timing.

These run AFTER the daily agent picks which stocks to buy.
They decide the optimal intraday moment to enter/exit each position.

Three strategies (each independently backtestable):
─────────────────────────────────────────────────
1. VWAP Reversion
   Buy when price drops below VWAP + RSI(5) oversold on hourly bars.
   Exit when price recrosses VWAP or stop-loss hit.

2. Opening Range Breakout (ORB)
   Record the high/low of the first N hours after open.
   Enter on breakout above range high (long) with volume confirmation.
   Stop at range low.

3. Hourly Momentum
   Enter when hourly close > VWAP AND hourly return > threshold.
   Ride momentum with a trailing stop.

Each strategy returns a signal Series (+1=long, 0=flat) on hourly bars.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# ── Shared technical helpers ──────────────────────────────────────────────────

def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Rolling intraday VWAP, reset at each calendar day.
    VWAP = cumulative(price * volume) / cumulative(volume)
    """
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    dollar_vol = typical * df["Volume"]

    vwap_vals = []
    for date, group in df.groupby(df.index.date):
        cum_dv  = dollar_vol.loc[group.index].cumsum()
        cum_vol = df["Volume"].loc[group.index].cumsum()
        vwap_vals.append(cum_dv / cum_vol.replace(0, np.nan))

    return pd.concat(vwap_vals).reindex(df.index)


def calc_rsi_intraday(closes: pd.Series, period: int = 5) -> pd.Series:
    delta    = closes.diff()
    gain     = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss     = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs       = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_atr_intraday(df: pd.DataFrame, period: int = 10) -> pd.Series:
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


# ── Strategy dataclasses ──────────────────────────────────────────────────────

@dataclass
class VWAPReversionParams:
    rsi_period:       int   = 5
    rsi_oversold:     float = 35.0
    rsi_overbought:   float = 65.0
    vwap_entry_pct:   float = 0.002   # must be this % below VWAP to enter
    stop_loss_pct:    float = 0.015   # 1.5% hard stop
    take_profit_pct:  float = 0.030   # 3% take profit


@dataclass
class ORBParams:
    range_hours:      int   = 1       # first N hours define the range
    volume_mult:      float = 1.3     # breakout bar volume must be > N×avg
    stop_loss_pct:    float = 0.020
    take_profit_pct:  float = 0.040
    max_entry_hour:   int   = 14      # don't enter after 2pm ET


@dataclass
class HourlyMomentumParams:
    momentum_threshold: float = 0.005  # hourly return must exceed this
    rsi_min:            float = 45.0   # not overbought
    rsi_max:            float = 70.0
    trailing_stop_pct:  float = 0.020
    take_profit_pct:    float = 0.040
    volume_mult:        float = 1.2    # volume confirmation


# ── Strategy 1: VWAP Reversion ────────────────────────────────────────────────

def vwap_reversion_signals(
    df: pd.DataFrame,
    params: Optional[VWAPReversionParams] = None,
) -> pd.Series:
    """
    Returns a signal Series: +1 = enter long, 0 = stay flat.
    Entry: price drops > vwap_entry_pct below VWAP AND RSI oversold.
    Exit handled in backtester via stop/take-profit.
    """
    if params is None:
        params = VWAPReversionParams()

    if len(df) < 20:
        return pd.Series(0, index=df.index)

    vwap    = calc_vwap(df)
    rsi     = calc_rsi_intraday(df["Close"], params.rsi_period)
    price   = df["Close"]

    # Entry: price meaningfully below VWAP + oversold RSI
    below_vwap = (vwap - price) / vwap > params.vwap_entry_pct
    rsi_low    = rsi < params.rsi_oversold

    # Only trade during regular hours (9:30–15:30 ET)
    in_hours = (df.index.hour >= 9) & (
        (df.index.hour < 15) | ((df.index.hour == 9) & (df.index.minute >= 30))
    )

    signal = (below_vwap & rsi_low & in_hours).astype(int)
    return signal


# ── Strategy 2: Opening Range Breakout ───────────────────────────────────────

def orb_signals(
    df: pd.DataFrame,
    params: Optional[ORBParams] = None,
) -> pd.Series:
    """
    Returns signal Series: +1 = enter long on ORB breakout.
    Range is defined by the first `range_hours` bars each day.
    """
    if params is None:
        params = ORBParams()

    if len(df) < 10:
        return pd.Series(0, index=df.index)

    signal    = pd.Series(0, index=df.index)
    avg_vol   = df["Volume"].rolling(20).mean()

    for date, day_df in df.groupby(df.index.date):
        if len(day_df) < params.range_hours + 2:
            continue

        # Opening range: first N hourly bars
        opening   = day_df.iloc[: params.range_hours]
        range_hi  = opening["High"].max()
        range_lo  = opening["Low"].min()

        # Rest of day: look for breakout
        rest = day_df.iloc[params.range_hours :]
        for ts, bar in rest.iterrows():
            if ts.hour >= params.max_entry_hour:
                break
            vol_ok  = bar["Volume"] > avg_vol.get(ts, 0) * params.volume_mult
            breakout = bar["Close"] > range_hi and vol_ok
            if breakout:
                signal[ts] = 1
                break   # only one entry per day

    return signal


# ── Strategy 3: Hourly Momentum ───────────────────────────────────────────────

def hourly_momentum_signals(
    df: pd.DataFrame,
    params: Optional[HourlyMomentumParams] = None,
) -> pd.Series:
    """
    Returns signal Series: +1 = enter long on hourly momentum confirmation.
    Entry when: hourly return > threshold AND price > VWAP AND RSI in range.
    """
    if params is None:
        params = HourlyMomentumParams()

    if len(df) < 20:
        return pd.Series(0, index=df.index)

    vwap     = calc_vwap(df)
    rsi      = calc_rsi_intraday(df["Close"], 5)
    hourly_ret = df["Close"].pct_change()
    avg_vol  = df["Volume"].rolling(20).mean()

    above_vwap   = df["Close"] > vwap
    strong_move  = hourly_ret > params.momentum_threshold
    rsi_ok       = (rsi > params.rsi_min) & (rsi < params.rsi_max)
    vol_ok       = df["Volume"] > avg_vol * params.volume_mult
    in_hours     = (df.index.hour >= 10) & (df.index.hour < 15)

    signal = (above_vwap & strong_move & rsi_ok & vol_ok & in_hours).astype(int)
    return signal


# ── Combined entry signal ─────────────────────────────────────────────────────

def get_entry_signal(
    df: pd.DataFrame,
    strategy: str = "vwap_reversion",
    params: Optional[dict] = None,
) -> pd.Series:
    """
    Unified entry point. Returns signal Series for the chosen strategy.

    strategy: "vwap_reversion" | "orb" | "hourly_momentum"
    params:   dict of parameter overrides (matched to the strategy's dataclass)
    """
    if strategy == "vwap_reversion":
        p = VWAPReversionParams(**(params or {}))
        return vwap_reversion_signals(df, p)

    elif strategy == "orb":
        p = ORBParams(**(params or {}))
        return orb_signals(df, p)

    elif strategy == "hourly_momentum":
        p = HourlyMomentumParams(**(params or {}))
        return hourly_momentum_signals(df, p)

    else:
        raise ValueError(f"Unknown intraday strategy: {strategy}")


# ── Current bar recommendation (used by live agent) ──────────────────────────

def intraday_entry_now(
    df: pd.DataFrame,
    strategy: str = "vwap_reversion",
    params: Optional[dict] = None,
) -> dict:
    """
    Check if the latest bar has an entry signal.
    Returns dict with signal, price, reasoning.
    """
    signal = get_entry_signal(df, strategy, params)
    latest = signal.iloc[-1]
    price  = df["Close"].iloc[-1]
    vwap   = calc_vwap(df).iloc[-1]
    rsi    = calc_rsi_intraday(df["Close"]).iloc[-1]

    return {
        "strategy":      strategy,
        "signal":        int(latest),
        "enter_now":     bool(latest == 1),
        "price":         round(float(price), 2),
        "vwap":          round(float(vwap), 2) if not np.isnan(vwap) else None,
        "rsi":           round(float(rsi), 2) if not np.isnan(rsi) else None,
        "price_vs_vwap": round((price / vwap - 1) * 100, 3) if not np.isnan(vwap) else None,
        "bar_time":      str(df.index[-1]),
    }
