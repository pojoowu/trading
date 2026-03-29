"""
alpha_researcher.py — discovers better signal parameters and new signals.

Three jobs:
1. Parameter search: re-tests each signal family with different lookback
   periods / thresholds and picks the variant with the best IC.

2. Walk-forward backtest: splits data into train/test windows so the
   Sharpe estimate is out-of-sample, not in-sample.

3. New signal testing: safely exec()s a Python function proposed by
   Claude, computes its IC, and returns whether it's worth keeping.
"""
import logging
import traceback
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _clip(v, lo=-1.0, hi=1.0):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return 0.0
    return max(lo, min(hi, float(v)))

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _rsi(closes: pd.Series, n: int) -> pd.Series:
    d = closes.diff()
    g = d.clip(lower=0).ewm(com=n-1, min_periods=n).mean()
    l = (-d.clip(upper=0)).ewm(com=n-1, min_periods=n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _spearman_ic(xs, ys):
    if len(xs) < 10:
        return 0.0, 1.0
    try:
        corr, pval = scipy_stats.spearmanr(xs, ys)
        return (float(corr) if not np.isnan(corr) else 0.0), float(pval)
    except Exception:
        return 0.0, 1.0


# ── 1. Parameter grid search ──────────────────────────────────────────────────

def _compute_variants(df: pd.DataFrame) -> dict[str, float]:
    """
    Compute signal values for all parameter variants on the last bar of df.
    Returns {variant_name: value}.
    """
    if len(df) < 65:
        return {}
    c = df["Close"]
    v = df["Volume"]
    h = df["High"]
    lo = df["Low"]
    results = {}

    # RSI variants
    for n in [5, 7, 10, 14, 21]:
        try:
            val = _rsi(c, n).iloc[-1]
            results[f"rsi_{n}"] = _clip((50 - val) / 50)
        except Exception:
            pass

    # Momentum variants (return over N bars, normalised by sqrt(N))
    for n in [1, 3, 5, 10, 15, 30, 60]:
        if len(df) > n:
            try:
                ret = c.iloc[-1] / c.iloc[-(n+1)] - 1
                norm = 0.01 * (n ** 0.5)
                results[f"mom_{n}m"] = _clip(ret / norm)
            except Exception:
                pass

    # EMA cross variants (fast, slow)
    for fast, slow in [(3, 10), (5, 15), (5, 20), (8, 21), (10, 30)]:
        if len(df) > slow + 2:
            try:
                e_fast = _ema(c, fast).iloc[-1]
                e_slow = _ema(c, slow).iloc[-1]
                gap = (e_fast - e_slow) / e_slow if e_slow != 0 else 0
                results[f"ema_{fast}_{slow}"] = _clip(gap / 0.01)
            except Exception:
                pass

    # Bollinger variants (period, std multiplier)
    for n, k in [(10, 2.0), (15, 2.0), (20, 1.5), (20, 2.0), (20, 2.5), (30, 2.0)]:
        if len(df) > n + 2:
            try:
                mid = c.rolling(n).mean().iloc[-1]
                std = c.rolling(n).std().iloc[-1]
                if std > 0:
                    results[f"bb_{n}_{k}"] = _clip((c.iloc[-1] - mid) / (k * std))
            except Exception:
                pass

    # MACD variants (fast_ema, slow_ema, signal_ema)
    for f, s, sig in [(8, 17, 9), (12, 26, 9), (5, 35, 5), (3, 10, 5)]:
        if len(df) > s + sig + 2:
            try:
                macd = _ema(c, f) - _ema(c, s)
                hist = (macd - _ema(macd, sig)).iloc[-1]
                results[f"macd_{f}_{s}"] = _clip(hist / (c.iloc[-1] * 0.002 + 1e-9))
            except Exception:
                pass

    # Volume surge variants
    for n in [10, 20, 30]:
        if len(df) > n + 2:
            try:
                avg = v.iloc[-n:].mean()
                if avg > 0:
                    results[f"vol_surge_{n}"] = _clip((v.iloc[-1] / avg - 1) / 3)
            except Exception:
                pass

    # Mean reversion variants
    for n in [10, 15, 20, 30]:
        if len(df) > n + 2:
            try:
                mean = c.iloc[-n:].mean()
                std  = c.iloc[-n:].std()
                if std > 0:
                    results[f"mean_rev_{n}"] = _clip(-(c.iloc[-1] - mean) / (2 * std))
            except Exception:
                pass

    # Stochastic variants
    for n in [9, 14, 21]:
        if len(df) > n + 2:
            try:
                lo_n = lo.rolling(n).min().iloc[-1]
                hi_n = h.rolling(n).max().iloc[-1]
                if hi_n != lo_n:
                    k = (c.iloc[-1] - lo_n) / (hi_n - lo_n) * 100
                    results[f"stoch_{n}"] = _clip((50 - k) / 50)
            except Exception:
                pass

    return results


def param_search(
    bars: dict[str, pd.DataFrame],
    n_fwd: int = 15,
) -> dict:
    """
    For each signal variant, compute Spearman IC vs n_fwd-bar forward return
    across all symbols and bars. Returns ranked variants.

    Parameters
    ----------
    bars   : {symbol: OHLCV DataFrame}
    n_fwd  : forward return horizon in bars

    Returns
    -------
    {variant_name: {"ic": float, "pval": float, "n": int}}
    sorted by abs(IC) descending
    """
    logger.info("Running parameter search across %d symbols…", len(bars))
    signal_xs: dict[str, list] = {}
    forward_ys: list = []

    for sym, df in bars.items():
        if len(df) < 100:
            continue
        closes = df["Close"].values
        for i in range(60, len(df) - n_fwd):
            slice_df = df.iloc[:i+1].copy()
            fwd_ret  = closes[i + n_fwd] / closes[i] - 1
            variants = _compute_variants(slice_df)
            if not variants:
                continue
            for name, val in variants.items():
                signal_xs.setdefault(name, []).append(val)
            # We need the same length for all, so only append if we got signals
            if variants:
                forward_ys.append(fwd_ret)
                # Pad missing signals with 0
                for name in signal_xs:
                    if len(signal_xs[name]) < len(forward_ys):
                        signal_xs[name].append(0.0)

    results = {}
    for name, xs in signal_xs.items():
        ys = forward_ys[:len(xs)]
        if len(xs) < 20:
            continue
        ic, pval = _spearman_ic(xs, ys)
        results[name] = {"ic": round(ic, 4), "pval": round(pval, 4), "n": len(xs)}

    # Sort by abs IC
    sorted_results = dict(sorted(results.items(), key=lambda x: abs(x[1]["ic"]), reverse=True))
    logger.info("Param search done. Top 5: %s",
                [(k, v["ic"]) for k, v in list(sorted_results.items())[:5]])
    return sorted_results


# ── 2. Walk-forward backtest ──────────────────────────────────────────────────

def walk_forward_backtest(
    bars: dict[str, pd.DataFrame],
    weights: dict,
    params: dict,
    n_folds: int = 3,
) -> dict:
    """
    Split bars into n_folds chunks and backtest on each fold independently
    (the signal uses only bars up to that point, never future data).

    Returns average and std of Sharpe, return, win_rate across folds.
    """
    from alpha_lab import rank_symbols

    # Trim all symbols to the same length
    min_len = min(len(df) for df in bars.values()) if bars else 0
    if min_len < 120:
        return {}

    trimmed = {s: df.iloc[-min_len:].reset_index(drop=True) for s, df in bars.items()}
    fold_size = min_len // n_folds
    if fold_size < 60:
        return {}

    fold_results = []
    for fold in range(n_folds):
        start = fold * fold_size
        end   = start + fold_size
        fold_bars = {s: df.iloc[start:end].copy() for s, df in trimmed.items()}

        trades = []
        cash   = params.get("initial_cash", 10_000.0)
        equity = [cash]
        open_p = {}  # {sym: {entry_px, entry_bar, stop, target}}

        n_ticks = fold_size
        for i in range(30, n_ticks):
            slices = {s: df.iloc[:i] for s, df in fold_bars.items()}
            prices = {s: float(df["Close"].iloc[-1]) for s, df in slices.items() if len(df) > 0}

            port = cash + sum(p["qty"] * prices.get(s, p["entry_px"]) for s, p in open_p.items())
            equity.append(port)

            for sym in list(open_p.keys()):
                pos   = open_p[sym]
                price = prices.get(sym, pos["entry_px"])
                hold  = i - pos["entry_bar"]
                if (price <= pos["stop"] or price >= pos["target"]
                        or hold >= params.get("max_hold_minutes", 120)):
                    pnl = price / pos["entry_px"] - 1
                    cash += pos["qty"] * price
                    del open_p[sym]
                    trades.append(pnl)

            if len(open_p) < params.get("max_positions", 5):
                try:
                    ranked = rank_symbols(slices, weights)
                    for sym, score, _ in ranked:
                        if len(open_p) >= params.get("max_positions", 5): break
                        if sym in open_p: continue
                        if score < params.get("entry_threshold", 0.05): break
                        price = prices.get(sym, 0)
                        if price <= 0: continue
                        invest = port * params.get("position_size_pct", 0.18)
                        if invest > cash * 0.99: continue
                        qty = invest / price
                        cash -= invest
                        open_p[sym] = {
                            "qty": qty, "entry_px": price, "entry_bar": i,
                            "stop":   price * (1 - params.get("stop_loss_pct", 0.015)),
                            "target": price * (1 + params.get("take_profit_pct", 0.03)),
                        }
                except Exception:
                    pass

        if len(equity) < 2 or not trades:
            continue

        eq  = pd.Series(equity)
        ret = eq.pct_change().dropna()
        sharpe    = ret.mean() / ret.std() * (60 * 24 * 365) ** 0.5 if ret.std() > 0 else 0
        total_ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
        win_rate  = sum(1 for t in trades if t > 0) / len(trades) * 100
        fold_results.append({
            "sharpe": sharpe, "total_return_pct": total_ret,
            "win_rate_pct": win_rate, "n_trades": len(trades),
        })

    if not fold_results:
        return {}

    def _avg(key): return round(sum(r[key] for r in fold_results) / len(fold_results), 3)
    def _std(key):
        vals = [r[key] for r in fold_results]
        mean = sum(vals) / len(vals)
        return round((sum((v - mean)**2 for v in vals) / len(vals)) ** 0.5, 3)

    return {
        "sharpe_mean":      _avg("sharpe"),
        "sharpe_std":       _std("sharpe"),
        "return_mean_pct":  _avg("total_return_pct"),
        "win_rate_mean":    _avg("win_rate_pct"),
        "n_trades_mean":    _avg("n_trades"),
        "n_folds":          len(fold_results),
        "folds":            fold_results,
    }


# ── 3. New signal testing ─────────────────────────────────────────────────────

def test_new_signal(
    name: str,
    code: str,
    bars: dict[str, pd.DataFrame],
    n_fwd: int = 15,
) -> dict:
    """
    Safely test a new signal function proposed by Claude.

    Parameters
    ----------
    name : signal name, e.g. "btc_relative_strength"
    code : Python source of a function:
           def sig_NAME(df: pd.DataFrame) -> float: ...
    bars : {symbol: OHLCV DataFrame}

    Returns
    -------
    {"ic": float, "pval": float, "n": int, "valid": bool, "error": str}
    """
    fn_name = f"sig_{name}"

    # ── Safe exec environment: only numpy + pandas allowed ───────────────────
    safe_globals = {
        "np": np, "pd": pd,
        "__builtins__": {
            "abs": abs, "min": min, "max": max, "len": len,
            "range": range, "enumerate": enumerate, "zip": zip,
            "float": float, "int": int, "bool": bool, "list": list,
            "sum": sum, "round": round, "print": print,
        },
    }
    try:
        exec(compile(code, "<proposed_signal>", "exec"), safe_globals)
    except Exception as e:
        return {"ic": 0.0, "pval": 1.0, "n": 0, "valid": False,
                "error": f"compile/exec error: {e}"}

    fn = safe_globals.get(fn_name)
    if fn is None:
        return {"ic": 0.0, "pval": 1.0, "n": 0, "valid": False,
                "error": f"function '{fn_name}' not found in code"}

    # ── Compute signal + forward return across bars ───────────────────────────
    xs, ys = [], []
    for sym, df in bars.items():
        if len(df) < 60 + n_fwd:
            continue
        closes = df["Close"].values
        for i in range(50, len(df) - n_fwd, 3):  # step=3 for speed
            try:
                val = fn(df.iloc[:i+1])
                if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                    continue
                val = float(np.clip(val, -1.0, 1.0))
                fwd = closes[i + n_fwd] / closes[i] - 1
                xs.append(val)
                ys.append(fwd)
            except Exception:
                pass

    if len(xs) < 20:
        return {"ic": 0.0, "pval": 1.0, "n": len(xs), "valid": False,
                "error": f"only {len(xs)} valid observations"}

    ic, pval = _spearman_ic(xs, ys)
    logger.info("New signal '%s': IC=%.4f  p=%.4f  n=%d", name, ic, pval, len(xs))

    return {
        "ic":    round(ic, 4),
        "pval":  round(pval, 4),
        "n":     len(xs),
        "valid": True,
        "error": "",
    }


def save_learned_signal(name: str, code: str, ic: float):
    """Append a validated new signal to data/learned_signals.py."""
    import os
    os.makedirs("data", exist_ok=True)
    path = "data/learned_signals.py"

    header = (
        "# Auto-generated by alpha_researcher. Do not edit manually.\n"
        "# Each function follows the same API as alpha_lab.py signals:\n"
        "#   def sig_NAME(df: pd.DataFrame) -> float  in [-1, +1]\n\n"
        "import numpy as np\nimport pandas as pd\n\n"
    )
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(header)

    with open(path, "a") as f:
        f.write(f"\n# IC={ic:.4f} (discovered by optimizer)\n")
        f.write(code.strip() + "\n")

    logger.info("Saved new signal '%s' (IC=%.4f) to %s", name, ic, path)
