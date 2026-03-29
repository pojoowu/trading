"""
Crypto optimizer: runs every 4 hours, uses accumulated signal logs to:
  1. Compute IC (Spearman correlation) for each of the 22 signals
  2. Reweight signals proportional to IC² (kill weak signals, boost strong ones)
  3. Tune entry/exit thresholds and position sizing via backtest on recent 1h bars
  4. Ask Claude to review the IC table and approve / override the new weights
  5. Save weights to data/signal_weights.json for the minute trader to pick up live

Self-improvement timeline with crypto 1-min bars:
  Hour 0  : trader starts, logs signals every minute
  Hour 1  : ~60 bars × 20 coins = 1,200 signal observations
             fwd_5m and fwd_15m filled → first IC computable
  Hour 4  : optimizer runs with ~240 bars × 20 coins = 4,800 observations
             Solid IC estimates → weights updated → trader improves
  Hour 8  : second optimizer run with even more data
  Day 1   : 1,440 bars × 20 coins = 28,800 observations — very reliable IC
"""
import json
import logging
import os
import random
import copy
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import anthropic

from alpha_lab import ALL_SIGNALS, compute_ic, ic_to_weights, DEFAULT_WEIGHTS, rank_symbols
from crypto_data import batch_history, DEFAULT_UNIVERSE

logger = logging.getLogger(__name__)

SIGNAL_LOG_FILE   = "data/crypto_signal_log.jsonl"
SIGNAL_WEIGHTS_FILE = "data/signal_weights.json"
TRADER_PARAMS_FILE  = "data/intraday_params.json"
OPT_HISTORY_FILE    = "data/optimizer_history.jsonl"

# ── Load signal log ───────────────────────────────────────────────────────────

def load_signal_log(n: int = 5000) -> list[dict]:
    if not os.path.exists(SIGNAL_LOG_FILE):
        return []
    records = []
    with open(SIGNAL_LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records[-n:]


def load_equity_log(n: int = 500) -> list[dict]:
    path = "data/crypto_equity.jsonl"
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records[-n:]


# ── Current weights ───────────────────────────────────────────────────────────

def load_weights() -> dict:
    if os.path.exists(SIGNAL_WEIGHTS_FILE):
        try:
            return json.load(open(SIGNAL_WEIGHTS_FILE))
        except Exception:
            pass
    return dict(DEFAULT_WEIGHTS)


def save_weights(weights: dict, reason: str = ""):
    os.makedirs("data", exist_ok=True)
    record = {
        "weights":    weights,
        "reason":     reason,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(SIGNAL_WEIGHTS_FILE, "w") as f:
        json.dump(record, f, indent=2)
    logger.info("Signal weights saved. Reason: %s", reason)


# ── Trader params ─────────────────────────────────────────────────────────────

def load_trader_params() -> dict:
    from minute_trader import DEFAULT_TRADER_PARAMS
    p = dict(DEFAULT_TRADER_PARAMS)
    if os.path.exists(TRADER_PARAMS_FILE):
        try:
            p.update(json.load(open(TRADER_PARAMS_FILE)))
        except Exception:
            pass
    return p


def save_trader_params(params: dict):
    os.makedirs("data", exist_ok=True)
    with open(TRADER_PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)


# ── Quick backtest of param candidates ───────────────────────────────────────

def _quick_backtest(
    bars_1m: dict,
    weights: dict,
    params: dict,
    n_bars: int = 500,
) -> dict:
    """
    Simulate the minute trader on recent bars with given weights + params.
    Returns {sharpe, total_return_pct, win_rate, avg_hold_minutes, n_trades}.
    """
    trades  = []
    equity  = [params.get("initial_cash", 10_000)]
    cash    = equity[0]
    open_positions = {}  # {sym: {entry_px, entry_bar, stop, target}}

    symbols = list(bars_1m.keys())

    # Use last n_bars only for speed
    trimmed = {s: df.iloc[-n_bars:].copy() for s, df in bars_1m.items() if len(df) >= n_bars}
    if not trimmed:
        return {}

    n_ticks = min(len(df) for df in trimmed.values())

    for i in range(60, n_ticks):
        # Slice up to bar i
        slices = {s: df.iloc[:i] for s, df in trimmed.items()}
        prices = {s: float(df["Close"].iloc[-1]) for s, df in slices.items()}

        # Update equity
        port_val = cash + sum(
            p["qty"] * prices.get(s, p["entry_px"])
            for s, p in open_positions.items()
        )
        equity.append(port_val)

        # Check exits
        for sym in list(open_positions.keys()):
            pos   = open_positions[sym]
            price = prices.get(sym, pos["entry_px"])
            hold  = i - pos["entry_bar"]
            reason = None
            if price <= pos["stop"]:    reason = "stop"
            elif price >= pos["target"]: reason = "tp"
            elif hold >= params.get("max_hold_minutes", 120): reason = "timeout"
            if reason:
                pnl_pct = price / pos["entry_px"] - 1
                cash   += pos["qty"] * price
                del open_positions[sym]
                trades.append(pnl_pct)

        # Enter new positions
        if len(open_positions) < params.get("max_positions", 5):
            ranked = rank_symbols(slices, weights)
            for sym, score, _ in ranked:
                if len(open_positions) >= params.get("max_positions", 5): break
                if sym in open_positions: continue
                if score < params.get("entry_threshold", 0.15): break
                price = prices.get(sym, 0)
                if price <= 0: continue
                invest = port_val * params.get("position_size_pct", 0.18)
                if invest > cash * 0.99: continue
                qty    = invest / price
                cash  -= invest
                open_positions[sym] = {
                    "qty":       qty,
                    "entry_px":  price,
                    "entry_bar": i,
                    "stop":      price * (1 - params.get("stop_loss_pct", 0.015)),
                    "target":    price * (1 + params.get("take_profit_pct", 0.030)),
                }

    if len(equity) < 2 or not trades:
        return {}

    import pandas as pd
    eq  = pd.Series(equity)
    ret = eq.pct_change().dropna()
    sharpe = ret.mean() / ret.std() * (60 * 24 * 365) ** 0.5 if ret.std() > 0 else 0  # annualised
    total_ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    win_rate  = sum(1 for t in trades if t > 0) / len(trades) * 100

    return {
        "sharpe":           round(sharpe, 3),
        "total_return_pct": round(total_ret, 3),
        "win_rate_pct":     round(win_rate, 2),
        "n_trades":         len(trades),
    }


# ── Claude meta-agent ─────────────────────────────────────────────────────────

OPTIMIZER_SYSTEM = """You are a quantitative crypto trading strategy optimizer.

You receive:
1. IC table: Spearman correlation of each signal with 15-min forward returns
2. IC-derived suggested weights (proportional to IC²)
3. Current weights being used
4. Backtest results comparing current vs suggested weights
5. Recent portfolio performance (equity curve stats)

Your task:
- Approve or modify the suggested weights
- Note which signals to boost (high positive IC) and which to zero out (negative/zero IC)
- Suggest any parameter changes (stop loss, take profit, entry threshold, max positions)
- Be conservative: don't make extreme changes all at once
- Explain your reasoning concisely

You MUST call update_strategy with your final decisions.
"""

OPTIMIZER_TOOLS = [{
    "name": "update_strategy",
    "description": "Save updated signal weights and trader parameters.",
    "input_schema": {
        "type": "object",
        "properties": {
            "signal_weights": {
                "type": "object",
                "description": "{signal_name: weight} — weights will be normalised to sum to 1"
            },
            "trader_params": {
                "type": "object",
                "description": (
                    "Optional parameter overrides — any subset of the following:\n"
                    "SIZING: position_size_pct (base %, e.g. 0.18), "
                    "min_position_pct (floor, e.g. 0.05), max_position_pct (ceiling, e.g. 0.25), "
                    "size_by_score (bool), score_factor_min (e.g. 0.5), score_factor_max (e.g. 1.5), "
                    "size_by_vol (bool), vol_factor_min (e.g. 0.3), vol_factor_max (e.g. 2.0), "
                    "vol_target_atr (baseline ATR, e.g. 0.015).\n"
                    "RISK PER TRADE: stop_loss_pct, take_profit_pct, trailing_stop_pct (0=off), "
                    "partial_tp_pct (0=off, e.g. 0.015), partial_tp_size (e.g. 0.5).\n"
                    "ENTRY/EXIT: entry_threshold, exit_threshold, max_positions, "
                    "confirm_ticks (1=immediate), cooldown_minutes (after loss exit), "
                    "max_hold_minutes.\n"
                    "REGIME: regime_filter (bool), regime_ema_bars (e.g. 20), "
                    "regime_threshold (e.g. -0.005), regime_size_penalty (e.g. 0.5).\n"
                    "CIRCUIT BREAKER: max_daily_loss_pct (e.g. 0.05).\n"
                    "IC TUNING: ic_blend_15m (e.g. 0.6), ic_blend_5m (e.g. 0.4), "
                    "ic_floor (min IC for non-zero weight, e.g. 0.02)."
                )
            },
            "reason": {
                "type": "string",
                "description": "2-4 sentence explanation of changes"
            },
        },
        "required": ["signal_weights", "reason"],
    },
}]


def run_crypto_optimizer(dry_run: bool = False) -> dict:
    """
    Run one full crypto optimization cycle.
    Returns dict with new_weights, ic_table, backtest_results, reason.
    """
    from config import ANTHROPIC_API_KEY, CLAUDE_MODEL
    logger.info("=" * 55)
    logger.info("CRYPTO OPTIMIZER  %s", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    logger.info("=" * 55)

    # ── 1. Load signal log & compute IC ──────────────────────────────────────
    signal_log = load_signal_log(5000)
    resolved   = [r for r in signal_log if r.get("fwd_15m") is not None]
    logger.info("Signal log: %d total, %d resolved (fwd_15m)", len(signal_log), len(resolved))

    if len(resolved) < 20:
        logger.warning("Not enough resolved signals (%d < 20). Skipping optimization.", len(resolved))
        return {"skipped": True, "reason": f"only {len(resolved)} resolved signals"}

    ic_15m = compute_ic(resolved, "fwd_15m")
    ic_5m  = compute_ic(resolved, "fwd_5m")

    # Load current params and weights first (needed for IC blend settings)
    current_weights = load_weights()
    if isinstance(current_weights, dict) and "weights" in current_weights:
        current_weights = current_weights["weights"]
    current_params = load_trader_params()

    # IC horizon blend (weights tunable via params)
    ic_15m_w = current_params.get("ic_blend_15m", 0.6)
    ic_5m_w  = current_params.get("ic_blend_5m",  0.4)
    ic_floor = current_params.get("ic_floor", 0.02)

    blended_ic = {}
    for sig in ALL_SIGNALS:
        blended_ic[sig] = round(
            ic_15m_w * ic_15m.get(sig, 0) + ic_5m_w * ic_5m.get(sig, 0), 4
        )

    suggested_weights = ic_to_weights(blended_ic, floor=ic_floor)

    # ── 2. Quick backtest: current vs suggested weights ───────────────────────
    logger.info("Fetching bars for backtest…")
    bars = batch_history(DEFAULT_UNIVERSE[:10], interval="1m", days=1)

    bt_current   = _quick_backtest(bars, current_weights,   current_params) if bars else {}
    bt_suggested = _quick_backtest(bars, suggested_weights, current_params) if bars else {}

    # ── 3. Equity curve stats ─────────────────────────────────────────────────
    equity_log = load_equity_log(200)
    eq_stats   = {}
    if len(equity_log) > 5:
        equities  = [e["equity"] for e in equity_log]
        total_ret = (equities[-1] / equities[0] - 1) * 100
        eq_stats  = {
            "n_snapshots":    len(equities),
            "start_equity":   round(equities[0],  2),
            "current_equity": round(equities[-1], 2),
            "total_return_pct": round(total_ret,  3),
            "current_positions": equity_log[-1].get("n_positions", 0),
        }

    # ── 4. Build IC table for Claude ─────────────────────────────────────────
    ic_table = sorted(blended_ic.items(), key=lambda x: abs(x[1]), reverse=True)
    ic_str   = "\n".join(
        f"  {sig:<22} IC={v:+.4f}  {'✓ predictive' if v > 0.05 else ('✗ inverse' if v < -0.05 else '~ noise')}"
        for sig, v in ic_table
    )

    # ── 5. Ask Claude ─────────────────────────────────────────────────────────
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = (
        f"Crypto optimizer run at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"Resolved signals used: {len(resolved)}\n\n"
        f"IC TABLE (Spearman corr with 15-min forward returns):\n{ic_str}\n\n"
        f"SUGGESTED WEIGHTS (IC²-proportional):\n"
        f"{json.dumps({k: round(v,4) for k,v in suggested_weights.items()}, indent=2)}\n\n"
        f"CURRENT WEIGHTS:\n"
        f"{json.dumps({k: round(float(v),4) for k,v in current_weights.items()}, indent=2)}\n\n"
        f"BACKTEST — current weights: {bt_current}\n"
        f"BACKTEST — suggested weights: {bt_suggested}\n\n"
        f"PORTFOLIO STATS: {eq_stats}\n\n"
        f"CURRENT PARAMS:\n"
        f"  Risk:     stop={current_params.get('stop_loss_pct')}, "
        f"tp={current_params.get('take_profit_pct')}, "
        f"trailing_stop={current_params.get('trailing_stop_pct', 0)}, "
        f"partial_tp={current_params.get('partial_tp_pct', 0)} "
        f"(size={current_params.get('partial_tp_size', 0.5)})\n"
        f"  Entry:    threshold={current_params.get('entry_threshold')}, "
        f"confirm_ticks={current_params.get('confirm_ticks', 1)}, "
        f"cooldown_min={current_params.get('cooldown_minutes', 15)}, "
        f"max_hold_min={current_params.get('max_hold_minutes')}\n"
        f"  Exit:     threshold={current_params.get('exit_threshold')}\n"
        f"  Sizing:   base={current_params.get('position_size_pct')}, "
        f"range=[{current_params.get('min_position_pct')}, {current_params.get('max_position_pct')}], "
        f"score_f=[{current_params.get('score_factor_min', 0.7)}, {current_params.get('score_factor_max', 1.3)}], "
        f"vol_f=[{current_params.get('vol_factor_min', 0.4)}, {current_params.get('vol_factor_max', 1.5)}], "
        f"vol_target_atr={current_params.get('vol_target_atr', 0.015)}\n"
        f"  Regime:   filter={current_params.get('regime_filter', True)}, "
        f"ema={current_params.get('regime_ema_bars', 20)}bars, "
        f"threshold={current_params.get('regime_threshold', -0.005)}, "
        f"penalty={current_params.get('regime_size_penalty', 0.5)}\n"
        f"  Safety:   max_daily_loss={current_params.get('max_daily_loss_pct', 0.05)}, "
        f"max_positions={current_params.get('max_positions')}\n"
        f"  IC:       blend=15m×{current_params.get('ic_blend_15m', 0.6)}+"
        f"5m×{current_params.get('ic_blend_5m', 0.4)}, "
        f"floor={current_params.get('ic_floor', 0.02)}\n\n"
        "Review and call update_strategy with your final weights and any param changes."
    )

    messages      = [{"role": "user", "content": prompt}]
    new_weights   = suggested_weights
    new_params    = {}
    reason        = "IC²-proportional reweighting"
    commentary    = ""

    for _ in range(6):
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=OPTIMIZER_SYSTEM,
            tools=OPTIMIZER_TOOLS,
            messages=messages,
        )
        for block in resp.content:
            if hasattr(block, "text"):
                commentary += block.text
        if resp.stop_reason == "end_turn":
            break
        if resp.stop_reason == "tool_use":
            for block in resp.content:
                if block.type == "tool_use" and block.name == "update_strategy":
                    inp         = block.input
                    raw_w       = inp.get("signal_weights", {})
                    total_w     = sum(abs(v) for v in raw_w.values())
                    new_weights = {k: round(v/total_w, 6) for k,v in raw_w.items()} if total_w > 0 else suggested_weights
                    new_params  = inp.get("trader_params", {})
                    reason      = inp.get("reason", reason)

            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": next(b.id for b in resp.content if b.type == "tool_use"),
                "content": json.dumps({"saved": True}),
            }]})
            break

    # ── 6. Save ───────────────────────────────────────────────────────────────
    if not dry_run:
        save_weights(new_weights, reason)
        if new_params:
            updated = {**current_params, **new_params}
            save_trader_params(updated)
            logger.info("Trader params updated: %s", new_params)

    # ── 7. Log optimizer run ──────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    history_record = {
        "ts":              datetime.now(timezone.utc).isoformat(),
        "n_resolved":      len(resolved),
        "ic_table":        dict(ic_table),
        "bt_current":      bt_current,
        "bt_suggested":    bt_suggested,
        "eq_stats":        eq_stats,
        "new_weights":     new_weights,
        "reason":          reason,
    }
    with open(OPT_HISTORY_FILE, "a") as f:
        f.write(json.dumps(history_record) + "\n")

    logger.info("Optimizer done. Reason: %s", reason)
    logger.info("Top 3 signals by weight: %s",
                sorted(new_weights.items(), key=lambda x: -x[1])[:3])

    return {
        "new_weights":   new_weights,
        "ic_table":      dict(ic_table),
        "bt_current":    bt_current,
        "bt_suggested":  bt_suggested,
        "eq_stats":      eq_stats,
        "reason":        reason,
        "commentary":    commentary.strip(),
    }
