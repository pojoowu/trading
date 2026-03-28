"""
Self-improvement optimizer: a second Claude agent that runs weekly,
reviews its own trading performance, and rewrites strategy parameters.

How it works
------------
1. Load the last N days of performance history + signal accuracy log
2. Run a grid / random search over parameter candidates using the backtester
3. Feed all results to Claude (the meta-agent) which reasons about:
   - Which signals were most predictive
   - Which parameters improved or hurt performance
   - What market regime we're in
4. Claude writes a new parameter set + a written explanation of the changes
5. Parameters are saved to data/strategy_params.json
6. The next daily run picks them up automatically

The optimizer never touches live trading — it only reads history and
writes parameters. The daily agent reads those parameters at startup.
"""
import json
import logging
import random
import copy
from datetime import datetime
from itertools import product
from typing import Any, Optional

import numpy as np
import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from performance_tracker import (
    load_history,
    load_signal_log,
    load_params,
    save_params,
    compute_recent_metrics,
    compute_signal_accuracy,
    DEFAULT_PARAMS,
)
from data_fetcher import batch_price_history
from backtester import backtest_portfolio

logger = logging.getLogger(__name__)


# ── Parameter search space ────────────────────────────────────────────────────

SEARCH_SPACE = {
    # How many candidates pass the screen
    "max_stocks_after_screen": [15, 20, 25, 30],

    # Momentum lookback in days
    "momentum_lookback": [126, 189, 252, 315],

    # RSI thresholds
    "rsi_overbought": [70, 75, 80],
    "rsi_oversold":   [25, 30, 35],

    # Portfolio sizing
    "max_positions":    [5, 8, 10, 12, 15],
    "max_position_pct": [0.10, 0.15, 0.20, 0.25],
    "cash_reserve":     [0.02, 0.05, 0.08, 0.10],

    # Risk management
    "stop_loss_pct":    [0.05, 0.07, 0.10, 0.15],
    "take_profit_pct":  [0.15, 0.20, 0.25, 0.35],
}

# Signal weight search: each weight is sampled from this set, then normalised
WEIGHT_OPTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


# ── Random parameter candidate generator ─────────────────────────────────────

def _random_params(base: dict, n: int = 20) -> list[dict]:
    """Generate n random parameter candidates by perturbing the base params."""
    candidates = []
    for _ in range(n):
        p = copy.deepcopy(base)

        # Randomly mutate 2-4 parameters
        keys = random.sample(list(SEARCH_SPACE.keys()), k=random.randint(2, 4))
        for key in keys:
            p[key] = random.choice(SEARCH_SPACE[key])

        # Randomly mutate signal weights
        if random.random() < 0.5:
            signal_names = list(DEFAULT_PARAMS["signal_weights"].keys())
            raw = {s: random.choice(WEIGHT_OPTIONS) for s in signal_names}
            total = sum(raw.values())
            p["signal_weights"] = {s: round(v / total, 4) for s, v in raw.items()}

        candidates.append(p)
    return candidates


# ── Backtest a parameter set ──────────────────────────────────────────────────

def _evaluate_params(params: dict, price_data: dict) -> dict:
    """
    Run a portfolio backtest with the given parameters and return metrics.
    Uses only parameters that affect the backtester (position count, stop/take).
    """
    result = backtest_portfolio(
        price_data,
        top_n=params.get("max_positions", 10),
        initial_capital=100_000,
    )
    m = result.get("metrics", {})
    return {
        "cagr_pct":         m.get("cagr_pct", 0),
        "sharpe":           m.get("sharpe", 0),
        "max_drawdown_pct": m.get("max_drawdown_pct", 0),
        "calmar":           m.get("calmar", 0),
        "total_return_pct": m.get("total_return_pct", 0),
        "alpha_vs_bh_pct":  m.get("alpha_vs_benchmark_pct", 0),
    }


def _score_backtest(metrics: dict) -> float:
    """Single composite score for ranking parameter sets."""
    sharpe  = metrics.get("sharpe", 0) or 0
    calmar  = metrics.get("calmar", 0) or 0
    alpha   = metrics.get("alpha_vs_bh_pct", 0) or 0
    dd      = metrics.get("max_drawdown_pct", 0) or 0
    # Penalise drawdowns beyond -20%
    dd_penalty = max(0, abs(dd) - 20) * 0.1
    return sharpe * 0.4 + calmar * 0.3 + alpha * 0.01 - dd_penalty


# ── Meta-agent (Claude reviews and decides) ───────────────────────────────────

OPTIMIZER_SYSTEM = """You are an expert quantitative trading strategy optimizer.

Your job is to review trading performance data and improve the strategy parameters.

You will receive:
1. Recent portfolio performance metrics (CAGR, Sharpe, drawdown, win rate)
2. Signal accuracy analysis (which signals best predicted returns)
3. Backtest results for candidate parameter sets
4. The current parameter set

Your task:
- Identify what is working and what is not
- Select the best parameter set from the candidates (or propose a modified version)
- Adjust signal weights based on signal accuracy — increase weight for predictive signals
- Consider current market regime (trending vs choppy, bull vs bear)
- Be conservative: do not make extreme changes all at once
- Explain your reasoning clearly

You MUST call the `update_strategy_params` tool with your final parameter decision.
Always include a clear `update_reason` explaining what you changed and why.
"""

OPTIMIZER_TOOLS = [
    {
        "name": "update_strategy_params",
        "description": (
            "Save the new optimised strategy parameters. "
            "This is the final output — call this once with your best parameter set."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "params": {
                    "type": "object",
                    "description": "The full updated parameter dict",
                },
                "update_reason": {
                    "type": "string",
                    "description": "Explanation of what changed and why (2-5 sentences)",
                },
                "expected_improvement": {
                    "type": "string",
                    "description": "What improvement you expect from these changes",
                },
            },
            "required": ["params", "update_reason"],
        },
    }
]


def run_optimizer(
    price_data: Optional[dict] = None,
    n_candidates: int = 15,
) -> dict:
    """
    Run one full optimization cycle.

    Parameters
    ----------
    price_data  : pre-fetched price data (will be fetched if None)
    n_candidates: number of random parameter sets to backtest

    Returns
    -------
    dict with new_params, update_reason, backtest_results, signal_accuracy
    """
    from config import UNIVERSE_TICKERS, BACKTEST_YEARS

    logger.info("=" * 55)
    logger.info("OPTIMIZER STARTING  %s", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
    logger.info("=" * 55)

    # ── Load data ─────────────────────────────────────────────────────────────
    history      = load_history(n_days=90)
    signal_log   = load_signal_log(n_records=500)
    current_params = load_params()

    recent_metrics   = compute_recent_metrics(history)
    signal_accuracy  = compute_signal_accuracy(signal_log)

    logger.info("History: %d days  |  Signal log: %d records", len(history), len(signal_log))
    logger.info("Recent metrics: %s", recent_metrics)
    logger.info("Signal accuracy: %s", signal_accuracy)

    # ── Fetch price data if not provided ──────────────────────────────────────
    if price_data is None:
        logger.info("Fetching price data for backtest…")
        # Use a representative subset to keep it fast
        sample_tickers = UNIVERSE_TICKERS[:30]
        price_data = batch_price_history(sample_tickers, years=BACKTEST_YEARS)

    if not price_data:
        logger.error("No price data available for optimization.")
        return {"error": "no price data"}

    # ── Generate and evaluate candidate parameters ────────────────────────────
    candidates = _random_params(current_params, n=n_candidates)
    # Always include the current params as baseline
    candidates.insert(0, copy.deepcopy(current_params))

    logger.info("Evaluating %d parameter candidates…", len(candidates))
    evaluated = []
    for i, params in enumerate(candidates):
        try:
            metrics = _evaluate_params(params, price_data)
            score   = _score_backtest(metrics)
            evaluated.append({
                "index":   i,
                "params":  params,
                "metrics": metrics,
                "score":   round(score, 4),
            })
            logger.debug(
                "Candidate %d: score=%.3f  sharpe=%.2f  cagr=%.1f%%  dd=%.1f%%",
                i, score,
                metrics.get("sharpe", 0),
                metrics.get("cagr_pct", 0),
                metrics.get("max_drawdown_pct", 0),
            )
        except Exception as exc:
            logger.warning("Candidate %d evaluation failed: %s", i, exc)

    # Sort by score
    evaluated.sort(key=lambda x: x["score"], reverse=True)
    top5 = evaluated[:5]

    # ── Build context for Claude ──────────────────────────────────────────────
    context = {
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "recent_performance": recent_metrics,
        "signal_accuracy": signal_accuracy,
        "current_params": current_params,
        "top_candidates": [
            {
                "rank":    i + 1,
                "score":   c["score"],
                "metrics": c["metrics"],
                "key_changes": {
                    k: c["params"][k]
                    for k in SEARCH_SPACE
                    if c["params"].get(k) != current_params.get(k)
                },
                "signal_weights": c["params"].get("signal_weights"),
            }
            for i, c in enumerate(top5)
        ],
        "baseline_metrics": evaluated[0]["metrics"] if evaluated else {},
    }

    # ── Call Claude meta-agent ────────────────────────────────────────────────
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    messages = [
        {
            "role": "user",
            "content": (
                f"Here is the trading strategy performance review for {context['date']}.\n\n"
                f"RECENT PERFORMANCE (last {recent_metrics.get('n_days', '?')} days):\n"
                f"{json.dumps(recent_metrics, indent=2)}\n\n"
                f"SIGNAL ACCURACY (correlation with 1-month forward returns):\n"
                f"{json.dumps(signal_accuracy, indent=2)}\n\n"
                f"CURRENT PARAMETERS:\n"
                f"{json.dumps({k: v for k, v in current_params.items() if k != 'signal_weights'}, indent=2)}\n"
                f"Current signal weights: {json.dumps(current_params.get('signal_weights'), indent=2)}\n\n"
                f"TOP 5 BACKTEST CANDIDATES (ranked by composite score):\n"
                f"{json.dumps(context['top_candidates'], indent=2)}\n\n"
                "Please review this data, identify what needs to change, "
                "select or modify the best parameter set, and call update_strategy_params."
            ),
        }
    ]

    new_params = None
    update_reason = "no change"
    optimizer_commentary = ""

    for turn in range(10):
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=OPTIMIZER_SYSTEM,
            tools=OPTIMIZER_TOOLS,
            messages=messages,
        )

        for block in response.content:
            if hasattr(block, "text"):
                optimizer_commentary += block.text + "\n"
                logger.info("[OPTIMIZER] %s", block.text[:400])

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            for block in response.content:
                if block.type == "tool_use" and block.name == "update_strategy_params":
                    proposed = block.input.get("params", {})
                    update_reason = block.input.get("update_reason", "")
                    expected      = block.input.get("expected_improvement", "")

                    # Validate and merge with current params
                    new_params = {**current_params, **proposed}
                    # Ensure version bump
                    new_params["version"] = current_params.get("version", 1) + 1
                    new_params["update_reason"] = update_reason

                    # Normalise signal weights if provided
                    sw = new_params.get("signal_weights", {})
                    total_w = sum(sw.values())
                    if total_w > 0:
                        new_params["signal_weights"] = {
                            k: round(v / total_w, 4) for k, v in sw.items()
                        }

                    logger.info("Optimizer update_reason: %s", update_reason)
                    logger.info("Expected improvement: %s", expected)

                    # Tool result
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type":        "tool_result",
                            "tool_use_id": block.id,
                            "content":     json.dumps({"saved": True, "version": new_params["version"]}),
                        }],
                    })
                    break
            if new_params:
                break

    # ── Save new params ───────────────────────────────────────────────────────
    if new_params:
        save_params(new_params)
        logger.info("Optimizer saved new params (version %d)", new_params.get("version", "?"))
    else:
        logger.info("Optimizer made no changes to params")
        new_params = current_params

    # ── Save optimization report ──────────────────────────────────────────────
    _save_optimizer_report(
        date_str=context["date"],
        recent_metrics=recent_metrics,
        signal_accuracy=signal_accuracy,
        top_candidates=top5,
        new_params=new_params,
        update_reason=update_reason,
        commentary=optimizer_commentary,
    )

    return {
        "new_params":       new_params,
        "update_reason":    update_reason,
        "recent_metrics":   recent_metrics,
        "signal_accuracy":  signal_accuracy,
        "n_candidates":     len(evaluated),
        "best_backtest":    top5[0]["metrics"] if top5 else {},
        "commentary":       optimizer_commentary.strip(),
    }


def _save_optimizer_report(
    date_str, recent_metrics, signal_accuracy,
    top_candidates, new_params, update_reason, commentary
) -> None:
    from config import REPORTS_DIR
    import os
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, f"optimizer_{date_str}.txt")
    with open(path, "w") as f:
        f.write(f"OPTIMIZER REPORT — {date_str}\n")
        f.write("=" * 60 + "\n\n")
        f.write("RECENT PERFORMANCE\n")
        f.write(json.dumps(recent_metrics, indent=2) + "\n\n")
        f.write("SIGNAL ACCURACY\n")
        f.write(json.dumps(signal_accuracy, indent=2) + "\n\n")
        f.write("TOP BACKTEST CANDIDATES\n")
        for c in top_candidates:
            f.write(f"  Score {c['score']:.3f}  |  {c['metrics']}\n")
        f.write("\n")
        f.write("UPDATE REASON\n")
        f.write(update_reason + "\n\n")
        f.write("CLAUDE COMMENTARY\n")
        f.write(commentary + "\n\n")
        f.write("NEW PARAMETERS\n")
        f.write(json.dumps(new_params, indent=2) + "\n")
    logger.info("Optimizer report saved to %s", path)


# ── Convenience: fill in forward returns for past signal logs ─────────────────

def update_forward_returns(price_data: dict) -> int:
    """
    Go through unresolved signal log entries and fill in forward returns
    now that enough time has passed. Returns number of records updated.
    """
    if not price_data:
        return 0

    signal_log = load_signal_log(n_records=2000)
    updated = 0
    new_records = []

    for record in signal_log:
        # Skip if already has forward return
        if record.get("forward_return_1m") is not None:
            new_records.append(record)
            continue

        ticker      = record.get("ticker")
        signal_date = record.get("signal_date")
        if not ticker or not signal_date or ticker not in price_data:
            new_records.append(record)
            continue

        try:
            df     = price_data[ticker]
            df     = df[df.index >= signal_date]
            if len(df) >= 21:   # 1 month
                ret_1m = float(df["AdjClose"].iloc[21] / df["AdjClose"].iloc[0] - 1)
                record["forward_return_1m"] = round(ret_1m, 5)
                if len(df) >= 5:
                    ret_1w = float(df["AdjClose"].iloc[5] / df["AdjClose"].iloc[0] - 1)
                    record["forward_return_1w"] = round(ret_1w, 5)
                updated += 1
        except Exception:
            pass

        new_records.append(record)

    # Rewrite the signal log
    if updated > 0:
        import os
        from performance_tracker import SIGNAL_LOG
        with open(SIGNAL_LOG, "w") as f:
            for r in new_records:
                f.write(json.dumps(r) + "\n")
        logger.info("Updated forward returns for %d signal records", updated)

    return updated


