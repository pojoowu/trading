"""
Trading Agent: Claude-powered orchestrator using the Anthropic tool-use API.

The agent is given a set of tools (screener, alpha, analyzer, backtester,
portfolio, executor) and autonomously decides when to call each one.
It then produces a final written investment decision and executes trades.

Flow
----
1. Screen universe → candidate list
2. Fetch price + fundamental data for candidates
3. Compute alpha scores
4. Run deep analysis on top candidates
5. Backtest the full portfolio strategy
6. Decide final allocation (with Claude reasoning)
7. Generate orders & execute
8. Write daily report
"""
import json
import logging
import os
from datetime import datetime
from typing import Any

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, REPORTS_DIR, BACKTEST_YEARS, MAX_POSITIONS
from data_fetcher import batch_price_history, get_fundamentals, get_quote
from screener import run_screen
from alpha_research import compute_alpha, alpha_breakdown
from analyzer import analyze_stock, format_analysis_report
from backtester import backtest_single, backtest_portfolio, format_backtest_report
from portfolio import PortfolioManager, compute_target_allocation
from executor import Executor
from performance_tracker import (
    load_params,
    record_daily_run,
    record_signal_outcome,
)

logger = logging.getLogger(__name__)

# ── Tool implementations ──────────────────────────────────────────────────────

def _tool_screen_stocks(params: dict) -> str:
    """Run the stock screener and return top candidates as JSON."""
    max_n = params.get("max_results", 20)
    result = run_screen()
    if result.empty:
        return json.dumps({"error": "screener returned no results"})
    top = result.head(max_n)[["ticker", "score", "mom_12m", "mom_6m", "sector", "mkt_cap_b"]].to_dict(orient="records")
    return json.dumps({"candidates": top, "count": len(top)})


def _tool_compute_alpha(params: dict, price_data: dict, fundamentals: dict) -> str:
    """Compute alpha scores for a list of tickers."""
    tickers = params.get("tickers", list(price_data.keys()))
    alpha_df = compute_alpha(tickers, price_data, fundamentals)
    if alpha_df.empty:
        return json.dumps({"error": "no alpha computed"})
    records = alpha_df[["ticker", "alpha_score", "cross_momentum", "sharpe_momentum",
                         "trend_following", "analyst_upside"]].head(15).to_dict(orient="records")
    return json.dumps({"alpha_scores": records})


def _tool_analyze_stock(params: dict, price_data: dict, fundamentals: dict) -> str:
    """Run deep technical + fundamental analysis on a single ticker."""
    ticker = params.get("ticker", "")
    df   = price_data.get(ticker)
    fund = fundamentals.get(ticker, {})
    if df is None:
        return json.dumps({"error": f"no price data for {ticker}"})
    analysis = analyze_stock(ticker, df, fund)
    report   = format_analysis_report(analysis)
    return json.dumps({"analysis": analysis, "report": report})


def _tool_backtest_stock(params: dict, price_data: dict) -> str:
    """Backtest a single-stock momentum strategy."""
    ticker = params.get("ticker", "")
    df = price_data.get(ticker)
    if df is None:
        return json.dumps({"error": f"no data for {ticker}"})
    result  = backtest_single(ticker, df)
    report  = format_backtest_report(result)
    metrics = result.get("metrics", {})
    return json.dumps({
        "report": report,
        "cagr_pct": metrics.get("cagr_pct"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "win_rate_pct": metrics.get("win_rate_pct"),
        "alpha_vs_bh_pct": metrics.get("alpha_vs_bh_pct"),
    })


def _tool_backtest_portfolio(params: dict, price_data: dict) -> str:
    """Backtest the full portfolio strategy across all candidates."""
    tickers = params.get("tickers", list(price_data.keys()))
    subset  = {t: price_data[t] for t in tickers if t in price_data}
    result  = backtest_portfolio(subset, top_n=params.get("top_n", MAX_POSITIONS))
    report  = format_backtest_report(result)
    return json.dumps({
        "report": report,
        **{k: v for k, v in result.get("metrics", {}).items()
           if k not in ("monthly_returns",)},
    })


def _tool_get_portfolio(portfolio: PortfolioManager) -> str:
    """Return the current portfolio snapshot."""
    state = portfolio.state
    positions = [
        {
            "ticker":          t,
            "shares":          p.shares,
            "avg_cost":        p.avg_cost,
            "last_price":      p.last_price,
            "market_value":    round(p.market_value, 2),
            "unrealised_pnl_pct": round(p.unrealised_pnl_pct, 2),
        }
        for t, p in state.positions.items()
    ]
    return json.dumps({
        "summary": portfolio.summary(),
        "cash": round(state.cash, 2),
        "total_equity": round(state.total_equity, 2),
        "positions": positions,
    })


def _tool_intraday_timing(params: dict) -> str:
    """
    Check intraday entry timing for a list of tickers using the best
    backtested intraday strategy (loaded from data/intraday_params.json).
    Returns per-ticker entry signals + reasoning.
    """
    from intraday_data import get_intraday_bars
    from intraday_strategy import intraday_entry_now
    from optimizer import load_intraday_params

    tickers     = params.get("tickers", [])
    intraday_p  = load_intraday_params()
    strategy    = intraday_p.get("strategy", "vwap_reversion")
    strat_params = intraday_p.get("params", {})

    results = {}
    for ticker in tickers:
        df = get_intraday_bars(ticker, interval="1h", days=5)
        if df is None or len(df) < 10:
            results[ticker] = {"enter_now": False, "reason": "no intraday data"}
            continue
        signal = intraday_entry_now(df, strategy=strategy, params=strat_params or None)
        results[ticker] = signal

    return json.dumps({
        "strategy":      strategy,
        "strategy_params": strat_params,
        "signals":       results,
        "enter_count":   sum(1 for v in results.values() if v.get("enter_now")),
    })


def _tool_backtest_intraday(params: dict) -> str:
    """
    Backtest and optimise all three intraday strategies for a ticker,
    returning the best strategy + parameters + metrics.
    """
    from intraday_data import get_intraday_bars
    from intraday_backtester import compare_strategies, format_intraday_report

    ticker = params.get("ticker", "")
    df     = get_intraday_bars(ticker, interval="1h", days=730)
    if df is None or len(df) < 50:
        return json.dumps({"error": f"insufficient intraday data for {ticker}"})

    result = compare_strategies(ticker, df)
    report = format_intraday_report(result)
    return json.dumps({
        "report":       report,
        "winner":       result["winner"],
        "best_params":  result["best_params"],
        "best_metrics": result["best_metrics"],
        "ranking":      [{
            "strategy": r["strategy"],
            "sharpe":   r["metrics"].get("sharpe"),
            "win_rate": r["metrics"].get("win_rate_pct"),
            "trades":   r["metrics"].get("total_trades"),
            "score":    r["score"],
        } for r in result["ranking"]],
    })


def _tool_set_allocation(
    params: dict,
    price_data: dict,
    portfolio: PortfolioManager,
    executor: Executor,
) -> str:
    """
    Set the target allocation and execute the rebalancing trades.

    params.tickers: ranked list of tickers to invest in (best first)
    params.alpha_scores: optional {ticker: score} for weighting
    """
    tickers = params.get("tickers", [])
    alpha_scores = params.get("alpha_scores")

    # Get current prices
    current_prices = {}
    for t in tickers:
        q = get_quote(t)
        if q.get("price"):
            current_prices[t] = q["price"]
        elif t in price_data:
            current_prices[t] = float(price_data[t]["AdjClose"].iloc[-1])

    # Refresh portfolio with latest prices
    portfolio.refresh(current_prices)

    # Compute target
    target = compute_target_allocation(tickers, alpha_scores)

    # Generate and execute orders
    orders = portfolio.get_orders(target, current_prices)
    report = executor.execute_and_report(orders)

    # Refresh after fills
    portfolio.refresh(current_prices)

    return json.dumps({
        "execution_report": report,
        "target_allocation": target,
        "orders_count": len(orders),
        "portfolio_summary": portfolio.summary(),
    })


# ── Tool definitions for Claude ───────────────────────────────────────────────

TOOLS = [
    {
        "name": "screen_stocks",
        "description": (
            "Screen the entire stock universe using momentum, volume, market-cap, "
            "and fundamental quality filters. Returns the top candidate stocks ranked "
            "by composite score."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of candidates to return (default 20)",
                },
            },
        },
    },
    {
        "name": "compute_alpha",
        "description": (
            "Compute multi-factor alpha scores (cross-momentum, Sharpe momentum, "
            "trend following, analyst upside, value/quality, volume surge, short reversal) "
            "for a list of tickers. Returns ranked scores."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols to score",
                },
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "analyze_stock",
        "description": (
            "Run a deep technical and fundamental analysis on a single stock. "
            "Computes RSI, MACD, Bollinger Bands, ATR, Stochastic, OBV, "
            "support/resistance levels, candlestick patterns, and gives a "
            "BUY/HOLD/SELL recommendation with a risk/reward setup."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "backtest_stock",
        "description": (
            "Backtest a momentum strategy on a single stock over the configured "
            "historical period. Returns CAGR, Sharpe ratio, max drawdown, "
            "win rate, and alpha vs buy-and-hold."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "backtest_portfolio",
        "description": (
            "Run a full portfolio backtest across all candidate stocks. "
            "Rebalances monthly into top-N momentum stocks. "
            "Returns portfolio CAGR, Sharpe, drawdown, and alpha vs benchmark."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Candidate tickers to include in the backtest",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Max positions to hold at once (default 10)",
                },
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "get_portfolio",
        "description": "Return the current portfolio: cash, positions, P&L, and equity.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_allocation",
        "description": (
            "Set the target stock allocation and execute all necessary buy/sell orders "
            "to rebalance the portfolio. This is the final step that actually places trades. "
            "Provide tickers ranked best-first and optional alpha scores for weighting."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Final ranked list of tickers to invest in (best first)",
                },
                "alpha_scores": {
                    "type": "object",
                    "description": "Optional {ticker: score} dict for position sizing weight",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief reasoning for this allocation decision",
                },
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "backtest_intraday",
        "description": (
            "Backtest and optimise all three intraday strategies (VWAP reversion, "
            "Opening Range Breakout, Hourly Momentum) on up to 2 years of hourly bars. "
            "Returns the winning strategy, its best parameters, and full metrics "
            "(win rate, profit factor, Sharpe, max drawdown). "
            "Run this on each of your top picks to find the best intraday entry style."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "intraday_timing",
        "description": (
            "Check the current intraday entry signal for a list of tickers using the "
            "best backtested intraday strategy. Returns which stocks have an active "
            "entry signal RIGHT NOW (based on latest hourly bar). "
            "Call this just before set_allocation to get the best entry timing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tickers to check for intraday entry signal",
                },
            },
            "required": ["tickers"],
        },
    },
]


# ── Agent loop ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert quantitative trading agent.

Your mission is to identify the best stocks to invest in TODAY and execute trades at the optimal intraday moment.

You must follow this disciplined process in order:
1. SCREEN: Call screen_stocks to get the top candidates from the universe.
2. ALPHA: Call compute_alpha on the screened candidates to rank by multi-factor alpha.
3. ANALYZE: Call analyze_stock on the top 5 candidates for deep technical/fundamental review.
4. BACKTEST STOCKS: Call backtest_stock on your top 3-5 picks to validate daily strategy performance.
5. BACKTEST INTRADAY: Call backtest_intraday on each finalist to find the best intraday entry strategy
   (VWAP reversion, Opening Range Breakout, or Hourly Momentum) and its optimal parameters.
6. BACKTEST PORTFOLIO: Call backtest_portfolio with all finalists for portfolio-level performance.
7. PORTFOLIO CHECK: Call get_portfolio to see current holdings and P&L.
8. DECIDE: Combine all signals to select the final list (max 10 stocks).
   - Only invest in stocks with: positive alpha, BUY/WEAK BUY recommendation,
     acceptable daily backtest (Sharpe > 0.5, max drawdown > -35%),
     AND a profitable intraday strategy (win rate > 50%, profit factor > 1.2).
   - Avoid overbought stocks (RSI > 75) unless fundamentals are exceptional.
9. INTRADAY TIMING: Call intraday_timing on your final list to check if NOW is a good entry point.
   - If fewer than half the stocks have an active intraday signal, consider waiting or reducing size.
10. EXECUTE: Call set_allocation with your final ranked list and reasoning.

Be analytical, disciplined, and risk-aware. The intraday timing step is critical —
entering at the right moment within the day significantly improves average entry price.
After execution, summarise your decisions with clear reasoning including the intraday strategy used.
"""


def run_agent(
    initial_capital: float = 100_000.0,
    dry_run: bool = False,
) -> str:
    """
    Run one full daily trading cycle.

    Parameters
    ----------
    initial_capital : used only when initialising a fresh portfolio
    dry_run         : if True, analyse but do NOT call set_allocation

    Returns
    -------
    str: final agent summary / report text
    """
    client    = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    portfolio = PortfolioManager()
    executor  = Executor(portfolio)

    logger.info("=" * 60)
    logger.info("TRADING AGENT STARTING  %s", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
    logger.info("=" * 60)

    # Pre-fetch universe data (screener will fetch what it needs, but we cache here
    # for efficiency so tools don't each make their own network calls)
    price_data: dict = {}
    fundamentals_data: dict = {}
    data_fetched_for: set = set()

    def _ensure_data(tickers: list[str]) -> None:
        """Lazily fetch price + fundamental data for tickers we haven't loaded yet."""
        missing = [t for t in tickers if t not in data_fetched_for]
        if not missing:
            return
        logger.info("Fetching data for %d tickers…", len(missing))
        new_prices = batch_price_history(missing, years=BACKTEST_YEARS)
        price_data.update(new_prices)
        for t in missing:
            fundamentals_data[t] = get_fundamentals(t)
            data_fetched_for.add(t)

    def _dispatch(tool_name: str, tool_input: dict) -> str:
        """Dispatch a tool call and return the result as a string."""
        nonlocal agent_picks, last_alpha_scores
        logger.info("Tool call: %s  input=%s", tool_name, str(tool_input)[:200])

        if tool_name == "screen_stocks":
            result = _tool_screen_stocks(tool_input)
            # Pre-fetch data for screened tickers
            try:
                candidates = json.loads(result).get("candidates", [])
                tickers = [c["ticker"] for c in candidates]
                _ensure_data(tickers)
            except Exception:
                pass
            return result

        elif tool_name == "compute_alpha":
            tickers = tool_input.get("tickers", [])
            _ensure_data(tickers)
            return _tool_compute_alpha(tool_input, price_data, fundamentals_data)

        elif tool_name == "analyze_stock":
            ticker = tool_input.get("ticker", "")
            _ensure_data([ticker])
            return _tool_analyze_stock(tool_input, price_data, fundamentals_data)

        elif tool_name == "backtest_stock":
            ticker = tool_input.get("ticker", "")
            _ensure_data([ticker])
            return _tool_backtest_stock(tool_input, price_data)

        elif tool_name == "backtest_portfolio":
            tickers = tool_input.get("tickers", [])
            _ensure_data(tickers)
            return _tool_backtest_portfolio(tool_input, price_data)

        elif tool_name == "get_portfolio":
            return _tool_get_portfolio(portfolio)

        elif tool_name == "backtest_intraday":
            return _tool_backtest_intraday(tool_input)

        elif tool_name == "intraday_timing":
            return _tool_intraday_timing(tool_input)

        elif tool_name == "set_allocation":
            tickers = tool_input.get("tickers", [])
            nonlocal agent_picks
            agent_picks = tickers
            if dry_run:
                return json.dumps({"note": "DRY RUN: allocation not executed", "tickers": tickers})
            _ensure_data(tickers)
            return _tool_set_allocation(tool_input, price_data, portfolio, executor)

        elif tool_name == "compute_alpha":
            result = _tool_compute_alpha(tool_input, price_data, fundamentals_data)
            # Cache alpha scores for performance tracking
            try:
                scores = json.loads(result).get("alpha_scores", [])
                for row in scores:
                    last_alpha_scores[row["ticker"]] = row
            except Exception:
                pass
            return result

        else:
            return json.dumps({"error": f"unknown tool: {tool_name}"})

    # ── Agentic loop ──────────────────────────────────────────────────────────
    messages = [
        {
            "role": "user",
            "content": (
                f"Today is {datetime.utcnow().strftime('%Y-%m-%d')}. "
                f"Run the full daily trading cycle: screen, research alpha, "
                f"analyze, backtest, and then execute the optimal trades. "
                f"{'DRY RUN: analyse but do not execute.' if dry_run else 'Execute live paper trades.'}"
            ),
        }
    ]

    # Load live strategy parameters (updated by optimizer)
    strategy_params = load_params()
    logger.info("Using strategy params version %d: %s",
                strategy_params.get("version", 1),
                strategy_params.get("update_reason", "defaults"))

    final_text = ""
    max_turns  = 30
    agent_picks: list[str] = []
    last_alpha_scores: dict = {}

    for turn in range(max_turns):
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        logger.debug("Agent turn %d: stop_reason=%s", turn, response.stop_reason)

        # Collect text from this response
        for block in response.content:
            if hasattr(block, "text"):
                final_text += block.text + "\n"
                logger.info("[AGENT] %s", block.text[:300])

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason != "tool_use":
            logger.warning("Unexpected stop_reason: %s", response.stop_reason)
            break

        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result_str = _dispatch(block.name, block.input)
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result_str,
                })

        # Append assistant turn + tool results
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    # ── Record performance + signal outcomes ──────────────────────────────────
    try:
        state = portfolio.state
        positions_snapshot = {
            t: {
                "shares":     p.shares,
                "avg_cost":   p.avg_cost,
                "last_price": p.last_price,
                "pnl_pct":    p.unrealised_pnl_pct,
            }
            for t, p in state.positions.items()
        }
        record_daily_run(
            date_str=datetime.utcnow().strftime("%Y-%m-%d"),
            portfolio_equity=state.total_equity,
            cash=state.cash,
            positions=positions_snapshot,
            trades_executed=[],   # executor logs trades separately
            agent_picks=agent_picks,
            signal_scores=last_alpha_scores,
            params_version=strategy_params.get("version", 1),
        )

        # Record signal outcomes (forward returns filled in later by optimizer)
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        for ticker, scores in last_alpha_scores.items():
            record_signal_outcome(
                ticker=ticker,
                signal_date=date_str,
                signals={k: v for k, v in scores.items() if k != "ticker"},
            )
    except Exception as exc:
        logger.warning("Performance recording failed: %s", exc)

    return final_text.strip()


# ── Report writing ────────────────────────────────────────────────────────────

def save_report(content: str) -> str:
    """Save the agent's final report to the reports directory."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    path = os.path.join(REPORTS_DIR, f"report_{date_str}.txt")
    with open(path, "w") as f:
        f.write(f"TRADING AGENT DAILY REPORT — {date_str}\n")
        f.write("=" * 60 + "\n\n")
        f.write(content)
    logger.info("Report saved to %s", path)
    return path
