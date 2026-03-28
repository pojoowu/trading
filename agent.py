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
]


# ── Agent loop ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert quantitative trading agent.

Your mission is to identify the best stocks to invest in TODAY and execute the trades.

You must follow this disciplined process in order:
1. SCREEN: Call screen_stocks to get the top candidates from the universe.
2. ALPHA: Call compute_alpha on the screened candidates to rank by multi-factor alpha.
3. ANALYZE: Call analyze_stock on the top 5 candidates individually for deep technical/fundamental review.
4. BACKTEST STOCKS: Call backtest_stock on each of your top 3-5 picks to validate historical performance.
5. BACKTEST PORTFOLIO: Call backtest_portfolio with all finalists to see portfolio-level performance.
6. PORTFOLIO CHECK: Call get_portfolio to see current holdings and P&L.
7. DECIDE: Combine all signals (alpha, analysis, backtest) to select the final list (max 10 stocks).
   - Only invest in stocks that have: positive alpha, BUY/WEAK BUY technical recommendation,
     acceptable backtest (Sharpe > 0.5, max drawdown < -35%), and strong momentum.
   - Avoid overbought stocks (RSI > 75) unless fundamentals are exceptional.
8. EXECUTE: Call set_allocation with your final ranked list and reasoning.

Be analytical, disciplined, and risk-aware. After execution, summarise your decisions with clear reasoning.
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

        elif tool_name == "set_allocation":
            if dry_run:
                return json.dumps({"note": "DRY RUN: allocation not executed", "tickers": tool_input.get("tickers", [])})
            tickers = tool_input.get("tickers", [])
            _ensure_data(tickers)
            return _tool_set_allocation(tool_input, price_data, portfolio, executor)

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

    final_text = ""
    max_turns  = 30

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
