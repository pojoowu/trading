"""
Portfolio manager: tracks holdings, computes position sizes, and
decides what to buy / sell to align the live portfolio with the
target allocation produced by the agent.

Supports both the local paper-trading simulation and Alpaca paper API.
"""
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd

from config import (
    PORTFOLIO_FILE,
    MAX_POSITIONS,
    MAX_POSITION_PCT,
    MIN_POSITION_PCT,
    PORTFOLIO_CASH_RESERVE,
)

logger = logging.getLogger(__name__)


# ── Portfolio state ───────────────────────────────────────────────────────────

@dataclass
class Position:
    ticker:      str
    shares:      float
    avg_cost:    float
    entry_date:  str
    last_price:  float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * self.last_price

    @property
    def unrealised_pnl(self) -> float:
        return (self.last_price - self.avg_cost) * self.shares

    @property
    def unrealised_pnl_pct(self) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (self.last_price / self.avg_cost - 1) * 100


@dataclass
class PortfolioState:
    cash:      float
    positions: dict   # {ticker: Position}
    created:   str = ""
    last_updated: str = ""

    @classmethod
    def default(cls, initial_cash: float = 100_000.0) -> "PortfolioState":
        return cls(
            cash=initial_cash,
            positions={},
            created=datetime.utcnow().isoformat(),
            last_updated=datetime.utcnow().isoformat(),
        )

    @property
    def total_market_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_equity(self) -> float:
        return self.cash + self.total_market_value

    @property
    def allocation(self) -> dict[str, float]:
        total = self.total_equity
        if total == 0:
            return {}
        return {t: p.market_value / total for t, p in self.positions.items()}


# ── Persistence ───────────────────────────────────────────────────────────────

def _load_portfolio() -> PortfolioState:
    """Load portfolio from JSON file, or create a default one."""
    os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE) as f:
                data = json.load(f)
            positions = {
                t: Position(**p) for t, p in data.get("positions", {}).items()
            }
            return PortfolioState(
                cash=data["cash"],
                positions=positions,
                created=data.get("created", ""),
                last_updated=data.get("last_updated", ""),
            )
        except Exception as exc:
            logger.warning("Could not load portfolio (%s). Using fresh portfolio.", exc)
    return PortfolioState.default()


def _save_portfolio(state: PortfolioState) -> None:
    state.last_updated = datetime.utcnow().isoformat()
    os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
    data = {
        "cash": state.cash,
        "positions": {t: asdict(p) for t, p in state.positions.items()},
        "created": state.created,
        "last_updated": state.last_updated,
    }
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Price refresh ─────────────────────────────────────────────────────────────

def refresh_prices(state: PortfolioState, prices: dict[str, float]) -> None:
    """Update last_price for all held positions."""
    for ticker, pos in state.positions.items():
        if ticker in prices:
            pos.last_price = prices[ticker]


# ── Target allocation ─────────────────────────────────────────────────────────

def compute_target_allocation(
    ranked_tickers: list[str],
    alpha_scores: Optional[dict[str, float]] = None,
    n_positions: int = MAX_POSITIONS,
    max_pct: float = MAX_POSITION_PCT,
    min_pct: float = MIN_POSITION_PCT,
    cash_reserve: float = PORTFOLIO_CASH_RESERVE,
) -> dict[str, float]:
    """
    Given a ranked list of tickers (best first), compute target % allocations.

    Strategy: alpha-score-proportional weight, capped at max_pct.
    If alpha_scores is None, uses equal-weight within top-N.

    Returns {ticker: fraction_of_equity}  (values sum to ≤ 1 - cash_reserve)
    """
    candidates = ranked_tickers[:n_positions]
    if not candidates:
        return {}

    invest_fraction = 1.0 - cash_reserve

    if alpha_scores:
        raw_scores = np.array([max(alpha_scores.get(t, 0), 0.01) for t in candidates])
    else:
        raw_scores = np.ones(len(candidates))

    # Normalise to sum to 1
    weights = raw_scores / raw_scores.sum()

    # Apply min / max caps (iteratively)
    for _ in range(10):
        capped = np.minimum(weights, max_pct / invest_fraction)
        floored = np.maximum(capped, min_pct / invest_fraction)
        floored = floored / floored.sum()     # re-normalise
        if np.allclose(floored, weights, atol=1e-6):
            break
        weights = floored

    target = {t: round(w * invest_fraction, 4) for t, w in zip(candidates, weights)}
    return target


# ── Trade orders ──────────────────────────────────────────────────────────────

@dataclass
class Order:
    ticker:  str
    action:  str       # "BUY" | "SELL"
    shares:  float
    price:   float
    reason:  str = ""

    @property
    def value(self) -> float:
        return self.shares * self.price


def generate_orders(
    state: PortfolioState,
    target_allocation: dict[str, float],
    current_prices: dict[str, float],
    min_trade_value: float = 100.0,
) -> list[Order]:
    """
    Compare current portfolio to target allocation and produce a list of orders
    (sells first, then buys).

    Parameters
    ----------
    state             : current portfolio state
    target_allocation : {ticker: fraction_of_equity}  from compute_target_allocation
    current_prices    : {ticker: current_price}
    min_trade_value   : skip orders below this dollar threshold

    Returns
    -------
    List of Order objects (sells first, then buys)
    """
    equity = state.total_equity
    orders: list[Order] = []

    # Current allocation
    current_alloc = state.allocation

    # --- SELL orders (reduce or close positions not in target) ---
    for ticker, pos in list(state.positions.items()):
        price = current_prices.get(ticker, pos.last_price)
        if price <= 0:
            continue

        target_pct = target_allocation.get(ticker, 0.0)
        current_pct = pos.market_value / equity if equity > 0 else 0

        if ticker not in target_allocation:
            # Close full position
            orders.append(Order(ticker, "SELL", pos.shares, price, reason="not_in_target"))
        elif current_pct - target_pct > 0.01:
            # Trim to target
            target_value = target_pct * equity
            sell_value   = pos.market_value - target_value
            if sell_value >= min_trade_value:
                sell_shares = sell_value / price
                orders.append(Order(ticker, "SELL", round(sell_shares, 4), price, reason="rebalance_trim"))

    # --- BUY orders ---
    # Simulate cash after sells
    simulated_cash = state.cash + sum(o.value for o in orders if o.action == "SELL")

    for ticker, target_pct in target_allocation.items():
        price = current_prices.get(ticker)
        if not price or price <= 0:
            logger.warning("No price for %s, skipping buy order", ticker)
            continue

        current_pos  = state.positions.get(ticker)
        current_pct  = (current_pos.market_value / equity) if (current_pos and equity > 0) else 0

        if target_pct - current_pct > 0.005:
            buy_value  = (target_pct - current_pct) * equity
            buy_shares = buy_value / price
            if buy_value >= min_trade_value and simulated_cash >= buy_value:
                orders.append(Order(ticker, "BUY", round(buy_shares, 4), price, reason="rebalance_buy"))
                simulated_cash -= buy_value

    # Sort: sells before buys
    orders.sort(key=lambda o: 0 if o.action == "SELL" else 1)
    return orders


# ── Portfolio summary ─────────────────────────────────────────────────────────

def portfolio_summary(state: PortfolioState) -> str:
    """Return a human-readable portfolio snapshot."""
    lines = [
        f"{'=' * 55}",
        f"  PORTFOLIO SNAPSHOT  ({datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')})",
        f"{'=' * 55}",
        f"  Cash              : ${state.cash:>12,.2f}",
        f"  Invested          : ${state.total_market_value:>12,.2f}",
        f"  Total Equity      : ${state.total_equity:>12,.2f}",
        f"",
        f"  {'Ticker':<8} {'Shares':>8} {'Avg Cost':>10} {'Price':>10} {'Value':>12} {'P&L':>10} {'P&L%':>7}",
        f"  {'-'*70}",
    ]
    for ticker, pos in sorted(state.positions.items()):
        lines.append(
            f"  {ticker:<8} {pos.shares:>8.2f} {pos.avg_cost:>10.2f} "
            f"{pos.last_price:>10.2f} {pos.market_value:>12,.2f} "
            f"{pos.unrealised_pnl:>10,.2f} {pos.unrealised_pnl_pct:>6.1f}%"
        )
    lines.append(f"{'=' * 55}")
    return "\n".join(lines)


# ── Public API (used by executor) ─────────────────────────────────────────────

class PortfolioManager:
    """Stateful wrapper around portfolio load/save/compute."""

    def __init__(self):
        self.state = _load_portfolio()

    def refresh(self, prices: dict[str, float]) -> None:
        refresh_prices(self.state, prices)
        _save_portfolio(self.state)

    def get_target(
        self,
        ranked_tickers: list[str],
        alpha_scores: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        return compute_target_allocation(ranked_tickers, alpha_scores)

    def get_orders(
        self,
        target: dict[str, float],
        prices: dict[str, float],
    ) -> list[Order]:
        return generate_orders(self.state, target, prices)

    def apply_fill(self, order: Order) -> None:
        """Update portfolio state after an order is filled."""
        state = self.state
        if order.action == "BUY":
            cost = order.shares * order.price
            if cost > state.cash:
                logger.warning("Insufficient cash for %s BUY. Skipping.", order.ticker)
                return
            state.cash -= cost
            if order.ticker in state.positions:
                pos = state.positions[order.ticker]
                total_shares = pos.shares + order.shares
                pos.avg_cost = (pos.avg_cost * pos.shares + order.price * order.shares) / total_shares
                pos.shares   = total_shares
                pos.last_price = order.price
            else:
                state.positions[order.ticker] = Position(
                    ticker=order.ticker,
                    shares=order.shares,
                    avg_cost=order.price,
                    entry_date=datetime.utcnow().strftime("%Y-%m-%d"),
                    last_price=order.price,
                )

        elif order.action == "SELL":
            if order.ticker not in state.positions:
                logger.warning("Tried to sell %s but no position. Skipping.", order.ticker)
                return
            pos = state.positions[order.ticker]
            sell_shares = min(order.shares, pos.shares)
            state.cash += sell_shares * order.price
            pos.shares -= sell_shares
            if pos.shares < 0.001:
                del state.positions[order.ticker]

        _save_portfolio(state)

    def summary(self) -> str:
        return portfolio_summary(self.state)
