"""
Order executor: routes orders to either the local paper-trading
simulator or Alpaca's paper-trading REST API.

Usage
-----
    exec = Executor()
    fills = exec.execute(orders, prices)   # returns list of filled Order
"""
import json
import logging
import os
from datetime import datetime
from typing import Optional

import requests

from config import (
    BROKER,
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_BASE_URL,
    TRADES_LOG_FILE,
)
from portfolio import Order, PortfolioManager

logger = logging.getLogger(__name__)


# ── Trade log ─────────────────────────────────────────────────────────────────

def _log_trade(order: Order, status: str, note: str = "") -> None:
    """Append a trade record to the trades log file."""
    os.makedirs(os.path.dirname(TRADES_LOG_FILE), exist_ok=True)
    record = {
        "ts":     datetime.utcnow().isoformat(),
        "action": order.action,
        "ticker": order.ticker,
        "shares": order.shares,
        "price":  order.price,
        "value":  round(order.value, 2),
        "reason": order.reason,
        "status": status,
        "note":   note,
    }
    with open(TRADES_LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Paper simulator ───────────────────────────────────────────────────────────

class PaperExecutor:
    """Local paper-trading: instantly fills at the given price."""

    def __init__(self, portfolio: PortfolioManager):
        self.portfolio = portfolio

    def execute(self, orders: list[Order]) -> list[Order]:
        filled = []
        for order in orders:
            try:
                self.portfolio.apply_fill(order)
                _log_trade(order, "filled", "paper")
                logger.info(
                    "[PAPER] %s %s x%.2f @ $%.2f  ($%.0f)",
                    order.action, order.ticker, order.shares, order.price, order.value,
                )
                filled.append(order)
            except Exception as exc:
                _log_trade(order, "error", str(exc))
                logger.error("Paper fill error for %s: %s", order.ticker, exc)
        return filled


# ── Alpaca paper executor ─────────────────────────────────────────────────────

class AlpacaExecutor:
    """Submit market orders via Alpaca REST API (paper mode)."""

    def __init__(self, portfolio: PortfolioManager):
        self.portfolio = portfolio
        self.base      = ALPACA_BASE_URL.rstrip("/")
        self.headers   = {
            "APCA-API-KEY-ID":     ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            "Content-Type":        "application/json",
        }

    def _post(self, endpoint: str, body: dict) -> dict:
        url = f"{self.base}{endpoint}"
        r   = requests.post(url, json=body, headers=self.headers, timeout=15)
        r.raise_for_status()
        return r.json()

    def execute(self, orders: list[Order]) -> list[Order]:
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            logger.error("Alpaca API keys not set. Cannot execute orders.")
            return []

        filled = []
        for order in orders:
            side = order.action.lower()   # "buy" | "sell"
            body = {
                "symbol":        order.ticker,
                "qty":           str(round(order.shares, 4)),
                "side":          side,
                "type":          "market",
                "time_in_force": "day",
            }
            try:
                resp = self._post("/v2/orders", body)
                order_id = resp.get("id", "?")
                fill_price = float(resp.get("filled_avg_price") or order.price)
                # Update order price with actual fill
                order.price = fill_price
                self.portfolio.apply_fill(order)
                _log_trade(order, "filled", f"alpaca order_id={order_id}")
                logger.info(
                    "[ALPACA] %s %s x%.4f @ $%.2f  (id=%s)",
                    order.action, order.ticker, order.shares, fill_price, order_id,
                )
                filled.append(order)
            except Exception as exc:
                _log_trade(order, "error", str(exc))
                logger.error("Alpaca order error for %s %s: %s", order.action, order.ticker, exc)
        return filled


# ── Unified executor ──────────────────────────────────────────────────────────

class Executor:
    """
    Unified interface.  Selects paper or Alpaca based on BROKER config var.
    """

    def __init__(self, portfolio: Optional[PortfolioManager] = None):
        self.portfolio = portfolio or PortfolioManager()
        if BROKER == "alpaca":
            self._impl = AlpacaExecutor(self.portfolio)
            logger.info("Using Alpaca paper executor")
        else:
            self._impl = PaperExecutor(self.portfolio)
            logger.info("Using local paper executor")

    def execute(self, orders: list[Order]) -> list[Order]:
        """Execute all orders and return the filled subset."""
        if not orders:
            logger.info("No orders to execute.")
            return []
        logger.info("Executing %d orders…", len(orders))
        return self._impl.execute(orders)

    def execute_and_report(self, orders: list[Order]) -> str:
        """Execute orders and return a formatted execution report."""
        filled = self.execute(orders)
        lines = [
            f"{'=' * 50}",
            f"  EXECUTION REPORT  ({datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')})",
            f"{'=' * 50}",
            f"  Orders submitted : {len(orders)}",
            f"  Orders filled    : {len(filled)}",
            "",
        ]
        for o in filled:
            lines.append(
                f"  {'✓' if BROKER == 'paper' else '→'} {o.action:<4} {o.ticker:<6} "
                f"x{o.shares:.2f} @ ${o.price:.2f}  =  ${o.value:,.0f}  ({o.reason})"
            )
        if not filled:
            lines.append("  No orders were filled.")
        lines.append(f"{'=' * 50}")
        return "\n".join(lines)
