"""
Minute trader: continuous 1-minute loop, 24/7, for crypto.

Every tick (default 60 s):
  1. Fetch latest 1-min bars for all universe symbols
  2. Compute all 22 alpha signals
  3. Rank symbols by composite alpha score (using latest IC-derived weights)
  4. Enter longs on top-N symbols above entry threshold
  5. Exit positions that hit stop-loss, take-profit, or max hold time
  6. Log every signal observation with timestamps for forward-return fill

After every 15-min window:
  - Fill in forward returns (fwd_5m, fwd_15m) for past signal logs

State is persisted to data/crypto_portfolio.json every tick.
"""
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from crypto_data import (
    DEFAULT_UNIVERSE,
    get_history,
    get_current_price,
    batch_history,
)
from alpha_lab import rank_symbols, compute_signals, composite_score, ic_to_weights
from performance_tracker import load_params

logger = logging.getLogger(__name__)

PORTFOLIO_FILE    = "data/crypto_portfolio.json"
SIGNAL_LOG_FILE   = "data/crypto_signal_log.jsonl"
TRADE_LOG_FILE    = "data/crypto_trades.jsonl"
EQUITY_LOG_FILE   = "data/crypto_equity.jsonl"
INTRADAY_PARAMS   = "data/intraday_params.json"

# ── Config (overridable via env / intraday_params.json) ───────────────────────

DEFAULT_TRADER_PARAMS = {
    "universe":          DEFAULT_UNIVERSE,
    "max_positions":     5,
    "position_size_pct": 0.18,       # base allocation per position (% of equity)
    "size_by_score":     True,       # scale size up/down with signal strength
    "size_by_vol":       True,       # shrink size for high-volatility coins
    "min_position_pct":  0.05,       # floor: never less than 5% of equity
    "max_position_pct":  0.25,       # ceiling: never more than 25% of equity
    "entry_threshold":   0.05,       # composite score must exceed this
    "exit_threshold":    -0.08,      # exit if score drops below this
    "stop_loss_pct":     0.015,      # 1.5% hard stop
    "take_profit_pct":   0.030,      # 3% take profit
    "max_hold_minutes":  120,        # force-close after 2h
    "tick_seconds":      60,         # loop interval
    "min_volume_usdt":   5_000_000,  # skip illiquid coins
    "bar_history_bars":  120,        # bars to fetch per symbol
}


def _load_trader_params() -> dict:
    p = dict(DEFAULT_TRADER_PARAMS)
    if os.path.exists(INTRADAY_PARAMS):
        try:
            saved = json.load(open(INTRADAY_PARAMS))
            for k in ("max_positions", "position_size_pct", "entry_threshold",
                      "exit_threshold", "stop_loss_pct", "take_profit_pct",
                      "max_hold_minutes", "tick_seconds"):
                if k in saved:
                    p[k] = saved[k]
        except Exception:
            pass
    return p


def _load_signal_weights() -> Optional[dict]:
    """Load IC-derived signal weights saved by the optimizer."""
    path = "data/signal_weights.json"
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            pass
    return None


# ── Portfolio state ───────────────────────────────────────────────────────────

@dataclass
class CryptoPosition:
    symbol:        str
    entry_price:   float
    qty:           float           # in base currency (e.g. BTC)
    entry_time:    str             # ISO string
    entry_score:   float
    stop_price:    float
    target_price:  float
    last_price:    float = 0.0

    @property
    def cost_basis(self) -> float:
        return self.qty * self.entry_price

    @property
    def market_value(self) -> float:
        return self.qty * (self.last_price or self.entry_price)

    @property
    def unrealised_pnl_pct(self) -> float:
        if self.entry_price == 0: return 0.0
        return (self.last_price / self.entry_price - 1) * 100

    @property
    def hold_minutes(self) -> float:
        try:
            entry_dt = datetime.fromisoformat(self.entry_time).replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - entry_dt).total_seconds() / 60
        except Exception:
            return 0.0


class CryptoPortfolio:
    def __init__(self, initial_cash: float = 10_000.0):
        self.positions: dict[str, CryptoPosition] = {}
        self.cash:      float = initial_cash
        self._load()

    def _load(self):
        if os.path.exists(PORTFOLIO_FILE):
            try:
                d = json.load(open(PORTFOLIO_FILE))
                self.cash = d.get("cash", self.cash)
                for sym, p in d.get("positions", {}).items():
                    self.positions[sym] = CryptoPosition(**p)
            except Exception as exc:
                logger.warning("Could not load portfolio: %s", exc)

    def save(self):
        os.makedirs("data", exist_ok=True)
        d = {
            "cash":        self.cash,
            "positions":   {s: asdict(p) for s, p in self.positions.items()},
            "equity":      self.equity,
            "updated_at":  datetime.now(timezone.utc).isoformat(),
        }
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(d, f, indent=2)

    @property
    def equity(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    def update_prices(self, prices: dict[str, float]):
        for sym, pos in self.positions.items():
            if sym in prices:
                pos.last_price = prices[sym]

    def open_position(
        self,
        symbol:  str,
        price:   float,
        score:   float,
        params:  dict,
        df:      "Optional[pd.DataFrame]" = None,
    ) -> Optional[CryptoPosition]:
        if symbol in self.positions:
            return None
        if len(self.positions) >= params["max_positions"]:
            return None

        # ── Position sizing ───────────────────────────────────────────────────
        # Base: flat fraction of current equity (shrinks with losses, grows with gains)
        base_pct = params["position_size_pct"]

        # 1. Scale by signal strength: strong signal → bigger bet
        #    score is in [entry_threshold, 1.0]; map linearly to [0.7, 1.3]
        if params.get("size_by_score", True):
            entry_thr = params.get("entry_threshold", 0.05)
            score_factor = 0.7 + 0.6 * min(score / max(entry_thr * 4, 0.20), 1.0)
        else:
            score_factor = 1.0

        # 2. Scale by volatility: high ATR → smaller bet (same dollar risk)
        #    Target risk = stop_loss_pct of invest; adjust so ATR risk is constant
        vol_factor = 1.0
        if params.get("size_by_vol", True) and df is not None and len(df) >= 15:
            try:
                atr   = float(np.array([
                    max(h - l, abs(h - pc), abs(l - pc))
                    for h, l, pc in zip(
                        df["High"].iloc[-14:],
                        df["Low"].iloc[-14:],
                        df["Close"].iloc[-15:-1],
                    )
                ]).mean())
                atr_pct = atr / price if price > 0 else 0.015
                # Normalise: 1.5% ATR → factor=1.0; higher ATR → smaller size
                vol_factor = max(0.4, min(1.5, 0.015 / max(atr_pct, 0.001)))
            except Exception:
                vol_factor = 1.0

        raw_pct = base_pct * score_factor * vol_factor
        final_pct = max(
            params.get("min_position_pct", 0.05),
            min(params.get("max_position_pct", 0.25), raw_pct),
        )

        invest = self.equity * final_pct
        if invest > self.cash * 0.99:
            invest = self.cash * 0.99
        if invest < 10:
            return None

        logger.debug(
            "SIZE  %-12s score=%.3f score_f=%.2f vol_f=%.2f => %.1f%% ($%.0f)",
            symbol, score, score_factor, vol_factor, final_pct * 100, invest,
        )

        qty   = invest / price
        stop  = price * (1 - params["stop_loss_pct"])
        tgt   = price * (1 + params["take_profit_pct"])

        pos = CryptoPosition(
            symbol=symbol,
            entry_price=price,
            qty=qty,
            entry_time=datetime.now(timezone.utc).isoformat(),
            entry_score=score,
            stop_price=stop,
            target_price=tgt,
            last_price=price,
        )
        self.cash         -= invest
        self.positions[symbol] = pos
        _log_trade("BUY", symbol, qty, price, score, invest)
        logger.info("OPEN  %-12s qty=%.6f  @ $%.4f  score=%.3f", symbol, qty, price, score)
        return pos

    def close_position(self, symbol: str, price: float, reason: str):
        pos = self.positions.pop(symbol, None)
        if not pos:
            return
        proceeds     = pos.qty * price
        pnl          = proceeds - pos.cost_basis
        pnl_pct      = pos.unrealised_pnl_pct
        self.cash   += proceeds
        _log_trade("SELL", symbol, pos.qty, price, pos.entry_score,
                   proceeds, pnl=pnl, pnl_pct=pnl_pct, reason=reason)
        logger.info(
            "CLOSE %-12s @ $%.4f  pnl=%+.2f%%  reason=%s  hold=%.0fmin",
            symbol, price, pnl_pct, reason, pos.hold_minutes,
        )


# ── Logging helpers ───────────────────────────────────────────────────────────

def _log_trade(action, symbol, qty, price, score, value, pnl=None, pnl_pct=None, reason=""):
    os.makedirs("data", exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action, "symbol": symbol,
        "qty": qty, "price": price,
        "score": score, "value": value,
        "pnl": pnl, "pnl_pct": pnl_pct, "reason": reason,
    }
    with open(TRADE_LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def _log_signal(symbol: str, signals: dict, score: float, price: float):
    """Log signal observation for later IC computation."""
    os.makedirs("data", exist_ok=True)
    record = {
        "ts":      datetime.now(timezone.utc).isoformat(),
        "symbol":  symbol,
        "price":   price,
        "score":   score,
        "signals": signals,
        "fwd_5m":  None,
        "fwd_15m": None,
        "fwd_1h":  None,
    }
    with open(SIGNAL_LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def _log_equity(equity: float, cash: float, n_positions: int):
    os.makedirs("data", exist_ok=True)
    record = {
        "ts":          datetime.now(timezone.utc).isoformat(),
        "equity":      round(equity, 4),
        "cash":        round(cash, 4),
        "n_positions": n_positions,
    }
    with open(EQUITY_LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def fill_forward_returns(bars: dict[str, pd.DataFrame]):
    """
    Go through unresolved signal log entries and fill in forward returns
    now that enough time has passed.
    """
    import pandas as pd
    if not os.path.exists(SIGNAL_LOG_FILE):
        return

    records = []
    updated = 0
    with open(SIGNAL_LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue

            sym = r.get("symbol")
            df  = bars.get(sym)
            if df is None or (r.get("fwd_5m") is not None and r.get("fwd_15m") is not None):
                records.append(r)
                continue

            try:
                entry_ts = pd.Timestamp(r["ts"]).tz_convert("UTC")
                future   = df[df.index > entry_ts]
                entry_px = r.get("price", 0)
                if entry_px <= 0:
                    records.append(r)
                    continue
                if len(future) >= 5 and r.get("fwd_5m") is None:
                    r["fwd_5m"] = round(float(future["Close"].iloc[4] / entry_px - 1), 6)
                if len(future) >= 15 and r.get("fwd_15m") is None:
                    r["fwd_15m"] = round(float(future["Close"].iloc[14] / entry_px - 1), 6)
                if len(future) >= 60 and r.get("fwd_1h") is None:
                    r["fwd_1h"] = round(float(future["Close"].iloc[59] / entry_px - 1), 6)
                updated += 1
            except Exception:
                pass

            records.append(r)

    if updated > 0:
        with open(SIGNAL_LOG_FILE, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        logger.info("Filled forward returns for %d signal records", updated)


# ── Main trading loop ─────────────────────────────────────────────────────────

def run_forever(
    initial_cash: float = 10_000.0,
    dry_run:      bool  = False,
):
    """
    Run the crypto minute trader indefinitely.
    Ctrl-C to stop.
    """
    logger.info("=" * 60)
    logger.info("CRYPTO MINUTE TRADER STARTING  (%s)", "DRY RUN" if dry_run else "PAPER TRADING")
    logger.info("24/7 — Ctrl-C to stop")
    logger.info("=" * 60)

    portfolio    = CryptoPortfolio(initial_cash=initial_cash)
    tick_count   = 0
    last_fwd_fill = 0    # timestamp of last forward-return fill

    while True:
        tick_start = time.time()
        tick_count += 1
        params   = _load_trader_params()
        weights  = _load_signal_weights()
        universe = params["universe"]

        try:
            # ── 1. Fetch latest bars ────────────────────────────────────────
            n_bars = params["bar_history_bars"]
            bars   = batch_history(universe, interval="1m", days=max(1, n_bars // 1440 + 1))
            if not bars:
                logger.warning("No bar data — skipping tick")
                time.sleep(params["tick_seconds"])
                continue

            # ── 2. Get current prices ───────────────────────────────────────
            prices = {sym: float(df["Close"].iloc[-1]) for sym, df in bars.items()}

            # ── 3. Update portfolio mark-to-market ──────────────────────────
            portfolio.update_prices(prices)

            # ── 4. Rank all symbols by alpha ────────────────────────────────
            ranked = rank_symbols(bars, weights)

            # ── 5. Log signals (every tick, for IC computation) ─────────────
            for sym, score, sigs in ranked:
                price = prices.get(sym, 0)
                if price > 0:
                    _log_signal(sym, sigs, score, price)

            # ── 6. Exit positions ───────────────────────────────────────────
            for sym in list(portfolio.positions.keys()):
                pos   = portfolio.positions[sym]
                price = prices.get(sym)
                if price is None:
                    continue

                # Score of current position
                current_score = next(
                    (s for s2, s, _ in ranked if s2 == sym), 0.0
                )

                reason = None
                if price <= pos.stop_price:
                    reason = "stop_loss"
                elif price >= pos.target_price:
                    reason = "take_profit"
                elif pos.hold_minutes >= params["max_hold_minutes"]:
                    reason = "max_hold"
                elif current_score < params["exit_threshold"]:
                    reason = "signal_exit"

                if reason and not dry_run:
                    portfolio.close_position(sym, price, reason)

            # ── 7. Enter new positions ──────────────────────────────────────
            n_open = len(portfolio.positions)
            for sym, score, sigs in ranked:
                if n_open >= params["max_positions"]:
                    break
                if sym in portfolio.positions:
                    continue
                if score < params["entry_threshold"]:
                    break   # ranked list, so everything below is weaker

                price = prices.get(sym, 0)
                if price <= 0:
                    continue

                if not dry_run:
                    pos = portfolio.open_position(sym, price, score, params, df=bars.get(sym))
                    if pos:
                        n_open += 1
                else:
                    logger.info("[DRY] Would BUY %-12s score=%.3f  @ $%.4f", sym, score, price)

            # ── 8. Log equity snapshot ──────────────────────────────────────
            _log_equity(portfolio.equity, portfolio.cash, len(portfolio.positions))
            portfolio.save()

            logger.info(
                "Tick %d | equity=$%.2f | cash=$%.2f | positions=%d | top: %s",
                tick_count,
                portfolio.equity,
                portfolio.cash,
                len(portfolio.positions),
                ", ".join(f"{s}({sc:+.2f})" for s, sc, _ in ranked[:3]),
            )

            # ── 9. Fill forward returns every 15 min ────────────────────────
            if time.time() - last_fwd_fill > 900:
                fill_forward_returns(bars)
                last_fwd_fill = time.time()

        except KeyboardInterrupt:
            logger.info("Minute trader stopped by user.")
            break
        except Exception as exc:
            logger.exception("Tick error: %s", exc)

        # Sleep for remainder of tick interval
        elapsed = time.time() - tick_start
        sleep_s = max(1, params["tick_seconds"] - elapsed)
        time.sleep(sleep_s)
