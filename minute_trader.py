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
    "universe":            DEFAULT_UNIVERSE,
    # ── Portfolio ─────────────────────────────────────────────────────────────
    "max_positions":       5,          # max concurrent open positions
    # ── Position sizing ───────────────────────────────────────────────────────
    "position_size_pct":   0.18,       # base allocation per position (% of equity)
    "size_by_score":       True,       # scale size with signal conviction
    "size_by_vol":         True,       # shrink size for high-volatility coins
    "min_position_pct":    0.05,       # floor: never less than 5% of equity
    "max_position_pct":    0.25,       # ceiling: never more than 25% of equity
    "score_factor_min":    0.7,        # min size multiplier (weakest signal above threshold)
    "score_factor_max":    1.3,        # max size multiplier (strongest signal)
    "vol_factor_min":      0.4,        # min size multiplier (most volatile)
    "vol_factor_max":      1.5,        # max size multiplier (calmest)
    "vol_target_atr":      0.015,      # baseline ATR% for vol normalisation (1.5%)
    # ── Risk per trade ────────────────────────────────────────────────────────
    "stop_loss_pct":       0.015,      # 1.5% hard stop below entry
    "take_profit_pct":     0.030,      # 3.0% take profit above entry
    "trailing_stop_pct":   0.0,        # >0: trail this % below peak (0 = disabled)
    "partial_tp_pct":      0.0,        # >0: sell partial_tp_size at this gain (0 = disabled)
    "partial_tp_size":     0.5,        # fraction to sell at partial TP (0.5 = half)
    # ── Entry / exit signals ──────────────────────────────────────────────────
    "entry_threshold":     0.05,       # min composite score to open a position
    "exit_threshold":      -0.08,      # close if score drops below this
    "confirm_ticks":       1,          # score must exceed threshold for N consecutive ticks
    "cooldown_minutes":    15,         # don't re-enter same coin for N min after a loss exit
    "max_hold_minutes":    120,        # force-close after 2h regardless of signal
    # ── Market regime filter ──────────────────────────────────────────────────
    "regime_filter":       True,       # scale down sizing when BTC is in a downtrend
    "regime_ema_bars":     20,         # EMA period for regime detection
    "regime_threshold":    -0.005,     # if BTC is this % below its EMA → bear regime
    "regime_size_penalty": 0.5,        # multiply position size by this in bear regime
    # ── Risk circuit breakers ─────────────────────────────────────────────────
    "max_daily_loss_pct":  0.05,       # pause new entries if daily equity drops 5%
    # ── Optimizer IC settings (read by crypto_optimizer) ─────────────────────
    "ic_blend_15m":        0.6,        # weight of 15-min IC in blended IC
    "ic_blend_5m":         0.4,        # weight of 5-min IC in blended IC
    "ic_floor":            0.02,       # signals below this IC get zero weight
    # ── Realism ───────────────────────────────────────────────────────────────
    "fee_pct":             0.001,      # 0.1% per trade (Binance taker fee)
    "slippage_pct":        0.0005,     # 0.05% adverse fill vs close price
    "tick_seconds":        60,         # loop interval
    "min_volume_usdt":     5_000_000,  # skip illiquid coins
    "bar_history_bars":    120,        # bars to fetch per symbol
}


def _load_trader_params() -> dict:
    p = dict(DEFAULT_TRADER_PARAMS)
    if os.path.exists(INTRADAY_PARAMS):
        try:
            saved = json.load(open(INTRADAY_PARAMS))
            p.update(saved)   # load ALL saved keys; optimizer can tune any param
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
    symbol:          str
    entry_price:     float
    qty:             float           # in base currency (e.g. BTC)
    entry_time:      str             # ISO string
    entry_score:     float
    stop_price:      float
    target_price:    float
    last_price:      float = 0.0
    peak_price:      float = 0.0    # highest price seen since entry (trailing stop)
    partial_tp_done: bool  = False  # True after first partial take-profit fired

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
                if prices[sym] > pos.peak_price:
                    pos.peak_price = prices[sym]

    def open_position(
        self,
        symbol:        str,
        price:         float,
        score:         float,
        params:        dict,
        df:            "Optional[pd.DataFrame]" = None,
        regime_factor: float = 1.0,
    ) -> Optional[CryptoPosition]:
        if symbol in self.positions:
            return None
        if len(self.positions) >= params["max_positions"]:
            return None

        # ── Position sizing ───────────────────────────────────────────────────
        base_pct = params["position_size_pct"]

        # 1. Scale by signal conviction
        if params.get("size_by_score", True):
            entry_thr  = max(params.get("entry_threshold", 0.05) * 4, 0.20)
            sf_min     = params.get("score_factor_min", 0.7)
            sf_max     = params.get("score_factor_max", 1.3)
            score_factor = sf_min + (sf_max - sf_min) * min(score / entry_thr, 1.0)
        else:
            score_factor = 1.0

        # 2. Scale by volatility: high ATR → smaller bet to keep dollar-risk constant
        vol_factor = 1.0
        if params.get("size_by_vol", True) and df is not None and len(df) >= 15:
            try:
                atr = float(np.array([
                    max(h - l, abs(h - pc), abs(l - pc))
                    for h, l, pc in zip(
                        df["High"].iloc[-14:],
                        df["Low"].iloc[-14:],
                        df["Close"].iloc[-15:-1],
                    )
                ]).mean())
                atr_pct    = atr / price if price > 0 else params.get("vol_target_atr", 0.015)
                vf_min     = params.get("vol_factor_min", 0.4)
                vf_max     = params.get("vol_factor_max", 1.5)
                vol_target = params.get("vol_target_atr", 0.015)
                vol_factor = max(vf_min, min(vf_max, vol_target / max(atr_pct, 0.001)))
            except Exception:
                vol_factor = 1.0

        # 3. Regime penalty (bear market → reduce size)
        raw_pct   = base_pct * score_factor * vol_factor * regime_factor
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
            "SIZE  %-12s score=%.3f sf=%.2f vf=%.2f rf=%.2f => %.1f%% ($%.0f)",
            symbol, score, score_factor, vol_factor, regime_factor, final_pct * 100, invest,
        )

        # Apply slippage (buy at ask = close + slippage) and fee
        slippage   = params.get("slippage_pct", 0.0005)
        fee_pct    = params.get("fee_pct", 0.001)
        fill_price = price * (1 + slippage)          # paid slightly more than close
        fee        = invest * fee_pct                 # broker commission
        qty        = (invest - fee) / fill_price      # net shares after fee
        stop       = fill_price * (1 - params["stop_loss_pct"])
        tgt        = fill_price * (1 + params["take_profit_pct"])

        pos = CryptoPosition(
            symbol=symbol,
            entry_price=fill_price,
            qty=qty,
            entry_time=datetime.now(timezone.utc).isoformat(),
            entry_score=score,
            stop_price=stop,
            target_price=tgt,
            last_price=fill_price,
            peak_price=fill_price,
        )
        self.cash -= invest   # invest includes fee (qty was net of fee)
        self.positions[symbol] = pos
        _log_trade("BUY", symbol, qty, fill_price, score, invest, fee=fee)
        logger.info(
            "OPEN  %-12s qty=%.6f  @ $%.4f  (slip+fee=$%.2f)  score=%.3f",
            symbol, qty, fill_price, fee + invest * slippage, score,
        )
        return pos

    def close_position(self, symbol: str, price: float, reason: str,
                       params: Optional[dict] = None):
        pos = self.positions.pop(symbol, None)
        if not pos:
            return
        # Apply slippage (sell at bid = close - slippage) and fee
        slippage   = (params or {}).get("slippage_pct", 0.0005)
        fee_pct    = (params or {}).get("fee_pct", 0.001)
        fill_price = price * (1 - slippage)
        gross      = pos.qty * fill_price
        fee        = gross * fee_pct
        proceeds   = gross - fee
        pnl        = proceeds - pos.cost_basis
        pnl_pct    = (fill_price / pos.entry_price - 1) * 100 if pos.entry_price else 0
        self.cash += proceeds
        _log_trade("SELL", symbol, pos.qty, fill_price, pos.entry_score,
                   proceeds, pnl=pnl, pnl_pct=pnl_pct, reason=reason, fee=fee)
        logger.info(
            "CLOSE %-12s @ $%.4f  pnl=%+.2f%%  reason=%s  hold=%.0fmin",
            symbol, fill_price, pnl_pct, reason, pos.hold_minutes,
        )


# ── Logging helpers ───────────────────────────────────────────────────────────

def _log_trade(action, symbol, qty, price, score, value,
               pnl=None, pnl_pct=None, reason="", fee=None):
    os.makedirs("data", exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action, "symbol": symbol,
        "qty": qty, "price": price,
        "score": score, "value": value,
        "pnl": pnl, "pnl_pct": pnl_pct,
        "fee": round(fee, 6) if fee else 0.0,
        "reason": reason,
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


def _log_equity(equity: float, cash: float, n_positions: int,
                btc_price: Optional[float] = None, btc_benchmark: Optional[float] = None):
    os.makedirs("data", exist_ok=True)
    record = {
        "ts":            datetime.now(timezone.utc).isoformat(),
        "equity":        round(equity, 4),
        "cash":          round(cash, 4),
        "n_positions":   n_positions,
        "btc_price":     round(btc_price, 2) if btc_price else None,
        "btc_benchmark": round(btc_benchmark, 4) if btc_benchmark else None,
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


# ── Market regime filter ──────────────────────────────────────────────────────

def _compute_regime_factor(bars: dict, params: dict) -> float:
    """
    Detect bull/bear regime from BTC trend.
    Returns 1.0 in a neutral/bull regime, regime_size_penalty in a bear regime.

    Bear = BTC close is more than regime_threshold% below its EMA.
    E.g. regime_threshold=-0.005: if BTC is 0.5% below its 20-bar EMA → bear.
    """
    if not params.get("regime_filter", True):
        return 1.0
    btc = bars.get("BTCUSDT")
    if btc is None or len(btc) < params.get("regime_ema_bars", 20) + 2:
        return 1.0
    ema_n   = int(params.get("regime_ema_bars", 20))
    ema_val = float(btc["Close"].ewm(span=ema_n, adjust=False).mean().iloc[-1])
    price   = float(btc["Close"].iloc[-1])
    if ema_val == 0:
        return 1.0
    deviation = (price - ema_val) / ema_val
    threshold = params.get("regime_threshold", -0.005)
    if deviation < threshold:
        factor = params.get("regime_size_penalty", 0.5)
        return float(factor)
    return 1.0


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

    portfolio      = CryptoPortfolio(initial_cash=initial_cash)
    tick_count     = 0
    last_fwd_fill  = 0

    # ── Session-level state ───────────────────────────────────────────────────
    daily_open_equity    = portfolio.equity         # reset each calendar day
    last_reset_day       = datetime.now(timezone.utc).date()
    last_exit_times:     dict[str, datetime] = {}   # cooldown tracking
    above_thresh_ticks:  dict[str, int]      = {}   # confirmation-tick counters
    consecutive_losses   = 0                         # circuit breaker counter
    btc_start_price: Optional[float] = None          # for buy-and-hold benchmark

    while True:
        tick_start = time.time()
        tick_count += 1
        params   = _load_trader_params()
        weights  = _load_signal_weights()
        universe = params["universe"]

        try:
            # ── 0. Daily reset ──────────────────────────────────────────────
            today = datetime.now(timezone.utc).date()
            if today != last_reset_day:
                daily_open_equity = portfolio.equity
                last_reset_day    = today
                logger.info("New day. Daily open equity reset to $%.2f", daily_open_equity)

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

                current_score = next(
                    (s for s2, s, _ in ranked if s2 == sym), 0.0
                )

                # 6a. Partial take-profit: sell a fraction at partial_tp_pct,
                #     then slide stop to breakeven so the remainder rides for free
                partial_tp_pct = params.get("partial_tp_pct", 0.0)
                if partial_tp_pct > 0 and not pos.partial_tp_done and not dry_run:
                    if price >= pos.entry_price * (1 + partial_tp_pct):
                        slippage   = params.get("slippage_pct", 0.0005)
                        fee_pct    = params.get("fee_pct", 0.001)
                        fill_p     = price * (1 - slippage)
                        sell_qty   = pos.qty * params.get("partial_tp_size", 0.5)
                        gross      = sell_qty * fill_p
                        fee        = gross * fee_pct
                        proceeds   = gross - fee
                        pnl        = proceeds - sell_qty * pos.entry_price
                        pos.qty   -= sell_qty
                        portfolio.cash      += proceeds
                        pos.partial_tp_done  = True
                        pos.stop_price       = pos.entry_price   # move stop to breakeven
                        _log_trade("SELL_PARTIAL", sym, sell_qty, fill_p, pos.entry_score,
                                   proceeds, pnl=pnl,
                                   pnl_pct=(fill_p / pos.entry_price - 1) * 100,
                                   reason="partial_tp", fee=fee)
                        logger.info(
                            "PARTIAL_TP %-10s qty=%.6f @ $%.4f  stop->breakeven",
                            sym, sell_qty, fill_p,
                        )

                # 6b. Full exit conditions (checked in priority order)
                reason = None
                trailing_stop_pct = params.get("trailing_stop_pct", 0.0)
                if trailing_stop_pct > 0 and pos.peak_price > 0:
                    trail_level = pos.peak_price * (1 - trailing_stop_pct)
                    if price <= trail_level:
                        reason = "trailing_stop"
                if reason is None and price <= pos.stop_price:
                    reason = "stop_loss"
                elif reason is None and price >= pos.target_price:
                    reason = "take_profit"
                elif reason is None and pos.hold_minutes >= params["max_hold_minutes"]:
                    reason = "max_hold"
                elif reason is None and current_score < params["exit_threshold"]:
                    reason = "signal_exit"

                if reason and not dry_run:
                    portfolio.close_position(sym, price, reason, params=params)
                    last_exit_times[sym] = datetime.now(timezone.utc)
                    if reason in ("stop_loss", "trailing_stop", "signal_exit"):
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0

            # ── 7. Pre-entry checks ─────────────────────────────────────────
            # Daily loss circuit breaker
            daily_pnl_pct = (portfolio.equity / daily_open_equity - 1) if daily_open_equity > 0 else 0
            can_enter = daily_pnl_pct >= -params.get("max_daily_loss_pct", 0.05)
            if not can_enter:
                logger.warning(
                    "Daily loss limit hit (%.1f%%) — pausing new entries",
                    daily_pnl_pct * 100,
                )

            # Market regime
            regime_factor = _compute_regime_factor(bars, params)
            if regime_factor < 1.0:
                logger.info("Bear regime — position size penalty %.0f%%", regime_factor * 100)

            # ── 8. Enter new positions ──────────────────────────────────────
            n_open        = len(portfolio.positions)
            confirm_ticks = int(params.get("confirm_ticks", 1))
            cooldown_m    = params.get("cooldown_minutes", 0)

            for sym, score, sigs in ranked:
                if n_open >= params["max_positions"]:
                    break
                if sym in portfolio.positions:
                    continue
                if score < params["entry_threshold"]:
                    # Update counter even for below-threshold to reset it
                    above_thresh_ticks.pop(sym, None)
                    break   # list is sorted, all remaining are weaker

                # Confirmation ticks: must be above threshold for N consecutive ticks
                above_thresh_ticks[sym] = above_thresh_ticks.get(sym, 0) + 1
                if above_thresh_ticks[sym] < confirm_ticks:
                    continue

                # Re-entry cooldown after a loss exit
                if cooldown_m > 0 and sym in last_exit_times:
                    elapsed_m = (datetime.now(timezone.utc) - last_exit_times[sym]).total_seconds() / 60
                    if elapsed_m < cooldown_m:
                        continue

                if not can_enter:
                    break

                price = prices.get(sym, 0)
                if price <= 0:
                    continue

                if not dry_run:
                    pos = portfolio.open_position(
                        sym, price, score, params,
                        df=bars.get(sym),
                        regime_factor=regime_factor,
                    )
                    if pos:
                        n_open += 1
                        above_thresh_ticks.pop(sym, None)   # reset counter after entry
                else:
                    logger.info("[DRY] Would BUY %-12s score=%.3f  @ $%.4f", sym, score, price)

            # ── 9. Log equity snapshot with BTC benchmark ───────────────────
            btc_price = prices.get("BTCUSDT")
            if btc_price and btc_start_price is None:
                btc_start_price = btc_price   # anchor on first tick
            btc_benchmark = None
            if btc_price and btc_start_price:
                # What would $initial_cash be worth if we just held BTC?
                btc_benchmark = initial_cash * (btc_price / btc_start_price)
            _log_equity(portfolio.equity, portfolio.cash, len(portfolio.positions),
                        btc_price=btc_price, btc_benchmark=btc_benchmark)
            portfolio.save()

            logger.info(
                "Tick %d | equity=$%.2f | daily=%.1f%% | regime=%.0f%% | "
                "positions=%d | cons_loss=%d | top: %s",
                tick_count,
                portfolio.equity,
                daily_pnl_pct * 100,
                regime_factor * 100,
                len(portfolio.positions),
                consecutive_losses,
                ", ".join(f"{s}({sc:+.2f})" for s, sc, _ in ranked[:3]),
            )

            # ── 10. Fill forward returns every 15 min ───────────────────────
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
