"""
Crypto data fetcher via Binance public REST API.
No API key required for market data.

Intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d
Max bars per request: 1000
"""
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

_BASE = "https://api.binance.com/api/v3"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

# Default universe — major liquid pairs
DEFAULT_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT",
    "LINKUSDT", "DOTUSDT", "MATICUSDT", "ADAUSDT", "ATOMUSDT",
    "LTCUSDT", "XRPUSDT", "UNIUSDT", "AAVEUSDT", "NEARUSDT",
    "FILUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT",
]

# Max lookback per interval before hitting Binance limits
MAX_BARS = 1000   # per request; paginate for more


def _get(endpoint: str, params: dict, retries: int = 4) -> list | dict:
    url = f"{_BASE}{endpoint}"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            wait = 2 ** attempt
            logger.warning("Binance fetch attempt %d failed (%s). Retrying in %ds…", attempt + 1, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Binance fetch failed after {retries} attempts: {endpoint}")


def get_klines(
    symbol: str,
    interval: str = "1m",
    limit: int = 1000,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV klines from Binance.

    Parameters
    ----------
    symbol   : e.g. "BTCUSDT"
    interval : "1m" | "5m" | "15m" | "1h" | "4h" | "1d"
    limit    : number of bars (max 1000 per request)
    start_ms : start time in milliseconds (optional)
    end_ms   : end time in milliseconds (optional)

    Returns
    -------
    DataFrame with columns [Open, High, Low, Close, Volume]
    indexed by UTC datetime. Returns None on failure.
    """
    params: dict = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms:
        params["startTime"] = start_ms
    if end_ms:
        params["endTime"] = end_ms

    try:
        raw = _get("/klines", params)
        if not raw:
            return None

        df = pd.DataFrame(raw, columns=[
            "open_time", "Open", "High", "Low", "Close", "Volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        df["Open"]   = df["Open"].astype(float)
        df["High"]   = df["High"].astype(float)
        df["Low"]    = df["Low"].astype(float)
        df["Close"]  = df["Close"].astype(float)
        df["Volume"] = df["Volume"].astype(float)
        df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df.index.name = "Datetime"
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        return df
    except Exception as exc:
        logger.error("get_klines(%s, %s): %s", symbol, interval, exc)
        return None


def get_history(
    symbol: str,
    interval: str = "1m",
    days: int = 7,
) -> Optional[pd.DataFrame]:
    """
    Fetch up to `days` of history, paginating across multiple requests
    if needed (Binance limit is 1000 bars per call).
    """
    interval_minutes = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
        "12h": 720, "1d": 1440,
    }.get(interval, 1)

    total_bars  = int(days * 24 * 60 / interval_minutes)
    end_ms      = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms    = end_ms - int(days * 24 * 3600 * 1000)

    frames = []
    cur_start = start_ms

    while cur_start < end_ms:
        df = get_klines(symbol, interval, limit=1000, start_ms=cur_start, end_ms=end_ms)
        if df is None or df.empty:
            break
        frames.append(df)
        last_ts = int(df.index[-1].timestamp() * 1000)
        if last_ts <= cur_start:
            break
        cur_start = last_ts + interval_minutes * 60 * 1000
        time.sleep(0.05)   # gentle rate limit

    if not frames:
        return None
    combined = pd.concat(frames).drop_duplicates().sort_index()
    return combined


def get_current_price(symbol: str) -> Optional[float]:
    """Return the latest trade price for a symbol."""
    try:
        data = _get("/ticker/price", {"symbol": symbol})
        return float(data["price"])
    except Exception as exc:
        logger.error("get_current_price(%s): %s", symbol, exc)
        return None


def get_ticker_24h(symbol: str) -> dict:
    """Return 24h stats: priceChangePercent, volume, highPrice, lowPrice."""
    try:
        data = _get("/ticker/24hr", {"symbol": symbol})
        return {
            "symbol":             symbol,
            "price":              float(data.get("lastPrice", 0)),
            "change_pct_24h":     float(data.get("priceChangePercent", 0)),
            "volume_usdt":        float(data.get("quoteVolume", 0)),
            "high_24h":           float(data.get("highPrice", 0)),
            "low_24h":            float(data.get("lowPrice", 0)),
            "trades_24h":         int(data.get("count", 0)),
        }
    except Exception as exc:
        logger.error("get_ticker_24h(%s): %s", symbol, exc)
        return {"symbol": symbol}


def batch_history(
    symbols: list[str],
    interval: str = "1m",
    days: int = 3,
) -> dict[str, pd.DataFrame]:
    """Fetch history for multiple symbols. Returns {symbol: df}."""
    results = {}
    for sym in symbols:
        df = get_history(sym, interval=interval, days=days)
        if df is not None and len(df) > 10:
            results[sym] = df
        time.sleep(0.1)
    logger.info("Fetched %s bars for %d/%d symbols", interval, len(results), len(symbols))
    return results


def batch_ticker_24h(symbols: list[str]) -> dict[str, dict]:
    """Return 24h stats for all symbols at once (single bulk API call)."""
    try:
        raw = _get("/ticker/24hr", {})   # returns all symbols
        lookup = {d["symbol"]: d for d in raw}
        results = {}
        for sym in symbols:
            if sym in lookup:
                d = lookup[sym]
                results[sym] = {
                    "symbol":         sym,
                    "price":          float(d.get("lastPrice", 0)),
                    "change_pct_24h": float(d.get("priceChangePercent", 0)),
                    "volume_usdt":    float(d.get("quoteVolume", 0)),
                    "high_24h":       float(d.get("highPrice", 0)),
                    "low_24h":        float(d.get("lowPrice", 0)),
                    "trades_24h":     int(d.get("count", 0)),
                }
        return results
    except Exception as exc:
        logger.error("batch_ticker_24h: %s", exc)
        return {}
