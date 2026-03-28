"""
Intraday data fetcher: pulls sub-daily OHLCV bars from Yahoo Finance.

Available history per interval:
  1m  → last 7 days only
  5m  → last 60 days
  15m → last 60 days
  1h  → last 730 days  ← best for backtesting intraday strategies
  1d  → unlimited
"""
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)

_YF_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

# Max lookback per interval (Yahoo Finance limits)
MAX_DAYS = {"1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "1h": 730}


def get_intraday_bars(
    ticker: str,
    interval: str = "1h",
    days: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch intraday OHLCV bars for a ticker.

    Parameters
    ----------
    ticker   : e.g. "AAPL"
    interval : "1m" | "5m" | "15m" | "1h"
    days     : lookback in days (capped to Yahoo's limit per interval)

    Returns
    -------
    DataFrame indexed by datetime (timezone-aware, US/Eastern)
    Columns: Open, High, Low, Close, Volume
    Returns None on failure.
    """
    max_d = MAX_DAYS.get(interval, 60)
    if days is None:
        days = max_d
    days = min(days, max_d)

    end   = int(datetime.utcnow().timestamp())
    start = int((datetime.utcnow() - timedelta(days=days)).timestamp())

    params = {
        "period1":  start,
        "period2":  end,
        "interval": interval,
    }

    try:
        from data_fetcher import _get
        data   = _get(_YF_CHART.format(ticker=ticker), params)
        result = data["chart"]["result"][0]

        timestamps = result["timestamp"]
        ohlcv      = result["indicators"]["quote"][0]

        df = pd.DataFrame(
            {
                "Open":   ohlcv["open"],
                "High":   ohlcv["high"],
                "Low":    ohlcv["low"],
                "Close":  ohlcv["close"],
                "Volume": ohlcv["volume"],
            },
            index=pd.to_datetime(timestamps, unit="s", utc=True)
                    .tz_convert("US/Eastern"),
        )
        df.index.name = "Datetime"
        df = df.dropna(subset=["Close"])
        df = df[df["Volume"] > 0]
        logger.debug(
            "Fetched %d %s bars for %s (%s → %s)",
            len(df), interval, ticker,
            df.index[0].strftime("%Y-%m-%d") if len(df) else "?",
            df.index[-1].strftime("%Y-%m-%d") if len(df) else "?",
        )
        return df

    except Exception as exc:
        logger.error("Failed to fetch intraday bars for %s (%s): %s", ticker, interval, exc)
        return None


def batch_intraday(
    tickers: list[str],
    interval: str = "1h",
    days: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """Fetch intraday bars for multiple tickers. Returns {ticker: df}."""
    results = {}
    for ticker in tickers:
        df = get_intraday_bars(ticker, interval=interval, days=days)
        if df is not None and len(df) >= 10:
            results[ticker] = df
        time.sleep(0.15)
    return results
