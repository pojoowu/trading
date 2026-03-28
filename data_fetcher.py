"""
Data fetcher: pulls historical price data and fundamental info from Yahoo Finance.
Falls back gracefully when network is unavailable.
"""
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Yahoo Finance query endpoints
_YF_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_YF_SUMMARY = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    )
}


def _get(url: str, params: dict, retries: int = 3) -> dict:
    """GET with simple retry / back-off."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            wait = 2 ** attempt
            logger.warning("Fetch attempt %d failed (%s). Retrying in %ds…", attempt + 1, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


# ── Price history ─────────────────────────────────────────────────────────────

def get_price_history(
    ticker: str,
    years: int = 3,
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """
    Return a DataFrame with columns [Open, High, Low, Close, Volume, AdjClose]
    indexed by date.  Returns None on failure.
    """
    end = int(datetime.utcnow().timestamp())
    start = int((datetime.utcnow() - timedelta(days=years * 365 + 30)).timestamp())
    params = {
        "period1": start,
        "period2": end,
        "interval": interval,
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    try:
        data = _get(_YF_CHART.format(ticker=ticker), params)
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        ohlcv = result["indicators"]["quote"][0]
        adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose", ohlcv["close"])

        df = pd.DataFrame(
            {
                "Open": ohlcv["open"],
                "High": ohlcv["high"],
                "Low": ohlcv["low"],
                "Close": ohlcv["close"],
                "Volume": ohlcv["volume"],
                "AdjClose": adj,
            },
            index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("US/Eastern").normalize(),
        )
        df.index.name = "Date"
        df = df.dropna(subset=["Close"])
        return df
    except Exception as exc:
        logger.error("get_price_history(%s): %s", ticker, exc)
        return None


# ── Fundamental / quote summary ───────────────────────────────────────────────

def get_fundamentals(ticker: str) -> dict:
    """
    Return key fundamental metrics for a ticker:
    marketCap, trailingPE, forwardPE, priceToBook, dividendYield,
    returnOnEquity, debtToEquity, revenueGrowth, earningsGrowth, sector, industry.
    Returns empty dict on failure.
    """
    modules = "summaryDetail,defaultKeyStatistics,financialData,assetProfile"
    params = {"modules": modules, "formatted": "false"}
    try:
        data = _get(_YF_SUMMARY.format(ticker=ticker), params)
        result = data.get("quoteSummary", {}).get("result", [{}])[0] or {}

        def _get_val(section: str, key: str, default=None):
            return result.get(section, {}).get(key, default)

        return {
            "ticker": ticker,
            "sector": _get_val("assetProfile", "sector", "Unknown"),
            "industry": _get_val("assetProfile", "industry", "Unknown"),
            "marketCap": _get_val("summaryDetail", "marketCap"),
            "trailingPE": _get_val("summaryDetail", "trailingPE"),
            "forwardPE": _get_val("summaryDetail", "forwardPE"),
            "priceToBook": _get_val("defaultKeyStatistics", "priceToBook"),
            "dividendYield": _get_val("summaryDetail", "dividendYield"),
            "beta": _get_val("summaryDetail", "beta"),
            "fiftyTwoWeekHigh": _get_val("summaryDetail", "fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": _get_val("summaryDetail", "fiftyTwoWeekLow"),
            "returnOnEquity": _get_val("financialData", "returnOnEquity"),
            "returnOnAssets": _get_val("financialData", "returnOnAssets"),
            "debtToEquity": _get_val("financialData", "debtToEquity"),
            "revenueGrowth": _get_val("financialData", "revenueGrowth"),
            "earningsGrowth": _get_val("financialData", "earningsGrowth"),
            "grossMargins": _get_val("financialData", "grossMargins"),
            "operatingMargins": _get_val("financialData", "operatingMargins"),
            "currentPrice": _get_val("financialData", "currentPrice"),
            "targetMeanPrice": _get_val("financialData", "targetMeanPrice"),
            "recommendationKey": _get_val("financialData", "recommendationKey"),
            "numberOfAnalystOpinions": _get_val("financialData", "numberOfAnalystOpinions"),
            "shortRatio": _get_val("defaultKeyStatistics", "shortRatio"),
        }
    except Exception as exc:
        logger.error("get_fundamentals(%s): %s", ticker, exc)
        return {"ticker": ticker}


# ── Quick quote (current price + day stats) ───────────────────────────────────

def get_quote(ticker: str) -> dict:
    """Return current price, day change, volume. Fast single-day call."""
    params = {"period1": int(time.time()) - 86400, "period2": int(time.time()), "interval": "1m"}
    try:
        data = _get(_YF_CHART.format(ticker=ticker), params)
        meta = data["chart"]["result"][0]["meta"]
        return {
            "ticker": ticker,
            "price": meta.get("regularMarketPrice"),
            "previousClose": meta.get("previousClose") or meta.get("chartPreviousClose"),
            "volume": meta.get("regularMarketVolume"),
        }
    except Exception as exc:
        logger.error("get_quote(%s): %s", ticker, exc)
        return {"ticker": ticker}


# ── Batch helper ──────────────────────────────────────────────────────────────

def batch_price_history(tickers: list[str], years: int = 3) -> dict[str, pd.DataFrame]:
    """
    Fetch price history for multiple tickers.
    Returns a dict {ticker: DataFrame}; missing tickers are omitted.
    """
    results = {}
    for ticker in tickers:
        df = get_price_history(ticker, years=years)
        if df is not None and not df.empty:
            results[ticker] = df
        time.sleep(0.1)   # gentle rate-limit
    return results
