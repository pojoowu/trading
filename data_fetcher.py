"""
Data fetcher: pulls historical price data and fundamental info from Yahoo Finance.

Yahoo Finance now requires a session cookie + crumb for most endpoints.
This module handles that automatically with a shared session and crumb cache.
"""
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Yahoo Finance endpoints
_YF_CHART   = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_YF_SUMMARY = "https://query1.finance.yahoo.com/v11/finance/quoteSummary/{ticker}"
_YF_CRUMB   = "https://query1.finance.yahoo.com/v1/test/getcrumb"
_YF_CONSENT = "https://consent.yahoo.com/v2/collectConsent"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Origin":          "https://finance.yahoo.com",
    "Referer":         "https://finance.yahoo.com/",
}

# ── Session + crumb management ────────────────────────────────────────────────

_session: Optional[requests.Session] = None
_crumb:   Optional[str]              = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(_HEADERS)
    return _session


def _refresh_crumb() -> str:
    """
    Obtain a fresh Yahoo Finance crumb by visiting the finance page
    and calling the crumb endpoint. Stores cookie in the shared session.
    """
    global _crumb
    session = _get_session()

    # Step 1: hit the main finance page to get the initial cookie
    try:
        session.get("https://finance.yahoo.com", timeout=10)
    except Exception:
        pass

    # Step 2: handle EU consent if redirected
    try:
        r = session.get(
            "https://finance.yahoo.com/quote/AAPL",
            timeout=10,
            allow_redirects=True,
        )
        # If we land on the consent page, post through it
        if "consent.yahoo.com" in r.url:
            session.post(
                _YF_CONSENT,
                data={"agree": "agree", "consentUUID": "default", "sessionId": "default"},
                timeout=10,
            )
    except Exception:
        pass

    # Step 3: fetch the crumb
    for attempt in range(3):
        try:
            r = session.get(_YF_CRUMB, timeout=10)
            if r.status_code == 200 and r.text.strip():
                _crumb = r.text.strip()
                logger.debug("Yahoo Finance crumb refreshed: %s…", _crumb[:8])
                return _crumb
        except Exception as exc:
            logger.debug("Crumb attempt %d failed: %s", attempt + 1, exc)
        time.sleep(1)

    raise RuntimeError("Could not obtain Yahoo Finance crumb after 3 attempts")


def _get_crumb() -> str:
    global _crumb
    if not _crumb:
        _refresh_crumb()
    return _crumb


# ── Core GET with auto-retry + crumb refresh ──────────────────────────────────

def _get(url: str, params: dict, retries: int = 3) -> dict:
    """GET with crumb injection, cookie session, and retry / back-off."""
    global _crumb
    session = _get_session()

    for attempt in range(retries):
        try:
            crumb = _get_crumb()
            p = {**params, "crumb": crumb}
            r = session.get(url, params=p, timeout=20)

            # 401 / 403 → crumb expired, refresh and retry
            if r.status_code in (401, 403):
                logger.debug("Got %d, refreshing crumb…", r.status_code)
                _crumb = None
                time.sleep(1)
                continue

            r.raise_for_status()
            return r.json()

        except requests.HTTPError as exc:
            wait = 2 ** attempt
            logger.warning("HTTP error attempt %d (%s). Retrying in %ds…", attempt + 1, exc, wait)
            time.sleep(wait)
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
    end   = int(datetime.utcnow().timestamp())
    start = int((datetime.utcnow() - timedelta(days=years * 365 + 30)).timestamp())
    params = {
        "period1": start,
        "period2": end,
        "interval": interval,
        "events":   "div,splits",
        "includeAdjustedClose": "true",
    }
    try:
        data   = _get(_YF_CHART.format(ticker=ticker), params)
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        ohlcv  = result["indicators"]["quote"][0]
        adj    = result["indicators"].get("adjclose", [{}])[0].get("adjclose", ohlcv["close"])

        df = pd.DataFrame(
            {
                "Open":     ohlcv["open"],
                "High":     ohlcv["high"],
                "Low":      ohlcv["low"],
                "Close":    ohlcv["close"],
                "Volume":   ohlcv["volume"],
                "AdjClose": adj,
            },
            index=pd.to_datetime(timestamps, unit="s", utc=True)
                    .tz_convert("US/Eastern").normalize(),
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
    Return key fundamental metrics for a ticker.
    Uses v11 quoteSummary which works with the crumb/cookie auth flow.
    Returns empty dict on failure.
    """
    modules = "summaryDetail,defaultKeyStatistics,financialData,assetProfile"
    params  = {"modules": modules, "formatted": "false", "lang": "en-US", "region": "US"}
    try:
        data   = _get(_YF_SUMMARY.format(ticker=ticker), params)
        result = data.get("quoteSummary", {}).get("result") or [{}]
        result = result[0] if result else {}

        def _v(section: str, key: str, default=None):
            val = result.get(section, {}).get(key, default)
            # v11 sometimes wraps values as {"raw": x, "fmt": "..."}
            if isinstance(val, dict) and "raw" in val:
                return val["raw"]
            return val

        return {
            "ticker":                  ticker,
            "sector":                  _v("assetProfile",          "sector",                  "Unknown"),
            "industry":                _v("assetProfile",          "industry",                "Unknown"),
            "marketCap":               _v("summaryDetail",         "marketCap"),
            "trailingPE":              _v("summaryDetail",         "trailingPE"),
            "forwardPE":               _v("summaryDetail",         "forwardPE"),
            "priceToBook":             _v("defaultKeyStatistics",  "priceToBook"),
            "dividendYield":           _v("summaryDetail",         "dividendYield"),
            "beta":                    _v("summaryDetail",         "beta"),
            "fiftyTwoWeekHigh":        _v("summaryDetail",         "fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow":         _v("summaryDetail",         "fiftyTwoWeekLow"),
            "returnOnEquity":          _v("financialData",         "returnOnEquity"),
            "returnOnAssets":          _v("financialData",         "returnOnAssets"),
            "debtToEquity":            _v("financialData",         "debtToEquity"),
            "revenueGrowth":           _v("financialData",         "revenueGrowth"),
            "earningsGrowth":          _v("financialData",         "earningsGrowth"),
            "grossMargins":            _v("financialData",         "grossMargins"),
            "operatingMargins":        _v("financialData",         "operatingMargins"),
            "currentPrice":            _v("financialData",         "currentPrice"),
            "targetMeanPrice":         _v("financialData",         "targetMeanPrice"),
            "recommendationKey":       _v("financialData",         "recommendationKey"),
            "numberOfAnalystOpinions": _v("financialData",         "numberOfAnalystOpinions"),
            "shortRatio":              _v("defaultKeyStatistics",  "shortRatio"),
        }
    except Exception as exc:
        logger.warning("get_fundamentals(%s) failed (non-fatal): %s", ticker, exc)
        return {"ticker": ticker}   # screener/analyzer continue with price-only signals


# ── Quick quote ───────────────────────────────────────────────────────────────

def get_quote(ticker: str) -> dict:
    """Return current price, previousClose, volume."""
    params = {
        "period1": int(time.time()) - 86400,
        "period2": int(time.time()),
        "interval": "1m",
    }
    try:
        data = _get(_YF_CHART.format(ticker=ticker), params)
        meta = data["chart"]["result"][0]["meta"]
        return {
            "ticker":        ticker,
            "price":         meta.get("regularMarketPrice"),
            "previousClose": meta.get("previousClose") or meta.get("chartPreviousClose"),
            "volume":        meta.get("regularMarketVolume"),
        }
    except Exception as exc:
        logger.error("get_quote(%s): %s", ticker, exc)
        return {"ticker": ticker}


# ── Batch helper ──────────────────────────────────────────────────────────────

def batch_price_history(tickers: list[str], years: int = 3) -> dict[str, pd.DataFrame]:
    """Fetch price history for multiple tickers. Returns {ticker: DataFrame}."""
    results = {}
    for ticker in tickers:
        df = get_price_history(ticker, years=years)
        if df is not None and not df.empty:
            results[ticker] = df
        time.sleep(0.15)
    return results
