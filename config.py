"""
Configuration for the Automated Trading Agent.
Load settings from environment variables or defaults.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Anthropic ───────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

# ── Broker / execution ───────────────────────────────────────────────────────
# Set BROKER=alpaca to use Alpaca paper trading. Default is "paper" (local sim).
BROKER = os.getenv("BROKER", "paper")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# ── Universe ─────────────────────────────────────────────────────────────────
# Comma-separated list of tickers to screen, or "SP500" to auto-fetch
UNIVERSE = os.getenv(
    "UNIVERSE",
    "AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,JPM,V,JNJ,"
    "UNH,XOM,PG,MA,HD,CVX,MRK,LLY,ABBV,PEP,"
    "KO,AVGO,COST,TMO,MCD,DHR,ACN,WMT,BAC,CRM,"
    "ADBE,CSCO,ABT,DIS,TXN,VZ,CMCSA,NEE,PM,"
    "INTC,AMD,QCOM,AMGN,HON,IBM,GS,MS,BLK,SPGI",
)
UNIVERSE_TICKERS = [t.strip() for t in UNIVERSE.split(",") if t.strip()]

# ── Screening thresholds ──────────────────────────────────────────────────────
MIN_MARKET_CAP_B = float(os.getenv("MIN_MARKET_CAP_B", "5"))       # $5B+
MIN_AVG_VOLUME = int(os.getenv("MIN_AVG_VOLUME", "500000"))         # 500K shares/day
MAX_STOCKS_AFTER_SCREEN = int(os.getenv("MAX_STOCKS_AFTER_SCREEN", "20"))

# ── Alpha signals ─────────────────────────────────────────────────────────────
MOMENTUM_LOOKBACK_DAYS = int(os.getenv("MOMENTUM_LOOKBACK_DAYS", "252"))  # ~1 year
SHORT_MOMENTUM_DAYS = int(os.getenv("SHORT_MOMENTUM_DAYS", "21"))          # 1 month
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", "9"))
BOLLINGER_PERIOD = int(os.getenv("BOLLINGER_PERIOD", "20"))
BOLLINGER_STD = float(os.getenv("BOLLINGER_STD", "2.0"))

# ── Backtesting ───────────────────────────────────────────────────────────────
BACKTEST_YEARS = int(os.getenv("BACKTEST_YEARS", "3"))
BACKTEST_INITIAL_CAPITAL = float(os.getenv("BACKTEST_INITIAL_CAPITAL", "100000"))
BACKTEST_COMMISSION = float(os.getenv("BACKTEST_COMMISSION", "0.001"))  # 0.1%

# ── Portfolio ─────────────────────────────────────────────────────────────────
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "10"))
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.15"))   # 15% per stock
MIN_POSITION_PCT = float(os.getenv("MIN_POSITION_PCT", "0.03"))   # 3% minimum
PORTFOLIO_CASH_RESERVE = float(os.getenv("PORTFOLIO_CASH_RESERVE", "0.05"))  # 5% cash

# ── Paper portfolio state ─────────────────────────────────────────────────────
PORTFOLIO_FILE = os.getenv("PORTFOLIO_FILE", "data/portfolio.json")
TRADES_LOG_FILE = os.getenv("TRADES_LOG_FILE", "data/trades.log")
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")

# ── Scheduler ─────────────────────────────────────────────────────────────────
# Time (HH:MM, 24h, Eastern) to run daily analysis
DAILY_RUN_TIME = os.getenv("DAILY_RUN_TIME", "09:30")
TIMEZONE = os.getenv("TIMEZONE", "US/Eastern")
