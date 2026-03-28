"""
Daily runner: entry point for the automated trading agent.

Modes
-----
  python daily_runner.py                  # run once immediately
  python daily_runner.py --schedule       # run daily at DAILY_RUN_TIME (config)
  python daily_runner.py --dry-run        # analyse only, no trades
  python daily_runner.py --optimize       # run optimizer once and exit
  python daily_runner.py --dry-run --schedule

Schedule
--------
  - Daily at DAILY_RUN_TIME : trading cycle (screen → alpha → analyse → backtest → trade)
  - Every Sunday at 18:00   : self-improvement optimizer (review performance, tune params)
  - Daily (after trading)   : fill forward returns for past signal records

For production, run inside a Docker container, systemd service, or cron job.
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime, date

import schedule

from config import DAILY_RUN_TIME, REPORTS_DIR

# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(log_dir: str = "logs") -> None:
    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"trading_{date_str}.log")
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    for noisy in ("urllib3", "httpx", "httpcore", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Single trading run ────────────────────────────────────────────────────────

def run_once(dry_run: bool = False) -> None:
    from agent import run_agent, save_report
    logger = logging.getLogger("daily_runner")
    start  = datetime.utcnow()
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║   DAILY TRADING CYCLE STARTING               ║")
    logger.info("║   %s UTC  (%s)         ║",
                start.strftime("%Y-%m-%d %H:%M"),
                "DRY RUN" if dry_run else "LIVE PAPER TRADING")
    logger.info("╚══════════════════════════════════════════════╝")
    try:
        summary = run_agent(dry_run=dry_run)
        report_path = save_report(summary)
        elapsed = (datetime.utcnow() - start).total_seconds()
        logger.info("Cycle complete in %.0fs  |  Report: %s", elapsed, report_path)
        print("\n" + "=" * 60)
        print("AGENT FINAL SUMMARY")
        print("=" * 60)
        print(summary)
        print("=" * 60 + "\n")
    except Exception as exc:
        logger.exception("Trading cycle failed: %s", exc)


# ── Weekly self-improvement run ───────────────────────────────────────────────

def run_optimizer_cycle() -> None:
    """
    Weekly optimizer: reviews the last 90 days of performance,
    backtests parameter candidates, and lets Claude rewrite strategy params.
    """
    from optimizer import run_optimizer, update_forward_returns
    from data_fetcher import batch_price_history
    from config import UNIVERSE_TICKERS, BACKTEST_YEARS
    logger = logging.getLogger("optimizer_runner")

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║   WEEKLY OPTIMIZER STARTING                  ║")
    logger.info("╚══════════════════════════════════════════════╝")

    try:
        # Fetch price data once — share between forward-return fill and backtest
        logger.info("Fetching price data for optimizer…")
        price_data = batch_price_history(UNIVERSE_TICKERS[:30], years=BACKTEST_YEARS)

        # Fill forward returns for past signal records
        n_updated = update_forward_returns(price_data)
        logger.info("Updated forward returns for %d records", n_updated)

        # Run the meta-agent optimizer
        result = run_optimizer(price_data=price_data)

        print("\n" + "=" * 60)
        print("OPTIMIZER SUMMARY")
        print("=" * 60)
        print(f"Update reason  : {result.get('update_reason', 'none')}")
        print(f"Recent Sharpe  : {result.get('recent_metrics', {}).get('sharpe', 'N/A')}")
        print(f"Recent CAGR    : {result.get('recent_metrics', {}).get('cagr_pct', 'N/A')}%")
        print(f"Signal accuracy: {result.get('signal_accuracy', {})}")
        print(f"\nClaude commentary:\n{result.get('commentary', '')}")
        print("=" * 60 + "\n")

    except Exception as exc:
        logger.exception("Optimizer cycle failed: %s", exc)


# ── Scheduled runner ──────────────────────────────────────────────────────────

def run_scheduled(dry_run: bool = False) -> None:
    logger = logging.getLogger("daily_runner")
    logger.info(
        "Scheduling:\n"
        "  Daily trading  : %s ET\n"
        "  Weekly optimize: Sunday 18:00 UTC\n"
        "Press Ctrl-C to stop.",
        DAILY_RUN_TIME,
    )

    # Daily trading job
    schedule.every().day.at(DAILY_RUN_TIME).do(run_once, dry_run=dry_run)

    # Weekly self-improvement (Sunday evening, before Monday open)
    schedule.every().sunday.at("18:00").do(run_optimizer_cycle)

    # Run trading immediately on startup
    logger.info("Running trading cycle immediately on startup…")
    run_once(dry_run=dry_run)

    while True:
        schedule.run_pending()
        time.sleep(30)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Automated Trading Agent\n"
            "  Screens stocks → researches alpha → deep analysis → backtest → trade\n"
            "  Self-improves weekly via performance review + parameter optimization"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help="Run on a daily schedule + weekly optimizer (default: run once and exit)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analyse and report but do NOT execute any trades",
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="Run the self-improvement optimizer once and exit",
    )
    parser.add_argument(
        "--log-dir", default="logs",
        help="Directory for log files (default: logs/)",
    )
    args = parser.parse_args()

    _setup_logging(args.log_dir)

    from config import ANTHROPIC_API_KEY
    if not ANTHROPIC_API_KEY:
        print(
            "ERROR: ANTHROPIC_API_KEY is not set.\n"
            "Copy .env.example to .env and add your key."
        )
        sys.exit(1)

    if args.optimize:
        run_optimizer_cycle()
    elif args.schedule:
        run_scheduled(dry_run=args.dry_run)
    else:
        run_once(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
