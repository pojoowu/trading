"""
Daily runner: entry point for the automated trading agent.

Modes
-----
  python daily_runner.py                  # run once immediately
  python daily_runner.py --schedule       # run daily at DAILY_RUN_TIME (config)
  python daily_runner.py --dry-run        # analyse only, no trades
  python daily_runner.py --dry-run --schedule

The scheduler uses the `schedule` library and blocks indefinitely.
For production, run this inside a Docker container, systemd service,
or a cron job (recommended).
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime

import schedule

from config import DAILY_RUN_TIME, REPORTS_DIR
from agent import run_agent, save_report

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
    # Quieten noisy libraries
    for noisy in ("urllib3", "httpx", "httpcore", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Single run ────────────────────────────────────────────────────────────────

def run_once(dry_run: bool = False) -> None:
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
        logger.info("╔══════════════════════════════════════════════╗")
        logger.info("║   CYCLE COMPLETE in %.0fs                   ║", elapsed)
        logger.info("║   Report: %-37s║", report_path)
        logger.info("╚══════════════════════════════════════════════╝")

        # Print report to stdout
        print("\n" + "=" * 60)
        print("AGENT FINAL SUMMARY")
        print("=" * 60)
        print(summary)
        print("=" * 60 + "\n")

    except Exception as exc:
        logger.exception("Trading cycle failed: %s", exc)


# ── Scheduled runner ──────────────────────────────────────────────────────────

def run_scheduled(dry_run: bool = False) -> None:
    logger = logging.getLogger("daily_runner")
    logger.info(
        "Scheduling daily run at %s (Eastern). Press Ctrl-C to stop.",
        DAILY_RUN_TIME,
    )

    def _job():
        run_once(dry_run=dry_run)

    schedule.every().day.at(DAILY_RUN_TIME).do(_job)

    # Also run immediately on start
    logger.info("Running immediately on startup…")
    _job()

    while True:
        schedule.run_pending()
        time.sleep(30)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated Trading Agent — daily stock screening, alpha research, and execution"
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help="Run on a daily schedule (default: run once and exit)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analyse and produce report but do NOT execute any trades",
    )
    parser.add_argument(
        "--log-dir", default="logs",
        help="Directory for log files (default: logs/)",
    )
    args = parser.parse_args()

    _setup_logging(args.log_dir)

    from config import ANTHROPIC_API_KEY
    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY is not set. Please copy .env.example to .env and fill it in.")
        sys.exit(1)

    if args.schedule:
        run_scheduled(dry_run=args.dry_run)
    else:
        run_once(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
