"""
Runner: entry point for the automated trading system.

Modes
-----
  python daily_runner.py                      # crypto minute trader (default)
  python daily_runner.py --dry-run            # analyse only, no trades
  python daily_runner.py --optimize           # run crypto optimizer once and exit
  python daily_runner.py --stocks             # legacy daily stock trader
  python daily_runner.py --stocks --schedule  # stock trader on daily schedule

Crypto mode (default):
  - Runs minute_trader 24/7 (1-min loop)
  - Runs crypto_optimizer every OPTIMIZER_INTERVAL_HOURS (default 4h)
  - Both run in the same process via threads

Self-improvement timeline:
  Hour 1  : first IC data available (60 × 20 coins resolved signals)
  Hour 4  : optimizer runs → weights updated → trader improves immediately
  Hour 8  : second optimizer pass → more refinement
  Day 1   : 28,800 observations → very reliable IC → near-optimal weights
"""
import argparse
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone

# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(log_dir: str = "logs") -> None:
    os.makedirs(log_dir, exist_ok=True)
    date_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file  = os.path.join(log_dir, f"trading_{date_str}.log")
    fmt       = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers  = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    for noisy in ("urllib3", "httpx", "httpcore", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Crypto mode ───────────────────────────────────────────────────────────────

def run_crypto(dry_run: bool = False):
    """
    Launch the minute trader + periodic optimizer in parallel threads.
    Blocks until Ctrl-C.
    """
    from config import CRYPTO_INITIAL_CASH, OPTIMIZER_INTERVAL_HOURS
    from minute_trader import run_forever
    from crypto_optimizer import run_crypto_optimizer

    logger = logging.getLogger("runner")
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║   CRYPTO TRADING SYSTEM STARTING             ║")
    logger.info("║   Mode: %-36s║", "DRY RUN" if dry_run else "PAPER TRADING")
    logger.info("║   Optimizer every %dh                        ║", OPTIMIZER_INTERVAL_HOURS)
    logger.info("╚══════════════════════════════════════════════╝")

    stop_event = threading.Event()

    # ── Optimizer thread ──────────────────────────────────────────────────────
    def optimizer_loop():
        interval_s = OPTIMIZER_INTERVAL_HOURS * 3600
        # Wait a bit before first run so trader has time to log some signals
        initial_wait = min(3600, interval_s)
        logger.info("Optimizer: first run in %.0f min", initial_wait / 60)
        time.sleep(initial_wait)

        while not stop_event.is_set():
            try:
                logger.info("--- OPTIMIZER STARTING ---")
                result = run_crypto_optimizer(dry_run=dry_run)
                n_resolved = result.get("n_resolved", 0) if isinstance(result, dict) else 0
                reason     = result.get("reason", "") if isinstance(result, dict) else ""
                logger.info("--- OPTIMIZER DONE | resolved=%s | %s ---", n_resolved, reason[:80])
            except Exception as exc:
                logger.exception("Optimizer error: %s", exc)
            stop_event.wait(interval_s)

    opt_thread = threading.Thread(target=optimizer_loop, daemon=True, name="optimizer")
    opt_thread.start()

    # ── Minute trader (main thread) ───────────────────────────────────────────
    try:
        run_forever(initial_cash=CRYPTO_INITIAL_CASH, dry_run=dry_run)
    except KeyboardInterrupt:
        logger.info("Shutting down…")
    finally:
        stop_event.set()


# ── One-shot optimizer ────────────────────────────────────────────────────────

def run_optimizer_once(dry_run: bool = False):
    from crypto_optimizer import run_crypto_optimizer
    logger = logging.getLogger("runner")
    result = run_crypto_optimizer(dry_run=dry_run)
    print("\n" + "=" * 60)
    print("OPTIMIZER RESULT")
    print("=" * 60)
    if isinstance(result, dict):
        print(f"Resolved signals : {result.get('n_resolved', result.get('reason', 'N/A'))}")
        print(f"Reason           : {result.get('reason', 'N/A')}")
        print(f"Backtest current : {result.get('bt_current', {})}")
        print(f"Backtest new     : {result.get('bt_suggested', {})}")
        print(f"Portfolio stats  : {result.get('eq_stats', {})}")
        if result.get("commentary"):
            print(f"\nClaude commentary:\n{result['commentary']}")
        print("\nTop signals by new weight:")
        nw = result.get("new_weights", {})
        if isinstance(nw, dict):
            for sig, w in sorted(nw.items(), key=lambda x: -x[1])[:8]:
                print(f"  {sig:<22} {w:.4f}")
    print("=" * 60)


# ── Legacy stock trader ───────────────────────────────────────────────────────

def run_stocks_once(dry_run: bool = False):
    from agent import run_agent, save_report
    logger = logging.getLogger("runner")
    start  = datetime.now(timezone.utc)
    try:
        summary     = run_agent(dry_run=dry_run)
        report_path = save_report(summary)
        elapsed     = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info("Stock cycle complete in %.0fs | report: %s", elapsed, report_path)
        print(f"\n{'='*60}\n{summary}\n{'='*60}")
    except Exception as exc:
        logger.exception("Stock cycle failed: %s", exc)


def run_stocks_scheduled(dry_run: bool = False):
    import schedule
    from config import DAILY_RUN_TIME
    logger = logging.getLogger("runner")
    logger.info("Scheduling stocks at %s ET daily. Ctrl-C to stop.", DAILY_RUN_TIME)
    schedule.every().day.at(DAILY_RUN_TIME).do(run_stocks_once, dry_run=dry_run)
    run_stocks_once(dry_run=dry_run)
    while True:
        schedule.run_pending()
        time.sleep(30)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Automated Trading System\n"
            "  Default: 24/7 crypto minute trader + IC-based self-improvement every 4h\n"
            "  --stocks: legacy daily stock screener + weekly optimizer"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run",   action="store_true", help="No trades, analysis only")
    parser.add_argument("--optimize",  action="store_true", help="Run optimizer once and exit")
    parser.add_argument("--stocks",    action="store_true", help="Use stock mode instead of crypto")
    parser.add_argument("--schedule",  action="store_true", help="(stocks mode) daily schedule")
    parser.add_argument("--log-dir",   default="logs",      help="Log directory")
    args = parser.parse_args()

    _setup_logging(args.log_dir)

    from config import ANTHROPIC_API_KEY
    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    if args.stocks:
        if args.optimize:
            from optimizer import run_optimizer
            run_optimizer()
        elif args.schedule:
            run_stocks_scheduled(dry_run=args.dry_run)
        else:
            run_stocks_once(dry_run=args.dry_run)
    elif args.optimize:
        run_optimizer_once(dry_run=args.dry_run)
    else:
        run_crypto(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
