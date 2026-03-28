"""
monitor.py — live terminal dashboard for the trading agent.

Run any time to see current status:
    python monitor.py            # snapshot and exit
    python monitor.py --watch    # refresh every 60 seconds
    python monitor.py --trades   # show full trade history
    python monitor.py --report   # print latest daily report
    python monitor.py --params   # show current strategy parameters
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path


# ── Colour helpers (work on Windows 10+ with ANSI enabled) ───────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _green(s):  return f"{GREEN}{s}{RESET}"
def _red(s):    return f"{RED}{s}{RESET}"
def _yellow(s): return f"{YELLOW}{s}{RESET}"
def _cyan(s):   return f"{CYAN}{s}{RESET}"
def _bold(s):   return f"{BOLD}{s}{RESET}"

def _pnl_color(val):
    if val is None: return "N/A"
    return _green(f"+{val:.2f}%") if val >= 0 else _red(f"{val:.2f}%")


# ── Load data files ───────────────────────────────────────────────────────────

def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def _load_jsonl(path, n=50):
    if not os.path.exists(path):
        return []
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except Exception:
                    pass
    return lines[-n:]

def _load_history(n=90):
    return _load_jsonl("data/performance_history.jsonl", n)

def _load_trades(n=50):
    return _load_jsonl("data/trades.log", n)

def _load_params():
    p = _load_json("data/strategy_params.json")
    return p if p else {}

def _latest_report():
    reports = sorted(Path("reports").glob("report_*.txt")) if Path("reports").exists() else []
    if not reports:
        return None, None
    path = reports[-1]
    return path.name, path.read_text()


# ── Portfolio panel ───────────────────────────────────────────────────────────

def show_portfolio():
    data = _load_json("data/portfolio.json")
    if not data:
        print(_yellow("  No portfolio data found. Run the agent first."))
        return

    cash      = data.get("cash", 0)
    positions = data.get("positions", {})
    invested  = sum(
        p.get("shares", 0) * p.get("last_price", p.get("avg_cost", 0))
        for p in positions.values()
    )
    equity    = cash + invested
    updated   = data.get("last_updated", "unknown")[:19].replace("T", " ")

    print(_bold("─" * 68))
    print(_bold(f"  PORTFOLIO SNAPSHOT   (updated {updated} UTC)"))
    print(_bold("─" * 68))
    print(f"  {'Cash':<20} ${cash:>14,.2f}")
    print(f"  {'Invested':<20} ${invested:>14,.2f}")
    print(f"  {'Total Equity':<20} ${equity:>14,.2f}  {_bold('←')}")
    print()

    if not positions:
        print(_yellow("  No open positions."))
    else:
        print(f"  {'Ticker':<8} {'Shares':>8} {'Avg Cost':>10} "
              f"{'Last Price':>11} {'Value':>12} {'P&L %':>8}")
        print("  " + "─" * 62)
        for ticker, pos in sorted(positions.items()):
            shares     = pos.get("shares", 0)
            avg_cost   = pos.get("avg_cost", 0)
            last_price = pos.get("last_price", avg_cost)
            value      = shares * last_price
            pnl_pct    = (last_price / avg_cost - 1) * 100 if avg_cost else 0
            pnl_str    = _green(f"+{pnl_pct:.1f}%") if pnl_pct >= 0 else _red(f"{pnl_pct:.1f}%")
            print(f"  {ticker:<8} {shares:>8.2f} {avg_cost:>10.2f} "
                  f"{last_price:>11.2f} {value:>12,.2f} {pnl_str:>16}")

    print(_bold("─" * 68))


# ── Equity curve (sparkline) ──────────────────────────────────────────────────

def show_equity_curve():
    history = _load_history(60)
    if len(history) < 3:
        print(_yellow("  Equity curve: not enough history yet (need 3+ days)."))
        return

    equities = [r["portfolio_equity"] for r in history if "portfolio_equity" in r]
    if len(equities) < 2:
        return

    # Simple ASCII sparkline
    lo, hi = min(equities), max(equities)
    span    = hi - lo or 1
    chars   = " ▁▂▃▄▅▆▇█"
    spark   = "".join(chars[int((v - lo) / span * 8)] for v in equities[-40:])

    total_ret = (equities[-1] / equities[0] - 1) * 100
    color     = _green if total_ret >= 0 else _red

    print(_bold("  EQUITY CURVE (last 60 days)"))
    print(f"  ${equities[0]:,.0f} ▶  {spark}  ▶ ${equities[-1]:,.0f}")
    print(f"  Period return: {color(f'{total_ret:+.2f}%')}  "
          f"over {len(equities)} trading days")
    print()


# ── Performance metrics ───────────────────────────────────────────────────────

def show_metrics():
    history = _load_history(90)
    if len(history) < 5:
        print(_yellow("  Performance metrics: need 5+ days of history."))
        return

    import numpy as np
    equities = [r["portfolio_equity"] for r in history if "portfolio_equity" in r]
    eq   = equities
    rets = [eq[i] / eq[i-1] - 1 for i in range(1, len(eq))]

    total_ret = eq[-1] / eq[0] - 1
    n_years   = len(eq) / 252
    cagr      = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    ann_ret   = sum(rets) / len(rets) * 252
    ann_vol   = (sum(r**2 for r in rets) / len(rets)) ** 0.5 * (252 ** 0.5)
    sharpe    = ann_ret / ann_vol if ann_vol > 0 else 0

    peak   = eq[0]
    max_dd = 0.0
    for v in eq:
        peak   = max(peak, v)
        max_dd = min(max_dd, (v - peak) / peak)

    print(_bold("  PERFORMANCE METRICS"))
    print(f"  {'Total Return':<22} {_pnl_color(total_ret * 100)}")
    print(f"  {'CAGR (annualised)':<22} {_pnl_color(cagr * 100)}")
    print(f"  {'Sharpe Ratio':<22} {sharpe:.3f}")
    print(f"  {'Max Drawdown':<22} {_red(f'{max_dd*100:.2f}%')}")
    print(f"  {'Days tracked':<22} {len(eq)}")
    print()


# ── Recent trades ─────────────────────────────────────────────────────────────

def show_trades(n=15):
    trades = _load_trades(n)
    if not trades:
        print(_yellow("  No trades recorded yet."))
        return

    print(_bold(f"  RECENT TRADES (last {len(trades)})"))
    print(f"  {'Date':<12} {'Action':<5} {'Ticker':<8} "
          f"{'Shares':>8} {'Price':>8} {'Value':>10} {'Reason':<18} Status")
    print("  " + "─" * 76)
    for t in reversed(trades):
        date   = t.get("ts", "")[:10]
        action = t.get("action", "")
        ticker = t.get("ticker", "")
        shares = t.get("shares", 0)
        price  = t.get("price", 0)
        value  = t.get("value", 0)
        reason = t.get("reason", "")
        status = t.get("status", "")

        a_str = _green(f"{action:<5}") if action == "BUY" else _red(f"{action:<5}")
        s_str = _green("filled") if status == "filled" else _red(status)
        print(f"  {date:<12} {a_str} {ticker:<8} "
              f"{shares:>8.2f} {price:>8.2f} {value:>10,.0f} {reason:<18} {s_str}")
    print()


# ── Strategy parameters ───────────────────────────────────────────────────────

def show_params():
    params = _load_params()
    if not params:
        print(_yellow("  No saved strategy params. Using defaults."))
        return

    version = params.get("version", 1)
    updated = params.get("updated_at", "never")[:19].replace("T", " ")
    reason  = params.get("update_reason", "initial defaults")

    print(_bold(f"  STRATEGY PARAMETERS  (v{version}, updated {updated} UTC)"))
    print(f"  Reason: {_cyan(reason)}")
    print()
    print(f"  {'max_positions':<28} {params.get('max_positions', 'N/A')}")
    print(f"  {'max_position_pct':<28} {params.get('max_position_pct', 'N/A')}")
    print(f"  {'cash_reserve':<28} {params.get('cash_reserve', 'N/A')}")
    print(f"  {'stop_loss_pct':<28} {params.get('stop_loss_pct', 'N/A')}")
    print(f"  {'take_profit_pct':<28} {params.get('take_profit_pct', 'N/A')}")
    print(f"  {'momentum_lookback':<28} {params.get('momentum_lookback', 'N/A')} days")
    print(f"  {'rsi_overbought':<28} {params.get('rsi_overbought', 'N/A')}")
    print()
    print("  Signal weights:")
    sw = params.get("signal_weights", {})
    for sig, w in sorted(sw.items(), key=lambda x: -x[1]):
        bar = "█" * int(w * 40)
        print(f"    {sig:<22} {w:.3f}  {_cyan(bar)}")
    print()


# ── Last agent picks ──────────────────────────────────────────────────────────

def show_last_picks():
    history = _load_history(5)
    if not history:
        return
    last = history[-1]
    picks = last.get("agent_picks", [])
    date  = last.get("date", "unknown")
    if picks:
        print(_bold(f"  LAST AGENT PICKS  ({date})"))
        print("  " + "  ".join(_cyan(t) for t in picks))
        print()


# ── Full report ───────────────────────────────────────────────────────────────

def show_report():
    name, content = _latest_report()
    if not content:
        print(_yellow("  No reports found yet. Run the agent first."))
        return
    print(_bold(f"  LATEST REPORT: {name}"))
    print("─" * 68)
    print(content)


# ── Optimizer history ─────────────────────────────────────────────────────────

def show_optimizer_history():
    reports = sorted(Path("reports").glob("optimizer_*.txt")) if Path("reports").exists() else []
    if not reports:
        print(_yellow("  No optimizer runs yet (runs weekly after 3+ weeks of data)."))
        return
    last = reports[-1]
    print(_bold(f"  LAST OPTIMIZER RUN: {last.name}"))
    print("─" * 68)
    # Print just the first 40 lines
    lines = last.read_text().splitlines()[:40]
    print("\n".join(lines))
    if len(last.read_text().splitlines()) > 40:
        print("  … (truncated, open file for full report)")


# ── Main dashboard ────────────────────────────────────────────────────────────

def dashboard():
    # Enable ANSI on Windows
    os.system("")

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    print()
    print(_bold(_cyan("╔══════════════════════════════════════════════════════════════════╗")))
    print(_bold(_cyan(f"║  TRADING AGENT MONITOR   {now:<42}║")))
    print(_bold(_cyan("╚══════════════════════════════════════════════════════════════════╝")))
    print()

    show_portfolio()
    print()
    show_equity_curve()
    show_metrics()
    show_last_picks()
    show_params()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trading agent monitor")
    parser.add_argument("--watch",     action="store_true", help="Refresh every 60s")
    parser.add_argument("--trades",    action="store_true", help="Show trade history")
    parser.add_argument("--report",    action="store_true", help="Print latest daily report")
    parser.add_argument("--params",    action="store_true", help="Show strategy parameters")
    parser.add_argument("--optimizer", action="store_true", help="Show latest optimizer run")
    parser.add_argument("--interval",  type=int, default=60, help="Watch refresh seconds")
    args = parser.parse_args()

    # Single-panel modes
    if args.report:
        os.system("")
        show_report()
        return
    if args.trades:
        os.system("")
        show_trades(50)
        return
    if args.params:
        os.system("")
        show_params()
        return
    if args.optimizer:
        os.system("")
        show_optimizer_history()
        return

    # Dashboard (with optional watch loop)
    if args.watch:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            dashboard()
            print(_yellow(f"  Auto-refresh every {args.interval}s — Ctrl+C to stop"))
            time.sleep(args.interval)
    else:
        dashboard()


if __name__ == "__main__":
    main()
