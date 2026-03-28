"""
monitor.py — live terminal dashboard for the trading agent.

Run any time to see current status:
    python monitor.py            # snapshot and exit
    python monitor.py --watch    # refresh every 60 seconds
    python monitor.py --trades   # show full trade history
    python monitor.py --report   # print latest daily report
    python monitor.py --params   # show current strategy parameters
    python monitor.py --compare  # before/after improvement per optimizer version
    python monitor.py --signals  # which signals are most predictive over time
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


# ── Version comparison (before/after each optimizer run) ─────────────────────

def show_comparison():
    """
    Split performance history by strategy params version and show
    key metrics (CAGR, Sharpe, win rate, drawdown) for each version
    so you can see whether the optimizer actually improved things.
    """
    history = _load_history(180)
    if len(history) < 5:
        print(_yellow("  Not enough history yet (need 5+ days). Run the agent for a few weeks first."))
        return

    import numpy as np

    # Group records by params_version
    versions = {}
    for r in history:
        v = r.get("params_version", 1)
        versions.setdefault(v, []).append(r)

    print(_bold("─" * 72))
    print(_bold("  PERFORMANCE BY STRATEGY VERSION  (did the optimizer help?)"))
    print(_bold("─" * 72))
    print(f"  {'Ver':<5} {'Period':<24} {'Days':<6} {'Return':>8} "
          f"{'CAGR':>7} {'Sharpe':>8} {'MaxDD':>8} {'Reason'}")
    print("  " + "─" * 70)

    prev_equity = None
    for ver in sorted(versions.keys()):
        records  = versions[ver]
        equities = [r["portfolio_equity"] for r in records if "portfolio_equity" in r]
        if len(equities) < 2:
            continue

        start_date = records[0].get("date", "?")
        end_date   = records[-1].get("date", "?")
        n_days     = len(equities)

        total_ret = equities[-1] / equities[0] - 1
        n_years   = n_days / 252
        cagr      = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
        rets      = [equities[i] / equities[i-1] - 1 for i in range(1, len(equities))]
        sharpe    = (sum(rets) / len(rets)) / (max((sum(r**2 for r in rets)/len(rets))**0.5, 1e-9)) * (252**0.5)
        peak      = equities[0]
        max_dd    = 0.0
        for v in equities:
            peak   = max(peak, v)
            max_dd = min(max_dd, (v - peak) / peak)

        ret_str = _green(f"{total_ret*100:+.1f}%") if total_ret >= 0 else _red(f"{total_ret*100:+.1f}%")
        cagr_str = _green(f"{cagr*100:+.1f}%") if cagr >= 0 else _red(f"{cagr*100:+.1f}%")
        sh_str   = _green(f"{sharpe:.2f}") if sharpe >= 0.5 else (_yellow(f"{sharpe:.2f}") if sharpe >= 0 else _red(f"{sharpe:.2f}"))
        dd_str   = _red(f"{max_dd*100:.1f}%")

        # Show improvement arrow vs previous version
        arrow = ""
        if prev_equity is not None:
            prev_cagr = prev_equity
            arrow = _green(" ▲ improved") if cagr > prev_cagr else _red(" ▼ declined")
        prev_equity = cagr

        # Get update reason from params history
        params = _load_json("data/strategy_params.json") or {}
        reason = ""
        if params.get("version") == ver:
            reason = (params.get("update_reason") or "")[:35]

        print(f"  v{ver:<4} {start_date} → {end_date}  {n_days:<6} {ret_str:>16} "
              f"{cagr_str:>15} {sh_str:>16} {dd_str:>16}  {_cyan(reason)}{arrow}")

    print()

    # Summary: best version
    best_ver = None
    best_sharpe = -999
    for ver, records in versions.items():
        equities = [r["portfolio_equity"] for r in records if "portfolio_equity" in r]
        if len(equities) < 2:
            continue
        rets   = [equities[i] / equities[i-1] - 1 for i in range(1, len(equities))]
        std    = max((sum(r**2 for r in rets)/len(rets))**0.5, 1e-9)
        sharpe = (sum(rets)/len(rets)) / std * (252**0.5)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_ver    = ver

    if best_ver:
        print(f"  {_bold('Best version so far:')} v{best_ver}  (Sharpe {best_sharpe:.2f})")
    print(_bold("─" * 72))
    print()

    # Show what changed between versions
    opt_reports = sorted(Path("reports").glob("optimizer_*.txt")) if Path("reports").exists() else []
    if opt_reports:
        print(_bold("  OPTIMIZER CHANGE LOG"))
        print("  " + "─" * 50)
        for rpt in opt_reports[-5:]:
            lines = rpt.read_text().splitlines()
            date  = rpt.stem.replace("optimizer_", "")
            # Extract update reason line
            for i, line in enumerate(lines):
                if "UPDATE REASON" in line and i + 1 < len(lines):
                    reason = lines[i + 1].strip()[:70]
                    print(f"  {_cyan(date)}  {reason}")
                    break
        print()


# ── Signal accuracy over time ─────────────────────────────────────────────────

def show_signals():
    """
    Show which alpha signals have been most predictive of actual returns,
    and how their accuracy has changed over time (improving = optimizer is learning).
    """
    signal_log = _load_jsonl("data/signal_log.jsonl", 1000)
    resolved   = [r for r in signal_log if r.get("forward_return_1m") is not None]

    print(_bold("─" * 65))
    print(_bold("  SIGNAL ACCURACY  (correlation with actual 1-month returns)"))
    print(_bold("─" * 65))

    if len(resolved) < 10:
        print(_yellow(f"  Only {len(resolved)} resolved signals so far."))
        print(_yellow("  Need ~21 trading days before forward returns are filled in."))
        print(_yellow("  Check back after the first optimizer run (Sunday 18:00 UTC)."))
        print()
        # Still show what signals exist
        if signal_log:
            print(f"  {len(signal_log)} signals logged, {len(resolved)} resolved so far.")
            earliest = signal_log[0].get("signal_date", "?")
            latest   = signal_log[-1].get("signal_date", "?")
            print(f"  Date range: {earliest} → {latest}")
        return

    import numpy as np

    signal_names = [
        "cross_momentum", "sharpe_momentum", "trend_following",
        "analyst_upside", "value_quality", "volume_surge", "short_reversal",
    ]

    # Load current weights for comparison
    params  = _load_json("data/strategy_params.json") or {}
    weights = params.get("signal_weights", {})

    print(f"  {'Signal':<22} {'Corr':>6}  {'Weight':>7}  {'Bar':<30}  Verdict")
    print("  " + "─" * 75)

    correlations = {}
    for sig in signal_names:
        xs, ys = [], []
        for r in resolved:
            val = r.get("signals", {}).get(sig)
            ret = r.get("forward_return_1m")
            if val is not None and ret is not None:
                xs.append(float(val))
                ys.append(float(ret))
        if len(xs) >= 5:
            corr = float(np.corrcoef(xs, ys)[0, 1])
            correlations[sig] = corr

    # Sort by absolute correlation
    sorted_sigs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    for sig, corr in sorted_sigs:
        weight  = weights.get(sig, 0.0)
        bar_len = int(abs(corr) * 25)
        bar_chr = "█" if corr >= 0 else "░"
        bar     = bar_chr * bar_len

        if corr >= 0.15:
            verdict = _green("predictive ✓")
            bar_col = _green(bar)
        elif corr >= 0.05:
            verdict = _yellow("weak signal")
            bar_col = _yellow(bar)
        elif corr >= -0.05:
            verdict = "  noise    "
            bar_col = bar
        else:
            verdict = _red("inverse    ")
            bar_col = _red(bar)

        w_str = f"{weight:.3f}" if weight else "default"
        print(f"  {sig:<22} {corr:>+6.3f}  {w_str:>7}  {bar_col:<40}  {verdict}")

    print()
    print(f"  Based on {len(resolved)} resolved signals  ({len(signal_log) - len(resolved)} pending forward returns)")
    print()

    # Show if optimizer has been adjusting weights in the right direction
    if weights and correlations:
        aligned = sum(
            1 for sig, corr in correlations.items()
            if weights.get(sig, 0) > 0.10 and corr > 0.05   # high weight + predictive
            or weights.get(sig, 0) < 0.10 and corr < 0.05   # low weight + weak
        )
        total = len(correlations)
        pct   = aligned / total * 100
        msg   = (
            _green(f"  Optimizer weights are aligned with signal accuracy ({aligned}/{total} correct direction)")
            if pct >= 60 else
            _yellow(f"  Optimizer still learning ({aligned}/{total} weights aligned with accuracy)")
        )
        print(msg)
    print(_bold("─" * 65))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trading agent monitor")
    parser.add_argument("--watch",     action="store_true", help="Refresh every 60s")
    parser.add_argument("--trades",    action="store_true", help="Show trade history")
    parser.add_argument("--report",    action="store_true", help="Print latest daily report")
    parser.add_argument("--params",    action="store_true", help="Show strategy parameters")
    parser.add_argument("--optimizer", action="store_true", help="Show latest optimizer run")
    parser.add_argument("--compare",   action="store_true", help="Before/after improvement per optimizer version")
    parser.add_argument("--signals",   action="store_true", help="Signal accuracy vs actual returns")
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
    if args.compare:
        os.system("")
        show_comparison()
        return
    if args.signals:
        os.system("")
        show_signals()
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
