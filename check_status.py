"""Quick diagnostic: check what data files exist and their contents."""
import os
import json

files = [
    "data/crypto_signal_log.jsonl",
    "data/crypto_trades.jsonl",
    "data/crypto_equity.jsonl",
    "data/crypto_portfolio.json",
    "data/optimizer_history.jsonl",
    "data/signal_weights.json",
    "data/intraday_params.json",
]

for path in files:
    if not os.path.exists(path):
        print(f"MISSING  {path}")
        continue

    with open(path) as f:
        content = f.read().strip()

    if not content:
        print(f"EMPTY    {path}")
        continue

    lines = [l for l in content.splitlines() if l.strip()]

    if path.endswith(".jsonl"):
        print(f"OK  {len(lines):>6} lines  {path}")
        try:
            last = json.loads(lines[-1])
            ts = last.get("ts") or last.get("updated_at") or "?"
            print(f"           last ts: {ts}")
            # For signal log, show resolved count
            if "signal_log" in path:
                resolved = sum(1 for l in lines if '"fwd_15m": ' in l and '"fwd_15m": null' not in l)
                print(f"           resolved (fwd_15m filled): {resolved}")
        except Exception:
            pass
    else:
        print(f"OK  {len(content):>6} bytes  {path}")
        try:
            data = json.loads(content)
            ts = data.get("updated_at") or data.get("ts") or "?"
            print(f"           last ts: {ts}")
        except Exception:
            pass
