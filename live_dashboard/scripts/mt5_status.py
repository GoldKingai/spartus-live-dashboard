"""Quick MT5 live status snapshot — shows open positions, account, and
protection state so Claude (or the user) can diagnose issues without
needing the dashboard UI or screenshots.

Usage:
    python scripts/mt5_status.py
    python scripts/mt5_status.py --watch     # refresh every 5s
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

# --- path bootstrap ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 package not installed")
    sys.exit(1)


def _connect() -> bool:
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return False
    return True


def _fmt(v, decimals=2):
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def snapshot() -> dict:
    result = {}

    # Account info
    ai = mt5.account_info()
    if ai:
        result["account"] = {
            "balance": round(ai.balance, 2),
            "equity": round(ai.equity, 2),
            "margin": round(ai.margin, 2),
            "free_margin": round(ai.margin_free, 2),
            "profit": round(ai.profit, 2),
            "currency": ai.currency,
        }

    # Open positions
    positions = mt5.positions_get()
    if positions is None:
        positions = []
    result["positions"] = []
    for p in positions:
        tick = mt5.symbol_info_tick(p.symbol)
        current_price = tick.ask if p.type == 1 else tick.bid  # 1=SELL, 0=BUY
        entry = p.price_open
        sl = p.sl
        tp = p.tp
        side = "SHORT" if p.type == 1 else "LONG"
        pnl = p.profit
        pips_moved = (entry - current_price) if side == "SHORT" else (current_price - entry)
        sl_distance = (current_price - sl) if side == "SHORT" else (sl - current_price)
        result["positions"].append({
            "ticket": p.ticket,
            "symbol": p.symbol,
            "side": side,
            "lots": p.volume,
            "entry": round(entry, 2),
            "current_price": round(current_price, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "pnl_gbp": round(pnl, 2),
            "pips_moved": round(pips_moved, 2),
            "sl_distance_from_price": round(sl_distance, 2),
            "comment": p.comment,
            "magic": p.magic,
        })

    # Protection state from disk
    prot_file = BASE_DIR / "storage" / "protection_state.json"
    if prot_file.exists():
        try:
            prot = json.loads(prot_file.read_text(encoding="utf-8"))
            result["protection_state"] = prot
        except Exception:
            result["protection_state"] = {"error": "unreadable"}
    else:
        result["protection_state"] = None

    # Recent log tail
    log_file = BASE_DIR / "storage" / "logs" / "dashboard.log"
    if log_file.exists():
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        result["recent_log"] = lines[-20:]

    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    return result


def print_snapshot(data: dict):
    print("\n" + "=" * 60)
    print(f"  MT5 LIVE STATUS  {data['timestamp'][:19]} UTC")
    print("=" * 60)

    acc = data.get("account", {})
    if acc:
        print(f"\nACCOUNT ({acc.get('currency', '?')})")
        print(f"  Balance:     {_fmt(acc.get('balance'))}")
        print(f"  Equity:      {_fmt(acc.get('equity'))}")
        print(f"  Open P/L:    {_fmt(acc.get('profit'))}")

    positions = data.get("positions", [])
    if not positions:
        print("\nPOSITIONS: none open")
    else:
        print(f"\nPOSITIONS ({len(positions)} open)")
        for p in positions:
            print(f"\n  Ticket:  {p['ticket']}  {p['symbol']}  {p['side']}  {p['lots']} lots")
            print(f"  Entry:   {p['entry']:.2f}  ->  Current: {p['current_price']:.2f}")
            print(f"  SL:      {p['sl']:.2f}  (distance from price: {p['sl_distance_from_price']:.2f} pts)")
            print(f"  TP:      {p['tp']:.2f}")
            print(f"  P/L:     {p['pnl_gbp']:.2f} {acc.get('currency', 'GBP')}")
            print(f"  Moved:   {p['pips_moved']:.2f} pts in trade direction")
            if p['magic']:
                print(f"  Magic:   {p['magic']}")

    prot = data.get("protection_state")
    if prot:
        print("\nPROTECTION STATE (from disk)")
        for k, v in prot.items():
            if k != "saved_at":
                print(f"  {k}: {v}")
        print(f"  saved_at: {prot.get('saved_at', '—')[:19]}")

        # Derived values
        positions_list = data.get("positions", [])
        for p in positions_list:
            if p["ticket"] == prot.get("ticket"):
                r_dist = abs(prot.get("entry_price", 0) - prot.get("initial_sl", 0))
                mfe = prot.get("max_favorable", 0)
                r_val = prot.get("r_value_gbp", 0)
                r_mult = mfe / r_dist if r_dist > 0 else 0
                be_r = min(1.0, 2.0 / r_val) if r_val > 0 else 1.0
                lock_r = min(1.5, 3.0 / r_val) if r_val > 0 else 1.5
                trail_r = min(2.0, 4.0 / r_val) if r_val > 0 else 2.0
                print(f"\n  DERIVED:")
                print(f"  R-distance: {r_dist:.2f} pts  |  MFE: {mfe:.2f} pts  |  R-multiple: {r_mult:.3f}")
                print(f"  Effective thresholds: BE={be_r:.3f}R  Lock={lock_r:.3f}R  Trail={trail_r:.3f}R")
                sl_in_mt5 = p["sl"]
                sl_in_prot = prot.get("initial_sl", 0)
                print(f"  MT5 SL: {sl_in_mt5:.2f}  |  Initial SL: {sl_in_prot:.2f}")
                if abs(sl_in_mt5 - sl_in_prot) > 0.1:
                    print(f"  [OK] SL HAS MOVED since initial ({sl_in_prot:.2f} -> {sl_in_mt5:.2f})")
                else:
                    print(f"  [!!] SL has NOT moved from initial")

    print("\nRECENT LOG (last 10 lines)")
    for line in data.get("recent_log", [])[-10:]:
        print(f"  {line}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="Refresh every 5 seconds")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if not _connect():
        sys.exit(1)

    try:
        while True:
            data = snapshot()
            if args.json:
                print(json.dumps(data, indent=2, default=str))
            else:
                print_snapshot(data)
            if not args.watch:
                break
            time.sleep(5)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
