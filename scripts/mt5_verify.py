"""Verify P/L calculations against real MT5 broker data."""
import MetaTrader5 as mt5
from datetime import datetime, timedelta

if not mt5.initialize():
    print("MT5 init failed"); exit()

info = mt5.symbol_info("XAUUSD")
acc = mt5.account_info()

print("=== CORRECT P/L VERIFICATION ===")
for price_move in [0.10, 0.50, 1.00, 2.00, 5.00]:
    ticks = price_move / info.trade_tick_size
    mt5_pnl = ticks * info.trade_tick_value_profit * 0.01
    our_pnl = price_move * 0.01 * 100
    ratio = our_pnl / mt5_pnl if mt5_pnl > 0 else 0
    print(f"  Move: ${price_move:.2f} | MT5: {mt5_pnl:.4f} GBP | Ours: {our_pnl:.4f} | Ratio: {ratio:.3f}x")

usd_to_gbp = info.trade_tick_value_profit / (info.trade_tick_size * info.trade_contract_size)
print(f"\nUSD->GBP rate: {usd_to_gbp:.6f}")
print(f"Overstatement: {(1/usd_to_gbp - 1)*100:.1f}%")

print(f"\n=== VANTAGE COMMISSION MODEL ===")
print(f"BTCUSD trade history: Commission = 0.0, Fee = 0.0")
print(f"Vantage is SPREAD-ONLY (no per-lot commission)")
print(f"Our config: commission_per_lot = $7.00 <-- SHOULD BE $0")
print(f"Phantom cost over 1039 trades: ${1039*0.01*7:.2f}")

mt5.shutdown()
