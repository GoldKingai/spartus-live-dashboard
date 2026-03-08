"""Generate feature_index.json -- maps 670 observation indices to feature names.

The 670-dim observation is 10 stacked copies (frame_stack=10) of 67 features.
Frame 0 is the OLDEST bar, frame 9 is the CURRENT bar.

Usage:
    python scripts/generate_feature_index.py

Output:
    storage/feature_index.json
"""
import json
from pathlib import Path

FRAME_STACK = 10
NUM_FEATURES = 67

# Canonical feature order -- must match _FULL_FEATURE_ORDER in feature_pipeline.py
FULL_FEATURE_ORDER = [
    # Group A: Price & Returns (7)
    "close_frac_diff", "returns_1bar", "returns_5bar", "returns_20bar",
    "bar_range", "close_position", "body_ratio",
    # Group B: Volatility (4)
    "atr_14_norm", "atr_ratio", "bb_width", "bb_position",
    # Group C: Momentum & Trend (6)
    "rsi_14", "macd_signal", "adx_14", "ema_cross", "price_vs_ema200", "stoch_k",
    # Group D: Volume (2)
    "volume_ratio", "obv_slope",
    # Group E: Multi-Timeframe Context (6)
    "h1_trend_dir", "h4_trend_dir", "d1_trend_dir",
    "h1_rsi", "mtf_alignment", "htf_momentum",
    # Group F: Time & Session (4)
    "hour_sin", "hour_cos", "day_of_week", "session_quality",
    # Upgrade 1: Correlated Instruments (11)
    "eurusd_returns_20", "eurusd_rsi_14", "eurusd_trend",
    "xagusd_returns_20", "xagusd_rsi_14",
    "usdjpy_returns_20", "usdjpy_trend",
    "us500_returns_20", "us500_rsi_14",
    "usoil_returns_20",
    "gold_silver_ratio_z",
    # Upgrade 2: Calendar & Events (6)
    "hours_to_next_high_impact", "hours_to_next_nfp_fomc",
    "in_event_window", "daily_event_density",
    "london_fix_proximity", "comex_session_active",
    # Upgrade 3: Spread & Liquidity (2)
    "spread_estimate_norm", "volume_spike",
    # Upgrade 4: Regime Detection (2)
    "corr_gold_usd_100", "corr_gold_spx_100",
    # Upgrade 5: Session Microstructure (4)
    "asian_range_norm", "asian_range_position",
    "session_momentum", "london_ny_overlap",
    # Group G: Account (8)
    "has_position", "position_side", "unrealized_pnl", "position_duration",
    "current_drawdown", "equity_ratio", "sl_distance_ratio", "profit_locked_pct",
    # Group H: Memory (5)
    "recent_win_rate", "similar_pattern_winrate",
    "trend_prediction_accuracy", "tp_hit_rate", "avg_sl_trail_profit",
]

assert len(FULL_FEATURE_ORDER) == NUM_FEATURES, (
    f"Expected {NUM_FEATURES} features, got {len(FULL_FEATURE_ORDER)}"
)

index_map = {}
for frame in range(FRAME_STACK):
    for feat_idx, feat_name in enumerate(FULL_FEATURE_ORDER):
        obs_idx = frame * NUM_FEATURES + feat_idx
        index_map[str(obs_idx)] = {
            "feature": feat_name,
            "frame": frame,
            "frame_label": "oldest" if frame == 0 else ("current" if frame == 9 else f"t-{9 - frame}"),
        }

output = {
    "description": "Maps 670-dim observation indices to feature names and frame positions",
    "num_features": NUM_FEATURES,
    "frame_stack": FRAME_STACK,
    "obs_dim": NUM_FEATURES * FRAME_STACK,
    "feature_order": FULL_FEATURE_ORDER,
    "index_map": index_map,
}

out_path = Path(__file__).resolve().parent.parent / "storage" / "feature_index.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"Written {out_path} ({len(index_map)} indices)")
