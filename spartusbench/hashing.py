"""Reproducibility hashing for SpartusBench.

Every benchmark run carries version hashes so any result can be re-derived
from the same data + config + seed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def week_idx_to_year_week(week_idx: int) -> Tuple[int, int]:
    """Convert a linear week index to (year, week_number).

    Assumes week 0 = 2015 week 1, matching the training system's convention.
    """
    # M5 data starts 2015. ~52 weeks per year.
    year = 2015 + week_idx // 52
    week = (week_idx % 52) + 1
    return year, week


def compute_data_manifest_hash(
    week_indices: List[int],
    data_dir: Path = Path("storage/data"),
) -> str:
    """Hash of (file_path, file_size, mtime) for all data files used."""
    entries = []
    for week_idx in sorted(week_indices):
        year, wk = week_idx_to_year_week(week_idx)
        for tf in ["M5", "H1", "H4", "D1"]:
            path = data_dir / str(year) / f"week_{wk:02d}_{tf}.parquet"
            if path.exists():
                stat = path.stat()
                entries.append(f"{path}|{stat.st_size}|{stat.st_mtime_ns}")
        # Feature cache
        feat_path = Path(f"storage/features/{year}/week_{wk:02d}_features.parquet")
        if feat_path.exists():
            stat = feat_path.stat()
            entries.append(f"{feat_path}|{stat.st_size}|{stat.st_mtime_ns}")
    manifest = "\n".join(entries)
    return hashlib.sha256(manifest.encode()).hexdigest()


def compute_split_hash(
    train_weeks: List[int],
    val_weeks: List[int],
    test_weeks: List[int],
    purge_gap: int = 2,
) -> str:
    """Hash of the exact train/val/test partition."""
    payload = json.dumps({
        "train": sorted(train_weeks),
        "val": sorted(val_weeks),
        "test": sorted(test_weeks),
        "purge_gap": purge_gap,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def compute_feature_hash(config: Any) -> str:
    """Hash of feature set identity (protects against reordering/additions)."""
    payload = json.dumps({
        "num_features": config.num_features,
        "obs_dim": config.obs_dim,
        "frame_stack": config.frame_stack,
        "market_feature_names": list(config.market_feature_names),
        "norm_exempt_features": list(config.norm_exempt_features),
        "account_feature_names": list(getattr(config, 'account_feature_names', [])),
        "memory_feature_names": list(getattr(config, 'memory_feature_names', [])),
        "norm_window": getattr(config, 'norm_window', 200),
        "norm_clip": getattr(config, 'norm_clip', 5.0),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def compute_config_hash(config: Any) -> str:
    """Hash of all config fields that affect evaluation outcomes."""
    fields = {
        "initial_balance": config.initial_balance,
        "direction_threshold": getattr(config, 'direction_threshold', 0.3),
        "min_conviction": getattr(config, 'min_conviction', 0.15),
        "max_dd": config.max_dd,
        "daily_trade_hard_cap": config.daily_trade_hard_cap,
        "risk_per_trade": getattr(config, 'max_risk_pct', 0.02),
        "spread_london_pips": config.spread_london_pips,
        "spread_ny_pips": config.spread_ny_pips,
        "spread_asia_pips": config.spread_asia_pips,
        "spread_off_hours_pips": config.spread_off_hours_pips,
        "slippage_mean_pips": getattr(config, 'slippage_mean_pips', 0.5),
        "slippage_std_pips": getattr(config, 'slippage_std_pips', 0.3),
        "commission_per_lot": config.commission_per_lot,
        "pip_price": getattr(config, 'pip_price', 0.1),
        "trade_tick_value": config.trade_tick_value,
        "trade_tick_size": config.trade_tick_size,
        "observation_noise_std": 0.0,
        "spread_jitter": 0.0,
        "slippage_jitter": 0.0,
    }
    payload = json.dumps(fields, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def compute_model_file_hash(model_path: Path) -> str:
    """SHA256 of the model ZIP file."""
    if not model_path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_result_hash(result_json: str) -> str:
    """SHA256 of the results JSON for tamper detection."""
    return hashlib.sha256(result_json.encode()).hexdigest()
