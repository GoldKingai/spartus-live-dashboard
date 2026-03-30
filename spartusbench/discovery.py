"""Model discovery and loading for SpartusBench."""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .types import EvalBundle

log = logging.getLogger("spartusbench.discovery")

MODEL_DIR = Path("storage/models")
FEATURE_DIR = Path("storage/features")
DATA_DIR = Path("storage/data")


def discover_models(model_dir: Path = MODEL_DIR) -> List[Dict[str, Any]]:
    """Discover all available model files and their metadata."""
    models = []

    if not model_dir.exists():
        return models

    # Weekly checkpoints: spartus_week_NNNN.zip
    for p in sorted(model_dir.glob("spartus_week_*.zip")):
        match = re.match(r"spartus_week_(\d{4})\.zip", p.name)
        if match:
            week_num = int(match.group(1))
            model_id = f"W{week_num:04d}"
            models.append(_model_info(model_id, p))

    # Best model
    best = model_dir / "spartus_best.zip"
    if best.exists():
        models.append(_model_info("best", best))

    # Champion packages
    for p in sorted(model_dir.glob("spartus_champion_*.zip")):
        stem = p.stem.replace("spartus_champion_", "")
        models.append(_model_info(f"champion_{stem}", p))

    # Latest
    latest = model_dir / "spartus_latest.zip"
    if latest.exists():
        models.append(_model_info("latest", latest))

    return models


def _model_info(model_id: str, path: Path) -> Dict[str, Any]:
    """Build model info dict."""
    meta_path = path.with_suffix(".meta.json")
    reward_state_path = path.with_suffix(".reward_state.json")
    size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0

    return {
        "model_id": model_id,
        "path": str(path),
        "size_mb": round(size_mb, 1),
        "has_meta": meta_path.exists(),
        "has_reward_state": reward_state_path.exists(),
    }


def resolve_model_path(model_ref: str, model_dir: Path = MODEL_DIR) -> Tuple[str, Path]:
    """Resolve a model reference to (model_id, absolute_path).

    Accepts:
    - "W0170" -> storage/models/spartus_week_0170.zip
    - "best"  -> storage/models/spartus_best.zip
    - Explicit path -> that path
    """
    # Explicit path
    p = Path(model_ref)
    if p.exists() and p.suffix == ".zip":
        model_id = p.stem
        # Try to extract week number
        match = re.match(r"spartus_week_(\d{4})", p.stem)
        if match:
            model_id = f"W{int(match.group(1)):04d}"
        return model_id, p

    # Named references
    if model_ref.lower() == "best":
        path = model_dir / "spartus_best.zip"
        return "best", path

    if model_ref.lower() == "latest":
        path = model_dir / "spartus_latest.zip"
        return "latest", path

    # Week reference: W0170 or W170
    match = re.match(r"W?(\d+)", model_ref, re.IGNORECASE)
    if match:
        week_num = int(match.group(1))
        path = model_dir / f"spartus_week_{week_num:04d}.zip"
        return f"W{week_num:04d}", path

    # Champion reference
    if model_ref.lower().startswith("champion"):
        path = model_dir / f"spartus_{model_ref.lower()}.zip"
        return model_ref, path

    raise ValueError(f"Cannot resolve model reference: {model_ref}")


def load_model_for_benchmark(
    model_ref: str,
    model_dir: Path = MODEL_DIR,
) -> EvalBundle:
    """Load a model and all companion data needed for benchmarking.

    Uses CPU for deterministic evaluation (no GPU non-determinism).
    """
    from stable_baselines3 import SAC
    from src.config import TrainingConfig

    model_id, model_path = resolve_model_path(model_ref, model_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    log.info("Loading model %s from %s", model_id, model_path)

    # Load SB3 model on CPU
    model = SAC.load(str(model_path), device="cpu")

    # Load config
    config = _extract_config(model_path)

    # Validate dimensions
    obs_dim = model.observation_space.shape[0]
    if obs_dim != config.obs_dim:
        raise ValueError(
            f"Model obs_dim={obs_dim} != config.obs_dim={config.obs_dim}. "
            "Old checkpoint incompatible with current feature set."
        )

    # Load companion files
    meta_path = model_path.with_suffix(".meta.json")
    metadata = None
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    reward_state_path = model_path.with_suffix(".reward_state.json")
    reward_state = None
    if reward_state_path.exists():
        with open(reward_state_path, "r", encoding="utf-8") as f:
            reward_state = json.load(f)

    # Discover weeks and split
    val_weeks, test_weeks = discover_and_split_weeks(config)

    return EvalBundle(
        model=model,
        config=config,
        model_id=model_id,
        model_path=model_path,
        val_weeks=val_weeks,
        test_weeks=test_weeks,
        reward_state=reward_state,
        metadata=metadata,
    )


def _extract_config(model_path: Path) -> Any:
    """Extract config from model package or use defaults."""
    from src.config import TrainingConfig

    # Check for bundled config in deployment ZIP
    config_json = model_path.with_suffix(".config.json")
    if config_json.exists():
        log.info("Using bundled config from %s", config_json)
        # Could parse and construct TrainingConfig from JSON
        # For now, fall through to defaults

    # Check companion meta.json
    meta_path = model_path.with_suffix(".meta.json")
    if meta_path.exists():
        log.info("Found metadata at %s", meta_path)

    # Fallback: use current TrainingConfig defaults
    return TrainingConfig()


def discover_weeks(feature_dir: Path = FEATURE_DIR) -> List[int]:
    """Discover all available feature cache weeks as linear indices."""
    weeks = []
    if not feature_dir.exists():
        return weeks

    for year_dir in sorted(feature_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        try:
            year = int(year_dir.name)
        except ValueError:
            continue
        for f in sorted(year_dir.glob("week_*_features.parquet")):
            match = re.match(r"week_(\d+)_features\.parquet", f.name)
            if match:
                wk = int(match.group(1))
                # Convert to linear index: (year - 2015) * 52 + (wk - 1)
                week_idx = (year - 2015) * 52 + (wk - 1)
                weeks.append(week_idx)

    return sorted(weeks)


def discover_and_split_weeks(
    config: Any,
    feature_dir: Path = FEATURE_DIR,
    purge_gap: int = 2,
) -> Tuple[List[int], List[int]]:
    """Discover weeks and split into val/test sets.

    Split ratios (from training system):
    - Train: 70%
    - Val: 15%
    - Test: 15%
    - Purge gap between sets
    """
    all_weeks = discover_weeks(feature_dir)
    if not all_weeks:
        log.warning("No feature caches found in %s", feature_dir)
        return [], []

    n = len(all_weeks)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_weeks = all_weeks[:train_end]
    val_start = train_end + purge_gap
    val_weeks = all_weeks[val_start:val_end]
    test_start = val_end + purge_gap
    test_weeks = all_weeks[test_start:]

    log.info(
        "Week split: %d train, %d val, %d test (total %d, purge_gap=%d)",
        len(train_weeks), len(val_weeks), len(test_weeks), n, purge_gap,
    )

    return val_weeks, test_weeks


def load_week_features(week_idx: int) -> Any:
    """Load feature cache for a specific week index.

    Returns pandas DataFrame or raises FileNotFoundError.
    """
    import pandas as pd
    from .hashing import week_idx_to_year_week

    year, wk = week_idx_to_year_week(week_idx)
    cache_path = FEATURE_DIR / str(year) / f"week_{wk:02d}_features.parquet"

    if not cache_path.exists():
        raise FileNotFoundError(f"Feature cache missing: {cache_path}")

    return pd.read_parquet(cache_path)


def validate_feature_caches(
    week_indices: List[int],
    expected_columns: int = 54,
) -> Tuple[List[int], List[str]]:
    """Validate that feature caches exist and have correct column count.

    Returns (valid_weeks, warnings).
    """
    import pandas as pd
    from .hashing import week_idx_to_year_week

    valid = []
    warnings = []

    for week_idx in week_indices:
        year, wk = week_idx_to_year_week(week_idx)
        cache_path = FEATURE_DIR / str(year) / f"week_{wk:02d}_features.parquet"

        if not cache_path.exists():
            warnings.append(f"Missing: {cache_path}")
            continue

        try:
            df = pd.read_parquet(cache_path)
            # Count feature columns (exclude time/OHLCV metadata)
            feat_cols = [c for c in df.columns if c not in
                         ("open", "high", "low", "close", "volume",
                          "time", "datetime", "timestamp")]
            if len(feat_cols) < expected_columns:
                warnings.append(
                    f"Week {week_idx} ({year}/w{wk:02d}): "
                    f"{len(feat_cols)} features (expected >= {expected_columns})"
                )
            valid.append(week_idx)
        except Exception as e:
            warnings.append(f"Week {week_idx}: read error: {e}")

    return valid, warnings
