"""Model export and packaging for live deployment.

Handles:
- Model packaging: bundles SB3 model + config + metadata + reward state +
  feature baseline + stress results into a single deployment ZIP.
- Feature baseline: computes per-feature mean/std from validation data
  for live drift detection.
- Stress results: bundles stress matrix JSON for training-vs-live comparison.
- ONNX export: stub for future MT5 native integration.

Usage (from training dashboard or script):
    exporter = ModelExporter(config)
    checks = exporter.validate_model()          # pre-flight checks
    out_path = exporter.package_model()          # create deployment ZIP
"""

import datetime
import json
import logging
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import TrainingConfig

log = logging.getLogger(__name__)


class ModelExporter:
    """Packages trained models for live dashboard deployment."""

    def __init__(self, config: TrainingConfig = None):
        self.cfg = config or TrainingConfig()

    # ── Validation ──────────────────────────────────────────────────────────

    def validate_model(
        self,
        model_path: Optional[str] = None,
        skip_load: bool = False,
    ) -> dict:
        """Run pre-packaging validation checks.

        Args:
            model_path: Path to SB3 .zip model. Defaults to best model.
            skip_load: If True, skip the expensive SAC.load() check.

        Returns:
            Dict of {check_name: (passed: bool, detail: str)}.
        """
        model_path = Path(model_path) if model_path else self.cfg.best_model_path
        meta_path = model_path.with_suffix(".meta.json")
        rs_sidecar = model_path.with_suffix(".reward_state.json")

        checks = {}

        # 1. Model file exists
        checks["model_file_exists"] = (
            model_path.exists(),
            str(model_path) if model_path.exists() else "NOT FOUND",
        )

        # 2. Meta or reward_state sidecar exists
        has_meta = meta_path.exists()
        has_sidecar = rs_sidecar.exists()
        checks["meta_file_exists"] = (
            has_meta or has_sidecar,
            str(meta_path) if has_meta else (
                str(rs_sidecar) if has_sidecar else "NOT FOUND"
            ),
        )

        # 3. Feature count
        checks["feature_count"] = (
            self.cfg.num_features == 67,
            f"{self.cfg.num_features} features (expected 67)",
        )

        # 4. Obs dim
        checks["obs_dim"] = (
            self.cfg.obs_dim == 670,
            f"obs_dim={self.cfg.obs_dim} (expected 670)",
        )

        # 5. Reward state present
        reward_present = False
        if has_meta:
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                reward_present = bool(meta.get("reward_state"))
            except Exception:
                pass
        if not reward_present and has_sidecar:
            reward_present = True
        checks["reward_state_present"] = (
            reward_present,
            "present" if reward_present else "MISSING",
        )

        # 6. Model loads OK (expensive)
        if not skip_load and model_path.exists():
            try:
                from stable_baselines3 import SAC
                SAC.load(str(model_path), device="cpu")
                checks["model_loads_ok"] = (True, "loads OK")
            except Exception as e:
                checks["model_loads_ok"] = (False, f"FAILED: {str(e)[:80]}")
        elif skip_load:
            pass  # omit from results — caller knows it was skipped
        else:
            checks["model_loads_ok"] = (False, "model file missing")

        # 7. Stress results available
        week_num = self._resolve_week(model_path)
        stress_path = self._find_stress_results(week_num)
        if stress_path:
            checks["stress_results"] = (True, stress_path.name)
        else:
            checks["stress_results"] = (
                False, f"stress_matrix_W{week_num:04d}.json not found")

        # 8. Feature baseline can be generated (val feature caches exist)
        val_weeks = self._discover_val_weeks()
        cached_count = sum(1 for _, _, p in val_weeks if p.exists())
        if cached_count >= 10:
            checks["feature_baseline"] = (
                True, f"{cached_count}/{len(val_weeks)} val weeks cached")
        elif cached_count > 0:
            checks["feature_baseline"] = (
                True, f"partial: {cached_count}/{len(val_weeks)} val weeks")
        else:
            checks["feature_baseline"] = (
                False, "no val feature caches found")

        # 9. Correlation baseline (same val caches, derived from #8)
        if cached_count >= 10:
            checks["correlation_baseline"] = (
                True, f"top-20 from {cached_count} val weeks")
        else:
            checks["correlation_baseline"] = (
                False, "needs val feature caches (see #8)")

        return checks

    # ── Packaging ───────────────────────────────────────────────────────────

    def package_model(
        self,
        model_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        output_filename: Optional[str] = None,
    ) -> str:
        """Bundle model + config + metadata + reward state into a deployment ZIP.

        Args:
            model_path: Path to SB3 .zip model. Defaults to best model.
            output_dir: Output directory. Defaults to cfg.model_dir.
            output_filename: Output filename. Auto-generated if None.

        Returns:
            Absolute path string of the created package ZIP.

        Raises:
            FileNotFoundError: If model file does not exist.
            ValueError: If model fails validation.
        """
        model_path = Path(model_path) if model_path else self.cfg.best_model_path
        output_dir = Path(output_dir) if output_dir else self.cfg.model_dir

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # ── Load metadata ──────────────────────────────────────────────
        meta_path = model_path.with_suffix(".meta.json")
        rs_sidecar = model_path.with_suffix(".reward_state.json")

        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        else:
            # Week checkpoint — build stub metadata from filename
            week_num = self._parse_week_from_filename(model_path.stem)
            meta = {
                "week": week_num,
                "val_sharpe": None,
                "balance": None,
                "timestamp": os.path.getmtime(str(model_path)),
            }

        # ── Extract reward state ───────────────────────────────────────
        reward_state = meta.pop("reward_state", {})
        if not reward_state and rs_sidecar.exists():
            with open(rs_sidecar, encoding="utf-8") as f:
                reward_state = json.load(f)

        # ── Build config dict ──────────────────────────────────────────
        config_dict = self.cfg.to_dict()

        # ── Build clean metadata dict ──────────────────────────────────
        week = meta.get("week", 0)
        val_sharpe = meta.get("val_sharpe")
        metadata_dict = {
            "week": week,
            "val_sharpe": val_sharpe,
            "balance": meta.get("balance"),
            "timestamp": meta.get("timestamp"),
            "packaged_at": time.time(),
            "obs_dim": self.cfg.obs_dim,
            "num_features": self.cfg.num_features,
            "frame_stack": self.cfg.frame_stack,
        }

        # ── Generate feature baseline ─────────────────────────────────
        feature_baseline = self.generate_feature_baseline()
        has_baseline = bool(feature_baseline)

        # ── Generate correlation baseline ─────────────────────────────
        corr_baseline = self.generate_correlation_baseline(n_top=20)
        has_corr = bool(corr_baseline)

        # ── Load stress results ───────────────────────────────────────
        stress_results = self._load_stress_results(week)
        has_stress = bool(stress_results)

        # ── EWC (Fisher) state ────────────────────────────────────────
        ewc_state_path = self.cfg.finetune_dir / "ewc_state.pt"
        has_ewc = ewc_state_path.exists()

        # ── Build README ──────────────────────────────────────────────
        readme = self._build_readme(
            metadata_dict, has_baseline=has_baseline, has_stress=has_stress,
            has_corr=has_corr, has_ewc=has_ewc)

        # ── Determine output filename ─────────────────────────────────
        if output_filename is None:
            sharpe_str = "none"
            if val_sharpe is not None:
                sharpe_str = f"{val_sharpe:.3f}".replace("-", "neg")
            output_filename = f"spartus_model_w{week:04d}_sharpe{sharpe_str}.zip"

        out_path = output_dir / output_filename
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Write package ZIP ─────────────────────────────────────────
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(str(model_path), "model.zip")
            zf.writestr("config.json", json.dumps(config_dict, indent=2))
            zf.writestr("metadata.json", json.dumps(metadata_dict, indent=2))
            zf.writestr("reward_state.json", json.dumps(reward_state, indent=2))
            if has_baseline:
                zf.writestr("feature_baseline.json",
                            json.dumps(feature_baseline, indent=2))
            if has_corr:
                zf.writestr("correlation_baseline.json",
                            json.dumps(corr_baseline, indent=2))
            if has_stress:
                zf.writestr("stress_results.json",
                            json.dumps(stress_results, indent=2))
            if has_ewc:
                zf.write(str(ewc_state_path), "ewc_state.pt")
            zf.writestr("README.txt", readme)

        size_mb = out_path.stat().st_size / (1024 * 1024)
        included = ["model", "config", "metadata", "reward_state"]
        if has_baseline:
            included.append("feature_baseline")
        if has_corr:
            included.append("correlation_baseline")
        if has_stress:
            included.append("stress_results")
        if has_ewc:
            included.append("ewc_state")
        log.info(f"Model packaged: {out_path} ({size_mb:.1f} MB) "
                 f"[{', '.join(included)}]")
        return str(out_path)

    # ── Feature Baseline ────────────────────────────────────────────────────

    def generate_feature_baseline(self) -> Dict:
        """Compute per-feature mean/std from validation set feature caches.

        Returns dict: {feature_name: {"mean": float, "std": float}, ...}
        Used by live dashboard for drift detection (Tab 6).
        """
        from src.environment.trade_env import PRECOMPUTED_FEATURES

        val_weeks = self._discover_val_weeks()
        if not val_weeks:
            log.warning("No val weeks discovered — cannot generate feature baseline")
            return {}

        all_frames = []
        for year, wk, cache_path in val_weeks:
            if not cache_path.exists():
                continue
            try:
                df = pd.read_parquet(cache_path)
                present = [c for c in PRECOMPUTED_FEATURES if c in df.columns]
                if len(present) >= 50:  # sanity: expect 54 precomputed
                    all_frames.append(df[present])
            except Exception as e:
                log.warning(f"Skip {cache_path.name}: {e}")

        if not all_frames:
            log.warning("No valid feature caches loaded for baseline")
            return {}

        combined = pd.concat(all_frames, ignore_index=True)
        baseline = {}
        for col in combined.columns:
            vals = combined[col].dropna()
            if len(vals) > 0:
                baseline[col] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "count": int(len(vals)),
                }

        log.info(f"Feature baseline: {len(baseline)} features from "
                 f"{len(all_frames)} val weeks ({len(combined)} bars)")
        return baseline

    # ── Correlation Baseline ────────────────────────────────────────────────

    def generate_correlation_baseline(self, n_top: int = 20) -> Dict:
        """Compute correlation matrix for top-N most variable features.

        Selects the N features with highest variance from the validation set,
        then computes their pairwise correlation matrix. This captures the
        inter-feature correlation structure that per-feature drift detection
        can miss (e.g., gold-USD correlation flipping sign).

        Returns dict: {"features": [...], "correlation_matrix": [[...]], ...}
        Used by live dashboard Tab 6 for correlation drift detection.
        """
        from src.environment.trade_env import PRECOMPUTED_FEATURES

        val_weeks = self._discover_val_weeks()
        if not val_weeks:
            log.warning("No val weeks — cannot generate correlation baseline")
            return {}

        all_frames = []
        for year, wk, cache_path in val_weeks:
            if not cache_path.exists():
                continue
            try:
                df = pd.read_parquet(cache_path)
                present = [c for c in PRECOMPUTED_FEATURES if c in df.columns]
                if len(present) >= 50:
                    all_frames.append(df[present])
            except Exception as e:
                log.warning(f"Skip {cache_path.name}: {e}")

        if not all_frames:
            log.warning("No valid feature caches for correlation baseline")
            return {}

        combined = pd.concat(all_frames, ignore_index=True).dropna()
        if len(combined) < 100:
            log.warning(f"Too few rows ({len(combined)}) for correlation baseline")
            return {}

        # Select top-N features by variance
        variances = combined.var().sort_values(ascending=False)
        top_features = list(variances.head(min(n_top, len(variances))).index)

        # Compute correlation matrix and its Frobenius norm (for normalized drift score)
        corr_matrix = combined[top_features].corr()
        frob_norm = float(np.linalg.norm(corr_matrix.values, 'fro'))

        result = {
            "features": top_features,
            "correlation_matrix": corr_matrix.values.tolist(),
            "frobenius_norm_baseline": frob_norm,
            "n_bars": int(len(combined)),
            "n_features": len(top_features),
        }
        log.info(f"Correlation baseline: {len(top_features)} features, "
                 f"{len(combined)} bars, ‖C‖_F={frob_norm:.3f}")
        return result

    # ── ONNX Export (stub) ──────────────────────────────────────────────────

    def export_onnx(self, model_path: str, output_path: Optional[str] = None) -> str:
        """Export SAC actor to ONNX format for MT5 integration.

        TODO: Implement in deployment phase.
        """
        raise NotImplementedError(
            "ONNX export will be implemented in the deployment phase. "
            "For now, use the SB3 .zip model directly."
        )

    # ── Private helpers ─────────────────────────────────────────────────────

    def _discover_val_weeks(self) -> List[Tuple[int, int, Path]]:
        """Discover validation set weeks and their feature cache paths.

        Uses the same 70/15/15 split as training with 2-week purge gaps.
        Returns list of (year, week_num, cache_path) tuples.
        """
        data_dir = Path(self.cfg.data_dir)
        all_weeks = []

        for year_dir in sorted(data_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            try:
                year = int(year_dir.name)
            except ValueError:
                continue
            for wk in range(1, 54):
                m5 = year_dir / f"week_{wk:02d}_M5.parquet"
                h1 = year_dir / f"week_{wk:02d}_H1.parquet"
                if m5.exists() and h1.exists():
                    cache_path = (
                        self.cfg.feature_dir / str(year)
                        / f"week_{wk:02d}_features.parquet"
                    )
                    all_weeks.append((year, wk, cache_path))

        if not all_weeks:
            return []

        n = len(all_weeks)
        purge = 2
        train_end = int(n * 0.70)
        val_end = train_end + purge + int(n * 0.15)
        val_start = train_end + purge

        return all_weeks[val_start:min(val_end, n)]

    def _find_stress_results(self, week: int) -> Optional[Path]:
        """Find stress matrix JSON for a given checkpoint week."""
        stress_path = Path("storage/logs") / f"stress_matrix_W{week:04d}.json"
        if stress_path.exists():
            return stress_path
        # Also check absolute
        abs_path = Path(self.cfg.data_dir).parent.parent / "logs" / f"stress_matrix_W{week:04d}.json"
        if abs_path.exists():
            return abs_path
        return None

    def _load_stress_results(self, week: int) -> dict:
        """Load stress matrix results for inclusion in the package.

        Returns dict keyed by "{set}_{scenario}" for easy lookup, e.g.:
        {"VAL_base": {...}, "VAL_2x_spread": {...}, "TEST_base": {...}, ...}
        """
        path = self._find_stress_results(week)
        if not path:
            log.warning(f"No stress results found for W{week:04d}")
            return {}

        try:
            with open(path, encoding="utf-8") as f:
                results_list = json.load(f)
            # Convert flat list to keyed dict for easier lookup
            keyed = {}
            for r in results_list:
                key = f"{r.get('set', 'UNK')}_{r.get('scenario', 'base')}"
                keyed[key] = r
            keyed["_source_file"] = path.name
            keyed["_checkpoint"] = week
            return keyed
        except Exception as e:
            log.warning(f"Failed to load stress results: {e}")
            return {}

    def _resolve_week(self, model_path: Path) -> int:
        """Resolve week number from model path — tries meta.json then filename."""
        meta_path = model_path.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    return json.load(f).get("week", 0)
            except Exception:
                pass
        return self._parse_week_from_filename(model_path.stem)

    def _build_readme(self, metadata: dict, has_baseline: bool = False,
                      has_stress: bool = False, has_corr: bool = False,
                      has_ewc: bool = False) -> str:
        ts = metadata.get("packaged_at", time.time())
        dt = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        week = metadata.get("week", "?")
        sharpe = metadata.get("val_sharpe")
        sharpe_str = f"{sharpe:.4f}" if sharpe is not None else "N/A"
        balance = metadata.get("balance")
        balance_str = f"${balance:.2f}" if balance is not None else "N/A"
        obs_dim = metadata.get("obs_dim", "?")
        n_feats = metadata.get("num_features", "?")
        n_frames = metadata.get("frame_stack", "?")

        contents = (
            f"  model.zip               SB3 SAC model weights\n"
            f"  config.json             Full TrainingConfig ({len(self.cfg.to_dict())} fields)\n"
            f"  metadata.json           Training week, Sharpe, balance\n"
            f"  reward_state.json       RewardNormalizer + DiffSharpe state\n"
        )
        if has_baseline:
            contents += f"  feature_baseline.json   Per-feature mean/std for drift detection (54 features)\n"
        if has_corr:
            contents += f"  correlation_baseline.json  Top-20 feature correlation matrix (regime detection)\n"
        if has_stress:
            contents += f"  stress_results.json     Stress matrix results for live comparison\n"
        if has_ewc:
            contents += f"  ewc_state.pt            Fisher information matrix for live fine-tuning (EWC)\n"
        contents += f"  README.txt              This file\n"

        return (
            f"SPARTUS TRADING AI - Model Package\n"
            f"===================================\n\n"
            f"Packaged:    {dt}\n"
            f"Week:        {week}\n"
            f"Val Sharpe:  {sharpe_str}\n"
            f"Balance:     {balance_str}\n"
            f"Obs Dim:     {obs_dim} ({n_feats} features x {n_frames} frames)\n\n"
            f"Contents\n"
            f"--------\n"
            f"{contents}\n"
            f"Usage\n"
            f"-----\n"
            f"Place this ZIP in live_dashboard/model/ and start the dashboard.\n"
            f"The loader validates obs_dim={obs_dim} automatically.\n"
            f"\n"
            f"Live Dashboard Integration\n"
            f"-------------------------\n"
            f"  feature_baseline.json      → Tab 6 drift detection (flags features >2σ from training)\n"
            f"  correlation_baseline.json  → Tab 6 correlation drift (Frobenius norm of diff matrix)\n"
            f"  stress_results.json        → Tab 6 training-vs-live comparison panel\n"
            f"  ewc_state.pt               → Tab 7 Live Fine-Tune (Fisher matrix, skip recompute)\n"
        )

    @staticmethod
    def _parse_week_from_filename(stem: str) -> int:
        """Extract week number from checkpoint filename like 'spartus_week_0027'."""
        match = re.search(r"week_(\d+)", stem)
        return int(match.group(1)) if match else 0
