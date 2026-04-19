"""ModelLoader -- Load a model package and prepare for live inference.

A model package is a ZIP created by ``src.training.exporter.ModelExporter``
containing:

    model.zip                SB3 SAC model weights
    config.json              TrainingConfig snapshot
    metadata.json            Training week, val_sharpe, balance, obs_dim, etc.
    reward_state.json        RewardNormalizer + DiffSharpe EMA state
    feature_baseline.json    55-feature mean/std for drift detection  (optional)
    correlation_baseline.json  20x20 corr matrix for regime detection (optional)
    stress_results.json      Stress matrix results for live comparison (optional)
    README.txt               Human-readable summary

Usage:
    from config.live_config import LiveConfig
    from core.model_loader import ModelLoader

    loader = ModelLoader(LiveConfig())
    components = loader.load()           # auto-discover from model/
    model = components["model"]          # SB3 SAC object
    meta  = components["metadata"]       # dict
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.live_config import LiveConfig

log = logging.getLogger(__name__)


class ModelLoader:
    """Load a model package and prepare for live inference.

    Extracts the deployment ZIP into a temporary directory, loads the
    SB3 SAC model, and returns all bundled components as a dictionary.
    """

    def __init__(self, config: LiveConfig) -> None:
        self._config = config
        self._extract_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, package_path: Optional[str] = None) -> Dict[str, Any]:
        """Extract ZIP and load all components.

        Args:
            package_path: Absolute or config-relative path to the model
                package ZIP.  If *None*, auto-discovers via
                :meth:`discover_model`.

        Returns:
            Dict with keys:
                model               -- loaded SB3 SAC object
                config              -- TrainingConfig dict from package
                metadata            -- training week, val_sharpe, etc.
                reward_state        -- RewardNormalizer + DiffSharpe state
                feature_baseline    -- per-feature mean/std (may be empty)
                correlation_baseline -- corr matrix dict (may be empty)
                stress_results      -- stress matrix dict (may be empty)

        Raises:
            FileNotFoundError: If no package ZIP can be found.
            zipfile.BadZipFile: If the file is not a valid ZIP.
            RuntimeError: If the model fails to load.
        """
        # Resolve path
        if package_path is None:
            package_path = self.discover_model()
            if package_path is None:
                raise FileNotFoundError(
                    "No model package found.  Place a .zip file in the "
                    "model/ directory or set model_path in your config."
                )

        resolved = self._config.resolve_path(package_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Model package not found: {resolved}")

        log.info("Loading model package: %s", resolved)

        # Extract
        extract_dir = self._extract_package(str(resolved))

        # Load each component
        model = self._load_model(extract_dir)
        config_dict = self._load_json(extract_dir, "config.json")
        metadata = self._load_json(extract_dir, "metadata.json")
        reward_state = self._load_json(extract_dir, "reward_state.json")
        feature_baseline = self._load_json(extract_dir, "feature_baseline.json")
        correlation_baseline = self._load_json(extract_dir, "correlation_baseline.json")
        stress_results = self._load_json(extract_dir, "stress_results.json")

        components: Dict[str, Any] = {
            "model": model,
            "config": config_dict,
            "metadata": metadata,
            "reward_state": reward_state,
            "feature_baseline": feature_baseline,
            "correlation_baseline": correlation_baseline,
            "stress_results": stress_results,
        }

        # Validate and log any warnings
        warnings = self._validate(components)
        for w in warnings:
            log.warning("Validation: %s", w)

        week = metadata.get("week", "?")
        sharpe = metadata.get("val_sharpe")
        sharpe_str = f"{sharpe:.4f}" if sharpe is not None else "N/A"
        log.info(
            "Model loaded: week=%s  val_sharpe=%s  obs_dim=%s  "
            "baseline=%s  corr=%s  stress=%s",
            week,
            sharpe_str,
            metadata.get("obs_dim", "?"),
            "yes" if feature_baseline else "no",
            "yes" if correlation_baseline else "no",
            "yes" if stress_results else "no",
        )

        return components

    def discover_model(self) -> Optional[str]:
        """Scan model/ directory for .zip files.

        Returns:
            Path string to the first model package found (relative to
            live_dashboard base dir), or *None* if nothing is found.

        Discovery order:
            1. The explicit ``model_path`` from config (if the file exists).
            2. The first ``.zip`` file found in model/ (alphabetical).
        """
        base_dir = self._config.get_base_dir()

        # 1. Try the configured model_path first
        configured = base_dir / self._config.model_path
        if configured.exists() and configured.suffix == ".zip":
            log.info("Discovered model (from config): %s", configured)
            return self._config.model_path

        # 2. Scan model/ directory
        model_dir = base_dir / "model"
        if not model_dir.is_dir():
            log.warning("Model directory does not exist: %s", model_dir)
            return None

        zip_files = sorted(model_dir.glob("*.zip"))
        if not zip_files:
            log.warning("No .zip files found in %s", model_dir)
            return None

        # Return relative path (config-style)
        selected = zip_files[0]
        relative = str(selected.relative_to(base_dir))
        log.info("Discovered model (auto): %s", relative)
        return relative

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_package(self, zip_path: str) -> str:
        """Extract the model package ZIP to a temporary directory.

        Args:
            zip_path: Absolute path to the ZIP file.

        Returns:
            Path string to the temporary extraction directory.

        Raises:
            zipfile.BadZipFile: If the file is corrupt or not a ZIP.
        """
        extract_dir = tempfile.mkdtemp(prefix="spartus_model_")
        log.info("Extracting model package to: %s", extract_dir)

        with zipfile.ZipFile(zip_path, "r") as zf:
            # Log package contents
            names = zf.namelist()
            log.info("Package contents: %s", ", ".join(names))
            zf.extractall(extract_dir)

        self._extract_dir = extract_dir
        return extract_dir

    def _load_model(self, extract_dir: str) -> object:
        """Load the SB3 SAC model from the extracted package.

        Args:
            extract_dir: Path to the temporary extraction directory.

        Returns:
            The loaded SAC model object.

        Raises:
            RuntimeError: If model.zip is missing or fails to load.
        """
        model_zip = Path(extract_dir) / "model.zip"
        if not model_zip.exists():
            raise RuntimeError(
                f"model.zip not found in package (extracted to {extract_dir})"
            )

        log.info("Loading SAC model from: %s", model_zip)
        try:
            from stable_baselines3 import SAC

            # Override pickled lr_schedule / learning_rate. SB3 invokes the
            # schedule once on load (with progress_remaining=1) to validate
            # it; if the trainer's schedule function had any edge case at
            # progress=1 (e.g. min() on an empty sequence), the load fails.
            # Inference doesn't need the schedule — only continued training
            # would, and we never resume training in the live dashboard.
            model = SAC.load(
                str(model_zip),
                device="cpu",
                custom_objects={
                    "lr_schedule": lambda _: 0.0,
                    "learning_rate": 0.0,
                },
            )
            log.info("SAC model loaded successfully (device=cpu)")
            return model
        except Exception as exc:
            raise RuntimeError(f"Failed to load SAC model: {exc}") from exc

    def _load_json(self, extract_dir: str, filename: str) -> Dict[str, Any]:
        """Load a JSON file from the extracted package.

        Args:
            extract_dir: Path to the temporary extraction directory.
            filename:    Name of the JSON file (e.g. "metadata.json").

        Returns:
            Parsed dict, or empty dict if the file is missing.
        """
        json_path = Path(extract_dir) / filename
        if not json_path.exists():
            log.info("Optional file not in package: %s", filename)
            return {}

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            log.info("Loaded %s (%d keys)", filename, len(data) if isinstance(data, dict) else 0)
            return data
        except json.JSONDecodeError as exc:
            log.warning("Failed to parse %s: %s", filename, exc)
            return {}
        except Exception as exc:
            log.warning("Error reading %s: %s", filename, exc)
            return {}

    def _validate(self, components: Dict[str, Any]) -> List[str]:
        """Validate loaded components for internal consistency.

        Checks that the model's actual observation space matches the
        metadata and packaged config.  Does NOT compare against hardcoded
        constants -- the model package is the source of truth.

        Args:
            components: Dict of loaded components.

        Returns:
            List of warning strings (empty if all checks pass).
        """
        warnings: List[str] = []

        metadata = components.get("metadata", {})
        config_dict = components.get("config", {})
        model = components.get("model")

        # Get the model's actual obs_dim as the ground truth
        model_obs_dim: Optional[int] = None
        if model is not None:
            try:
                model_obs_dim = model.observation_space.shape[0]
            except Exception as exc:
                warnings.append(f"Cannot read model observation space: {exc}")

        # 1. metadata.obs_dim should match the model's actual obs space
        meta_obs = metadata.get("obs_dim")
        if meta_obs is not None and model_obs_dim is not None:
            if meta_obs != model_obs_dim:
                warnings.append(
                    f"metadata.obs_dim={meta_obs} != "
                    f"model.observation_space={model_obs_dim}"
                )

        # 2. num_features * frame_stack should equal obs_dim
        meta_feats = metadata.get("num_features")
        meta_fs = metadata.get("frame_stack")
        if meta_feats is not None and meta_fs is not None and meta_obs is not None:
            expected = meta_feats * meta_fs
            if expected != meta_obs:
                warnings.append(
                    f"num_features({meta_feats}) * frame_stack({meta_fs}) "
                    f"= {expected} != obs_dim({meta_obs})"
                )

        # 3. Config obs_dim (from the training config snapshot) should match
        cfg_obs = config_dict.get("obs_dim")
        if cfg_obs is not None and model_obs_dim is not None:
            if cfg_obs != model_obs_dim:
                warnings.append(
                    f"config.obs_dim={cfg_obs} != "
                    f"model.observation_space={model_obs_dim}"
                )

        # 4. Reward state
        reward_state = components.get("reward_state", {})
        if not reward_state:
            warnings.append("reward_state is empty (reward normalisation may differ)")

        # 6. Optional but recommended components
        if not components.get("feature_baseline"):
            warnings.append(
                "feature_baseline missing -- drift detection will be unavailable"
            )
        if not components.get("correlation_baseline"):
            warnings.append(
                "correlation_baseline missing -- correlation drift detection unavailable"
            )
        if not components.get("stress_results"):
            warnings.append(
                "stress_results missing -- training-vs-live comparison unavailable"
            )

        return warnings

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Remove the temporary extraction directory if it exists."""
        if self._extract_dir is not None:
            try:
                shutil.rmtree(self._extract_dir, ignore_errors=True)
                log.info("Cleaned up temp dir: %s", self._extract_dir)
            except Exception as exc:
                log.warning("Failed to clean temp dir: %s", exc)
            self._extract_dir = None
