"""Live Fine-Tuner — orchestrates the entire live fine-tuning process.

Coordinates:
    - LiveDataCollector: pulls M5 bars from MT5
    - CuratedReplayBuffer: tier-managed replay buffer
    - EWCProtection: weight regularization
    - StrategyMemory: regime-strategy tracking
    - ValidationGate: pre-promotion safety check
    - TradeEnv: same environment as training (feature parity)
    - SAC model: gradient updates via SB3

State machine:
    IDLE → INITIALIZING → COLLECTING → TRAINING → VALIDATING → COLLECTING
    Any state → STOPPED (on stop request)
    VALIDATING → PROMOTING (on user promote request)

All operations run in a background thread. Dashboard communicates via
the shared_metrics dict (same pattern as training dashboard).
"""

import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.training.convergence import LiveConvergenceDetector

log = logging.getLogger(__name__)

# Fine-tuner states
FT_IDLE         = "IDLE"
FT_INITIALIZING = "INITIALIZING"
FT_COLLECTING   = "COLLECTING"
FT_TRAINING     = "TRAINING"
FT_VALIDATING   = "VALIDATING"
FT_PROMOTING    = "PROMOTING"
FT_STOPPED      = "STOPPED"
FT_ERROR        = "ERROR"


class LiveFineTuner:
    """Orchestrator for live fine-tuning of the SAC model on MT5 live data.

    Usage:
        tuner = LiveFineTuner(config, shared_metrics)
        tuner.start(model_path="storage/models/spartus_best.zip")
        # ... runs in background ...
        tuner.validate_now()
        tuner.promote(output_name="spartus_finetuned_v1.zip")
        tuner.stop()
    """

    def __init__(self, config, shared_metrics: Optional[Dict] = None):
        self.cfg = config
        self.shared_metrics = shared_metrics if shared_metrics is not None else {}

        # Sub-components (initialized lazily in _initialize)
        self._collector: Optional[object] = None   # LiveDataCollector
        self._ewc: Optional[object] = None         # EWCProtection
        self._buffer: Optional[object] = None      # CuratedReplayBuffer
        self._strategy_memory: Optional[object] = None  # StrategyMemory
        self._val_gate: Optional[object] = None    # ValidationGate
        self._memory: Optional[object] = None      # TradingMemory (single connection)

        # SAC model references
        self._model = None             # Fine-tuned model (learns)
        self._baseline_model = None    # Frozen reference (never modified)
        self._model_path: Optional[str] = None

        # State
        self._state: str = FT_IDLE
        self._thread: Optional[threading.Thread] = None
        self._stop_requested: bool = False
        self._validate_requested: bool = False
        self._promote_requested: bool = False

        # Metrics
        self._episode_count: int = 0
        self._grad_steps_total: int = 0
        self._last_episode_time: float = 0.0
        self._baseline_sharpe: float = 0.0
        self._kl_divergence: float = 0.0
        self._ewc_penalty_last: float = 0.0
        self._consecutive_val_failures: int = 0

        # Checkpoint management
        self._finetune_dir = Path(getattr(config, "finetune_dir", "storage/finetune"))
        self._checkpoint_dir = self._finetune_dir / "checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints: list = []  # List of checkpoint paths (newest first)

        # Resume cursor — tracks the last bar that completed a fine-tune episode.
        # Persisted to disk so restarts pick up exactly where they left off.
        self._cursor_path = self._finetune_dir / "cursor.json"
        self._cursor: Optional[Dict] = None  # Populated in _initialize()

        # Episode result history for dashboard
        self._episode_history: list = []

        # Live convergence detector
        self._convergence = LiveConvergenceDetector(config=config)

        # EWC deferred flag (set when buffer was empty at init time)
        self._ewc_pending: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Public Control API
    # ─────────────────────────────────────────────────────────────────────────

    def start(
        self,
        model_path: str,
        mode: str = "live",  # "live" or "historical"
        trainer=None,
    ) -> None:
        """Start fine-tuning in a background thread.

        Args:
            model_path: Path to trained model ZIP to fine-tune.
            mode: "live" = pull from MT5. "historical" = use training data.
            trainer: Trainer instance (for validation gate access). Optional.
        """
        if self._thread and self._thread.is_alive():
            log.warning("LiveFineTuner: already running")
            return

        self._model_path = model_path
        self._stop_requested = False
        self._trainer_ref = trainer
        self._mode = mode

        self._thread = threading.Thread(
            target=self._run_loop,
            name="LiveFineTuner",
            daemon=True,
        )
        self._thread.start()
        log.info(f"LiveFineTuner: started (mode={mode}, model={Path(model_path).name})")

    def stop(self) -> None:
        """Request graceful stop."""
        self._stop_requested = True
        self._set_state(FT_STOPPED)
        log.info("LiveFineTuner: stop requested")

    def validate_now(self) -> None:
        """Request an immediate validation cycle."""
        self._validate_requested = True

    def promote(self, output_name: Optional[str] = None) -> Optional[str]:
        """Export the current fine-tuned model as a live deployment package.

        Returns:
            Path to exported ZIP, or None if not ready.
        """
        if self._model is None:
            log.warning("LiveFineTuner: no model to promote")
            return None

        try:
            self._set_state(FT_PROMOTING)
            self._promote_requested = True
            output_path = self._export_model(output_name)
            self._set_state(FT_COLLECTING if self._mode == "live" else FT_IDLE)
            return output_path
        except Exception as e:
            log.error(f"LiveFineTuner: promote failed: {e}")
            self._set_state(FT_ERROR)
            return None

    def rollback(self, checkpoint_idx: int = 0) -> bool:
        """Revert to a previous checkpoint.

        Args:
            checkpoint_idx: 0 = most recent, 1 = second most recent, etc.

        Returns:
            True if rollback succeeded.
        """
        if checkpoint_idx >= len(self._checkpoints):
            log.warning(f"LiveFineTuner: no checkpoint at index {checkpoint_idx}")
            return False

        checkpoint_path = self._checkpoints[checkpoint_idx]
        try:
            self._load_model(checkpoint_path)
            log.info(f"LiveFineTuner: rolled back to {Path(checkpoint_path).name}")
            self._update_shared_metrics()
            return True
        except Exception as e:
            log.error(f"LiveFineTuner: rollback failed: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Main Loop
    # ─────────────────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Main fine-tuning loop (runs in background thread)."""
        try:
            self._set_state(FT_INITIALIZING)
            if not self._initialize():
                self._set_state(FT_ERROR)
                return

            self._set_state(FT_COLLECTING)

            while not self._stop_requested:
                # Handle user-triggered actions
                if self._validate_requested:
                    self._validate_requested = False
                    self._run_validation()

                # Collection phase
                if self._mode == "live":
                    self._collect_live_bars()
                    is_ready = self._collector.is_ready()
                else:
                    is_ready = True  # Historical mode: always ready

                # Training phase: trigger every N hours OR on demand
                should_train = (
                    is_ready and self._should_run_episode()
                )

                if should_train:
                    self._set_state(FT_TRAINING)
                    self._run_training_episode()
                    self._set_state(FT_COLLECTING)

                    # Periodic validation
                    if self._episode_count % self.cfg.finetune_checkpoint_interval == 0:
                        self._set_state(FT_VALIDATING)
                        self._run_validation()
                        self._set_state(FT_COLLECTING)

                    # Auto-rollback check
                    if self._consecutive_val_failures >= self.cfg.finetune_auto_rollback_failures:
                        log.warning(
                            f"LiveFineTuner: {self._consecutive_val_failures} consecutive "
                            f"validation failures — auto-rolling back"
                        )
                        self.rollback(0)
                        self._consecutive_val_failures = 0

                self._update_shared_metrics()
                time.sleep(10)  # Poll interval

        except Exception as e:
            log.error(f"LiveFineTuner: unhandled exception: {e}", exc_info=True)
            self.shared_metrics["_ft_error"] = str(e)
            self._set_state(FT_ERROR)

    def _should_run_episode(self) -> bool:
        """Check if it's time to run a training episode."""
        interval_h = self.cfg.finetune_episode_interval_hours
        elapsed_h = (time.time() - self._last_episode_time) / 3600
        return elapsed_h >= interval_h

    # ─────────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────────

    def _initialize(self) -> bool:
        """Initialize all sub-components."""
        try:
            from src.training.ewc_protection import EWCProtection
            from src.training.strategy_memory import StrategyMemory
            from src.training.validation_gate import ValidationGate

            # Load model
            log.info(f"LiveFineTuner: loading model from {self._model_path}")
            self._load_model(self._model_path)

            # Load frozen baseline (separate copy, never modified)
            self._load_baseline(self._model_path)

            # EWC protection
            self._ewc = EWCProtection(ewc_lambda=self.cfg.finetune_ewc_lambda)
            ewc_path = str(self._finetune_dir / "ewc_state")
            if not self._ewc.load(ewc_path):
                if self.cfg.finetune_ewc_enabled:
                    buf_size = self._model.replay_buffer.size() if (
                        hasattr(self._model, "replay_buffer") and
                        self._model.replay_buffer is not None
                    ) else 0
                    if buf_size >= 256:
                        log.info("LiveFineTuner: computing Fisher matrix (first run)...")
                        self._ewc.compute_fisher(
                            self._model,
                            n_samples=self.cfg.finetune_ewc_fisher_samples,
                        )
                        self._ewc.save(ewc_path)
                    else:
                        log.info(
                            f"LiveFineTuner: replay buffer too small ({buf_size}) for Fisher "
                            f"— will compute after first episode"
                        )
                        self._ewc_pending = True

            # Strategy memory
            self._strategy_memory = StrategyMemory(
                db_path=str(self._finetune_dir / "strategy_memory.json"),
                forgetting_threshold=self.cfg.finetune_forgetting_threshold,
            )

            # Validation gate — prefer explicit trainer ref, fall back to shared_metrics
            _trainer = (
                getattr(self, "_trainer_ref", None)
                or self.shared_metrics.get("_trainer_ref")
            )
            self._val_gate = ValidationGate(
                config=self.cfg,
                trainer=_trainer,
            )
            self._val_gate.load_history(str(self._finetune_dir / "val_history.json"))

            # Compute baseline Sharpe if not known
            self._baseline_sharpe = self.shared_metrics.get("_ft_baseline_sharpe", 0.0)

            # Resume cursor — determines where prefill should start so we don't
            # re-train on bars that already completed a fine-tune episode.
            self._cursor = self._load_cursor()

            # Live data collector (only in live mode)
            if self._mode == "live":
                from src.training.live_data_collector import LiveDataCollector
                from datetime import datetime, timezone
                self._collector = LiveDataCollector(self.cfg)
                terminal = getattr(self.cfg, "mt5_terminal_path", None)
                if not self._collector.connect(terminal_path=terminal):
                    log.error("LiveFineTuner: MT5 connection failed")
                    return False

                # Determine prefill start:
                #   - If a cursor exists, resume from last_bar_time (skip already-trained data).
                #   - Otherwise, start from Jan 1 of the current year (full year context).
                now = datetime.now(timezone.utc)
                if self._cursor and self._cursor.get("last_bar_time"):
                    prefill_start = self._cursor["last_bar_time"]
                    log.info(
                        f"LiveFineTuner: cursor found — resuming from {prefill_start.date()} "
                        f"(skipping {self._cursor.get('total_bars_trained', 0):,} already-trained bars)"
                    )
                else:
                    prefill_start = datetime(now.year, 1, 1, tzinfo=timezone.utc)
                    log.info(
                        f"LiveFineTuner: no cursor — fresh start from {prefill_start.date()}"
                    )

                n_prefilled = self._collector.prefill_from_date(prefill_start)
                log.info(
                    f"LiveFineTuner: prefilled {n_prefilled} bars from "
                    f"{prefill_start.date()} — transitioning to live stream"
                )
                self.shared_metrics["_ft_prefill_bars"] = n_prefilled

            # Single TradingMemory instance for the entire fine-tune session.
            # Creating a new one per episode causes competing SQLite connections.
            from src.memory.trading_memory import TradingMemory
            self._memory = TradingMemory(
                db_path=str(self._finetune_dir / "finetune_memory.db")
            )

            # Setup curated replay buffer (overrides model's standard buffer)
            self._setup_curated_buffer()

            # Reset convergence detector for fresh session
            self._convergence.reset()

            log.info("LiveFineTuner: initialization complete")
            return True

        except Exception as e:
            log.error(f"LiveFineTuner: initialization failed: {e}", exc_info=True)
            self.shared_metrics["_ft_error"] = f"Init failed: {e}"
            return False

    def _load_model(self, model_path: str) -> None:
        """Load the SAC model with fine-tuning LR."""
        from stable_baselines3 import SAC

        lr = self.cfg.finetune_lr
        self._model = SAC.load(
            model_path,
            custom_objects={"learning_rate": lr},
            device="auto",
        )
        # Training uses DummyVecEnv(n_envs=4); fine-tuning uses n_envs=1.
        # Patch here so set_env() in _run_training_episode() passes the assertion.
        self._model.n_envs = 1
        log.info(f"LiveFineTuner: model loaded (lr={lr}), n_envs patched to 1")

    def _load_baseline(self, model_path: str) -> None:
        """Load a frozen copy of the baseline model for KL divergence tracking."""
        from stable_baselines3 import SAC
        self._baseline_model = SAC.load(model_path, device="cpu")
        # Freeze baseline — disable gradients
        for param in self._baseline_model.actor.parameters():
            param.requires_grad = False
        log.info("LiveFineTuner: baseline model loaded (frozen)")

    def _setup_curated_buffer(self) -> None:
        """Replace model's replay buffer with curated version."""
        try:
            from src.training.curated_replay_buffer import CuratedReplayBuffer

            obs_space = self._model.observation_space
            act_space = self._model.action_space
            buf_size = self.cfg.finetune_buffer_size
            n_envs = 1  # Fine-tuning uses single env

            curated_buf = CuratedReplayBuffer(
                buffer_size=buf_size,
                observation_space=obs_space,
                action_space=act_space,
                device=self._model.device,
                n_envs=n_envs,
                config=self.cfg,
            )

            # Seed from existing buffer if available
            if hasattr(self._model, "replay_buffer") and self._model.replay_buffer is not None:
                existing_buf = self._model.replay_buffer
                if existing_buf.size() > 0:
                    seeded = curated_buf.seed_from_historical(existing_buf)
                    log.info(f"LiveFineTuner: seeded {seeded:,} transitions from existing buffer")

            self._model.replay_buffer = curated_buf
            self._buffer = curated_buf
            log.info("LiveFineTuner: curated replay buffer installed")
        except Exception as e:
            log.warning(f"LiveFineTuner: curated buffer setup failed: {e} — using standard buffer")
            self._buffer = getattr(self._model, "replay_buffer", None)

    # ─────────────────────────────────────────────────────────────────────────
    # Data Collection
    # ─────────────────────────────────────────────────────────────────────────

    def _collect_live_bars(self) -> None:
        """Poll MT5 for new bars."""
        if self._collector is not None:
            self._collector.update()

    # ─────────────────────────────────────────────────────────────────────────
    # Training Episode
    # ─────────────────────────────────────────────────────────────────────────

    def _run_training_episode(self) -> None:
        """Run one fine-tuning episode on the accumulated data."""
        try:
            # Roll back any uncommitted transaction left by a previous crashed episode.
            # Without this, a stale open transaction keeps the DB locked indefinitely.
            if self._memory is not None:
                try:
                    self._memory.conn.rollback()
                except Exception:
                    pass

            # Get features DataFrame
            if self._mode == "live":
                features_df = self._collector.get_features_df()
            else:
                features_df = self._get_historical_features()

            if features_df is None or len(features_df) < 100:
                log.warning("LiveFineTuner: insufficient features for episode")
                return

            # Create environment — reuse the single TradingMemory connection
            # created in _initialize() to avoid competing SQLite locks.
            from src.environment.trade_env import SpartusTradeEnv

            env = SpartusTradeEnv(
                features_df=features_df,
                config=self.cfg,
                memory=self._memory,
                initial_balance=self.cfg.val_initial_balance,
                week=self._episode_count,
                seed=int(time.time()) % 10000,
                is_validation=False,
            )

            # Set env on model
            from stable_baselines3.common.vec_env import DummyVecEnv
            from stable_baselines3.common.monitor import Monitor

            vec_env = DummyVecEnv([lambda: Monitor(env)])
            self._model.set_env(vec_env)

            # Adjust LR based on KL divergence (Layer 1: Speed guardrail)
            kl = self._kl_divergence
            if kl > self.cfg.finetune_kl_emergency_threshold:
                lr = self.cfg.finetune_lr_emergency
                self._model.lr_schedule = lambda _: lr
                log.warning(f"LiveFineTuner: KL={kl:.3f} > threshold — emergency LR={lr}")
            elif kl > self.cfg.finetune_max_kl_divergence:
                self._set_state(FT_STOPPED)
                log.error(f"LiveFineTuner: KL={kl:.3f} > max — pausing fine-tuning")
                return

            # Run learning
            steps = self.cfg.steps_per_week  # Same as training
            self._model.learn(
                total_timesteps=steps,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            self._grad_steps_total += self.cfg.finetune_gradient_steps

            # Apply EWC penalty (Layer 3: Weight protection)
            if self._ewc is not None and self.cfg.finetune_ewc_enabled:
                # Deferred Fisher computation: compute now that buffer has data
                if self._ewc_pending and self._model.replay_buffer.size() >= 256:
                    ewc_path = str(self._finetune_dir / "ewc_state")
                    log.info("LiveFineTuner: computing deferred Fisher matrix...")
                    self._ewc.compute_fisher(
                        self._model,
                        n_samples=self.cfg.finetune_ewc_fisher_samples,
                    )
                    self._ewc.save(ewc_path)
                    self._ewc_pending = False
                self._ewc_penalty_last = self._ewc.apply_penalty(self._model)

            # Compute KL divergence from baseline (Layer 1: KL anchor)
            self._kl_divergence = self._compute_kl(features_df)

            # Record episode stats
            ep_stats = self._collect_episode_stats(env)
            self._episode_history.append(ep_stats)
            if len(self._episode_history) > 50:
                self._episode_history = self._episode_history[-50:]

            # Update strategy memory (Layer 4)
            self._update_strategy_memory(env, features_df)

            # Update live convergence detector
            ep_sharpe = self._compute_episode_sharpe(env)
            action_std = self._sample_action_std()
            self._convergence.update(
                episode_sharpe=ep_sharpe,
                kl_divergence=self._kl_divergence,
                action_std=action_std,
            )
            ep_stats["episode_sharpe"] = ep_sharpe
            ep_stats["convergence_state"] = self._convergence.state

            # Save resume cursor — record the latest bar time so a restart
            # can skip all data that was already included in this episode.
            if self._mode == "live" and self._collector is not None:
                latest_bar_time = self._collector.get_latest_bar_time()
                if latest_bar_time is not None:
                    self._save_cursor(latest_bar_time)

            # Save checkpoint
            self._episode_count += 1
            self._last_episode_time = time.time()

            if self._episode_count % self.cfg.finetune_checkpoint_interval == 0:
                self._save_checkpoint()

            log.info(
                f"LiveFineTuner: episode {self._episode_count} complete — "
                f"KL={self._kl_divergence:.3f}, EWC={self._ewc_penalty_last:.1f}, "
                f"trades={ep_stats.get('trades', 0)}, P/L={ep_stats.get('pnl', 0):.2f}"
            )

        except Exception as e:
            log.error(f"LiveFineTuner: training episode failed: {e}", exc_info=True)

    def _get_historical_features(self) -> Optional[object]:
        """Get features from training dataset for historical fine-tune mode."""
        # Use trainer's data loading if available
        trainer = getattr(self, "_trainer_ref", None)
        if trainer is not None and hasattr(trainer, "_weeks_data") and trainer._weeks_data:
            import random
            week_data = random.choice(trainer._train_weeks[-50:])  # Recent training data
            try:
                return trainer._load_features(trainer._weeks_data[week_data])
            except Exception:
                pass
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # KL Divergence (Layer 1)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_kl(self, features_df) -> float:
        """Estimate KL divergence between fine-tuned and baseline policy.

        KL(π_new || π_frozen) estimated from a sample of observations.
        Uses closed-form KL for Gaussian distributions:
            KL = 0.5 * [log(σ_f/σ_n) + (σ_n² + (μ_n-μ_f)²)/σ_f² - 1]
        """
        if self._baseline_model is None:
            return 0.0

        try:
            import torch
            from src.environment.trade_env import SpartusTradeEnv
            from src.data.normalizer import FeatureNormalizer

            # Sample a batch of observations from features_df
            n_sample = min(100, len(features_df) - 10)
            if n_sample < 10:
                return 0.0

            # Build normalizer to get obs vectors
            normalizer = FeatureNormalizer(self.cfg)
            obs_list = []
            for i in range(10, 10 + n_sample):
                row = features_df.iloc[i]
                try:
                    obs = normalizer.normalize_row(row.to_dict())
                    if obs is not None and not np.any(np.isnan(obs)):
                        obs_list.append(obs)
                except Exception:
                    continue

            if len(obs_list) < 5:
                return 0.0

            obs_tensor = torch.FloatTensor(np.array(obs_list))

            # Get action distributions from both policies
            with torch.no_grad():
                dist_new = self._model.actor.get_distribution(obs_tensor)
                dist_base = self._baseline_model.actor.get_distribution(obs_tensor)

                # Gaussian KL divergence (mean + std)
                mean_new = dist_new.distribution.loc
                std_new = dist_new.distribution.scale
                mean_base = dist_base.distribution.loc
                std_base = dist_base.distribution.scale

                # Per-dimension KL
                kl = (
                    torch.log(std_base / (std_new + 1e-8))
                    + (std_new.pow(2) + (mean_new - mean_base).pow(2)) / (2 * std_base.pow(2) + 1e-8)
                    - 0.5
                )
                kl_mean = float(kl.mean().item())
                return max(0.0, kl_mean)

        except Exception as e:
            log.debug(f"LiveFineTuner: KL computation failed: {e}")
            return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────────

    def _run_validation(self) -> Dict:
        """Run validation gate and handle result."""
        if self._val_gate is None or self._model is None:
            return {}

        try:
            # Lazily refresh trainer ref — training may have started after fine-tuning init
            if self._val_gate._trainer is None:
                trainer = (
                    getattr(self, "_trainer_ref", None)
                    or self.shared_metrics.get("_trainer_ref")
                )
                if trainer is not None:
                    self._val_gate._trainer = trainer

            # Measure action std from buffer
            action_std = self._measure_action_std()

            result = self._val_gate.evaluate(
                model=self._model,
                baseline_sharpe=self._baseline_sharpe,
                action_std=action_std,
                strategy_memory=self._strategy_memory,
            )

            if result.get("passed"):
                self._consecutive_val_failures = 0
            else:
                self._consecutive_val_failures += 1

            # Save validation history
            self._val_gate.save_history(str(self._finetune_dir / "val_history.json"))
            self._update_shared_metrics()
            return result

        except Exception as e:
            log.warning(f"LiveFineTuner: validation failed: {e}")
            return {}

    def _measure_action_std(self) -> Optional[float]:
        """Measure action std from a buffer sample."""
        try:
            import torch
            buf = self._model.replay_buffer
            if buf is None or buf.size() < 64:
                return None
            sample = buf.sample(64)
            actions = sample.actions.cpu().numpy()
            return float(np.std(actions))
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Strategy Memory Update (Layer 4)
    # ─────────────────────────────────────────────────────────────────────────

    def _update_strategy_memory(self, env, features_df) -> None:
        """Record episode trades in strategy memory."""
        if self._strategy_memory is None:
            return

        try:
            from src.training.strategy_memory import RegimeClassifier

            # Get the current regime from the last row of features
            if len(features_df) > 0:
                last_row = features_df.iloc[-1]
                regime_key = RegimeClassifier.classify_from_features(last_row.to_dict())
            else:
                regime_key = "UNKNOWN"

            # Extract trades from env (if available)
            if hasattr(env, "trade_history"):
                for trade in env.trade_history:
                    self._strategy_memory.record_trade(
                        regime_key=regime_key,
                        pnl=trade.get("pnl", 0.0),
                        conviction=trade.get("conviction", 0.5),
                        hold_bars=trade.get("hold_bars", 6),
                        source="finetune",
                    )
        except Exception as e:
            log.debug(f"LiveFineTuner: strategy memory update failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Resume Cursor
    # ─────────────────────────────────────────────────────────────────────────

    def _load_cursor(self) -> Optional[Dict]:
        """Load the resume cursor from disk.

        Returns a dict with keys:
            last_bar_time       — datetime (UTC) of the last bar in the completed episode
            session_start       — datetime (UTC) when fine-tuning originally started
            total_bars_trained  — cumulative count of bars processed across all sessions
            episodes_completed  — total episodes completed across all sessions

        Returns None if no cursor exists yet (fresh start).
        """
        if not self._cursor_path.exists():
            log.debug("LiveFineTuner: no resume cursor found (fresh start)")
            return None

        try:
            from datetime import datetime, timezone
            with open(self._cursor_path, encoding="utf-8") as f:
                raw = json.load(f)

            # Parse ISO timestamps back to datetime objects
            for key in ("last_bar_time", "session_start"):
                if raw.get(key):
                    try:
                        dt = datetime.fromisoformat(raw[key])
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        raw[key] = dt
                    except Exception:
                        raw[key] = None

            log.info(
                f"LiveFineTuner: cursor loaded — last_bar={raw.get('last_bar_time')}, "
                f"total_bars={raw.get('total_bars_trained', 0):,}, "
                f"episodes={raw.get('episodes_completed', 0)}"
            )
            return raw

        except Exception as e:
            log.warning(f"LiveFineTuner: cursor read failed ({e}) — treating as fresh start")
            return None

    def _save_cursor(self, latest_bar_time) -> None:
        """Persist the resume cursor to disk after a completed episode.

        Args:
            latest_bar_time: datetime (UTC) of the last bar processed this episode.
        """
        from datetime import datetime, timezone

        prev = self._cursor or {}
        session_start = prev.get("session_start") or datetime.now(timezone.utc)

        # Accumulate totals across sessions
        prev_bars = prev.get("total_bars_trained", 0)
        bars_this_session = self.shared_metrics.get("_ft_prefill_bars", 0)
        # Estimate bars processed: use bar_count from collector if available
        if self._collector is not None:
            current_bars = self._collector.get_bar_count()
        else:
            current_bars = bars_this_session

        cursor = {
            "last_bar_time": latest_bar_time.isoformat(),
            "session_start": session_start.isoformat() if hasattr(session_start, "isoformat") else str(session_start),
            "total_bars_trained": prev_bars + current_bars,
            "episodes_completed": prev.get("episodes_completed", 0) + 1,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            self._cursor_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cursor_path, "w", encoding="utf-8") as f:
                json.dump(cursor, f, indent=2)
            # Keep in-memory cursor in sync (convert back to datetime for next read)
            cursor["last_bar_time"] = latest_bar_time
            cursor["session_start"] = session_start
            self._cursor = cursor
            log.debug(
                f"LiveFineTuner: cursor saved — last_bar={latest_bar_time}, "
                f"total_bars={cursor['total_bars_trained']:,}"
            )
        except Exception as e:
            log.warning(f"LiveFineTuner: cursor save failed: {e}")

    def reset_cursor(self) -> None:
        """Delete the resume cursor, forcing a full fresh start on next launch.

        Use this when you intentionally want to re-process all 2026 data
        (e.g., after promoting a new base model).
        """
        try:
            self._cursor_path.unlink(missing_ok=True)
            self._cursor = None
            log.info("LiveFineTuner: cursor reset — next start will begin from Jan 1")
        except Exception as e:
            log.warning(f"LiveFineTuner: cursor reset failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoints
    # ─────────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self) -> None:
        """Save current model as a numbered checkpoint."""
        try:
            path = self._checkpoint_dir / f"ft_ep{self._episode_count:04d}.zip"
            self._model.save(str(path))
            self._checkpoints.insert(0, str(path))

            # Prune old checkpoints
            max_ckpt = self.cfg.finetune_max_checkpoints
            while len(self._checkpoints) > max_ckpt:
                old = self._checkpoints.pop()
                try:
                    Path(old).unlink()
                    log.debug(f"LiveFineTuner: pruned checkpoint {old}")
                except Exception:
                    pass

            log.info(f"LiveFineTuner: checkpoint saved: {path.name}")
        except Exception as e:
            log.warning(f"LiveFineTuner: checkpoint save failed: {e}")

    def _export_model(self, output_name: Optional[str] = None) -> str:
        """Export fine-tuned model as a live deployment package."""
        if output_name is None:
            ts = int(time.time())
            output_name = f"spartus_finetuned_ep{self._episode_count:04d}_{ts}.zip"

        output_path = self._finetune_dir / output_name

        try:
            # Save current model
            tmp = self._finetune_dir / "_tmp_export.zip"
            self._model.save(str(tmp))

            # Use exporter to package with metadata
            from src.training.exporter import ModelExporter
            exporter = ModelExporter(self.cfg)
            final_path = exporter.package_model(
                model_path=str(tmp),
                output_dir=str(self._finetune_dir),
                output_filename=output_name,
            )

            # Clean up tmp
            tmp.unlink(missing_ok=True)

            log.info(f"LiveFineTuner: model exported to {final_path}")
            return final_path

        except Exception as e:
            log.error(f"LiveFineTuner: export failed: {e}")
            raise

    # ─────────────────────────────────────────────────────────────────────────
    # Episode Stats
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_episode_sharpe(self, env) -> float:
        """Compute Sharpe ratio from episode trade P/L returns."""
        try:
            if hasattr(env, "trade_history") and len(env.trade_history) >= 3:
                returns = [float(t.get("pnl", 0.0)) for t in env.trade_history]
                ret_arr = np.array(returns)
                std = ret_arr.std()
                if std > 0:
                    return float(ret_arr.mean() / std * np.sqrt(252))
        except Exception:
            pass
        return 0.0

    def _sample_action_std(self) -> Optional[float]:
        """Sample action std from current replay buffer (proxy for policy diversity)."""
        try:
            if self._model is not None and self._model.replay_buffer is not None:
                buf = self._model.replay_buffer
                if buf.pos > 32:
                    replay_data = buf.sample(32)
                    import torch
                    with torch.no_grad():
                        obs_tensor = replay_data.observations
                        if hasattr(self._model.actor, "get_action_dist_params"):
                            mean_actions, log_std, _ = self._model.actor.get_action_dist_params(obs_tensor)
                            return float(log_std.exp().mean().item())
        except Exception:
            pass
        return None

    def _collect_episode_stats(self, env) -> Dict:
        """Extract episode metrics from the environment."""
        stats = {
            "episode": self._episode_count + 1,
            "timestamp": time.time(),
            "trades": 0,
            "wins": 0,
            "pnl": 0.0,
            "win_rate": 0.0,
            "balance": 0.0,
            "kl_divergence": self._kl_divergence,
            "ewc_penalty": self._ewc_penalty_last,
        }

        try:
            if hasattr(env, "balance"):
                stats["balance"] = float(env.balance)
            if hasattr(env, "episode_trades"):
                stats["trades"] = int(env.episode_trades)
            if hasattr(env, "_total_wins"):
                stats["wins"] = int(env._total_wins)
                stats["pnl"] = float(env.balance - self.cfg.val_initial_balance)
                if stats["trades"] > 0:
                    stats["win_rate"] = stats["wins"] / stats["trades"]
        except Exception:
            pass

        return stats

    # ─────────────────────────────────────────────────────────────────────────
    # State Management & Dashboard Updates
    # ─────────────────────────────────────────────────────────────────────────

    def _set_state(self, state: str) -> None:
        self._state = state
        self.shared_metrics["_ft_state"] = state

    def _update_shared_metrics(self) -> None:
        """Push all metrics to shared dict for dashboard polling."""
        m = self.shared_metrics

        m["_ft_state"] = self._state
        m["_ft_episode_count"] = self._episode_count
        m["_ft_grad_steps"] = self._grad_steps_total
        m["_ft_last_episode"] = self._last_episode_time
        m["_ft_kl_divergence"] = self._kl_divergence
        m["_ft_ewc_penalty"] = self._ewc_penalty_last
        m["_ft_consecutive_failures"] = self._consecutive_val_failures
        m["_ft_checkpoints"] = [Path(c).name for c in self._checkpoints]
        m["_ft_baseline_sharpe"] = self._baseline_sharpe

        # Collector status
        if self._collector is not None:
            status = self._collector.get_status()
            m["_ft_bars"] = status.get("bar_count", 0)
            m["_ft_buffer_pct"] = status.get("buffer_pct", 0)
            m["_ft_collector_ready"] = status.get("is_ready", False)
            m["_ft_mt5_connected"] = status.get("connected", False)

        # Buffer tiers
        if self._buffer is not None and hasattr(self._buffer, "get_tier_status"):
            tier_status = self._buffer.get_tier_status()
            m["_ft_buffer_tiers"] = tier_status

        # EWC stats
        if self._ewc is not None:
            m["_ft_ewc_enabled"] = self._ewc.enabled
            m["_ft_weight_divergence"] = self._ewc.weight_divergence(self._model) if self._model else 0.0

        # Strategy memory summary
        if self._strategy_memory is not None:
            m["_ft_strategy_summary"] = self._strategy_memory.get_summary()

        # Validation gate summary
        if self._val_gate is not None:
            m["_ft_val_summary"] = self._val_gate.get_summary()

        # Live convergence state
        m["_ft_convergence_state"] = self._convergence.state
        m["_ft_convergence_summary"] = self._convergence.get_summary()

        # Last episode summary
        if self._episode_history:
            m["_ft_last_episode_stats"] = self._episode_history[-1]

        # Resume cursor info
        if self._cursor:
            m["_ft_cursor_last_bar"] = str(self._cursor.get("last_bar_time", ""))
            m["_ft_cursor_total_bars"] = self._cursor.get("total_bars_trained", 0)
            m["_ft_cursor_episodes"] = self._cursor.get("episodes_completed", 0)

    def get_state(self) -> str:
        return self._state

    def get_checkpoints(self) -> list:
        """Return list of available checkpoint names."""
        return [Path(c).name for c in self._checkpoints]
