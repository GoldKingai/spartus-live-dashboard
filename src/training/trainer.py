"""SpartusTrainer — main training orchestrator.

Responsibilities:
1. Load weekly data and pre-compute features
2. Create SAC agent with custom LR schedule
3. Curriculum learning (3 stages)
4. Weekly training loop with checkpointing
5. Validation every N weeks
6. Crash recovery from training_state.json
"""

import json
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, Future
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure as sb3_configure
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn as nn


class ClippedSAC(SAC):
    """SAC with gradient clipping — SB3's SAC has NO gradient clipping by default.

    Without clipping, critic gradient norms of 30,000+ cause catastrophic instability
    in early training, especially with LayerNorm injection.
    """

    def __init__(self, *args, max_grad_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_grad_norm = max_grad_norm
        # Store pre-clip gradient norms for monitoring (callback reads these)
        self._preclip_actor_grad = 0.0
        self._preclip_critic_grad = 0.0

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """Override to inject gradient clipping after backward passes."""
        import torch as th
        import torch.nn.functional as F
        import numpy as np
        from stable_baselines3.common.utils import polyak_update

        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
                # FIX-CRITIC: Clip Q-value targets to prevent Bellman backup cascade.
                # Without clipping, a hard data week produces Q_target → reward/(1-γ) = -166,
                # creating massive MSE gradients that corrupt the critic permanently.
                # Bounded by ±(clip_reward / (1-γ)) × safety_margin.
                _q_bound = (5.0 / (1.0 - float(self.gamma))) * 1.5  # ±250 with γ=0.97
                target_q_values = target_q_values.clamp(-_q_bound, _q_bound)

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            # FIX-CRITIC: Huber loss instead of MSE.
            # MSE gradient = 2 × error (unbounded) → norm 3000+ on hard weeks.
            # Huber gradient caps at δ=1.0 regardless of Q-error magnitude.
            # This is the standard fix used in DQN, TD3, and robust SAC variants.
            critic_loss = 0.5 * sum(
                F.huber_loss(current_q, target_q_values) for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize critic WITH gradient clipping
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self._preclip_critic_grad = th.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self._max_grad_norm).item()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize actor WITH gradient clipping
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self._preclip_actor_grad = th.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self._max_grad_norm).item()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

from src.config import TrainingConfig
from src.data.feature_builder import FeatureBuilder
from src.data.normalizer import ExpandingWindowNormalizer
from src.data.storage_manager import StorageManager
from src.environment.trade_env import PRECOMPUTED_FEATURES, SpartusTradeEnv
from src.memory.trading_memory import TradingMemory
from src.training.callback import SpartusCallback
from src.training.convergence import ConvergenceDetector, EraPerformanceTracker
from src.training.curated_training_buffer import CuratedTrainingBuffer
from src.training.logger import TrainingLogger


class SpartusTrainer:
    """Main training orchestrator."""

    def __init__(
        self,
        config: TrainingConfig = None,
        shared_metrics: Optional[Dict] = None,
        seed: int = 42,
    ):
        # Enable TF32 on Ampere GPUs — 2-4x faster matmuls, negligible precision loss
        torch.set_float32_matmul_precision("high")

        self.cfg = config or TrainingConfig()
        self.seed = seed
        self.shared_metrics = shared_metrics if shared_metrics is not None else {}

        # Components
        self.storage = StorageManager(str(self.cfg.data_dir))
        self.feature_builder = FeatureBuilder(self.cfg)
        self.normalizer = ExpandingWindowNormalizer(self.cfg)
        self.memory = TradingMemory(config=self.cfg)
        self.logger = TrainingLogger(self.cfg)
        self.convergence = ConvergenceDetector(self.cfg)
        self.era_tracker = EraPerformanceTracker(window=10)

        # State
        self.current_week = 0
        self.balance = self.cfg.initial_balance
        self.peak_balance = self.balance
        self.model: Optional[SAC] = None
        self._weeks_data: List[Dict] = []
        self._week_difficulties: Dict[int, float] = {}  # idx -> difficulty score
        self._week_profiles: Dict[int, Dict] = {}  # idx -> {trending_up, trending_down, ranging, volatile}
        self._train_weeks: List[int] = []
        self._val_weeks: List[int] = []
        self._test_weeks: List[int] = []
        self._stage3_order: List[int] = []  # Year-interleaved order for Stage 3 (FIX-B1)
        self._bankruptcies = 0
        self._best_checkpoint_week = 0
        self._last_checkpoint_week = 0

        # Background data prefetching (Phase 3 optimization)
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")
        self._prefetch_future: Optional[Future] = None
        self._prefetch_cache_key: Optional[str] = None

        # Mutable timestep counter for LR schedule closure (avoids capturing
        # `self` or model reference which would drag non-picklable objects
        # like ThreadPoolExecutor locks into cloudpickle during model.save())
        self._lr_timesteps: List[int] = [0]

        # LR plateau reduction: halve LR when overfitting/plateau persists.
        # Mutable list so the LR schedule closure can read it without capturing `self`.
        self._lr_multiplier: List[float] = [1.0]  # Mutable list for pickle-safe closure
        self._plateau_lr_reductions: int = 0
        self._stagnation_weeks: int = 0  # Consecutive weeks in OVERFITTING or PLATEAU

        # Graduated overfitting defense state
        self._soft_correction_active: bool = False
        self._rollback_count: int = 0  # Total rollbacks this training run
        self._last_rollback_week: int = -999  # Week of last rollback (for cooldown)
        self._forced_stage2_until: int = 0  # Force Stage 2 sampling until this week (hard reset)

    # === Public API ==========================================================

    def run(self, max_weeks: Optional[int] = None, resume: bool = False):
        """Run the full training loop.

        Args:
            max_weeks: Override total training weeks (for testing).
            resume: If True, resume from saved training state.

        Fine-tune mode: set shared_metrics["_finetune"] = True before calling.
        Loads model weights from checkpoint but resets week counter, balance,
        convergence, and memory. The model keeps its learned patterns but
        trains through all data again with the current config.
        """
        total_weeks = max_weeks or self.cfg.total_training_weeks
        finetune = self.shared_metrics.get("_finetune", False)

        # Auto-populate correlated M5 history from MT5 (one-time bulk fetch).
        # Runs before week discovery so the weekly cache-miss fallback is rarely needed.
        self._bulk_populate_correlated_from_mt5()

        # Load and classify all weekly data
        self._discover_weeks()
        self._split_weeks()

        if finetune:
            self._load_finetune_model()
        elif resume:
            self._load_training_state()
        else:
            # Fresh start — clear stale data from prior runs
            self.memory.reset_for_fresh_training()
            self._create_model()

        mode_str = "FINE-TUNE" if finetune else ("RESUME" if resume else "FRESH")
        self.logger.log_alert("INFO", f"Training started ({mode_str}): {total_weeks} weeks, "
                              f"{len(self._train_weeks)} train, {len(self._val_weeks)} val")

        # Push initial dashboard metrics
        self.shared_metrics["_start_time"] = time.time()
        self.shared_metrics["total_weeks"] = total_weeks
        self.shared_metrics["bankruptcies"] = self._bankruptcies
        self.shared_metrics["_trainer_ref"] = self  # For live fine-tuner validation gate

        if resume and not finetune:
            self.shared_metrics["_resumed"] = True
            self.shared_metrics["_resumed_week"] = self.current_week

        start_week = self.current_week
        try:
            for week_idx in range(start_week, total_weeks):
                # Check quit request from dashboard keyboard controls
                if self.shared_metrics.get("_quit_requested", False):
                    self.logger.log_alert("INFO", "Quit requested via dashboard")
                    break

                self.current_week = week_idx
                self.logger.set_week(week_idx)
                self.shared_metrics["current_week"] = week_idx
                self.shared_metrics["total_weeks"] = total_weeks
                self.shared_metrics["convergence_state"] = self.convergence.state
                self.shared_metrics["best_val_sharpe"] = self.convergence.best_val_sharpe
                self.shared_metrics["weeks_since_best"] = self.convergence.weeks_since_best

                # Push memory stats for dashboard
                self._update_memory_stats()

                # Select training week (curriculum)
                data_week_idx = self._select_week(week_idx)
                week_data = self._weeks_data[data_week_idx]

                # Train on this week
                week_result = self._train_week(week_data, week_idx)

                # Health check: auto-reset reward normalizer if variance has collapsed
                self._check_reward_normalizer_health(week_idx)

                # Tag buffer with current era and record performance
                data_year = week_data.get("year", 0)
                if hasattr(self.model, "replay_buffer") and hasattr(self.model.replay_buffer, "set_current_year"):
                    self.model.replay_buffer.set_current_year(data_year)
                episode_sharpe = week_result.get("sharpe", 0.0)
                self.era_tracker.record(data_year, episode_sharpe)

                # Log weak eras as health flags (catastrophic forgetting detection)
                weak_eras = self.era_tracker.get_weak_eras()
                for era in weak_eras:
                    self.logger.log_alert(
                        "WARNING",
                        f"W{week_idx} | Era forgetting: {era['year']} "
                        f"Sharpe {era['recent_avg']:.2f} vs best {era['best']:.2f} "
                        f"(severity {era['severity']:.0%})",
                    )

                # Check quit again (callback may have set it mid-week)
                if self.shared_metrics.get("_quit_requested", False):
                    self.logger.log_alert("INFO", "Quit requested during training")
                    self._save_checkpoint(week_idx, week_result)
                    break

                # Log weekly summary
                self.logger.log_weekly_summary(week_result)
                self.logger.clear_week_buffers()

                # Checkpoint
                if week_idx % self.cfg.checkpoint_interval == 0:
                    self._save_checkpoint(week_idx, week_result)

                # Validation
                if week_idx > 0 and week_idx % self.cfg.validation_interval == 0:
                    val_result = self._validate(week_idx)
                    self.convergence.update(
                        week_idx,
                        val_sharpe=val_result.get("sharpe", 0.0),
                        val_return=val_result.get("total_return", 0.0),
                        train_sharpe=week_result.get("sharpe", 0.0),
                        action_std=week_result.get("action_std", 0.5),
                    )
                else:
                    self.convergence.update(
                        week_idx,
                        train_sharpe=week_result.get("sharpe", 0.0),
                        action_std=week_result.get("action_std", 0.5),
                    )

                # === Graduated Overfitting Defense ===
                self._graduated_defense(week_idx)

                # Early stopping: policy collapse -> halt training
                if (self.convergence.state == "COLLAPSED"
                        and self.cfg.collapsed_auto_stop):
                    self.logger.log_alert(
                        "CRITICAL",
                        f"Policy COLLAPSED (action_std < {self.cfg.collapsed_action_std} for "
                        f"{self.cfg.collapsed_duration} weeks). Training halted.",
                    )
                    break

                # Save state for crash recovery
                self._save_training_state()

                # Prefetch next week's data in background while we checkpoint/validate
                next_week_idx = week_idx + 1
                if next_week_idx < total_weeks:
                    try:
                        # push_profile=False: don't overwrite shared_metrics with
                        # next week's profile before current week finishes (FIX-B6)
                        next_data_idx = self._select_week(next_week_idx, push_profile=False)
                        self._prefetch_next_week(self._weeks_data[next_data_idx])
                    except Exception as e:
                        try:
                            self.logger.log_alert("WARNING", f"Prefetch submit failed: {e}")
                        except Exception:
                            pass

                # Check convergence
                if self.convergence.state in ("CONVERGED", "STABLE"):
                    self.logger.log_alert(
                        "INFO",
                        f"Training {self.convergence.state} at week {week_idx}",
                    )

        except KeyboardInterrupt:
            self.logger.log_alert("WARNING", "Training interrupted by user")
        except Exception as e:
            try:
                self.logger.log_alert("CRITICAL", f"Training error: {e}")
            except Exception:
                pass  # Don't let log failure mask the original error
            self.shared_metrics["_error"] = str(e)
            raise
        finally:
            self._prefetch_executor.shutdown(wait=False)
            self._save_training_state()
            self._save_checkpoint(self.current_week, {})
            self.shared_metrics["_training_done"] = True
            # Report best checkpoint
            if self._best_checkpoint_week > 0:
                self.logger.log_alert(
                    "INFO",
                    f"Training complete. Best checkpoint: week {self._best_checkpoint_week} "
                    f"(val_sharpe={self.convergence.best_val_sharpe:.4f}). "
                    f"Best model: {self.cfg.best_model_path}",
                )
            self.memory.close()

    # === Week Training ======================================================

    def _make_vec_env(self, features_df: pd.DataFrame, week_idx: int) -> DummyVecEnv:
        """Create a DummyVecEnv with n_envs parallel environments.

        All environments share the same features_df and TradingMemory instance
        but get different seeds for domain randomization diversity.
        """
        def make_env(env_idx: int):
            def _init():
                return SpartusTradeEnv(
                    features_df=features_df,
                    config=self.cfg,
                    memory=self.memory,
                    initial_balance=self.cfg.initial_balance,
                    week=week_idx,
                    seed=self.seed + week_idx * 100 + env_idx,
                )
            return _init

        return DummyVecEnv([make_env(i) for i in range(self.cfg.n_envs)])

    def _train_week(self, week_data: Dict, week_idx: int) -> Dict:
        """Train on a single week of data."""
        features_df = self._get_prefetched_or_load(week_data)

        # Capture reward normalizer state from previous env (before creating new one)
        prev_reward_state = self._get_reward_state()

        # Create vectorized environment (n_envs parallel envs)
        vec_env = self._make_vec_env(features_df, week_idx)

        # Update model's environment
        self.model.set_env(vec_env)

        # Transfer reward normalizer state to new envs (prevents cold-start each week)
        if prev_reward_state:
            self._set_reward_state(prev_reward_state)

        # Create callback
        callback = SpartusCallback(
            config=self.cfg,
            logger=self.logger,
            shared_metrics=self.shared_metrics,
        )

        # Sync LR schedule timestep counter before training
        self._lr_timesteps[0] = self.model.num_timesteps

        # Train
        t0 = time.time()
        self.model.learn(
            total_timesteps=self.cfg.steps_per_week,
            callback=callback,
            reset_num_timesteps=False,
        )
        train_time = time.time() - t0

        # Sync LR schedule timestep counter after training
        self._lr_timesteps[0] = self.model.num_timesteps

        # Read per-episode result from primary environment (env 0)
        # Balance is NOT carried between episodes — each starts fresh at initial_balance
        base_env = vec_env.envs[0]
        episode_return_pct = (base_env.balance - self.cfg.initial_balance) / self.cfg.initial_balance * 100

        # Collect results — per-episode metrics (no cumulative balance)
        result = {
            "week": week_idx,
            "data_year": week_data.get("year", 0),
            "data_week": week_data.get("week", 0),
            "balance": base_env.balance,             # End-of-episode balance (for logging)
            "peak_balance": base_env.peak_balance,   # Episode peak (for logging)
            "episode_return_pct": episode_return_pct,
            "episode_trades": base_env.episode_trades,
            "train_time_s": train_time,
            "convergence_state": self.convergence.state,
        }

        # Compute train Sharpe from logged metrics
        step_metrics = self.logger.get_step_metrics()
        if step_metrics:
            returns = [m.get("raw_reward", 0) for m in step_metrics]
            if len(returns) > 1 and np.std(returns) > 0:
                result["sharpe"] = np.mean(returns) / np.std(returns)
            else:
                result["sharpe"] = 0.0

            result["action_std"] = np.mean([
                m.get("action_std", 0.5) for m in step_metrics
                if "action_std" in m
            ]) if any("action_std" in m for m in step_metrics) else 0.5

        # Track bankruptcies — episode ended below 50% of initial
        bankruptcy_threshold = self.cfg.initial_balance * 0.50
        if base_env.balance <= bankruptcy_threshold:
            self._bankruptcies += 1
            self.shared_metrics["bankruptcies"] = self._bankruptcies
            self.logger.log_alert(
                "WARNING",
                f"Episode bankruptcy #{self._bankruptcies} at week {week_idx} "
                f"(end_balance={base_env.balance:.2f}, return={episode_return_pct:.1f}%)")

        return result

    # === Validation ==========================================================

    def _validate(self, week_idx: int) -> Dict:
        """Run validation on held-out weeks.

        FIX-7: Sharpe from per-week equity returns (not step rewards).
        FIX-8: Validate on a capped subset of val weeks for speed.
               Was ALL 87 weeks (~1.1hr each run). Now capped at max_val_weeks (30)
               to keep each validation under ~25 minutes.
        """
        if not self._val_weeks:
            return {"sharpe": 0.0}

        weekly_returns = []
        val_trades = 0
        val_wins = 0

        # Cap validation weeks for speed.
        # max_val_weeks=0 means use ALL val weeks (no cap).
        # Otherwise cap to max_val_weeks with a fixed random subset.
        val_subset = self._val_weeks
        if self.cfg.max_val_weeks > 0 and len(val_subset) > self.cfg.max_val_weeks:
            rng = np.random.RandomState(self.seed)  # Fixed seed: same subset every validation
            val_subset = list(rng.choice(
                self._val_weeks,
                size=self.cfg.max_val_weeks,
                replace=False,
            ))

        for vi in val_subset:
            if vi >= len(self._weeks_data):
                continue
            week_data = self._weeks_data[vi]
            features_df = self._get_features(week_data)

            env = SpartusTradeEnv(
                features_df=features_df,
                config=self.cfg,
                memory=self.memory,
                initial_balance=self.cfg.val_initial_balance,  # FIX-VAL: £10K avoids SKIP_CAP on high-ATR weeks
                week=week_idx,
                seed=self.seed,
                is_validation=True,  # Prevents writing trades to shared memory
            )

            obs, _ = env.reset()
            done = False
            truncated = False

            while not done and not truncated:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)

            # FIX-7: Compute weekly equity return (not step rewards)
            week_return = (env.balance - self.cfg.val_initial_balance) / self.cfg.val_initial_balance
            weekly_returns.append(week_return)
            val_trades += env.episode_trades
            trades = self.memory.get_recent_trades(n=env.episode_trades)
            val_wins += sum(1 for t in trades if t.get("pnl", 0) > 0)

        # FIX-7: Sharpe from weekly equity returns — comparable to benchmark
        if len(weekly_returns) > 1 and np.std(weekly_returns) > 1e-10:
            val_sharpe = np.mean(weekly_returns) / np.std(weekly_returns)
        else:
            val_sharpe = 0.0

        total_return = float(np.sum(weekly_returns))

        result = {
            "sharpe": val_sharpe,
            "total_return": total_return,
            "trades": val_trades,
            "win_rate": val_wins / max(val_trades, 1),
        }
        val_score = self.convergence.compute_val_score(val_sharpe, total_return)

        self.logger.log_alert(
            "INFO",
            f"Validation at week {week_idx}: Sharpe={val_sharpe:.3f}, "
            f"return={total_return:.3f}, score={val_score:.3f}, "
            f"trades={val_trades}, WR={result['win_rate']:.1%}",
        )

        # FIX-VAL: Track best checkpoint using composite score (Sharpe * sqrt(1+return))
        # instead of pure Sharpe — prevents stale checkpoint when model profits more
        # but with higher per-week variance from conviction sizing.
        if val_score > self.convergence.best_val_score:
            self._best_checkpoint_week = week_idx
            self.shared_metrics["best_checkpoint_week"] = week_idx
            self._save_best_model(week_idx, val_sharpe)

        # Save checkpoint meta in memory
        model_path = str(self.cfg.model_dir / f"spartus_week_{week_idx:04d}.zip")
        self.memory.save_checkpoint_meta(
            week=week_idx, model_path=model_path,
            val_sharpe=val_sharpe,
            val_return=total_return,
            val_win_rate=result["win_rate"],
            training_weeks=week_idx,
        )

        return result

    # === Data Management =====================================================

    def _discover_weeks(self):
        """Scan data directory and build list of available weeks."""
        self._weeks_data.clear()
        summary = self.storage.get_summary()

        for year_str, tfs in sorted(summary.items()):
            year = int(year_str)
            # Get M5 week numbers
            m5_count = tfs.get("M5", 0)
            for wk in range(1, 54):
                m5_path = Path(self.cfg.data_dir) / str(year) / f"week_{wk:02d}_M5.parquet"
                h1_path = Path(self.cfg.data_dir) / str(year) / f"week_{wk:02d}_H1.parquet"
                if m5_path.exists() and h1_path.exists():
                    self._weeks_data.append({
                        "year": year,
                        "week": wk,
                        "m5_path": m5_path,
                        "h1_path": h1_path,
                        "h4_path": Path(self.cfg.data_dir) / str(year) / f"week_{wk:02d}_H4.parquet",
                        "d1_path": Path(self.cfg.data_dir) / str(year) / f"week_{wk:02d}_D1.parquet",
                    })

        self.logger.log_alert("INFO", f"Discovered {len(self._weeks_data)} weeks of data")

        # Score difficulty for curriculum
        self._score_week_difficulties()

    def _score_week_difficulties(self):
        """Score each week's difficulty based on trend clarity, volatility, whipsaws.

        Score 0.0 = easy (trending, low vol), 1.0 = hard (ranging, volatile, whipsaw).
        Used by curriculum to feed easy weeks first in Stage 1.
        Also stores per-week profile (trending_up, trending_down, ranging, volatile).
        """
        for idx, week_data in enumerate(self._weeks_data):
            try:
                m5 = pd.read_parquet(week_data["m5_path"])
                if len(m5) < 100:
                    self._week_difficulties[idx] = 0.5
                    self._week_profiles[idx] = {
                        "trending_up": 0.0, "trending_down": 0.0,
                        "ranging": 1.0, "volatile": 0.0,
                    }
                    continue

                close = m5["close"].values

                # Trend clarity: abs(start-to-end return) / total path length
                # XAUUSD M5 clarity values are tiny (median ~0.028, max ~0.12)
                # because bar-to-bar noise dominates. Use *15 so median maps
                # to ~40% and genuinely trending weeks reach 80-100%.
                total_return = abs(close[-1] - close[0])
                net_return = close[-1] - close[0]
                path_length = np.sum(np.abs(np.diff(close)))
                trend_clarity = total_return / (path_length + 1e-8)
                trend_strength = min(trend_clarity * 15, 1.0)
                # Higher clarity = easier. Invert for difficulty.
                trend_score = 1.0 - trend_strength

                # Volatility: normalized ATR proxy
                returns = np.diff(close) / close[:-1]
                vol = np.std(returns) * 100  # as percentage
                vol_score = min(vol / 2.0, 1.0)  # 2% std = max difficulty

                # Whipsaw: count direction changes
                signs = np.sign(np.diff(close))
                sign_changes = np.sum(np.abs(np.diff(signs[signs != 0])) > 0)
                whipsaw_rate = sign_changes / max(len(close), 1)
                whipsaw_score = min(whipsaw_rate * 5, 1.0)

                # Combined difficulty (weighted average)
                difficulty = 0.4 * trend_score + 0.3 * vol_score + 0.3 * whipsaw_score
                self._week_difficulties[idx] = np.clip(difficulty, 0.0, 1.0)

                # Per-week profile for dashboard
                self._week_profiles[idx] = {
                    "trending_up": trend_strength if net_return > 0 else 0.0,
                    "trending_down": trend_strength if net_return < 0 else 0.0,
                    "ranging": trend_score,   # high when NOT trending
                    "volatile": vol_score,
                }

            except Exception:
                self._week_difficulties[idx] = 0.5
                self._week_profiles[idx] = {
                    "trending_up": 0.0, "trending_down": 0.0,
                    "ranging": 0.5, "volatile": 0.5,
                }

    def _split_weeks(self):
        """Split weeks into train/val/test with purge gaps.

        Uses config percentages (train_split/val_split/test_split) and
        purge_weeks gap between sets to prevent data leakage.
        """
        n = len(self._weeks_data)
        purge = self.cfg.purge_weeks

        train_end = int(n * self.cfg.train_split)
        val_end = train_end + purge + int(n * self.cfg.val_split)

        self._train_weeks = list(range(0, train_end))
        self._val_weeks = list(range(train_end + purge, min(val_end, n)))
        self._test_weeks = list(range(min(val_end + purge, n), n))

        # Build interleaved Stage 3 order immediately after split (FIX-B1)
        self._stage3_order = self._build_stage3_order()

        self.logger.log_alert(
            "INFO",
            f"Split: {len(self._train_weeks)} train, {len(self._val_weeks)} val, "
            f"{len(self._test_weeks)} test (purge={purge}, "
            f"ratios={self.cfg.train_split:.0%}/{self.cfg.val_split:.0%}/{self.cfg.test_split:.0%})",
        )

    def _build_stage3_order(self) -> List[int]:
        """Build year-interleaved chronological order for Stage 3 curriculum.

        FIX-B8: Sort by (week_number, year) NOT (year, week_number).
        Old sort produced: 2015w01, 2015w02, ..., 2015w52, 2016w01, ...
        which fed 50+ consecutive weeks from one year — catastrophic forgetting.

        New sort produces: 2015w01, 2016w01, 2017w01, ..., 2024w01,
        2015w02, 2016w02, ..., ensuring the model sees ALL years every ~10
        weeks while still progressing chronologically within each year.
        """
        if not self._train_weeks:
            return []
        # Sort by (week_number, year) for true year interleaving
        return sorted(
            self._train_weeks,
            key=lambda w: (self._weeks_data[w]["week"], self._weeks_data[w]["year"])
        )

    def _select_week(self, training_week: int, push_profile: bool = True) -> int:
        """Select a data week based on curriculum stage and difficulty scoring.

        Stage 1 (weeks 0-30):  Easy weeks (low difficulty score, trending markets)
        Stage 2 (weeks 31-80): Mixed difficulty, random sampling
        Stage 3 (weeks 81+):   Year-interleaved chronological order (FIX-B1)
        """
        if not self._train_weeks:
            return 0

        # Layer 4 hard reset: force Stage 2 (mixed) sampling temporarily
        if training_week < self._forced_stage2_until:
            rng = np.random.RandomState(self.seed + training_week)
            selected = rng.choice(self._train_weeks)
            self._push_week_profile(selected)
            return selected

        if training_week < self.cfg.stage1_end_week:
            # Stage 1: sample from easiest weeks (difficulty < 0.5)
            easy_weeks = [
                w for w in self._train_weeks
                if self._week_difficulties.get(w, 0.5) < 0.5
            ]
            if not easy_weeks:
                # Fallback: use lower third by difficulty
                sorted_weeks = sorted(
                    self._train_weeks,
                    key=lambda w: self._week_difficulties.get(w, 0.5),
                )
                easy_weeks = sorted_weeks[:max(1, len(sorted_weeks) // 3)]
            rng = np.random.RandomState(self.seed + training_week)
            selected = rng.choice(easy_weeks)
        elif training_week < self.cfg.stage2_end_week:
            # Stage 2: random from all training weeks
            rng = np.random.RandomState(self.seed + training_week)
            selected = rng.choice(self._train_weeks)
        else:
            # Stage 3: year-interleaved chronological order (FIX-B1/B8)
            # Sorted by (week_number, year): sees all years at week 1, then
            # all years at week 2, etc. Prevents single-year lock-in.
            if not self._stage3_order:
                self._stage3_order = self._build_stage3_order()
            idx = (training_week - self.cfg.stage2_end_week) % len(self._stage3_order)
            selected = self._stage3_order[idx]

        if push_profile:
            self._push_week_profile(selected)
        return selected

    def _push_week_profile(self, week_idx: int):
        """Push selected week's difficulty and profile to shared_metrics."""
        self.shared_metrics["week_difficulty"] = self._week_difficulties.get(week_idx, 0.5)

        profile = self._week_profiles.get(week_idx, {})
        self.shared_metrics["buffer_trending_up"] = profile.get("trending_up", 0.0)
        self.shared_metrics["buffer_trending_down"] = profile.get("trending_down", 0.0)
        self.shared_metrics["buffer_ranging"] = profile.get("ranging", 0.0)
        self.shared_metrics["buffer_volatile"] = profile.get("volatile", 0.0)

        # Determine regime label from profile
        up = profile.get("trending_up", 0)
        dn = profile.get("trending_down", 0)
        rng = profile.get("ranging", 0)
        vol = profile.get("volatile", 0)
        if max(up, dn) > 0.5:
            self.shared_metrics["regime"] = "TRENDING" if up > dn else "TRENDING DOWN"
        elif rng > 0.6:
            self.shared_metrics["regime"] = "RANGING"
        elif vol > 0.6:
            self.shared_metrics["regime"] = "VOLATILE"
        else:
            self.shared_metrics["regime"] = "MIXED"

        # Pattern confidence from memory
        try:
            pat_count = self.shared_metrics.get("total_patterns", 0)
            total_trades = self.shared_metrics.get("total_trades", 0)
            if total_trades > 10 and pat_count > 0:
                self.shared_metrics["pattern_confidence_avg"] = min(pat_count / (total_trades * 2), 1.0)
        except Exception:
            pass

    def _get_features(self, week_data: Dict) -> pd.DataFrame:
        """Load and compute features for a week, with caching."""
        cache_path = (
            self.cfg.feature_dir
            / str(week_data["year"])
            / f"week_{week_data['week']:02d}_features.parquet"
        )

        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            # Validate schema: reject stale caches with wrong column count
            expected = set(PRECOMPUTED_FEATURES)
            present = set(cached.columns) & expected
            if len(present) < len(expected):
                self.logger.log_alert(
                    "WARNING",
                    f"Stale cache {cache_path.name}: {len(present)}/{len(expected)} "
                    f"expected features. Rebuilding.",
                )
                cache_path.unlink()
            elif self._correlated_cache_stale(cached, week_data["year"], week_data["week"]):
                # Correlated features are zeros but M5 data now exists — rebuild
                cache_path.unlink()
            else:
                return cached

        # Load raw data — include prior week for context
        m5_frames = []
        h1_frames = []
        h4_frames = []
        d1_frames = []

        year, wk = week_data["year"], week_data["week"]

        # Load 2 prior weeks for warmup context
        for offset in range(-2, 1):
            target_wk = wk + offset
            target_year = year
            if target_wk < 1:
                target_year -= 1
                target_wk += 52
            elif target_wk > 52:
                target_year += 1
                target_wk -= 52

            for tf, frames_list in [("M5", m5_frames), ("H1", h1_frames),
                                     ("H4", h4_frames), ("D1", d1_frames)]:
                path = Path(self.cfg.data_dir) / str(target_year) / f"week_{target_wk:02d}_{tf}.parquet"
                if path.exists():
                    frames_list.append(pd.read_parquet(path))

        empty_ohlcv = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        m5 = pd.concat(m5_frames, ignore_index=True).sort_values("time").reset_index(drop=True) if m5_frames else empty_ohlcv.copy()
        h1 = pd.concat(h1_frames, ignore_index=True).sort_values("time").reset_index(drop=True) if h1_frames else empty_ohlcv.copy()
        h4 = pd.concat(h4_frames, ignore_index=True).sort_values("time").reset_index(drop=True) if h4_frames else empty_ohlcv.copy()
        d1 = pd.concat(d1_frames, ignore_index=True).sort_values("time").reset_index(drop=True) if d1_frames else empty_ohlcv.copy()

        if m5.empty:
            raise ValueError(f"No M5 data for year={year} week={wk}")

        # Load correlated instrument M5 data (Upgrade 1)
        correlated_m5 = self._load_correlated_m5(year, wk)

        # Build features (now includes Upgrades 1-5)
        features = self.feature_builder.build_features(
            m5, h1, h4, d1, correlated_m5=correlated_m5
        )

        # Preserve key raw values before normalization (for pattern binning / memory)
        for col in ("rsi_14", "atr_ratio", "h1_trend_dir"):
            if col in features.columns:
                features[f"{col}_raw"] = features[col].copy()

        features = self.normalizer.normalize(features)

        # Cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(cache_path, engine="pyarrow", index=False)

        return features

    # Correlated feature columns that are always 0 when data is missing
    _CORR_SENTINEL_COLS = ("us500_returns_20", "usoil_returns_20", "eurusd_returns_20")

    def _correlated_cache_stale(
        self, cached: pd.DataFrame, year: int, wk: int
    ) -> bool:
        """Return True if cached features have zero correlated data but M5 data now exists.

        Allows training to rebuild feature files after correlated M5 data is populated
        (e.g. after the first successful MT5 bulk fetch).
        """
        # Check if any sentinel correlated column is non-zero in the cache
        for col in self._CORR_SENTINEL_COLS:
            if col in cached.columns and cached[col].abs().max() > 1e-6:
                return False  # Cache has real correlated data — not stale

        # All sentinels are zero — check if correlated M5 data exists now
        for symbol in self.cfg.correlated_symbols:
            path = (
                self.cfg.correlated_data_dir / symbol
                / str(year) / f"week_{wk:02d}_M5.parquet"
            )
            if path.exists():
                return True  # Data available but cache was built without it

        return False  # No correlated data available — zeros are correct

    def _load_correlated_m5(self, year: int, wk: int) -> Dict[str, pd.DataFrame]:
        """Load M5 data for correlated instruments (same week range as XAUUSD).

        For each symbol/week:
          1. Load from cached parquet if available.
          2. Otherwise fetch from MT5 and cache to disk for future runs.
          3. If MT5 is unavailable, return empty DataFrame (features fill with 0).

        Returns dict mapping symbol name to M5 DataFrame.
        """
        correlated = {}
        empty_ohlcv = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        for symbol in self.cfg.correlated_symbols:
            frames = []
            for offset in range(-2, 1):
                target_wk = wk + offset
                target_year = year
                if target_wk < 1:
                    target_year -= 1
                    target_wk += 52
                elif target_wk > 52:
                    target_year += 1
                    target_wk -= 52

                path = (
                    self.cfg.correlated_data_dir / symbol
                    / str(target_year) / f"week_{target_wk:02d}_M5.parquet"
                )
                if path.exists():
                    frames.append(pd.read_parquet(path))
                else:
                    # Cache miss — try MT5
                    df = self._fetch_correlated_from_mt5(symbol, target_year, target_wk, path)
                    if df is not None and not df.empty:
                        frames.append(df)

            if frames:
                df = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)
                correlated[symbol] = df
            else:
                correlated[symbol] = empty_ohlcv.copy()

        return correlated

    def _bulk_populate_correlated_from_mt5(self) -> None:
        """Pull maximum available M5 history for all correlated symbols from MT5.

        Runs once at training startup. Pulls as many bars as possible per symbol
        (brokers typically keep 2-5 years of M5), slices into weekly parquet files,
        and caches them. After this runs, `_load_correlated_m5` finds cache hits for
        all covered weeks and the per-week MT5 fallback is rarely needed.

        Skipped silently if MT5 is unavailable (features stay at 0 for missing weeks).
        Already-cached weeks are never overwritten.
        """
        try:
            from src.data.mt5_connection import MT5Connection
            import MetaTrader5 as mt5_mod

            terminal = getattr(self.cfg, "mt5_terminal_path", None)
            conn = MT5Connection.acquire(
                terminal_path=terminal,
                symbol_map=self.cfg.symbol_map,
            )
            if conn is None:
                return  # MT5 not available — skip silently

            try:
                new_files = 0
                for symbol in self.cfg.correlated_symbols:
                    new_files += self._bulk_fetch_symbol(conn, symbol, mt5_mod)
            finally:
                conn.release()

            if new_files > 0:
                self.logger.log_alert(
                    "INFO",
                    f"MT5 bulk fetch: cached {new_files} correlated weekly files"
                )
        except Exception:
            pass  # Never block training startup due to MT5 issues

    def _bulk_fetch_symbol(self, conn, symbol: str, mt5_mod) -> int:
        """Pull all available M5 history for one symbol and slice into weekly parquets.

        Returns the number of new weekly files written.
        """
        try:
            # Pull max available bars (position-based, most brokers allow 50K-200K M5 bars)
            rates = conn.copy_rates_from_pos(symbol, mt5_mod.TIMEFRAME_M5, 0, 200_000)
            if rates is None or len(rates) == 0:
                return 0

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.rename(columns={"tick_volume": "volume"}, inplace=True, errors="ignore")
            df = df[["time", "open", "high", "low", "close", "volume"]].copy()
            df = df.sort_values("time").reset_index(drop=True)

            if df.empty:
                return 0

            # Assign ISO year/week to each bar and split into weekly files
            df["_iso_year"] = df["time"].dt.isocalendar().year.astype(int)
            df["_iso_week"] = df["time"].dt.isocalendar().week.astype(int)

            new_files = 0
            for (iso_year, iso_week), week_df in df.groupby(["_iso_year", "_iso_week"]):
                path = (
                    self.cfg.correlated_data_dir / symbol
                    / str(iso_year) / f"week_{iso_week:02d}_M5.parquet"
                )
                if path.exists():
                    continue  # Never overwrite cached data

                week_df = week_df.drop(columns=["_iso_year", "_iso_week"]).reset_index(drop=True)
                if week_df.empty:
                    continue

                path.parent.mkdir(parents=True, exist_ok=True)
                week_df.to_parquet(path, index=False)
                new_files += 1

            return new_files

        except Exception:
            return 0

    def _fetch_correlated_from_mt5(
        self,
        symbol: str,
        year: int,
        wk: int,
        cache_path: Path,
    ) -> Optional[pd.DataFrame]:
        """Pull one week of M5 data for a correlated symbol from MT5 and cache it.

        Uses the shared MT5ConnectionManager so training never disrupts the
        fine-tuner's live data stream (and vice-versa). The connection is
        acquired and released around this single fetch — if the fine-tuner
        already holds the connection it stays open after we release.

        Returns the DataFrame, or None if MT5 is unavailable.
        """
        try:
            from datetime import datetime, timezone
            from src.data.mt5_connection import MT5Connection

            terminal = getattr(self.cfg, "mt5_terminal_path", None)
            conn = MT5Connection.acquire(
                terminal_path=terminal,
                symbol_map=self.cfg.symbol_map,
            )
            if conn is None:
                return None

            try:
                # Date range: full ISO week (Monday 00:00 -> Sunday 23:59 UTC)
                monday = datetime.fromisocalendar(year, wk, 1).replace(tzinfo=timezone.utc)
                sunday = datetime.fromisocalendar(year, wk, 7).replace(
                    hour=23, minute=59, second=59, tzinfo=timezone.utc
                )

                import MetaTrader5 as mt5
                tf = mt5.TIMEFRAME_M5

                rates = conn.copy_rates_range(symbol, tf, monday, sunday)
                if rates is None or len(rates) == 0:
                    rates = conn.copy_rates_from(symbol, tf, monday, 2000)

                if rates is None or len(rates) == 0:
                    return None

            finally:
                conn.release()

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
            df = df[["time", "open", "high", "low", "close", "volume"]].copy()
            df = df.sort_values("time").reset_index(drop=True)
            df = df[(df["time"] >= monday) & (df["time"] <= sunday)].reset_index(drop=True)

            if df.empty:
                return None

            # Cache to disk — next training run won't need MT5 for this week
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
            self.logger.log_alert(
                "INFO",
                f"Cached {symbol} {year}W{wk:02d} from MT5 ({len(df)} bars)"
            )

            return df

        except Exception:
            # MT5 unavailable or symbol not found — silently skip (features fill with 0)
            return None

    def _prefetch_next_week(self, next_week_data: Dict):
        """Submit background thread to pre-load next week's features."""
        cache_key = f"{next_week_data['year']}_{next_week_data['week']}"
        self._prefetch_cache_key = cache_key
        self._prefetch_future = self._prefetch_executor.submit(
            self._get_features, next_week_data
        )

    def _get_prefetched_or_load(self, week_data: Dict) -> pd.DataFrame:
        """Get features from prefetch if available, else load synchronously."""
        cache_key = f"{week_data['year']}_{week_data['week']}"

        # Check if prefetch is ready for this week
        if (self._prefetch_future is not None
                and self._prefetch_cache_key == cache_key):
            try:
                df = self._prefetch_future.result(timeout=60)
                self._prefetch_future = None
                return df
            except TimeoutError:
                self._prefetch_future = None
                try:
                    self.logger.log_alert("WARNING", f"Prefetch timeout for {cache_key}, sync loading")
                except Exception:
                    pass
            except Exception as e:
                self._prefetch_future = None
                try:
                    self.logger.log_alert("WARNING", f"Prefetch failed for {cache_key}: {e}")
                except Exception:
                    pass

        # Synchronous load
        return self._get_features(week_data)

    def _update_memory_stats(self):
        """Push memory and prediction stats to shared_metrics for dashboard."""
        try:
            # Trend prediction stats
            pred_stats = self.memory.get_prediction_stats()
            self.shared_metrics["trend_accuracy"] = pred_stats.get("accuracy", 0)
            self.shared_metrics["pending_predictions"] = pred_stats.get("pending", 0)
            self.shared_metrics["verified_predictions"] = pred_stats.get("verified", 0)

            # UP/DOWN accuracy (query predictions by direction sign)
            try:
                up_rows = self.memory.conn.execute("""
                    SELECT correct FROM predictions
                    WHERE verified_at_step IS NOT NULL AND predicted_direction > 0
                    ORDER BY id DESC LIMIT 50
                """).fetchall()
                down_rows = self.memory.conn.execute("""
                    SELECT correct FROM predictions
                    WHERE verified_at_step IS NOT NULL AND predicted_direction < 0
                    ORDER BY id DESC LIMIT 50
                """).fetchall()
                if len(up_rows) >= 5:
                    self.shared_metrics["trend_acc_up"] = sum(r[0] for r in up_rows) / len(up_rows)
                if len(down_rows) >= 5:
                    self.shared_metrics["trend_acc_down"] = sum(r[0] for r in down_rows) / len(down_rows)
            except Exception:
                pass

            # Pattern count
            try:
                pat_count = self.memory.conn.execute(
                    "SELECT COUNT(*) FROM patterns"
                ).fetchone()[0]
                self.shared_metrics["total_patterns"] = pat_count
            except Exception:
                pass

            # TP/SL stats
            tp_stats = self.memory.get_tp_stats()
            self.shared_metrics["tp_hit_rate"] = tp_stats.get("tp_hit_rate", 0)
            self.shared_metrics["tp_reachable_rate"] = tp_stats.get("tp_reachable_rate", 0)
            self.shared_metrics["sl_hit_rate"] = tp_stats.get("sl_hit_rate", 0)
        except Exception:
            pass  # Don't crash training if memory query fails

    # === Model Management ====================================================

    def _get_tb_run_dir(self) -> str:
        """Get a run-specific TensorBoard directory.

        Each training run gets its own subfolder (e.g. run_20260223_143000)
        so runs don't overlap and TensorBoard can compare them side by side.
        """
        run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir = self.cfg.tensorboard_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return str(run_dir)

    def _create_model(self):
        """Create SAC model with custom architecture and LR schedule."""
        # Need a dummy env for model creation — must be VecEnv to match n_envs
        dummy_week = self._weeks_data[0] if self._weeks_data else None
        if dummy_week is None:
            raise ValueError("No training data available")

        features_df = self._get_features(dummy_week)
        dummy_vec_env = self._make_vec_env(features_df, week_idx=0)

        # LR schedule
        lr_schedule = self._build_lr_schedule()

        # Configure SB3 logger — each run gets its own TB subdirectory
        self._tb_run_dir = self._get_tb_run_dir()
        sb3_logger = sb3_configure(
            self._tb_run_dir,
            ["tensorboard"],
        )

        self.model = ClippedSAC(
            "MlpPolicy",
            dummy_vec_env,
            learning_rate=lr_schedule,
            buffer_size=self.cfg.buffer_size,
            batch_size=self.cfg.batch_size,
            gamma=self.cfg.gamma,
            tau=self.cfg.tau,
            ent_coef=self.cfg.ent_coef,
            target_entropy=self.cfg.target_entropy,
            learning_starts=self.cfg.learning_starts,
            train_freq=self.cfg.train_freq,
            gradient_steps=self.cfg.gradient_steps,
            max_grad_norm=self.cfg.max_grad_norm,
            policy_kwargs={
                "net_arch": {
                    "pi": self.cfg.net_arch_pi,
                    "qf": self.cfg.net_arch_qf,
                },
                "optimizer_kwargs": {
                    "weight_decay": self.cfg.weight_decay,
                },
            },
            verbose=0,
            seed=self.seed,
            device="auto",
        )
        self.model.set_logger(sb3_logger)

        # Layer 1: Inject Dropout + LayerNorm into networks
        self._inject_regularization()

        # Curated training buffer: replace default FIFO with era-aware reservoir buffer
        self._inject_curated_buffer()

        # Update model reference for LR schedule closure
        self._lr_timesteps[0] = self.model.num_timesteps

    def _build_lr_schedule(self) -> Callable:
        """Build warmup -> hold -> cosine decay LR schedule.

        IMPORTANT: SB3's progress_remaining resets each model.learn() call
        because _total_timesteps accumulates (num_timesteps + steps_per_week).
        This causes the schedule to compress into each week instead of spanning
        all 200 weeks. Fix: use model.num_timesteps for true global progress.

        NOTE: The closure must NOT capture `self` (the trainer), because SB3
        pickles learning_rate during model.save(). Capturing the trainer would
        drag in ThreadPoolExecutor which contains _thread.lock objects.
        Instead we capture a mutable list that holds a model reference,
        updated by the trainer after model creation.
        """
        cfg = self.cfg
        # n_envs multiplier: SB3 increments num_timesteps by n_envs per step
        true_total = cfg.steps_per_week * cfg.total_training_weeks * cfg.n_envs
        # Mutable timestep counter — only captures a list of [int], fully picklable.
        # Updated by the trainer before each learn() call.
        timesteps = self._lr_timesteps

        # Extract all config values as plain floats (no reference to cfg/trainer)
        lr_warmup_end = float(cfg.lr_warmup_end)
        lr_warmup_start = float(cfg.lr_warmup_start)
        learning_rate = float(cfg.learning_rate)
        lr_decay_start = float(cfg.lr_decay_start)
        lr_min = float(cfg.lr_min)

        # Mutable list — updated in-place by plateau detection, read by closure.
        # Pickle-safe (just a list of float).
        lr_multiplier = self._lr_multiplier

        def lr_fn(progress_remaining: float) -> float:
            """Returns the actual learning rate (SB3 uses it directly).

            Uses accumulated timesteps for true global progress instead of
            SB3's progress_remaining which resets each learn() call.
            Applies lr_multiplier from plateau detection (halved on persistent overfitting).
            """
            progress = min(timesteps[0] / true_total, 1.0)

            if progress < lr_warmup_end:
                # Warmup: linear from lr_warmup_start to learning_rate
                t = progress / lr_warmup_end
                base_lr = lr_warmup_start + t * (learning_rate - lr_warmup_start)
            elif progress < lr_decay_start:
                # Hold at full LR
                base_lr = learning_rate
            else:
                # Cosine decay to lr_min
                t = (progress - lr_decay_start) / (1.0 - lr_decay_start)
                base_lr = lr_min + 0.5 * (learning_rate - lr_min) * (1.0 + np.cos(np.pi * t))

            return base_lr * lr_multiplier[0]

        return lr_fn

    # === Crash Recovery ======================================================

    def _get_reward_state(self) -> Optional[Dict]:
        """Extract reward calculator state from the primary environment."""
        try:
            if self.model and self.model.get_env():
                env = self.model.get_env()
                base_env = env.envs[0] if hasattr(env, "envs") else env
                if hasattr(base_env, "reward_calc"):
                    return base_env.reward_calc.get_state()
        except Exception:
            pass
        return None

    def _set_reward_state(self, reward_state: Dict):
        """Restore reward calculator state to all environments."""
        try:
            if self.model and self.model.get_env():
                env = self.model.get_env()
                if hasattr(env, "envs"):
                    for sub_env in env.envs:
                        if hasattr(sub_env, "reward_calc"):
                            sub_env.reward_calc.set_state(reward_state)
                elif hasattr(env, "reward_calc"):
                    env.reward_calc.set_state(reward_state)
        except Exception as e:
            self.logger.log_alert("WARNING", f"Failed to restore reward state: {e}")

    def _check_reward_normalizer_health(self, week_idx: int):
        """Auto-reset reward normalizer variance if it has collapsed.

        The EMA normalizer (tau=0.001) compresses variance slowly but inexorably
        as training rewards become smaller (model learns, mean loss shrinks).
        Below var=0.10 the reward signal is so compressed the model receives
        near-zero gradient from reward shaping — equivalent to training blind.

        Threshold 0.10 gives a 10x safety margin above the 0.062 crisis point
        observed at W279, while avoiding false resets during normal operation.
        """
        reward_state = self._get_reward_state()
        if not reward_state:
            return

        norm = reward_state.get("normalizer", {})
        var = norm.get("var", 1.0)

        if var < 0.10:
            reset_state = {
                "normalizer": {"mean": 0.0, "var": 1.0, "_count": 0},
                "diff_sharpe": reward_state.get(
                    "diff_sharpe", {"A": 0.0, "B": 0.0, "_initialized": False}
                ),
            }
            self._set_reward_state(reset_state)
            self.logger.log_alert(
                "CRITICAL",
                f"W{week_idx} | Reward normalizer variance COLLAPSED (var={var:.4f} < 0.10). "
                f"Auto-reset to mean=0.0, var=1.0. Learning signal restored.",
            )

    def _save_training_state(self):
        """Save training state for crash recovery."""
        state = {
            "current_week": self.current_week,
            "balance": self.balance,
            "peak_balance": self.peak_balance,
            "convergence": self.convergence.get_summary(),
            "era_tracker": self.era_tracker.to_dict(),
            "seed": self.seed,
            "timestamp": time.time(),
            "best_checkpoint_week": self._best_checkpoint_week,
            "bankruptcies": self._bankruptcies,
            # Graduated defense state
            "rollback_count": self._rollback_count,
            "last_rollback_week": self._last_rollback_week,
            "forced_stage2_until": self._forced_stage2_until,
            "stagnation_weeks": self._stagnation_weeks,
            "plateau_lr_reductions": self._plateau_lr_reductions,
            "lr_multiplier": self._lr_multiplier[0],
        }
        # Save reward normalizer / differential Sharpe state
        reward_state = self._get_reward_state()
        if reward_state:
            state["reward_state"] = reward_state
        state_path = self.cfg.training_state_path
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

        # Save model (strip regularization for save/load compatibility)
        if self.model:
            model_path = self.cfg.model_dir / "spartus_latest.zip"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_model_safe(model_path)

    def _load_training_state(self):
        """Resume from saved training state."""
        # Resolve override path first — needed to find checkpoint sidecar
        override_path = self.shared_metrics.get("_resume_model_path")

        # Prefer the checkpoint's own training-state sidecar when resuming from
        # a specific checkpoint. This ensures convergence history, balance, and
        # all defense counters exactly match the checkpoint's moment in training.
        state_path = self.cfg.training_state_path
        if override_path:
            sidecar = Path(override_path).with_suffix(".training_state.json")
            if sidecar.exists():
                state_path = sidecar
                self.logger.log_alert(
                    "INFO", f"Resume: using training state sidecar {sidecar.name}"
                )

        if not state_path.exists():
            self.logger.log_alert("WARNING", "No training state found, starting fresh")
            self._create_model()
            return

        with open(state_path) as f:
            state = json.load(f)

        # No sidecar for this checkpoint — trim convergence history to prevent
        # post-checkpoint collapse data (negative val_sharpes, inflated drawdown)
        # from causing immediate false VAL_DECLINING / PLATEAU / DD-stop triggers.
        if override_path and state_path == self.cfg.training_state_path:
            state = self._trim_state_to_checkpoint(state, Path(override_path))

        self.current_week = state.get("current_week", 0)
        self.balance = state.get("balance", self.cfg.initial_balance)
        self.peak_balance = state.get("peak_balance", self.balance)
        self._best_checkpoint_week = state.get("best_checkpoint_week", 0)
        self._bankruptcies = state.get("bankruptcies", 0)

        # Restore graduated defense state
        self._rollback_count = state.get("rollback_count", 0)
        self._last_rollback_week = state.get("last_rollback_week", -999)
        self._forced_stage2_until = state.get("forced_stage2_until", 0)
        self._stagnation_weeks = state.get("stagnation_weeks", 0)
        self._plateau_lr_reductions = state.get("plateau_lr_reductions", 0)
        self._lr_multiplier[0] = state.get("lr_multiplier", 1.0)

        # Restore convergence state (including history lists for proper detection)
        conv_state = state.get("convergence", {})
        if conv_state:
            self.convergence.restore_from_summary(conv_state)

        # Restore era performance tracker (cross-era forgetting history)
        era_state = state.get("era_tracker", {})
        if era_state:
            self.era_tracker = EraPerformanceTracker.from_dict(era_state)

        # Load model — override_path resolved at top of method
        if override_path:
            model_path = Path(override_path)
        else:
            model_path = self.cfg.model_dir / "spartus_latest.zip"
        if model_path.exists():
            dummy_week = self._weeks_data[0] if self._weeks_data else None
            if dummy_week:
                features_df = self._get_features(dummy_week)
                dummy_vec_env = self._make_vec_env(features_df, week_idx=0)
                lr_schedule = self._build_lr_schedule()
                self.model = ClippedSAC.load(
                    str(model_path), env=dummy_vec_env,
                    custom_objects={"learning_rate": lr_schedule},
                )
                self.model._max_grad_norm = self.cfg.max_grad_norm
                self._lr_timesteps[0] = self.model.num_timesteps

                # Override target_entropy from config (saved checkpoint may have old value)
                if hasattr(self.model, 'target_entropy'):
                    old_te = self.model.target_entropy
                    self.model.target_entropy = self.cfg.target_entropy
                    if old_te != self.cfg.target_entropy:
                        self.logger.log_alert(
                            "INFO",
                            f"Updated target_entropy: {old_te:.1f} -> {self.cfg.target_entropy:.1f}",
                        )

                # Enforce entropy alpha floor on loaded checkpoint
                if hasattr(self.model, 'log_ent_coef') and self.cfg.min_entropy_alpha > 0:
                    import math, torch
                    min_log = math.log(self.cfg.min_entropy_alpha)
                    old_alpha = torch.exp(self.model.log_ent_coef).item()
                    if self.model.log_ent_coef.item() < min_log:
                        with torch.no_grad():
                            self.model.log_ent_coef.data.clamp_(min=min_log)
                        new_alpha = torch.exp(self.model.log_ent_coef).item()
                        self.logger.log_alert(
                            "INFO",
                            f"Entropy alpha floored: {old_alpha:.6f} -> {new_alpha:.4f}",
                        )

                self._tb_run_dir = self._get_tb_run_dir()
                sb3_logger = sb3_configure(
                    self._tb_run_dir,
                    ["tensorboard"],
                )
                self.model.set_logger(sb3_logger)

                # Re-inject regularization into loaded model
                self._inject_regularization()
                self._inject_curated_buffer()

                # Restore reward normalizer / differential Sharpe state
                reward_state = state.get("reward_state")
                if reward_state:
                    self._set_reward_state(reward_state)
                    self.logger.log_alert("INFO", "Restored reward normalizer state")

                self.logger.log_alert(
                    "INFO",
                    f"Resumed from week {self.current_week}, balance={self.balance:.2f}",
                )
        else:
            self.logger.log_alert("WARNING", "No model file found, creating new")
            self._create_model()

    def _trim_state_to_checkpoint(self, state: dict, checkpoint_path: Path) -> dict:
        """Trim training state to match a specific checkpoint week (FIX-RESUME).

        When resuming from an older checkpoint without a training_state sidecar,
        training_state.json may contain convergence history from weeks AFTER the
        checkpoint — including collapsed val_sharpes and inflated drawdown from the
        post-W110 collapse period.  Without trimming:
            - VAL_DECLINING fires immediately (recent avg << best)
            - PLATEAU fires immediately (weeks_since_best already = 5)
            - 81% drawdown blocks all trades via risk manager

        Trims val/train history to the val points that existed at checkpoint_week,
        recomputes weeks_since_best, resets consecutive counters, and restores
        balance from the checkpoint's .meta.json if available.
        """
        import copy
        try:
            checkpoint_week = int(checkpoint_path.stem.split("_")[-1])
        except (ValueError, IndexError):
            # Filename not parseable (e.g. "spartus_best") — try .meta.json (FIX-RESUME Bug#2)
            meta_path = checkpoint_path.with_suffix(".meta.json")
            if meta_path.exists():
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        checkpoint_week = int(json.load(f).get("week", 0))
                except Exception:
                    return state
            else:
                return state  # Can't determine week — return unchanged

        state_week = state.get("current_week") or 0
        if checkpoint_week >= (state_week or 0):
            return state  # State is at or before checkpoint — no trimming needed

        state = copy.deepcopy(state)
        state["current_week"] = checkpoint_week

        # Trim convergence history lists
        expected_val_points = checkpoint_week // self.cfg.validation_interval
        conv = state.get("convergence", {})

        for key in ("val_sharpes", "train_sharpes_at_val"):
            lst = conv.get(key, [])
            if len(lst) > expected_val_points:
                conv[key] = lst[:expected_val_points]

        # train_sharpes and action_stds accumulate ~1 entry per training week
        for key in ("train_sharpes", "action_stds"):
            lst = conv.get(key, [])
            if len(lst) > checkpoint_week:
                conv[key] = lst[:checkpoint_week]

        # Recompute weeks_since_best from trimmed val_sharpes
        val_sharpes = conv.get("val_sharpes", [])
        if val_sharpes:
            best_val = max(val_sharpes)
            best_idx = len(val_sharpes) - 1 - val_sharpes[::-1].index(best_val)
            conv["weeks_since_best"] = len(val_sharpes) - 1 - best_idx
            conv["best_val_sharpe"] = best_val
        conv["overfitting_weeks"] = 0   # Give checkpoint a clean start
        conv["collapsed_weeks"] = 0
        state["convergence"] = conv

        # Reset stagnation/defense counters (were counting the collapse period)
        state["stagnation_weeks"] = 0
        state["plateau_lr_reductions"] = 0

        # Restore balance from checkpoint .meta.json to prevent 80%+ drawdown block
        meta_path = checkpoint_path.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("balance"):
                    state["balance"] = meta["balance"]
                    state["peak_balance"] = meta["balance"]
            except Exception:
                pass

        self.logger.log_alert(
            "INFO",
            f"Resume trim: W{checkpoint_week} — "
            f"{len(val_sharpes)} val pts, best_sharpe={conv.get('best_val_sharpe', 0):.3f}, "
            f"weeks_since_best={conv.get('weeks_since_best', 0)}, "
            f"balance={state.get('balance', 0):.0f}",
        )
        return state

    def _load_finetune_model(self):
        """Load model weights from checkpoint but reset all training state.

        Fine-tune keeps the model's learned patterns (actor/critic/entropy weights)
        but starts a fresh training run: week 0, fresh balance, fresh convergence.
        The LR schedule and num_timesteps reset so the model gets the full
        warmup -> hold -> decay cycle again.

        The replay buffer from the checkpoint is kept — it contains real market
        experience that helps the model adapt faster under new reward settings.
        """
        # Determine which model to load
        override_path = self.shared_metrics.get("_resume_model_path")
        if override_path:
            model_path = Path(override_path)
        else:
            model_path = self.cfg.model_dir / "spartus_latest.zip"

        if not model_path.exists():
            self.logger.log_alert("WARNING", f"Fine-tune model not found: {model_path}, creating new")
            self.memory.reset_for_fresh_training()
            self._create_model()
            return

        # Load model weights
        dummy_week = self._weeks_data[0] if self._weeks_data else None
        if not dummy_week:
            raise ValueError("No training data available for fine-tuning")

        features_df = self._get_features(dummy_week)
        dummy_vec_env = self._make_vec_env(features_df, week_idx=0)
        lr_schedule = self._build_lr_schedule()
        self.model = ClippedSAC.load(
            str(model_path), env=dummy_vec_env,
            custom_objects={"learning_rate": lr_schedule},
        )
        self.model._max_grad_norm = self.cfg.max_grad_norm

        # Reset LR schedule — fine-tune gets the full warmup -> hold -> decay cycle
        self.model.num_timesteps = 0
        self._lr_timesteps[0] = 0

        # Clear replay buffer — old transitions contain rewards from the previous
        # reward function. Training the critic on stale rewards would fight the
        # new reward signals. The model retains its policy/critic WEIGHTS as
        # initialization but collects fresh experience under the new config.
        self.model.replay_buffer.reset()

        # Skip the learning_starts warmup — the model already has learned weights,
        # it doesn't need 5000 steps of random collection before learning.
        self.model.learning_starts = 0

        # Enforce entropy alpha floor
        if hasattr(self.model, 'log_ent_coef') and self.cfg.min_entropy_alpha > 0:
            import math, torch
            min_log = math.log(self.cfg.min_entropy_alpha)
            if self.model.log_ent_coef.item() < min_log:
                with torch.no_grad():
                    self.model.log_ent_coef.data.clamp_(min=min_log)

        # Override target_entropy from config
        if hasattr(self.model, 'target_entropy'):
            self.model.target_entropy = self.cfg.target_entropy

        # Set up logger
        self._tb_run_dir = self._get_tb_run_dir()
        sb3_logger = sb3_configure(self._tb_run_dir, ["tensorboard"])
        self.model.set_logger(sb3_logger)

        # Re-inject regularization into loaded model
        self._inject_regularization()

        # Reset ALL training state — fresh run with existing weights
        self.current_week = 0
        self.balance = self.cfg.initial_balance
        self.peak_balance = self.balance
        self._bankruptcies = 0
        self._best_checkpoint_week = 0
        self._lr_multiplier[0] = 1.0
        self._plateau_lr_reductions = 0
        self._stagnation_weeks = 0
        self._soft_correction_active = False
        self._rollback_count = 0
        self._last_rollback_week = -999
        self._forced_stage2_until = 0
        self.convergence = ConvergenceDetector(self.cfg)
        self.memory.reset_for_fresh_training()

        self.logger.log_alert(
            "INFO",
            f"Fine-tune loaded from {model_path.name}. "
            f"Weights preserved, counters reset. Week=0, Balance={self.balance:.0f}",
        )

    # === Graduated Overfitting Defense =======================================

    def _graduated_defense(self, week_idx: int):
        """4-layer graduated defense against overfitting.

        Layer 1: Prevention — dropout + LayerNorm (applied at model creation)
        Layer 2: Soft correction — boost entropy + weight decay after 10 weeks
        Layer 3: Rollback with modification — load best, boost entropy, clear buffer
        Layer 4: Hard reset — critic layer reset + force curriculum Stage 2
        """
        ow = self.convergence.overfitting_weeks
        state = self.convergence.state

        # Track stagnation (OVERFITTING, VAL_DECLINING, or PLATEAU) (FIX-B4)
        # VAL_DECLINING added: regime-shift causes both metrics to drop, which
        # bypassed this check entirely since state was always "IMPROVING".
        if state in ("OVERFITTING", "VAL_DECLINING", "PLATEAU"):
            self._stagnation_weeks += 1
        else:
            self._stagnation_weeks = 0
            # Clear soft correction if state recovered
            if self._soft_correction_active:
                self._undo_soft_correction()
            return

        # --- Layer 2: Soft correction (10+ consecutive overfitting weeks) ---
        if ow >= self.cfg.soft_correction_weeks and not self._soft_correction_active:
            self._apply_soft_correction()

        # --- Layer 3: Rollback with modification (30+ weeks) ---
        if (ow >= self.cfg.rollback_trigger_weeks
                and self._best_checkpoint_week > 0):
            # Check if this is a repeat rollback within cooldown -> Layer 4
            weeks_since_last_rollback = week_idx - self._last_rollback_week
            if (self._rollback_count > 0
                    and weeks_since_last_rollback < self.cfg.hard_reset_cooldown):
                # Layer 4: Hard reset
                self._hard_reset(week_idx)
            else:
                # Layer 3: Standard rollback with modification
                self._rollback_with_modification(week_idx)

            self.convergence.overfitting_weeks = 0  # Give corrected model a fresh chance
            self._stagnation_weeks = 0
            return

        # --- LR plateau reduction (20+ stagnation weeks, up to 2 times) ---
        if (self._stagnation_weeks >= 20
                and self._plateau_lr_reductions < 2):
            self._lr_multiplier[0] *= 0.5
            self._plateau_lr_reductions += 1
            self.logger.log_alert(
                "WARNING",
                f"LR plateau reduction #{self._plateau_lr_reductions}: "
                f"multiplier now {self._lr_multiplier[0]:.2f} "
                f"(stagnating for {self._stagnation_weeks} weeks, "
                f"state={state})",
            )
            self._stagnation_weeks = 0

        # --- PLATEAU escape: force rollback after N stagnation weeks with LR maxed ---
        # PLATEAU does not increment overfitting_weeks, so Layer 3 (ow>=30) never fires
        # from PLATEAU alone. This is a dead-end: the model is stuck but defenses are silent.
        # After plateau_rollback_weeks of continuous stagnation with LR at floor, force a rollback.
        if (state == "PLATEAU"
                and self._stagnation_weeks >= self.cfg.plateau_rollback_weeks
                and self._plateau_lr_reductions >= 2
                and self._best_checkpoint_week > 0):
            self.logger.log_alert(
                "WARNING",
                f"[PLATEAU ESCAPE] Stagnating {self._stagnation_weeks} weeks with "
                f"LR maxed (mult={self._lr_multiplier[0]:.3f}). Forcing rollback.",
            )
            weeks_since_last_rollback = week_idx - self._last_rollback_week
            if (self._rollback_count > 0
                    and weeks_since_last_rollback < self.cfg.hard_reset_cooldown):
                self._hard_reset(week_idx)
            else:
                self._rollback_with_modification(week_idx)
            self.convergence.overfitting_weeks = 0
            self._stagnation_weeks = 0

    def _apply_soft_correction(self):
        """Layer 2: Boost entropy and weight decay to nudge model back on track."""
        if not self.model:
            return
        import torch, math

        # Boost entropy alpha
        if hasattr(self.model, 'log_ent_coef'):
            old_alpha = torch.exp(self.model.log_ent_coef).item()
            new_alpha = old_alpha * self.cfg.entropy_boost_mult
            new_log = math.log(max(new_alpha, 1e-8))
            with torch.no_grad():
                self.model.log_ent_coef.data.fill_(new_log)
            self.logger.log_alert(
                "WARNING",
                f"[SOFT CORRECTION] Entropy boosted: {old_alpha:.4f} -> {new_alpha:.4f} "
                f"(×{self.cfg.entropy_boost_mult})",
            )

        # Boost weight decay in actor + critic optimizers
        for optimizer in [self.model.actor.optimizer, self.model.critic.optimizer]:
            for param_group in optimizer.param_groups:
                old_wd = param_group.get("weight_decay", 0)
                param_group["weight_decay"] = old_wd * self.cfg.weight_decay_boost_mult
        self.logger.log_alert(
            "WARNING",
            f"[SOFT CORRECTION] Weight decay boosted ×{self.cfg.weight_decay_boost_mult}",
        )

        self._soft_correction_active = True

    def _undo_soft_correction(self):
        """Revert soft correction when overfitting clears."""
        if not self.model:
            return

        # Restore weight decay to original
        for optimizer in [self.model.actor.optimizer, self.model.critic.optimizer]:
            for param_group in optimizer.param_groups:
                old_wd = param_group.get("weight_decay", 0)
                if old_wd > 0 and self.cfg.weight_decay_boost_mult > 0:
                    param_group["weight_decay"] = old_wd / self.cfg.weight_decay_boost_mult

        self._soft_correction_active = False
        self.logger.log_alert("INFO", "[SOFT CORRECTION] Cleared — model recovered from overfitting")

    def _rollback_with_modification(self, week_idx: int):
        """Layer 3: Rollback to best checkpoint + mild entropy boost.

        FIX-B8: Reduced destabilization.  Old version did 4 things at once
        (revert + 2x entropy + clear 30% buffer + reset LR to full) which
        caused actor loss to jump 4x and destroyed a strong model.

        New version: revert weights + mild 1.25x entropy only.
        Buffer and LR are preserved for continuity.
        """
        self.logger.log_alert(
            "CRITICAL",
            f"[ROLLBACK L3] Overfitting for {self.convergence.overfitting_weeks} weeks. "
            f"Rolling back to best checkpoint (week {self._best_checkpoint_week}) "
            f"with mild entropy boost.",
        )

        # Load best checkpoint
        self._revert_to_best_checkpoint()

        # Mild entropy boost (1.25x, not 2x — reduces destabilization)
        if self.model and hasattr(self.model, 'log_ent_coef'):
            import torch, math
            old_alpha = torch.exp(self.model.log_ent_coef).item()
            new_alpha = old_alpha * 1.25  # Mild, not cfg.rollback_entropy_mult (2.0)
            new_log = math.log(max(new_alpha, 1e-8))
            with torch.no_grad():
                self.model.log_ent_coef.data.fill_(new_log)
            self.logger.log_alert(
                "INFO",
                f"[ROLLBACK L3] Entropy boosted: {old_alpha:.4f} -> {new_alpha:.4f} (×1.25)",
            )

        # FIX-B8: Do NOT clear replay buffer — existing experience is valuable
        # and removing it forces the model to re-learn from scratch.

        # FIX-B8: Do NOT reset LR — keep current LR schedule position.
        # Jumping from 1.34e-4 back to 3.0e-4 destabilized the actor.

        self._soft_correction_active = False
        self._rollback_count += 1
        self._last_rollback_week = week_idx

        # Trim val history to prevent pre-rollback era scores from poisoning the
        # overfitting/val-declining comparison windows after recovery.
        # Old era scores in [-6:-3] window inflate prev_avg, making post-rollback
        # recovery look like a decline (false-positive OVERFITTING).
        val_len = len(self.convergence.val_sharpes)
        if val_len > 3:
            self.convergence.val_sharpes = self.convergence.val_sharpes[-3:]
            self.convergence.val_returns = self.convergence.val_returns[-3:]
            self.convergence.train_sharpes_at_val = self.convergence.train_sharpes_at_val[-3:]
            # Recompute best_val_score from trimmed window
            self.convergence.best_val_score = max(
                self.convergence.compute_val_score(s, r)
                for s, r in zip(
                    self.convergence.val_sharpes, self.convergence.val_returns
                )
            )
            self.convergence.weeks_since_best = 0
            self.logger.log_alert(
                "INFO",
                f"[ROLLBACK L3] Val history trimmed {val_len} -> 3 points. "
                f"Recomputed best_val_score={self.convergence.best_val_score:.3f}. "
                f"Pre-rollback era scores removed from comparison window.",
            )

        # Partially recover LR — the rolled-back model is better; give it room to improve.
        # Double the multiplier (capped at 0.5) and remove one LR reduction credit.
        if self._plateau_lr_reductions > 0:
            recovered_mult = min(self._lr_multiplier[0] * 2.0, 0.5)
            self._lr_multiplier[0] = recovered_mult
            self._plateau_lr_reductions = max(0, self._plateau_lr_reductions - 1)
            self.logger.log_alert(
                "INFO",
                f"[ROLLBACK L3] LR partially recovered: "
                f"multiplier={self._lr_multiplier[0]:.3f}, "
                f"plateau_lr_reductions={self._plateau_lr_reductions}",
            )

    def _hard_reset(self, week_idx: int):
        """Layer 4: Hard reset — rollback + buffer clear + LR reset + critic reset + Stage 2.

        FIX-B8: Buffer clear and LR reset moved here from Layer 3.
        Layer 4 is the nuclear option — Layer 3 should be gentler.
        """
        self.logger.log_alert(
            "CRITICAL",
            f"[HARD RESET L4] Second rollback within {self.cfg.hard_reset_cooldown} weeks. "
            f"Resetting critic layers + clearing buffer + forcing Stage 2 for "
            f"{self.cfg.hard_reset_curriculum_weeks} weeks.",
        )

        # First do standard rollback (weights + mild entropy)
        self._rollback_with_modification(week_idx)

        # Layer 4 additions: buffer clear + LR reset (moved from Layer 3)
        if self.model and hasattr(self.model, 'replay_buffer'):
            buf = self.model.replay_buffer
            if buf.full or buf.pos > 0:
                n_stored = buf.buffer_size if buf.full else buf.pos
                n_clear = int(n_stored * self.cfg.rollback_buffer_clear_pct)
                if n_clear > 0:
                    buf.pos = max(0, buf.pos - n_clear)
                    if buf.full:
                        buf.full = False
                    self.logger.log_alert(
                        "INFO",
                        f"[HARD RESET L4] Cleared {n_clear:,} newest replay buffer entries",
                    )

        self._lr_multiplier[0] = 1.0
        self._plateau_lr_reductions = 0

        # Reset last 2 layers of each critic Q-network (plasticity preservation)
        if self.model:
            import torch
            for qnet in self.model.critic.q_networks:
                modules = list(qnet.children())
                for module in modules[-3:]:
                    if isinstance(module, torch.nn.Linear):
                        torch.nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            torch.nn.init.zeros_(module.bias)
            self.model.critic_target.load_state_dict(self.model.critic.state_dict())
            self.logger.log_alert(
                "INFO",
                "[HARD RESET L4] Critic output layers re-initialized (plasticity reset)",
            )

        # Force curriculum back to Stage 2 (mixed sampling) for N weeks
        self._forced_stage2_until = week_idx + self.cfg.hard_reset_curriculum_weeks
        self.logger.log_alert(
            "INFO",
            f"[HARD RESET L4] Forced Stage 2 sampling until week {self._forced_stage2_until}",
        )

    def _strip_regularization(self):
        """Remove injected Dropout + LayerNorm so state_dict is save/load compatible.

        Called before model.save(). Re-inject after save with _inject_regularization().
        """
        if not self.model:
            return
        import torch.nn as nn

        def _strip_sequential(seq: nn.Sequential) -> nn.Sequential:
            """Remove LayerNorm and Dropout, keep only Linear + ReLU."""
            kept = [m for m in seq if not isinstance(m, (nn.LayerNorm, nn.Dropout))]
            return nn.Sequential(*kept)

        # Strip critic
        for i, qnet in enumerate(self.model.critic.q_networks):
            stripped = _strip_sequential(qnet)
            self.model.critic.q_networks[i] = stripped
            setattr(self.model.critic, f"qf{i}", stripped)

        # Strip critic target
        for i, qnet in enumerate(self.model.critic_target.q_networks):
            stripped = _strip_sequential(qnet)
            self.model.critic_target.q_networks[i] = stripped
            setattr(self.model.critic_target, f"qf{i}", stripped)

        # Strip actor
        if hasattr(self.model.actor, 'latent_pi'):
            self.model.actor.latent_pi = _strip_sequential(self.model.actor.latent_pi)

        device = next(self.model.policy.parameters()).device
        self.model.critic.to(device)
        self.model.critic_target.to(device)
        self.model.actor.to(device)

    def _inject_curated_buffer(self):
        """Replace the model's standard ReplayBuffer with CuratedTrainingBuffer.

        Called after model creation and after resume load. The curated buffer
        maintains a reservoir-sampled core tier (~20% of capacity) that permanently
        retains diverse samples from all training eras, preventing catastrophic
        forgetting as the curriculum advances from 2015→2026.

        Memory overhead: ~1.5 MB (3 small metadata arrays). No duplicate obs storage.
        """
        if not self.model:
            return
        old_buf = self.model.replay_buffer
        curated = CuratedTrainingBuffer(
            buffer_size=old_buf.buffer_size * old_buf.n_envs,  # restore pre-n_envs-division size
            observation_space=self.model.observation_space,
            action_space=self.model.action_space,
            device=self.model.device,
            n_envs=self.model.n_envs,
            optimize_memory_usage=False,
            handle_timeout_termination=True,
            core_fraction=0.20,
            sample_core_pct=0.35,
        )
        self.model.replay_buffer = curated
        self.logger.log_alert(
            "INFO",
            f"[CURATED BUFFER] Installed — core={curated._core_capacity} slots "
            f"({20}% of {curated.buffer_size}), sample_core={35}%",
        )

    def _save_model_safe(self, path):
        """Save model with strip/re-inject to ensure save/load compatibility."""
        self._strip_regularization()
        self.model.save(str(path))
        self._inject_regularization()

    def _inject_regularization(self):
        """Layer 1: Inject Dropout + LayerNorm into actor/critic networks post-creation.

        Called once after model creation. Modifies the Sequential networks
        in-place by inserting LayerNorm before ReLU and Dropout after ReLU.
        """
        if not self.model:
            return
        import torch.nn as nn

        dropout_p = self.cfg.critic_dropout
        use_ln = self.cfg.use_layer_norm

        if dropout_p <= 0 and not use_ln:
            return

        def _inject_into_sequential(seq: nn.Sequential, apply_dropout: bool) -> nn.Sequential:
            """Rebuild a Sequential with LayerNorm + Dropout injected."""
            new_modules = []
            for module in seq:
                if isinstance(module, nn.ReLU):
                    if use_ln and new_modules:
                        # Insert LayerNorm before ReLU (based on previous Linear's out_features)
                        prev = new_modules[-1]
                        if isinstance(prev, nn.Linear):
                            new_modules.append(nn.LayerNorm(prev.out_features))
                    new_modules.append(module)
                    if apply_dropout and dropout_p > 0:
                        new_modules.append(nn.Dropout(p=dropout_p))
                else:
                    new_modules.append(module)
            return nn.Sequential(*new_modules)

        # Inject into critic Q-networks (dropout + LayerNorm)
        for i, qnet in enumerate(self.model.critic.q_networks):
            new_qnet = _inject_into_sequential(qnet, apply_dropout=True)
            self.model.critic.q_networks[i] = new_qnet
            setattr(self.model.critic, f"qf{i}", new_qnet)

        # Inject into critic_target FIRST (same architecture), then copy weights
        for i, qnet in enumerate(self.model.critic_target.q_networks):
            new_qnet = _inject_into_sequential(qnet, apply_dropout=True)
            self.model.critic_target.q_networks[i] = new_qnet
            setattr(self.model.critic_target, f"qf{i}", new_qnet)

        # Now both have matching architecture — copy weights
        self.model.critic_target.load_state_dict(self.model.critic.state_dict())

        # Inject into actor (LayerNorm only, no dropout — dropout destabilizes policy gradient)
        if hasattr(self.model.actor, 'latent_pi'):
            new_latent = _inject_into_sequential(self.model.actor.latent_pi, apply_dropout=False)
            self.model.actor.latent_pi = new_latent

        # Move new modules to same device as model
        device = next(self.model.policy.parameters()).device
        self.model.critic.to(device)
        self.model.critic_target.to(device)
        self.model.actor.to(device)

        parts = []
        if use_ln:
            parts.append("LayerNorm")
        if dropout_p > 0:
            parts.append(f"Dropout(p={dropout_p})")
        self.logger.log_alert(
            "INFO",
            f"[LAYER 1] Injected {' + '.join(parts)} into actor/critic networks",
        )

    def _revert_to_best_checkpoint(self):
        """Revert model to best validation checkpoint (early stopping)."""
        best_path = self.cfg.best_model_path
        if not best_path.exists():
            # Fallback: try the weekly checkpoint
            best_path = self.cfg.model_dir / f"spartus_week_{self._best_checkpoint_week:04d}.zip"
        if not best_path.exists():
            self.logger.log_alert("WARNING", "No best checkpoint found to revert to")
            return

        try:
            dummy_week = self._weeks_data[0] if self._weeks_data else None
            if dummy_week:
                features_df = self._get_features(dummy_week)
                dummy_vec_env = self._make_vec_env(features_df, week_idx=0)
                lr_schedule = self._build_lr_schedule()
                self.model = ClippedSAC.load(
                    str(best_path), env=dummy_vec_env,
                    custom_objects={"learning_rate": lr_schedule},
                )
                self.model._max_grad_norm = self.cfg.max_grad_norm
                # FIX-12: Set num_timesteps to current training position, not old checkpoint's
                # This prevents LR schedule desync (old checkpoint has fewer timesteps)
                current_timesteps = self.current_week * self.cfg.steps_per_week * self.cfg.n_envs
                self.model.num_timesteps = current_timesteps
                self._lr_timesteps[0] = current_timesteps

                # Restore reward state if available
                meta_path = self.cfg.best_model_path.with_suffix(".meta.json")
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    rs = meta.get("reward_state")
                    if rs:
                        self._set_reward_state(rs)

                self._tb_run_dir = self._get_tb_run_dir()
                sb3_logger = sb3_configure(self._tb_run_dir, ["tensorboard"])
                self.model.set_logger(sb3_logger)

                # Re-inject regularization into loaded model
                self._inject_regularization()

                self.logger.log_alert(
                    "INFO",
                    f"Reverted to best model from week {self._best_checkpoint_week}",
                )
        except Exception as e:
            self.logger.log_alert("WARNING", f"Failed to revert: {e}")

    def _save_best_model(self, week: int, val_sharpe: float):
        """Save current model as the best model (validation Sharpe improved)."""
        if not self.model:
            return
        best_path = self.cfg.best_model_path
        best_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_model_safe(best_path)

        # Save metadata + reward state alongside best model
        meta = {
            "week": week,
            "val_sharpe": val_sharpe,
            "balance": self.balance,
            "timestamp": time.time(),
        }
        reward_state = self._get_reward_state()
        if reward_state:
            meta["reward_state"] = reward_state

        meta_path = best_path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

        # Save training state sidecar (FIX-RESUME Bug#1): same as _save_checkpoint,
        # so that "Resume from best" works correctly without collapsing convergence history.
        ts_sidecar = best_path.with_suffix(".training_state.json")
        ts = {
            "current_week": week,
            "balance": self.balance,
            "peak_balance": self.peak_balance,
            "convergence": self.convergence.get_summary(),
            "seed": self.seed,
            "best_checkpoint_week": self._best_checkpoint_week,
            "bankruptcies": self._bankruptcies,
            "rollback_count": self._rollback_count,
            "last_rollback_week": self._last_rollback_week,
            "forced_stage2_until": self._forced_stage2_until,
            "stagnation_weeks": self._stagnation_weeks,
            "plateau_lr_reductions": self._plateau_lr_reductions,
            "lr_multiplier": self._lr_multiplier[0],
        }
        if reward_state:
            ts["reward_state"] = reward_state
        with open(ts_sidecar, "w", encoding="utf-8") as f:
            json.dump(ts, f, indent=2, default=str)

        self.logger.log_alert(
            "INFO",
            f"Best model saved: week {week}, val_sharpe={val_sharpe:.4f}",
        )

    def _save_checkpoint(self, week: int, result: Dict):
        """Save a named checkpoint and clean up old ones.

        Retention policy:
        - Always keep: spartus_latest.zip, every 10th week, last 3 checkpoints
        - Delete intermediate checkpoints to save disk space
        """
        if not self.model:
            return
        path = self.cfg.model_dir / f"spartus_week_{week:04d}.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._save_model_safe(path)

        # Save reward state alongside checkpoint
        reward_state = self._get_reward_state()
        if reward_state:
            rs_path = path.with_suffix(".reward_state.json")
            with open(rs_path, "w", encoding="utf-8") as f:
                json.dump(reward_state, f, indent=2)

        # Save training state sidecar (FIX-RESUME): enables accurate Resume-from-checkpoint
        # by preserving convergence history, balance, and defense counters at this exact week.
        ts_sidecar = path.with_suffix(".training_state.json")
        ts = {
            "current_week": week,
            "balance": self.balance,
            "peak_balance": self.peak_balance,
            "convergence": self.convergence.get_summary(),
            "seed": self.seed,
            "best_checkpoint_week": self._best_checkpoint_week,
            "bankruptcies": self._bankruptcies,
            "rollback_count": self._rollback_count,
            "last_rollback_week": self._last_rollback_week,
            "forced_stage2_until": self._forced_stage2_until,
            "stagnation_weeks": self._stagnation_weeks,
            "plateau_lr_reductions": self._plateau_lr_reductions,
            "lr_multiplier": self._lr_multiplier[0],
        }
        if reward_state:
            ts["reward_state"] = reward_state
        with open(ts_sidecar, "w", encoding="utf-8") as f:
            json.dump(ts, f, indent=2, default=str)

        # Track checkpoint for dashboard
        self._last_checkpoint_week = week
        self.shared_metrics["last_checkpoint_week"] = week

        # Cleanup: keep latest, every 10th, and last 3
        self._cleanup_old_checkpoints(week)

    def _cleanup_old_checkpoints(self, current_week: int):
        """Remove old checkpoints, keeping only important ones."""
        keep_every_n = 10
        keep_recent = 3

        for f in sorted(self.cfg.model_dir.glob("spartus_week_*.zip")):
            # Parse week number from filename
            try:
                wk = int(f.stem.split("_")[-1])
            except ValueError:
                continue

            # Keep: every Nth, recent N, current, and best
            if wk % keep_every_n == 0:
                continue
            if wk >= current_week - keep_recent:
                continue
            if wk == self._best_checkpoint_week:
                continue

            f.unlink()  # Delete old checkpoint
            # Also remove sidecars so they don't accumulate on disk
            for suffix in (".reward_state.json", ".training_state.json"):
                sidecar = f.with_suffix(suffix)
                if sidecar.exists():
                    sidecar.unlink()
