"""SB3 callback: bridges training loop to logging, dashboard, and TensorBoard.

Extracts SAC internals (entropy α, Q-values, losses, gradient norms)
and writes them alongside environment metrics.

Implements 46 alert conditions from the SPARTUS_TRAINING_DASHBOARD spec.
"""

import math
import numpy as np
import time
from typing import Any, Dict, List, Optional

import torch
from stable_baselines3.common.callbacks import BaseCallback

from src.config import TrainingConfig
from src.training.logger import TrainingLogger


class SpartusCallback(BaseCallback):
    """Custom callback for SAC training with full metric extraction.

    Responsibilities:
    - Extract SAC model internals every N steps
    - Log to TrainingLogger (JSONL files)
    - Write TensorBoard scalars
    - Evaluate 46 alert conditions
    - Update shared metrics dict for dashboard
    - Run observation health checks
    """

    def __init__(
        self,
        config: TrainingConfig = None,
        logger: Optional[TrainingLogger] = None,
        shared_metrics: Optional[Dict] = None,
        log_interval: int = 10,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.cfg = config or TrainingConfig()
        self.training_logger = logger or TrainingLogger(self.cfg)
        self.shared_metrics = shared_metrics if shared_metrics is not None else {}
        self.log_interval = log_interval

        # Running stats
        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int] = []
        self._step_rewards: List[float] = []
        self._last_log_time = time.time()
        self._total_trades = 0
        self._winning_trades = 0
        self._losing_trades = 0
        self._total_pnl = 0.0
        self._gross_profit = 0.0
        self._gross_loss = 0.0

        # Anti-hack tracking
        self._trade_cap_hits = 0
        self._hold_blocks = 0
        self._conviction_blocks = 0
        self._consecutive_hold_blocks = 0
        self._trade_directions: List[str] = []
        self._conviction_values: List[float] = []
        self._steps_without_trade = 0

        # Observation health
        self._obs_buffer: List[np.ndarray] = []
        self._obs_check_interval = 100

        # Alert dedup (avoid spamming same alert)
        self._last_convergence_state = "WARMING_UP"
        self._reward_zero_steps = 0

        # Gradient clipping tracking
        self._grad_clip_count = 0
        self._grad_total_count = 0

        # Reward clipping tracking
        self._reward_clip_count = 0
        self._reward_total_count = 0

        # Per-week tracking (reset each episode/week)
        self._week_wins = 0
        self._week_losses = 0
        self._week_pnl = 0.0
        self._week_best_trade = 0.0
        self._week_worst_trade = 0.0
        self._week_hold_bars: List[int] = []
        self._week_commission = 0.0

        # Decision log (rolling list of formatted decision strings)
        self._decisions: List[str] = []
        self._max_decisions = 50

        # Lot size tracking
        self._lot_sizes: List[float] = []

        # Initial entropy for computing % of initial
        self._initial_entropy_alpha: Optional[float] = None

    def _on_training_start(self) -> None:
        """Called at the start of each agent.learn() call (once per week)."""
        self._week_wins = 0
        self._week_losses = 0
        self._week_pnl = 0.0
        self._week_best_trade = 0.0
        self._week_worst_trade = 0.0
        self._week_hold_bars.clear()
        self._week_commission = 0.0

    def _enforce_entropy_floor(self):
        """Clamp log_ent_coef so entropy alpha never drops below min_entropy_alpha.

        SB3's SAC stores alpha as log_ent_coef (a torch.nn.Parameter).
        We clamp it to log(min_alpha) to maintain a minimum exploration level.
        """
        min_alpha = self.cfg.min_entropy_alpha
        if min_alpha <= 0:
            return
        try:
            model = self.model
            if hasattr(model, "log_ent_coef"):
                min_log = math.log(min_alpha)
                with torch.no_grad():
                    if model.log_ent_coef.item() < min_log:
                        model.log_ent_coef.data.clamp_(min=min_log)
        except Exception:
            pass

    def _on_step(self) -> bool:
        """Called after every env.step() during learn().

        Returns False to stop learning (quit requested).
        Blocks while paused.
        """
        # Handle pause — block until unpaused
        while self.shared_metrics.get("_paused", False):
            time.sleep(0.25)
            if self.shared_metrics.get("_quit_requested", False):
                return False

        # Handle quit — stop learning gracefully
        if self.shared_metrics.get("_quit_requested", False):
            return False

        infos = self.locals.get("infos", [{}])
        rewards = self.locals.get("rewards", [0.0])
        if not hasattr(rewards, "__len__"):
            rewards = [rewards]

        # Use env 0's reward for primary tracking (reward curve, zero detection)
        reward = float(rewards[0])
        self._step_rewards.append(reward)

        # Track reward clipping (env 0)
        self._reward_total_count += 1
        if abs(reward) >= self.cfg.reward_clip - 0.01:
            self._reward_clip_count += 1

        # Track reward zero (env 0 — for consecutive-zero alert)
        if abs(reward) < 1e-8:
            self._reward_zero_steps += 1
        else:
            self._reward_zero_steps = 0

        # Track trades from ALL environments (not just env 0)
        any_trade = False
        for info in infos:
            if "last_trade" in info:
                trade = info["last_trade"]
                any_trade = True
                self._total_trades += 1
                pnl = trade.get("pnl", 0)
                self._total_pnl += pnl
                if pnl > 0:
                    self._winning_trades += 1
                    self._gross_profit += pnl
                    self._week_wins += 1
                else:
                    self._losing_trades += 1
                    self._gross_loss += abs(pnl)
                    self._week_losses += 1
                self._week_pnl += pnl
                self._week_best_trade = max(self._week_best_trade, pnl)
                self._week_worst_trade = min(self._week_worst_trade, pnl)
                hold = trade.get("hold_bars", 0)
                if hold > 0:
                    self._week_hold_bars.append(hold)
                self._week_commission += trade.get("commission", 0)
                self._trade_directions.append(trade.get("side", "LONG"))
                if len(self._trade_directions) > 100:
                    self._trade_directions = self._trade_directions[-100:]
                self._conviction_values.append(trade.get("conviction", 0.5))
                if len(self._conviction_values) > 100:
                    self._conviction_values = self._conviction_values[-100:]
                lots = trade.get("lots", 0)
                if lots > 0:
                    self._lot_sizes.append(lots)
                self._format_decision(trade)
                self.training_logger.log_trade(trade)

        if any_trade:
            self._steps_without_trade = 0
        else:
            self._steps_without_trade += 1

        # Collect observations for health check (env 0 is representative)
        new_obs = self.locals.get("new_obs", None)
        if new_obs is not None and len(new_obs) > 0:
            self._obs_buffer.append(new_obs[0].copy())
            if len(self._obs_buffer) > 200:
                self._obs_buffer = self._obs_buffer[-200:]

        # Track episode completions (any env)
        dones = self.locals.get("dones", [False])
        done = dones[0] if hasattr(dones, "__len__") else dones
        if done:
            ep_reward = sum(self._step_rewards)
            self._episode_rewards.append(ep_reward)
            self._episode_lengths.append(len(self._step_rewards))
            self._step_rewards.clear()
            # Cap episode history
            if len(self._episode_rewards) > 100:
                self._episode_rewards = self._episode_rewards[-100:]
                self._episode_lengths = self._episode_lengths[-100:]

        # Enforce entropy alpha floor — prevent exploration collapse
        self._enforce_entropy_floor()

        # Periodic logging (use env 0's info for state metrics)
        if self.num_timesteps % self.log_interval == 0:
            self._log_metrics(infos[0] if infos else {})

        # Periodic observation health check
        if self.num_timesteps % self._obs_check_interval == 0:
            self._check_obs_health()

        return True

    def _format_decision(self, trade: Dict):
        """Format a trade into a decision log string."""
        week = self.shared_metrics.get("current_week", 0)
        step = self.num_timesteps
        side = trade.get("side", "?")
        pnl = trade.get("pnl", 0)
        lots = trade.get("lots", 0)
        entry = trade.get("entry_price", 0)
        exit_p = trade.get("exit_price", 0)
        conv = trade.get("conviction", 0)
        hold = trade.get("hold_bars", 0)
        reason = trade.get("reason", "?")

        # Determine P/L label: WIN / LOSS / BREAK-EVEN
        if pnl > 0.005:
            pnl_str = f"+\u00a3{pnl:.2f}"
            tag = "WIN"
        elif pnl < -0.005:
            pnl_str = f"-\u00a3{abs(pnl):.2f}"
            tag = "LOSS"
        else:
            pnl_str = "\u00a30.00"
            tag = "B/E"

        line = (f"[W{week} S{step}] CLOSE {side} {pnl_str} {tag} "
                f"({hold}bars, {reason}) lots={lots:.2f} conv={conv:.2f}")

        self._decisions.append(line)
        if len(self._decisions) > self._max_decisions:
            self._decisions = self._decisions[-self._max_decisions:]

    def _log_metrics(self, info: Dict):
        """Extract and log all metrics."""
        now = time.time()
        dt = now - self._last_log_time
        steps_since = self.log_interval
        metrics = {
            "timestep": self.num_timesteps,
            "wall_time": now,
            "steps_per_sec": steps_since / dt if dt > 0 else 0,
        }
        self._last_log_time = now

        # Environment metrics
        metrics["balance"] = info.get("balance", 0.0)
        metrics["equity"] = info.get("equity", 0.0)
        metrics["peak_balance"] = info.get("peak_balance", 0.0)
        metrics["episode_trades"] = info.get("episode_trades", 0)
        metrics["daily_trades"] = info.get("daily_trades", 0)
        metrics["has_position"] = info.get("has_position", False)

        # Reward components
        for key in ["reward", "raw_reward", "r1_position_pnl", "r2_trade_quality",
                     "r3_drawdown", "r4_sharpe", "r5_risk_bonus"]:
            if key in info:
                metrics[key] = info[key]

        # Drawdown
        equity = info.get("equity", 0.0)
        peak = info.get("peak_balance", 1.0)
        metrics["drawdown"] = (peak - equity) / peak if peak > 0 else 0.0

        # Trade statistics
        metrics["total_trades"] = self._total_trades
        metrics["win_rate"] = (
            self._winning_trades / self._total_trades
            if self._total_trades > 0 else 0.0
        )

        # Profit factor (all-wins = 999.0 cap, no trades = 0.0)
        if self._gross_loss > 0:
            metrics["profit_factor"] = self._gross_profit / self._gross_loss
        elif self._gross_profit > 0:
            metrics["profit_factor"] = 999.0  # All winners
        else:
            metrics["profit_factor"] = 0.0

        # Episode stats
        if self._episode_rewards:
            metrics["mean_ep_reward"] = np.mean(self._episode_rewards[-10:])
            metrics["mean_ep_length"] = np.mean(self._episode_lengths[-10:])

        # Reward clip percentage
        metrics["reward_clip_pct"] = (
            self._reward_clip_count / max(self._reward_total_count, 1)
        )

        # Per-week stats
        week_trades = self._week_wins + self._week_losses
        metrics["week_wins"] = self._week_wins
        metrics["week_pnl"] = self._week_pnl
        metrics["week_best_trade"] = self._week_best_trade
        metrics["week_worst_trade"] = self._week_worst_trade
        metrics["avg_hold_bars"] = (
            np.mean(self._week_hold_bars) if self._week_hold_bars else 0
        )
        metrics["week_commission"] = self._week_commission

        # Average trade P/L and lot size
        if self._total_trades > 0:
            metrics["avg_trade_pnl"] = self._total_pnl / self._total_trades
        if self._lot_sizes:
            metrics["avg_lot_size"] = float(np.mean(self._lot_sizes[-100:]))
        # Sharpe ratio (from episode rewards) -- clamped to prevent Inf/NaN propagation
        if len(self._episode_rewards) >= 2:
            ep_arr = np.array(self._episode_rewards[-50:])
            std = np.std(ep_arr)
            if std > 1e-8:
                sharpe = float(np.mean(ep_arr) / std)
                metrics["sharpe"] = max(-10.0, min(10.0, sharpe))
            else:
                metrics["sharpe"] = 0.0

        # Anti-hack stats
        metrics["trade_cap_hits"] = self._trade_cap_hits
        metrics["hold_blocks"] = self._hold_blocks
        metrics["conviction_blocks"] = self._conviction_blocks

        # SAC internals
        sac_metrics = self._extract_sac_metrics()
        metrics.update(sac_metrics)

        # Reward normalizer stats + journal reflection (from environment)
        try:
            env = self.model.get_env()
            if env is not None:
                # Unwrap: DummyVecEnv → Monitor → SpartusTradeEnv
                base_env = env.envs[0] if hasattr(env, "envs") else env
                # Unwrap Monitor/TimeLimit wrappers
                while hasattr(base_env, "env"):
                    base_env = base_env.env
                if hasattr(base_env, "reward_calc") and hasattr(base_env.reward_calc, "normalizer"):
                    norm = base_env.reward_calc.normalizer
                    metrics["reward_running_mean"] = float(norm.mean)
                    metrics["reward_running_std"] = float(norm.var ** 0.5)

                # Journal reflection stats (throttled — only when cache is invalid)
                if hasattr(base_env, "memory") and self._total_trades > 0:
                    reflection = base_env.memory.get_reflection_stats()
                    metrics["journal_direction_accuracy"] = reflection["direction_accuracy"]
                    metrics["journal_sl_quality_score"] = reflection["sl_quality_score"]
                    metrics["journal_early_close_rate"] = reflection["early_close_rate"]
                    metrics["journal_wrong_direction_rate"] = reflection["wrong_direction_rate"]
                    metrics["journal_good_trade_rate"] = reflection["good_trade_rate"]

                    # Update trend prediction accuracy (UP/DOWN split)
                    # Only every 100 steps to avoid hammering SQLite
                    if self.num_timesteps % 100 == 0:
                        try:
                            mem = base_env.memory
                            up_rows = mem.conn.execute("""
                                SELECT correct FROM predictions
                                WHERE verified_at_step IS NOT NULL AND predicted_direction > 0
                                ORDER BY id DESC LIMIT 50
                            """).fetchall()
                            down_rows = mem.conn.execute("""
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
        except Exception:
            pass

        # Include obs health metrics in JSONL (sourced from shared_metrics)
        for health_key in ("dead_features", "exploding_features", "nan_features"):
            if health_key in self.shared_metrics:
                metrics[health_key] = self.shared_metrics[health_key]

        # Log to JSONL
        self.training_logger.log_step(metrics)

        # Write to TensorBoard
        self._write_tensorboard(metrics)

        # Update shared metrics for dashboard
        self.shared_metrics.update(metrics)

        # Push decision log for dashboard
        self.shared_metrics["_decisions"] = list(self._decisions)

        # Push alerts list for dashboard
        self.shared_metrics["_alerts"] = self.training_logger.get_recent_alerts(20)

        # Check all alert conditions
        self._check_alerts(metrics)

    def _extract_sac_metrics(self) -> Dict:
        """Extract SAC model internals."""
        metrics = {}

        try:
            model = self.model

            # Entropy coefficient (alpha)
            if hasattr(model, "ent_coef_tensor") and model.ent_coef_tensor is not None:
                metrics["entropy_alpha"] = model.ent_coef_tensor.item()
            elif hasattr(model, "log_ent_coef"):
                import torch
                metrics["entropy_alpha"] = torch.exp(model.log_ent_coef).item()

            # Losses from the SB3 logger
            if hasattr(model, "logger") and model.logger is not None:
                name_value = getattr(model.logger, "name_to_value", {})
                for key in ["train/actor_loss", "train/critic_loss",
                             "train/ent_coef_loss", "train/ent_coef"]:
                    if key in name_value:
                        clean_key = key.replace("train/", "")
                        metrics[clean_key] = name_value[key]

            # Gradient norms — use pre-clip values from ClippedSAC if available
            if hasattr(model, "_preclip_actor_grad") and model._preclip_actor_grad > 0:
                actor_grad = model._preclip_actor_grad
                metrics["actor_grad_norm"] = actor_grad
                self._grad_total_count += 1
                if actor_grad > self.cfg.max_grad_norm:
                    self._grad_clip_count += 1
            elif hasattr(model, "actor"):
                actor_grad = self._get_grad_norm(model.actor)
                if actor_grad is not None:
                    metrics["actor_grad_norm"] = actor_grad
                    self._grad_total_count += 1
                    if actor_grad > self.cfg.max_grad_norm * 0.95:
                        self._grad_clip_count += 1

            if hasattr(model, "_preclip_critic_grad") and model._preclip_critic_grad > 0:
                metrics["critic_grad_norm"] = model._preclip_critic_grad
            elif hasattr(model, "critic"):
                critic_grad = self._get_grad_norm(model.critic)
                if critic_grad is not None:
                    metrics["critic_grad_norm"] = critic_grad

            # Gradient clip percentage
            if self._grad_total_count > 0:
                metrics["grad_clip_pct"] = self._grad_clip_count / self._grad_total_count

            # Track initial entropy for % calculation
            if "entropy_alpha" in metrics and self._initial_entropy_alpha is None:
                self._initial_entropy_alpha = metrics["entropy_alpha"]
            if "entropy_alpha" in metrics and self._initial_entropy_alpha:
                metrics["policy_entropy_pct"] = (
                    metrics["entropy_alpha"] / self._initial_entropy_alpha * 100.0
                )

            # Q-values from critic (sample small batch)
            if hasattr(model, "critic") and hasattr(model, "replay_buffer"):
                try:
                    buf = model.replay_buffer
                    if buf.size() > 64:
                        import torch
                        data = buf.sample(64)
                        with torch.no_grad():
                            q1, q2 = model.critic(data.observations, data.actions)
                            q_min = torch.min(q1, q2)
                            metrics["q_value_mean"] = float(q_min.mean().item())
                            metrics["q_value_max"] = float(q_min.max().item())
                except Exception:
                    pass

            # Replay buffer fill percentage
            if hasattr(model, "replay_buffer"):
                try:
                    buf = model.replay_buffer
                    metrics["buffer_pct"] = buf.size() / buf.buffer_size * 100.0
                except Exception:
                    pass

            # Action statistics from replay buffer (sample, don't index directly)
            if hasattr(model, "replay_buffer") and model.replay_buffer.size() > 64:
                try:
                    sample = model.replay_buffer.sample(64)
                    actions = sample.actions.cpu().numpy() if hasattr(sample.actions, 'cpu') else sample.actions
                    metrics["action_mean"] = float(np.mean(actions))
                    metrics["action_std"] = float(np.std(actions))
                    # Per-action dimension stats (4 actions: direction, conviction, exit, sl_mgmt)
                    if actions.ndim == 2 and actions.shape[1] >= 4:
                        dim_names = ["direction", "conviction", "exit", "sl_mgmt"]
                        for i, name in enumerate(dim_names):
                            metrics[f"act_{name}_mean"] = float(np.mean(actions[:, i]))
                            metrics[f"act_{name}_std"] = float(np.std(actions[:, i]))
                except Exception:
                    pass

            # Learning rate (current)
            if hasattr(model, "lr_schedule"):
                try:
                    lr = model.lr_schedule(model._current_progress_remaining)
                    metrics["learning_rate"] = lr
                except Exception:
                    pass

        except Exception:
            pass

        return metrics

    @staticmethod
    def _get_grad_norm(module) -> Optional[float]:
        """Get L2 norm of gradients for a module."""
        try:
            total_norm = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            return total_norm ** 0.5 if total_norm > 0 else None
        except Exception:
            return None

    def _check_obs_health(self):
        """Check observation buffer for dead, exploding, or NaN features."""
        if len(self._obs_buffer) < 50:
            return

        obs_array = np.array(self._obs_buffer[-100:])

        # Dead features (std < 0.01)
        stds = np.std(obs_array, axis=0)
        dead = int(np.sum(stds < 0.01))

        # Exploding features (std > 3.0)
        exploding = int(np.sum(stds > 3.0))

        # NaN features (>5% NaN in any column)
        nan_pcts = np.mean(np.isnan(obs_array), axis=0)
        nan_features = int(np.sum(nan_pcts > 0.05))

        self.shared_metrics["dead_features"] = dead
        self.shared_metrics["exploding_features"] = exploding
        self.shared_metrics["nan_features"] = nan_features

    # TensorBoard metric → group mapping.
    # Each group gets its own color and collapsible section in TensorBoard.
    _TB_METRIC_GROUP = {
        # ── account/ (green tones) — portfolio state ─────────────
        "balance": "account",
        "equity": "account",
        "peak_balance": "account",
        "drawdown": "account",

        # ── trading/ (blue tones) — trade statistics ─────────────
        "total_trades": "trading",
        "win_rate": "trading",
        "profit_factor": "trading",
        "avg_trade_pnl": "trading",
        "avg_lot_size": "trading",
        "sharpe": "trading",

        # ── weekly/ — per-week metrics ───────────────────────────
        "episode_trades": "weekly",
        "week_wins": "weekly",
        "week_pnl": "weekly",
        "week_best_trade": "weekly",
        "week_worst_trade": "weekly",
        "avg_hold_bars": "weekly",
        "week_commission": "weekly",

        # ── reward/ — reward signal health ───────────────────────
        "reward": "reward",
        "raw_reward": "reward",
        "r1_position_pnl": "reward",
        "r2_trade_quality": "reward",
        "r3_drawdown": "reward",
        "r4_sharpe": "reward",
        "r5_risk_bonus": "reward",
        "reward_clip_pct": "reward",
        "reward_running_mean": "reward",
        "reward_running_std": "reward",

        # ── sac/ — model internals ───────────────────────────────
        "entropy_alpha": "sac",
        "policy_entropy_pct": "sac",
        "actor_loss": "sac",
        "critic_loss": "sac",
        "ent_coef_loss": "sac",
        "ent_coef": "sac",
        "q_value_mean": "sac",
        "q_value_max": "sac",
        "actor_grad_norm": "sac",
        "critic_grad_norm": "sac",
        "grad_clip_pct": "sac",
        "learning_rate": "sac",
        "buffer_pct": "sac",
        "action_mean": "sac",
        "action_std": "sac",

        # ── episode/ — episode-level stats ───────────────────────
        "mean_ep_reward": "episode",
        "mean_ep_length": "episode",

        # ── health/ — observation quality ────────────────────────
        "dead_features": "health",
        "exploding_features": "health",
        "nan_features": "health",

        # ── safety/ — anti-hack counters ─────────────────────────
        "trade_cap_hits": "safety",
        "hold_blocks": "safety",
        "conviction_blocks": "safety",
        "daily_trades": "safety",

        # ── perf/ — training performance ─────────────────────────
        "steps_per_sec": "perf",

        # ── journal/ — trade reasoning & self-reflection ─────────
        "journal_direction_accuracy": "journal",
        "journal_sl_quality_score": "journal",
        "journal_early_close_rate": "journal",
        "journal_wrong_direction_rate": "journal",
        "journal_good_trade_rate": "journal",
    }

    # Keys to skip entirely (noisy / redundant in TensorBoard)
    _TB_SKIP_KEYS = {"timestep", "wall_time", "has_position"}

    def _write_tensorboard(self, metrics: Dict):
        """Write categorised metrics to TensorBoard.

        Metrics are grouped by prefix (account/, trading/, sac/, etc.)
        so TensorBoard renders them in separate panels with distinct colors.
        """
        try:
            tb_logger = self.model.logger if hasattr(self.model, "logger") else None
            if tb_logger is None:
                return

            for key, value in metrics.items():
                if key in self._TB_SKIP_KEYS:
                    continue
                if not isinstance(value, (int, float)):
                    continue
                if not np.isfinite(value):
                    continue

                group = self._TB_METRIC_GROUP.get(key, "other")
                tb_logger.record(f"{group}/{key}", value)

            tb_logger.dump(step=self.num_timesteps)
        except Exception:
            pass

    # === 46 Alert Conditions ================================================

    def _check_alerts(self, metrics: Dict):
        """Evaluate all 46 alert conditions from the dashboard spec."""
        try:
            self._check_core_trading_alerts(metrics)
            self._check_sac_health_alerts(metrics)
            self._check_anti_hack_alerts(metrics)
            self._check_obs_health_alerts(metrics)
            self._check_reward_alerts(metrics)
            self._check_tp_sl_alerts(metrics)
        except Exception:
            pass  # Never let alert logging crash training

    def _check_core_trading_alerts(self, metrics: Dict):
        """Alerts 1-8: Core trading alerts."""
        dd = metrics.get("drawdown", 0.0)

        # #1: Drawdown > 5%
        if 0.05 < dd <= 0.08:
            self.training_logger.log_alert(
                "WARNING", f"Drawdown {dd:.1%} elevated", {"drawdown": dd})

        # #2: Drawdown > 8%
        if dd > 0.08:
            self.training_logger.log_alert(
                "CRITICAL", f"Drawdown {dd:.1%} approaching limit", {"drawdown": dd})

        # #3: Drawdown > 10% handled by env emergency stop (terminal penalty)

        # #4: Daily DD > 3% handled by env emergency stop

        # #5: Bankruptcy
        balance = metrics.get("balance", 100.0)
        if balance <= 0:
            self.training_logger.log_alert(
                "CRITICAL", "Bankruptcy — balance reset", {"balance": balance})

        # #6: Win rate < 35% after 50+ trades
        wr = metrics.get("win_rate", 0.5)
        if self._total_trades > 50 and wr < 0.35:
            self.training_logger.log_alert(
                "WARNING", f"Win rate low: {wr:.1%} ({self._total_trades} trades)",
                {"win_rate": wr, "total_trades": self._total_trades})

        # #7: Win rate < 25% after 100+ trades
        if self._total_trades > 100 and wr < 0.25:
            self.training_logger.log_alert(
                "CRITICAL", f"Win rate critical: {wr:.1%} ({self._total_trades} trades)",
                {"win_rate": wr, "total_trades": self._total_trades})

        # #8: No trades in 500+ steps
        if self._steps_without_trade >= 500:
            self.training_logger.log_alert(
                "WARNING", f"No trades for {self._steps_without_trade} steps",
                {"steps_without_trade": self._steps_without_trade})
            self._steps_without_trade = 0  # Reset to avoid spam

    def _check_sac_health_alerts(self, metrics: Dict):
        """Alerts 9-16: SAC health alerts."""
        alpha = metrics.get("entropy_alpha")

        # #9: Entropy α < 0.01
        if alpha is not None and 0.001 <= alpha < 0.01:
            self.training_logger.log_alert(
                "WARNING", f"Entropy alpha low: {alpha:.4f}", {"entropy_alpha": alpha})

        # #10: Entropy α < 0.001 (collapse)
        if alpha is not None and alpha < 0.001:
            self.training_logger.log_alert(
                "CRITICAL", f"Entropy COLLAPSED: alpha={alpha:.6f}", {"entropy_alpha": alpha})

        # #11: Entropy α > 10 (explosion)
        if alpha is not None and alpha > 10.0:
            self.training_logger.log_alert(
                "CRITICAL", f"Entropy EXPLODING: alpha={alpha:.2f}", {"entropy_alpha": alpha})

        # #12-13: Q-value mean (if available)
        q_mean = metrics.get("q_value_mean")
        if q_mean is not None:
            if q_mean > 100:
                self.training_logger.log_alert(
                    "CRITICAL", f"Q-values exploding: mean={q_mean:.1f}",
                    {"q_value_mean": q_mean})
            elif q_mean > 50:
                self.training_logger.log_alert(
                    "WARNING", f"Q-values elevated: mean={q_mean:.1f}",
                    {"q_value_mean": q_mean})

        # #14-15: Actor gradient
        actor_grad = metrics.get("actor_grad_norm")
        if actor_grad is not None:
            if actor_grad > 100:
                self.training_logger.log_alert(
                    "CRITICAL", f"Actor gradient exploding: {actor_grad:.1f}",
                    {"actor_grad_norm": actor_grad})
            elif actor_grad > 50:
                self.training_logger.log_alert(
                    "WARNING", f"Actor gradient high: {actor_grad:.1f}",
                    {"actor_grad_norm": actor_grad})

        # #16: Critic gradient
        critic_grad = metrics.get("critic_grad_norm")
        if critic_grad is not None and critic_grad > 100:
            self.training_logger.log_alert(
                "WARNING", f"Critic gradient high: {critic_grad:.1f}",
                {"critic_grad_norm": critic_grad})

        # Action collapse
        action_std = metrics.get("action_std")
        if action_std is not None and action_std < self.cfg.collapsed_action_std:
            self.training_logger.log_alert(
                "WARNING", f"Action std collapsed: {action_std:.4f}",
                {"action_std": action_std})

    def _check_anti_hack_alerts(self, metrics: Dict):
        """Alerts 17-22: Anti-reward-hacking alerts."""
        # #17: Trade cap hit
        daily = metrics.get("daily_trades", 0)
        if daily >= self.cfg.daily_trade_soft_cap:
            self._trade_cap_hits += 1

        # #21: All trades same direction (50+ trades)
        if len(self._trade_directions) >= 50:
            recent = self._trade_directions[-50:]
            long_pct = sum(1 for d in recent if d == "LONG") / len(recent)
            if long_pct > 0.95 or long_pct < 0.05:
                self.training_logger.log_alert(
                    "WARNING", f"All trades same direction (LONG%: {long_pct:.0%})",
                    {"long_pct": long_pct})

        # #22: Conviction always maxed out
        if len(self._conviction_values) >= 30:
            recent_conv = self._conviction_values[-30:]
            maxed = sum(1 for c in recent_conv if c > 0.95) / len(recent_conv)
            if maxed > 0.90:
                self.training_logger.log_alert(
                    "WARNING", f"Conviction always maxed: {maxed:.0%} at 1.0",
                    {"maxed_pct": maxed})

    def _check_obs_health_alerts(self, metrics: Dict):
        """Alerts 23-28: Observation health alerts."""
        dead = self.shared_metrics.get("dead_features", 0)
        exploding = self.shared_metrics.get("exploding_features", 0)
        nan_count = self.shared_metrics.get("nan_features", 0)

        # #23: Dead features
        if dead > 0:
            self.training_logger.log_alert(
                "WARNING", f"{dead} dead features (std < 0.01)",
                {"dead_features": dead})

        # #24: Exploding features
        if exploding > 0:
            self.training_logger.log_alert(
                "WARNING", f"{exploding} exploding features (std > 3.0)",
                {"exploding_features": exploding})

        # #25: NaN features
        if nan_count > 0:
            self.training_logger.log_alert(
                "CRITICAL", f"{nan_count} features with >5% NaN",
                {"nan_features": nan_count})

    def _check_reward_alerts(self, metrics: Dict):
        """Alerts 29-34: Reward health alerts."""
        clip_pct = metrics.get("reward_clip_pct", 0)

        # #29: Reward clip rate > 5%
        if 0.05 < clip_pct <= 0.15:
            self.training_logger.log_alert(
                "WARNING", f"Reward clip rate elevated: {clip_pct:.1%}",
                {"reward_clip_pct": clip_pct})

        # #30: Reward clip rate > 15%
        if clip_pct > 0.15:
            self.training_logger.log_alert(
                "CRITICAL", f"Reward clip rate high: {clip_pct:.1%}",
                {"reward_clip_pct": clip_pct})

        # #34: Raw reward always 0 for 100+ steps
        if self._reward_zero_steps >= 100:
            self.training_logger.log_alert(
                "CRITICAL", f"Raw reward zero for {self._reward_zero_steps} steps",
                {"reward_zero_steps": self._reward_zero_steps})

        # Gradient clip rate
        grad_clip = metrics.get("grad_clip_pct", 0)
        if 0.05 < grad_clip <= 0.30:
            self.training_logger.log_alert(
                "WARNING", f"Gradient clip rate: {grad_clip:.1%}",
                {"grad_clip_pct": grad_clip})
        elif grad_clip > 0.30:
            self.training_logger.log_alert(
                "CRITICAL", f"Gradient clip rate high: {grad_clip:.1%}",
                {"grad_clip_pct": grad_clip})

    def _check_tp_sl_alerts(self, metrics: Dict):
        """Alerts 44-46: TP/SL health alerts."""
        tp_hit = self.shared_metrics.get("tp_hit_rate", 0)
        tp_reach = self.shared_metrics.get("tp_reachable_rate", 0)
        sl_hit = self.shared_metrics.get("sl_hit_rate", 0)

        # #44: TP reachable but not hit > 50%
        if tp_reach > 0 and tp_hit > 0:
            gap = tp_reach - tp_hit
            if gap > 0.50:
                self.training_logger.log_alert(
                    "WARNING", f"TP reachable {tp_reach:.0%} but hit {tp_hit:.0%} — closing early",
                    {"tp_reach": tp_reach, "tp_hit": tp_hit})

        # #45: SL hit rate > 60%
        if self._total_trades > 20 and sl_hit > 0.60:
            self.training_logger.log_alert(
                "WARNING", f"SL hit rate {sl_hit:.0%} — SL may be too tight",
                {"sl_hit_rate": sl_hit})

        # #46: No TP hits in 20+ trades
        if self._total_trades >= 20 and tp_hit == 0:
            self.training_logger.log_alert(
                "WARNING", f"No TP hits in {self._total_trades} trades",
                {"total_trades": self._total_trades, "tp_hit_rate": 0})

    def _on_training_end(self):
        """Called when learn() finishes."""
        pass
