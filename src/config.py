"""Central configuration for the Spartus Trading AI system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class TrainingConfig:
    """All system parameters in one place. Every module imports from here."""

    # === Paths ===
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "storage" / "data" / "processed")
    feature_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "storage" / "features")
    model_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "storage" / "models")
    memory_db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "storage" / "memory" / "spartus_memory.db")
    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "storage" / "logs")
    report_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "storage" / "reports")
    tensorboard_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "storage" / "logs" / "tensorboard")
    training_state_path: Path = field(default_factory=lambda: PROJECT_ROOT / "storage" / "training_state.json")
    best_model_path: Path = field(default_factory=lambda: PROJECT_ROOT / "storage" / "models" / "spartus_best.zip")

    # === Feature Engineering ===
    num_features: int = 67
    frame_stack: int = 10
    lookback: int = 200          # Warmup bars for indicators
    frac_diff_d: float = 0.35    # Fractional differentiation order

    # Features that are NOT normalized (already bounded)
    norm_exempt_features: tuple = (
        # Group F: Time & Session (original)
        "hour_sin", "hour_cos", "day_of_week", "session_quality",
        # Groups G-H: Account & Memory (live, original)
        "has_position", "position_side", "unrealized_pnl", "position_duration",
        "current_drawdown", "equity_ratio", "sl_distance_ratio", "profit_locked_pct",
        "recent_win_rate", "similar_pattern_winrate",
        "trend_prediction_accuracy", "tp_hit_rate", "avg_sl_trail_profit",
        # Upgrade 2: Calendar & Events (bounded/binary)
        "hours_to_next_high_impact", "hours_to_next_nfp_fomc",
        "in_event_window", "daily_event_density",
        "london_fix_proximity", "comex_session_active",
        # Upgrade 3: Spread & Liquidity (already ATR-normalized or capped)
        "spread_estimate_norm", "volume_spike",
        # Upgrade 5: Session Microstructure (already ATR-normalized/clipped/binary)
        "asian_range_norm", "asian_range_position",
        "session_momentum", "london_ny_overlap",
    )

    # Market features to normalize via rolling z-score (#1-25 original + 13 new)
    market_feature_names: tuple = (
        # Groups A-E: Original 25 market features
        "close_frac_diff", "returns_1bar", "returns_5bar", "returns_20bar",
        "bar_range", "close_position", "body_ratio",
        "atr_14_norm", "atr_ratio", "bb_width", "bb_position",
        "rsi_14", "macd_signal", "adx_14", "ema_cross", "price_vs_ema200", "stoch_k",
        "volume_ratio", "obv_slope",
        "h1_trend_dir", "h4_trend_dir", "d1_trend_dir",
        "h1_rsi", "mtf_alignment", "htf_momentum",
        # Upgrade 1: Correlated Instruments (11 features)
        "eurusd_returns_20", "eurusd_rsi_14", "eurusd_trend",
        "xagusd_returns_20", "xagusd_rsi_14",
        "usdjpy_returns_20", "usdjpy_trend",
        "us500_returns_20", "us500_rsi_14",
        "usoil_returns_20",
        "gold_silver_ratio_z",
        # Upgrade 4: Regime Detection (2 features)
        "corr_gold_usd_100", "corr_gold_spx_100",
    )

    # === Correlated Instruments (Upgrade 1) ===
    correlated_symbols: tuple = ("EURUSD", "XAGUSD", "USDJPY", "US500", "USOIL")
    correlated_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "storage" / "data" / "correlated")
    # Broker symbol overrides (canonical → broker name).
    # Empty = canonical names used as-is (works for Vantage standard symbols).
    # Example: {"US500": "SP500", "USOIL": "OIL.WTI"}
    symbol_map: dict = field(default_factory=dict)

    # === Economic Calendar (Upgrade 2) ===
    calendar_csv_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "calendar" / "economic_calendar.csv")

    # === Regime Detection (Upgrade 4) ===
    regime_corr_window: int = 100   # Rolling correlation window (bars)

    # === Session Microstructure (Upgrade 5) ===
    asian_session_end_utc: int = 7  # UTC hour when Asian range completes
    session_boundaries_utc: tuple = (0, 7, 12)  # Session open times in UTC

    norm_window: int = 200       # Rolling z-score window
    norm_clip: float = 5.0       # Clip normalized values to [-5, +5]

    # === SAC Hyperparameters ===
    learning_rate: float = 3e-4
    buffer_size: int = 500_000       # Increased from 200K: holds ~12 weeks + core retention
    batch_size: int = 1024
    gamma: float = 0.97
    tau: float = 0.002   # FIX-CRITIC: Reduced from 0.005 — slower target network tracking prevents diverging critic from corrupting stable target
    ent_coef: str = "auto"
    target_entropy: float = -1.0    # Higher = more exploration; "auto" default would be -4.0 (action dims)
    min_entropy_alpha: float = 0.05  # Floor: raised from 0.01 to prevent near-deterministic policy
    learning_starts: int = 5_000
    train_freq: int = 1
    gradient_steps: int = 4          # Reduced from 8: halves overfitting pressure per env step
    max_grad_norm: float = 1.0
    weight_decay: float = 1e-4       # L2 regularization (AdamW) — prevents weight explosion / memorization
    net_arch_pi: list = field(default_factory=lambda: [256, 256])
    net_arch_qf: list = field(default_factory=lambda: [256, 256])

    # LR schedule phases
    lr_warmup_end: float = 0.05   # First 5%: warm-up
    lr_decay_start: float = 0.75  # After 75%: cosine decay
    lr_min: float = 5e-5
    lr_warmup_start: float = 1e-5

    # === Trading Environment ===
    initial_balance: float = 10_000.0     # £10K per episode — matches val and realistic live scale
    val_initial_balance: float = 10_000.0  # Validation balance — same scale as training
    steps_per_week: int = 10_000    # Timesteps per week for agent.learn()
    n_envs: int = 4                 # Parallel environments via DummyVecEnv

    # === Risk Parameters ===
    max_risk_pct: float = 0.02      # Max 2% risk per trade
    max_dd: float = 0.10            # 10% total DD → end episode
    max_daily_dd: float = 0.03      # 3% daily DD → force close
    min_sl_atr: float = 1.0         # Minimum SL distance: 1.0 ATR
    min_sl_trail_atr: float = 0.5   # Minimum trailing SL distance: 0.5 ATR
    max_positions: int = 1          # Phase 1: single position

    # === Profit Protection (Rule-Based Staged SL) ===
    protection_be_trigger_r: float = 1.0       # Stage 1: move SL to breakeven at +1.0R
    protection_be_buffer_pips: float = 0.5     # Buffer above entry for BE (cover spread)
    protection_lock_trigger_r: float = 1.5     # Stage 2: lock profit at +1.5R
    protection_lock_amount_r: float = 0.5      # Lock +0.5R guaranteed
    protection_trail_trigger_r: float = 2.0    # Stage 3: activate ATR trail at +2.0R
    protection_trail_atr_mult: float = 1.0     # Trail distance = 1.0 * ATR

    # === Re-Entry Penalty ===
    reentry_penalty_bars: int = 6              # Window: 30 min on M5
    reentry_penalty_mult: float = 1.5          # 50% amplified loss signal on R1
    reentry_win_discount: float = 0.7          # FIX-11: Discount winning re-entries too

    # === Asymmetric Loss Penalty ===
    loss_penalty_mult: float = 1.2             # Reduced from 1.5: less punishing to prevent "don't trade" convergence

    # === Anti-Reward-Hacking ===
    min_hold_bars: int = 6          # No close before 6 bars (30 min)
    daily_trade_soft_cap: int = 10
    daily_trade_hard_cap: int = 20    # Absolute max — no exceptions, even with high conviction
    normal_conviction_threshold: float = 0.3
    elevated_conviction_threshold: float = 0.6
    direction_threshold: float = 0.3  # |direction| > 0.3 to open
    exit_threshold: float = 0.6       # exit_urgency > 0.6 to close (FIX-9: raised from 0.5)
    min_combined_signal: float = 0.15 # |direction| * conviction floor (FIX-3)

    # === Reward Weights ===
    r1_weight: float = 0.40  # Position P/L
    r2_weight: float = 0.20  # Trade completion quality
    r3_weight: float = 0.15  # Drawdown penalty
    r4_weight: float = 0.15  # Differential Sharpe
    r5_weight: float = 0.05  # Risk-adjusted bonus (FIX-2: halved from 0.10)
    r5_flat_penalty: float = -0.02  # Opportunity cost when flat (FIX-2: reduced from -0.05)
    r5_conviction_scaling: bool = True  # FIX-1: Scale R5 by entry conviction
    reward_normalizer_tau: float = 0.001
    reward_clip: float = 5.0
    sharpe_eta: float = 0.01  # Differential Sharpe EMA rate

    # === Market Simulation ===
    spread_london_pips: float = 1.5
    spread_ny_pips: float = 2.0
    spread_asia_pips: float = 3.0
    spread_off_hours_pips: float = 5.0
    spread_news_multiplier: float = 3.0
    pip_price: float = 0.10           # 1 pip = $0.10 for XAUUSD
    slippage_mean_pips: float = 0.5
    slippage_std_pips: float = 0.3
    commission_per_lot: float = 0.0   # Vantage: spread-only, $0 commission (verified via MT5 API)

    # Domain randomization (per episode)
    spread_jitter: float = 0.30       # ±30%
    slippage_jitter: float = 0.50     # ±50%
    commission_jitter: float = 0.20   # ±20%
    observation_noise_std: float = 0.02
    start_offset_max: int = 500       # Random skip 0-500 bars (~1.7 days) for session diversity

    # === Account & Symbol Info ===
    # Account currency determines P/L conversion. XAUUSD raw P/L is in USD.
    # trade_tick_value converts it to account currency automatically.
    #   USD account: tick_value = 1.00 (no conversion needed)
    #   GBP account: tick_value = 0.745 (from MT5 symbol_info, fluctuates with GBP/USD)
    #   EUR account: tick_value = ~0.92 (pull from MT5 symbol_info)
    # For live: pull real-time tick_value from mt5.symbol_info("XAUUSD").trade_tick_value
    # SET TO MATCH YOUR BROKER ACCOUNT CURRENCY for realistic lot sizing & P/L.
    account_currency: str = "GBP"     # Vantage International — GBP account
    trade_tick_value: float = 0.745   # GBP per tick per 1.0 lot (from live MT5, avg of Week 1)
    trade_tick_size: float = 0.01     # Minimum price increment (XAUUSD)
    trade_contract_size: float = 100  # 1 lot = 100 troy oz (reference only — P/L uses tick formula)
    volume_min: float = 0.01
    volume_max: float = 5.0             # Safety cap — realistic for £10K account
    volume_step: float = 0.01
    point: float = 0.01

    # === Curriculum ===
    stage1_end_week: int = 30    # Easy mode
    stage2_end_week: int = 80    # Normal mode
    total_training_weeks: int = 700  # Must exceed train split count (~497 weeks at 85% of 585). Stage 3 cycles chronologically through all train weeks.

    # === Training Control ===
    validation_interval: int = 10    # Validate every 10 weeks (doubled frequency for earlier overfitting detection)
    max_val_weeks: int = 0           # 0 = use ALL val weeks (no cap). Was 30, causing val subset overfitting
    checkpoint_interval: int = 1    # Save every N weeks
    report_interval: int = 10       # Generate report every N weeks

    # === Data Split ===
    train_split: float = 0.85       # 85% train (2015-2024)
    val_split: float = 0.10         # 10% validation (2025)
    test_split: float = 0.05        # 5% test (2026) — holdout, never trained or validated on
    purge_weeks: int = 2            # Gap between splits to prevent leakage

    # === Convergence Detection ===
    convergence_window: int = 50
    convergence_sharpe_threshold: float = 0.001
    convergence_weeks_since_best: int = 5    # FIX-B3: was 50 (val points, not weeks) = 500 training weeks; 5 val points = 50 weeks
    collapsed_action_std: float = 0.05
    collapsed_duration: int = 20
    overfitting_patience: int = 60    # Reduced from 300: revert to best after 60 weeks of overfitting
    collapsed_auto_stop: bool = True  # Halt training on policy collapse

    # === Graduated Overfitting Defense (4-layer system) ===
    # Layer 1: Prevention (always active)
    critic_dropout: float = 0.10     # Dropout probability in critic Q-networks (ICLR 2024)
    use_layer_norm: bool = True      # LayerNorm in actor/critic (stabilizes regime shifts)

    # Layer 2: Soft correction (automatic, no rollback)
    soft_correction_weeks: int = 10   # Trigger after N consecutive OVERFITTING weeks
    entropy_boost_mult: float = 1.5   # Multiply entropy alpha by this on soft correction
    weight_decay_boost_mult: float = 2.0  # Multiply weight_decay by this on soft correction

    # Layer 3: Rollback with modification (replaces old bare rollback)
    rollback_trigger_weeks: int = 30  # Trigger after N consecutive OVERFITTING weeks
    rollback_entropy_mult: float = 2.0  # Entropy boost after rollback
    rollback_buffer_clear_pct: float = 0.30  # Clear newest X% of replay buffer after rollback

    # Layer 4: Hard reset (second rollback within cooldown window)
    hard_reset_cooldown: int = 100    # If second rollback within N weeks → hard reset
    hard_reset_curriculum_weeks: int = 20  # Force Stage 2 (mixed) for N weeks after hard reset

    # PLATEAU escape: rollback if stuck in PLATEAU with LR maxed (independent of overfitting_weeks)
    plateau_rollback_weeks: int = 40  # Stagnation weeks threshold before forcing rollback from PLATEAU

    # === Replay Buffer Retention ===
    buffer_core_retention_pct: float = 0.20  # Keep 20% of best experiences when cycling weeks in Stage 3

    # === Live Fine-Tuning ===
    # Layer 1: Slow Adaptation
    finetune_lr: float = 1e-4                    # Reduced LR (3x slower than training)
    finetune_kl_penalty_weight: float = 0.1      # KL penalty weight in SAC loss
    finetune_max_kl_divergence: float = 0.5      # Pause if KL exceeds this
    finetune_kl_emergency_threshold: float = 0.3 # Reduce LR to emergency level at this KL
    finetune_lr_emergency: float = 3e-5          # Emergency LR when KL > 0.3

    # Layer 2: Curated Replay Buffer
    finetune_buffer_size: int = 200_000          # Total buffer capacity
    finetune_buffer_core_pct: float = 0.30       # Core memories (never evicted)
    finetune_buffer_supporting_pct: float = 0.40 # Supporting memories (slow eviction)
    finetune_buffer_recent_pct: float = 0.30     # Live/recent data (fast turnover)
    finetune_min_reward_core: float = 1.0        # Min reward to qualify as core memory

    # Layer 3: EWC Weight Protection
    finetune_ewc_enabled: bool = True
    finetune_ewc_lambda: float = 5000.0          # EWC penalty strength
    finetune_ewc_fisher_samples: int = 2000      # Samples for Fisher matrix computation

    # Layer 4: Strategy Memory
    finetune_strategy_memory_enabled: bool = True
    finetune_regime_clusters: int = 8            # Number of market regime buckets
    finetune_forgetting_threshold: float = 0.70  # Alert if regime perf < 70% of historical

    # Data Collection
    finetune_min_bars: int = 500                 # Min bars before first training episode
    finetune_episode_interval_hours: float = 4.0 # Hours between training episodes
    finetune_bar_buffer_size: int = 2000         # Rolling bar buffer (~1 week M5)

    # Validation & Checkpoints
    finetune_gradient_steps: int = 8            # Gradient steps per episode (same as training)
    finetune_batch_size: int = 1024             # Batch size (same as training)
    finetune_checkpoint_interval: int = 5       # Save every N episodes
    finetune_max_checkpoints: int = 10          # Keep last 10 fine-tune checkpoints
    finetune_val_sharpe_threshold: float = 0.90 # Must be >= 90% of baseline Sharpe
    finetune_auto_rollback_failures: int = 3    # Rollback after N consecutive val failures
    finetune_max_drawdown_mult: float = 2.0     # Stop if drawdown > 2x baseline max DD
    finetune_action_std_min: float = 0.30       # Pause if action_std drops below this

    # Fine-tune storage paths
    finetune_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "storage" / "finetune")

    @property
    def obs_dim(self) -> int:
        return self.num_features * self.frame_stack  # 670

    @property
    def value_per_point_per_lot(self) -> float:
        """Account-currency value of 1 price point move on 1.0 lot.

        This is MT5's exact conversion: tick_value / tick_size.
        USD account: 1.00 / 0.01 = 100.0
        GBP account: 0.7412 / 0.01 = 74.12
        Used everywhere P/L is calculated — replaces hardcoded contract_size.
        """
        return self.trade_tick_value / self.trade_tick_size

    @property
    def symbol_info(self) -> Dict:
        return {
            "trade_tick_value": self.trade_tick_value,
            "trade_tick_size": self.trade_tick_size,
            "trade_contract_size": self.trade_contract_size,
            "volume_min": self.volume_min,
            "volume_max": self.volume_max,
            "volume_step": self.volume_step,
            "point": self.point,
        }

    def to_dict(self) -> dict:
        """Serialize all config fields to a JSON-safe dictionary.

        Converts Path -> str, tuple -> list. Includes computed properties.
        """
        from dataclasses import fields as dc_fields

        d = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if isinstance(val, Path):
                d[f.name] = str(val)
            elif isinstance(val, tuple):
                d[f.name] = list(val)
            else:
                d[f.name] = val
        # Computed properties (not in dataclasses.fields)
        d["obs_dim"] = self.obs_dim
        d["value_per_point_per_lot"] = self.value_per_point_per_lot
        d["symbol_info"] = self.symbol_info
        return d
