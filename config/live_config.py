"""LiveConfig dataclass for the Spartus Live Dashboard.

Centralises every runtime parameter. Fields carry sensible defaults
that match the training spec (v3.3.2, 67 features / 670 obs_dim).

Load user overrides from YAML via:
    cfg = LiveConfig.from_yaml("config/default_config.yaml")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def _default_symbol_map() -> Dict[str, str]:
    """Dukascopy symbol -> MT5 symbol mapping for correlated instruments."""
    return {
        "EURUSD": "EURUSD",
        "XAGUSD": "XAGUSD",
        "USDJPY": "USDJPY",
        "US500": "US500",
        "USOIL": "USOIL",
    }


def _default_market_feature_names() -> List[str]:
    """38 features that receive rolling z-score normalisation."""
    return [
        # --- Groups A-E: Original 25 market features ---
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
        # --- Upgrade 1: Correlated Instruments (11) ---
        "eurusd_returns_20", "eurusd_rsi_14", "eurusd_trend",
        "xagusd_returns_20", "xagusd_rsi_14",
        "usdjpy_returns_20", "usdjpy_trend",
        "us500_returns_20", "us500_rsi_14",
        "usoil_returns_20",
        "gold_silver_ratio_z",
        # --- Upgrade 4: Regime Detection (2) ---
        "corr_gold_usd_100", "corr_gold_spx_100",
    ]


def _default_norm_exempt_features() -> List[str]:
    """29 features that pass through without z-score normalisation.

    These are already bounded, binary, or computed live per step.
    """
    return [
        # --- Group F: Time & Session (4, precomputed) ---
        "hour_sin", "hour_cos", "day_of_week", "session_quality",
        # --- Groups G-H: Account & Memory (13, live) ---
        # Group G: Account state (8)
        "has_position", "position_side", "unrealized_pnl", "position_duration",
        "current_drawdown", "equity_ratio", "sl_distance_ratio", "profit_locked_pct",
        # Group H: Memory features (5)
        "recent_win_rate", "similar_pattern_winrate",
        "trend_prediction_accuracy", "tp_hit_rate", "avg_sl_trail_profit",
        # --- Upgrade 2: Calendar & Events (6) ---
        "hours_to_next_high_impact", "hours_to_next_nfp_fomc",
        "in_event_window", "daily_event_density",
        "london_fix_proximity", "comex_session_active",
        # --- Upgrade 3: Spread & Liquidity (2) ---
        "spread_estimate_norm", "volume_spike",
        # --- Upgrade 5: Session Microstructure (4) ---
        "asian_range_norm", "asian_range_position",
        "session_momentum", "london_ny_overlap",
    ]


@dataclass
class LiveConfig:
    """Complete configuration for the Spartus Live Dashboard.

    Every module in the live system imports from here.  Defaults match
    the training spec so a bare ``LiveConfig()`` is immediately usable.
    """

    # ---- Model -----------------------------------------------------------
    # Leave empty to auto-discover the first .zip in model/ directory.
    # Set explicitly only if you have multiple models and want a specific one.
    model_path: str = ""

    # ---- MT5 -------------------------------------------------------------
    mt5_symbol: str = "XAUUSD"
    mt5_terminal_path: str = ""

    # ---- Symbol mapping (Dukascopy -> MT5 broker names) ------------------
    symbol_map: Dict[str, str] = field(default_factory=_default_symbol_map)

    # ---- Risk ------------------------------------------------------------
    max_risk_pct: float = 0.02
    max_dd: float = 0.10
    max_daily_dd: float = 0.03
    daily_trade_hard_cap: int = 20
    min_hold_bars: int = 3
    direction_threshold: float = 0.3
    min_conviction: float = 0.15          # Separate conviction floor (model output compressed)
    exit_threshold: float = 0.5
    allow_min_lot_override: bool = False  # Disabled: min-lot must still pass hard risk cap

    # ---- Circuit breakers ------------------------------------------------
    consecutive_loss_pause: int = 5
    consecutive_loss_pause_minutes: int = 15
    severe_loss_pause: int = 8
    severe_loss_pause_minutes: int = 60
    daily_dd_halt_pct: float = 0.05
    daily_dd_close_all_pct: float = 0.08
    weekly_dd_halt_pct: float = 0.10

    # ---- Weekend ---------------------------------------------------------
    friday_close_utc_hour: int = 20
    friday_block_new_utc_hour: int = 19
    monday_resume_utc_hour: int = 0
    monday_resume_utc_minute: int = 30

    # ---- Feature pipeline ------------------------------------------------
    warmup_bars: int = 500
    norm_window: int = 200
    norm_clip: float = 5.0
    frame_stack: int = 10

    # ---- Calendar --------------------------------------------------------
    # MQL5 bridge: CalendarBridge.mq5 writes to MT5 Common Files folder.
    # Auto-detected at runtime from mt5.terminal_info().commondata_path.
    # Set to "" to auto-detect, or an absolute path to override.
    calendar_bridge_path: str = ""
    calendar_csv_path: str = "data/calendar/economic_calendar.csv"
    calendar_static_path: str = "data/calendar/known_events.json"
    calendar_bridge_max_age_s: int = 7200

    # ---- Paper trading ---------------------------------------------------
    paper_trading: bool = True

    # ---- Observation period (optional, off by default) -------------------
    observation_period_enabled: bool = False
    observation_period_days: int = 14
    observation_lot_cap: float = 0.01

    # ---- Post-rounding risk cap (always on) --------------------------------
    enable_post_rounding_risk_cap: bool = True
    post_rounding_risk_cap: float = 1.5
    absolute_risk_cap_pct: float = 0.03

    # ---- Broker constraints / spread gate -----------------------------------
    spread_hard_max_points: int = 50       # Block entry if spread > N points
    spread_spike_multiplier: float = 2.5   # Block if spread > EMA * multiplier
    min_sl_buffer_points: int = 5          # Buffer above stops_level+spread for SL
    broker_heavy_refresh_s: int = 600      # Full symbol_info refresh (10 min)
    broker_light_refresh_s: int = 10       # Spread/tick refresh (10s)

    # ---- Daily reset -----------------------------------------------------
    daily_reset_utc_hour: int = 0

    # ---- Logging ---------------------------------------------------------
    log_every_action: bool = True
    log_observations_every_n_bars: int = 12
    log_feature_stats: bool = True

    # ---- Analytics -------------------------------------------------------
    feature_drift_threshold_sigma: float = 2.0
    corr_drift_yellow_threshold: float = 0.15
    corr_drift_red_threshold: float = 0.25
    corr_drift_yellow_persist: int = 3
    corr_drift_red_persist: int = 2
    action_flat_rate_warn: float = 0.90
    action_std_collapse_warn: float = 0.10
    weekly_report_auto: bool = True
    retrain_trigger_pf_weeks: int = 3

    # ---- Normalizer persistence ------------------------------------------
    normalizer_state_path: str = "storage/state/normalizer_state.json"
    normalizer_backup_interval_s: int = 3600
    normalizer_max_age_s: int = 3600

    # ---- Feature counts (fixed, from training) ---------------------------
    n_features: int = 67
    n_market_features: int = 38
    n_exempt_features: int = 29
    n_precomputed: int = 54
    n_live: int = 13
    obs_dim: int = 670

    # ---- Feature name lists ----------------------------------------------
    market_feature_names: List[str] = field(
        default_factory=_default_market_feature_names,
    )
    norm_exempt_features: List[str] = field(
        default_factory=_default_norm_exempt_features,
    )

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def get_base_dir(cls) -> Path:
        """Return the absolute path to the live_dashboard directory."""
        return Path(__file__).resolve().parent.parent

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "LiveConfig":
        """Load a LiveConfig from a YAML file, merging with defaults.

        Any key present in the YAML overrides the dataclass default.
        Keys not in the YAML keep their default value.

        Args:
            path: Path to the YAML file.  If *None* or the file does
                  not exist, returns a pure-default LiveConfig.

        Returns:
            A fully populated LiveConfig instance.
        """
        if path is None:
            path = str(cls.get_base_dir() / "config" / "default_config.yaml")

        yaml_path = Path(path)
        if not yaml_path.exists():
            return cls()

        with open(yaml_path, "r", encoding="utf-8") as fh:
            raw: dict = yaml.safe_load(fh) or {}

        # Collect all dataclass field names so we can distinguish between
        # a field that happens to be a dict (e.g. symbol_map) and a YAML
        # grouping section (e.g. risk, circuit_breakers) that should be
        # flattened into individual kwargs.
        from dataclasses import fields as dc_fields
        field_names = {f.name for f in dc_fields(cls)}

        kwargs: Dict = {}
        for key, value in raw.items():
            if key in field_names:
                # Exact field name -- pass the value as-is (dict, list, scalar).
                kwargs[key] = value
            elif isinstance(value, dict):
                # Grouping section -- flatten sub-keys into kwargs.
                for sub_key, sub_value in value.items():
                    kwargs[sub_key] = sub_value
            else:
                kwargs[key] = value

        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def resolve_path(self, relative: str) -> Path:
        """Resolve a config-relative path against get_base_dir()."""
        p = Path(relative)
        if p.is_absolute():
            return p
        return self.get_base_dir() / p

    def to_dict(self) -> dict:
        """Serialise all fields to a JSON-safe dictionary."""
        from dataclasses import fields as dc_fields

        d: Dict = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if isinstance(val, Path):
                d[f.name] = str(val)
            elif isinstance(val, (tuple, set)):
                d[f.name] = list(val)
            else:
                d[f.name] = val
        return d
