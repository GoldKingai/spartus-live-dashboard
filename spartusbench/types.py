"""Dataclasses and type definitions for SpartusBench."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TradeRecord:
    """Single trade from a benchmark rollout."""
    trade_num: int
    week: int
    step: int
    side: str               # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    lots: float
    pnl: float
    pnl_pct: float
    hold_bars: int
    conviction: float
    close_reason: str       # TP_HIT, SL_HIT, AGENT_CLOSE, EMERGENCY_STOP, TIMEOUT
    lesson_type: str        # from TradeAnalyzer classification
    session: str            # London, NY_Overlap, NY_PM, Asia, Off_hours
    atr_at_entry: float
    max_favorable: float    # max favorable excursion
    initial_sl: float
    initial_tp: float
    final_sl: float
    scenario: str = "base"


@dataclass
class StressResult:
    """Results for a single stress scenario."""
    scenario: str
    trades: int = 0
    win_pct: float = 0.0
    net_pnl: float = 0.0
    pf: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_dd_pct: float = 0.0
    tim_pct: float = 0.0
    avg_hold: float = 0.0
    trades_per_day: float = 0.0
    long_count: int = 0
    short_count: int = 0
    long_pnl: float = 0.0
    short_pnl: float = 0.0
    pf_retention: float = 1.0
    sharpe_retention: float = 1.0
    trade_list: List[TradeRecord] = field(default_factory=list)


@dataclass
class RegimeSlice:
    """Results for a single regime slice (ATR quartile, session, day, etc.)."""
    slice_type: str         # "atr_quartile", "session", "day", "direction", "close_reason"
    slice_value: str        # "Q1", "London", "Monday", "LONG", "TP_HIT"
    trades: int = 0
    win_pct: float = 0.0
    net_pnl: float = 0.0
    pf: float = 0.0
    avg_pnl: float = 0.0
    avg_hold: float = 0.0


@dataclass
class ChurnResult:
    """Churn diagnostic results (T4)."""
    trading_days: int = 0
    trades_per_day: float = 0.0
    avg_spread_pips: float = 0.0
    avg_slippage_pips: float = 0.0
    avg_cost_pips: float = 0.0
    avg_cost_points: float = 0.0
    avg_lot: float = 0.0
    est_cost_per_trade: float = 0.0
    total_est_cost: float = 0.0
    net_pnl: float = 0.0
    est_gross_pnl: float = 0.0
    net_edge_per_trade: float = 0.0
    gross_edge_per_trade: float = 0.0
    cost_to_edge_ratio: float = 0.0


@dataclass
class RewardAblation:
    """Reward ablation results (T5)."""
    r1_sum: float = 0.0
    r2_sum: float = 0.0
    r3_sum: float = 0.0
    r4_sum: float = 0.0
    r5_sum: float = 0.0
    r1_weighted: float = 0.0
    r2_weighted: float = 0.0
    r3_weighted: float = 0.0
    r4_weighted: float = 0.0
    r5_weighted: float = 0.0
    total_weighted: float = 0.0
    r1_pct: float = 0.0
    r2_pct: float = 0.0
    r3_pct: float = 0.0
    r4_pct: float = 0.0
    r5_pct: float = 0.0
    r5_positive_steps: int = 0
    r5_negative_steps: int = 0
    r5_zero_steps: int = 0


@dataclass
class GatingResult:
    """Gating diagnostics results (T6)."""
    total_bars: int = 0
    direction_pass_count: int = 0
    direction_pass_pct: float = 0.0
    conviction_pass_live_count: int = 0
    conviction_pass_live_pct: float = 0.0
    conviction_pass_train_count: int = 0
    conviction_pass_train_pct: float = 0.0
    lot_pass_count: int = 0
    lot_pass_pct: float = 0.0
    spread_pass_count: int = 0
    spread_pass_pct: float = 0.0
    overall_pass_count: int = 0
    overall_pass_pct: float = 0.0
    promote_rate: float = 0.0


@dataclass
class ConvictionStats:
    """Statistics about conviction values across all trades."""
    mean: float = 0.0
    std: float = 0.0
    p10: float = 0.0
    p50: float = 0.0
    p90: float = 0.0


@dataclass
class ActionStats:
    """Statistics about raw action outputs across all steps."""
    direction_mean: float = 0.0
    direction_std: float = 0.0
    conviction_mean: float = 0.0
    conviction_std: float = 0.0
    exit_mean: float = 0.0
    exit_std: float = 0.0
    sl_mean: float = 0.0
    sl_std: float = 0.0


@dataclass
class ScoreBreakdown:
    """SpartusScore component breakdown."""
    val_sharpe_component: float = 0.0
    val_pf_component: float = 0.0
    stress_component: float = 0.0
    max_dd_component: float = 0.0
    quality_component: float = 0.0
    spartus_score: float = 0.0


@dataclass
class DetectorResult:
    """Result from a single detector."""
    name: str
    detected: bool = False
    severity: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalBundle:
    """Everything needed to run a benchmark evaluation."""
    model: Any                          # SB3 SAC model
    config: Any                         # TrainingConfig
    model_id: str = ""
    model_path: Path = field(default_factory=Path)
    val_weeks: List[int] = field(default_factory=list)
    test_weeks: List[int] = field(default_factory=list)
    reward_state: Optional[Dict] = None
    metadata: Optional[Dict] = None


@dataclass
class BenchmarkResult:
    """Complete results from a full benchmark run."""
    run_id: str = ""
    timestamp: str = ""
    model_id: str = ""
    model_path: str = ""
    model_file_hash: str = ""
    suite: str = "full"
    seed: int = 42
    operator: str = ""

    # Reproducibility hashes
    data_manifest_hash: str = ""
    split_hash: str = ""
    feature_hash: str = ""
    config_hash: str = ""

    # T1: Validation metrics
    val_trades: int = 0
    val_win_pct: float = 0.0
    val_pf: float = 0.0
    val_sharpe: float = 0.0
    val_sortino: float = 0.0
    val_max_dd_pct: float = 0.0
    val_net_pnl: float = 0.0
    val_tim_pct: float = 0.0
    val_trades_day: float = 0.0
    val_avg_hold: float = 0.0
    val_median_hold: float = 0.0
    val_calmar: float = 0.0
    val_recovery_factor: float = 0.0
    val_tail_ratio: float = 0.0
    val_expectancy: float = 0.0
    val_max_consec_loss: int = 0
    val_max_consec_win: int = 0
    val_gross_profit: float = 0.0
    val_gross_loss: float = 0.0
    val_avg_win: float = 0.0
    val_avg_loss: float = 0.0
    val_win_loss_ratio: float = 0.0
    val_flat_bar_pct: float = 0.0
    val_entry_timing: float = 0.0
    val_sl_quality: float = 0.0
    val_long_count: int = 0
    val_short_count: int = 0
    val_long_pnl: float = 0.0
    val_short_pnl: float = 0.0
    val_long_pf: float = 0.0
    val_short_pf: float = 0.0

    # T2: Stress
    stress_results: Dict[str, StressResult] = field(default_factory=dict)
    stress_robustness_score: float = 0.0
    stress_worst_retention: float = 0.0
    stress_worst_scenario: str = ""

    # T3: Regime
    regime_slices: List[RegimeSlice] = field(default_factory=list)

    # T4: Churn
    churn: ChurnResult = field(default_factory=ChurnResult)

    # T5: Reward ablation
    reward_ablation: RewardAblation = field(default_factory=RewardAblation)

    # T6: Gating
    gating: GatingResult = field(default_factory=GatingResult)

    # Conviction & action stats
    conviction_stats: ConvictionStats = field(default_factory=ConvictionStats)
    action_stats: ActionStats = field(default_factory=ActionStats)

    # Scoring
    score: ScoreBreakdown = field(default_factory=ScoreBreakdown)

    # Hard-fail / detectors
    hard_fails: List[str] = field(default_factory=list)
    is_disqualified: bool = False
    detectors: List[DetectorResult] = field(default_factory=list)

    # Champion
    is_champion: bool = False

    # Locked test (T7)
    test_trades: Optional[int] = None
    test_win_pct: Optional[float] = None
    test_pf: Optional[float] = None
    test_sharpe: Optional[float] = None
    test_sortino: Optional[float] = None
    test_max_dd_pct: Optional[float] = None
    test_net_pnl: Optional[float] = None
    test_tim_pct: Optional[float] = None
    test_weeks_used: Optional[List[int]] = None

    # All trades for audit
    base_trades: List[TradeRecord] = field(default_factory=list)

    # Weekly returns for Sharpe/Sortino
    weekly_returns: List[float] = field(default_factory=list)

    # Total steps for TIM calculation
    total_steps: int = 0
    in_position_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization (excluding large lists)."""
        d = {}
        for k, v in self.__dict__.items():
            if k in ("base_trades", "weekly_returns"):
                continue
            if isinstance(v, (StressResult, RegimeSlice, ChurnResult,
                              RewardAblation, GatingResult, ConvictionStats,
                              ActionStats, ScoreBreakdown, DetectorResult)):
                d[k] = asdict(v)
            elif isinstance(v, dict):
                d[k] = {sk: asdict(sv) if hasattr(sv, '__dataclass_fields__') else sv
                        for sk, sv in v.items()}
            elif isinstance(v, list) and v and hasattr(v[0], '__dataclass_fields__'):
                d[k] = [asdict(item) for item in v]
            elif isinstance(v, Path):
                d[k] = str(v)
            else:
                d[k] = v
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)
