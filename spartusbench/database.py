"""SQLite persistence layer for SpartusBench.

Append-only database: rows are NEVER updated or deleted.
Only exception: leaderboard.dethroned_at (set when new champion replaces old).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import (
    BenchmarkResult, StressResult, RegimeSlice, TradeRecord,
    ScoreBreakdown, DetectorResult,
)

DB_PATH = Path("storage/benchmark/spartusbench.db")

SCHEMA_SQL = """
-- =====================================================
-- Table: benchmark_runs
-- =====================================================
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT UNIQUE NOT NULL,
    timestamp           TEXT NOT NULL,
    model_id            TEXT NOT NULL,
    model_path          TEXT NOT NULL,
    model_file_hash     TEXT,
    suite               TEXT NOT NULL,
    seed                INTEGER NOT NULL DEFAULT 42,
    operator            TEXT,
    data_manifest_hash  TEXT NOT NULL,
    split_hash          TEXT NOT NULL,
    feature_hash        TEXT NOT NULL,
    config_hash         TEXT NOT NULL,

    val_trades          INTEGER,
    val_win_pct         REAL,
    val_pf              REAL,
    val_sharpe          REAL,
    val_sortino         REAL,
    val_max_dd_pct      REAL,
    val_net_pnl         REAL,
    val_tim_pct         REAL,
    val_trades_day      REAL,
    val_avg_hold        REAL,
    val_median_hold     REAL,
    val_calmar          REAL,
    val_recovery_factor REAL,
    val_tail_ratio      REAL,
    val_expectancy      REAL,
    val_max_consec_loss INTEGER,
    val_max_consec_win  INTEGER,
    val_gross_profit    REAL,
    val_gross_loss      REAL,
    val_avg_win         REAL,
    val_avg_loss        REAL,
    val_win_loss_ratio  REAL,
    val_flat_bar_pct    REAL,
    val_entry_timing    REAL,
    val_sl_quality      REAL,
    val_long_count      INTEGER,
    val_short_count     INTEGER,
    val_long_pnl        REAL,
    val_short_pnl       REAL,
    val_long_pf         REAL,
    val_short_pf        REAL,

    stress_base_pf          REAL,
    stress_2x_spread_pf     REAL,
    stress_3x_spread_pf     REAL,
    stress_5x_spread_pf     REAL,
    stress_2x_slip_mean_pf  REAL,
    stress_2x_slip_std_pf   REAL,
    stress_combined_pf      REAL,
    stress_robustness_score REAL,
    stress_worst_retention  REAL,
    stress_worst_scenario   TEXT,

    churn_cost_per_trade    REAL,
    churn_total_cost        REAL,
    churn_net_edge          REAL,
    churn_gross_edge        REAL,
    churn_cost_to_edge      REAL,

    r1_pct_of_total         REAL,
    r2_pct_of_total         REAL,
    r3_pct_of_total         REAL,
    r4_pct_of_total         REAL,
    r5_pct_of_total         REAL,

    gate_direction_pass     REAL,
    gate_conviction_pass    REAL,
    gate_spread_pass        REAL,
    gate_lot_pass           REAL,
    gate_overall_pass       REAL,

    conv_mean               REAL,
    conv_std                REAL,
    conv_p10                REAL,
    conv_p50                REAL,
    conv_p90                REAL,

    action_direction_mean   REAL,
    action_direction_std    REAL,
    action_conviction_mean  REAL,
    action_conviction_std   REAL,
    action_exit_mean        REAL,
    action_exit_std         REAL,
    action_sl_mean          REAL,
    action_sl_std           REAL,

    spartus_score           REAL,
    score_val_sharpe        REAL,
    score_val_pf            REAL,
    score_stress            REAL,
    score_max_dd            REAL,
    score_quality           REAL,

    hard_fails              TEXT,
    is_disqualified         INTEGER DEFAULT 0,
    detector_aggression     INTEGER DEFAULT 0,
    detector_collapse       INTEGER DEFAULT 0,
    detector_fragility      INTEGER DEFAULT 0,
    detector_overfitting    INTEGER DEFAULT 0,
    detector_reward_hack    INTEGER DEFAULT 0,
    detector_details        TEXT,

    is_champion             INTEGER DEFAULT 0,

    test_trades             INTEGER,
    test_win_pct            REAL,
    test_pf                 REAL,
    test_sharpe             REAL,
    test_sortino            REAL,
    test_max_dd_pct         REAL,
    test_net_pnl            REAL,
    test_tim_pct            REAL,
    test_weeks_used         TEXT
);

-- =====================================================
-- Table: stress_details
-- =====================================================
CREATE TABLE IF NOT EXISTS stress_details (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES benchmark_runs(run_id),
    scenario        TEXT NOT NULL,
    set_name        TEXT NOT NULL DEFAULT 'VAL',
    trades          INTEGER,
    win_pct         REAL,
    net_pnl         REAL,
    pf              REAL,
    sharpe          REAL,
    sortino         REAL,
    max_dd_pct      REAL,
    tim_pct         REAL,
    avg_hold        REAL,
    trades_per_day  REAL,
    long_count      INTEGER,
    short_count     INTEGER,
    long_pnl        REAL,
    short_pnl       REAL,
    pf_retention    REAL
);

-- =====================================================
-- Table: regime_details
-- =====================================================
CREATE TABLE IF NOT EXISTS regime_details (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES benchmark_runs(run_id),
    slice_type      TEXT NOT NULL,
    slice_value     TEXT NOT NULL,
    trades          INTEGER,
    win_pct         REAL,
    net_pnl         REAL,
    pf              REAL,
    avg_pnl         REAL,
    avg_hold        REAL
);

-- =====================================================
-- Table: benchmark_trades
-- =====================================================
CREATE TABLE IF NOT EXISTS benchmark_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES benchmark_runs(run_id),
    scenario        TEXT NOT NULL DEFAULT 'base',
    trade_num       INTEGER,
    week            INTEGER,
    step            INTEGER,
    side            TEXT,
    entry_price     REAL,
    exit_price      REAL,
    lots            REAL,
    pnl             REAL,
    pnl_pct         REAL,
    hold_bars       INTEGER,
    conviction      REAL,
    close_reason    TEXT,
    lesson_type     TEXT,
    session         TEXT,
    atr_at_entry    REAL,
    max_favorable   REAL,
    initial_sl      REAL,
    initial_tp      REAL,
    final_sl        REAL
);

-- =====================================================
-- Table: leaderboard
-- =====================================================
CREATE TABLE IF NOT EXISTS leaderboard (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES benchmark_runs(run_id),
    model_id        TEXT NOT NULL,
    spartus_score   REAL NOT NULL,
    crowned_at      TEXT NOT NULL,
    dethroned_at    TEXT,
    notes           TEXT
);

-- =====================================================
-- Table: locked_test_audit
-- =====================================================
CREATE TABLE IF NOT EXISTS locked_test_audit (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL REFERENCES benchmark_runs(run_id),
    timestamp           TEXT NOT NULL,
    operator            TEXT NOT NULL,
    model_id            TEXT NOT NULL,
    model_file_hash     TEXT NOT NULL,
    data_manifest_hash  TEXT NOT NULL,
    split_hash          TEXT NOT NULL,
    feature_hash        TEXT NOT NULL,
    config_hash         TEXT NOT NULL,
    seed                INTEGER NOT NULL,
    test_weeks_used     TEXT NOT NULL,
    result_hash         TEXT NOT NULL
);

-- Indices
CREATE INDEX IF NOT EXISTS idx_runs_model     ON benchmark_runs(model_id);
CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON benchmark_runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_runs_champion  ON benchmark_runs(is_champion);
CREATE INDEX IF NOT EXISTS idx_stress_run     ON stress_details(run_id);
CREATE INDEX IF NOT EXISTS idx_regime_run     ON regime_details(run_id);
CREATE INDEX IF NOT EXISTS idx_trades_run     ON benchmark_trades(run_id);
CREATE INDEX IF NOT EXISTS idx_trades_scenario ON benchmark_trades(run_id, scenario);
CREATE INDEX IF NOT EXISTS idx_leader_score   ON leaderboard(spartus_score DESC);
CREATE INDEX IF NOT EXISTS idx_audit_model    ON locked_test_audit(model_id);
"""


class BenchmarkDB:
    """Append-only SQLite database for benchmark results."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_run(self, result: BenchmarkResult) -> None:
        """Insert a complete benchmark run into the database."""
        # Detector flags
        det_map = {d.name: d for d in result.detectors}

        # Stress PFs
        stress = result.stress_results
        base_pf = stress.get("base", StressResult(scenario="base")).pf

        self.conn.execute("""
            INSERT INTO benchmark_runs (
                run_id, timestamp, model_id, model_path, model_file_hash,
                suite, seed, operator,
                data_manifest_hash, split_hash, feature_hash, config_hash,
                val_trades, val_win_pct, val_pf, val_sharpe, val_sortino,
                val_max_dd_pct, val_net_pnl, val_tim_pct, val_trades_day,
                val_avg_hold, val_median_hold, val_calmar, val_recovery_factor,
                val_tail_ratio, val_expectancy, val_max_consec_loss, val_max_consec_win,
                val_gross_profit, val_gross_loss, val_avg_win, val_avg_loss,
                val_win_loss_ratio, val_flat_bar_pct, val_entry_timing, val_sl_quality,
                val_long_count, val_short_count, val_long_pnl, val_short_pnl,
                val_long_pf, val_short_pf,
                stress_base_pf, stress_2x_spread_pf, stress_3x_spread_pf,
                stress_5x_spread_pf, stress_2x_slip_mean_pf, stress_2x_slip_std_pf,
                stress_combined_pf, stress_robustness_score,
                stress_worst_retention, stress_worst_scenario,
                churn_cost_per_trade, churn_total_cost, churn_net_edge,
                churn_gross_edge, churn_cost_to_edge,
                r1_pct_of_total, r2_pct_of_total, r3_pct_of_total,
                r4_pct_of_total, r5_pct_of_total,
                gate_direction_pass, gate_conviction_pass, gate_spread_pass,
                gate_lot_pass, gate_overall_pass,
                conv_mean, conv_std, conv_p10, conv_p50, conv_p90,
                action_direction_mean, action_direction_std,
                action_conviction_mean, action_conviction_std,
                action_exit_mean, action_exit_std,
                action_sl_mean, action_sl_std,
                spartus_score, score_val_sharpe, score_val_pf,
                score_stress, score_max_dd, score_quality,
                hard_fails, is_disqualified,
                detector_aggression, detector_collapse, detector_fragility,
                detector_overfitting, detector_reward_hack, detector_details,
                is_champion,
                test_trades, test_win_pct, test_pf, test_sharpe, test_sortino,
                test_max_dd_pct, test_net_pnl, test_tim_pct, test_weeks_used
            ) VALUES (
                ?,?,?,?,?, ?,?,?, ?,?,?,?,
                ?,?,?,?,?, ?,?,?,?, ?,?,?,?,
                ?,?,?,?, ?,?,?,?,
                ?,?,?,?, ?,?,?,?,?,?,
                ?,?,?, ?,?,?, ?,?,?,?,
                ?,?,?,?,?,
                ?,?,?,?,?,
                ?,?,?,?,?,
                ?,?,?,?,?,
                ?,?,?,?, ?,?,?,?,
                ?,?,?, ?,?,?,
                ?,?,
                ?,?,?, ?,?,?,
                ?,
                ?,?,?,?,?, ?,?,?,?
            )
        """, (
            result.run_id, result.timestamp, result.model_id,
            result.model_path, result.model_file_hash,
            result.suite, result.seed, result.operator,
            result.data_manifest_hash, result.split_hash,
            result.feature_hash, result.config_hash,
            # T1
            result.val_trades, result.val_win_pct, result.val_pf,
            result.val_sharpe, result.val_sortino,
            result.val_max_dd_pct, result.val_net_pnl, result.val_tim_pct,
            result.val_trades_day, result.val_avg_hold, result.val_median_hold,
            result.val_calmar, result.val_recovery_factor,
            result.val_tail_ratio, result.val_expectancy,
            result.val_max_consec_loss, result.val_max_consec_win,
            result.val_gross_profit, result.val_gross_loss,
            result.val_avg_win, result.val_avg_loss,
            result.val_win_loss_ratio, result.val_flat_bar_pct,
            result.val_entry_timing, result.val_sl_quality,
            result.val_long_count, result.val_short_count,
            result.val_long_pnl, result.val_short_pnl,
            result.val_long_pf, result.val_short_pf,
            # T2 Stress
            base_pf,
            stress.get("2x_spread", StressResult(scenario="")).pf,
            stress.get("3x_spread", StressResult(scenario="")).pf,
            stress.get("5x_spread", StressResult(scenario="")).pf,
            stress.get("2x_slip_mean", StressResult(scenario="")).pf,
            stress.get("2x_slip_std", StressResult(scenario="")).pf,
            stress.get("combined_2x2x", StressResult(scenario="")).pf,
            result.stress_robustness_score,
            result.stress_worst_retention, result.stress_worst_scenario,
            # T4 Churn
            result.churn.est_cost_per_trade if result.churn else 0,
            result.churn.total_est_cost if result.churn else 0,
            result.churn.net_edge_per_trade if result.churn else 0,
            result.churn.gross_edge_per_trade if result.churn else 0,
            result.churn.cost_to_edge_ratio if result.churn else 0,
            # T5 Reward
            result.reward_ablation.r1_pct if result.reward_ablation else 0,
            result.reward_ablation.r2_pct if result.reward_ablation else 0,
            result.reward_ablation.r3_pct if result.reward_ablation else 0,
            result.reward_ablation.r4_pct if result.reward_ablation else 0,
            result.reward_ablation.r5_pct if result.reward_ablation else 0,
            # T6 Gating
            result.gating.direction_pass_pct if result.gating else 0,
            result.gating.conviction_pass_live_pct if result.gating else 0,
            result.gating.spread_pass_pct if result.gating else 0,
            result.gating.lot_pass_pct if result.gating else 0,
            result.gating.overall_pass_pct if result.gating else 0,
            # Conviction stats
            result.conviction_stats.mean, result.conviction_stats.std,
            result.conviction_stats.p10, result.conviction_stats.p50,
            result.conviction_stats.p90,
            # Action stats
            result.action_stats.direction_mean, result.action_stats.direction_std,
            result.action_stats.conviction_mean, result.action_stats.conviction_std,
            result.action_stats.exit_mean, result.action_stats.exit_std,
            result.action_stats.sl_mean, result.action_stats.sl_std,
            # Score
            result.score.spartus_score, result.score.val_sharpe_component,
            result.score.val_pf_component, result.score.stress_component,
            result.score.max_dd_component, result.score.quality_component,
            # Hard-fails / detectors
            json.dumps(result.hard_fails), int(result.is_disqualified),
            int(det_map.get("aggression_drift", DetectorResult(name="")).detected),
            int(det_map.get("conviction_collapse", DetectorResult(name="")).detected),
            int(det_map.get("stress_fragility", DetectorResult(name="")).detected),
            int(det_map.get("overfitting", DetectorResult(name="")).detected),
            int(det_map.get("reward_hacking", DetectorResult(name="")).detected),
            json.dumps([d.details for d in result.detectors]),
            # Champion
            int(result.is_champion),
            # Test
            result.test_trades, result.test_win_pct, result.test_pf,
            result.test_sharpe, result.test_sortino,
            result.test_max_dd_pct, result.test_net_pnl, result.test_tim_pct,
            json.dumps(result.test_weeks_used) if result.test_weeks_used else None,
        ))

        # Stress details
        for scenario, sr in result.stress_results.items():
            self.conn.execute("""
                INSERT INTO stress_details (
                    run_id, scenario, set_name, trades, win_pct, net_pnl, pf,
                    sharpe, sortino, max_dd_pct, tim_pct, avg_hold, trades_per_day,
                    long_count, short_count, long_pnl, short_pnl, pf_retention
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                result.run_id, scenario, "VAL", sr.trades, sr.win_pct,
                sr.net_pnl, sr.pf, sr.sharpe, sr.sortino, sr.max_dd_pct,
                sr.tim_pct, sr.avg_hold, sr.trades_per_day,
                sr.long_count, sr.short_count, sr.long_pnl, sr.short_pnl,
                sr.pf_retention,
            ))

        # Regime details
        for rs in result.regime_slices:
            self.conn.execute("""
                INSERT INTO regime_details (
                    run_id, slice_type, slice_value, trades, win_pct,
                    net_pnl, pf, avg_pnl, avg_hold
                ) VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                result.run_id, rs.slice_type, rs.slice_value,
                rs.trades, rs.win_pct, rs.net_pnl, rs.pf, rs.avg_pnl, rs.avg_hold,
            ))

        # Benchmark trades (base scenario only to save space; stress trades skipped)
        for t in result.base_trades:
            self.conn.execute("""
                INSERT INTO benchmark_trades (
                    run_id, scenario, trade_num, week, step, side,
                    entry_price, exit_price, lots, pnl, pnl_pct, hold_bars,
                    conviction, close_reason, lesson_type, session,
                    atr_at_entry, max_favorable, initial_sl, initial_tp, final_sl
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                result.run_id, t.scenario, t.trade_num, t.week, t.step, t.side,
                t.entry_price, t.exit_price, t.lots, t.pnl, t.pnl_pct, t.hold_bars,
                t.conviction, t.close_reason, t.lesson_type, t.session,
                t.atr_at_entry, t.max_favorable, t.initial_sl, t.initial_tp, t.final_sl,
            ))

        self.conn.commit()

    def save_locked_test_audit(
        self, result: BenchmarkResult, result_hash: str,
    ) -> None:
        """Record a locked test run in the audit trail."""
        self.conn.execute("""
            INSERT INTO locked_test_audit (
                run_id, timestamp, operator, model_id, model_file_hash,
                data_manifest_hash, split_hash, feature_hash, config_hash,
                seed, test_weeks_used, result_hash
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            result.run_id, result.timestamp, result.operator,
            result.model_id, result.model_file_hash,
            result.data_manifest_hash, result.split_hash,
            result.feature_hash, result.config_hash,
            result.seed,
            json.dumps(result.test_weeks_used or []),
            result_hash,
        ))
        self.conn.commit()

    def promote_champion(self, run_id: str, model_id: str, score: float,
                         timestamp: str) -> None:
        """Add a new champion to the leaderboard and dethrone the old one."""
        # Dethrone current champion
        self.conn.execute("""
            UPDATE leaderboard SET dethroned_at = ?
            WHERE dethroned_at IS NULL
        """, (timestamp,))

        # Mark run as champion
        self.conn.execute("""
            UPDATE benchmark_runs SET is_champion = 1 WHERE run_id = ?
        """, (run_id,))

        # Add to leaderboard
        self.conn.execute("""
            INSERT INTO leaderboard (run_id, model_id, spartus_score, crowned_at)
            VALUES (?,?,?,?)
        """, (run_id, model_id, score, timestamp))

        self.conn.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_current_champion(self) -> Optional[Dict[str, Any]]:
        """Get the current champion's benchmark run."""
        row = self.conn.execute("""
            SELECT br.* FROM benchmark_runs br
            JOIN leaderboard l ON br.run_id = l.run_id
            WHERE l.dethroned_at IS NULL
            ORDER BY l.spartus_score DESC
            LIMIT 1
        """).fetchone()
        return dict(row) if row else None

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM benchmark_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_latest_run(self, model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if model_id:
            row = self.conn.execute(
                "SELECT * FROM benchmark_runs WHERE model_id = ? ORDER BY timestamp DESC LIMIT 1",
                (model_id,)
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM benchmark_runs ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def get_leaderboard(self, top_n: int = 10, include_dq: bool = False) -> List[Dict[str, Any]]:
        if include_dq:
            rows = self.conn.execute("""
                SELECT br.model_id, br.spartus_score, br.val_sharpe, br.val_pf,
                       br.val_max_dd_pct, br.stress_robustness_score,
                       br.is_disqualified, br.is_champion, br.run_id, br.timestamp,
                       l.crowned_at, l.dethroned_at
                FROM benchmark_runs br
                LEFT JOIN leaderboard l ON br.run_id = l.run_id
                ORDER BY br.spartus_score DESC
                LIMIT ?
            """, (top_n,)).fetchall()
        else:
            rows = self.conn.execute("""
                SELECT br.model_id, br.spartus_score, br.val_sharpe, br.val_pf,
                       br.val_max_dd_pct, br.stress_robustness_score,
                       br.is_disqualified, br.is_champion, br.run_id, br.timestamp,
                       l.crowned_at, l.dethroned_at
                FROM benchmark_runs br
                LEFT JOIN leaderboard l ON br.run_id = l.run_id
                WHERE br.is_disqualified = 0
                ORDER BY br.spartus_score DESC
                LIMIT ?
            """, (top_n,)).fetchall()
        return [dict(r) for r in rows]

    def get_runs_for_model(self, model_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM benchmark_runs WHERE model_id = ? ORDER BY timestamp",
            (model_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stress_details(self, run_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM stress_details WHERE run_id = ?", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_regime_details(self, run_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM regime_details WHERE run_id = ?", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_trades(self, run_id: str, scenario: str = "base") -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM benchmark_trades WHERE run_id = ? AND scenario = ?",
            (run_id, scenario)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_locked_test_audit(self, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if model_id:
            rows = self.conn.execute(
                "SELECT * FROM locked_test_audit WHERE model_id = ? ORDER BY timestamp",
                (model_id,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM locked_test_audit ORDER BY timestamp"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_prior_results(self, model_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent benchmark results (for overfitting detector)."""
        rows = self.conn.execute("""
            SELECT * FROM benchmark_runs
            WHERE model_id != ? AND suite IN ('full', 'validation_only')
            ORDER BY timestamp DESC
            LIMIT ?
        """, (model_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_run_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()
        return row[0] if row else 0
