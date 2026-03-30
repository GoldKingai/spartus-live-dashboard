"""Persistent trading memory using SQLite.

5 tables:
- trades: Every completed trade with entry conditions
- patterns: Binned market conditions → outcomes (Bayesian)
- predictions: Trend prediction verification cycle
- tp_tracking: TP/SL hit analysis
- checkpoints: Model checkpoint metadata

Provides 5 memory features (#38-42) to the observation vector.
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import TrainingConfig


class TradingMemory:
    """SQLite-backed persistent memory for the trading agent."""

    def __init__(self, db_path: Optional[Path] = None, config: TrainingConfig = None):
        cfg = config or TrainingConfig()
        self.db_path = str(db_path or cfg.memory_db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")   # Safe with WAL, faster writes
        self.conn.execute("PRAGMA cache_size=-10000")     # 10 MB page cache
        self.conn.execute("PRAGMA mmap_size=268435456")   # 256 MB memory-mapped I/O
        self._create_tables()

        # Caches to avoid per-step DB queries
        self._cache_valid = False
        self._cached_features = np.array([0.5, 0.5, 0.5, 0.5, 0.0])
        self._cache_step = 0
        self._cache_ttl = 50  # Serve cached values for up to 50 steps

    def __getstate__(self):
        """Exclude sqlite3.Connection from pickle (SB3 model.save)."""
        state = self.__dict__.copy()
        state["conn"] = None
        return state

    def __setstate__(self, state):
        """Reconnect to SQLite after unpickling."""
        self.__dict__.update(state)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-10000")
        self.conn.execute("PRAGMA mmap_size=268435456")

    def _create_tables(self):
        c = self.conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                week INTEGER,
                step INTEGER,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                lot_size REAL,
                pnl REAL,
                pnl_pct REAL,
                hold_bars INTEGER,
                close_reason TEXT,
                conviction REAL,
                rsi_at_entry REAL,
                trend_dir_at_entry REAL,
                session_at_entry TEXT,
                vol_regime_at_entry REAL,
                entry_conditions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rsi_bin INTEGER,
                trend_bin INTEGER,
                session_bin TEXT,
                vol_bin INTEGER,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                avg_hold_bars REAL DEFAULT 0.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                week INTEGER,
                step INTEGER,
                predicted_direction REAL,
                predicted_confidence REAL,
                price_at_prediction REAL,
                verify_at_step INTEGER,
                actual_direction REAL,
                correct INTEGER,
                verified_at_step INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS tp_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER,
                tp_price REAL,
                sl_price REAL,
                tp_hit INTEGER DEFAULT 0,
                sl_hit INTEGER DEFAULT 0,
                max_favorable REAL DEFAULT 0.0,
                profit_locked_by_trail REAL DEFAULT 0.0,
                FOREIGN KEY (trade_id) REFERENCES trades(id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                week INTEGER,
                model_path TEXT,
                val_sharpe REAL,
                val_return REAL,
                val_max_dd REAL,
                val_win_rate REAL,
                training_weeks INTEGER,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS trade_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER,
                week INTEGER,
                lesson_type TEXT,
                entry_reasoning TEXT,
                exit_analysis TEXT,
                summary TEXT,
                direction_correct INTEGER,
                sl_quality TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trade_id) REFERENCES trades(id)
            )
        """)

        # Index for fast queries
        c.execute("CREATE INDEX IF NOT EXISTS idx_trades_week ON trades(week)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_predictions_verify ON predictions(verify_at_step)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_patterns_bins ON patterns(rsi_bin, trend_bin, session_bin, vol_bin)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_journal_week ON trade_journal(week)")

        self.conn.commit()

    # === Trade Recording ====================================================

    def record_trade(
        self,
        week: int,
        step: int,
        side: str,
        entry_price: float,
        exit_price: float,
        lot_size: float,
        pnl: float,
        pnl_pct: float,
        hold_bars: int,
        close_reason: str,
        conviction: float = 0.0,
        rsi_at_entry: float = 0.5,
        trend_dir_at_entry: float = 0.0,
        session_at_entry: str = "unknown",
        vol_regime_at_entry: float = 1.0,
        entry_conditions: Optional[Dict] = None,
    ) -> int:
        """Record a completed trade. Returns trade_id."""
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO trades (week, step, side, entry_price, exit_price, lot_size,
                                pnl, pnl_pct, hold_bars, close_reason, conviction,
                                rsi_at_entry, trend_dir_at_entry, session_at_entry,
                                vol_regime_at_entry, entry_conditions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            week, step, side, entry_price, exit_price, lot_size,
            pnl, pnl_pct, hold_bars, close_reason, conviction,
            rsi_at_entry, trend_dir_at_entry, session_at_entry,
            vol_regime_at_entry, json.dumps(entry_conditions or {}),
        ))
        self._cache_valid = False
        trade_id = c.lastrowid

        # Update pattern table (no intermediate commit)
        self._update_pattern(rsi_at_entry, trend_dir_at_entry,
                             session_at_entry, vol_regime_at_entry,
                             pnl > 0, pnl, hold_bars)
        # Single commit for trade + pattern
        self.conn.commit()
        return trade_id

    def record_tp_tracking(
        self,
        trade_id: int,
        tp_price: float,
        sl_price: float,
        tp_hit: bool,
        sl_hit: bool,
        max_favorable: float,
        profit_locked_by_trail: float,
    ):
        """Record TP/SL tracking data for a completed trade."""
        self.conn.execute("""
            INSERT INTO tp_tracking (trade_id, tp_price, sl_price, tp_hit, sl_hit,
                                     max_favorable, profit_locked_by_trail)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (trade_id, tp_price, sl_price, int(tp_hit), int(sl_hit),
              max_favorable, profit_locked_by_trail))
        self.conn.commit()
        self._cache_valid = False

    # === Pattern Tracking ===================================================

    def _update_pattern(
        self,
        rsi: float,
        trend_dir: float,
        session: str,
        vol_regime: float,
        is_win: bool,
        pnl: float,
        hold_bars: int,
    ):
        """Update pattern stats for binned market conditions."""
        rsi_bin = int(rsi * 10)           # 0-10 bins
        trend_bin = int(np.clip(trend_dir + 3, 0, 6))  # 0-6 bins
        vol_bin = int(np.clip(vol_regime * 5, 0, 10))   # 0-10 bins

        existing = self.conn.execute("""
            SELECT id, wins, losses, total_pnl, avg_hold_bars
            FROM patterns
            WHERE rsi_bin=? AND trend_bin=? AND session_bin=? AND vol_bin=?
        """, (rsi_bin, trend_bin, session, vol_bin)).fetchone()

        if existing:
            pid, wins, losses, total_pnl, avg_hold = existing
            count = wins + losses
            new_wins = wins + (1 if is_win else 0)
            new_losses = losses + (0 if is_win else 1)
            new_total = total_pnl + pnl
            new_avg_hold = (avg_hold * count + hold_bars) / (count + 1)
            self.conn.execute("""
                UPDATE patterns SET wins=?, losses=?, total_pnl=?, avg_hold_bars=?,
                                    updated_at=CURRENT_TIMESTAMP
                WHERE id=?
            """, (new_wins, new_losses, new_total, new_avg_hold, pid))
        else:
            self.conn.execute("""
                INSERT INTO patterns (rsi_bin, trend_bin, session_bin, vol_bin,
                                      wins, losses, total_pnl, avg_hold_bars)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (rsi_bin, trend_bin, session, vol_bin,
                  1 if is_win else 0, 0 if is_win else 1, pnl, float(hold_bars)))
        # Commit handled by caller (record_trade)

    # === Prediction Tracking ================================================

    def store_prediction(
        self,
        week: int,
        step: int,
        predicted_direction: float,
        confidence: float,
        current_price: float,
        lookforward: int = 20,
    ):
        """Store a trend prediction to be verified later."""
        verify_at = step + lookforward
        self.conn.execute("""
            INSERT INTO predictions (week, step, predicted_direction, predicted_confidence,
                                     price_at_prediction, verify_at_step)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (week, step, predicted_direction, confidence, current_price, verify_at))
        # Deferred commit — will be committed on next verify or trade recording

    def verify_predictions(self, current_step: int, current_price: float):
        """Verify any predictions that are due for checking."""
        pending = self.conn.execute("""
            SELECT id, predicted_direction, price_at_prediction
            FROM predictions
            WHERE verify_at_step <= ? AND verified_at_step IS NULL
        """, (current_step,)).fetchall()

        for pid, pred_dir, pred_price in pending:
            actual_dir = 1.0 if current_price > pred_price else -1.0
            correct = 1 if (pred_dir > 0 and actual_dir > 0) or (pred_dir < 0 and actual_dir < 0) else 0
            self.conn.execute("""
                UPDATE predictions SET actual_direction=?, correct=?, verified_at_step=?
                WHERE id=?
            """, (actual_dir, correct, current_step, pid))

        if pending:
            self.conn.commit()
            self._cache_valid = False

    # === Checkpoint Management ==============================================

    def save_checkpoint_meta(
        self,
        week: int,
        model_path: str,
        val_sharpe: float = 0.0,
        val_return: float = 0.0,
        val_max_dd: float = 0.0,
        val_win_rate: float = 0.0,
        training_weeks: int = 0,
        notes: str = "",
    ):
        self.conn.execute("""
            INSERT INTO checkpoints (week, model_path, val_sharpe, val_return,
                                     val_max_dd, val_win_rate, training_weeks, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (week, model_path, val_sharpe, val_return, val_max_dd,
              val_win_rate, training_weeks, notes))
        self.conn.commit()

    def get_best_checkpoint(self) -> Optional[Dict]:
        """Get the checkpoint with the best validation Sharpe ratio."""
        row = self.conn.execute("""
            SELECT week, model_path, val_sharpe, val_return, val_max_dd, val_win_rate
            FROM checkpoints ORDER BY val_sharpe DESC LIMIT 1
        """).fetchone()
        if not row:
            return None
        return {
            "week": row[0], "model_path": row[1], "val_sharpe": row[2],
            "val_return": row[3], "val_max_dd": row[4], "val_win_rate": row[5],
        }

    # === Memory Feature Queries (for observation vector) ====================

    def get_memory_features(self, market_state: Optional[Dict] = None,
                            current_step: int = 0) -> np.ndarray:
        """Return 5 memory features for the observation vector.

        Features (reflection-enhanced):
            [0] direction_accuracy — blend of win rate + journal direction accuracy
            [1] similar_pattern_winrate (Bayesian)
            [2] trend_prediction_accuracy
            [3] tp_hit_rate — blend with good_trade_rate from journal
            [4] sl_quality_score — blend of SL trail profit + journal SL quality

        Returns cached values if nothing changed since last call,
        or if fewer than _cache_ttl steps have passed (memory features
        change slowly — rolling stats over the last 20 trades).
        """
        if self._cache_valid and (current_step - self._cache_step) < self._cache_ttl:
            return self._cached_features.copy()

        win_rate = self._get_recent_win_rate()
        reflection = self.get_reflection_stats()

        # Blend base memory features with self-reflection stats
        # [0] Direction accuracy: 60% win rate + 40% journal direction accuracy
        dir_acc = reflection["direction_accuracy"]
        feat_0 = 0.6 * win_rate + 0.4 * dir_acc

        # [1] Pattern winrate: unchanged (already Bayesian)
        feat_1 = self._get_similar_pattern_winrate(market_state)

        # [2] Trend prediction accuracy: unchanged
        feat_2 = self._get_trend_accuracy()

        # [3] TP quality: 50% tp_hit_rate + 50% good_trade_rate
        tp_rate = self._get_tp_hit_rate()
        good_rate = reflection["good_trade_rate"]
        feat_3 = 0.5 * tp_rate + 0.5 * good_rate

        # [4] SL quality: 50% trail profit + 50% journal SL quality score
        trail = self._get_avg_sl_trail_profit()
        sl_q = reflection["sl_quality_score"]
        feat_4 = 0.5 * trail + 0.5 * sl_q

        feats = np.array([feat_0, feat_1, feat_2, feat_3, feat_4])

        self._cached_features = feats
        self._cache_valid = True
        self._cache_step = current_step
        return feats.copy()

    def invalidate_cache(self):
        """Force cache refresh on next get_memory_features() call."""
        self._cache_valid = False

    def _get_recent_win_rate(self, n: int = 20) -> float:
        """Win rate of last N trades."""
        rows = self.conn.execute(
            "SELECT pnl FROM trades ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        if not rows:
            return 0.5  # Prior
        wins = sum(1 for r in rows if r[0] > 0)
        return wins / len(rows)

    def _get_similar_pattern_winrate(self, market_state: Optional[Dict] = None) -> float:
        """Win rate for similar market conditions with Bayesian shrinkage."""
        if not market_state:
            return 0.5

        rsi_bin = int(market_state.get("rsi", 0.5) * 10)
        trend_bin = int(np.clip(market_state.get("trend_dir", 0) + 3, 0, 6))
        session = market_state.get("session", "unknown")
        vol_bin = int(np.clip(market_state.get("vol_regime", 1.0) * 5, 0, 10))

        # Look in adjacent bins too for more data
        rows = self.conn.execute("""
            SELECT wins, losses FROM patterns
            WHERE abs(rsi_bin - ?) <= 1 AND abs(trend_bin - ?) <= 1
              AND session_bin = ? AND abs(vol_bin - ?) <= 1
        """, (rsi_bin, trend_bin, session, vol_bin)).fetchall()

        if not rows:
            return 0.5

        total_wins = sum(r[0] for r in rows)
        total_losses = sum(r[1] for r in rows)
        count = total_wins + total_losses

        if count < 10:
            return 0.5

        raw_wr = total_wins / count
        # Bayesian shrinkage: blend with prior (0.5)
        credibility = count / (count + 30)
        return raw_wr * credibility + 0.5 * (1 - credibility)

    def _get_trend_accuracy(self, n: int = 100) -> float:
        """Rolling accuracy of verified trend predictions."""
        rows = self.conn.execute("""
            SELECT correct FROM predictions
            WHERE verified_at_step IS NOT NULL
            ORDER BY id DESC LIMIT ?
        """, (n,)).fetchall()

        if len(rows) < 10:
            return 0.5
        return sum(r[0] for r in rows) / len(rows)

    def _get_tp_hit_rate(self, n: int = 20) -> float:
        """Percentage of recent trades where TP was hit."""
        rows = self.conn.execute("""
            SELECT tp_hit FROM tp_tracking
            ORDER BY id DESC LIMIT ?
        """, (n,)).fetchall()

        if not rows:
            return 0.5
        return sum(r[0] for r in rows) / len(rows)

    def _get_avg_sl_trail_profit(self, n: int = 20) -> float:
        """Average profit locked by SL trailing (normalized)."""
        rows = self.conn.execute("""
            SELECT profit_locked_by_trail FROM tp_tracking
            ORDER BY id DESC LIMIT ?
        """, (n,)).fetchall()

        if not rows:
            return 0.0
        vals = [r[0] for r in rows if r[0] is not None]
        if not vals:
            return 0.0
        # Clip to [0, 1] for observation
        return float(np.clip(np.mean(vals), 0.0, 1.0))

    # === Dashboard Stats ====================================================

    def get_prediction_stats(self) -> Dict:
        """Get prediction stats for dashboard display."""
        pending = self.conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE verified_at_step IS NULL"
        ).fetchone()[0]
        verified = self.conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE verified_at_step IS NOT NULL"
        ).fetchone()[0]
        accuracy = self._get_trend_accuracy()
        return {"pending": pending, "verified": verified, "accuracy": accuracy}

    def get_tp_stats(self) -> Dict:
        """Get TP/SL stats for dashboard display."""
        total = self.conn.execute("SELECT COUNT(*) FROM tp_tracking").fetchone()[0]
        if total == 0:
            return {"tp_hit_rate": 0, "tp_reachable_rate": 0, "sl_hit_rate": 0}

        tp_hits = self.conn.execute(
            "SELECT COUNT(*) FROM tp_tracking WHERE tp_hit=1"
        ).fetchone()[0]
        sl_hits = self.conn.execute(
            "SELECT COUNT(*) FROM tp_tracking WHERE sl_hit=1"
        ).fetchone()[0]

        # TP reachable: max_favorable >= (tp_price - entry_price equivalent)
        # Approximation: if max_favorable > 0 and trade was profitable or TP was hit
        tp_reachable = self.conn.execute("""
            SELECT COUNT(*) FROM tp_tracking
            WHERE max_favorable > 0 AND (tp_hit=1 OR max_favorable > abs(tp_price - sl_price) * 0.8)
        """).fetchone()[0]

        return {
            "tp_hit_rate": tp_hits / total,
            "tp_reachable_rate": tp_reachable / total,
            "sl_hit_rate": sl_hits / total,
        }

    # === Trade Journal ======================================================

    def record_journal_entry(
        self,
        trade_id: int,
        week: int,
        lesson_type: str,
        entry_reasoning: str,
        exit_analysis: str,
        summary: str,
        direction_correct: int,
        sl_quality: str,
    ):
        """Record a post-trade journal entry with reasoning and analysis."""
        self.conn.execute("""
            INSERT INTO trade_journal (trade_id, week, lesson_type, entry_reasoning,
                                       exit_analysis, summary, direction_correct, sl_quality)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (trade_id, week, lesson_type, entry_reasoning, exit_analysis,
              summary, direction_correct, sl_quality))
        self.conn.commit()

    def get_reflection_stats(self, n: int = 30) -> Dict:
        """Get self-reflection stats from recent journal entries.

        Returns rolling metrics the AI can use to adjust its behavior.
        """
        rows = self.conn.execute("""
            SELECT lesson_type, direction_correct, sl_quality
            FROM trade_journal ORDER BY id DESC LIMIT ?
        """, (n,)).fetchall()

        if len(rows) < 5:
            return {
                "direction_accuracy": 0.5,
                "sl_quality_score": 0.5,
                "early_close_rate": 0.0,
                "wrong_direction_rate": 0.0,
                "good_trade_rate": 0.0,
            }

        total = len(rows)
        dir_correct = sum(1 for r in rows if r[1] == 1)
        sl_good = sum(1 for r in rows if r[2] in ("GOOD", "OK"))
        early_close = sum(1 for r in rows if r[0] in ("CORRECT_DIR_CLOSED_EARLY", "CORRECT_DIR_BAD_SL"))
        wrong_dir = sum(1 for r in rows if r[0] == "WRONG_DIRECTION")
        good = sum(1 for r in rows if r[0] == "GOOD_TRADE")

        return {
            "direction_accuracy": dir_correct / total,
            "sl_quality_score": sl_good / total,
            "early_close_rate": early_close / total,
            "wrong_direction_rate": wrong_dir / total,
            "good_trade_rate": good / total,
        }

    # === Utility Methods ====================================================

    def get_recent_trades(self, n: int = 20) -> List[Dict]:
        """Get the N most recent trades as dicts."""
        rows = self.conn.execute("""
            SELECT id, week, step, side, entry_price, exit_price, lot_size,
                   pnl, pnl_pct, hold_bars, close_reason, conviction
            FROM trades ORDER BY id DESC LIMIT ?
        """, (n,)).fetchall()

        return [{
            "id": r[0], "week": r[1], "step": r[2], "side": r[3],
            "entry_price": r[4], "exit_price": r[5], "lot_size": r[6],
            "pnl": r[7], "pnl_pct": r[8], "hold_bars": r[9],
            "close_reason": r[10], "conviction": r[11],
        } for r in rows]

    def get_trade_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM trades").fetchone()
        return row[0] if row else 0

    def get_total_pnl(self) -> float:
        row = self.conn.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades").fetchone()
        return row[0]

    def reset_for_new_episode(self):
        """Called at episode reset — cache is invalidated but data persists."""
        self._cache_valid = False

    def reset_for_fresh_training(self):
        """Clear all tables for a fresh training run (not resume).

        Prevents stale data from prior runs polluting pattern matching
        and prediction accuracy.
        """
        for table in ("trades", "patterns", "predictions", "tp_tracking", "checkpoints", "trade_journal"):
            self.conn.execute(f"DELETE FROM {table}")
        self.conn.commit()
        self._cache_valid = False
        self._cached_features = np.array([0.5, 0.5, 0.5, 0.5, 0.0])

    def close(self):
        self.conn.close()
