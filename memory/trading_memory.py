"""SQLite-backed trading memory for the live dashboard.

Standalone memory system -- no imports from training codebase.
Same conceptual schema as src/memory/trading_memory.py but fully
self-contained for the live dashboard.

5 tables:
- trades: every completed trade
- patterns: binned market conditions -> outcomes (Bayesian)
- predictions: trend prediction verification cycle
- tp_tracking: TP/SL hit analysis
- journal: trade quality classification
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _session_from_utc_hour(hour: int) -> str:
    """Determine trading session from UTC hour.

    London   07-12
    NY       12-16
    NY_PM    16-20
    Asia     00-07
    Off      20-24
    """
    if 0 <= hour < 7:
        return "Asia"
    elif 7 <= hour < 12:
        return "London"
    elif 12 <= hour < 16:
        return "NY"
    elif 16 <= hour < 20:
        return "NY_PM"
    else:
        return "Off"


class TradingMemory:
    """SQLite-backed trading memory for live dashboard.

    Tables:
    - trades: every completed trade
    - patterns: binned market conditions -> outcomes
    - predictions: trend prediction verification cycle
    - tp_tracking: TP/SL hit analysis
    - journal: trade quality classification
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, db_path: str):
        """Create / connect to the SQLite database.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database file.  Parent directories are
            created automatically.
        """
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-10000")       # 10 MB page cache
        self.conn.execute("PRAGMA mmap_size=268435456")      # 256 MB mmap I/O
        self._create_tables()

        # Feature cache -- memory features change slowly, no need to
        # recompute every step.
        self._cache_valid = False
        self._cached_features = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        self._cache_step = 0
        self._cache_ttl = 50  # Serve cached values for up to 50 steps

        logger.info("TradingMemory initialised: %s", self.db_path)

    # ------------------------------------------------------------------
    # Pickle support (safety -- if anything tries to serialise us)
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        state["conn"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-10000")
        self.conn.execute("PRAGMA mmap_size=268435456")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self):
        c = self.conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                week INTEGER DEFAULT 0,
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
                paper_trade BOOLEAN DEFAULT 0
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rsi_bin INTEGER,
                trend_bin INTEGER,
                session TEXT,
                vol_bin INTEGER,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                avg_hold_bars REAL DEFAULT 0.0
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                step INTEGER,
                predicted_direction REAL,
                confidence REAL,
                price_at_prediction REAL,
                actual_direction REAL,
                correct INTEGER
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS tp_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER,
                tp_price REAL,
                sl_price REAL,
                tp_hit BOOLEAN,
                sl_hit BOOLEAN,
                max_favorable REAL,
                profit_locked_by_trail REAL,
                FOREIGN KEY (trade_id) REFERENCES trades(id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER,
                lesson_type TEXT,
                notes TEXT,
                FOREIGN KEY (trade_id) REFERENCES trades(id)
            )
        """)

        # Indices for fast look-ups
        c.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_trades_week ON trades(week)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_patterns_bins ON patterns(rsi_bin, trend_bin, session, vol_bin)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_predictions_step ON predictions(step)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_journal_trade ON journal(trade_id)")

        # --- Migration: add mt5_ticket column if not present ---
        try:
            c.execute("SELECT mt5_ticket FROM trades LIMIT 1")
        except sqlite3.OperationalError:
            c.execute("ALTER TABLE trades ADD COLUMN mt5_ticket INTEGER DEFAULT 0")
            c.execute("CREATE INDEX IF NOT EXISTS idx_trades_mt5_ticket ON trades(mt5_ticket)")
            logger.info("Migration: added mt5_ticket column to trades table")

        self.conn.commit()
        logger.debug("All 5 tables created / verified.")

    # ==================================================================
    # Trade Recording
    # ==================================================================

    def record_trade(self, trade_data: dict) -> int:
        """Insert a completed trade.  Returns the new trade id.

        Expected keys in *trade_data*:
            timestamp, week, step, side, entry_price, exit_price, lot_size,
            pnl, pnl_pct, hold_bars, close_reason, conviction,
            rsi_at_entry, trend_dir_at_entry, session_at_entry,
            vol_regime_at_entry, entry_conditions (dict or JSON str),
            paper_trade (bool, optional), mt5_ticket (int, optional).
        """
        d = trade_data
        entry_cond = d.get("entry_conditions", {})
        if isinstance(entry_cond, dict):
            entry_cond = json.dumps(entry_cond)

        c = self.conn.cursor()
        c.execute("""
            INSERT INTO trades (
                timestamp, week, step, side, entry_price, exit_price, lot_size,
                pnl, pnl_pct, hold_bars, close_reason, conviction,
                rsi_at_entry, trend_dir_at_entry, session_at_entry,
                vol_regime_at_entry, entry_conditions, paper_trade,
                mt5_ticket
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            d.get("timestamp", datetime.now(timezone.utc).isoformat()),
            d.get("week", 0),
            d.get("step", 0),
            d.get("side", "LONG"),
            d.get("entry_price", 0.0),
            d.get("exit_price", 0.0),
            d.get("lot_size", 0.0),
            d.get("pnl", 0.0),
            d.get("pnl_pct", 0.0),
            d.get("hold_bars", 0),
            d.get("close_reason", "AGENT"),
            d.get("conviction", 0.0),
            d.get("rsi_at_entry", 0.5),
            d.get("trend_dir_at_entry", 0.0),
            d.get("session_at_entry", "unknown"),
            d.get("vol_regime_at_entry", 1.0),
            entry_cond,
            int(d.get("paper_trade", False)),
            d.get("mt5_ticket", 0),
        ))
        trade_id = c.lastrowid
        self._cache_valid = False

        # Also update pattern table
        self._update_pattern(
            rsi=d.get("rsi_at_entry", 0.5),
            trend_dir=d.get("trend_dir_at_entry", 0.0),
            session=d.get("session_at_entry", "unknown"),
            vol_regime=d.get("vol_regime_at_entry", 1.0),
            won=d.get("pnl", 0.0) > 0,
            pnl=d.get("pnl", 0.0),
            hold_bars=d.get("hold_bars", 0),
        )

        self.conn.commit()
        logger.debug("Recorded trade id=%d  pnl=%.2f", trade_id, d.get("pnl", 0.0))
        return trade_id

    def has_mt5_ticket(self, ticket: int) -> bool:
        """Check if a trade with the given MT5 ticket already exists."""
        if ticket <= 0:
            return False
        try:
            row = self.conn.execute(
                "SELECT 1 FROM trades WHERE mt5_ticket = ? LIMIT 1",
                (ticket,),
            ).fetchone()
            return row is not None
        except sqlite3.OperationalError as e:
            logger.warning("has_mt5_ticket query failed: %s", e)
            return False

    def get_known_mt5_tickets(self) -> set:
        """Return the set of all MT5 tickets we've recorded."""
        try:
            rows = self.conn.execute(
                "SELECT mt5_ticket FROM trades WHERE mt5_ticket > 0"
            ).fetchall()
            return {r[0] for r in rows}
        except sqlite3.OperationalError as e:
            logger.warning("get_known_mt5_tickets query failed: %s", e)
            return set()

    def has_matching_trade(
        self,
        side: str,
        entry_price: float,
        exit_price: float,
        lot_size: float,
        pnl: float,
    ) -> bool:
        """Check if a trade with matching characteristics already exists.

        Used to prevent duplicates when reconciling MT5 history against
        trades that were recorded before the mt5_ticket column existed.
        Matches on side + entry_price + exit_price + lot_size + pnl
        (rounded to 2 decimals for price tolerance).
        """
        row = self.conn.execute(
            """
            SELECT 1 FROM trades
            WHERE side = ?
              AND ROUND(entry_price, 2) = ROUND(?, 2)
              AND ROUND(exit_price, 2) = ROUND(?, 2)
              AND ROUND(lot_size, 3) = ROUND(?, 3)
              AND ROUND(pnl, 1) = ROUND(?, 1)
            LIMIT 1
            """,
            (side, entry_price, exit_price, lot_size, pnl),
        ).fetchone()
        return row is not None

    def backfill_mt5_ticket(
        self,
        side: str,
        entry_price: float,
        exit_price: float,
        lot_size: float,
        pnl: float,
        mt5_ticket: int,
    ) -> bool:
        """Set the mt5_ticket on a matching trade that has ticket=0.

        Returns True if a row was updated.
        """
        c = self.conn.cursor()
        # Use subquery to update only the first match (SQLite has no
        # UPDATE ... LIMIT)
        c.execute(
            """
            UPDATE trades
            SET mt5_ticket = ?
            WHERE id = (
                SELECT id FROM trades
                WHERE mt5_ticket = 0
                  AND side = ?
                  AND ROUND(entry_price, 2) = ROUND(?, 2)
                  AND ROUND(exit_price, 2) = ROUND(?, 2)
                  AND ROUND(lot_size, 3) = ROUND(?, 3)
                  AND ROUND(pnl, 1) = ROUND(?, 1)
                ORDER BY id LIMIT 1
            )
            """,
            (mt5_ticket, side, entry_price, exit_price, lot_size, pnl),
        )
        if c.rowcount > 0:
            self.conn.commit()
            logger.info(
                "Backfilled mt5_ticket=%d on existing trade "
                "(%s entry=%.2f exit=%.2f)",
                mt5_ticket, side, entry_price, exit_price,
            )
            return True
        return False

    # ==================================================================
    # Pattern Tracking
    # ==================================================================

    def record_pattern(
        self,
        rsi_bin: int,
        trend_bin: int,
        session: str,
        vol_bin: int,
        won: bool,
        pnl: float,
        hold_bars: int,
    ):
        """Directly record / update a pattern row (public API)."""
        self._update_pattern_binned(rsi_bin, trend_bin, session, vol_bin, won, pnl, hold_bars)
        self.conn.commit()
        self._cache_valid = False

    def _update_pattern(
        self,
        rsi: float,
        trend_dir: float,
        session: str,
        vol_regime: float,
        won: bool,
        pnl: float,
        hold_bars: int,
    ):
        """Update pattern stats -- raw values are binned first."""
        rsi_bin = int(np.clip(rsi * 10, 0, 10))
        trend_bin = int(np.clip((trend_dir + 3) / 6 * 6, 0, 6))
        vol_bin = int(np.clip(vol_regime * 10, 0, 10))
        self._update_pattern_binned(rsi_bin, trend_bin, session, vol_bin, won, pnl, hold_bars)

    def _update_pattern_binned(
        self,
        rsi_bin: int,
        trend_bin: int,
        session: str,
        vol_bin: int,
        won: bool,
        pnl: float,
        hold_bars: int,
    ):
        """Insert or update a pattern row using pre-binned values."""
        existing = self.conn.execute("""
            SELECT id, wins, losses, total_pnl, avg_hold_bars
            FROM patterns
            WHERE rsi_bin=? AND trend_bin=? AND session=? AND vol_bin=?
        """, (rsi_bin, trend_bin, session, vol_bin)).fetchone()

        if existing:
            pid, wins, losses, total_pnl, avg_hold = existing
            count = wins + losses
            new_wins = wins + (1 if won else 0)
            new_losses = losses + (0 if won else 1)
            new_total = total_pnl + pnl
            new_avg_hold = (avg_hold * count + hold_bars) / (count + 1)
            self.conn.execute("""
                UPDATE patterns SET wins=?, losses=?, total_pnl=?, avg_hold_bars=?
                WHERE id=?
            """, (new_wins, new_losses, new_total, new_avg_hold, pid))
        else:
            self.conn.execute("""
                INSERT INTO patterns (rsi_bin, trend_bin, session, vol_bin,
                                      wins, losses, total_pnl, avg_hold_bars)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (rsi_bin, trend_bin, session, vol_bin,
                  1 if won else 0, 0 if won else 1, pnl, float(hold_bars)))
        # Commit handled by caller

    # ==================================================================
    # Prediction Tracking
    # ==================================================================

    def record_prediction(self, step: int, direction: float, confidence: float, price: float):
        """Store a trend prediction to be verified later.

        Parameters
        ----------
        step : int
            Current step / bar index.
        direction : float
            Predicted direction (positive = long, negative = short).
        confidence : float
            Confidence in [0, 1].
        price : float
            Price at the moment of prediction.
        """
        self.conn.execute("""
            INSERT INTO predictions (timestamp, step, predicted_direction, confidence,
                                     price_at_prediction)
            VALUES (?, ?, ?, ?, ?)
        """, (datetime.now(timezone.utc).isoformat(), step, direction, confidence, price))
        self.conn.commit()
        logger.debug("Prediction recorded at step %d: dir=%.3f conf=%.3f", step, direction, confidence)

    def verify_predictions(self, step: int, current_price: float, lookforward: int = 20):
        """Verify predictions older than *lookforward* bars.

        A prediction is correct if the predicted direction matches the
        actual price movement from the prediction price to *current_price*.
        """
        cutoff_step = step - lookforward
        pending = self.conn.execute("""
            SELECT id, predicted_direction, price_at_prediction
            FROM predictions
            WHERE actual_direction IS NULL AND step <= ?
        """, (cutoff_step,)).fetchall()

        if not pending:
            return

        for pid, pred_dir, pred_price in pending:
            actual_dir = 1.0 if current_price > pred_price else -1.0
            correct = 1 if (pred_dir > 0 and actual_dir > 0) or (pred_dir < 0 and actual_dir < 0) else 0
            self.conn.execute("""
                UPDATE predictions SET actual_direction=?, correct=?
                WHERE id=?
            """, (actual_dir, correct, pid))

        self.conn.commit()
        self._cache_valid = False
        logger.debug("Verified %d predictions at step %d", len(pending), step)

    # ==================================================================
    # TP / SL Tracking
    # ==================================================================

    def record_tp_tracking(
        self,
        trade_id: int,
        tp_price: float,
        sl_price: float,
        tp_hit: bool,
        sl_hit: bool,
        max_favorable: float,
        profit_locked: float,
    ):
        """Record TP/SL tracking data for a completed trade."""
        self.conn.execute("""
            INSERT INTO tp_tracking (trade_id, tp_price, sl_price, tp_hit, sl_hit,
                                     max_favorable, profit_locked_by_trail)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (trade_id, tp_price, sl_price, int(tp_hit), int(sl_hit),
              max_favorable, profit_locked))
        self.conn.commit()
        self._cache_valid = False

    # ==================================================================
    # Journal
    # ==================================================================

    def record_journal(self, trade_id: int, lesson_type: str, notes: str = ""):
        """Record a journal entry classifying a trade."""
        self.conn.execute("""
            INSERT INTO journal (trade_id, lesson_type, notes)
            VALUES (?, ?, ?)
        """, (trade_id, lesson_type, notes))
        self.conn.commit()

    # ==================================================================
    # Memory Features (5-dim vector for observation)
    # ==================================================================

    def get_memory_features(self, market_state: dict, step: int) -> np.ndarray:
        """Compute 5 memory features for the observation vector.

        Features:
            [0] recent_win_rate         -- wins / total from last 20 trades
            [1] similar_pattern_winrate -- Bayesian: (wins+1)/(wins+losses+2)
            [2] trend_prediction_accuracy -- correct / total verified
            [3] tp_hit_rate             -- blended TP + good trade rate
            [4] avg_sl_trail_profit     -- blended trail profit + SL quality

        Uses a 50-step cache to avoid hammering the DB every step.
        """
        if self._cache_valid and (step - self._cache_step) < self._cache_ttl:
            return self._cached_features.copy()

        feat_0 = self._get_recent_win_rate()
        feat_1 = self._get_similar_pattern_winrate(market_state)
        feat_2 = self._get_trend_accuracy()

        # [3] TP quality: 50% tp_hit_rate + 50% good_trade_rate from journal
        tp_rate = self._get_tp_hit_rate()
        good_rate = self._get_good_trade_rate()
        feat_3 = 0.5 * tp_rate + 0.5 * good_rate

        # [4] SL quality: 50% avg trail profit + 50% journal SL quality score
        trail = self._get_avg_sl_trail_profit()
        sl_q = self._get_sl_quality_score()
        feat_4 = 0.5 * trail + 0.5 * sl_q

        feats = np.array([feat_0, feat_1, feat_2, feat_3, feat_4], dtype=np.float32)

        self._cached_features = feats
        self._cache_valid = True
        self._cache_step = step
        return feats.copy()

    def get_memory_features_safe(self, market_state: dict, step: int) -> np.ndarray:
        """Safe memory features with blending for fresh accounts.

        For accounts with fewer than 50 trades the raw memory features
        are blended towards the neutral default (0.5) so that noisy
        early statistics do not mislead the agent.

        Blend schedule:
            < 20 trades  -> pure defaults [0.5, 0.5, 0.5, 0.5, 0.5]
            20-50 trades -> linear blend from defaults to raw features
            > 50 trades  -> pure raw features
        """
        defaults = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        trade_count = self.get_trade_count()

        if trade_count < 20:
            return defaults.copy()

        raw = self.get_memory_features(market_state, step)

        if trade_count >= 50:
            return raw

        # Linear blend between 20 and 50 trades
        alpha = (trade_count - 20) / 30.0  # 0.0 at 20, 1.0 at 50
        blended = defaults * (1.0 - alpha) + raw * alpha
        return blended

    # ------------------------------------------------------------------
    # Internal feature helpers
    # ------------------------------------------------------------------

    def _get_recent_win_rate(self, n: int = 20) -> float:
        """Win rate of the last *n* trades (default 0.5)."""
        rows = self.conn.execute(
            "SELECT pnl FROM trades ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        if not rows:
            return 0.5
        wins = sum(1 for r in rows if r[0] > 0)
        return wins / len(rows)

    def _get_similar_pattern_winrate(self, market_state: Optional[dict] = None) -> float:
        """Win rate for similar market conditions with Bayesian prior.

        Bayesian formula: (wins + 1) / (wins + losses + 2)
        This gives a prior of 0.5 with minimal data.
        """
        if not market_state:
            return 0.5

        rsi_bin = int(np.clip(market_state.get("rsi", 0.5) * 10, 0, 10))
        trend_bin = int(np.clip((market_state.get("trend_dir", 0.0) + 3) / 6 * 6, 0, 6))
        session = market_state.get("session", "unknown")
        vol_bin = int(np.clip(market_state.get("vol_regime", 1.0) * 10, 0, 10))

        # Look at adjacent bins for more data
        rows = self.conn.execute("""
            SELECT wins, losses FROM patterns
            WHERE abs(rsi_bin - ?) <= 1 AND abs(trend_bin - ?) <= 1
              AND session = ? AND abs(vol_bin - ?) <= 1
        """, (rsi_bin, trend_bin, session, vol_bin)).fetchall()

        if not rows:
            return 0.5

        total_wins = sum(r[0] for r in rows)
        total_losses = sum(r[1] for r in rows)

        # Bayesian: (wins + 1) / (wins + losses + 2)
        return (total_wins + 1) / (total_wins + total_losses + 2)

    def _get_trend_accuracy(self, n: int = 100) -> float:
        """Rolling accuracy of verified trend predictions (default 0.5)."""
        rows = self.conn.execute("""
            SELECT correct FROM predictions
            WHERE correct IS NOT NULL
            ORDER BY id DESC LIMIT ?
        """, (n,)).fetchall()

        if len(rows) < 10:
            return 0.5
        return sum(r[0] for r in rows) / len(rows)

    def _get_tp_hit_rate(self, n: int = 20) -> float:
        """Percentage of recent trades where TP was hit (default 0.5)."""
        rows = self.conn.execute("""
            SELECT tp_hit FROM tp_tracking
            ORDER BY id DESC LIMIT ?
        """, (n,)).fetchall()
        if not rows:
            return 0.5
        return sum(r[0] for r in rows) / len(rows)

    def _get_avg_sl_trail_profit(self, n: int = 20) -> float:
        """Average profit locked by SL trailing, normalised to [0, 1] (default 0.5)."""
        rows = self.conn.execute("""
            SELECT profit_locked_by_trail FROM tp_tracking
            ORDER BY id DESC LIMIT ?
        """, (n,)).fetchall()
        if not rows:
            return 0.5
        vals = [r[0] for r in rows if r[0] is not None]
        if not vals:
            return 0.5
        return float(np.clip(np.mean(vals), 0.0, 1.0))

    def _get_good_trade_rate(self, n: int = 30) -> float:
        """Fraction of recent journal entries classified as GOOD_TRADE."""
        rows = self.conn.execute("""
            SELECT lesson_type FROM journal
            ORDER BY id DESC LIMIT ?
        """, (n,)).fetchall()
        if not rows:
            return 0.5
        good = sum(1 for r in rows if r[0] == "GOOD_TRADE")
        return good / len(rows)

    def _get_sl_quality_score(self, n: int = 30) -> float:
        """Quality score derived from journal entries.

        Counts GOOD_TRADE and SMALL_WIN as 'good SL management'.
        """
        rows = self.conn.execute("""
            SELECT lesson_type FROM journal
            ORDER BY id DESC LIMIT ?
        """, (n,)).fetchall()
        if not rows:
            return 0.5
        good_sl = sum(1 for r in rows if r[0] in ("GOOD_TRADE", "SMALL_WIN"))
        return good_sl / len(rows)

    # ==================================================================
    # Query Methods
    # ==================================================================

    def get_trade_count(self) -> int:
        """Total number of trades recorded."""
        row = self.conn.execute("SELECT COUNT(*) FROM trades").fetchone()
        return row[0] if row else 0

    def get_recent_trades(self, limit: int = 20) -> list:
        """Return the *limit* most recent trades as dicts."""
        rows = self.conn.execute("""
            SELECT id, timestamp, week, step, side, entry_price, exit_price,
                   lot_size, pnl, pnl_pct, hold_bars, close_reason,
                   conviction, rsi_at_entry, trend_dir_at_entry,
                   session_at_entry, vol_regime_at_entry, entry_conditions,
                   paper_trade
            FROM trades ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()

        results: List[Dict] = []
        for r in rows:
            entry_cond = r[17]
            try:
                entry_cond = json.loads(entry_cond) if entry_cond else {}
            except (json.JSONDecodeError, TypeError):
                entry_cond = {}

            results.append({
                "id": r[0],
                "timestamp": r[1],
                "week": r[2],
                "step": r[3],
                "side": r[4],
                "entry_price": r[5],
                "exit_price": r[6],
                "lot_size": r[7],
                "pnl": r[8],
                "pnl_pct": r[9],
                "hold_bars": r[10],
                "close_reason": r[11],
                "conviction": r[12],
                "rsi_at_entry": r[13],
                "trend_dir_at_entry": r[14],
                "session_at_entry": r[15],
                "vol_regime_at_entry": r[16],
                "entry_conditions": entry_cond,
                "paper_trade": bool(r[18]),
            })
        return results

    def get_today_summary(self) -> dict:
        """Return today's trading summary computed from the database.

        Returns dict with: trades, wins, losses, pnl, win_rate,
        max_dd, profit_factor.  All values default to 0 if no trades today.
        """
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        row = self.conn.execute("""
            SELECT
                COUNT(*)                                          AS trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)         AS wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END)        AS losses,
                COALESCE(SUM(pnl), 0.0)                           AS net_pnl,
                COALESCE(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), 0.0)
                                                                  AS gross_profit,
                COALESCE(SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END), 0.0)
                                                                  AS gross_loss
            FROM trades
            WHERE timestamp LIKE ?
        """, (f"{today_str}%",)).fetchone()

        trades = row[0] or 0
        wins = row[1] or 0
        losses = row[2] or 0
        net_pnl = row[3] or 0.0
        gross_profit = row[4] or 0.0
        gross_loss = row[5] or 0.0

        win_rate = (wins / trades * 100) if trades > 0 else 0.0
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else
            (999.0 if gross_profit > 0 else 0.0)
        )

        # Max intra-day drawdown: peak-to-trough of cumulative P/L
        pnls = self.conn.execute("""
            SELECT pnl FROM trades
            WHERE timestamp LIKE ?
            ORDER BY id ASC
        """, (f"{today_str}%",)).fetchall()

        max_dd = 0.0
        if pnls:
            cumulative = 0.0
            peak = 0.0
            for (p,) in pnls:
                cumulative += p
                if cumulative > peak:
                    peak = cumulative
                dd = peak - cumulative
                if dd > max_dd:
                    max_dd = dd

        return {
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "pnl": round(net_pnl, 2),
            "win_rate": round(win_rate, 1),
            "max_dd": round(max_dd, 2),
            "profit_factor": round(profit_factor, 2),
        }

    def get_lesson_summary(self) -> dict:
        """Count of each lesson_type in the journal."""
        rows = self.conn.execute("""
            SELECT lesson_type, COUNT(*) FROM journal
            GROUP BY lesson_type
        """).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_session_breakdown(self) -> dict:
        """Performance breakdown by trading session.

        Returns dict mapping session name to aggregated stats.
        Win rate is returned as a percentage (0-100), not a fraction.
        """
        rows = self.conn.execute("""
            SELECT session_at_entry,
                   COUNT(*) as cnt,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(pnl) as net_pnl,
                   AVG(pnl) as avg_pnl,
                   SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
                   SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as gross_loss
            FROM trades
            GROUP BY session_at_entry
        """).fetchall()

        result = {}
        for r in rows:
            session = r[0] or "unknown"
            cnt = r[1]
            wins = r[2]
            gross_profit = r[5] or 0.0
            gross_loss = r[6] or 0.0
            pf = gross_profit / gross_loss if gross_loss > 0 else 0.0
            result[session] = {
                "trades": cnt,
                "wins": wins,
                "losses": cnt - wins,
                "win_rate": round(wins / cnt * 100, 1) if cnt > 0 else 0.0,
                "pf": round(pf, 2),
                "net_pnl": r[3] or 0.0,
                "avg_pnl": r[4] or 0.0,
            }
        return result

    def get_day_of_week_breakdown(self) -> dict:
        """Net P/L by day of the week (Mon=0 .. Sun=6).

        Parses the stored timestamp to extract the weekday.
        """
        rows = self.conn.execute("""
            SELECT timestamp, pnl FROM trades
            WHERE timestamp IS NOT NULL
        """).fetchall()

        day_map: Dict[str, float] = {
            "Mon": 0.0, "Tue": 0.0, "Wed": 0.0,
            "Thu": 0.0, "Fri": 0.0, "Sat": 0.0, "Sun": 0.0,
        }
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        for ts_str, pnl in rows:
            try:
                dt = datetime.fromisoformat(ts_str)
                day_name = day_names[dt.weekday()]
                day_map[day_name] += pnl or 0.0
            except (ValueError, TypeError):
                continue

        return day_map

    def get_weekly_summary(self, start: str, end: str) -> dict:
        """Aggregated metrics for trades between *start* and *end* (ISO strings).

        Returns dict with keys: trades, wins, losses, win_rate, net_pnl,
        avg_pnl, avg_hold_bars, best_trade, worst_trade.
        """
        rows = self.conn.execute("""
            SELECT pnl, hold_bars FROM trades
            WHERE timestamp >= ? AND timestamp <= ?
        """, (start, end)).fetchall()

        if not rows:
            return {
                "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "net_pnl": 0.0, "avg_pnl": 0.0, "avg_hold_bars": 0.0,
                "best_trade": 0.0, "worst_trade": 0.0,
            }

        pnls = [r[0] for r in rows]
        holds = [r[1] for r in rows]
        wins = sum(1 for p in pnls if p > 0)
        total = len(pnls)

        return {
            "trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": wins / total if total > 0 else 0.0,
            "net_pnl": sum(pnls),
            "avg_pnl": sum(pnls) / total,
            "avg_hold_bars": sum(holds) / total if holds else 0.0,
            "best_trade": max(pnls),
            "worst_trade": min(pnls),
        }

    # ==================================================================
    # Utility
    # ==================================================================

    def invalidate_cache(self):
        """Force a cache refresh on the next get_memory_features() call."""
        self._cache_valid = False

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("TradingMemory connection closed.")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
