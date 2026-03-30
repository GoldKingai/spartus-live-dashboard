"""Strategy Memory — Layer 4 of the anti-forgetting system.

Explicit declarative memory: a JSON database mapping market regimes to the
strategies the model successfully used. Like a human trader's strategy book.

When the model encounters a regime it recognizes, we can check whether it's
still performing at historical levels. If performance degrades, we know the
model is forgetting.

Regime classification uses 3 dimensions:
    volatility_bucket: LOW / NORMAL / HIGH / EXTREME
    trend_bucket:      STRONG_UP / WEAK_UP / RANGING / WEAK_DOWN / STRONG_DOWN
    session_bucket:    ASIA / LONDON / NY / OVERLAP / OFF_HOURS

This gives up to 4 * 5 * 5 = 100 possible regimes, but in practice only
8-15 are frequently observed.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


class RegimeClassifier:
    """Classifies market conditions into discrete regime buckets."""

    @staticmethod
    def classify(
        atr_ratio: float,      # Current ATR / 200-bar ATR baseline
        trend_strength: float, # Signed trend: positive = up, negative = down [-1, 1]
        hour_utc: int,         # UTC hour 0-23
    ) -> str:
        """Return a regime key string like 'HIGH_STRONG_UP_LONDON'."""
        # Volatility bucket
        if atr_ratio < 0.5:
            vol = "LOW"
        elif atr_ratio < 1.0:
            vol = "NORMAL"
        elif atr_ratio < 2.0:
            vol = "HIGH"
        else:
            vol = "EXTREME"

        # Trend bucket
        if trend_strength > 0.5:
            trend = "STRONG_UP"
        elif trend_strength > 0.15:
            trend = "WEAK_UP"
        elif trend_strength < -0.5:
            trend = "STRONG_DOWN"
        elif trend_strength < -0.15:
            trend = "WEAK_DOWN"
        else:
            trend = "RANGING"

        # Session bucket
        if 7 <= hour_utc < 9:
            session = "LONDON_OPEN"
        elif 9 <= hour_utc < 13:
            session = "LONDON"
        elif 13 <= hour_utc < 17:
            session = "OVERLAP"
        elif 17 <= hour_utc < 22:
            session = "NY"
        elif 22 <= hour_utc or hour_utc < 7:
            session = "ASIA_OFF"
        else:
            session = "TRANSITION"

        return f"{vol}_{trend}_{session}"

    @staticmethod
    def classify_from_features(features: Dict) -> str:
        """Classify regime from a features dict (e.g., last row of features_df)."""
        atr_ratio = float(features.get("atr_ratio", 1.0))
        # Use h1_trend_dir as trend proxy [-1, 0, 1]
        trend = float(features.get("h1_trend_dir", 0.0))
        # Use hour from hour_sin/cos: hour = atan2(sin, cos) * 12 / pi + 12
        h_sin = float(features.get("hour_sin", 0.0))
        h_cos = float(features.get("hour_cos", 1.0))
        hour = int((np.arctan2(h_sin, h_cos) * 12 / np.pi + 12) % 24)
        return RegimeClassifier.classify(atr_ratio, trend, hour)


class StrategyMemory:
    """Persistent strategy database mapping regimes to model performance.

    JSON-backed. Survives process restarts. Never modified by gradient updates.

    Schema per regime entry:
    {
        "regime_key": "HIGH_STRONG_UP_LONDON",
        "total_trades": 247,
        "wins": 152,
        "total_pnl": 184.32,
        "gross_profit": 341.20,
        "gross_loss": 156.88,
        "avg_conviction": 0.48,
        "avg_hold_bars": 7.3,
        "historical_win_rate": 0.615,
        "historical_pf": 2.17,
        "last_seen": 1710000000.0,
        "last_updated": 1710000000.0,
        "source": "training",   # "training" or "finetune"
        "finetune_trades": 0,
        "finetune_wins": 0,
        "finetune_pnl": 0.0
    }
    """

    def __init__(
        self,
        db_path: str = "storage/finetune/strategy_memory.json",
        forgetting_threshold: float = 0.70,
    ):
        self._db_path = Path(db_path)
        self._forgetting_threshold = forgetting_threshold
        self._db: Dict[str, Dict] = {}
        self._load()

    # ─────────────────────────────────────────────────────────────────────────
    # Recording Trades
    # ─────────────────────────────────────────────────────────────────────────

    def record_trade(
        self,
        regime_key: str,
        pnl: float,
        conviction: float,
        hold_bars: int,
        source: str = "training",  # "training" or "finetune"
    ) -> None:
        """Record a completed trade for a given regime."""
        if regime_key not in self._db:
            self._db[regime_key] = self._new_entry(regime_key)

        entry = self._db[regime_key]
        won = pnl > 0

        # Update aggregate stats
        entry["total_trades"] += 1
        if won:
            entry["wins"] += 1
            entry["gross_profit"] += pnl
        else:
            entry["gross_loss"] += abs(pnl)
        entry["total_pnl"] += pnl

        # Running averages
        n = entry["total_trades"]
        entry["avg_conviction"] = (
            (entry["avg_conviction"] * (n - 1) + conviction) / n
        )
        entry["avg_hold_bars"] = (
            (entry["avg_hold_bars"] * (n - 1) + hold_bars) / n
        )
        entry["last_seen"] = time.time()
        entry["last_updated"] = time.time()
        entry["source"] = source

        # Update historical baseline after enough training trades
        if source == "training" and entry["total_trades"] >= 10:
            wr = entry["wins"] / entry["total_trades"]
            gl = entry["gross_loss"]
            pf = entry["gross_profit"] / gl if gl > 0 else 999.0
            entry["historical_win_rate"] = wr
            entry["historical_pf"] = min(pf, 99.0)

        # Track fine-tune performance separately
        if source == "finetune":
            entry["finetune_trades"] += 1
            if won:
                entry["finetune_wins"] += 1
            entry["finetune_pnl"] += pnl

        self._save_debounced()

    def bulk_record_from_training(self, trades: List[Dict]) -> int:
        """Bulk-ingest completed trades from training history.

        Args:
            trades: List of trade dicts with keys: regime, pnl, conviction,
                    hold_bars (all optional with fallbacks).

        Returns:
            Number of trades recorded.
        """
        n = 0
        for t in trades:
            regime = t.get("regime", "UNKNOWN")
            if not regime:
                continue
            self.record_trade(
                regime_key=regime,
                pnl=float(t.get("pnl", 0.0)),
                conviction=float(t.get("conviction", 0.5)),
                hold_bars=int(t.get("hold_bars", 6)),
                source="training",
            )
            n += 1
        self._save()
        return n

    # ─────────────────────────────────────────────────────────────────────────
    # Forgetting Detection
    # ─────────────────────────────────────────────────────────────────────────

    def check_forgetting(self, min_finetune_trades: int = 10) -> List[Dict]:
        """Check if fine-tuning has degraded performance on any known regime.

        Returns:
            List of forgetting alert dicts with keys:
            {regime, historical_wr, current_wr, degradation_pct, severity}
        """
        alerts = []

        for key, entry in self._db.items():
            ft_trades = entry.get("finetune_trades", 0)
            hist_wr = entry.get("historical_win_rate", 0.0)

            # Skip if no historical baseline or insufficient fine-tune data
            if hist_wr <= 0 or ft_trades < min_finetune_trades:
                continue

            ft_wins = entry.get("finetune_wins", 0)
            current_wr = ft_wins / ft_trades

            # Check if current performance is below threshold * historical
            if current_wr < hist_wr * self._forgetting_threshold:
                degradation = 1.0 - (current_wr / hist_wr) if hist_wr > 0 else 1.0
                severity = "CRITICAL" if degradation > 0.5 else "WARNING"
                alerts.append({
                    "regime": key,
                    "historical_wr": round(hist_wr, 3),
                    "current_wr": round(current_wr, 3),
                    "degradation_pct": round(degradation * 100, 1),
                    "finetune_trades": ft_trades,
                    "severity": severity,
                })

        return sorted(alerts, key=lambda a: a["degradation_pct"], reverse=True)

    def get_active_regime_performance(self, regime_key: str) -> Optional[Dict]:
        """Get performance stats for the current active regime."""
        return self._db.get(regime_key)

    # ─────────────────────────────────────────────────────────────────────────
    # Dashboard Summary
    # ─────────────────────────────────────────────────────────────────────────

    def get_summary(self) -> Dict:
        """Return summary for dashboard display."""
        regimes = list(self._db.keys())
        profitable = sum(
            1 for e in self._db.values()
            if e.get("historical_win_rate", 0) > 0.50 and e.get("total_trades", 0) >= 10
        )
        alerts = self.check_forgetting()

        # Most recently active regime
        active = None
        if self._db:
            active = max(self._db, key=lambda k: self._db[k].get("last_seen", 0))

        return {
            "known_regimes": len(regimes),
            "profitable_regimes": profitable,
            "forgetting_alerts": len(alerts),
            "active_regime": active,
            "alert_details": alerts[:3],  # Top 3 most severe
        }

    def get_all_regimes(self) -> List[Dict]:
        """Return all regime entries sorted by trade count."""
        entries = []
        for key, entry in self._db.items():
            e = dict(entry)
            e["regime_key"] = key
            if e.get("total_trades", 0) > 0:
                e["win_rate"] = e.get("wins", 0) / e["total_trades"]
                gl = e.get("gross_loss", 0)
                e["profit_factor"] = (
                    e.get("gross_profit", 0) / gl if gl > 0 else 999.0
                )
            entries.append(e)
        return sorted(entries, key=lambda e: e.get("total_trades", 0), reverse=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._db_path.exists():
            try:
                with open(self._db_path, encoding="utf-8") as f:
                    self._db = json.load(f)
                log.info(f"StrategyMemory: loaded {len(self._db)} regimes from {self._db_path}")
            except Exception as e:
                log.warning(f"StrategyMemory: load failed: {e} — starting fresh")
                self._db = {}
        else:
            self._db = {}

    def _save(self) -> None:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._db_path, "w", encoding="utf-8") as f:
                json.dump(self._db, f, indent=2)
        except Exception as e:
            log.warning(f"StrategyMemory: save failed: {e}")

    _last_save_time: float = 0.0

    def _save_debounced(self, interval_s: float = 60.0) -> None:
        """Save at most once per interval_s to avoid disk thrash."""
        now = time.time()
        if now - self._last_save_time > interval_s:
            self._save()
            self._last_save_time = now

    @staticmethod
    def _new_entry(regime_key: str) -> Dict:
        return {
            "regime_key": regime_key,
            "total_trades": 0,
            "wins": 0,
            "total_pnl": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "avg_conviction": 0.0,
            "avg_hold_bars": 0.0,
            "historical_win_rate": 0.0,
            "historical_pf": 0.0,
            "last_seen": time.time(),
            "last_updated": time.time(),
            "source": "training",
            "finetune_trades": 0,
            "finetune_wins": 0,
            "finetune_pnl": 0.0,
        }
