"""Trend prediction tracking and verification.

Wraps the prediction-verification cycle:
- When |direction| > threshold, record a prediction
- After lookforward bars, verify if the prediction was correct
- Provides trend_prediction_accuracy feature (#40)
"""

import numpy as np
from typing import Optional

from src.memory.trading_memory import TradingMemory
from src.config import TrainingConfig


class TrendTracker:
    """Manages the trend prediction -> verification cycle.

    Predictions are stored when the agent's action indicates a strong
    directional view (|direction| > threshold). After a lookforward period,
    we check if the prediction was correct.

    Performance: verification and storage are throttled to avoid hitting
    SQLite on every single environment step.
    """

    def __init__(
        self,
        memory: TradingMemory,
        config: TrainingConfig = None,
    ):
        self.memory = memory
        cfg = config or TrainingConfig()
        self.direction_threshold = cfg.direction_threshold  # 0.3
        self.lookforward = 20  # Bars to look forward for verification
        self._current_week = 0
        self._verify_interval = 10  # Only verify every N steps
        self._store_interval = 5   # Only store predictions every N strong-signal steps

    def set_week(self, week: int):
        """Set current training week for prediction recording."""
        self._current_week = week

    def on_step(
        self,
        step: int,
        action_direction: float,
        current_price: float,
    ):
        """Called every environment step.

        1. Verify any predictions that are due (throttled)
        2. If agent has a strong directional view, store a new prediction (throttled)
        """
        # Verify old predictions (throttled — every N steps)
        if step % self._verify_interval == 0:
            self.memory.verify_predictions(step, current_price)

        # Store new prediction if conviction is strong enough (throttled)
        # Skip first 10 steps — agent hasn't seen enough bars for a meaningful prediction
        if step >= 10 and abs(action_direction) > self.direction_threshold:
            if step % self._store_interval == 0:
                self.memory.store_prediction(
                    week=self._current_week,
                    step=step,
                    predicted_direction=np.sign(action_direction),
                    confidence=abs(action_direction),
                    current_price=current_price,
                    lookforward=self.lookforward,
                )

    def get_accuracy(self) -> float:
        """Get current trend prediction accuracy (feature #40)."""
        return self.memory._get_trend_accuracy()
