"""Predict -> verify cycle for trend prediction accuracy.

Wraps TradingMemory's prediction storage with a threshold filter
(only records predictions where |direction| > 0.3) and a convenient
verify-pending interface.
"""

import logging
from typing import Optional

from memory.trading_memory import TradingMemory

logger = logging.getLogger(__name__)


class TrendTracker:
    """Predict -> verify cycle for trend prediction accuracy.

    Records agent directional predictions and later verifies them against
    actual price movement to compute an ongoing accuracy metric that feeds
    back into the observation vector as memory feature [2].
    """

    def __init__(self, memory: TradingMemory):
        """
        Parameters
        ----------
        memory : TradingMemory
            The shared memory instance that owns the SQLite database.
        """
        self.memory = memory

    def record_prediction(
        self,
        step: int,
        direction: float,
        confidence: float,
        price: float,
    ):
        """Store a trend prediction if conviction is strong enough.

        Only predictions with |direction| > 0.3 are recorded so that
        neutral / indecisive signals do not dilute the accuracy metric.

        Parameters
        ----------
        step : int
            Current step / bar index.
        direction : float
            Predicted direction -- positive = bullish, negative = bearish.
        confidence : float
            Agent conviction in [0, 1].
        price : float
            Market price at the moment of prediction.
        """
        if abs(direction) <= 0.3:
            return  # Too weak -- do not pollute accuracy metric

        self.memory.record_prediction(step, direction, confidence, price)
        logger.debug(
            "TrendTracker: recorded prediction step=%d dir=%.3f conf=%.3f price=%.2f",
            step, direction, confidence, price,
        )

    def verify_pending(
        self,
        step: int,
        current_price: float,
        lookforward: int = 20,
    ):
        """Verify predictions that are at least *lookforward* bars old.

        Delegates to TradingMemory.verify_predictions() which marks each
        pending prediction as correct / incorrect based on actual price
        movement.

        Parameters
        ----------
        step : int
            Current step / bar index.
        current_price : float
            Current market price used to determine actual direction.
        lookforward : int
            Minimum number of bars a prediction must be old before it
            is eligible for verification (default 20).
        """
        self.memory.verify_predictions(step, current_price, lookforward=lookforward)
