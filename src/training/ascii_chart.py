"""ASCII/Unicode balance chart and sparklines for terminal dashboard."""

import numpy as np
from typing import List


SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: List[float], width: int = 20) -> str:
    """Generate a sparkline string from a list of values."""
    if not values:
        return ""

    # Sample to width if too many values
    if len(values) > width:
        indices = np.linspace(0, len(values) - 1, width, dtype=int)
        values = [values[i] for i in indices]

    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1.0

    chars = []
    for v in values:
        idx = int((v - mn) / rng * (len(SPARK_CHARS) - 1))
        idx = max(0, min(idx, len(SPARK_CHARS) - 1))
        chars.append(SPARK_CHARS[idx])

    return "".join(chars)


def balance_chart(
    balances: List[float],
    width: int = 60,
    height: int = 12,
    title: str = "Balance",
) -> str:
    """Generate a Unicode line chart of balance over time.

    Returns a multi-line string ready for rendering.
    """
    if not balances or len(balances) < 2:
        return f"  {title}: Insufficient data"

    # Sample to width
    if len(balances) > width:
        indices = np.linspace(0, len(balances) - 1, width, dtype=int)
        data = [balances[i] for i in indices]
    else:
        data = list(balances)

    mn = min(data)
    mx = max(data)
    rng = mx - mn if mx > mn else 1.0

    lines = []
    lines.append(f"  {title} (£{data[-1]:.2f})")

    # Y-axis labels
    for row in range(height - 1, -1, -1):
        y_val = mn + (row / (height - 1)) * rng
        label = f"{y_val:>8.2f} │"

        row_chars = []
        for col_idx in range(len(data)):
            normalized = (data[col_idx] - mn) / rng * (height - 1)
            if abs(normalized - row) < 0.5:
                row_chars.append("●")
            elif normalized > row:
                row_chars.append("│" if col_idx > 0 and row > 0 else " ")
            else:
                row_chars.append(" ")

        lines.append(label + "".join(row_chars))

    # X-axis
    lines.append(" " * 10 + "└" + "─" * len(data))
    lines.append(" " * 11 + f"W1{' ' * (len(data) - 5)}W{len(balances)}")

    return "\n".join(lines)


def reward_sparkline(rewards: List[float], width: int = 30) -> str:
    """Colored sparkline for rewards (green=positive, red=negative)."""
    if not rewards:
        return ""

    # Sample to width
    if len(rewards) > width:
        indices = np.linspace(0, len(rewards) - 1, width, dtype=int)
        rewards = [rewards[i] for i in indices]

    mn = min(min(rewards), -0.1)
    mx = max(max(rewards), 0.1)
    rng = mx - mn

    chars = []
    for v in rewards:
        idx = int((v - mn) / rng * (len(SPARK_CHARS) - 1))
        idx = max(0, min(idx, len(SPARK_CHARS) - 1))
        chars.append(SPARK_CHARS[idx])

    return "".join(chars)
