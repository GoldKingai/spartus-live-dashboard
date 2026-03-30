"""Data Validator - Validate OHLCV data quality before and after cleaning."""

import numpy as np
import pandas as pd


class DataValidator:
    """Validate OHLCV DataFrames against quality rules.

    Checks:
    1. OHLC integrity (high >= max(O,C), low <= min(O,C))
    2. No negative/zero prices
    3. No NaN values in OHLCV columns
    4. No duplicate timestamps
    5. Sorted ascending by time
    6. No price spikes > max_spike_pct in one bar
    7. Volume > 0 (allow up to zero_vol_tolerance % zero)
    8. M5 minimum bars per week (min_bars_per_week)
    """

    OHLCV_COLS = ["open", "high", "low", "close", "volume"]

    def __init__(
        self,
        max_spike_pct: float = 0.03,
        zero_vol_tolerance: float = 0.10,
        min_bars_per_week: int = 1000,
    ):
        self.max_spike_pct = max_spike_pct
        self.zero_vol_tolerance = zero_vol_tolerance
        self.min_bars_per_week = min_bars_per_week

    def validate(self, df: pd.DataFrame, timeframe: str = "M5") -> dict:
        """Run all validation checks on a DataFrame.

        Args:
            df: DataFrame with columns [time, open, high, low, close, volume].
            timeframe: Timeframe label for context in report.

        Returns:
            Dict with keys: passed (bool), checks (list of dicts), summary (str).
        """
        if df.empty:
            return {
                "passed": False,
                "checks": [{"name": "empty_data", "passed": False, "detail": "DataFrame is empty"}],
                "summary": "FAIL: No data to validate",
            }

        checks = []
        checks.append(self._check_columns(df))
        if not checks[-1]["passed"]:
            return {
                "passed": False,
                "checks": checks,
                "summary": f"FAIL: Missing columns - {checks[-1]['detail']}",
            }

        checks.append(self._check_ohlc_integrity(df))
        checks.append(self._check_no_negative_zero(df))
        checks.append(self._check_no_nans(df))
        checks.append(self._check_no_duplicates(df))
        checks.append(self._check_sorted(df))
        checks.append(self._check_spikes(df))
        checks.append(self._check_volume(df))

        if timeframe == "M5":
            checks.append(self._check_bar_count(df))

        all_passed = all(c["passed"] for c in checks)
        failed = [c["name"] for c in checks if not c["passed"]]
        summary = "PASS: All checks passed" if all_passed else f"FAIL: {', '.join(failed)}"

        return {"passed": all_passed, "checks": checks, "summary": summary}

    def _check_columns(self, df: pd.DataFrame) -> dict:
        required = ["time"] + self.OHLCV_COLS
        missing = [c for c in required if c not in df.columns]
        return {
            "name": "columns",
            "passed": len(missing) == 0,
            "detail": f"Missing: {missing}" if missing else "All required columns present",
        }

    def _check_ohlc_integrity(self, df: pd.DataFrame) -> dict:
        high_ok = df["high"] >= df[["open", "close"]].max(axis=1)
        low_ok = df["low"] <= df[["open", "close"]].min(axis=1)
        violations = (~high_ok | ~low_ok).sum()
        return {
            "name": "ohlc_integrity",
            "passed": violations == 0,
            "detail": f"{violations} OHLC integrity violations" if violations else "OK",
            "count": int(violations),
        }

    def _check_no_negative_zero(self, df: pd.DataFrame) -> dict:
        price_cols = ["open", "high", "low", "close"]
        bad = (df[price_cols] <= 0).any(axis=1).sum()
        return {
            "name": "no_negative_zero_prices",
            "passed": bad == 0,
            "detail": f"{bad} bars with zero/negative prices" if bad else "OK",
            "count": int(bad),
        }

    def _check_no_nans(self, df: pd.DataFrame) -> dict:
        nans = df[self.OHLCV_COLS].isna().any(axis=1).sum()
        return {
            "name": "no_nans",
            "passed": nans == 0,
            "detail": f"{nans} bars with NaN values" if nans else "OK",
            "count": int(nans),
        }

    def _check_no_duplicates(self, df: pd.DataFrame) -> dict:
        dupes = df["time"].duplicated().sum()
        return {
            "name": "no_duplicate_timestamps",
            "passed": dupes == 0,
            "detail": f"{dupes} duplicate timestamps" if dupes else "OK",
            "count": int(dupes),
        }

    def _check_sorted(self, df: pd.DataFrame) -> dict:
        is_sorted = df["time"].is_monotonic_increasing
        return {
            "name": "sorted_ascending",
            "passed": is_sorted,
            "detail": "OK" if is_sorted else "Data is not sorted by time",
        }

    def _check_spikes(self, df: pd.DataFrame) -> dict:
        if len(df) < 2:
            return {"name": "no_spikes", "passed": True, "detail": "Too few bars to check", "count": 0}
        pct_change = df["close"].pct_change().abs()
        spikes = (pct_change > self.max_spike_pct).sum()
        return {
            "name": "no_spikes",
            "passed": spikes == 0,
            "detail": f"{spikes} bars with >{self.max_spike_pct*100:.0f}% price spike" if spikes else "OK",
            "count": int(spikes),
        }

    def _check_volume(self, df: pd.DataFrame) -> dict:
        zero_vol = (df["volume"] <= 0).sum()
        zero_pct = zero_vol / len(df) if len(df) > 0 else 0
        passed = zero_pct <= self.zero_vol_tolerance
        return {
            "name": "volume_positive",
            "passed": passed,
            "detail": (
                f"{zero_vol} bars ({zero_pct*100:.1f}%) with zero volume "
                f"(tolerance: {self.zero_vol_tolerance*100:.0f}%)"
                if zero_vol else "OK"
            ),
            "count": int(zero_vol),
        }

    def _check_bar_count(self, df: pd.DataFrame) -> dict:
        n = len(df)
        passed = n >= self.min_bars_per_week
        return {
            "name": "min_bar_count",
            "passed": passed,
            "detail": (
                f"{n} bars (minimum: {self.min_bars_per_week})"
                if not passed else f"{n} bars - OK"
            ),
            "count": n,
        }

    def print_report(self, report: dict, label: str = ""):
        """Print a formatted validation report."""
        prefix = f"[{label}] " if label else ""
        print(f"\n{prefix}Validation: {report['summary']}")
        for check in report["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            print(f"  [{status}] {check['name']}: {check['detail']}")
