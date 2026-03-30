"""Feature engineering pipeline: 67 features from raw OHLCV data.

Groups A-F (29 base features) + Upgrades 1-5 (25 new features) = 54 precomputed.
Groups G-H (account + memory, 13 features) are computed live in the environment.

Uses the `ta` library for technical indicators.
Implements fractional differentiation manually (fracdiff package unavailable).
"""

import numpy as np
import pandas as pd
import ta
from pathlib import Path
from typing import Dict, Optional

from src.config import TrainingConfig
from src.data.correlation_features import calc_correlation_features
from src.data.calendar_features import calc_calendar_features, load_calendar
from src.data.regime_features import calc_regime_features
from src.data.session_features import calc_session_features


# ---------------------------------------------------------------------------
# Fractional differentiation (manual implementation)
# ---------------------------------------------------------------------------

def _get_frac_diff_weights(d: float, threshold: float = 1e-5, max_width: int = 200) -> np.ndarray:
    """Compute fractional differentiation weights using the expanding window method.

    The weights follow: w_k = -w_{k-1} * (d - k + 1) / k
    We truncate when |w_k| < threshold or width exceeds max_width.
    """
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
        if k >= max_width:
            break
    return np.array(weights[::-1])  # Oldest weight first


def frac_diff(series: pd.Series, d: float = 0.35, threshold: float = 1e-5,
              max_width: int = 200) -> pd.Series:
    """Apply fractional differentiation to a price series.

    Args:
        series: Price series (e.g., close prices).
        d: Differentiation order (0 < d < 1). 0.35 is default for XAUUSD.
        threshold: Weight cutoff for truncation.

    Returns:
        Fractionally differentiated series (NaN where insufficient history).
    """
    weights = _get_frac_diff_weights(d, threshold, max_width)
    width = len(weights)
    result = pd.Series(index=series.index, dtype=np.float64)

    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1: i + 1].values
        result.iloc[i] = np.dot(weights, window)

    return result


# ---------------------------------------------------------------------------
# Session quality mapping
# ---------------------------------------------------------------------------

def _session_quality(hour: int) -> float:
    """Map UTC hour to trading session quality score [0, 1]."""
    if 8 <= hour < 12:
        return 1.0    # London AM (best liquidity)
    elif 13 <= hour < 17:
        return 0.95   # NY overlap
    elif 12 <= hour < 13:
        return 0.9    # London PM
    elif 17 <= hour < 20:
        return 0.7    # NY PM
    elif 0 <= hour < 8:
        return 0.4    # Asia
    else:
        return 0.2    # Off hours


# ---------------------------------------------------------------------------
# Spread estimate mapping (Upgrade 3)
# ---------------------------------------------------------------------------

def _session_spread(hour: int) -> float:
    """Map UTC hour to estimated spread in pips (matches MarketSimulator)."""
    if 8 <= hour < 12:
        return 1.5    # London AM
    elif 13 <= hour < 17:
        return 2.0    # NY overlap
    elif 12 <= hour < 13:
        return 1.8    # London PM
    elif 17 <= hour < 20:
        return 2.5    # NY PM
    elif 0 <= hour < 8:
        return 3.0    # Asia
    else:
        return 5.0    # Off hours


# ---------------------------------------------------------------------------
# FeatureBuilder
# ---------------------------------------------------------------------------

class FeatureBuilder:
    """Computes 54 precomputed features from raw OHLCV + correlated data.

    Original Groups A-F (29 features) + Upgrades 1-5 (25 new features).
    Groups G-H (account + memory, 13 features) are added live by the environment.
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.cfg = config or TrainingConfig()
        self._calendar_df = None  # Loaded lazily

    # === Public API =========================================================

    def build_features(
        self,
        m5: pd.DataFrame,
        h1: pd.DataFrame,
        h4: pd.DataFrame,
        d1: pd.DataFrame,
        correlated_m5: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Compute all pre-computable features from multi-timeframe OHLCV.

        Args:
            m5: M5 OHLCV with columns [time, open, high, low, close, volume].
            h1: H1 OHLCV (same columns).
            h4: H4 OHLCV (same columns).
            d1: D1 OHLCV (same columns).
            correlated_m5: Dict mapping symbol name to M5 OHLCV DataFrames
                for correlated instruments (EURUSD, XAGUSD, USDJPY, US500, USOIL).
                None if correlated data not available.

        Returns:
            DataFrame indexed like m5 with 54 feature columns + time/OHLCV.
            First ~200 rows will have NaN (warmup period).
        """
        m5 = m5.copy().sort_values("time").reset_index(drop=True)
        h1 = h1.copy().sort_values("time").reset_index(drop=True)
        h4 = h4.copy().sort_values("time").reset_index(drop=True)
        d1 = d1.copy().sort_values("time").reset_index(drop=True)

        close = m5["close"]
        high = m5["high"]
        low = m5["low"]
        opn = m5["open"]
        vol = m5["volume"]

        # Pre-compute ATR(14) — used by multiple groups
        atr_14 = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()

        # Group A: Price & Returns (7)
        feats = self._calc_price_returns(close, high, low, opn)
        # Group B: Volatility (4)
        feats = pd.concat([feats, self._calc_volatility(close, high, low, atr_14)], axis=1)
        # Group C: Momentum & Trend (6)
        feats = pd.concat([feats, self._calc_momentum_trend(close, high, low, atr_14)], axis=1)
        # Group D: Volume (2)
        feats = pd.concat([feats, self._calc_volume(close, vol)], axis=1)
        # Group E: Multi-timeframe context (6)
        htf_feats = self._calc_mtf_context(m5, h1, h4, d1)
        feats = pd.concat([feats, htf_feats], axis=1)
        # Group F: Time & Session (4)
        feats = pd.concat([feats, self._calc_time_features(m5["time"])], axis=1)

        # === Upgrade 1: Correlated Instruments (11 features) ===
        _corr_cols = ("eurusd_returns_20", "eurusd_rsi_14", "eurusd_trend",
                      "xagusd_returns_20", "xagusd_rsi_14",
                      "usdjpy_returns_20", "usdjpy_trend",
                      "us500_returns_20", "us500_rsi_14",
                      "usoil_returns_20", "gold_silver_ratio_z")
        # Treat correlated_m5 as absent if ALL values are empty DataFrames
        _corr_has_data = correlated_m5 and any(
            not df.empty for df in correlated_m5.values()
        )
        if _corr_has_data:
            correlated_m5 = correlated_m5  # has real data
        else:
            correlated_m5 = None  # no data — skip and suppress validation warning
        if correlated_m5:
            corr_feats = calc_correlation_features(m5, correlated_m5, self.cfg)
            feats = pd.concat([feats, corr_feats], axis=1)
        # Fill any missing columns with zeros (e.g. USOIL unavailable)
        for col in _corr_cols:
            if col not in feats.columns:
                feats[col] = 0.0

        # === Upgrade 2: Economic Calendar & Events (6 features) ===
        calendar_df = self._get_calendar()
        cal_feats = calc_calendar_features(m5, calendar_df, self.cfg)
        feats = pd.concat([feats, cal_feats], axis=1)

        # === Upgrade 3: Spread & Liquidity (2 features, inline) ===
        feats = pd.concat([feats, self._calc_spread_liquidity(m5, atr_14, vol)], axis=1)

        # === Upgrade 4: Regime Detection (2 features) ===
        if correlated_m5:
            regime_feats = calc_regime_features(
                m5,
                correlated_m5.get("EURUSD"),
                correlated_m5.get("US500"),
                self.cfg,
            )
            feats = pd.concat([feats, regime_feats], axis=1)
        else:
            feats["corr_gold_usd_100"] = 0.0
            feats["corr_gold_spx_100"] = 0.0

        # === Upgrade 5: Session Microstructure (4 features) ===
        session_feats = calc_session_features(m5, self.cfg)
        feats = pd.concat([feats, session_feats], axis=1)

        # Attach raw OHLCV + time for the environment
        result = pd.concat([
            m5[["time", "open", "high", "low", "close", "volume"]],
            feats,
        ], axis=1)

        # Also keep ATR for the environment (SL/TP sizing)
        result["atr_14_raw"] = atr_14

        # Feature health validation
        # Optional columns are expected to be 0 when their data source is absent.
        # Only warn about CONSTANT for those columns when the data WAS provided.
        # Suppress per-symbol: a symbol with an empty DataFrame should not fire warnings
        # even when other correlated symbols do have data.
        _sym_col_map = {
            "EURUSD":  ("eurusd_returns_20", "eurusd_rsi_14", "eurusd_trend"),
            "XAGUSD":  ("xagusd_returns_20", "xagusd_rsi_14", "gold_silver_ratio_z"),
            "USDJPY":  ("usdjpy_returns_20", "usdjpy_trend"),
            "US500":   ("us500_returns_20", "us500_rsi_14", "corr_gold_spx_100"),
            "USOIL":   ("usoil_returns_20",),
        }
        optional_absent = set()
        if not correlated_m5:
            # No correlated data at all — suppress everything
            optional_absent.update(_corr_cols)
            optional_absent.update(("corr_gold_usd_100", "corr_gold_spx_100"))
        else:
            # Partial data — suppress only columns for symbols that are absent/empty
            for sym, cols in _sym_col_map.items():
                sym_df = correlated_m5.get(sym)
                if sym_df is None or (hasattr(sym_df, "empty") and sym_df.empty):
                    optional_absent.update(cols)
            # corr_gold_usd_100 depends on EURUSD; already covered above if EURUSD absent
            if correlated_m5.get("EURUSD") is None or (
                hasattr(correlated_m5.get("EURUSD"), "empty") and correlated_m5["EURUSD"].empty
            ):
                optional_absent.add("corr_gold_usd_100")
        if calendar_df is None or calendar_df.empty:
            optional_absent.update((
                "hours_to_next_high_impact", "hours_to_next_nfp_fomc",
                "in_event_window", "daily_event_density",
                "is_london_fix", "is_comex_open",
            ))
        # D1 features can be genuinely constant when the data window covers a single
        # sustained trend (common in live fine-tuning with limited D1 history).
        # Suppress the constant warning when D1 data is sparse (< 5 real D1 bars).
        if len(d1) < 5:
            optional_absent.update(("d1_trend_dir", "d1_range_pct", "d1_momentum"))
        self._validate_features(result, skip_constant=optional_absent)

        return result

    # === Feature Health Validation ============================================

    def _validate_features(
        self,
        df: pd.DataFrame,
        skip_constant: set = None,
    ) -> None:
        """Check for NaN, constant, or infinite values in feature columns.

        Args:
            skip_constant: Column names to exclude from the CONSTANT check.
                Used for optional feature groups (correlated instruments, calendar)
                that are intentionally zero when their data source is absent.
        """
        import warnings

        skip_cols = {"time", "open", "high", "low", "close", "volume", "atr_14_raw"}
        skip_constant = skip_constant or set()
        feature_cols = [c for c in df.columns if c not in skip_cols and not c.endswith("_raw")]

        # Only check after warmup region — first `lookback` bars are expected to have NaN
        start = min(self.cfg.lookback, len(df))
        check = df.iloc[start:]
        if check.empty or len(check) < 50:
            return

        issues = []
        for col in feature_cols:
            if col not in check.columns:
                continue
            s = check[col]
            nan_pct = s.isna().mean()
            if nan_pct > 0.05:
                issues.append(f"NaN: {col} ({nan_pct:.0%})")
            if col not in skip_constant and s.std() < 1e-10 and len(s) > 50:
                issues.append(f"CONSTANT: {col} (val={s.iloc[0]:.4f})")
            if s.dtype in (np.float32, np.float64) and np.isinf(s).any():
                issues.append(f"INF: {col}")

        if issues:
            warnings.warn(
                f"Feature health ({len(issues)} issues): {'; '.join(issues[:10])}",
                RuntimeWarning,
                stacklevel=2,
            )

    # === Group A: Price & Returns (7 features) ==============================

    def _calc_price_returns(
        self, close: pd.Series, high: pd.Series, low: pd.Series, opn: pd.Series
    ) -> pd.DataFrame:
        return pd.DataFrame({
            # #1: Fractionally differentiated close
            "close_frac_diff": frac_diff(close, d=self.cfg.frac_diff_d),
            # #2-4: Log returns at different horizons
            "returns_1bar": np.log(close / close.shift(1)),
            "returns_5bar": np.log(close / close.shift(5)),
            "returns_20bar": np.log(close / close.shift(20)),
            # #5: Bar range as % of close
            "bar_range": (high - low) / close,
            # #6: Where close sits within bar [0, 1]
            "close_position": (close - low) / (high - low + 1e-8),
            # #7: Candle body strength [0, 1]
            "body_ratio": (close - opn).abs() / (high - low + 1e-8),
        })

    # === Group B: Volatility (4 features) ===================================

    def _calc_volatility(
        self, close: pd.Series, high: pd.Series, low: pd.Series,
        atr_14: pd.Series,
    ) -> pd.DataFrame:
        # ATR(7) and ATR(21) for the ratio
        atr_7 = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=7
        ).average_true_range()
        atr_21 = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=21
        ).average_true_range()

        # Bollinger Bands (20, 2)
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)

        return pd.DataFrame({
            # #8: ATR normalized by price
            "atr_14_norm": atr_14 / close * 100,
            # #9: Volatility expansion/contraction
            "atr_ratio": (atr_7 / (atr_21 + 1e-8)).clip(0.0, 10.0),
            # #10: BB width
            "bb_width": (bb.bollinger_hband() - bb.bollinger_lband()) / (bb.bollinger_mavg() + 1e-8),
            # #11: Position within bands [0, 1]
            "bb_position": (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-8),
        })

    # === Group C: Momentum & Trend (6 features) ============================

    def _calc_momentum_trend(
        self, close: pd.Series, high: pd.Series, low: pd.Series,
        atr_14: pd.Series,
    ) -> pd.DataFrame:
        # RSI(14)
        rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()

        # MACD (12, 26, 9)
        macd_ind = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        macd_hist = macd_ind.macd_diff()  # histogram

        # ADX(14)
        adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()

        # EMAs
        ema_20 = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
        ema_50 = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
        ema_200 = ta.trend.EMAIndicator(close=close, window=200).ema_indicator()

        # Stochastic %K
        stoch = ta.momentum.StochasticOscillator(
            high=high, low=low, close=close,
            window=14, smooth_window=3,
        )

        return pd.DataFrame({
            # #12: RSI scaled to [0, 1]
            "rsi_14": rsi / 100.0,
            # #13: MACD histogram scaled by volatility
            "macd_signal": macd_hist / (atr_14 + 1e-8),
            # #14: ADX scaled to [0, 1]
            "adx_14": adx / 100.0,
            # #15: EMA crossover signal
            "ema_cross": (ema_20 - ema_50) / (atr_14 + 1e-8),
            # #16: Long-term trend position
            "price_vs_ema200": (close - ema_200) / (atr_14 + 1e-8),
            # #17: Stochastic %K scaled to [0, 1]
            "stoch_k": stoch.stoch() / 100.0,
        })

    # === Group D: Volume (2 features) =======================================

    def _calc_volume(self, close: pd.Series, volume: pd.Series) -> pd.DataFrame:
        # Volume relative to 20-bar SMA
        vol_sma_20 = volume.rolling(20).mean()

        # OBV
        obv = ta.volume.OnBalanceVolumeIndicator(
            close=close, volume=volume
        ).on_balance_volume()

        # OBV slope: linear regression over 10 bars, normalized by 50-bar std
        obv_slope = obv.rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0,
            raw=False,
        )
        obv_slope_std = obv_slope.rolling(50).std()
        obv_slope_norm = obv_slope / (obv_slope_std + 1e-8)

        return pd.DataFrame({
            # #18: Volume ratio
            "volume_ratio": volume / (vol_sma_20 + 1e-8),
            # #19: OBV slope (normalized)
            "obv_slope": obv_slope_norm,
        })

    # === Group E: Multi-Timeframe Context (6 features) ======================

    def _calc_mtf_context(
        self,
        m5: pd.DataFrame,
        h1: pd.DataFrame,
        h4: pd.DataFrame,
        d1: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute HTF trend/momentum and merge into M5 timeline.

        Uses merge_asof(direction='backward') to prevent look-ahead bias.
        """
        # Compute HTF features on each timeframe
        h1_feats = self._htf_features(h1, prefix="h1")
        h4_feats = self._htf_features(h4, prefix="h4")
        d1_feats = self._htf_features(d1, prefix="d1")

        # H4 MACD for htf_momentum
        h4_close = h4["close"]
        h4_atr = ta.volatility.AverageTrueRange(
            high=h4["high"], low=h4["low"], close=h4_close, window=14
        ).average_true_range()
        h4_macd = ta.trend.MACD(
            close=h4_close, window_slow=26, window_fast=12, window_sign=9
        ).macd_diff()
        h4_momentum = (h4_macd / (h4_atr + 1e-8)).clip(-3, 3)

        # H1 RSI
        h1_rsi_raw = ta.momentum.RSIIndicator(close=h1["close"], window=14).rsi() / 100.0

        # Build HTF DataFrames for merge
        h1_merge = pd.DataFrame({
            "time": h1["time"],
            "h1_trend_dir": h1_feats,
            "h1_rsi": h1_rsi_raw,
        })
        h4_merge = pd.DataFrame({
            "time": h4["time"],
            "h4_trend_dir": h4_feats,
            "htf_momentum": h4_momentum,
        })
        d1_merge = pd.DataFrame({
            "time": d1["time"],
            "d1_trend_dir": d1_feats,
        })

        # CRITICAL: merge_asof with direction='backward' — no look-ahead bias
        m5_time = m5[["time"]].copy()

        result = pd.merge_asof(
            m5_time.sort_values("time"),
            h1_merge.sort_values("time"),
            on="time",
            direction="backward",
        )
        result = pd.merge_asof(
            result.sort_values("time"),
            h4_merge.sort_values("time"),
            on="time",
            direction="backward",
        )
        result = pd.merge_asof(
            result.sort_values("time"),
            d1_merge.sort_values("time"),
            on="time",
            direction="backward",
        )

        # Forward-fill HTF features (NaN at start means no prior HTF bar)
        htf_cols = ["h1_trend_dir", "h4_trend_dir", "d1_trend_dir",
                     "h1_rsi", "htf_momentum"]
        for col in htf_cols:
            result[col] = result[col].ffill().fillna(0.0)

        # #24: MTF alignment — average of trend directions clipped to [-1, 1]
        result["mtf_alignment"] = (
            (result["h1_trend_dir"]
             + result["h4_trend_dir"]
             + result["d1_trend_dir"]) / 3.0
        ).clip(-1, 1)

        # Return only feature columns (drop time)
        return result[["h1_trend_dir", "h4_trend_dir", "d1_trend_dir",
                        "h1_rsi", "mtf_alignment", "htf_momentum"]].reset_index(drop=True)

    def _htf_features(self, htf: pd.DataFrame, prefix: str) -> pd.Series:
        """Compute trend direction for a higher-timeframe DataFrame.

        Returns: Series of trend direction values clipped to [-3, 3].
        """
        ema_20 = ta.trend.EMAIndicator(close=htf["close"], window=20).ema_indicator()
        # Slope over 3 bars, normalized by price
        slope = ema_20.diff(3) / (htf["close"] * 0.01 + 1e-8)
        return slope.clip(-3, 3)

    # === Group F: Time & Session (4 features) ===============================

    def _calc_time_features(self, time_series: pd.Series) -> pd.DataFrame:
        """Time/session features — EXEMPT from normalization."""
        hour = time_series.dt.hour
        day = time_series.dt.dayofweek  # Mon=0, Fri=4

        return pd.DataFrame({
            # #26: Cyclical hour encoding (sin)
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            # #27: Cyclical hour encoding (cos)
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            # #28: Day of week [0, 1]
            "day_of_week": day / 4.0,
            # #29: Session quality [0, 1]
            "session_quality": hour.apply(_session_quality),
        })

    # === Upgrade 3: Spread & Liquidity (2 features, inline) =================

    def _calc_spread_liquidity(
        self, m5: pd.DataFrame, atr_14: pd.Series, volume: pd.Series,
    ) -> pd.DataFrame:
        """Compute spread estimate and volume spike features."""
        hour = m5["time"].dt.hour

        # Spread estimate based on session (same logic as MarketSimulator)
        spread_pips = hour.map(lambda h: _session_spread(h))
        spread_price = spread_pips * self.cfg.pip_price  # Convert pips to price
        spread_norm = (spread_price / (atr_14 + 1e-8)).clip(0.0, 5.0)

        # Volume spike: current / SMA(20), capped at 5x, normalized to [0,1]
        vol_sma_20 = volume.rolling(20, min_periods=1).mean()
        vol_ratio = volume / (vol_sma_20 + 1e-8)
        volume_spike = (vol_ratio / 5.0).clip(0.0, 1.0)

        return pd.DataFrame({
            "spread_estimate_norm": spread_norm,
            "volume_spike": volume_spike,
        })

    # === Calendar Loading ===================================================

    def _get_calendar(self) -> Optional[pd.DataFrame]:
        """Load economic calendar, cached after first load."""
        if self._calendar_df is not None:
            return self._calendar_df

        csv_path = self.cfg.calendar_csv_path
        if csv_path.exists():
            self._calendar_df = load_calendar(csv_path)
            return self._calendar_df

        return None  # No calendar data — features default to neutral

    # === Feature Caching ====================================================

    def build_and_cache(
        self,
        m5: pd.DataFrame,
        h1: pd.DataFrame,
        h4: pd.DataFrame,
        d1: pd.DataFrame,
        cache_path: Path,
        correlated_m5: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Build features and save to Parquet cache.

        If cache_path exists and is newer than source data, load from cache.
        """
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        features = self.build_features(m5, h1, h4, d1, correlated_m5=correlated_m5)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(cache_path, engine="pyarrow", index=False)
        return features
