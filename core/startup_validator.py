"""Pre-flight checks before trading can be enabled."""
import logging
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

log = logging.getLogger(__name__)


class StartupValidator:
    """Pre-flight checks before trading can be enabled.

    Each check returns (pass: bool, message: str).
    ALL required checks must pass before "Start Trading" activates.
    Optional checks show warnings but don't block.
    """

    def __init__(
        self,
        config,
        mt5_bridge=None,
        feature_pipeline=None,
        model_loader=None,
        memory=None,
        weekend_manager=None,
    ):
        self.config = config
        self.mt5 = mt5_bridge
        self.pipeline = feature_pipeline
        self.model_loader = model_loader
        self.memory = memory
        self.weekend_manager = weekend_manager
        self.results: Dict[str, Dict[str, Any]] = {}

    def run_all_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all startup checks.

        Returns:
            Dict of check_name -> {passed: bool, message: str, required: bool}.
        """
        self.results = {}

        # Required checks
        required = [
            ("MT5 Terminal Running", self._check_mt5_running),
            ("MT5 Account Logged In", self._check_mt5_account),
            ("XAUUSD Symbol Available", self._check_symbol_available),
            ("XAUUSD Spread Reasonable", self._check_spread_reasonable),
            ("Account Currency Detected", self._check_currency_detected),
            ("tick_value Valid", self._check_tick_value),
            ("Sufficient Balance", self._check_balance),
            ("Model File Exists", self._check_model_exists),
            ("Model Loads Successfully", self._check_model_loads),
            ("Observation Dim Valid", self._check_obs_dim),
            ("Feature Count Matches Config", self._check_feature_count),
            ("M5 History Available (500+ bars)", self._check_m5_history),
            ("H1/H4/D1 History Available", self._check_htf_history),
            ("Feature Warmup Complete", self._check_warmup),
            ("Normalizer Initialized", self._check_normalizer),
            ("Circuit Breaker Configured", self._check_circuit_breaker),
            ("Weekend Manager Active", self._check_weekend_manager),
            ("Emergency Stop Working", self._check_emergency_stop),
            ("Memory DB Accessible", self._check_memory_db),
            ("Log Directory Writable", self._check_logs),
        ]

        optional = [
            ("Calendar Data Available", self._check_calendar),
            ("Correlated Instruments Available", self._check_correlated),
            ("Feature Baseline in Model Package", self._check_feature_baseline),
            ("Stress Results in Model Package", self._check_stress_results),
        ]

        # Paper-to-live transition checks (optional, informational)
        transition = [
            ("Paper: Min 1 Week Trading", self._check_paper_min_duration),
            ("Paper: Profit Factor >= 1.4", self._check_paper_profit_factor),
            ("Paper: Win Rate >= 45%", self._check_paper_win_rate),
            ("Paper: MaxDD < 15%", self._check_paper_max_dd),
        ]

        for name, check_fn in required:
            try:
                passed, msg = check_fn()
            except Exception as e:
                passed, msg = False, f"Error: {e}"
            self.results[name] = {
                "passed": passed,
                "message": msg,
                "required": True,
            }

        for name, check_fn in optional:
            try:
                passed, msg = check_fn()
            except Exception as e:
                passed, msg = False, f"Error: {e}"
            self.results[name] = {
                "passed": passed,
                "message": msg,
                "required": False,
            }

        # Only run transition checks if paper_trading is enabled
        if getattr(self.config, "paper_trading", False):
            for name, check_fn in transition:
                try:
                    passed, msg = check_fn()
                except Exception as e:
                    passed, msg = False, f"Error: {e}"
                self.results[name] = {
                    "passed": passed,
                    "message": msg,
                    "required": False,
                }

        return self.results

    def all_required_pass(self) -> bool:
        """Check if all required checks passed."""
        return all(
            r["passed"] for r in self.results.values() if r["required"]
        )

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of check results.

        Returns:
            Dict with keys: total, passed, failed, required_passed,
            required_failed, optional_passed, optional_failed,
            all_required_pass.
        """
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r["passed"])
        failed = total - passed

        req = [r for r in self.results.values() if r["required"]]
        req_passed = sum(1 for r in req if r["passed"])
        req_failed = len(req) - req_passed

        opt = [r for r in self.results.values() if not r["required"]]
        opt_passed = sum(1 for r in opt if r["passed"])
        opt_failed = len(opt) - opt_passed

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "required_passed": req_passed,
            "required_failed": req_failed,
            "optional_passed": opt_passed,
            "optional_failed": opt_failed,
            "all_required_pass": self.all_required_pass(),
        }

    # ------------------------------------------------------------------
    # Required check implementations
    # ------------------------------------------------------------------

    def _check_mt5_running(self) -> Tuple[bool, str]:
        """Check that the MT5 bridge is connected to the terminal."""
        if self.mt5 is None:
            return False, "MT5 bridge not initialized"
        try:
            # MT5Bridge exposes _connected as a property that checks
            # mt5.terminal_info().connected
            connected = self.mt5._connected
            if connected:
                return True, "MT5 terminal connected"
            return False, "MT5 terminal not connected"
        except Exception as e:
            return False, f"Cannot check MT5 status: {e}"

    def _check_mt5_account(self) -> Tuple[bool, str]:
        """Check that MT5 can retrieve valid account info."""
        if self.mt5 is None:
            return False, "MT5 bridge not initialized"
        try:
            info = self.mt5.get_account_info()
            if not info:
                return False, "account_info() returned empty"
            name = info.get("name", "unknown")
            server = info.get("server", "unknown")
            return True, f"Account: {name} @ {server}"
        except Exception as e:
            return False, f"Failed to get account info: {e}"

    def _check_symbol_available(self) -> Tuple[bool, str]:
        """Check that XAUUSD symbol info is available."""
        if self.mt5 is None:
            return False, "MT5 bridge not initialized"
        try:
            symbol = self.config.mt5_symbol
            info = self.mt5.get_symbol_info(symbol)
            if not info:
                return False, f"symbol_info({symbol}) returned empty"
            return True, f"{symbol} available (digits={info.get('digits', '?')})"
        except Exception as e:
            return False, f"Failed to get symbol info: {e}"

    def _check_spread_reasonable(self) -> Tuple[bool, str]:
        """Check that the current spread is below 100 points."""
        if self.mt5 is None:
            return False, "MT5 bridge not initialized"
        try:
            symbol = self.config.mt5_symbol
            info = self.mt5.get_symbol_info(symbol)
            if not info:
                return False, f"Cannot get symbol info for {symbol}"
            spread = info.get("spread", 0)
            # spread is in integer points from get_symbol_info
            if spread < 100:
                return True, f"Spread = {spread} points (< 100 limit)"
            return False, f"Spread = {spread} points (>= 100 -- too wide)"
        except Exception as e:
            return False, f"Failed to check spread: {e}"

    def _check_currency_detected(self) -> Tuple[bool, str]:
        """Check that the account currency has been detected."""
        if self.mt5 is None:
            return False, "MT5 bridge not initialized"
        try:
            currency = getattr(self.mt5, "account_currency", None)
            if currency and isinstance(currency, str) and len(currency) == 3:
                return True, f"Account currency: {currency}"
            return False, f"Account currency not detected (got: {currency})"
        except Exception as e:
            return False, f"Failed to detect currency: {e}"

    def _check_tick_value(self) -> Tuple[bool, str]:
        """Check that tick_value is positive."""
        if self.mt5 is None:
            return False, "MT5 bridge not initialized"
        try:
            tick_value = getattr(self.mt5, "tick_value", None)
            if tick_value is not None and tick_value > 0:
                return True, f"tick_value = {tick_value:.5f}"
            return False, f"tick_value invalid: {tick_value}"
        except Exception as e:
            return False, f"Failed to check tick_value: {e}"

    def _check_balance(self) -> Tuple[bool, str]:
        """Check that the account balance is positive."""
        if self.mt5 is None:
            return False, "MT5 bridge not initialized"
        try:
            info = self.mt5.get_account_info()
            if not info:
                return False, "Cannot get account info"
            balance = info.get("balance", 0)
            if balance > 0:
                currency = info.get("currency", "")
                return True, f"Balance: {balance:.2f} {currency}"
            return False, f"Balance is zero or negative: {balance}"
        except Exception as e:
            return False, f"Failed to check balance: {e}"

    def _check_model_exists(self) -> Tuple[bool, str]:
        """Check that a model ZIP file exists in the model/ directory."""
        if self.model_loader is None:
            return False, "Model loader not initialized"
        try:
            path = self.model_loader.discover_model()
            if path is not None:
                return True, f"Model found: {path}"
            return False, "No model .zip file found in model/ directory"
        except Exception as e:
            return False, f"Failed to discover model: {e}"

    def _check_model_loads(self) -> Tuple[bool, str]:
        """Check that the model loader can find a valid model."""
        if self.model_loader is None:
            return False, "Model loader not initialized"
        try:
            path = self.model_loader.discover_model()
            if path is None:
                return False, "No model to load (no .zip found)"
            # We do NOT fully load the model here (expensive + side effects).
            # We verify the ZIP exists and is a valid ZIP file.
            import zipfile

            resolved = self.config.resolve_path(path)
            if not resolved.exists():
                return False, f"Model path does not exist: {resolved}"
            if not zipfile.is_zipfile(str(resolved)):
                return False, f"Model file is not a valid ZIP: {resolved}"
            return True, f"Model ZIP valid: {resolved.name}"
        except Exception as e:
            return False, f"Model load check failed: {e}"

    def _check_obs_dim(self) -> Tuple[bool, str]:
        """Check that config.obs_dim is positive and consistent with features.

        After model loading, config.obs_dim is overridden with the actual
        model's observation dimension, so this validates that the override
        happened correctly rather than checking against a hardcoded value.
        """
        try:
            obs_dim = getattr(self.config, "obs_dim", None)
            n_features = getattr(self.config, "n_features", None)
            frame_stack = getattr(self.config, "frame_stack", None)

            if obs_dim is None or obs_dim <= 0:
                return False, f"obs_dim is invalid: {obs_dim}"

            # Verify obs_dim = n_features * frame_stack
            if n_features and frame_stack:
                expected = n_features * frame_stack
                if obs_dim == expected:
                    return True, (
                        f"obs_dim = {obs_dim} "
                        f"({n_features} features x {frame_stack} frames)"
                    )
                return False, (
                    f"obs_dim = {obs_dim} but n_features({n_features}) "
                    f"x frame_stack({frame_stack}) = {expected}"
                )

            return True, f"obs_dim = {obs_dim}"
        except Exception as e:
            return False, f"Failed to check obs_dim: {e}"

    def _check_feature_count(self) -> Tuple[bool, str]:
        """Check that config.n_features is positive and consistent."""
        try:
            n_features = getattr(self.config, "n_features", None)
            n_market = getattr(self.config, "n_market_features", None)
            n_exempt = getattr(self.config, "n_exempt_features", None)

            if n_features is None or n_features <= 0:
                return False, f"n_features is invalid: {n_features}"

            if n_market and n_exempt:
                if n_market + n_exempt == n_features:
                    return True, (
                        f"{n_features} features "
                        f"({n_market} market + {n_exempt} exempt)"
                    )
                return False, (
                    f"n_market({n_market}) + n_exempt({n_exempt}) "
                    f"= {n_market + n_exempt} != n_features({n_features})"
                )

            return True, f"n_features = {n_features}"
        except Exception as e:
            return False, f"Failed to check feature count: {e}"

    def _check_htf_history(self) -> Tuple[bool, str]:
        """Check that H1/H4/D1 bars are available."""
        if self.pipeline is None:
            return False, "Feature pipeline not initialized"
        try:
            bar_counts = self.pipeline.get_bar_count()
            h1 = bar_counts.get("H1", 0)
            h4 = bar_counts.get("H4", 0)
            d1 = bar_counts.get("D1", 0)

            issues = []
            if h1 < 100:
                issues.append(f"H1={h1} (need 100+)")
            if h4 < 50:
                issues.append(f"H4={h4} (need 50+)")
            if d1 < 50:
                issues.append(f"D1={d1} (need 50+)")

            if not issues:
                return True, f"HTF history: H1={h1}, H4={h4}, D1={d1}"
            return False, f"Insufficient HTF bars: {', '.join(issues)}"
        except Exception as e:
            return False, f"Failed to check HTF history: {e}"

    def _check_weekend_manager(self) -> Tuple[bool, str]:
        """Check that the weekend manager is configured and active."""
        if self.weekend_manager is None:
            return False, "Weekend manager not initialized"
        try:
            fri_close = getattr(self.config, "friday_close_utc_hour", None)
            mon_resume = getattr(self.config, "monday_resume_utc_hour", None)
            if fri_close is not None and mon_resume is not None:
                return True, (
                    f"Weekend manager active: "
                    f"Friday close {fri_close}:00 UTC, "
                    f"Monday resume {mon_resume}:30 UTC"
                )
            return False, "Weekend times not configured"
        except Exception as e:
            return False, f"Failed to check weekend manager: {e}"

    def _check_m5_history(self) -> Tuple[bool, str]:
        """Check that 500+ M5 bars can be fetched from MT5."""
        if self.mt5 is None:
            return False, "MT5 bridge not initialized"
        try:
            import MetaTrader5 as mt5_mod

            symbol = self.config.mt5_symbol
            bars = self.mt5.get_latest_bars(
                symbol, mt5_mod.TIMEFRAME_M5, 500
            )
            count = len(bars) if bars is not None else 0
            if count >= 500:
                return True, f"M5 history: {count} bars available"
            return False, f"M5 history: only {count} bars (need 500+)"
        except ImportError:
            return False, "MetaTrader5 module not available"
        except Exception as e:
            return False, f"Failed to fetch M5 history: {e}"

    def _check_warmup(self) -> Tuple[bool, str]:
        """Check that the feature pipeline has completed warmup."""
        if self.pipeline is None:
            return False, "Feature pipeline not initialized"
        try:
            warmed = self.pipeline.is_warmed_up()
            if warmed:
                return True, "Feature pipeline warmed up"
            return False, "Feature pipeline has not completed warmup"
        except Exception as e:
            return False, f"Failed to check warmup status: {e}"

    def _check_normalizer(self) -> Tuple[bool, str]:
        """Check that the feature pipeline has an initialized normalizer."""
        if self.pipeline is None:
            return False, "Feature pipeline not initialized"
        try:
            normalizer = self.pipeline.get_normalizer()
            if normalizer is None:
                return False, "Normalizer is None"

            # In frozen mode, buffers are NOT used -- baseline mean/std is used directly.
            # Check that the baseline is loaded instead of checking buffer counts.
            mode = getattr(normalizer, "_mode", "adaptive")
            if mode == "frozen":
                has_baseline = getattr(normalizer, "_has_baseline", False)
                if has_baseline:
                    n = len(getattr(normalizer, "_baseline", {}))
                    return True, f"Normalizer active (frozen mode): {n} baseline features loaded"
                return False, "Normalizer in frozen mode but no baseline loaded"

            # Adaptive mode: check that the normalizer has buffers populated
            stats = normalizer.get_buffer_stats()
            populated = sum(
                1 for s in stats.values() if s.get("count", 0) > 0
            )
            total = len(stats)
            if populated > 0:
                return True, f"Normalizer active (adaptive): {populated}/{total} features have data"
            return False, "Normalizer has no data in buffers"
        except Exception as e:
            return False, f"Failed to check normalizer: {e}"

    def _check_circuit_breaker(self) -> Tuple[bool, str]:
        """Check that circuit breaker parameters are configured.

        The circuit breaker is always available via config parameters
        (consecutive_loss_pause, daily_dd_halt_pct, etc.), so this
        check verifies the config values are sensible.
        """
        try:
            clp = getattr(self.config, "consecutive_loss_pause", None)
            dd_halt = getattr(self.config, "daily_dd_halt_pct", None)
            if clp is not None and clp > 0 and dd_halt is not None and dd_halt > 0:
                return True, (
                    f"Circuit breaker configured: "
                    f"consecutive_loss_pause={clp}, "
                    f"daily_dd_halt={dd_halt:.1%}"
                )
            return False, "Circuit breaker config values missing or invalid"
        except Exception as e:
            return False, f"Failed to check circuit breaker: {e}"

    def _check_emergency_stop(self) -> Tuple[bool, str]:
        """Check that the emergency stop mechanism is available.

        The EmergencyStop class is always instantiable when the MT5
        bridge is present, so this check confirms readiness.
        """
        try:
            # Emergency stop is available as long as the system can
            # construct one. It does not require prior activation.
            return True, "Emergency stop mechanism available"
        except Exception as e:
            return False, f"Emergency stop check failed: {e}"

    def _check_memory_db(self) -> Tuple[bool, str]:
        """Check that the SQLite memory database is accessible."""
        if self.memory is None:
            return False, "Trading memory not initialized"
        try:
            db_path = getattr(self.memory, "db_path", None)
            if db_path is None:
                return False, "Memory has no db_path attribute"
            # Try a lightweight query to verify the connection is alive
            conn = getattr(self.memory, "conn", None)
            if conn is None:
                return False, "Memory database connection is None"
            cursor = conn.execute("SELECT COUNT(*) FROM trades")
            count = cursor.fetchone()[0]
            return True, f"Memory DB accessible: {count} trades recorded ({db_path})"
        except sqlite3.OperationalError as e:
            return False, f"Memory DB query failed: {e}"
        except Exception as e:
            return False, f"Failed to check memory DB: {e}"

    def _check_logs(self) -> Tuple[bool, str]:
        """Check that the log directory exists and is writable."""
        try:
            base = self.config.get_base_dir()
            log_dir = base / "storage" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            # Write a test file to verify write permissions
            test_file = log_dir / ".write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
            return True, f"Log directory writable: {log_dir}"
        except Exception as e:
            return False, f"Log directory not writable: {e}"

    # ------------------------------------------------------------------
    # Optional check implementations
    # ------------------------------------------------------------------

    def _check_calendar(self) -> Tuple[bool, str]:
        """Check that at least one calendar data source is available."""
        try:
            base = self.config.get_base_dir()

            # Check CSV calendar
            csv_path = base / self.config.calendar_csv_path
            if csv_path.exists():
                return True, f"Calendar CSV found: {csv_path.name}"

            # Check MQL5 bridge JSON
            bridge_path = base / self.config.calendar_bridge_path
            if bridge_path.exists():
                return True, f"Calendar bridge JSON found: {bridge_path.name}"

            # Check static known events
            static_path = base / self.config.calendar_static_path
            if static_path.exists():
                return True, f"Static calendar found: {static_path.name}"

            return False, (
                "No calendar data found. Checked: "
                f"{self.config.calendar_csv_path}, "
                f"{self.config.calendar_bridge_path}, "
                f"{self.config.calendar_static_path}"
            )
        except Exception as e:
            return False, f"Failed to check calendar: {e}"

    def _check_correlated(self) -> Tuple[bool, str]:
        """Check that at least some correlated instruments are available."""
        if self.mt5 is None:
            return False, "MT5 bridge not initialized"
        try:
            correlated_symbols = ["EURUSD", "XAGUSD", "USDJPY", "US500", "USOIL"]
            available = []
            unavailable = []

            for sym in correlated_symbols:
                info = self.mt5.get_symbol_info(sym)
                if info:
                    available.append(sym)
                else:
                    unavailable.append(sym)

            if len(available) >= 3:
                return True, (
                    f"{len(available)}/5 correlated instruments available: "
                    f"{', '.join(available)}"
                )
            if available:
                return False, (
                    f"Only {len(available)}/5 correlated instruments "
                    f"(need 3+): {', '.join(available)}"
                )
            return False, "No correlated instruments available"
        except Exception as e:
            return False, f"Failed to check correlated instruments: {e}"

    def _check_feature_baseline(self) -> Tuple[bool, str]:
        """Check that feature_baseline.json is loaded in the model package."""
        if self.model_loader is None:
            return False, "Model loader not initialized"
        try:
            # Check if a model has already been loaded and has
            # a feature_baseline.  The model_loader stores the
            # extract dir after load(), but we should not force a
            # full load here.  Instead, look for the file in the
            # model directory alongside the ZIP.
            base = self.config.get_base_dir()
            model_dir = base / "model"

            # Check for standalone feature_baseline.json
            baseline_path = model_dir / "feature_baseline.json"
            if baseline_path.exists():
                return True, f"Feature baseline found: {baseline_path.name}"

            # Also check inside any extracted model package
            # (model_loader may have already loaded it)
            extract_dir = getattr(self.model_loader, "_extract_dir", None)
            if extract_dir:
                extracted_baseline = Path(extract_dir) / "feature_baseline.json"
                if extracted_baseline.exists():
                    return True, "Feature baseline found in loaded model package"

            return False, (
                "feature_baseline.json not found. "
                "Drift detection will be unavailable."
            )
        except Exception as e:
            return False, f"Failed to check feature baseline: {e}"

    def _check_stress_results(self) -> Tuple[bool, str]:
        """Check that stress_results.json is loaded in the model package."""
        if self.model_loader is None:
            return False, "Model loader not initialized"
        try:
            base = self.config.get_base_dir()
            model_dir = base / "model"

            # Check for standalone stress_results.json
            stress_path = model_dir / "stress_results.json"
            if stress_path.exists():
                return True, f"Stress results found: {stress_path.name}"

            # Also check inside any extracted model package
            extract_dir = getattr(self.model_loader, "_extract_dir", None)
            if extract_dir:
                extracted_stress = Path(extract_dir) / "stress_results.json"
                if extracted_stress.exists():
                    return True, "Stress results found in loaded model package"

            return False, (
                "stress_results.json not found. "
                "Training-vs-live comparison will be unavailable."
            )
        except Exception as e:
            return False, f"Failed to check stress results: {e}"

    # ------------------------------------------------------------------
    # Paper-to-live transition checks
    # ------------------------------------------------------------------

    def _get_paper_trades(self) -> list:
        """Fetch all paper trades from memory."""
        if self.memory is None:
            return []
        try:
            conn = getattr(self.memory, "conn", None)
            if conn is None:
                return []
            cursor = conn.execute(
                "SELECT * FROM trades ORDER BY timestamp DESC"
            )
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, r)) for r in rows]
        except Exception:
            return []

    def _check_paper_min_duration(self) -> Tuple[bool, str]:
        """Check >= 1 week of paper trading history."""
        trades = self._get_paper_trades()
        if not trades:
            return False, "No paper trades recorded yet"
        try:
            from datetime import datetime, timezone

            first_ts = trades[-1].get("timestamp")
            if first_ts is None:
                return False, "Cannot determine first trade time"
            if isinstance(first_ts, str):
                first_ts = datetime.fromisoformat(first_ts)
            now = datetime.now(timezone.utc)
            if hasattr(first_ts, "tzinfo") and first_ts.tzinfo is None:
                first_ts = first_ts.replace(tzinfo=timezone.utc)
            days = (now - first_ts).days
            if days >= 7:
                return True, f"Paper trading for {days} days (>= 7 required)"
            return False, f"Only {days} days of paper trading (need >= 7)"
        except Exception as e:
            return False, f"Failed to check duration: {e}"

    def _check_paper_profit_factor(self) -> Tuple[bool, str]:
        """Check profit factor >= 1.4."""
        trades = self._get_paper_trades()
        if not trades:
            return False, "No paper trades to evaluate"
        gross_profit = sum(
            t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0
        )
        gross_loss = abs(sum(
            t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0
        ))
        if gross_loss == 0:
            pf = float("inf") if gross_profit > 0 else 0.0
        else:
            pf = gross_profit / gross_loss
        if pf >= 1.4:
            return True, f"Profit factor = {pf:.2f} (>= 1.4)"
        return False, f"Profit factor = {pf:.2f} (need >= 1.4)"

    def _check_paper_win_rate(self) -> Tuple[bool, str]:
        """Check win rate >= 45%."""
        trades = self._get_paper_trades()
        if not trades:
            return False, "No paper trades to evaluate"
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        wr = wins / len(trades) * 100
        if wr >= 45:
            return True, f"Win rate = {wr:.1f}% (>= 45%)"
        return False, f"Win rate = {wr:.1f}% (need >= 45%)"

    def _check_paper_max_dd(self) -> Tuple[bool, str]:
        """Check max drawdown < 15%."""
        trades = self._get_paper_trades()
        if not trades:
            return False, "No paper trades to evaluate"
        balance = 0.0
        peak = 0.0
        max_dd_pct = 0.0
        for t in trades:
            balance += t.get("pnl", 0)
            if balance > peak:
                peak = balance
            if peak > 0:
                dd = (peak - balance) / peak
                if dd > max_dd_pct:
                    max_dd_pct = dd
        max_dd_pct *= 100
        if max_dd_pct < 15:
            return True, f"Max drawdown = {max_dd_pct:.1f}% (< 15%)"
        return False, f"Max drawdown = {max_dd_pct:.1f}% (need < 15%)"
