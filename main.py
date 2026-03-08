"""Spartus Live Trading Dashboard -- Entry Point.

Usage:
    python main.py                    # Normal launch
    python main.py --config path.yaml # Custom config
    python main.py --paper            # Force paper trading mode
"""
import sys
import os
import signal
import logging
import argparse
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Set up base directory
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))

# Ensure storage/logs directory exists BEFORE configuring file handler
_log_dir = BASE_DIR / "storage" / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            _log_dir / "dashboard.log",
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("spartus.live")

# ---------------------------------------------------------------------------
# Graceful MetaTrader5 import
# ---------------------------------------------------------------------------
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None
    log.warning(
        "MetaTrader5 module not installed.  "
        "Install with: pip install MetaTrader5"
    )

# ---------------------------------------------------------------------------
# Component imports
# ---------------------------------------------------------------------------
from config.live_config import LiveConfig
from core.mt5_bridge import MT5Bridge
from core.model_loader import ModelLoader
from core.inference_engine import InferenceEngine
from core.feature_pipeline import LiveFeaturePipeline
from core.risk_manager import LiveRiskManager
from core.trade_executor import TradeExecutor, TradingState as ExecutorState
from core.position_manager import PositionManager
from core.startup_validator import StartupValidator

from memory.trading_memory import TradingMemory
from memory.trend_tracker import TrendTracker
from memory.trade_analyzer import TradeAnalyzer

from core.broker_constraints import BrokerConstraints

from safety.circuit_breaker import CircuitBreaker
from safety.weekend_manager import WeekendManager
from safety.emergency_stop import EmergencyStop
from safety.connection_monitor import ConnectionMonitor

from utils.logger import LiveLogger

from features.account_features import compute_account_features
from features.memory_features import get_memory_features_array

# Dashboard imports (PyQt6)
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
import pyqtgraph as pg

from dashboard.main_window import LiveDashboard, TradingState as DashState
from dashboard.tab_live_status import LiveStatusTab
from dashboard.tab_performance import PerformanceTab
from dashboard.tab_trade_journal import TradeJournalTab
from dashboard.tab_model_state import ModelStateTab
from dashboard.tab_alerts import AlertsTab
from dashboard.tab_analytics import AnalyticsTab
from dashboard import currency

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TRADING_LOOP_INTERVAL_MS = 5000      # Check for new M5 bars every 5 seconds
_DASHBOARD_UPDATE_INTERVAL_MS = 1000  # Refresh dashboard UI every 1 second
_NORMALIZER_SAVE_INTERVAL_S = 3600    # Save normalizer state every hour


# ===========================================================================
# SpartusOrchestrator -- Main entry point and trading loop
# ===========================================================================

class SpartusOrchestrator:
    """Orchestrates all components and drives the live trading loop.

    Initialization order:
        1. Parse args, load config
        2. Ensure storage directories
        3. MT5Bridge -> connect
        4. TradingMemory (SQLite)
        5. ModelLoader -> load model package
        6. InferenceEngine
        7. LiveRiskManager
        8. PositionManager
        9. TradeExecutor (STOPPED state)
        10. LiveFeaturePipeline -> warmup
        11. CircuitBreaker, WeekendManager, EmergencyStop, ConnectionMonitor
        12. LiveLogger
        13. StartupValidator -> run checks
        14. TrendTracker, TradeAnalyzer
        15. Create QApplication + LiveDashboard
        16. Inject components + build tabs
        17. Wire signals
        18. Start timers
        19. app.exec()
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args

        # ---- Config ----
        self._config: Optional[LiveConfig] = None

        # ---- Core components ----
        self._mt5_bridge: Optional[MT5Bridge] = None
        self._memory: Optional[TradingMemory] = None
        self._model_loader: Optional[ModelLoader] = None
        self._model_components: Dict[str, Any] = {}
        self._inference: Optional[InferenceEngine] = None
        self._risk_manager: Optional[LiveRiskManager] = None
        self._position_manager: Optional[PositionManager] = None
        self._executor: Optional[TradeExecutor] = None
        self._pipeline: Optional[LiveFeaturePipeline] = None

        # ---- Safety ----
        self._circuit_breaker: Optional[CircuitBreaker] = None
        self._weekend_manager: Optional[WeekendManager] = None
        self._emergency_stop: Optional[EmergencyStop] = None
        self._connection_monitor: Optional[ConnectionMonitor] = None

        # ---- Logging / Analytics ----
        self._live_logger: Optional[LiveLogger] = None
        self._validator: Optional[StartupValidator] = None
        self._trend_tracker: Optional[TrendTracker] = None
        self._trade_analyzer: Optional[TradeAnalyzer] = None

        # ---- Dashboard ----
        self._app: Optional[QApplication] = None
        self._dashboard: Optional[LiveDashboard] = None
        self._tab_live_status: Optional[LiveStatusTab] = None
        self._tab_performance: Optional[PerformanceTab] = None
        self._tab_journal: Optional[TradeJournalTab] = None
        self._tab_model_state: Optional[ModelStateTab] = None
        self._tab_alerts: Optional[AlertsTab] = None
        self._tab_analytics: Optional[AnalyticsTab] = None

        # ---- Timers ----
        self._trading_timer: Optional[QTimer] = None
        self._dashboard_timer: Optional[QTimer] = None

        # ---- State tracking ----
        self._last_bar_time: Optional[datetime] = None
        self._step_count: int = 0
        self._last_normalizer_save: float = time.monotonic()
        self._last_daily_reset_day: int = -1
        self._last_action: Dict[str, float] = {}
        self._last_decision: str = ""
        self._initial_balance: float = 0.0
        self._peak_balance: float = 0.0

        # ---- Heartbeat / market status ----
        self._last_heartbeat_time: float = 0.0
        self._bars_since_heartbeat: int = 0
        self._market_status: str = "INITIALIZING"  # OPEN, CLOSED, INITIALIZING
        self._last_block_reason: str = ""  # Track last safety block reason for alert dedup

        # ---- History buffers for dashboard ----
        self._balance_history: deque = deque(maxlen=5000)
        self._action_history: deque = deque(maxlen=500)
        self._decision_log: deque = deque(maxlen=50)
        self._alert_log: deque = deque(maxlen=200)

        # ---- Validation results (cached) ----
        self._validation_results: Dict[str, Any] = {}

        # ---- Performance data cache (avoid 500-trade SQL query every second) ----
        self._perf_cache: Dict[str, Any] = {}
        self._perf_cache_time: float = 0.0
        _PERF_CACHE_TTL_S = 5.0  # refresh at most every 5 seconds

    # ==================================================================
    # Initialization
    # ==================================================================

    def initialize(self) -> bool:
        """Run the full initialization sequence.

        Returns:
            True if all critical components initialized successfully.
        """
        log.info("=" * 60)
        log.info("  SPARTUS LIVE TRADING DASHBOARD")
        log.info("  Starting initialization...")
        log.info("=" * 60)

        # 1. Load config
        config_path = self._args.config
        self._config = LiveConfig.from_yaml(config_path)
        if self._args.paper:
            self._config.paper_trading = True
        log.info(
            "Config loaded: symbol=%s  paper=%s  model=%s",
            self._config.mt5_symbol,
            self._config.paper_trading,
            self._config.model_path,
        )

        # Validate config
        config_issues = self._config.validate()
        for issue in config_issues:
            log.warning("Config validation: %s", issue)

        # 2. Ensure storage directories
        self._ensure_storage_dirs()

        # 3. Check MetaTrader5 is available
        if mt5 is None:
            log.error(
                "MetaTrader5 module is not installed. Cannot proceed. "
                "Install with: pip install MetaTrader5"
            )
            return False

        # 4. MT5 Bridge -> connect
        self._mt5_bridge = MT5Bridge(self._config)
        if not self._mt5_bridge.connect():
            log.error("Failed to connect to MT5 terminal")
            return False
        log.info("MT5 bridge connected")

        # 4b. Detect account currency and set global symbol for UI formatting
        _acct = self._mt5_bridge.get_account_info()
        _acct_ccy = _acct.get("currency", "USD") if _acct else "USD"
        currency.set_currency(_acct_ccy)
        log.info("Account currency: %s (%s)", _acct_ccy, currency.sym())

        # Persist currency to state file so CLI tools can read it
        import json as _json
        _state_dir = self._config.resolve_path("storage/state")
        _state_dir.mkdir(parents=True, exist_ok=True)
        _ccy_path = _state_dir / "account_currency.json"
        _ccy_path.write_text(
            _json.dumps({"currency": _acct_ccy, "symbol": currency.sym()}),
            encoding="utf-8",
        )

        # 5. Trading Memory (SQLite)
        db_path = self._config.resolve_path("storage/memory/spartus_live.db")
        self._memory = TradingMemory(str(db_path))
        log.info("Trading memory initialized: %s", db_path)

        # 6. Model Loader -> load model package
        self._model_loader = ModelLoader(self._config)
        try:
            self._model_components = self._model_loader.load()
            log.info("Model package loaded successfully")
        except (FileNotFoundError, RuntimeError) as exc:
            log.error("Failed to load model: %s", exc)
            return False

        # 6b. Update config.model_path with the actually-discovered path
        _discovered = self._model_loader.discover_model()
        if _discovered:
            self._config.model_path = _discovered

        # 6c. Override config with actual model metadata (source of truth)
        _meta = self._model_components.get("metadata", {})
        if _meta.get("obs_dim"):
            self._config.obs_dim = _meta["obs_dim"]
        if _meta.get("num_features"):
            self._config.n_features = _meta["num_features"]
        if _meta.get("frame_stack"):
            self._config.frame_stack = _meta["frame_stack"]
        log.info(
            "Config from model metadata: model=%s, obs_dim=%d, "
            "n_features=%d, frame_stack=%d",
            self._config.model_path,
            self._config.obs_dim,
            self._config.n_features,
            self._config.frame_stack,
        )

        # 7. Inference Engine
        model_obj = self._model_components.get("model")
        if model_obj is None:
            log.error("No model object in loaded package")
            return False
        self._inference = InferenceEngine(model_obj)
        log.info("Inference engine initialized")

        # 8. Risk Manager
        self._risk_manager = LiveRiskManager(self._config)

        # 9. Position Manager
        self._position_manager = PositionManager(self._mt5_bridge, self._config)
        self._position_manager.update_from_mt5()

        # 10. Trade Executor (starts in STOPPED state)
        self._executor = TradeExecutor(
            mt5_bridge=self._mt5_bridge,
            risk_manager=self._risk_manager,
            memory=self._memory,
            config=self._config,
        )
        log.info("Trade executor initialized (state=STOPPED)")

        # 10b. Broker Constraints cache (dynamic, refreshes from MT5)
        self._broker_constraints = BrokerConstraints(self._config, self._mt5_bridge)
        self._broker_constraints.heavy_refresh(force=True)
        self._broker_constraints.light_refresh(force=True)
        self._executor.set_broker_constraints(self._broker_constraints)
        # Inject model version for trade traceability
        _meta = self._model_components.get("metadata", {})
        self._executor.set_model_version(
            version=f"W{_meta.get('week', '?')}",
            file_hash=self._model_components.get("config", {}).get("model_file_hash", ""),
        )
        log.info("Broker constraints initialized and injected into executor")

        # 10c. Auto-detect MT5 Common Files path for CalendarBridge
        if not self._config.calendar_bridge_path:
            try:
                _ti = mt5.terminal_info()
                if _ti and hasattr(_ti, "commondata_path"):
                    _common = Path(_ti.commondata_path) / "Files" / "calendar_events.json"
                    self._config.calendar_bridge_path = str(_common)
                    log.info("Calendar bridge path (auto): %s", _common)
            except Exception:
                pass

        # 11. Feature Pipeline -> warmup
        self._pipeline = LiveFeaturePipeline(self._config)

        # Load training feature baseline (for frozen mode + adaptive clamp cap)
        feature_baseline = self._model_components.get("feature_baseline", {})
        if feature_baseline:
            self._pipeline.set_feature_baseline(feature_baseline)
            log.info(
                "Feature baseline from model package: %d features (mode=%s)",
                len(feature_baseline),
                self._config.normalization_mode,
            )
        elif self._config.normalization_mode == "frozen":
            log.warning(
                "FROZEN mode selected but no feature_baseline in model package! "
                "Falling back to adaptive mode."
            )
            self._config.normalization_mode = "adaptive"

        # Try loading saved normalizer state first (adaptive mode only)
        if self._config.normalization_mode == "adaptive":
            norm_loaded = self._pipeline.load_normalizer_state()
            if norm_loaded:
                log.info("Normalizer state restored from disk")

        warmup_ok = self._pipeline.warmup(self._mt5_bridge)
        if not warmup_ok:
            log.error("Feature pipeline warmup FAILED")
            return False
        log.info("Feature pipeline warmed up successfully")

        # 12. Safety components
        self._circuit_breaker = CircuitBreaker(self._config)
        self._weekend_manager = WeekendManager(self._config)
        self._emergency_stop = EmergencyStop(self._mt5_bridge)
        self._connection_monitor = ConnectionMonitor(self._mt5_bridge)

        # Wire emergency stop callback on MT5 bridge
        self._mt5_bridge.on_emergency_stop = self._handle_emergency_stop

        # 13. Live Logger
        log_dir = str(self._config.resolve_path("storage/logs"))
        self._live_logger = LiveLogger(log_dir)

        # 14. Startup Validator -> run checks
        self._validator = StartupValidator(
            config=self._config,
            mt5_bridge=self._mt5_bridge,
            feature_pipeline=self._pipeline,
            model_loader=self._model_loader,
            memory=self._memory,
            weekend_manager=self._weekend_manager,
        )
        self._validation_results = self._validator.run_all_checks()
        summary = self._validator.get_summary()
        log.info(
            "Startup validation: %d/%d passed (%d required pass, %d required fail)",
            summary["passed"],
            summary["total"],
            summary["required_passed"],
            summary["required_failed"],
        )
        for name, result in self._validation_results.items():
            status = "PASS" if result["passed"] else "FAIL"
            req = "REQ" if result["required"] else "OPT"
            log.info("  [%s] [%s] %s: %s", status, req, name, result["message"])

        if not self._validator.all_required_pass():
            log.warning(
                "Not all required checks passed. "
                "Trading will be blocked until issues are resolved."
            )
            # Continue launching the dashboard so the user can see what failed

        # 15. TrendTracker, TradeAnalyzer
        self._trend_tracker = TrendTracker(self._memory)
        self._trade_analyzer = TradeAnalyzer(self._memory)

        # 16. Seed initial balance tracking
        acct = self._mt5_bridge.get_account_info()
        if acct:
            self._initial_balance = acct.get("balance", 0.0)
            self._peak_balance = self._initial_balance
            self._balance_history.append(self._initial_balance)

        log.info("Initialization COMPLETE")
        return True

    def _ensure_storage_dirs(self) -> None:
        """Create all required storage directories."""
        base = self._config.get_base_dir()
        dirs = [
            base / "storage" / "logs",
            base / "storage" / "state",
            base / "storage" / "memory",
            base / "storage" / "reports" / "weekly",
            base / "data" / "calendar",
            base / "model",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Dashboard Launch
    # ==================================================================

    def launch(self) -> int:
        """Create the Qt application and dashboard, start timers, run event loop.

        Returns:
            Application exit code.
        """
        # Create QApplication
        self._app = QApplication(sys.argv)
        self._app.setApplicationName("Spartus Live Trading")

        # Set pyqtgraph global config for dark mode (already done in main_window
        # import, but ensure it here too)
        pg.setConfigOption("background", "#0d1117")
        pg.setConfigOption("foreground", "#e6edf3")

        # Create main window
        self._dashboard = LiveDashboard()

        # Build tabs and insert into dashboard containers
        self._build_tabs()

        # Inject backend components into dashboard
        model_info = self._model_components.get("metadata", {})
        engine_info = self._inference.get_model_info() if self._inference else {}
        combined_model_info = {**model_info, **engine_info}

        self._dashboard.set_components(
            mt5_bridge=self._mt5_bridge,
            feature_pipeline=self._pipeline,
            inference_engine=self._inference,
            trade_executor=self._executor,
            risk_manager=self._risk_manager,
            position_manager=self._position_manager,
            memory=self._memory,
            model_info=combined_model_info,
        )

        # Set paper trading display
        self._dashboard.set_paper_trading(self._config.paper_trading)

        # Wire button signals from dashboard and Tab 5 (AlertsTab)
        self._wire_signals()

        # Show window
        self._dashboard.show()

        # Start trading loop timer (5-second interval to detect new M5 bars)
        self._trading_timer = QTimer()
        self._trading_timer.timeout.connect(self._trading_loop)
        self._trading_timer.start(_TRADING_LOOP_INTERVAL_MS)

        # Start dashboard UI update timer (1-second interval)
        self._dashboard_timer = QTimer()
        self._dashboard_timer.timeout.connect(self._dashboard_update)
        self._dashboard_timer.start(_DASHBOARD_UPDATE_INTERVAL_MS)

        log.info(
            "Dashboard launched. Trading loop=%dms, UI update=%dms",
            _TRADING_LOOP_INTERVAL_MS,
            _DASHBOARD_UPDATE_INTERVAL_MS,
        )

        # Run Qt event loop
        exit_code = self._app.exec()

        # Clean shutdown
        self._shutdown()

        return exit_code

    def _build_tabs(self) -> None:
        """Instantiate all 6 tab widgets and add them to dashboard containers."""
        # Tab 1: Live Status
        container_1 = self._dashboard.get_tab("LIVE STATUS")
        self._tab_live_status = LiveStatusTab()
        container_1.layout().addWidget(self._tab_live_status)

        # Tab 2: Performance
        container_2 = self._dashboard.get_tab("PERFORMANCE")
        self._tab_performance = PerformanceTab()
        container_2.layout().addWidget(self._tab_performance)

        # Tab 3: Trade Journal
        container_3 = self._dashboard.get_tab("TRADE JOURNAL")
        self._tab_journal = TradeJournalTab()
        container_3.layout().addWidget(self._tab_journal)

        # Tab 4: Model & Features
        container_4 = self._dashboard.get_tab("MODEL & FEATURES")
        self._tab_model_state = ModelStateTab()
        container_4.layout().addWidget(self._tab_model_state)

        # Tab 5: Alerts & Safety
        container_5 = self._dashboard.get_tab("ALERTS & SAFETY")
        self._tab_alerts = AlertsTab()
        container_5.layout().addWidget(self._tab_alerts)

        # Tab 6: Analytics
        container_6 = self._dashboard.get_tab("ANALYTICS")
        self._tab_analytics = AnalyticsTab()
        container_6.layout().addWidget(self._tab_analytics)

    def _wire_signals(self) -> None:
        """Connect button signals from dashboard and Tab 5 to executor methods."""
        # Dashboard header buttons -- safely disconnect any default handlers
        try:
            self._dashboard._btn_start.clicked.disconnect()
        except TypeError:
            pass
        self._dashboard._btn_start.clicked.connect(self._on_start_trading)

        try:
            self._dashboard._btn_stop.clicked.disconnect()
        except TypeError:
            pass
        self._dashboard._btn_stop.clicked.connect(self._on_stop_trading)

        try:
            self._dashboard._btn_wind_down.clicked.disconnect()
        except TypeError:
            pass
        self._dashboard._btn_wind_down.clicked.connect(self._on_wind_down)

        try:
            self._dashboard._btn_emergency.clicked.disconnect()
        except TypeError:
            pass
        self._dashboard._btn_emergency.clicked.connect(self._on_emergency_stop)

        # Tab 5 (AlertsTab) signals
        self._tab_alerts.start_requested.connect(self._on_start_trading)
        self._tab_alerts.wind_down_requested.connect(self._on_wind_down)
        self._tab_alerts.stop_requested.connect(self._on_stop_trading)
        self._tab_alerts.emergency_stop_requested.connect(self._on_emergency_stop)
        self._tab_alerts.reset_cb_requested.connect(self._on_reset_circuit_breaker)
        self._tab_alerts.reset_normalizer_requested.connect(self._on_reset_normalizer)

    # ==================================================================
    # Button Callbacks
    # ==================================================================

    def _on_start_trading(self) -> None:
        """Start trading -- user clicked Start Trading."""
        if self._validator is None or not self._validator.all_required_pass():
            log.warning("Cannot start: not all required checks passed")
            self._add_alert("WARN", "Cannot start: required startup checks failed")
            return
        if self._executor is None:
            self._add_alert("WARN", "Cannot start: trade executor not initialized")
            return

        self._executor.start_trading()
        self._dashboard.set_trading_state(DashState.RUNNING)

        # Reconcile MT5 deal history -- recover any trades that closed
        # while the dashboard was down (TP/SL hit during restart)
        reconciled = self._executor.reconcile_trade_history()
        if reconciled > 0:
            self._add_alert(
                "WARN",
                f"Recovered {reconciled} missed trade(s) from MT5 history",
            )

        if self._executor.has_position():
            self._add_alert("INFO", "Trading STARTED (recovered existing position)")
        else:
            self._add_alert("INFO", "Trading STARTED")
        log.info("User started trading")

    def _on_stop_trading(self) -> None:
        """Stop trading immediately -- close positions."""
        if self._executor is None:
            return
        self._executor.stop_trading()
        self._dashboard.set_trading_state(DashState.STOPPED)
        self._add_alert("INFO", "Trading STOPPED (positions closed)")
        log.info("User stopped trading")

    def _on_wind_down(self) -> None:
        """Wind down -- no new trades, manage open position until close."""
        if self._executor is None:
            return
        self._executor.wind_down()
        self._dashboard.set_trading_state(DashState.WINDING_DOWN)
        self._add_alert("INFO", "Winding down -- no new trades")
        log.info("User initiated wind down")

    def _on_emergency_stop(self) -> None:
        """Emergency stop -- close all immediately."""
        if self._executor is not None:
            self._executor.emergency_stop()
        if self._emergency_stop is not None:
            self._emergency_stop.activate("user_dashboard")
        self._dashboard.set_trading_state(DashState.STOPPED)
        self._add_alert("ERROR", "EMERGENCY STOP activated")
        log.critical("EMERGENCY STOP activated by user")

    def _on_reset_circuit_breaker(self) -> None:
        """Reset circuit breaker -- allow trading to resume."""
        if self._circuit_breaker is None:
            return
        self._circuit_breaker.reset()
        self._add_alert("INFO", "Circuit breaker RESET by user")
        log.info("Circuit breaker reset by user")

    def _on_reset_normalizer(self) -> None:
        """Reset normalizer -- clear contaminated z-score buffers and re-warm.

        This is the recovery action for conviction collapse caused by
        violent market moves poisoning the rolling z-score buffers.
        """
        if self._pipeline is None:
            self._add_alert("WARN", "Cannot reset normalizer: pipeline not initialized")
            return
        self._pipeline.reset_normalizer()
        self._add_alert("INFO", "Normalizer RESET -- z-score buffers cleared and re-warmed")
        log.info("Normalizer reset by user (conviction collapse recovery)")

    def _handle_emergency_stop(self) -> None:
        """Called by MT5Bridge heartbeat when connection is lost."""
        log.critical("MT5 connection lost -- activating emergency stop")
        if self._executor is not None:
            self._executor.emergency_stop()
        self._dashboard.set_trading_state(DashState.STOPPED)
        self._add_alert("ERROR", "MT5 connection lost -- EMERGENCY STOP")

    # ==================================================================
    # Trading Loop (QTimer, every 5 seconds)
    # ==================================================================

    def _trading_loop(self) -> None:
        """Main trading loop -- check for new M5 bars and execute actions.

        Called every 5 seconds by QTimer.  Only processes when a new M5
        bar has closed (compare current bar time vs last processed).
        All exceptions are caught to prevent dashboard crashes.
        """
        try:
            self._trading_loop_inner()
        except Exception:
            log.exception("Error in trading loop")
            self._add_alert("ERROR", "Trading loop error -- check logs")

    def _trading_loop_inner(self) -> None:
        """Inner trading loop logic (unwrapped from exception handler)."""
        # Guard: need all core components for trading
        if self._connection_monitor is None or self._pipeline is None \
                or self._inference is None or self._executor is None:
            return  # Limited mode -- no trading possible

        # Check MT5 connection
        if not self._connection_monitor.is_connected():
            conn_status = self._connection_monitor.check_connection()
            if not conn_status.get("connected", False):
                self._market_status = "DISCONNECTED"
                return  # Wait for reconnection

        # Refresh broker constraints (self-throttled, returns fast if not due)
        if hasattr(self, "_broker_constraints") and self._broker_constraints is not None:
            self._broker_constraints.heavy_refresh()
            self._broker_constraints.light_refresh()

        # Check for new M5 bar
        if not self._is_new_bar():
            self._emit_heartbeat_if_due()
            return

        self._step_count += 1

        # --- Get account info ---
        account = self._mt5_bridge.get_account_info()
        if not account:
            log.warning("Could not get account info -- skipping cycle")
            return

        balance = account.get("balance", 0.0)
        equity = account.get("equity", balance)
        if balance > self._peak_balance:
            self._peak_balance = balance

        # --- Sync position with MT5 (detect SL/TP hit) ---
        sync_result = self._executor.sync_position(account)
        if sync_result in ("TP_HIT", "SL_HIT", "EXTERNAL_CLOSE"):
            self._add_alert(
                "INFO",
                f"Position closed by MT5: {sync_result}",
            )
            # Record in circuit breaker -- sync_position already called
            # _record_trade_close which updates risk_manager.  Use
            # consecutive_losses as the indicator: if > 0, last trade lost.
            risk_status = self._risk_manager.get_safety_status()
            if risk_status.get("consecutive_losses", 0) > 0:
                self._circuit_breaker.record_loss()
            else:
                self._circuit_breaker.record_win()

        # --- Compute account features ---
        pos = self._executor.get_position()
        current_price = 0.0
        m5_bars = self._mt5_bridge.get_latest_bars(
            self._config.mt5_symbol,
            mt5.TIMEFRAME_M5 if mt5 else 5,
            1,
        )
        if not m5_bars.empty:
            current_price = float(m5_bars["close"].iloc[-1])

        atr = self._pipeline.get_current_atr()
        account_state = compute_account_features(
            has_position=pos is not None,
            position_side=pos["side"] if pos else None,
            entry_price=pos["entry_price"] if pos else 0.0,
            current_price=current_price,
            lots=pos["lots"] if pos else 0.0,
            stop_loss=pos["sl"] if pos else 0.0,
            entry_step=self._executor._entry_step if pos else 0,
            current_step=self._step_count,
            balance=balance,
            equity=equity,
            peak_balance=self._peak_balance,
            initial_balance=self._initial_balance,
            atr=atr,
            value_per_point_per_lot=self._mt5_bridge.value_per_point,
        )

        # --- Compute memory features ---
        market_state = {
            "rsi": self._pipeline.get_feature_snapshot().get("rsi_14", 0.5),
            "trend_dir": self._pipeline.get_feature_snapshot().get("h1_trend_dir", 0.0),
            "session": self._get_current_session(),
            "vol_regime": self._pipeline.get_feature_snapshot().get("atr_ratio", 1.0),
        }
        memory_features = get_memory_features_array(
            memory=self._memory,
            market_state=market_state,
            current_step=self._step_count,
        )

        # --- Run feature pipeline on_new_bar ---
        observation = self._pipeline.on_new_bar(
            mt5_bridge=self._mt5_bridge,
            account_state=account_state,
            memory_features=memory_features,
        )

        if observation is None:
            # Duplicate bar or pipeline not ready -- skip
            return

        # --- Update executor ATR ---
        self._executor.update_atr(atr)

        # --- Check circuit breakers + spread gate ---
        cb_allowed, cb_reason = self._circuit_breaker.should_trade()
        wk_allowed, wk_reason = self._weekend_manager.should_trade()
        spread_allowed, spread_reason = (True, "ok")
        if hasattr(self, "_broker_constraints") and self._broker_constraints is not None:
            spread_allowed, spread_reason = self._broker_constraints.check_spread_gate()

        # Update circuit breaker drawdown
        if self._peak_balance > 0:
            daily_dd = max(0.0, -self._risk_manager.daily_pnl) / self._peak_balance
            weekly_dd = max(0.0, -self._risk_manager.weekly_pnl) / self._peak_balance
            self._circuit_breaker.update_dd(daily_dd, weekly_dd)

        # If circuit breaker triggered close-all
        if self._circuit_breaker.close_all_triggered:
            log.critical("Circuit breaker CLOSE ALL triggered")
            self._executor.emergency_stop()
            self._circuit_breaker.acknowledge_close_all()
            self._dashboard.set_trading_state(DashState.CB_PAUSED)
            self._add_alert("ERROR", "Circuit breaker: CLOSE ALL positions")
            return

        # Weekend forced close
        if self._weekend_manager.should_close_all():
            if self._executor.has_position():
                log.warning("Weekend forced close")
                self._executor.stop_trading()
                self._add_alert("WARN", "Weekend close: all positions closed")
                self._dashboard.set_trading_state(DashState.STOPPED)
            return

        # --- Run inference ---
        action = self._inference.predict(observation)

        # Validate action output for NaN/Inf
        for key in ("direction", "conviction", "exit_urgency", "sl_adjustment"):
            val = action.get(key, 0.0)
            if not np.isfinite(val):
                log.error("Inference produced %s=%s -- replacing with 0.0", key, val)
                action[key] = 0.0

        self._last_action = action

        # Record action in history for analytics
        self._action_history.append(action)

        # --- Execute action ---
        bar_data = {
            "close": current_price,
            "time": datetime.now(timezone.utc),
            "rsi_14": self._pipeline.get_feature_snapshot().get("rsi_14", 0.5),
            "h1_trend_dir": self._pipeline.get_feature_snapshot().get("h1_trend_dir", 0.0),
            "atr_ratio": self._pipeline.get_feature_snapshot().get("atr_ratio", 1.0),
            "session": self._get_current_session(),
        }
        if not m5_bars.empty:
            bar_data["high"] = float(m5_bars["high"].iloc[-1])
            bar_data["low"] = float(m5_bars["low"].iloc[-1])

        # Pass bar context to executor for trade-entry snapshots
        _spread = 0.0
        if hasattr(self, "_broker_constraints") and self._broker_constraints is not None:
            _spread = self._broker_constraints.spread_current_points
        self._executor.set_bar_context(
            observation=observation,
            risk_state=self._risk_manager.get_safety_status(),
            spread=_spread,
        )

        # Only execute if safety systems allow (or if already in position)
        if self._executor.has_position() or (cb_allowed and wk_allowed and spread_allowed):
            decision = self._executor.execute_action(action, bar_data, account)
        else:
            # Safety systems blocked -- do NOT call execute_action()
            if not cb_allowed:
                decision = f"CB_BLOCKED_{cb_reason}"
            elif not wk_allowed:
                decision = f"WK_BLOCKED_{wk_reason}"
            elif not spread_allowed:
                decision = f"BLOCKED_{spread_reason}"
            else:
                decision = "BLOCKED"

        self._last_decision = decision

        # --- Log decision ---
        self._decision_log.append({
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "action": decision,
            "details": (
                f"dir={action['direction']:+.2f} "
                f"conv={action['conviction']:.2f} "
                f"exit={action['exit_urgency']:.2f} "
                f"sl={action['sl_adjustment']:.2f}"
            ),
        })

        # --- Log action to file (comprehensive) ---
        if self._config.log_every_action and self._live_logger:
            self._live_logger.log_action({
                "bar_time": bar_data["time"].isoformat(),
                "action_raw": [
                    action["direction"],
                    action["conviction"],
                    action["exit_urgency"],
                    action["sl_adjustment"],
                ],
                "direction": action["direction"],
                "conviction": action["conviction"],
                "exit_urgency": action["exit_urgency"],
                "sl_adjustment": action["sl_adjustment"],
                "decision": decision,
                "has_position": self._executor.has_position(),
                "balance": balance,
                "equity": equity,
                "price": current_price,
                "atr": atr,
                "spread_points": _spread,
                "cb_allowed": cb_allowed,
                "cb_reason": cb_reason if not cb_allowed else None,
                "wk_allowed": wk_allowed,
                "spread_allowed": spread_allowed,
                "spread_reason": spread_reason if not spread_allowed else None,
                "consecutive_losses": self._risk_manager.consecutive_losses,
                "daily_pnl": self._risk_manager.daily_pnl,
                # V2: structured block analysis
                "blocked_by": (
                    "circuit_breaker" if not cb_allowed else
                    "weekend_manager" if not wk_allowed else
                    "spread_filter" if not spread_allowed else
                    None
                ),
                "protection_stage": self._executor._protection_stage if self._executor.has_position() else None,
            })

        # --- Compute minlot risk % and simplified reason for transparency ---
        _conv = action["conviction"]
        _sl_dist = max(2.5 - _conv, 1.0) * atr
        if hasattr(self, "_broker_constraints") and self._broker_constraints and self._broker_constraints.point > 0:
            _vpp = self._broker_constraints.tick_value / self._broker_constraints.point
        else:
            _vpp = 74.6  # fallback GBP VPP
        _minlot_risk = (0.01 * _sl_dist * _vpp / balance * 100) if balance > 0 else 0.0

        if "OPEN_LONG" in decision or "OPEN_SHORT" in decision:
            _reason = "PROMOTE"
        elif decision == "LOTS_ZERO":
            _reason = "SKIP_CAP"
        elif "low_conviction" in decision:
            _reason = "BLOCKED_CONV"
        elif "high_spread" in decision or "spread_spike" in decision:
            _reason = "BLOCKED_SPREAD"
        elif "CB_BLOCKED" in decision:
            _reason = "CB_BLOCK"
        elif "WK_BLOCKED" in decision:
            _reason = "WK_BLOCK"
        elif decision == "HOLD_FLAT":
            _reason = "HOLD"
        elif decision == "AI_STOPPED":
            _reason = "STOPPED"
        else:
            _reason = decision[:15]

        # --- Structured per-bar log for full observability ---
        log.info(
            "BAR #%d | price=%.2f atr=%.4f | "
            "dir=%+.3f conv=%.3f exit=%.3f sl=%.3f | "
            "decision=%s | minlot_risk=%.1f%% reason=%s | "
            "bal=%.2f eq=%.2f",
            self._step_count, current_price, atr,
            action["direction"], action["conviction"],
            action["exit_urgency"], action["sl_adjustment"],
            decision, _minlot_risk, _reason,
            balance, equity,
        )

        # --- Log observation periodically ---
        if self._live_logger:
            self._live_logger.log_observation(
                {
                    "observation_670": observation,
                    "action": [
                        action["direction"],
                        action["conviction"],
                        action["exit_urgency"],
                        action["sl_adjustment"],
                    ],
                    "decision": decision,
                },
                every_n_bars=self._config.log_observations_every_n_bars,
            )

        # --- Update memory: trend tracker ---
        self._trend_tracker.record_prediction(
            step=self._step_count,
            direction=action["direction"],
            confidence=action["conviction"],
            price=current_price,
        )
        self._trend_tracker.verify_pending(
            step=self._step_count,
            current_price=current_price,
        )

        # --- Update balance history ---
        self._balance_history.append(equity)

        # --- Alert on trade events ---
        if decision.startswith("OPEN_"):
            self._add_alert("INFO", f"Opened {decision[5:]} position")
            self._last_block_reason = ""  # Clear block tracking on trade
        elif decision.startswith("CLOSE_"):
            self._add_alert("INFO", f"Closed position: {decision}")
            self._last_block_reason = ""
        elif decision.startswith("WK_BLOCKED_") or decision.startswith("CB_BLOCKED_"):
            # Alert on first block or when reason changes (avoid spam)
            block_reason = decision.split("_", 2)[-1] if "_" in decision else decision
            if block_reason != self._last_block_reason:
                self._last_block_reason = block_reason
                if decision.startswith("WK_BLOCKED_"):
                    self._add_alert(
                        "WARN",
                        f"Weekend manager blocking trades: {block_reason} "
                        f"(AI active: dir={action['direction']:+.2f} conv={action['conviction']:.2f})",
                    )
                else:
                    self._add_alert(
                        "WARN",
                        f"Circuit breaker blocking trades: {block_reason}",
                    )
        elif decision == "LOTS_ZERO":
            self._add_alert(
                "WARN",
                f"Lot size too small for balance "
                f"(dir={action['direction']:+.2f} conv={action['conviction']:.2f} "
                f"ATR={atr:.2f} bal={balance:.0f})",
            )
            self._last_block_reason = ""
        elif decision.startswith("BLOCKED_"):
            self._add_alert(
                "WARN",
                f"Risk gate blocked: {decision[8:]} "
                f"(dir={action['direction']:+.2f} conv={action['conviction']:.2f})",
            )
            self._last_block_reason = ""
        elif decision in ("HOLD", "FLAT", "HOLD_FLAT", "BELOW_THRESHOLD"):
            # Normal non-trade decisions -- clear block tracking
            self._last_block_reason = ""

        # --- Dashboard trading state sync ---
        executor_state = self._executor.get_state()
        state_map = {
            ExecutorState.STOPPED: DashState.STOPPED,
            ExecutorState.RUNNING: DashState.RUNNING,
            ExecutorState.WINDING_DOWN: DashState.WINDING_DOWN,
        }
        dash_state = state_map.get(executor_state, DashState.STOPPED)
        if self._circuit_breaker.is_paused() and executor_state == ExecutorState.RUNNING:
            dash_state = DashState.CB_PAUSED
        self._dashboard.set_trading_state(dash_state)

        # --- Periodic tasks ---
        self._periodic_tasks()

    def _is_new_bar(self) -> bool:
        """Check if a new M5 bar has closed since last processing.

        Compares the latest M5 bar timestamp from MT5 against the
        last processed bar time.

        Returns:
            True if a new bar is available.
        """
        try:
            bars = self._mt5_bridge.get_latest_bars(
                self._config.mt5_symbol,
                mt5.TIMEFRAME_M5 if mt5 else 5,
                1,
            )
            if bars.empty:
                return False

            bar_time = bars["time"].iloc[-1]
            if hasattr(bar_time, "to_pydatetime"):
                bar_time = bar_time.to_pydatetime()

            if self._last_bar_time is None:
                self._last_bar_time = bar_time
                return True

            if bar_time > self._last_bar_time:
                self._last_bar_time = bar_time
                self._market_status = "OPEN"
                self._bars_since_heartbeat += 1
                return True

            return False
        except Exception:
            log.exception("Error checking for new bar")
            return False

    # ------------------------------------------------------------------
    # Heartbeat -- feedback when market is closed or no new bars
    # ------------------------------------------------------------------

    _HEARTBEAT_INTERVAL_S = 60  # emit a heartbeat alert every 60 seconds

    def _emit_heartbeat_if_due(self) -> None:
        """Emit periodic heartbeat alerts so the user knows the loop is alive.

        Called every 5s when _is_new_bar() returns False.  Only logs
        an alert every _HEARTBEAT_INTERVAL_S seconds to avoid spam.
        """
        now = time.monotonic()

        # First heartbeat after state change
        if self._last_heartbeat_time == 0.0:
            self._last_heartbeat_time = now

        elapsed = now - self._last_heartbeat_time
        if elapsed < self._HEARTBEAT_INTERVAL_S:
            return

        self._last_heartbeat_time = now

        # Determine market status
        is_market_open = False
        try:
            is_market_open = self._mt5_bridge.is_market_open()
        except Exception:
            pass

        if is_market_open:
            self._market_status = "OPEN"
            # Market is open but no new bar yet -- normal between candles
            msg = (
                f"Heartbeat: market OPEN, waiting for next M5 bar "
                f"(last bar: {self._last_bar_time.strftime('%H:%M UTC') if self._last_bar_time else 'none'}, "
                f"bars processed: {self._step_count})"
            )
        else:
            self._market_status = "CLOSED"
            msg = (
                f"Heartbeat: market CLOSED, waiting for market open "
                f"(last bar: {self._last_bar_time.strftime('%Y-%m-%d %H:%M UTC') if self._last_bar_time else 'none'})"
            )

        # Only add alert if executor is in RUNNING state
        executor_state = getattr(self._executor, '_state', None)
        if executor_state is not None and executor_state.value == "running":
            self._add_alert("INFO", msg)
            log.info(msg)

    def _periodic_tasks(self) -> None:
        """Run periodic maintenance tasks: normalizer save, daily reset, etc."""
        now = time.monotonic()

        # Normalizer save (hourly)
        if now - self._last_normalizer_save >= _NORMALIZER_SAVE_INTERVAL_S:
            try:
                self._pipeline.save_normalizer_state()
                self._last_normalizer_save = now
                log.info("Normalizer state saved")
            except Exception:
                log.exception("Failed to save normalizer state")

        # Daily reset at 00:00 UTC
        utc_now = datetime.now(timezone.utc)
        if utc_now.hour == self._config.daily_reset_utc_hour:
            day_of_year = utc_now.timetuple().tm_yday
            if day_of_year != self._last_daily_reset_day:
                try:
                    self._risk_manager.reset_daily()
                    self._circuit_breaker.reset_daily()
                    self._executor.reset_daily()
                    self._last_daily_reset_day = day_of_year
                    self._add_alert("INFO", "Daily reset completed")
                    log.info("Daily reset completed")
                except Exception:
                    log.exception("Daily reset failed -- will retry next cycle")

        # Weekly summary generation (Friday 22:00 UTC)
        if (
            utc_now.weekday() == 4  # Friday
            and utc_now.hour == 22
            and self._live_logger
            and not getattr(self, "_weekly_summary_done", False)
        ):
            try:
                summary = self._live_logger.generate_weekly_summary()
                if summary:
                    log.info("Weekly summary generated: %d trades, PF=%.2f",
                             summary.get("total_trades", 0),
                             summary.get("profit_factor", 0))
                    self._add_alert("INFO", f"Weekly summary: {summary.get('total_trades', 0)} trades, "
                                    f"PF={summary.get('profit_factor', 0):.2f}, "
                                    f"P/L={summary.get('net_pnl', 0):+.2f}")
                self._weekly_summary_done = True
            except Exception:
                log.exception("Weekly summary generation failed")

        # Reset weekly summary flag on Saturday
        if utc_now.weekday() == 5:
            self._weekly_summary_done = False

        # Feature stats logging (every 12 bars)
        if (
            self._config.log_feature_stats
            and self._live_logger
            and self._step_count % 12 == 0
            and self._step_count > 0
        ):
            stats = self._pipeline.get_normalizer_stats()
            self._live_logger.log_feature_stats(
                {"normalizer_stats": {k: v for k, v in stats.items()}}
            )

    @staticmethod
    def _get_current_session() -> str:
        """Return the current trading session based on UTC hour."""
        hour = datetime.now(timezone.utc).hour
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

    def _add_alert(self, level: str, message: str) -> None:
        """Add an alert to the in-memory log and write to disk."""
        entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "level": level,
            "message": message,
        }
        self._alert_log.append(entry)
        if self._live_logger:
            self._live_logger.log_alert(level, message)

    # ==================================================================
    # Dashboard Update (QTimer, every 1 second)
    # ==================================================================

    def _dashboard_update(self) -> None:
        """Refresh all dashboard tabs with current data.

        Called every 1 second by QTimer.  Catches all exceptions
        to prevent dashboard crashes.
        """
        try:
            self._dashboard_update_inner()
        except Exception:
            log.exception("Error in dashboard update")

    def _dashboard_update_inner(self) -> None:
        """Inner dashboard update logic.

        Only the active tab + Tab 1 (Live Status) are updated every
        second.  Other tabs refresh lazily when selected, saving
        ~30-60ms of wasted CPU per update cycle.
        """
        # Guard: need tabs to be built
        if self._tab_live_status is None:
            return  # Dashboard not fully initialized yet

        # Tab 1 (Live Status) always updates -- it's the primary view
        self._tab_live_status.update_data(self._prepare_live_status_data())

        # Only update the currently-visible tab (skip if it's tab 0 / Live Status)
        active_idx = (
            self._dashboard._tabs.currentIndex()
            if self._dashboard and self._dashboard._tabs
            else 0
        )
        if active_idx == 1:
            self._tab_performance.update_data(self._prepare_performance_data())
        elif active_idx == 2:
            self._tab_journal.update_data(self._prepare_journal_data())
        elif active_idx == 3:
            self._tab_model_state.update_data(self._prepare_model_state_data())
        elif active_idx == 4:
            self._tab_alerts.update_data(self._prepare_alerts_data())
        elif active_idx == 5:
            self._tab_analytics.update_data(self._prepare_analytics_data())

    # ==================================================================
    # Data Preparation for Dashboard Tabs
    # ==================================================================

    def _prepare_live_status_data(self) -> Dict[str, Any]:
        """Prepare data dict for Tab 1: Live Status.

        Keys: connection, account, position, today, decisions.
        """
        # Connection
        conn_status = (
            self._connection_monitor.check_connection()
            if self._connection_monitor is not None
            else {"connected": False, "latency_ms": 0.0}
        )
        account = (
            self._mt5_bridge.get_account_info()
            if self._mt5_bridge is not None
            else {}
        )
        spread = (
            self._mt5_bridge.get_current_spread(self._config.mt5_symbol)
            if self._mt5_bridge is not None
            else 0.0
        )

        connection = {
            "connected": conn_status.get("connected", False),
            "server": account.get("server", "--"),
            "latency_ms": conn_status.get("latency_ms", 0.0),
            "spread": round(spread, 2),
        }

        # Account
        account_data = {
            "currency": account.get("currency", "USD"),
            "balance": account.get("balance", 0.0),
            "equity": account.get("equity", 0.0),
            "margin": account.get("margin", 0.0),
            "free_margin": account.get("free_margin", 0.0),
        }

        # Position
        pos = self._executor.get_position() if self._executor else None
        position_data = None
        if pos:
            current_price = 0.0
            bars = self._mt5_bridge.get_latest_bars(
                self._config.mt5_symbol,
                mt5.TIMEFRAME_M5 if mt5 else 5,
                1,
            )
            if not bars.empty:
                current_price = float(bars["close"].iloc[-1])

            # Compute P&L directly from executor position (position_manager
            # is not synced after new trades -- this avoids that gap).
            entry = pos["entry_price"]
            lots = pos["lots"]
            if current_price > 0 and entry > 0:
                sym_info = self._mt5_bridge.get_symbol_info()
                tick_value = sym_info.get("tick_value", 0.745)
                tick_size = sym_info.get("tick_size", 0.01)
                price_move = (
                    (current_price - entry) if pos["side"] == "LONG"
                    else (entry - current_price)
                )
                ticks = price_move / tick_size if tick_size > 0 else 0.0
                pnl = ticks * tick_value * lots
            else:
                pnl = 0.0

            # Duration from open_time stored by executor
            open_time = pos.get("open_time")
            if open_time:
                duration = (datetime.now(timezone.utc) - open_time).total_seconds() / 60.0
            else:
                duration = 0.0

            position_data = {
                "side": pos["side"],
                "lots": lots,
                "entry_price": entry,
                "current_price": current_price,
                "pnl": round(pnl, 2),
                "sl": pos.get("sl", 0.0),
                "tp": pos.get("tp", 0.0),
                "duration_min": int(duration),
                "trailing": pos.get("sl", 0) != self._executor._initial_sl,
            }

        # Today's summary — sourced from the database so data persists
        # across restarts.  Falls back to risk-manager in-memory counters
        # only when the database is unavailable.
        if self._memory is not None:
            today_data = self._memory.get_today_summary()
        else:
            risk_status = (
                self._risk_manager.get_safety_status()
                if self._risk_manager is not None
                else {}
            )
            today_data = {
                "trades": risk_status.get("daily_trade_count", 0),
                "wins": risk_status.get("total_wins", 0),
                "losses": (
                    risk_status.get("total_trades", 0)
                    - risk_status.get("total_wins", 0)
                ),
                "pnl": risk_status.get("daily_pnl", 0.0),
                "win_rate": round(
                    risk_status.get("win_rate", 0.0) * 100, 1
                ),
                "max_dd": 0.0,
                "profit_factor": 0.0,
            }

        # Decisions
        decisions = list(self._decision_log)

        # Market status
        market = {
            "status": self._market_status,
            "last_bar_time": (
                self._last_bar_time.strftime("%Y-%m-%d %H:%M UTC")
                if self._last_bar_time else "--"
            ),
            "bars_processed": self._step_count,
        }

        # Broker constraints snapshot
        broker_snapshot = None
        if hasattr(self, "_broker_constraints") and self._broker_constraints is not None:
            broker_snapshot = self._broker_constraints.get_snapshot()

        return {
            "connection": connection,
            "account": account_data,
            "position": position_data,
            "today": today_data,
            "decisions": decisions,
            "market": market,
            "broker": broker_snapshot,
        }

    def _prepare_performance_data(self) -> Dict[str, Any]:
        """Prepare data dict for Tab 2: Performance.

        Keys: balance_history, metrics, trades.

        Cached for 5 seconds to avoid repeated 500-trade SQL queries.
        Balance history is always fresh (it's in-memory).
        """
        now = time.monotonic()
        balance_hist = list(self._balance_history)

        # Return cached metrics if still fresh (just update balance)
        if self._perf_cache and now - self._perf_cache_time < 5.0:
            self._perf_cache["balance_history"] = balance_hist
            return self._perf_cache

        # Metrics from memory (expensive SQL query)
        trades = (
            self._memory.get_recent_trades(limit=500)
            if self._memory is not None
            else []
        )
        total = len(trades)
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        losses = total - wins
        pnls = [t.get("pnl", 0.0) for t in trades]
        net_pnl = sum(pnls)
        avg_trade = net_pnl / total if total > 0 else 0.0
        best = max(pnls) if pnls else 0.0
        worst = min(pnls) if pnls else 0.0
        win_rate = wins / total if total > 0 else 0.0

        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Max drawdown from balance history
        max_dd = 0.0
        if balance_hist:
            peak = balance_hist[0]
            for b in balance_hist:
                if b > peak:
                    peak = b
                dd = (peak - b) / peak if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd

        # Simple Sharpe estimate (last 30 trades)
        recent_pnls = pnls[:30]
        if len(recent_pnls) >= 5:
            mean_pnl = np.mean(recent_pnls)
            std_pnl = np.std(recent_pnls)
            sharpe = (mean_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0.0
        else:
            sharpe = 0.0

        metrics = {
            "sharpe": round(float(sharpe), 2),
            "win_rate": round(win_rate * 100, 1),
            "pf": round(pf, 2),
            "avg_trade": round(avg_trade, 2),
            "max_dd": round(max_dd * 100, 2),
            "total_trades": total,
            "best": round(best, 2),
            "worst": round(worst, 2),
        }

        # Trade list for table
        trade_list = [
            {
                "id": t.get("id", 0),
                "time": t.get("timestamp", ""),
                "side": t.get("side", ""),
                "lots": t.get("lot_size", 0.0),
                "pnl": t.get("pnl", 0.0),
                "reason": t.get("close_reason", ""),
            }
            for t in trades[:50]
        ]

        result = {
            "balance_history": balance_hist,
            "metrics": metrics,
            "trades": trade_list,
        }
        self._perf_cache = result
        self._perf_cache_time = now
        return result

    def _prepare_journal_data(self) -> Dict[str, Any]:
        """Prepare data dict for Tab 3: Trade Journal.

        Keys: lesson_summary, trades.
        """
        lesson_summary = (
            self._memory.get_lesson_summary()
            if self._memory is not None
            else {}
        )
        trades = (
            self._memory.get_recent_trades(limit=100)
            if self._memory is not None
            else []
        )

        trade_list = []
        for t in trades:
            trade_list.append({
                "id": t.get("id", 0),
                "side": t.get("side", ""),
                "entry_price": t.get("entry_price", 0.0),
                "exit_price": t.get("exit_price", 0.0),
                "exit_reason": t.get("close_reason", ""),
                "pnl": t.get("pnl", 0.0),
                "pnl_pct": t.get("pnl_pct", 0.0),
                "hold_min": t.get("hold_bars", 0) * 5,
                "hold_bars": t.get("hold_bars", 0),
                "lesson": "",  # Populated from journal if available
                "sl_quality": "",
                "sl_saved": 0.0,
                "pattern": "",
                "pattern_wins": 0,
                "pattern_losses": 0,
                "rsi": int(t.get("rsi_at_entry", 0.5) * 100),
                "trend": int(t.get("trend_dir_at_entry", 0)),
                "session": t.get("session_at_entry", ""),
                "volatility": int(t.get("vol_regime_at_entry", 1.0) * 10),
            })

        return {
            "lesson_summary": lesson_summary,
            "trades": trade_list,
        }

    def _prepare_model_state_data(self) -> Dict[str, Any]:
        """Prepare data dict for Tab 4: Model & Features.

        Keys: model_info, last_action, feature_health, correlated_feeds.
        """
        # Model info
        metadata = self._model_components.get("metadata", {})
        engine_info = self._inference.get_model_info() if self._inference else {}

        model_info = {
            "name": self._config.model_path,
            "trained_week": metadata.get("week", "?"),
            "val_sharpe": metadata.get("val_sharpe"),
            "architecture": engine_info.get("policy_class", "SAC"),
            "feature_count": self._config.n_features,
            "obs_dim": self._config.obs_dim,
            "action_count": engine_info.get("action_dim", 4),
        }

        # Last action
        last_action = dict(self._last_action) if self._last_action else {
            "direction": 0.0,
            "conviction": 0.0,
            "exit_urgency": 0.0,
            "sl_adjustment": 0.0,
        }

        # Feature health
        normalizer_stats = (
            self._pipeline.get_normalizer_stats()
            if self._pipeline is not None
            else {}
        )
        filled = sum(1 for s in normalizer_stats.values() if s.get("count", 0) > 0)
        total_norm = len(normalizer_stats)

        feature_health = {
            "active": self._config.n_features,
            "total": self._config.n_features,
            "normalizer_fill": filled,
            "normalizer_capacity": total_norm,
            "frame_fill": (
                self._pipeline.get_frame_buffer_depth()
                if self._pipeline is not None
                else 0
            ),
            "frame_capacity": self._config.frame_stack,
            "nan_rate": 0.0,
            "inf_count": 0,
            "constant_count": 0,
        }
        if self._inference:
            info = self._inference.get_model_info()
            feature_health["nan_rate"] = info.get("nan_warnings", 0)
            feature_health["inf_count"] = info.get("inf_warnings", 0)

        # Correlated feeds
        bar_counts = (
            self._pipeline.get_bar_count()
            if self._pipeline is not None
            else {}
        )
        correlated_feeds = {}
        for sym in ["EURUSD", "XAGUSD", "USDJPY", "US500", "USOIL"]:
            count = bar_counts.get(f"corr_{sym}", 0)
            status = "OK" if count > 50 else "STALE"
            correlated_feeds[sym] = {
                "status": status,
                "age_sec": 0.0,
                "note": f"{count} bars",
            }

        # Calendar info
        calendar_info = (
            self._pipeline.get_calendar_info()
            if self._pipeline is not None
            else {}
        )

        # Reward normalizer state (from model package)
        reward_state = self._model_components.get("reward_state", {})

        return {
            "model_info": model_info,
            "last_action": last_action,
            "feature_health": feature_health,
            "correlated_feeds": correlated_feeds,
            "calendar_info": calendar_info,
            "reward_state": reward_state,
        }

    def _prepare_alerts_data(self) -> Dict[str, Any]:
        """Prepare data dict for Tab 5: Alerts & Safety.

        Keys: safety_status, alerts, daily_risk, trading_state.
        """
        # Safety status
        cb_status = (
            self._circuit_breaker.get_status()
            if self._circuit_breaker is not None
            else {}
        )
        wk_status = (
            self._weekend_manager.get_status()
            if self._weekend_manager is not None
            else {}
        )
        conn_status = (
            self._connection_monitor.check_connection()
            if self._connection_monitor is not None
            else {"connected": False, "latency_ms": 0}
        )

        executor_state = (
            self._executor.get_state()
            if self._executor is not None
            else ExecutorState.STOPPED
        )
        state_map = {
            ExecutorState.STOPPED: "STOPPED",
            ExecutorState.RUNNING: "RUNNING",
            ExecutorState.WINDING_DOWN: "WINDING DOWN",
        }
        state_str = state_map.get(executor_state, "STOPPED")
        if cb_status.get("is_paused") and executor_state == ExecutorState.RUNNING:
            state_str = "CB PAUSED"

        # Circuit breaker display
        if cb_status.get("is_paused"):
            pause_s = cb_status.get("pause_remaining_s", 0)
            mins = pause_s // 60
            secs = pause_s % 60
            cb_display = f"PAUSED ({mins}:{secs:02d} remaining)"
        elif cb_status.get("daily_halted"):
            cb_display = "DAILY HALT"
        elif cb_status.get("weekly_halted"):
            cb_display = "WEEKLY HALT"
        else:
            cb_display = "OFF"

        # Weekend display
        if wk_status.get("is_weekend"):
            wk_display = wk_status.get("reason", "Weekend")
        else:
            hours = wk_status.get("hours_to_friday_close", 0)
            wk_display = f"{hours:.1f}h to Friday close"

        # Connection display
        if conn_status.get("connected"):
            latency = conn_status.get("latency_ms", 0)
            if latency > 200:
                conn_display = "Unstable"
            else:
                conn_display = "Stable"
        else:
            conn_display = "Disconnected"

        safety_status = {
            "state": state_str,
            "circuit_breaker": cb_display,
            "weekend_close": wk_display,
            "daily_dd_current": cb_status.get("daily_dd_pct", 0.0),
            "daily_dd_limit": self._config.daily_dd_close_all_pct * 100,
            "consec_losses": cb_status.get("consecutive_losses", 0),
            "connection": conn_display,
        }

        # Alerts
        alerts = list(self._alert_log)

        # Daily risk
        risk_status = (
            self._risk_manager.get_safety_status()
            if self._risk_manager is not None
            else {}
        )
        account = (
            self._mt5_bridge.get_account_info()
            if self._mt5_bridge is not None
            else {}
        )
        equity = account.get("equity", 0.0) if account else 0.0
        total_dd = 0.0
        if self._peak_balance > 0:
            total_dd = ((self._peak_balance - equity) / self._peak_balance) * 100

        daily_risk = {
            "trades_current": risk_status.get("daily_trade_count", 0),
            "trades_soft_cap": 10,
            "trades_hard_cap": self._config.daily_trade_hard_cap,
            "daily_dd_current": cb_status.get("daily_dd_pct", 0.0),
            "daily_dd_halt": self._config.daily_dd_halt_pct * 100,
            "total_dd_current": round(total_dd, 2),
            "total_dd_limit": self._config.max_dd * 100,
            "equity": equity,
        }

        return {
            "safety_status": safety_status,
            "alerts": alerts,
            "daily_risk": daily_risk,
            "trading_state": state_str,
        }

    def _prepare_analytics_data(self) -> Dict[str, Any]:
        """Prepare data dict for Tab 6: Analytics & Diagnostics.

        Keys: action_history, training_comparison, session_breakdown,
              day_of_week, feature_drift, correlation_drift, weekly_reports.
        """
        # Action history (last 500 actions)
        actions = list(self._action_history)
        direction_vals = [a.get("direction", 0.0) for a in actions]
        conviction_vals = [a.get("conviction", 0.0) for a in actions]
        exit_vals = [a.get("exit_urgency", 0.0) for a in actions]
        sl_vals = [a.get("sl_adjustment", 0.0) for a in actions]

        # Flat rate (bars where |direction| < threshold)
        flat_count = sum(
            1 for d in direction_vals
            if abs(d) < self._config.direction_threshold
        )
        flat_rate = flat_count / len(direction_vals) if direction_vals else 0.0

        action_history = {
            "direction": direction_vals,
            "conviction": conviction_vals,
            "exit": exit_vals,
            "sl_adj": sl_vals,
            "flat_rate": round(flat_rate, 3),
            "avg_trades_day": 0.0,
        }

        # Training comparison -- compute live metrics from trade history
        metadata = self._model_components.get("metadata", {})
        stress = self._model_components.get("stress_results", {})
        training_comparison = {}

        # Compute live metrics from recent trades
        recent_trades = (
            self._memory.get_recent_trades(limit=500)
            if self._memory is not None
            else []
        )
        live_metrics: Dict[str, Any] = {}
        if recent_trades:
            pnls = [t.get("pnl", 0.0) for t in recent_trades]
            total_t = len(pnls)
            wins_t = sum(1 for p in pnls if p > 0)
            gross_profit = sum(p for p in pnls if p > 0)
            gross_loss = abs(sum(p for p in pnls if p < 0))
            live_metrics["PF"] = round(
                gross_profit / gross_loss if gross_loss > 0 else 0.0, 2
            )
            live_metrics["Win Rate"] = f"{wins_t / total_t * 100:.0f}%" if total_t else "--"
            # Max drawdown from balance series
            bal = self._initial_balance or 1000.0
            peak = bal
            max_dd = 0.0
            for p in reversed(pnls):  # oldest first
                bal += p
                if bal > peak:
                    peak = bal
                dd = (peak - bal) / peak if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd
            live_metrics["MaxDD"] = f"{max_dd * 100:.1f}%"
            unique_days = set(
                t.get("timestamp", "")[:10] for t in recent_trades
                if t.get("timestamp")
            )
            live_metrics["Trades/Day"] = round(
                total_t / max(1, len(unique_days)), 1
            )
            # TIM% (time-in-market) — approximate from hold_bars
            total_bars = self._step_count or 1
            bars_in_market = sum(t.get("hold_bars", 0) for t in recent_trades)
            live_metrics["TIM%"] = f"{bars_in_market / total_bars * 100:.1f}%"
            # Avg Hold (minutes)
            avg_hold_bars = np.mean(
                [t.get("hold_bars", 0) for t in recent_trades]
            )
            live_metrics["Avg Hold"] = f"{avg_hold_bars * 5:.0f}m"
            # Sharpe (annualised estimate from trade P/Ls)
            if len(pnls) >= 5:
                mean_pnl = np.mean(pnls)
                std_pnl = np.std(pnls)
                sharpe_est = (
                    mean_pnl / std_pnl * np.sqrt(252) if std_pnl > 0 else 0.0
                )
                live_metrics["Sharpe"] = round(float(sharpe_est), 2)

        # Build comparison table: training (from metadata) vs live
        train_metrics = {
            "PF": metadata.get("val_pf", metadata.get("profit_factor", "--")),
            "Win Rate": metadata.get("val_win_rate", "--"),
            "MaxDD": metadata.get("val_max_dd", "--"),
            "Trades/Day": metadata.get("val_trades_per_day", "--"),
            "TIM%": metadata.get("val_tim_pct", "--"),
            "Avg Hold": metadata.get("val_avg_hold", "--"),
            "Sharpe": metadata.get("val_sharpe", "--"),
        }
        # If stress results available, use those for training columns
        if stress:
            for metric_name, values in stress.items():
                if isinstance(values, dict):
                    training_comparison[metric_name] = {
                        "val": values.get("val", "--"),
                        "test": values.get("test", "--"),
                        "live": live_metrics.get(metric_name, "--"),
                        "status": "green",
                    }
        else:
            # Fall back to metadata + live
            for metric_name in ["PF", "Win Rate", "MaxDD", "Trades/Day", "TIM%", "Avg Hold", "Sharpe"]:
                tv = train_metrics.get(metric_name, "--")
                lv = live_metrics.get(metric_name, "--")
                training_comparison[metric_name] = {
                    "val": tv,
                    "test": "--",
                    "live": lv,
                    "status": "green" if lv != "--" else "yellow",
                }

        # Session breakdown
        session_breakdown = (
            self._memory.get_session_breakdown()
            if self._memory is not None
            else {}
        )

        # Day of week
        day_of_week = (
            self._memory.get_day_of_week_breakdown()
            if self._memory is not None
            else {}
        )

        # Feature drift -- compare live normalizer stats to initial warmup
        normalizer_stats = (
            self._pipeline.get_normalizer_stats()
            if self._pipeline is not None
            else {}
        )
        drifted_features = []
        total_baselined = 0
        within_threshold = 0
        for fname, stats in normalizer_stats.items():
            count = stats.get("count", 0)
            if count < 50:
                continue  # Not enough data yet
            total_baselined += 1
            mean = stats.get("mean", 0.0)
            std = stats.get("std", 1.0)
            # For z-scored features, mean should be near 0, std near 1.
            # Drift = |mean| > 2.0 (feature mean shifted 2+ sigma)
            if abs(mean) <= 2.0:
                within_threshold += 1
            else:
                drifted_features.append({
                    "name": fname,
                    "live_mean": round(mean, 3),
                    "train_mean": 0.0,
                    "sigma_distance": round(abs(mean), 2),
                })

        feature_drift = {
            "total_baselined": total_baselined,
            "within_threshold": within_threshold,
            "drifted_features": drifted_features,
        }

        # Correlation drift -- basic stability check
        # Compare mean absolute z-score across features as a proxy
        if normalizer_stats:
            abs_means = [
                abs(s.get("mean", 0.0))
                for s in normalizer_stats.values()
                if s.get("count", 0) >= 50
            ]
            corr_score = round(np.mean(abs_means), 3) if abs_means else 0.0
            if corr_score > 1.5:
                corr_status = "CRITICAL"
            elif corr_score > 0.8:
                corr_status = "WARNING"
            else:
                corr_status = "OK"
        else:
            corr_score = 0.0
            corr_status = "OK"

        correlation_drift = {
            "score": corr_score,
            "status": corr_status,
            "yellow_consecutive": 0,
            "red_consecutive": 0,
        }

        # Weekly reports
        weekly_reports = []
        if self._live_logger:
            weekly_reports = self._live_logger.get_weekly_summaries()

        return {
            "action_history": action_history,
            "training_comparison": training_comparison,
            "session_breakdown": session_breakdown,
            "day_of_week": day_of_week,
            "feature_drift": feature_drift,
            "correlation_drift": correlation_drift,
            "weekly_reports": weekly_reports,
        }

    # ==================================================================
    # Clean Shutdown
    # ==================================================================

    def _shutdown(self) -> None:
        """Clean up all resources on exit."""
        log.info("Shutting down Spartus Live Dashboard...")

        # Stop timers
        if self._trading_timer:
            self._trading_timer.stop()
        if self._dashboard_timer:
            self._dashboard_timer.stop()

        # Save normalizer state
        if self._pipeline:
            try:
                self._pipeline.save_normalizer_state()
                log.info("Normalizer state saved on shutdown")
            except Exception:
                log.exception("Failed to save normalizer state on shutdown")

        # Disconnect MT5
        if self._mt5_bridge:
            try:
                self._mt5_bridge.disconnect()
            except Exception:
                log.exception("Error disconnecting MT5")

        # Close memory DB
        if self._memory:
            try:
                self._memory.close()
            except Exception:
                log.exception("Error closing memory DB")

        # Cleanup model loader temp dir
        if self._model_loader:
            try:
                self._model_loader.cleanup()
            except Exception:
                log.exception("Error cleaning up model loader")

        log.info("Shutdown complete")


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Spartus Live Trading Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                    # Normal launch\n"
            "  python main.py --config my.yaml   # Custom config\n"
            "  python main.py --paper            # Force paper trading\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: config/default_config.yaml)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=False,
        help="Force paper trading mode (overrides config)",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    log.info("Spartus Live Trading Dashboard starting...")
    log.info("Base directory: %s", BASE_DIR)
    log.info("Python: %s", sys.version)
    log.info("Time (UTC): %s", datetime.now(timezone.utc).isoformat())

    # Create orchestrator
    orchestrator = SpartusOrchestrator(args)

    # Install signal handlers for clean shutdown (Ctrl+C)
    def signal_handler(signum, frame):
        log.info("Signal %d received -- initiating shutdown", signum)
        if orchestrator._app:
            orchestrator._app.quit()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize all components
    if not orchestrator.initialize():
        log.error("Initialization failed -- launching dashboard in limited mode")
        # Still try to launch dashboard for diagnostics
        # But if MT5 is completely unavailable, we cannot proceed
        if mt5 is None:
            log.error("MetaTrader5 not installed -- cannot launch")
            return 1

    # Launch dashboard (blocks until window closes)
    exit_code = orchestrator.launch()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
