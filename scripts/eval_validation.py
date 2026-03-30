"""Validation evaluation harness — read-only, deterministic evaluation.

Components (per third-party audit specification):
    1. Base validation eval (current costs, full validation set)
    2. Stress test matrix (6 cost scenarios)
    3. Regime segmentation (ATR quartiles)
    4. Churn diagnostic (trade frequency & edge)
    5. R5 ablation (reward contribution analysis)
    6. Test set: NOT TOUCHED

Usage:
    python scripts/eval_validation.py [--model PATH] [--quick]
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import SAC

from src.config import TrainingConfig
from src.data.feature_builder import FeatureBuilder
from src.data.normalizer import ExpandingWindowNormalizer
from src.environment.trade_env import SpartusTradeEnv, PRECOMPUTED_FEATURES
from src.memory.trading_memory import TradingMemory


class ValidationEvaluator:
    """Read-only evaluation of a trained model on the validation set."""

    def __init__(self, model_path: str = None, config: TrainingConfig = None):
        self.cfg = config or TrainingConfig()
        self.model_path = Path(model_path) if model_path else self.cfg.best_model_path

        self.feature_builder = FeatureBuilder(self.cfg)
        self.normalizer = ExpandingWindowNormalizer(self.cfg)

        # Eval config: disable all randomization for deterministic results
        self._eval_cfg = self._make_eval_config()

        # Data
        self._weeks_data: List[Dict] = []
        self._train_weeks: List[int] = []
        self._val_weeks: List[int] = []
        self._test_weeks: List[int] = []
        self._features_cache: Dict[int, pd.DataFrame] = {}

        # Results (populated by components)
        self.base_trades: List[Dict] = []
        self.base_weekly_returns: List[float] = []
        self.base_reward_components: List[Dict] = []
        self.base_time_in_market: List[float] = []
        self.base_total_steps: int = 0
        self.model = None

    def _make_eval_config(self, **overrides) -> TrainingConfig:
        """Create a deterministic eval config (no randomization)."""
        cfg = deepcopy(self.cfg)
        cfg.start_offset_max = 0
        cfg.observation_noise_std = 0.0
        cfg.spread_jitter = 0.0
        cfg.slippage_jitter = 0.0
        cfg.commission_jitter = 0.0
        for key, val in overrides.items():
            setattr(cfg, key, val)
        return cfg

    # =========================================================================
    # Main entry
    # =========================================================================

    def run_all(self, quick: bool = False):
        print("=" * 70)
        print("SPARTUS VALIDATION EVALUATION HARNESS")
        print("=" * 70)
        print()

        # Load model
        if not self.model_path.exists():
            print("ERROR: Model not found at {}".format(self.model_path))
            return
        print("Loading model: {}".format(self.model_path))
        self.model = SAC.load(str(self.model_path))
        print("Model loaded. obs_dim={}".format(
            self.model.observation_space.shape[0]))

        # Load metadata if available
        meta_path = self.model_path.with_suffix(".meta.json")
        if meta_path.exists():
            import json
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            print("Model from week {}, val_sharpe={:.4f}".format(
                meta.get("week", "?"), meta.get("val_sharpe", 0.0)))
        print()

        # Discover and split weeks
        self._discover_weeks()
        self._split_weeks()

        if not self._val_weeks:
            print("ERROR: No validation weeks found.")
            return

        val_info = [self._weeks_data[i] for i in self._val_weeks]
        print("Validation set: {} weeks ({}-W{:02d} to {}-W{:02d})".format(
            len(self._val_weeks),
            val_info[0]["year"], val_info[0]["week"],
            val_info[-1]["year"], val_info[-1]["week"],
        ))
        print("Test set: {} weeks (NOT TOUCHED)".format(len(self._test_weeks)))
        print()

        # Component 1
        t0 = time.time()
        print("=" * 70)
        print("COMPONENT 1: BASE VALIDATION EVAL")
        print("=" * 70)
        self._run_base_eval()
        print("  [Completed in {:.1f}s]".format(time.time() - t0))
        print()

        # Component 2
        if not quick:
            t0 = time.time()
            print("=" * 70)
            print("COMPONENT 2: STRESS TEST MATRIX")
            print("=" * 70)
            self._run_stress_tests()
            print("  [Completed in {:.1f}s]".format(time.time() - t0))
            print()

        # Component 3
        t0 = time.time()
        print("=" * 70)
        print("COMPONENT 3: REGIME SEGMENTATION (ATR quartiles)")
        print("=" * 70)
        self._run_regime_segmentation()
        print("  [Completed in {:.1f}s]".format(time.time() - t0))
        print()

        # Component 4
        t0 = time.time()
        print("=" * 70)
        print("COMPONENT 4: CHURN DIAGNOSTIC")
        print("=" * 70)
        self._run_churn_diagnostic()
        print("  [Completed in {:.1f}s]".format(time.time() - t0))
        print()

        # Component 5
        t0 = time.time()
        print("=" * 70)
        print("COMPONENT 5: R5 ABLATION (reward contribution)")
        print("=" * 70)
        self._run_r5_ablation()
        print("  [Completed in {:.1f}s]".format(time.time() - t0))
        print()

        print("=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)

    # =========================================================================
    # Data discovery (replicated from trainer)
    # =========================================================================

    def _discover_weeks(self):
        self._weeks_data.clear()
        data_dir = Path(self.cfg.data_dir)

        for year_dir in sorted(data_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            try:
                year = int(year_dir.name)
            except ValueError:
                continue

            for wk in range(1, 54):
                m5_path = year_dir / "week_{:02d}_M5.parquet".format(wk)
                h1_path = year_dir / "week_{:02d}_H1.parquet".format(wk)
                if m5_path.exists() and h1_path.exists():
                    self._weeks_data.append({
                        "year": year,
                        "week": wk,
                        "m5_path": m5_path,
                        "h1_path": h1_path,
                        "h4_path": year_dir / "week_{:02d}_H4.parquet".format(wk),
                        "d1_path": year_dir / "week_{:02d}_D1.parquet".format(wk),
                    })

        print("Discovered {} total weeks of data".format(len(self._weeks_data)))

    def _split_weeks(self):
        n = len(self._weeks_data)
        purge = 2
        train_end = int(n * 0.70)
        val_end = train_end + purge + int(n * 0.15)

        self._train_weeks = list(range(0, train_end))
        self._val_weeks = list(range(train_end + purge, min(val_end, n)))
        self._test_weeks = list(range(min(val_end + purge, n), n))

        print("Split: {} train, {} val, {} test (purge={})".format(
            len(self._train_weeks), len(self._val_weeks),
            len(self._test_weeks), purge))

    # =========================================================================
    # Feature loading (replicated from trainer)
    # =========================================================================

    def _get_features_cached(self, week_idx: int) -> Optional[pd.DataFrame]:
        if week_idx in self._features_cache:
            return self._features_cache[week_idx]
        try:
            df = self._get_features(self._weeks_data[week_idx])
            self._features_cache[week_idx] = df
            return df
        except Exception as e:
            wd = self._weeks_data[week_idx]
            print("  WARNING: Feature load failed {}-W{:02d}: {}".format(
                wd["year"], wd["week"], e))
            return None

    def _get_features(self, week_data: Dict) -> pd.DataFrame:
        cache_path = (
            self.cfg.feature_dir
            / str(week_data["year"])
            / "week_{:02d}_features.parquet".format(week_data["week"])
        )

        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            expected = set(PRECOMPUTED_FEATURES)
            present = set(cached.columns) & expected
            if len(present) >= len(expected):
                return cached

        # Build from raw data
        m5_frames, h1_frames, h4_frames, d1_frames = [], [], [], []
        year, wk = week_data["year"], week_data["week"]

        for offset in range(-2, 1):
            target_wk = wk + offset
            target_year = year
            if target_wk < 1:
                target_year -= 1
                target_wk += 52
            elif target_wk > 52:
                target_year += 1
                target_wk -= 52

            for tf, fl in [("M5", m5_frames), ("H1", h1_frames),
                           ("H4", h4_frames), ("D1", d1_frames)]:
                p = Path(self.cfg.data_dir) / str(target_year) / "week_{:02d}_{}.parquet".format(target_wk, tf)
                if p.exists():
                    fl.append(pd.read_parquet(p))

        empty = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        m5 = pd.concat(m5_frames, ignore_index=True).sort_values("time").reset_index(drop=True) if m5_frames else empty.copy()
        h1 = pd.concat(h1_frames, ignore_index=True).sort_values("time").reset_index(drop=True) if h1_frames else empty.copy()
        h4 = pd.concat(h4_frames, ignore_index=True).sort_values("time").reset_index(drop=True) if h4_frames else empty.copy()
        d1 = pd.concat(d1_frames, ignore_index=True).sort_values("time").reset_index(drop=True) if d1_frames else empty.copy()

        if m5.empty:
            raise ValueError("No M5 data for year={} week={}".format(year, wk))

        correlated_m5 = self._load_correlated_m5(year, wk)
        features = self.feature_builder.build_features(m5, h1, h4, d1, correlated_m5=correlated_m5)

        for col in ("rsi_14", "atr_ratio", "h1_trend_dir"):
            if col in features.columns:
                features["{}_raw".format(col)] = features[col].copy()

        features = self.normalizer.normalize(features)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(cache_path, engine="pyarrow", index=False)
        return features

    def _load_correlated_m5(self, year: int, wk: int) -> Dict[str, pd.DataFrame]:
        correlated = {}
        empty = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        for symbol in self.cfg.correlated_symbols:
            frames = []
            for offset in range(-2, 1):
                target_wk = wk + offset
                target_year = year
                if target_wk < 1:
                    target_year -= 1
                    target_wk += 52
                elif target_wk > 52:
                    target_year += 1
                    target_wk -= 52

                path = (
                    self.cfg.correlated_data_dir / symbol
                    / str(target_year)
                    / "week_{:02d}_M5.parquet".format(target_wk)
                )
                if path.exists():
                    frames.append(pd.read_parquet(path))

            if frames:
                correlated[symbol] = pd.concat(
                    frames, ignore_index=True
                ).sort_values("time").reset_index(drop=True)
            else:
                correlated[symbol] = empty.copy()

        return correlated

    # =========================================================================
    # Rollout engine
    # =========================================================================

    def _rollout_week(self, week_idx: int, eval_cfg: TrainingConfig = None,
                      seed: int = 42) -> Optional[Dict]:
        """Run one deterministic rollout through a validation week."""
        week_data = self._weeks_data[week_idx]
        cfg = eval_cfg or self._eval_cfg

        features_df = self._get_features_cached(week_idx)
        if features_df is None:
            return None

        if len(features_df) < cfg.lookback + 10:
            return None

        # Use in-memory SQLite to avoid locking the training DB
        memory = TradingMemory(db_path=":memory:", config=cfg)
        env = SpartusTradeEnv(
            features_df=features_df,
            config=cfg,
            memory=memory,
            initial_balance=cfg.initial_balance,
            week=week_data["week"],
            seed=seed,
            is_validation=True,
        )

        obs, info = env.reset()

        trades = []
        equity_curve = [cfg.initial_balance]
        reward_components = []
        step_positions = []
        done = False
        truncated = False

        while not done and not truncated:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            equity_curve.append(info.get("equity", env.balance))
            step_positions.append(info.get("has_position", False))

            reward_components.append({
                "reward": info.get("reward", reward),
                "raw_reward": info.get("raw_reward", reward),
                "r1": info.get("r1_position_pnl", 0.0),
                "r2": info.get("r2_trade_quality", 0.0),
                "r3": info.get("r3_drawdown", 0.0),
                "r4": info.get("r4_sharpe", 0.0),
                "r5": info.get("r5_risk_bonus", 0.0),
            })

            if "last_trade" in info:
                trade = info["last_trade"].copy()
                trade["week_year"] = week_data["year"]
                trade["week_num"] = week_data["week"]
                # ATR at entry for regime segmentation
                entry_step = max(0, info.get("step", 0) - 1 - trade.get("hold_bars", 0))
                bar_idx = min(cfg.lookback + entry_step, len(features_df) - 1)
                if "atr_14_raw" in features_df.columns:
                    trade["atr_at_entry"] = float(
                        features_df.iloc[bar_idx].get("atr_14_raw", 1.0))
                else:
                    trade["atr_at_entry"] = 1.0
                trades.append(trade)

        time_in_market = sum(step_positions) / max(len(step_positions), 1)

        return {
            "trades": trades,
            "equity_curve": equity_curve,
            "reward_components": reward_components,
            "time_in_market": time_in_market,
            "total_steps": len(step_positions),
            "week_year": week_data["year"],
            "week_num": week_data["week"],
            "final_balance": env.balance,
        }

    # =========================================================================
    # Component 1: Base validation eval
    # =========================================================================

    def _run_base_eval(self):
        all_trades = []
        weekly_returns = []
        time_in_market = []
        reward_components = []
        total_steps = 0

        for i, week_idx in enumerate(self._val_weeks):
            wd = self._weeks_data[week_idx]
            label = "{}-W{:02d}".format(wd["year"], wd["week"])
            print("  [{}/{}] {} ...".format(i + 1, len(self._val_weeks), label),
                  end="", flush=True)

            result = self._rollout_week(week_idx)
            if result is None:
                print(" SKIP")
                continue

            all_trades.extend(result["trades"])
            reward_components.extend(result["reward_components"])
            time_in_market.append(result["time_in_market"])
            total_steps += result["total_steps"]

            ret = (result["final_balance"] - self.cfg.initial_balance) / self.cfg.initial_balance
            weekly_returns.append(ret)

            n_t = len(result["trades"])
            w_pnl = sum(t["pnl"] for t in result["trades"])
            print(" {} trades, P/L: {:+.2f}".format(n_t, w_pnl))

        # Store for other components
        self.base_trades = all_trades
        self.base_weekly_returns = weekly_returns
        self.base_reward_components = reward_components
        self.base_time_in_market = time_in_market
        self.base_total_steps = total_steps

        self._print_metrics("BASE EVAL", all_trades, weekly_returns,
                            time_in_market, total_steps)

    def _print_metrics(self, label: str, trades: List[Dict],
                       weekly_returns: List[float],
                       time_in_market: List[float], total_steps: int):
        print()
        print("--- {} METRICS ---".format(label))
        print()

        if not trades:
            print("  NO TRADES")
            return

        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        net_pnl = sum(pnls)
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = sum(losses) if losses else 0.0
        pf = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

        # Sharpe & Sortino (annualized from weekly returns)
        sharpe, sortino = 0.0, 0.0
        if len(weekly_returns) > 1:
            mr = np.mean(weekly_returns)
            sr = np.std(weekly_returns, ddof=1)
            sharpe = (mr / sr) * np.sqrt(52) if sr > 0 else 0.0
            down = [r for r in weekly_returns if r < 0]
            ds = np.std(down, ddof=1) if len(down) > 1 else 0.0
            sortino = (mr / ds) * np.sqrt(52) if ds > 0 else 0.0

        # Max DD from cumulative weekly equity
        equity = [self.cfg.initial_balance]
        for r in weekly_returns:
            equity.append(equity[-1] * (1 + r))
        eq_arr = np.array(equity)
        peak = np.maximum.accumulate(eq_arr)
        dd = (peak - eq_arr) / (peak + 1e-12)
        max_dd = float(np.max(dd))

        hold_bars = [t.get("hold_bars", 0) for t in trades]
        commissions = sum(t.get("commission", 0.0) for t in trades)
        avg_tim = np.mean(time_in_market) if time_in_market else 0.0

        rows = [
            ("Total Trades", "{:d}".format(len(trades))),
            ("Winning Trades", "{:d}".format(len(wins))),
            ("Losing Trades", "{:d}".format(len(losses))),
            ("Win Rate", "{:.1f}%".format(len(wins) / len(trades) * 100)),
            ("Net P/L", "{:.4f}".format(net_pnl)),
            ("Gross Profit", "{:.4f}".format(gross_profit)),
            ("Gross Loss", "{:.4f}".format(gross_loss)),
            ("Profit Factor", "{:.3f}".format(pf) if pf < 1000 else "inf"),
            ("Sharpe (annualized)", "{:.3f}".format(sharpe)),
            ("Sortino (annualized)", "{:.3f}".format(sortino)),
            ("Max Drawdown", "{:.1f}%".format(max_dd * 100)),
            ("Avg Hold Bars", "{:.1f}".format(np.mean(hold_bars))),
            ("Median Hold Bars", "{:.1f}".format(np.median(hold_bars))),
            ("Avg P/L per Trade", "{:.4f}".format(np.mean(pnls))),
            ("Avg Time in Market", "{:.1f}%".format(avg_tim * 100)),
            ("Total Steps", "{:d}".format(total_steps)),
            ("Total Commissions", "{:.4f}".format(commissions)),
            ("Validation Weeks", "{:d}".format(len(weekly_returns))),
        ]

        print("  {:<30} {:>15}".format("Metric", "Value"))
        print("  {} {}".format("-" * 30, "-" * 15))
        for name, val in rows:
            print("  {:<30} {:>15}".format(name, val))

    # =========================================================================
    # Component 2: Stress test matrix
    # =========================================================================

    def _run_stress_tests(self):
        scenarios = {
            "2x_spread": dict(
                spread_london_pips=self.cfg.spread_london_pips * 2,
                spread_ny_pips=self.cfg.spread_ny_pips * 2,
                spread_asia_pips=self.cfg.spread_asia_pips * 2,
                spread_off_hours_pips=self.cfg.spread_off_hours_pips * 2,
            ),
            "3x_spread": dict(
                spread_london_pips=self.cfg.spread_london_pips * 3,
                spread_ny_pips=self.cfg.spread_ny_pips * 3,
                spread_asia_pips=self.cfg.spread_asia_pips * 3,
                spread_off_hours_pips=self.cfg.spread_off_hours_pips * 3,
            ),
            "2x_slip_mean": dict(
                slippage_mean_pips=self.cfg.slippage_mean_pips * 2,
            ),
            "2x_slip_std": dict(
                slippage_std_pips=self.cfg.slippage_std_pips * 2,
            ),
            "combined_2x2x": dict(
                spread_london_pips=self.cfg.spread_london_pips * 2,
                spread_ny_pips=self.cfg.spread_ny_pips * 2,
                spread_asia_pips=self.cfg.spread_asia_pips * 2,
                spread_off_hours_pips=self.cfg.spread_off_hours_pips * 2,
                slippage_mean_pips=self.cfg.slippage_mean_pips * 2,
            ),
            "5pct_spike": None,  # handled specially
        }

        results = {}
        results["base"] = self._compute_summary(
            self.base_trades, self.base_weekly_returns)

        for name, overrides in scenarios.items():
            print("  Scenario: {} ...".format(name), flush=True)
            all_trades = []
            weekly_returns = []

            if name == "5pct_spike":
                for week_idx in self._val_weeks:
                    result = self._rollout_week_with_spikes(
                        week_idx, spike_prob=0.05, spike_mult=3.0)
                    if result is None:
                        continue
                    all_trades.extend(result["trades"])
                    ret = (result["final_balance"] - self.cfg.initial_balance) / self.cfg.initial_balance
                    weekly_returns.append(ret)
            else:
                stress_cfg = self._make_eval_config(**overrides)
                for week_idx in self._val_weeks:
                    result = self._rollout_week(week_idx, eval_cfg=stress_cfg)
                    if result is None:
                        continue
                    all_trades.extend(result["trades"])
                    ret = (result["final_balance"] - self.cfg.initial_balance) / self.cfg.initial_balance
                    weekly_returns.append(ret)

            results[name] = self._compute_summary(all_trades, weekly_returns)
            s = results[name]
            pf_s = "{:.3f}".format(s["profit_factor"]) if s["profit_factor"] < 1000 else "inf"
            print("    -> {} trades, Net: {:+.4f}, PF: {}".format(
                s["n_trades"], s["net_pnl"], pf_s))

        self._print_stress_table(results)

    def _rollout_week_with_spikes(self, week_idx: int,
                                  spike_prob: float = 0.05,
                                  spike_mult: float = 3.0,
                                  seed: int = 42) -> Optional[Dict]:
        """Rollout with random spread spikes injected at spike_prob of steps."""
        week_data = self._weeks_data[week_idx]
        cfg = self._eval_cfg

        features_df = self._get_features_cached(week_idx)
        if features_df is None or len(features_df) < cfg.lookback + 10:
            return None

        memory = TradingMemory(db_path=":memory:", config=cfg)
        env = SpartusTradeEnv(
            features_df=features_df,
            config=cfg,
            memory=memory,
            initial_balance=cfg.initial_balance,
            week=week_data["week"],
            seed=seed,
            is_validation=True,
        )
        obs, _ = env.reset()

        # Monkey-patch spread to inject spikes
        orig_get_spread = env.sim.get_spread
        spike_rng = np.random.RandomState(seed + week_idx)

        def spiked_get_spread(hour, is_news=False):
            spread = orig_get_spread(hour, is_news)
            if spike_rng.random() < spike_prob:
                spread *= spike_mult
            return spread

        env.sim.get_spread = spiked_get_spread

        trades = []
        done, truncated = False, False
        while not done and not truncated:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            if "last_trade" in info:
                trade = info["last_trade"].copy()
                trade["week_year"] = week_data["year"]
                trade["week_num"] = week_data["week"]
                trades.append(trade)

        return {"trades": trades, "final_balance": env.balance}

    def _compute_summary(self, trades: List[Dict],
                         weekly_returns: List[float]) -> Dict:
        if not trades:
            return dict(n_trades=0, win_rate=0, net_pnl=0, gross_profit=0,
                        gross_loss=0, profit_factor=0, sharpe=0, sortino=0,
                        max_dd=0, avg_hold=0, avg_pnl=0)

        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gp = sum(wins) if wins else 0.0
        gl = sum(losses) if losses else 0.0
        pf = abs(gp / gl) if gl != 0 else float("inf")

        sharpe, sortino = 0.0, 0.0
        if len(weekly_returns) > 1:
            mr = np.mean(weekly_returns)
            sr = np.std(weekly_returns, ddof=1)
            sharpe = (mr / sr) * np.sqrt(52) if sr > 0 else 0.0
            down = [r for r in weekly_returns if r < 0]
            ds = np.std(down, ddof=1) if len(down) > 1 else 0.0
            sortino = (mr / ds) * np.sqrt(52) if ds > 0 else 0.0

        equity = [self.cfg.initial_balance]
        for r in weekly_returns:
            equity.append(equity[-1] * (1 + r))
        eq_arr = np.array(equity)
        peak = np.maximum.accumulate(eq_arr)
        dd = (peak - eq_arr) / (peak + 1e-12)
        max_dd = float(np.max(dd))

        hold_bars = [t.get("hold_bars", 0) for t in trades]

        return dict(
            n_trades=len(trades),
            win_rate=len(wins) / len(trades) * 100,
            net_pnl=sum(pnls),
            gross_profit=gp,
            gross_loss=gl,
            profit_factor=pf,
            sharpe=sharpe,
            sortino=sortino,
            max_dd=max_dd * 100,
            avg_hold=np.mean(hold_bars),
            avg_pnl=np.mean(pnls),
        )

    def _print_stress_table(self, results: Dict):
        print()
        print("--- STRESS TEST COMPARISON TABLE ---")
        print()

        headers = ["Scenario", "Trades", "Win%", "Net P/L", "PF",
                    "Sharpe", "Sortino", "MaxDD%", "AvgP/L"]
        fmt =      "{:>15} {:>7} {:>6} {:>10} {:>7} {:>8} {:>8} {:>7} {:>10}"

        print(fmt.format(*headers))
        print("  " + "-" * 82)

        order = ["base", "2x_spread", "3x_spread", "2x_slip_mean",
                 "2x_slip_std", "combined_2x2x", "5pct_spike"]

        for name in order:
            if name not in results:
                continue
            s = results[name]
            pf_s = "{:.3f}".format(s["profit_factor"]) if s["profit_factor"] < 1000 else "inf"
            print(fmt.format(
                name,
                s["n_trades"],
                "{:.1f}".format(s["win_rate"]),
                "{:+.4f}".format(s["net_pnl"]),
                pf_s,
                "{:.3f}".format(s["sharpe"]),
                "{:.3f}".format(s["sortino"]),
                "{:.1f}".format(s["max_dd"]),
                "{:+.4f}".format(s["avg_pnl"]),
            ))

    # =========================================================================
    # Component 3: Regime segmentation (ATR quartiles)
    # =========================================================================

    def _run_regime_segmentation(self):
        if not self.base_trades:
            print("  No trades to segment.")
            return

        # Collect ATR values across all validation weeks for quartile boundaries
        all_atrs = []
        for week_idx in self._val_weeks:
            features_df = self._get_features_cached(week_idx)
            if features_df is not None and "atr_14_raw" in features_df.columns:
                atrs = features_df["atr_14_raw"].dropna().values
                all_atrs.extend(atrs.tolist())

        if not all_atrs:
            print("  No ATR data available.")
            return

        q25, q50, q75 = np.percentile(all_atrs, [25, 50, 75])
        print()
        print("  ATR Quartile Boundaries:")
        print("    Q1 (low vol):  ATR < {:.2f}".format(q25))
        print("    Q2:            {:.2f} <= ATR < {:.2f}".format(q25, q50))
        print("    Q3:            {:.2f} <= ATR < {:.2f}".format(q50, q75))
        print("    Q4 (high vol): ATR >= {:.2f}".format(q75))
        print()

        # Bucket trades
        buckets = {"Q1 (low vol)": [], "Q2": [], "Q3": [], "Q4 (high vol)": []}
        unbucketed = 0

        for trade in self.base_trades:
            atr = trade.get("atr_at_entry")
            if atr is None:
                unbucketed += 1
                continue
            if atr < q25:
                buckets["Q1 (low vol)"].append(trade)
            elif atr < q50:
                buckets["Q2"].append(trade)
            elif atr < q75:
                buckets["Q3"].append(trade)
            else:
                buckets["Q4 (high vol)"].append(trade)

        if unbucketed:
            print("  ({} trades without ATR data)".format(unbucketed))

        # Print table
        headers = ["Quartile", "Trades", "Win%", "Net P/L", "PF", "AvgP/L", "AvgHold"]
        fmt = "{:>15} {:>7} {:>6} {:>10} {:>7} {:>10} {:>8}"
        print(fmt.format(*headers))
        print("  " + "-" * 67)

        for q_name, q_trades in buckets.items():
            if not q_trades:
                print("{:>15}  (no trades)".format(q_name))
                continue

            pnls = [t["pnl"] for t in q_trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            gp = sum(wins) if wins else 0.0
            gl = sum(losses) if losses else 0.0
            pf = abs(gp / gl) if gl != 0 else float("inf")
            pf_s = "{:.3f}".format(pf) if pf < 1000 else "inf"
            holds = [t.get("hold_bars", 0) for t in q_trades]

            print(fmt.format(
                q_name,
                len(q_trades),
                "{:.1f}".format(len(wins) / len(q_trades) * 100),
                "{:+.4f}".format(sum(pnls)),
                pf_s,
                "{:+.4f}".format(np.mean(pnls)),
                "{:.1f}".format(np.mean(holds)),
            ))

    # =========================================================================
    # Component 4: Churn diagnostic
    # =========================================================================

    def _run_churn_diagnostic(self):
        if not self.base_trades:
            print("  No trades to analyze.")
            return

        trades = self.base_trades
        pnls = [t["pnl"] for t in trades]
        hold_bars = [t.get("hold_bars", 0) for t in trades]
        commissions = [t.get("commission", 0.0) for t in trades]
        lots = [t.get("lots", 0.01) for t in trades]

        # Estimated trading days in validation
        trading_days = len(self._val_weeks) * 5
        trades_per_day = len(trades) / max(trading_days, 1)

        # Estimated round-trip cost per trade (spread + slippage in account currency)
        # Average spread across sessions (weighted equally)
        avg_spread_pips = np.mean([
            self.cfg.spread_london_pips, self.cfg.spread_ny_pips,
            self.cfg.spread_asia_pips, self.cfg.spread_off_hours_pips,
        ])
        avg_slippage_pips = self.cfg.slippage_mean_pips * 2  # entry + exit
        avg_cost_pips = avg_spread_pips + avg_slippage_pips
        avg_cost_points = avg_cost_pips * self.cfg.pip_price
        avg_lot = np.mean(lots)
        vpp = self.cfg.value_per_point_per_lot
        est_cost_per_trade = avg_cost_points * avg_lot * vpp
        total_est_cost = est_cost_per_trade * len(trades)

        # Gross P/L estimate (net + estimated costs)
        net_pnl = sum(pnls)
        est_gross_pnl = net_pnl + total_est_cost

        print()
        rows = [
            ("Total Trades", "{:d}".format(len(trades))),
            ("Avg Hold Bars", "{:.1f}".format(np.mean(hold_bars))),
            ("Median Hold Bars", "{:.1f}".format(np.median(hold_bars))),
            ("Min / Max Hold Bars", "{} / {}".format(min(hold_bars), max(hold_bars))),
            ("Trades per Day (est.)", "{:.2f}".format(trades_per_day)),
            ("Avg Lot Size", "{:.4f}".format(avg_lot)),
            ("", ""),
            ("--- Cost Analysis ---", ""),
            ("Avg Spread (pips)", "{:.1f}".format(avg_spread_pips)),
            ("Avg Slippage RT (pips)", "{:.1f}".format(avg_slippage_pips)),
            ("Est. Cost per Trade", "{:.4f}".format(est_cost_per_trade)),
            ("Est. Total Cost", "{:.4f}".format(total_est_cost)),
            ("Commission per Trade", "{:.4f}".format(np.mean(commissions))),
            ("", ""),
            ("--- Edge Analysis ---", ""),
            ("Net P/L", "{:+.4f}".format(net_pnl)),
            ("Est. Gross P/L", "{:+.4f}".format(est_gross_pnl)),
            ("Net P/L per Trade", "{:+.6f}".format(np.mean(pnls))),
            ("Median P/L per Trade", "{:+.6f}".format(np.median(pnls))),
            ("Net Edge per Trade", "{:+.6f}".format(np.mean(pnls))),
            ("Gross Edge per Trade", "{:+.6f}".format(est_gross_pnl / len(trades))),
        ]

        print("  {:<30} {:>15}".format("Metric", "Value"))
        print("  {} {}".format("-" * 30, "-" * 15))
        for name, val in rows:
            if not name and not val:
                print()
            elif name.startswith("---"):
                print("  {}".format(name))
            else:
                print("  {:<30} {:>15}".format(name, val))

        # By close reason
        print()
        print("  --- By Close Reason ---")
        reasons = {}
        for t in trades:
            r = t.get("reason", "UNKNOWN")
            reasons.setdefault(r, []).append(t["pnl"])

        fmt = "  {:<20} {:>7} {:>7} {:>10} {:>12}"
        print(fmt.format("Reason", "Count", "Win%", "Avg P/L", "Total P/L"))
        print("  " + "-" * 60)
        for reason in sorted(reasons.keys()):
            rpnls = reasons[reason]
            rwins = [p for p in rpnls if p > 0]
            wr = len(rwins) / len(rpnls) * 100 if rpnls else 0
            print(fmt.format(
                reason, len(rpnls),
                "{:.1f}%".format(wr),
                "{:+.4f}".format(np.mean(rpnls)),
                "{:+.4f}".format(sum(rpnls)),
            ))

        # By side
        print()
        print("  --- By Side ---")
        sides = {}
        for t in trades:
            s = t.get("side", "UNKNOWN")
            sides.setdefault(s, []).append(t["pnl"])

        print(fmt.format("Side", "Count", "Win%", "Avg P/L", "Total P/L"))
        print("  " + "-" * 60)
        for side in sorted(sides.keys()):
            spnls = sides[side]
            swins = [p for p in spnls if p > 0]
            wr = len(swins) / len(spnls) * 100 if spnls else 0
            print(fmt.format(
                side, len(spnls),
                "{:.1f}%".format(wr),
                "{:+.4f}".format(np.mean(spnls)),
                "{:+.4f}".format(sum(spnls)),
            ))

    # =========================================================================
    # Component 5: R5 ablation (reward contribution analysis)
    # =========================================================================

    def _run_r5_ablation(self):
        print()
        print("  NOTE: In eval-only mode, policy is fixed. R5 weight change")
        print("  does NOT affect agent behavior. This measures R5's")
        print("  contribution to the reward signal only.")
        print()

        if not self.base_reward_components:
            print("  No reward data (run base eval first).")
            return

        comps = self.base_reward_components
        n = len(comps)

        r1_sum = sum(c["r1"] for c in comps)
        r2_sum = sum(c["r2"] for c in comps)
        r3_sum = sum(c["r3"] for c in comps)
        r4_sum = sum(c["r4"] for c in comps)
        r5_sum = sum(c["r5"] for c in comps)

        w1 = self.cfg.r1_weight * r1_sum
        w2 = self.cfg.r2_weight * r2_sum
        w3 = self.cfg.r3_weight * r3_sum
        w4 = self.cfg.r4_weight * r4_sum
        w5 = self.cfg.r5_weight * r5_sum
        total_w = w1 + w2 + w3 + w4 + w5

        # Decomposition table
        fmt = "  {:<20} {:>12} {:>8} {:>12} {:>10}"
        print(fmt.format("Component", "Raw Sum", "Weight", "Weighted", "% Total"))
        print("  " + "-" * 66)

        for name, raw, weight, weighted in [
            ("R1 (P/L)", r1_sum, self.cfg.r1_weight, w1),
            ("R2 (Quality)", r2_sum, self.cfg.r2_weight, w2),
            ("R3 (Drawdown)", r3_sum, self.cfg.r3_weight, w3),
            ("R4 (Sharpe)", r4_sum, self.cfg.r4_weight, w4),
            ("R5 (Risk Bonus)", r5_sum, self.cfg.r5_weight, w5),
        ]:
            pct = (weighted / total_w * 100) if abs(total_w) > 1e-10 else 0
            print(fmt.format(
                name,
                "{:.4f}".format(raw),
                "{:.2f}".format(weight),
                "{:.4f}".format(weighted),
                "{:.1f}%".format(pct),
            ))

        print("  " + "-" * 66)
        print(fmt.format("TOTAL", "", "", "{:.4f}".format(total_w), "100.0%"))
        print("  Steps analyzed: {:d}".format(n))
        print()

        # Ablation comparison
        total_without_r5 = w1 + w2 + w3 + w4
        print("  Total reward WITH R5:    {:.4f}".format(total_w))
        print("  Total reward WITHOUT R5: {:.4f}".format(total_without_r5))
        print("  R5 contribution:         {:.4f}".format(w5))
        if abs(total_w) > 1e-10:
            print("  R5 as % of total:        {:.1f}%".format(w5 / total_w * 100))
        print()

        # R5 per-step distribution
        r5_vals = [c["r5"] for c in comps]
        r5_pos = [v for v in r5_vals if v > 0]
        r5_neg = [v for v in r5_vals if v < 0]
        r5_zero = [v for v in r5_vals if v == 0]

        print("  R5 per-step distribution:")
        print("    Positive (in pos, low DD):  {:>8d} ({:.1f}%)".format(
            len(r5_pos), len(r5_pos) / n * 100))
        print("    Negative (flat penalty):    {:>8d} ({:.1f}%)".format(
            len(r5_neg), len(r5_neg) / n * 100))
        print("    Zero:                       {:>8d} ({:.1f}%)".format(
            len(r5_zero), len(r5_zero) / n * 100))
        if r5_pos:
            print("    Avg positive R5:            {:.4f}".format(np.mean(r5_pos)))
        if r5_neg:
            print("    Avg negative R5:            {:.4f}".format(np.mean(r5_neg)))


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spartus Validation Evaluation Harness")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model ZIP (default: best model)")
    parser.add_argument("--quick", action="store_true",
                        help="Skip stress tests (faster)")
    args = parser.parse_args()

    evaluator = ValidationEvaluator(model_path=args.model)
    evaluator.run_all(quick=args.quick)
