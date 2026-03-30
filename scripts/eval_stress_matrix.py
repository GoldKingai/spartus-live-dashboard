"""Stress matrix evaluation for champion checkpoint selection.

Runs multiple stress scenarios on specified checkpoints against both
validation and test sets. Designed for the final-phase audit.

Usage:
    python scripts/eval_stress_matrix.py --checkpoint 180 --mode val
    python scripts/eval_stress_matrix.py --checkpoint 180 --mode test
    python scripts/eval_stress_matrix.py --checkpoint 180 --mode both
"""

import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import SAC

from src.config import TrainingConfig
from src.data.feature_builder import FeatureBuilder
from src.data.normalizer import ExpandingWindowNormalizer
from src.environment.trade_env import SpartusTradeEnv, PRECOMPUTED_FEATURES
from src.memory.trading_memory import TradingMemory


# ── Stress scenario definitions ─────────────────────────────────────
STRESS_SCENARIOS = {
    "base": {},
    "2x_spread": {
        "spread_london_pips": 2.0,
        "spread_ny_pips": 2.0,
        "spread_asia_pips": 2.0,
        "spread_off_hours_pips": 2.0,
    },
    "3x_spread": {
        "spread_london_pips": 3.0,
        "spread_ny_pips": 3.0,
        "spread_asia_pips": 3.0,
        "spread_off_hours_pips": 3.0,
    },
    "2x_slip_mean": {
        "slippage_mean_pips": 2.0,
    },
    "2x_slip_std": {
        "slippage_std_pips": 2.0,
    },
    "combined_2x2x": {
        "spread_london_pips": 2.0,
        "spread_ny_pips": 2.0,
        "spread_asia_pips": 2.0,
        "spread_off_hours_pips": 2.0,
        "slippage_mean_pips": 2.0,
        "slippage_std_pips": 2.0,
    },
    "5x_spread": {
        "spread_london_pips": 5.0,
        "spread_ny_pips": 5.0,
        "spread_asia_pips": 5.0,
        "spread_off_hours_pips": 5.0,
    },
}


class StressMatrixEvaluator:
    """Evaluate a single checkpoint across multiple stress scenarios."""

    def __init__(self):
        self.cfg = TrainingConfig()
        self.feature_builder = FeatureBuilder(self.cfg)
        self.normalizer = ExpandingWindowNormalizer(self.cfg)

        self._base_eval_cfg = self._make_eval_config()

        self._weeks_data: List[Dict] = []
        self._train_weeks: List[int] = []
        self._val_weeks: List[int] = []
        self._test_weeks: List[int] = []
        self._features_cache: Dict[int, pd.DataFrame] = {}

    def _make_eval_config(self) -> TrainingConfig:
        cfg = deepcopy(self.cfg)
        cfg.start_offset_max = 0
        cfg.observation_noise_std = 0.0
        cfg.spread_jitter = 0.0
        cfg.slippage_jitter = 0.0
        cfg.commission_jitter = 0.0
        return cfg

    def _apply_stress(self, scenario_name: str) -> TrainingConfig:
        """Create a stressed config by multiplying base values."""
        cfg = deepcopy(self._base_eval_cfg)
        multipliers = STRESS_SCENARIOS.get(scenario_name, {})
        base = self._base_eval_cfg

        for attr, mult in multipliers.items():
            base_val = getattr(base, attr)
            setattr(cfg, attr, base_val * mult)

        return cfg

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
        train_end = int(n * 0.90)
        val_end = train_end + purge + int(n * 0.05)

        self._train_weeks = list(range(0, train_end))
        self._val_weeks = list(range(train_end + purge, min(val_end, n)))
        self._test_weeks = list(range(min(val_end + purge, n), n))

        val_info = [self._weeks_data[i] for i in self._val_weeks]
        test_info = [self._weeks_data[i] for i in self._test_weeks]
        print("Split: {} train, {} val ({}-W{:02d} to {}-W{:02d}), {} test ({}-W{:02d} to {}-W{:02d})".format(
            len(self._train_weeks),
            len(self._val_weeks),
            val_info[0]["year"], val_info[0]["week"],
            val_info[-1]["year"], val_info[-1]["week"],
            len(self._test_weeks),
            test_info[0]["year"], test_info[0]["week"],
            test_info[-1]["year"], test_info[-1]["week"],
        ))

    def _get_features_cached(self, week_idx: int) -> Optional[pd.DataFrame]:
        if week_idx in self._features_cache:
            return self._features_cache[week_idx]
        try:
            df = self._get_features(self._weeks_data[week_idx])
            self._features_cache[week_idx] = df
            return df
        except Exception as e:
            wd = self._weeks_data[week_idx]
            print("    WARN: Feature load failed {}-W{:02d}: {}".format(
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

    def _rollout_week(self, week_idx: int, model, cfg: TrainingConfig,
                      seed: int = 42) -> Optional[Dict]:
        week_data = self._weeks_data[week_idx]

        features_df = self._get_features_cached(week_idx)
        if features_df is None:
            return None

        if len(features_df) < cfg.lookback + 10:
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

        obs, info = env.reset()
        trades = []
        equity_curve = [cfg.initial_balance]
        step_positions = []
        hold_bars_list = []
        done = False
        truncated = False

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            equity_curve.append(info.get("equity", env.balance))
            step_positions.append(info.get("has_position", False))

            if "last_trade" in info:
                trade = info["last_trade"].copy()
                trade["week_year"] = week_data["year"]
                trade["week_num"] = week_data["week"]
                trades.append(trade)
                if "hold_bars" in trade:
                    hold_bars_list.append(trade["hold_bars"])

        time_in_market = sum(step_positions) / max(len(step_positions), 1)
        trades_per_day = len(trades) / 5.0  # 5 trading days per week

        return {
            "trades": trades,
            "equity_curve": equity_curve,
            "time_in_market": time_in_market,
            "total_steps": len(step_positions),
            "final_balance": equity_curve[-1] if equity_curve else cfg.initial_balance,
            "trades_per_day": trades_per_day,
            "avg_hold_bars": np.mean(hold_bars_list) if hold_bars_list else 0,
        }

    def _compute_metrics(self, all_trades: List[Dict], weekly_returns: List[float],
                         weekly_tim: List[float], total_steps: int,
                         weekly_tpd: List[float] = None,
                         weekly_hold: List[float] = None) -> Dict:
        if not all_trades:
            return {
                "trades": 0, "win_pct": 0, "net_pnl": 0, "pf": 0,
                "sharpe": 0, "sortino": 0, "max_dd_pct": 0,
                "avg_pnl": 0, "time_in_market": 0,
                "avg_trades_per_day": 0, "avg_hold_bars": 0,
            }

        pnls = [t.get("pnl", 0) for t in all_trades]
        wins = sum(1 for p in pnls if p > 0)
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))

        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        net_pnl = sum(pnls)

        if weekly_returns and len(weekly_returns) > 1:
            wr = np.array(weekly_returns)
            mean_r = wr.mean()
            std_r = wr.std()
            sharpe = (mean_r / std_r * np.sqrt(52)) if std_r > 1e-8 else 0.0

            downside = wr[wr < 0]
            down_std = downside.std() if len(downside) > 1 else 1e-8
            sortino = (mean_r / down_std * np.sqrt(52)) if down_std > 1e-8 else 0.0
        else:
            sharpe = 0.0
            sortino = 0.0

        cum_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = running_max - cum_pnl
        max_dd = drawdowns.max() if len(drawdowns) > 0 else 0
        start_bal = self.cfg.initial_balance
        max_dd_pct = max_dd / start_bal * 100

        avg_tim = np.mean(weekly_tim) if weekly_tim else 0
        avg_tpd = np.mean(weekly_tpd) if weekly_tpd else 0
        avg_hold = np.mean(weekly_hold) if weekly_hold else 0

        longs = [t for t in all_trades if t.get("direction", "") == "LONG"]
        shorts = [t for t in all_trades if t.get("direction", "") == "SHORT"]

        return {
            "trades": len(all_trades),
            "win_pct": wins / len(all_trades) * 100,
            "net_pnl": net_pnl,
            "pf": pf,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_dd_pct": max_dd_pct,
            "avg_pnl": net_pnl / len(all_trades),
            "time_in_market": avg_tim * 100,
            "avg_trades_per_day": avg_tpd,
            "avg_hold_bars": avg_hold,
            "long_count": len(longs),
            "short_count": len(shorts),
            "long_pnl": sum(t.get("pnl", 0) for t in longs),
            "short_pnl": sum(t.get("pnl", 0) for t in shorts),
        }

    def _eval_on_weeks(self, week_indices: List[int], model,
                       cfg: TrainingConfig, label: str) -> Dict:
        """Evaluate model on a specific set of weeks with given config."""
        all_trades = []
        weekly_returns = []
        weekly_tim = []
        weekly_tpd = []
        weekly_hold = []
        total_steps = 0

        for i, wi in enumerate(week_indices):
            result = self._rollout_week(wi, model, cfg)
            if result is None:
                continue

            all_trades.extend(result["trades"])
            weekly_returns.append(
                (result["final_balance"] - cfg.initial_balance) / cfg.initial_balance
            )
            weekly_tim.append(result["time_in_market"])
            weekly_tpd.append(result["trades_per_day"])
            weekly_hold.append(result["avg_hold_bars"])
            total_steps += result["total_steps"]

            if (i + 1) % 20 == 0:
                print("      {}/{} weeks...".format(i + 1, len(week_indices)))

        metrics = self._compute_metrics(
            all_trades, weekly_returns, weekly_tim, total_steps,
            weekly_tpd, weekly_hold
        )
        metrics["label"] = label
        return metrics

    def run(self, checkpoint_week: int, mode: str = "val",
            scenarios: List[str] = None):
        """Run full stress matrix for one checkpoint.

        Args:
            checkpoint_week: Week number of checkpoint to evaluate
            mode: 'val', 'test', or 'both'
            scenarios: List of scenario names (default: all 7)
        """
        if scenarios is None:
            scenarios = list(STRESS_SCENARIOS.keys())

        print("=" * 70)
        print("STRESS MATRIX — W{:04d}".format(checkpoint_week))
        print("=" * 70)

        # Setup data
        self._discover_weeks()
        self._split_weeks()

        # Resolve checkpoint path
        model_path = self.cfg.model_dir / "spartus_week_{:04d}.zip".format(checkpoint_week)
        if not model_path.exists():
            print("ERROR: {} not found".format(model_path))
            return {}

        # Load model ONCE
        print("\nLoading model: {} ...".format(model_path.name))
        t0 = time.time()
        model = SAC.load(str(model_path))
        print("  Model loaded in {:.1f}s".format(time.time() - t0))

        all_results = {}

        # Determine which week sets to evaluate
        eval_sets = []
        if mode in ("val", "both"):
            eval_sets.append(("VAL", self._val_weeks))
        if mode in ("test", "both"):
            eval_sets.append(("TEST", self._test_weeks))

        for set_name, week_indices in eval_sets:
            set_info = [self._weeks_data[i] for i in week_indices]
            print("\n--- {} SET: {} weeks ({}-W{:02d} to {}-W{:02d}) ---".format(
                set_name, len(week_indices),
                set_info[0]["year"], set_info[0]["week"],
                set_info[-1]["year"], set_info[-1]["week"],
            ))

            set_results = []
            for scenario in scenarios:
                cfg = self._apply_stress(scenario)
                label = "W{:04d}_{}_{}".format(checkpoint_week, set_name, scenario)
                print("\n  [{}] {} ...".format(set_name, scenario))
                t1 = time.time()
                metrics = self._eval_on_weeks(week_indices, model, cfg, label)
                elapsed = time.time() - t1
                metrics["scenario"] = scenario
                metrics["set"] = set_name
                metrics["checkpoint"] = checkpoint_week
                print("    Done in {:.1f}s — PF {:.3f}, Trades {}, MaxDD {:.1f}%".format(
                    elapsed, metrics["pf"], metrics["trades"], metrics["max_dd_pct"]))
                set_results.append(metrics)

            all_results[set_name] = set_results

            # Print summary table
            print("\n{}".format("=" * 70))
            print("W{:04d} — {} SET STRESS MATRIX".format(checkpoint_week, set_name))
            print("{}".format("=" * 70))
            header = "{:<15} {:>6} {:>6} {:>10} {:>6} {:>7} {:>7} {:>7} {:>5} {:>5}".format(
                "Scenario", "Trades", "Win%", "Net P/L", "PF", "Sharpe", "MaxDD%", "TIM%", "T/Day", "Hold"
            )
            print(header)
            print("-" * len(header))
            for r in set_results:
                print("{:<15} {:>6} {:>5.1f}% {:>+10.2f} {:>6.3f} {:>7.3f} {:>6.1f}% {:>6.1f}% {:>5.1f} {:>5.1f}".format(
                    r["scenario"],
                    r["trades"],
                    r["win_pct"],
                    r["net_pnl"],
                    r["pf"],
                    r["sharpe"],
                    r["max_dd_pct"],
                    r["time_in_market"],
                    r["avg_trades_per_day"],
                    r["avg_hold_bars"],
                ))

        # Save results
        output_path = Path("storage/logs/stress_matrix_W{:04d}.json".format(checkpoint_week))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        flat_results = []
        for set_results in all_results.values():
            flat_results.extend(set_results)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(flat_results, f, indent=2, default=str)
        print("\nResults saved to: {}".format(output_path))

        return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress matrix evaluation")
    parser.add_argument("--checkpoint", type=int, required=True,
                        help="Checkpoint week number to evaluate")
    parser.add_argument("--mode", choices=["val", "test", "both"], default="val",
                        help="Which set to evaluate on (default: val)")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="Comma-separated scenario names (default: all)")
    args = parser.parse_args()

    scenarios = None
    if args.scenarios:
        scenarios = [s.strip() for s in args.scenarios.split(",")]

    evaluator = StressMatrixEvaluator()
    evaluator.run(args.checkpoint, mode=args.mode, scenarios=scenarios)
