"""Validation PF/Sharpe comparison across training checkpoints.

Runs base validation eval (no stress tests) against multiple saved
checkpoints to answer: Is validation PF climbing alongside training PF,
or is training outpacing validation (overfitting)?

Usage:
    python scripts/eval_checkpoints.py [--checkpoints 100,110,120,...] [--include-best]
"""

import sys
import time
import json
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


class CheckpointEvaluator:
    """Evaluate multiple checkpoints on the same validation set."""

    def __init__(self):
        self.cfg = TrainingConfig()
        self.feature_builder = FeatureBuilder(self.cfg)
        self.normalizer = ExpandingWindowNormalizer(self.cfg)

        # Deterministic eval config
        self._eval_cfg = self._make_eval_config()

        # Data (shared across all checkpoint evals)
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

        print("Split: {} train, {} val, {} test".format(
            len(self._train_weeks), len(self._val_weeks), len(self._test_weeks)))

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

    def _rollout_week(self, week_idx: int, model, seed: int = 42) -> Optional[Dict]:
        week_data = self._weeks_data[week_idx]
        cfg = self._eval_cfg

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

        time_in_market = sum(step_positions) / max(len(step_positions), 1)

        return {
            "trades": trades,
            "equity_curve": equity_curve,
            "time_in_market": time_in_market,
            "total_steps": len(step_positions),
            "final_balance": equity_curve[-1] if equity_curve else cfg.initial_balance,
        }

    def _compute_metrics(self, all_trades: List[Dict], weekly_returns: List[float],
                         weekly_tim: List[float], total_steps: int) -> Dict:
        """Compute standard validation metrics from rollout results."""
        if not all_trades:
            return {
                "trades": 0, "win_pct": 0, "net_pnl": 0, "pf": 0,
                "sharpe": 0, "sortino": 0, "max_dd_pct": 0,
                "avg_pnl": 0, "time_in_market": 0,
            }

        pnls = [t.get("pnl", 0) for t in all_trades]
        wins = sum(1 for p in pnls if p > 0)
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))

        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        net_pnl = sum(pnls)

        # Weekly return stats
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

        # Max drawdown from cumulative P/L
        cum_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = running_max - cum_pnl
        max_dd = drawdowns.max() if len(drawdowns) > 0 else 0
        start_bal = self.cfg.initial_balance
        max_dd_pct = max_dd / start_bal * 100

        avg_tim = np.mean(weekly_tim) if weekly_tim else 0

        # Direction breakdown
        longs = [t for t in all_trades if t.get("direction", "") == "LONG"]
        shorts = [t for t in all_trades if t.get("direction", "") == "SHORT"]
        long_pnl = sum(t.get("pnl", 0) for t in longs)
        short_pnl = sum(t.get("pnl", 0) for t in shorts)

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
            "long_pnl": long_pnl,
            "short_pnl": short_pnl,
            "long_count": len(longs),
            "short_count": len(shorts),
        }

    def eval_checkpoint(self, model_path: Path, label: str) -> Dict:
        """Evaluate a single checkpoint on the full validation set."""
        print("\n  Loading: {} ...".format(model_path.name))
        try:
            model = SAC.load(str(model_path))
        except Exception as e:
            print("  ERROR loading {}: {}".format(model_path.name, e))
            return {"label": label, "error": str(e)}

        all_trades = []
        weekly_returns = []
        weekly_tim = []
        total_steps = 0

        for i, wi in enumerate(self._val_weeks):
            wd = self._weeks_data[wi]
            result = self._rollout_week(wi, model)
            if result is None:
                continue

            all_trades.extend(result["trades"])
            weekly_returns.append(
                (result["final_balance"] - self.cfg.initial_balance) / self.cfg.initial_balance
            )
            weekly_tim.append(result["time_in_market"])
            total_steps += result["total_steps"]

            if (i + 1) % 20 == 0:
                print("    {}/{} weeks done...".format(i + 1, len(self._val_weeks)))

        metrics = self._compute_metrics(all_trades, weekly_returns, weekly_tim, total_steps)
        metrics["label"] = label
        metrics["model_file"] = model_path.name
        return metrics

    def run(self, checkpoint_weeks: List[int], include_best: bool = True):
        """Run evaluation across all specified checkpoints."""
        print("=" * 70)
        print("CHECKPOINT VALIDATION COMPARISON")
        print("=" * 70)

        # Setup data
        self._discover_weeks()
        self._split_weeks()

        val_info = [self._weeks_data[i] for i in self._val_weeks]
        print("Validation: {} weeks ({}-W{:02d} to {}-W{:02d})".format(
            len(self._val_weeks),
            val_info[0]["year"], val_info[0]["week"],
            val_info[-1]["year"], val_info[-1]["week"],
        ))
        print()

        # Resolve checkpoint paths
        model_dir = self.cfg.model_dir
        evaluations = []

        for wk in checkpoint_weeks:
            path = model_dir / "spartus_week_{:04d}.zip".format(wk)
            if path.exists():
                evaluations.append((path, "W{:04d}".format(wk)))
            else:
                print("  SKIP: {} not found".format(path.name))

        if include_best:
            best_path = self.cfg.best_model_path
            if best_path.exists():
                evaluations.append((best_path, "BEST"))
            else:
                print("  SKIP: best model not found")

        print("Evaluating {} checkpoints on {} val weeks each".format(
            len(evaluations), len(self._val_weeks)))
        print()

        # Run evaluations
        results = []
        for i, (path, label) in enumerate(evaluations):
            t0 = time.time()
            print("[{}/{}] Evaluating {}".format(i + 1, len(evaluations), label))
            metrics = self.eval_checkpoint(path, label)
            elapsed = time.time() - t0
            print("  Done in {:.1f}s".format(elapsed))
            results.append(metrics)

        # Print results table
        print()
        print("=" * 70)
        print("VALIDATION PF COMPARISON TABLE")
        print("=" * 70)
        print()

        header = "{:<8} {:>6} {:>6} {:>10} {:>6} {:>7} {:>8} {:>7} {:>7} {:>7}".format(
            "Chkpt", "Trades", "Win%", "Net P/L", "PF", "Sharpe", "Sortino", "MaxDD%", "TIM%", "AvgP/L"
        )
        print(header)
        print("-" * len(header))

        for r in results:
            if "error" in r:
                print("{:<8} ERROR: {}".format(r["label"], r["error"]))
                continue
            print("{:<8} {:>6} {:>5.1f}% {:>+10.2f} {:>6.3f} {:>7.3f} {:>8.3f} {:>6.1f}% {:>6.1f}% {:>+7.4f}".format(
                r["label"],
                r["trades"],
                r["win_pct"],
                r["net_pnl"],
                r["pf"],
                r["sharpe"],
                r["sortino"],
                r["max_dd_pct"],
                r["time_in_market"],
                r["avg_pnl"],
            ))

        # Direction breakdown
        print()
        print("DIRECTION BREAKDOWN")
        print()
        dir_header = "{:<8} {:>6} {:>10} {:>6} {:>10}".format(
            "Chkpt", "Longs", "Long P/L", "Shorts", "Short P/L"
        )
        print(dir_header)
        print("-" * len(dir_header))
        for r in results:
            if "error" in r:
                continue
            print("{:<8} {:>6} {:>+10.2f} {:>6} {:>+10.2f}".format(
                r["label"],
                r.get("long_count", 0),
                r.get("long_pnl", 0),
                r.get("short_count", 0),
                r.get("short_pnl", 0),
            ))

        # Save results to JSON for later analysis
        output_path = Path("storage/logs/checkpoint_comparison.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print()
        print("Results saved to: {}".format(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checkpoint validation comparison")
    parser.add_argument("--checkpoints", type=str, default="100,110,120,130,140,150,160",
                        help="Comma-separated week numbers to evaluate")
    parser.add_argument("--include-best", action="store_true", default=True,
                        help="Include best model (default: True)")
    parser.add_argument("--no-best", action="store_true", default=False,
                        help="Exclude best model")
    args = parser.parse_args()

    weeks = [int(w.strip()) for w in args.checkpoints.split(",")]
    include_best = not args.no_best

    evaluator = CheckpointEvaluator()
    evaluator.run(weeks, include_best=include_best)
