"""Validation script — run a trained model on held-out data.

Usage:
    python scripts/validate.py --model storage/models/spartus_week_0050.zip
    python scripts/validate.py --model storage/models/spartus_latest.zip --weeks 5
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC

from src.config import TrainingConfig
from src.data.feature_builder import FeatureBuilder
from src.data.normalizer import ExpandingWindowNormalizer
from src.data.storage_manager import StorageManager
from src.environment.trade_env import SpartusTradeEnv
from src.memory.trading_memory import TradingMemory


def main():
    parser = argparse.ArgumentParser(description="Validate a trained Spartus model")
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip")
    parser.add_argument("--weeks", type=int, default=5, help="Number of validation weeks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = TrainingConfig()
    fb = FeatureBuilder(cfg)
    norm = ExpandingWindowNormalizer(cfg)
    storage = StorageManager(str(cfg.data_dir))
    memory = TradingMemory(config=cfg)

    # Discover weeks
    weeks_data = []
    summary = storage.get_summary()
    for year_str, tfs in sorted(summary.items()):
        year = int(year_str)
        for wk in range(1, 54):
            m5_path = cfg.data_dir / str(year) / f"week_{wk:02d}_M5.parquet"
            h1_path = cfg.data_dir / str(year) / f"week_{wk:02d}_H1.parquet"
            if m5_path.exists() and h1_path.exists():
                weeks_data.append({"year": year, "week": wk})

    # Use last N weeks as test set
    test_weeks = weeks_data[-args.weeks:]
    print(f"Validating on {len(test_weeks)} weeks")
    print(f"Model: {args.model}\n")

    # Load model
    model = SAC.load(args.model)

    all_returns = []
    all_trades = 0
    all_wins = 0
    starting_balance = cfg.initial_balance

    for i, wd in enumerate(test_weeks):
        # Load data with context
        m5_frames, h1_frames, h4_frames, d1_frames = [], [], [], []
        for offset in range(-2, 1):
            twk = wd["week"] + offset
            ty = wd["year"]
            if twk < 1:
                ty -= 1
                twk += 52
            elif twk > 52:
                ty += 1
                twk -= 52
            for tf, fl in [("M5", m5_frames), ("H1", h1_frames),
                           ("H4", h4_frames), ("D1", d1_frames)]:
                p = cfg.data_dir / str(ty) / f"week_{twk:02d}_{tf}.parquet"
                if p.exists():
                    fl.append(pd.read_parquet(p))

        m5 = pd.concat(m5_frames, ignore_index=True).sort_values("time").reset_index(drop=True)
        h1 = pd.concat(h1_frames, ignore_index=True).sort_values("time").reset_index(drop=True)
        h4 = pd.concat(h4_frames, ignore_index=True).sort_values("time").reset_index(drop=True) if h4_frames else pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        d1 = pd.concat(d1_frames, ignore_index=True).sort_values("time").reset_index(drop=True) if d1_frames else pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        features = norm.normalize(fb.build_features(m5, h1, h4, d1))

        env = SpartusTradeEnv(
            features_df=features, config=cfg, memory=memory,
            initial_balance=starting_balance, seed=args.seed,
        )

        obs, _ = env.reset()
        done = False
        truncated = False
        ep_rewards = []

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_rewards.append(reward)

        week_return = (env.balance - starting_balance) / starting_balance * 100
        all_returns.append(week_return)
        all_trades += env.episode_trades
        trades = memory.get_recent_trades(n=env.episode_trades)
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        all_wins += wins
        starting_balance = env.balance

        print(f"  Week {i+1} ({wd['year']}-W{wd['week']:02d}): "
              f"return={week_return:+.2f}%, trades={env.episode_trades}, "
              f"WR={wins}/{env.episode_trades}")

    # Summary
    print(f"\n{'='*50}")
    print(f"VALIDATION RESULTS ({len(test_weeks)} weeks)")
    print(f"{'='*50}")
    print(f"Final balance: £{starting_balance:.2f}")
    print(f"Total return: {sum(all_returns):+.2f}%")
    print(f"Mean weekly return: {np.mean(all_returns):+.2f}%")
    print(f"Total trades: {all_trades}")
    print(f"Win rate: {all_wins}/{all_trades} ({all_wins/max(all_trades,1):.1%})")
    if len(all_returns) > 1 and np.std(all_returns) > 0:
        print(f"Sharpe (weekly): {np.mean(all_returns)/np.std(all_returns):.3f}")
    print(f"Max weekly DD: {min(all_returns):.2f}%")

    memory.close()


if __name__ == "__main__":
    main()
