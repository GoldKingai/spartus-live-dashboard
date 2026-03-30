"""Phase 2 Upgrade verification tests.

Tests the 6 locked-in changes from Training_Upgrade_Plan.md:
1. Profit Protection (staged R-based SL)
2. Re-Entry Penalty
3. min_hold_bars = 6
4. R2 hold quality curve (denominator 20)
5. Conviction threshold = 0.30
6. Protection logging (tested via trade_executor in live system)

Usage:
    python tests/test_phase2_upgrade.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TrainingConfig
from src.risk.risk_manager import RiskManager
from src.environment.reward import RewardCalculator


def test_protection_stages_long():
    """Verify protection stages trigger correctly for LONG trades."""
    cfg = TrainingConfig()
    rm = RiskManager(cfg)

    entry = 2000.0
    initial_sl = 1990.0  # SL 10 points below entry
    r_distance = 10.0    # R = 10 points

    # Stage 0: MFE < 1R (e.g., +5 points)
    position = {
        "side": "LONG",
        "entry_price": entry,
        "stop_loss": initial_sl,
        "initial_sl": initial_sl,
        "max_favorable": 5.0,  # 0.5R
        "protection_stage": 0,
    }
    new_sl, stage = rm.apply_profit_protection(position, 2005.0, atr=5.0)
    assert stage == 0, f"Expected stage 0 at 0.5R, got {stage}"
    assert new_sl == initial_sl, f"SL should not move at stage 0, got {new_sl}"

    # Stage 1: MFE >= 1R (+10 points)
    position["max_favorable"] = 10.0  # 1.0R
    new_sl, stage = rm.apply_profit_protection(position, 2010.0, atr=5.0)
    assert stage == 1, f"Expected stage 1 at 1.0R, got {stage}"
    assert new_sl >= entry, f"BE SL should be >= entry ({entry}), got {new_sl}"

    # Stage 2: MFE >= 1.5R (+15 points)
    position["max_favorable"] = 15.0  # 1.5R
    position["protection_stage"] = 1
    new_sl, stage = rm.apply_profit_protection(position, 2015.0, atr=5.0)
    assert stage == 2, f"Expected stage 2 at 1.5R, got {stage}"
    expected_lock = entry + 0.5 * r_distance  # 2005.0
    assert new_sl >= expected_lock, f"Lock SL should be >= {expected_lock}, got {new_sl}"

    # Stage 3: MFE >= 2R (+20 points)
    position["max_favorable"] = 20.0  # 2.0R
    position["protection_stage"] = 2
    position["stop_loss"] = expected_lock  # From stage 2
    new_sl, stage = rm.apply_profit_protection(position, 2020.0, atr=5.0)
    assert stage == 3, f"Expected stage 3 at 2.0R, got {stage}"
    # ATR trail: sl = price - 1.0*ATR = 2020 - 5 = 2015
    assert new_sl >= 2014.0, f"Trail SL should be near 2015, got {new_sl}"

    print("  PASS: Protection stages trigger correctly (LONG)")


def test_protection_stages_short():
    """Verify protection stages for SHORT trades (mirrored logic)."""
    cfg = TrainingConfig()
    rm = RiskManager(cfg)

    entry = 2000.0
    initial_sl = 2010.0  # SL 10 points above entry
    r_distance = 10.0

    # Stage 1: MFE >= 1R
    position = {
        "side": "SHORT",
        "entry_price": entry,
        "stop_loss": initial_sl,
        "initial_sl": initial_sl,
        "max_favorable": 10.0,
        "protection_stage": 0,
    }
    new_sl, stage = rm.apply_profit_protection(position, 1990.0, atr=5.0)
    assert stage == 1, f"Expected stage 1, got {stage}"
    assert new_sl <= entry, f"SHORT BE SL should be <= entry ({entry}), got {new_sl}"

    # Stage 2: MFE >= 1.5R
    position["max_favorable"] = 15.0
    position["protection_stage"] = 1
    new_sl, stage = rm.apply_profit_protection(position, 1985.0, atr=5.0)
    assert stage == 2, f"Expected stage 2, got {stage}"
    expected_lock = entry - 0.5 * r_distance  # 1995.0
    assert new_sl <= expected_lock, f"Lock SL should be <= {expected_lock}, got {new_sl}"

    print("  PASS: Protection stages trigger correctly (SHORT)")


def test_protection_skip_stages():
    """Verify price gap through stages works (skips to highest eligible)."""
    cfg = TrainingConfig()
    rm = RiskManager(cfg)

    position = {
        "side": "LONG",
        "entry_price": 2000.0,
        "stop_loss": 1990.0,
        "initial_sl": 1990.0,
        "max_favorable": 25.0,  # 2.5R — past all triggers
        "protection_stage": 0,
    }
    new_sl, stage = rm.apply_profit_protection(position, 2025.0, atr=5.0)
    assert stage == 3, f"Expected stage 3 (skipped 1,2), got {stage}"

    print("  PASS: Stage skipping works (gap through +1R/+1.5R to +2.5R)")


def test_protection_never_regresses():
    """Verify stage never goes backward."""
    cfg = TrainingConfig()
    rm = RiskManager(cfg)

    position = {
        "side": "LONG",
        "entry_price": 2000.0,
        "stop_loss": 2005.0,  # Already above entry (from stage 2)
        "initial_sl": 1990.0,
        "max_favorable": 15.0,  # Was at 1.5R
        "protection_stage": 2,
    }
    # MFE drops to 0.5R (price retraced but stage should stay at 2)
    position["max_favorable"] = 5.0
    new_sl, stage = rm.apply_profit_protection(position, 2005.0, atr=5.0)
    assert stage == 2, f"Stage should not regress from 2, got {stage}"

    print("  PASS: Protection stage never regresses")


def test_protection_floor_overrides_ai():
    """Verify protection SL floor overrides AI sl_adjustment."""
    cfg = TrainingConfig()
    rm = RiskManager(cfg)

    entry = 2000.0
    initial_sl = 1990.0

    # Set up: stage 2 locked at +0.5R = 1995
    position = {
        "side": "LONG",
        "entry_price": entry,
        "stop_loss": 1990.0,
        "initial_sl": initial_sl,
        "max_favorable": 15.0,
        "protection_stage": 0,
    }
    protection_sl, stage = rm.apply_profit_protection(position, 2015.0, atr=5.0)
    assert stage == 2
    assert protection_sl >= 1995.0, f"Protection should lock at 1995+, got {protection_sl}"

    # Now AI tries to trail LOOSER (sl_adj=0 = keep current / don't trail)
    # adjust_stop_loss should not move SL below protection floor
    ai_sl = rm.adjust_stop_loss(
        current_sl=protection_sl,
        side="LONG",
        current_price=2015.0,
        atr=5.0,
        sl_adj=0.0,  # Don't trail
    )
    # sl_adj=0 → trail_distance = 0.5*5 + 1.0*2.0*5 = 12.5 → proposed = 2002.5
    # But since adjust_stop_loss only tightens, and current_sl is already ~1995,
    # the new sl should be max(2002.5, 1995) = 2002.5
    assert ai_sl >= protection_sl, (
        f"AI SL ({ai_sl}) should be >= protection floor ({protection_sl})"
    )

    print("  PASS: Protection floor overrides AI sl_adjustment")


def test_reentry_penalty_fires():
    """Verify re-entry penalty multiplier is applied on R1."""
    cfg = TrainingConfig()
    reward_calc = RewardCalculator(cfg)
    reward_calc.reset(100.0)

    # Simulate a losing re-entry trade result
    trade_result = {
        "pnl": -5.0,
        "risk_amount": 10.0,
        "hold_bars": 8,
        "reason": "SL_HIT",
        "is_reentry": True,
    }

    # R1 for a -1% equity return
    r1_reentry = reward_calc._calc_r1(
        equity_return=-0.01,
        position=None,
        trade_result=trade_result,
    )

    # R1 for same return WITHOUT re-entry
    trade_result_normal = dict(trade_result)
    trade_result_normal["is_reentry"] = False
    r1_normal = reward_calc._calc_r1(
        equity_return=-0.01,
        position=None,
        trade_result=trade_result_normal,
    )

    # Re-entry should be 1.5x more negative
    assert r1_reentry < r1_normal, (
        f"Re-entry R1 ({r1_reentry}) should be more negative than normal ({r1_normal})"
    )
    expected_ratio = cfg.reentry_penalty_mult
    actual_ratio = r1_reentry / r1_normal
    assert abs(actual_ratio - expected_ratio) < 0.01, (
        f"Re-entry penalty ratio should be {expected_ratio}, got {actual_ratio}"
    )

    # FIX-11: Winning re-entries now get DISCOUNTED (0.7x), not free
    r1_win = reward_calc._calc_r1(
        equity_return=0.01,
        position=None,
        trade_result=trade_result,  # is_reentry=True but winning
    )
    r1_win_normal = reward_calc._calc_r1(
        equity_return=0.01,
        position=None,
        trade_result=trade_result_normal,
    )
    expected_discount = cfg.reentry_win_discount  # 0.7
    assert abs(r1_win / r1_win_normal - expected_discount) < 0.01, (
        f"Winning re-entry R1 ({r1_win}) should be {expected_discount}x of normal ({r1_win_normal})"
    )

    print("  PASS: Re-entry penalty fires correctly (1.5x on losses, 0.7x on wins)")


def test_r2_hold_quality_curve():
    """Verify R2 hold quality uses denominator 20."""
    cfg = TrainingConfig()
    reward_calc = RewardCalculator(cfg)

    # Trade with 6 bars held (min_hold) and 1:1 R:R
    result_6bars = {"pnl": 10.0, "risk_amount": 10.0, "hold_bars": 6}
    r2_6 = reward_calc._calc_r2(result_6bars)
    expected_6 = (10.0 / 10.0) * min(6 / 20.0, 1.0)  # 1.0 * 0.3 = 0.3
    assert abs(r2_6 - expected_6) < 0.01, f"R2 at 6 bars: {r2_6} != expected {expected_6}"

    # Trade with 20 bars (full credit)
    result_20bars = {"pnl": 10.0, "risk_amount": 10.0, "hold_bars": 20}
    r2_20 = reward_calc._calc_r2(result_20bars)
    expected_20 = (10.0 / 10.0) * min(20 / 20.0, 1.0)  # 1.0 * 1.0 = 1.0
    assert abs(r2_20 - expected_20) < 0.01, f"R2 at 20 bars: {r2_20} != expected {expected_20}"

    # Trade with 10 bars (was 100% credit with old /10, now 50%)
    result_10bars = {"pnl": 10.0, "risk_amount": 10.0, "hold_bars": 10}
    r2_10 = reward_calc._calc_r2(result_10bars)
    expected_10 = (10.0 / 10.0) * min(10 / 20.0, 1.0)  # 1.0 * 0.5 = 0.5
    assert abs(r2_10 - expected_10) < 0.01, f"R2 at 10 bars: {r2_10} != expected {expected_10}"

    print("  PASS: R2 hold quality curve uses /20 (6 bars=30%, 10 bars=50%, 20 bars=100%)")


def test_config_values():
    """Verify config defaults match the upgrade plan."""
    cfg = TrainingConfig()

    assert cfg.min_hold_bars == 6, f"min_hold_bars should be 6, got {cfg.min_hold_bars}"
    assert cfg.protection_be_trigger_r == 1.0
    assert cfg.protection_lock_trigger_r == 1.5
    assert cfg.protection_lock_amount_r == 0.5
    assert cfg.protection_trail_trigger_r == 2.0
    assert cfg.protection_trail_atr_mult == 1.0
    assert cfg.reentry_penalty_bars == 6
    assert cfg.reentry_penalty_mult == 1.5
    assert cfg.normal_conviction_threshold == 0.3

    print("  PASS: Config values match Training_Upgrade_Plan.md")


def test_live_config_values():
    """Verify live config defaults match."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "live_dashboard"))
    from config.live_config import LiveConfig

    cfg = LiveConfig()
    assert cfg.min_hold_bars == 6, f"Live min_hold_bars should be 6, got {cfg.min_hold_bars}"
    assert cfg.min_conviction == 0.15, f"Live min_conviction should be 0.15 (lowered per Live Conviction Fix 2026-03-10), got {cfg.min_conviction}"
    assert cfg.protection_be_trigger_r == 1.0
    assert cfg.protection_lock_trigger_r == 1.5
    assert cfg.protection_trail_trigger_r == 2.0

    print("  PASS: Live config values match Training_Upgrade_Plan.md")


def test_live_risk_manager_protection():
    """Verify live risk manager has apply_profit_protection."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "live_dashboard"))
    from config.live_config import LiveConfig
    from core.risk_manager import LiveRiskManager

    cfg = LiveConfig()
    rm = LiveRiskManager(cfg)

    assert hasattr(rm, "apply_profit_protection"), "LiveRiskManager missing apply_profit_protection"

    # Quick smoke test
    position = {
        "side": "LONG",
        "entry_price": 2000.0,
        "stop_loss": 1990.0,
        "initial_sl": 1990.0,
        "max_favorable": 12.0,
        "protection_stage": 0,
    }
    new_sl, stage = rm.apply_profit_protection(position, 2012.0, atr=5.0)
    assert stage == 1, f"Expected stage 1 at 1.2R, got {stage}"
    assert new_sl >= 2000.0, f"BE SL should be >= entry, got {new_sl}"

    print("  PASS: Live risk manager protection works")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2 Upgrade Verification Tests")
    print("=" * 60)

    print("\n1. Protection stages (LONG)...")
    test_protection_stages_long()

    print("\n2. Protection stages (SHORT)...")
    test_protection_stages_short()

    print("\n3. Stage skipping (price gap)...")
    test_protection_skip_stages()

    print("\n4. Stage never regresses...")
    test_protection_never_regresses()

    print("\n5. Protection floor overrides AI...")
    test_protection_floor_overrides_ai()

    print("\n6. Re-entry penalty...")
    test_reentry_penalty_fires()

    print("\n7. R2 hold quality curve...")
    test_r2_hold_quality_curve()

    print("\n8. Config values...")
    test_config_values()

    print("\n9. Live config values...")
    test_live_config_values()

    print("\n10. Live risk manager protection...")
    test_live_risk_manager_protection()

    print("\n" + "=" * 60)
    print("ALL PHASE 2 UPGRADE TESTS PASSED")
    print("=" * 60)
