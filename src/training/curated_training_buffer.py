"""Curated 3-tier replay buffer for SAC training.

Addresses catastrophic forgetting in sequential curriculum training.
As the model trains through 2015→2026, a reservoir-sampled core tier
permanently retains diverse samples from ALL eras so the model never
fully forgets earlier market regimes.

Architecture
------------
  FIFO tier  (~80%): Standard circular buffer. Recent curriculum data.
  Core tier  (~20%): Reservoir-sampled from ALL transitions ever seen.
                     Protected from FIFO eviction. Zero extra memory —
                     uses parent's obs arrays with a boolean mask overlay.

Sample mix per batch: 35% core (cross-era), 65% FIFO (current curriculum).

Memory overhead vs standard ReplayBuffer
-----------------------------------------
  _core_mask:     buffer_size × 1 byte   (~125 KB)
  _core_year:     buffer_size × 2 bytes  (~250 KB)
  _core_added_at: buffer_size × 8 bytes  (~1 MB)
  ─────────────────────────────────────────────────
  Total:          ~1.4 MB  (vs zero for standard buffer)
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class CuratedTrainingBuffer(ReplayBuffer):
    """SAC replay buffer with reservoir-sampled core tier for era diversity.

    Drop-in replacement for SB3's ReplayBuffer. Pass via::

        model = SAC(
            ...,
            replay_buffer_class=CuratedTrainingBuffer,
            replay_buffer_kwargs={"core_fraction": 0.20, "sample_core_pct": 0.35},
        )

    Or inject post-construction::

        model.replay_buffer = CuratedTrainingBuffer(
            buffer_size, obs_space, act_space, device=device, n_envs=n_envs
        )

    Parameters
    ----------
    core_fraction : float
        Fraction of buffer slots reserved for core tier (default 0.20 = 20%).
    sample_core_pct : float
        Fraction of each training batch drawn from core tier (default 0.35).
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        core_fraction: float = 0.20,
        sample_core_pct: float = 0.35,
        **kwargs,
    ) -> None:
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        # Core tier parameters
        self._core_capacity = max(1, int(self.buffer_size * core_fraction))
        self._sample_core_pct = float(sample_core_pct)

        # Metadata arrays — tiny, no duplicate obs/action storage
        self._core_mask = np.zeros(self.buffer_size, dtype=bool)
        self._core_year = np.zeros(self.buffer_size, dtype=np.int16)
        self._core_added_at = np.zeros(self.buffer_size, dtype=np.int64)

        self._core_count: int = 0
        self._total_added: int = 0
        self._current_year: int = 0

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def set_current_year(self, year: int) -> None:
        """Call once per training week so core samples are tagged with era."""
        self._current_year = int(year)

    def get_era_distribution(self) -> dict:
        """Return {year: slot_count} for core tier. Used in logging."""
        if self._core_count == 0:
            return {}
        core_slots = np.where(self._core_mask)[0]
        years = self._core_year[core_slots]
        unique, counts = np.unique(years, return_counts=True)
        return {int(y): int(c) for y, c in zip(unique, counts)}

    def get_buffer_status(self) -> dict:
        """Return buffer health snapshot for logging."""
        upper = self.buffer_size if self.full else self.pos
        return {
            "total_slots": self.buffer_size,
            "used_slots": upper,
            "core_count": self._core_count,
            "fifo_count": upper - self._core_count,
            "total_added": self._total_added,
            "fill_pct": round(upper / self.buffer_size * 100, 1),
            "core_pct": round(self._core_count / max(upper, 1) * 100, 1),
            "era_distribution": self.get_era_distribution(),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Override: add()
    # ──────────────────────────────────────────────────────────────────────

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos,
    ) -> None:
        """Store transition, protecting core slots from FIFO eviction."""

        # ── Step 1: Skip over core slots so they aren't overwritten ──────
        if self._core_mask[self.pos]:
            start = self.pos
            for _ in range(self.buffer_size):
                self.pos = (self.pos + 1) % self.buffer_size
                if not self._core_mask[self.pos]:
                    break
            else:
                # All slots are core (rare edge case).
                # Evict the MOST RECENTLY added core slot to preserve
                # the oldest (most historically diverse) entries.
                newest = int(np.argmax(
                    np.where(self._core_mask, self._core_added_at, -1)
                ))
                self._core_mask[newest] = False
                self._core_count -= 1
                self.pos = newest

        # ── Step 2: Write transition to the located free slot ────────────
        super().add(obs, next_obs, action, reward, done, infos)
        self._total_added += 1

        # Correct the full flag (FIFO skip can confuse the parent's counter)
        if self._total_added >= self.buffer_size:
            self.full = True

        # Slot just written (pos was advanced by super().add)
        new_pos = (self.pos - 1) % self.buffer_size

        # ── Step 3: Reservoir sampling — decide if this slot joins core ──
        n = self._total_added
        k = self._core_capacity

        if n <= k:
            # Core not full yet: accept unconditionally
            if not self._core_mask[new_pos]:
                self._core_mask[new_pos] = True
                self._core_count += 1
            self._core_year[new_pos] = self._current_year
            self._core_added_at[new_pos] = n
        else:
            # Reservoir: keep with probability k/n
            j = np.random.randint(0, n)
            if j < k:
                # Evict one random existing core slot
                core_slots = np.where(self._core_mask)[0]
                if len(core_slots) > 0:
                    evict = core_slots[np.random.randint(len(core_slots))]
                    self._core_mask[evict] = False
                    # _core_count unchanged (replaced one slot)
                else:
                    self._core_count += 1
                self._core_mask[new_pos] = True
                self._core_year[new_pos] = self._current_year
                self._core_added_at[new_pos] = n

    # ──────────────────────────────────────────────────────────────────────
    # Override: sample()
    # ──────────────────────────────────────────────────────────────────────

    def sample(
        self,
        batch_size: int,
        env=None,
    ) -> ReplayBufferSamples:
        """Sample batch: ~35% cross-era core + ~65% recent FIFO."""
        core_slots = np.where(self._core_mask)[0]
        n_core_available = len(core_slots)

        if n_core_available == 0:
            # Core not populated yet — fall back to standard sampling
            return super().sample(batch_size, env=env)

        n_core = min(int(batch_size * self._sample_core_pct), n_core_available)
        n_fifo = batch_size - n_core

        # Sample from core (without replacement when possible)
        replace_core = n_core > n_core_available
        core_chosen = np.random.choice(n_core_available, n_core, replace=replace_core)
        core_inds = core_slots[core_chosen]

        # Sample from FIFO (non-core slots only)
        upper_bound = self.buffer_size if self.full else self.pos
        all_inds = np.arange(upper_bound)
        fifo_slots = all_inds[~self._core_mask[:upper_bound]]

        if len(fifo_slots) == 0:
            # No FIFO slots: sample extra from core
            extra = np.random.choice(n_core_available, n_fifo, replace=True)
            fifo_inds = core_slots[extra]
        elif len(fifo_slots) < n_fifo:
            fifo_inds = np.random.choice(fifo_slots, n_fifo, replace=True)
        else:
            fifo_inds = np.random.choice(fifo_slots, n_fifo, replace=False)

        # Combine and shuffle so core/FIFO are interleaved in the batch
        batch_inds = np.concatenate([core_inds, fifo_inds])
        np.random.shuffle(batch_inds)

        return self._get_samples(batch_inds, env=env)
