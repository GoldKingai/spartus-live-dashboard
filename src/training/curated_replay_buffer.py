"""Curated 3-tier replay buffer — Layer 2 of the anti-forgetting system.

The human brain replays important experiences during sleep to consolidate them
into long-term memory. This buffer does the same: it continuously reminds the
model of its most important past experiences so they don't fade.

Three tiers:
    CORE (30%):      Locked. Never evicted. Best trades, worst trades, regime
                     transitions. These are the foundational skills.
    SUPPORTING (40%): Slow eviction. Good variety of market conditions,
                     both sides, all sessions. Provides breadth.
    RECENT (30%):    Fast turnover. Live market transitions. This is where
                     adaptation happens.

Wraps SB3's ReplayBuffer, inheriting its data storage while overriding
sample() to enforce tier-weighted sampling.
"""

import logging
import random
from collections import deque
from typing import Any, Dict, Optional, Set

import numpy as np

try:
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.type_aliases import ReplayBufferSamples
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False
    ReplayBuffer = object  # type: ignore

log = logging.getLogger(__name__)


class CuratedReplayBuffer(ReplayBuffer if _SB3_AVAILABLE else object):
    """SB3 ReplayBuffer subclass with 3-tier priority sampling.

    Drop-in replacement for SAC's replay buffer:
        model.replay_buffer = CuratedReplayBuffer(...)

    Compatible with SB3's expected interface (add, sample, size, etc.).
    """

    def __init__(self, *args, config=None, **kwargs):
        if not _SB3_AVAILABLE:
            raise ImportError("stable_baselines3 required for CuratedReplayBuffer")
        super().__init__(*args, **kwargs)

        cfg = config
        self._core_pct = getattr(cfg, "finetune_buffer_core_pct", 0.30)
        self._supporting_pct = getattr(cfg, "finetune_buffer_supporting_pct", 0.40)
        self._recent_pct = getattr(cfg, "finetune_buffer_recent_pct", 0.30)
        self._min_reward_core = getattr(cfg, "finetune_min_reward_core", 1.0)

        # Tier index sets (track buffer positions by tier)
        self._core_indices: Set[int] = set()
        self._supporting_indices: Set[int] = set()
        self._recent_indices: Set[int] = set()

        # Metadata per transition: idx -> {reward, source, episode_pnl}
        self._metadata: Dict[int, Dict] = {}

        # Rolling deque of recent indices for fast "add to recent" tracking
        self._recent_queue: deque = deque(maxlen=int(self.buffer_size * self._recent_pct))

        # Track how many live transitions have been added
        self.live_transitions_added: int = 0
        self.historical_transitions_seeded: int = 0

        log.info(
            f"CuratedReplayBuffer: {self.buffer_size:,} capacity, "
            f"tiers: core={self._core_pct:.0%} / supporting={self._supporting_pct:.0%} "
            f"/ recent={self._recent_pct:.0%}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Add with tier classification
    # ─────────────────────────────────────────────────────────────────────────

    def add(self, obs, next_obs, action, reward, done, infos, *args, **kwargs):
        """Add transition and classify into the appropriate tier."""
        # Capture current write position before parent modifies it
        write_pos = self.pos

        # Call parent's add method
        super().add(obs, next_obs, action, reward, done, infos, *args, **kwargs)

        # Store metadata for this transition
        # reward may be a numpy array (n_envs), take env 0's value
        r = float(reward[0]) if hasattr(reward, "__len__") else float(reward)
        source = "live" if infos and infos[0].get("_live_source") else "historical"
        self._metadata[write_pos] = {"reward": r, "source": source}

        # Classify into tier
        tier = self._classify_tier(r, source)
        self._assign_to_tier(write_pos, tier)

        if source == "live":
            self.live_transitions_added += 1
        else:
            self.historical_transitions_seeded += 1

    def _classify_tier(self, reward: float, source: str) -> str:
        """Classify a transition into core / supporting / recent."""
        # Live data always goes to recent first
        if source == "live":
            return "recent"
        # High-reward historical transitions → core
        if abs(reward) >= self._min_reward_core:
            return "core"
        # Moderate historical transitions → supporting
        return "supporting"

    def _assign_to_tier(self, idx: int, tier: str) -> None:
        """Add index to tier set, evicting from other tiers if necessary."""
        # Remove from any existing tier assignment
        self._core_indices.discard(idx)
        self._supporting_indices.discard(idx)
        self._recent_indices.discard(idx)

        if tier == "core":
            # Core slots: once full, evict least important (lowest |reward|)
            max_core = int(self.buffer_size * self._core_pct)
            if len(self._core_indices) >= max_core:
                self._evict_lowest_reward(self._core_indices)
            self._core_indices.add(idx)
        elif tier == "supporting":
            max_supporting = int(self.buffer_size * self._supporting_pct)
            if len(self._supporting_indices) >= max_supporting:
                # Evict oldest (random from supporting, since we don't track order)
                if self._supporting_indices:
                    evict = random.choice(list(self._supporting_indices))
                    self._supporting_indices.discard(evict)
            self._supporting_indices.add(idx)
        else:  # recent
            # Recent uses the rolling deque — auto-evicts oldest
            if len(self._recent_queue) >= self._recent_queue.maxlen:
                old = self._recent_queue[0]  # oldest
                self._recent_indices.discard(old)
            self._recent_queue.append(idx)
            self._recent_indices.add(idx)

    def _evict_lowest_reward(self, tier_set: Set[int]) -> None:
        """Remove the lowest |reward| index from a tier set."""
        if not tier_set:
            return
        # Find index with lowest |reward| in metadata
        worst_idx = min(
            tier_set,
            key=lambda i: abs(self._metadata.get(i, {}).get("reward", 0.0))
        )
        tier_set.discard(worst_idx)

    # ─────────────────────────────────────────────────────────────────────────
    # Curated Sampling
    # ─────────────────────────────────────────────────────────────────────────

    def sample(self, batch_size: int, env=None) -> Any:
        """Sample with tier-weighted priority.

        30% from core, 40% from supporting, 30% from recent.
        Falls back to standard sampling if tiers are underpopulated.
        """
        n_core = int(batch_size * self._core_pct)
        n_supporting = int(batch_size * self._supporting_pct)
        n_recent = batch_size - n_core - n_supporting

        # Clamp to available
        n_core = min(n_core, len(self._core_indices))
        n_supporting = min(n_supporting, len(self._supporting_indices))
        n_recent = min(n_recent, len(self._recent_indices))

        total_curated = n_core + n_supporting + n_recent

        if total_curated < batch_size // 2:
            # Not enough curated indices yet — fall back to standard sampling
            return super().sample(batch_size, env=env)

        # Sample from each tier
        indices = []
        if n_core > 0:
            indices.extend(random.sample(list(self._core_indices), n_core))
        if n_supporting > 0:
            indices.extend(random.sample(list(self._supporting_indices), n_supporting))
        if n_recent > 0:
            indices.extend(random.sample(list(self._recent_indices), n_recent))

        # Fill remaining slots with standard random sampling
        n_fill = batch_size - len(indices)
        if n_fill > 0:
            valid_size = self.size()
            fill = np.random.randint(0, valid_size, size=n_fill)
            indices.extend(fill.tolist())

        # Use parent's _get_samples for actual data retrieval
        try:
            indices_arr = np.array(indices, dtype=np.int64)
            return self._get_samples(indices_arr, env=env)
        except Exception:
            # Fallback to standard sampling
            return super().sample(batch_size, env=env)

    # ─────────────────────────────────────────────────────────────────────────
    # Seeding from Historical Buffer
    # ─────────────────────────────────────────────────────────────────────────

    def seed_from_historical(self, historical_buffer, n_samples: Optional[int] = None) -> int:
        """Seed this buffer with curated transitions from a trained model's replay buffer.

        Args:
            historical_buffer: SB3 ReplayBuffer from trained model (loaded via
                               model.load_replay_buffer()).
            n_samples: Max transitions to copy. None = copy all available.

        Returns:
            Number of transitions seeded.
        """
        buf_size = historical_buffer.size()
        if buf_size == 0:
            log.warning("CuratedReplayBuffer: historical buffer is empty, skipping seed")
            return 0

        if n_samples is None or n_samples > buf_size:
            n_samples = buf_size

        log.info(f"CuratedReplayBuffer: seeding from {n_samples:,} historical transitions...")

        # Sample randomly from historical buffer
        batch_size = min(1024, n_samples)
        n_batches = n_samples // batch_size
        seeded = 0

        for _ in range(n_batches):
            try:
                data = historical_buffer.sample(batch_size)
                # Extract arrays from SB3 ReplayBufferSamples
                obs = data.observations.cpu().numpy()
                next_obs = data.next_observations.cpu().numpy()
                actions = data.actions.cpu().numpy()
                rewards = data.rewards.cpu().numpy()
                dones = data.dones.cpu().numpy()
                infos = [{"_live_source": False}] * batch_size

                for i in range(batch_size):
                    self.add(
                        obs[i:i+1], next_obs[i:i+1], actions[i:i+1],
                        rewards[i:i+1], dones[i:i+1], [infos[i]]
                    )
                    seeded += 1
            except Exception as e:
                log.warning(f"CuratedReplayBuffer: seed batch failed: {e}")
                break

        log.info(
            f"CuratedReplayBuffer: seeded {seeded:,} transitions — "
            f"core={len(self._core_indices)}, "
            f"supporting={len(self._supporting_indices)}, "
            f"recent={len(self._recent_indices)}"
        )
        return seeded

    # ─────────────────────────────────────────────────────────────────────────
    # Status & Monitoring
    # ─────────────────────────────────────────────────────────────────────────

    def get_tier_status(self) -> Dict:
        """Return current tier composition for dashboard display."""
        total = self.size()
        return {
            "total": total,
            "core": len(self._core_indices),
            "supporting": len(self._supporting_indices),
            "recent": len(self._recent_indices),
            "live_added": self.live_transitions_added,
            "historical_seeded": self.historical_transitions_seeded,
            "live_pct": (
                self.live_transitions_added / max(1, total) * 100
            ),
            "buffer_fill_pct": total / self.buffer_size * 100,
        }

    def promote_to_core(self, idx: int) -> None:
        """Manually promote a transition to the core tier (e.g., exceptional trade)."""
        self._supporting_indices.discard(idx)
        self._recent_indices.discard(idx)
        self._recent_queue = deque(
            (i for i in self._recent_queue if i != idx),
            maxlen=self._recent_queue.maxlen
        )
        self._core_indices.add(idx)
        if idx in self._metadata:
            self._metadata[idx]["tier_override"] = "core"
