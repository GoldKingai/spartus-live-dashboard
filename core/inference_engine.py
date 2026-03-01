"""InferenceEngine -- Run live inference with a trained SAC model.

Takes a 670-dim observation vector and produces 4 continuous action
values that the decision engine translates into trading decisions.

Action semantics (all outputs are in [-1, +1] from the SAC actor):
    action[0]  direction      -- negative = SELL, positive = BUY
    action[1]  conviction     -- mapped to [0, 1] via (x+1)/2
    action[2]  exit_urgency   -- mapped to [0, 1] via (x+1)/2
    action[3]  sl_adjustment  -- mapped to [0, 1] via (x+1)/2

Usage:
    from core.model_loader import ModelLoader
    from core.inference_engine import InferenceEngine

    loader = ModelLoader(config)
    components = loader.load()
    engine = InferenceEngine(components["model"])

    obs = build_observation(...)   # np.ndarray of shape (670,)
    decision = engine.predict(obs)
    # decision = {
    #     "direction": -0.72,
    #     "conviction": 0.85,
    #     "exit_urgency": 0.12,
    #     "sl_adjustment": 0.63,
    # }
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

log = logging.getLogger(__name__)

# Observation bounds matching training (trade_env clips to [-10, 10])
_OBS_CLIP_MIN = -10.0
_OBS_CLIP_MAX = 10.0


class InferenceEngine:
    """Loads a trained SAC model and runs deterministic inference.

    Takes an N-dim observation (determined by the model) and produces
    action values.  Uses deterministic policy (no exploration noise)
    for live trading.
    """

    def __init__(self, model: object) -> None:
        """Initialise the inference engine.

        Args:
            model: A loaded SB3 SAC model object (from ModelLoader).
        """
        self._model = model
        self._call_count: int = 0
        self._nan_warning_count: int = 0
        self._inf_warning_count: int = 0
        self._obs_dim: int = 0

        # Read model's actual observation dimension
        try:
            obs_shape = model.observation_space.shape
            self._obs_dim = obs_shape[0]
            log.info(
                "InferenceEngine initialised: obs_space=%s, action_space=%s",
                obs_shape,
                model.action_space.shape,
            )
        except Exception as exc:
            log.warning("Could not inspect model spaces: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, observation: np.ndarray) -> Dict[str, float]:
        """Run model inference and return interpreted action dict.

        Args:
            observation: 1-D numpy array of shape (670,) or (1, 670).
                Any NaN/Inf values are replaced with 0.0 and a warning
                is logged.  Values are clipped to [-10, +10].

        Returns:
            Dict with keys:
                direction      (float) -- raw action[0], range [-1, +1].
                                          Negative = SELL, positive = BUY.
                conviction     (float) -- (action[1]+1)/2, range [0, 1].
                                          Higher = more confident.
                exit_urgency   (float) -- (action[2]+1)/2, range [0, 1].
                                          Higher = more urgent to exit.
                sl_adjustment  (float) -- (action[3]+1)/2, range [0, 1].
                                          Higher = tighter trailing SL.
        """
        raw = self.predict_raw(observation)

        # Map tanh outputs [-1, +1] to semantic ranges
        direction = float(raw[0])                      # [-1, +1]
        conviction = float((raw[1] + 1.0) / 2.0)      # [0, 1]
        exit_urgency = float((raw[2] + 1.0) / 2.0)    # [0, 1]
        sl_adjustment = float((raw[3] + 1.0) / 2.0)   # [0, 1]

        return {
            "direction": direction,
            "conviction": conviction,
            "exit_urgency": exit_urgency,
            "sl_adjustment": sl_adjustment,
        }

    def predict_raw(self, observation: np.ndarray) -> np.ndarray:
        """Run model inference and return the raw 4-dim action array.

        This method handles:
            1. NaN/Inf sanitisation (replaced with 0.0).
            2. Clipping to [-10, +10].
            3. Reshaping to (1, 670) batch dimension.
            4. Deterministic prediction (no exploration noise).

        Args:
            observation: 1-D array of shape (670,) or 2-D of shape (1, 670).

        Returns:
            1-D numpy array of shape (4,) with raw action values in [-1, +1].
        """
        obs = np.asarray(observation, dtype=np.float32)

        # Flatten if needed (handle both (670,) and (1, 670))
        if obs.ndim > 1:
            obs = obs.flatten()

        # Validate dimension against the model's actual obs space
        if self._obs_dim > 0 and obs.shape[0] != self._obs_dim:
            log.warning(
                "Observation dimension mismatch: got %d, model expects %d",
                obs.shape[0],
                self._obs_dim,
            )

        # Sanitise NaN / Inf
        nan_mask = np.isnan(obs)
        inf_mask = np.isinf(obs)

        if nan_mask.any():
            nan_count = int(nan_mask.sum())
            self._nan_warning_count += 1
            # Log first 10, then every 100th occurrence to avoid log spam
            if self._nan_warning_count <= 10 or self._nan_warning_count % 100 == 0:
                nan_indices = np.where(nan_mask)[0]
                log.warning(
                    "Observation contains %d NaN values (indices: %s) "
                    "[warning #%d] -- replaced with 0.0",
                    nan_count,
                    nan_indices[:10].tolist(),
                    self._nan_warning_count,
                )
            obs[nan_mask] = 0.0

        if inf_mask.any():
            inf_count = int(inf_mask.sum())
            self._inf_warning_count += 1
            if self._inf_warning_count <= 10 or self._inf_warning_count % 100 == 0:
                inf_indices = np.where(inf_mask)[0]
                log.warning(
                    "Observation contains %d Inf values (indices: %s) "
                    "[warning #%d] -- replaced with 0.0",
                    inf_count,
                    inf_indices[:10].tolist(),
                    self._inf_warning_count,
                )
            obs[inf_mask] = 0.0

        # Clip to training bounds
        obs = np.clip(obs, _OBS_CLIP_MIN, _OBS_CLIP_MAX)

        # Reshape to (1, obs_dim) for SB3 batch dimension
        obs_batch = obs.reshape(1, -1)

        # Deterministic prediction (no exploration noise)
        action, _states = self._model.predict(obs_batch, deterministic=True)

        self._call_count += 1

        # action shape is (1, 4) from batch prediction; squeeze to (4,)
        return action.flatten()

    def get_model_info(self) -> Dict[str, Any]:
        """Return model architecture and state information.

        Returns:
            Dict with keys:
                obs_dim         -- observation space dimension
                action_dim      -- action space dimension
                policy_class    -- policy class name
                pi_net_arch     -- actor network layer sizes
                qf_net_arch     -- critic network layer sizes
                device          -- torch device the model is on
                total_timesteps -- total training timesteps (if available)
                call_count      -- number of predictions made
                nan_warnings    -- number of NaN warning events
                inf_warnings    -- number of Inf warning events
        """
        info: Dict[str, Any] = {
            "call_count": self._call_count,
            "nan_warnings": self._nan_warning_count,
            "inf_warnings": self._inf_warning_count,
        }

        try:
            info["obs_dim"] = self._model.observation_space.shape[0]
        except Exception:
            info["obs_dim"] = None

        try:
            info["action_dim"] = self._model.action_space.shape[0]
        except Exception:
            info["action_dim"] = None

        try:
            info["policy_class"] = type(self._model.policy).__name__
        except Exception:
            info["policy_class"] = None

        try:
            info["device"] = str(self._model.device)
        except Exception:
            info["device"] = None

        # Extract network architecture from policy kwargs
        try:
            policy_kwargs = self._model.policy.net_arch
            if isinstance(policy_kwargs, dict):
                info["pi_net_arch"] = policy_kwargs.get("pi", [])
                info["qf_net_arch"] = policy_kwargs.get("qf", [])
            elif isinstance(policy_kwargs, list):
                # SB3 v2.x format: net_arch is a list or dict
                info["pi_net_arch"] = policy_kwargs
                info["qf_net_arch"] = policy_kwargs
            else:
                info["pi_net_arch"] = str(policy_kwargs)
                info["qf_net_arch"] = str(policy_kwargs)
        except Exception:
            info["pi_net_arch"] = None
            info["qf_net_arch"] = None

        # Total timesteps
        try:
            info["total_timesteps"] = int(self._model.num_timesteps)
        except Exception:
            info["total_timesteps"] = None

        # Learning rate (current)
        try:
            lr = self._model.learning_rate
            if callable(lr):
                info["learning_rate"] = "scheduled"
            else:
                info["learning_rate"] = float(lr)
        except Exception:
            info["learning_rate"] = None

        # Entropy coefficient (alpha)
        try:
            log_ent = self._model.log_ent_coef
            if hasattr(log_ent, "item"):
                import torch
                info["entropy_alpha"] = float(torch.exp(log_ent).item())
            else:
                info["entropy_alpha"] = None
        except Exception:
            info["entropy_alpha"] = None

        return info
