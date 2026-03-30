"""Elastic Weight Consolidation (EWC) — Layer 3 of the anti-forgetting system.

Protects important weights from being overwritten during live fine-tuning.
Inspired by Kirkpatrick et al. (2017): "Overcoming catastrophic forgetting in
neural networks."

How it works:
1. After training completes, compute the Fisher Information diagonal for each
   weight — this measures how sensitive the model's performance is to changes
   in that weight.
2. High Fisher importance = weight encodes a fundamental trading skill (e.g.,
   "detect pullback → go short"). These resist change.
3. Low Fisher importance = weight is regime-specific (e.g., "volatility is
   usually X pips"). These can adapt freely.
4. During fine-tuning, an EWC penalty is added after each SAC training step:
   EWC_loss = λ * Σ_i F_i * (θ_i - θ*_i)²
   where F_i = Fisher importance, θ*_i = frozen reference weight.

Brain analogy: Synaptic consolidation — connections that were important for
learned skills become physically more resistant to change (through protein
synthesis at the synapse).
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


class EWCProtection:
    """Elastic Weight Consolidation for SAC actor and critic networks.

    Usage:
        # After training, compute Fisher matrix
        ewc = EWCProtection(ewc_lambda=5000.0)
        ewc.compute_fisher(model, replay_buffer, n_samples=2000)
        ewc.save("storage/finetune/ewc_state.json")

        # During fine-tuning, after each model.learn() call:
        ewc_loss = ewc.penalty(model)
        ewc_loss.backward()
        model.actor.optimizer.step()
        model.critic.optimizer.step()
    """

    def __init__(self, ewc_lambda: float = 5000.0):
        self.ewc_lambda = ewc_lambda
        self.enabled = False

        # Fisher diagonals: param_name -> torch.Tensor
        self._fisher: Dict[str, torch.Tensor] = {}

        # Frozen reference weights: param_name -> torch.Tensor
        self._frozen_params: Dict[str, torch.Tensor] = {}

        # Metadata
        self._n_samples_used: int = 0
        self._computed_at_week: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Fisher Computation
    # ─────────────────────────────────────────────────────────────────────────

    def compute_fisher(
        self,
        model,
        n_samples: int = 2000,
        device: Optional[str] = None,
    ) -> None:
        """Compute the diagonal Fisher Information Matrix.

        Uses the empirical Fisher approximation: for each sample, compute the
        gradient of the log-policy w.r.t. parameters, and accumulate the
        squared gradient as the Fisher diagonal estimate.

        Args:
            model: SB3 SAC model (must have .actor, .replay_buffer).
            n_samples: Number of replay buffer samples to use.
            device: Device to run computation on. None = model's device.
        """
        if not hasattr(model, "actor") or not hasattr(model, "replay_buffer"):
            log.warning("EWC: model missing actor or replay_buffer — skipping")
            return

        buf = model.replay_buffer
        if buf.size() < n_samples:
            n_samples = buf.size()
            log.info(f"EWC: buffer has {n_samples} samples, using all")

        if n_samples == 0:
            log.warning("EWC: replay buffer is empty — cannot compute Fisher, skipping")
            return

        if device is None:
            device = next(model.actor.parameters()).device

        log.info(f"EWC: computing Fisher diagonal from {n_samples} samples...")

        # Zero out Fisher accumulators
        fisher_acc: Dict[str, torch.Tensor] = {}
        for name, param in self._iter_params(model):
            fisher_acc[name] = torch.zeros_like(param.data, device=device)

        model.actor.zero_grad()

        # Accumulate squared gradients over samples
        n_valid = 0
        batch_size = min(256, n_samples)
        n_batches = max(1, n_samples // batch_size)

        for _ in range(n_batches):
            try:
                data = buf.sample(batch_size)
                obs = data.observations

                # Forward pass through actor to get action distribution
                actions_pi, log_prob = model.actor.action_log_prob(obs)

                # Use log_prob as the loss (Fisher is expectation of sq gradient of log-policy)
                loss = -log_prob.mean()
                loss.backward()

                for name, param in self._iter_params(model):
                    if param.grad is not None:
                        fisher_acc[name] += param.grad.data.pow(2)
                        n_valid += 1

                model.actor.zero_grad()
            except Exception as e:
                log.warning(f"EWC: sample batch failed: {e}")
                continue

        # Average over batches and normalize
        if n_valid > 0:
            scale = 1.0 / n_batches
            for name in fisher_acc:
                fisher_acc[name] *= scale

        # Store Fisher diagonals and frozen reference weights
        self._fisher = {}
        self._frozen_params = {}
        for name, param in self._iter_params(model):
            self._fisher[name] = fisher_acc[name].detach().clone()
            self._frozen_params[name] = param.data.detach().clone()

        self._n_samples_used = n_samples
        self.enabled = True

        total_params = sum(p.numel() for p in fisher_acc.values())
        log.info(
            f"EWC: Fisher computed. {len(self._fisher)} param tensors, "
            f"{total_params:,} total params, λ={self.ewc_lambda}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # EWC Penalty
    # ─────────────────────────────────────────────────────────────────────────

    def penalty(self, model) -> torch.Tensor:
        """Compute the EWC penalty loss.

        EWC_loss = λ * Σ_i F_i * (θ_i - θ*_i)²

        Args:
            model: SAC model with current (fine-tuned) weights.

        Returns:
            Scalar tensor with EWC penalty. 0.0 if not enabled.
        """
        if not self.enabled or not self._fisher:
            device = next(model.actor.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=False)

        device = next(model.actor.parameters()).device
        penalty = torch.tensor(0.0, device=device)

        for name, param in self._iter_params(model):
            if name not in self._fisher or name not in self._frozen_params:
                continue
            fisher = self._fisher[name].to(device)
            frozen = self._frozen_params[name].to(device)
            penalty = penalty + (fisher * (param - frozen).pow(2)).sum()

        return self.ewc_lambda * penalty

    def apply_penalty(self, model) -> float:
        """Compute and backprop the EWC penalty, step actor optimizers.

        Called after each model.learn() to correct weight drift.

        Returns:
            Float value of the EWC penalty for logging.
        """
        if not self.enabled:
            return 0.0

        try:
            # Zero gradients first
            for opt in self._get_optimizers(model):
                opt.zero_grad()

            ewc_loss = self.penalty(model)
            penalty_val = ewc_loss.item()

            if penalty_val > 0:
                ewc_loss.backward()
                for opt in self._get_optimizers(model):
                    opt.step()

            return penalty_val
        except Exception as e:
            log.warning(f"EWC: penalty application failed: {e}")
            return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Weight Divergence Monitoring
    # ─────────────────────────────────────────────────────────────────────────

    def weight_divergence(self, model) -> float:
        """Compute mean absolute weight change from frozen reference.

        Returns:
            Mean |θ_i - θ*_i| across all EWC-tracked parameters.
        """
        if not self.enabled or not self._frozen_params:
            return 0.0

        total_diff = 0.0
        total_params = 0

        for name, param in self._iter_params(model):
            if name not in self._frozen_params:
                continue
            frozen = self._frozen_params[name]
            diff = (param.data - frozen).abs().mean().item()
            total_diff += diff
            total_params += 1

        return total_diff / max(1, total_params)

    def fisher_importance_stats(self) -> Dict:
        """Return summary stats on Fisher importance scores."""
        if not self._fisher:
            return {}

        all_values = torch.cat([f.flatten().cpu() for f in self._fisher.values()])
        return {
            "mean": float(all_values.mean()),
            "max": float(all_values.max()),
            "pct_high": float((all_values > all_values.mean()).float().mean()),
            "n_params": int(all_values.numel()),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save Fisher diagonals and frozen weights to .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "fisher": self._fisher,
                "frozen_params": self._frozen_params,
                "ewc_lambda": self.ewc_lambda,
                "n_samples_used": self._n_samples_used,
                "computed_at_week": self._computed_at_week,
                "enabled": self.enabled,
            },
            str(path.with_suffix(".pt")),
        )
        log.info(f"EWC state saved to {path.with_suffix('.pt')}")

    def load(self, path: str) -> bool:
        """Load saved EWC state. Returns True on success."""
        path = Path(path).with_suffix(".pt")
        if not path.exists():
            log.debug(f"EWC state not found: {path} (will compute on first fine-tune)")
            return False
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=False)
            self._fisher = state.get("fisher", {})
            self._frozen_params = state.get("frozen_params", {})
            self.ewc_lambda = state.get("ewc_lambda", self.ewc_lambda)
            self._n_samples_used = state.get("n_samples_used", 0)
            self._computed_at_week = state.get("computed_at_week", 0)
            self.enabled = state.get("enabled", bool(self._fisher))
            log.info(
                f"EWC state loaded from {path}: {len(self._fisher)} param tensors, "
                f"λ={self.ewc_lambda}, enabled={self.enabled}"
            )
            return True
        except Exception as e:
            log.error(f"EWC load failed: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _iter_params(model):
        """Iterate named parameters of actor (and optionally critic)."""
        for name, param in model.actor.named_parameters():
            if param.requires_grad:
                yield f"actor.{name}", param

    @staticmethod
    def _get_optimizers(model) -> list:
        """Return list of optimizers to step after EWC backward."""
        opts = []
        if hasattr(model.actor, "optimizer"):
            opts.append(model.actor.optimizer)
        return opts
