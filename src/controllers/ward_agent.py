"""Ward-level RL inference wrapper.

Loads a trained PyTorch ``.pt`` model state dict and exposes
a clean ``get_action()`` interface for runtime use.

Does NOT train. Does NOT import ``traci``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

logger = logging.getLogger(__name__)


# Default policy network architecture (must match SB3 PPO MlpPolicy)
OBSERVATION_DIM = 12
ACTION_DIM = 10


class WardPolicyNetwork(nn.Module):
    """Minimal MLP policy matching SB3's default ``MlpPolicy`` architecture."""

    def __init__(
        self,
        obs_dim: int = OBSERVATION_DIM,
        action_dim: int = ACTION_DIM,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WardAgent:
    """Inference-only ward RL agent.

    Loads a trained model from ``models/{algorithm}/ward_{id}/model.pt``
    and returns actions given observations.
    """

    def __init__(self, ward_id: str, model_path: Path) -> None:
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for WardAgent")

        self.ward_id = ward_id
        self.model_path = model_path

        # Load config sidecar if exists
        config_path = model_path.parent / "config.json"
        self.config: dict = {}
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as fh:
                self.config = json.load(fh)

        obs_dim = self.config.get("obs_dim", OBSERVATION_DIM)
        action_dim = self.config.get("action_dim", ACTION_DIM)
        hidden = self.config.get("hidden_dim", 64)

        self.policy = WardPolicyNetwork(obs_dim, action_dim, hidden)

        if model_path.exists():
            self.policy.load_state_dict(
                torch.load(model_path, weights_only=True)
            )
            logger.info("Ward agent %s loaded ← %s", ward_id, model_path)
        else:
            logger.warning("No model found at %s, using random policy", model_path)

        self.policy.eval()

    def get_action(self, observation: np.ndarray) -> int:
        """Select an action given the observation vector.

        Args:
            observation: Ward observation array of shape ``(12,)``.

        Returns:
            Action index from ``WardAction`` enum.
        """
        with torch.no_grad():
            obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            logits = self.policy(obs_tensor)
            action = torch.argmax(logits, dim=-1).item()

        return int(action)

    def get_action_probs(self, observation: np.ndarray) -> np.ndarray:
        """Return action probability distribution for analysis."""
        with torch.no_grad():
            obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            logits = self.policy(obs_tensor)
            probs = torch.softmax(logits, dim=-1)

        return probs.squeeze(0).numpy()
