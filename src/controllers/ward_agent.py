"""Ward-level RL inference wrapper.

Loads trained PyTorch state dicts and exposes a clean ``get_action()``
interface for runtime use.

Does NOT train. Does NOT import ``traci``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

logger = logging.getLogger(__name__)

OBSERVATION_DIM = 12
ACTION_DIM = 10


def _resolve_device() -> torch.device:
    preferred = os.getenv("HMRL_INFER_DEVICE", "cuda").lower()
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WardMLPPolicyNetwork(nn.Module):
    """Compact MLP policy for policy-gradient checkpoints."""

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


class _SB3QNet(nn.Module):
    """Structure-compatible Q-network for SB3 DQN checkpoints."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_net(x)


class DQNPolicyNetwork(nn.Module):
    """Policy wrapper matching the state-dict layout saved by SB3 DQN."""

    def __init__(
        self,
        obs_dim: int = OBSERVATION_DIM,
        action_dim: int = ACTION_DIM,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.q_net = _SB3QNet(obs_dim, action_dim, hidden)
        self.q_net_target = _SB3QNet(obs_dim, action_dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_net(x)


class WardAgent:
    """Inference-only ward RL agent."""

    def __init__(self, ward_id: str, model_path: Path) -> None:
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for WardAgent")

        self.ward_id = ward_id
        self.model_path = model_path
        self.device = _resolve_device()

        config_path = model_path.parent / "config.json"
        self.config: dict = {}
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as fh:
                self.config = json.load(fh)

        self.algorithm = str(self.config.get("algorithm", "dqn")).lower()
        obs_dim = int(self.config.get("obs_dim", OBSERVATION_DIM))
        action_dim = int(self.config.get("action_dim", ACTION_DIM))
        hidden = int(self.config.get("hidden_dim", 64))

        if self.algorithm == "dqn":
            self.policy: nn.Module = DQNPolicyNetwork(obs_dim, action_dim, hidden)
        else:
            self.policy = WardMLPPolicyNetwork(obs_dim, action_dim, hidden)

        self.policy.to(self.device)

        if model_path.exists():
            state_dict = torch.load(
                model_path,
                map_location=self.device,
                weights_only=True,
            )
            self.policy.load_state_dict(state_dict, strict=True)
            logger.info(
                "Ward agent %s loaded [%s] on %s <- %s",
                ward_id, self.algorithm.upper(), self.device, model_path,
            )
        else:
            logger.warning("No model found at %s, using random policy", model_path)

        self.policy.eval()

    def _forward_logits(self, observation: np.ndarray) -> torch.Tensor:
        obs_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device,
        ).unsqueeze(0)
        return self.policy(obs_tensor)

    def get_action(self, observation: np.ndarray) -> int:
        """Select an action given the observation vector."""
        with torch.no_grad():
            logits = self._forward_logits(observation)
            action = torch.argmax(logits, dim=-1).item()
        return int(action)

    def get_action_probs(self, observation: np.ndarray) -> np.ndarray:
        """Return action probability distribution for analysis."""
        with torch.no_grad():
            logits = self._forward_logits(observation)
            probs = torch.softmax(logits, dim=-1)
        return probs.squeeze(0).detach().cpu().numpy()
