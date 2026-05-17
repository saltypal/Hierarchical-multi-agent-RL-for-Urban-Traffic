"""Isolated RL training module.

Supports PPO, A2C, DQN, and SAC via Stable-Baselines3.
Training is fully decoupled from runtime — invoked only from notebooks.

Saves models as PyTorch ``.pt`` state dicts + JSON config sidecars.
Auto-collects GNN training data during ward RL training episodes.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from stable_baselines3 import PPO, A2C, DQN, SAC
    from stable_baselines3.common.callbacks import BaseCallback

    HAS_SB3 = True
except ImportError:  # pragma: no cover
    HAS_SB3 = False

logger = logging.getLogger(__name__)

ALGORITHM_MAP = {
    "ppo": PPO if HAS_SB3 else None,
    "a2c": A2C if HAS_SB3 else None,
    "dqn": DQN if HAS_SB3 else None,
}


# ------------------------------------------------------------------
# GNN data collection callback
# ------------------------------------------------------------------

class GNNDataCollector(BaseCallback):
    """SB3 callback that logs ward state snapshots for GNN training.

    Every ``collection_interval`` steps, captures the ward summary
    from the environment and stores it for later GNN offline training.
    """

    def __init__(
        self,
        collection_interval: int = 30,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.collection_interval = collection_interval
        self.data_buffer: list[dict[str, Any]] = []
        self._step_count = 0

    def _on_step(self) -> bool:
        self._step_count += 1

        if self._step_count % self.collection_interval == 0:
            env = self.training_env.envs[0]
            if hasattr(env, "get_gnn_snapshot"):
                snapshot = env.get_gnn_snapshot()
                if snapshot is not None:
                    self.data_buffer.append(snapshot)

        return True

    def get_dataset(self) -> list[dict[str, Any]]:
        """Return collected GNN training samples with retroactive labels."""
        labeled: list[dict[str, Any]] = []

        for i in range(len(self.data_buffer) - 1):
            current = self.data_buffer[i]
            future = self.data_buffer[i + 1]
            labeled.append({
                "features": current["features"],
                "target": future["congestion"],
            })

        return labeled


# ------------------------------------------------------------------
# Training pipeline
# ------------------------------------------------------------------

def train_ward(
    ward_id: str,
    project_root: Path,
    algorithm: str = "ppo",
    total_timesteps: int = 10_000,
    gui: bool = False,
    scenario_id: str = "normal",
    collect_gnn_data: bool = True,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """Train a ward RL agent using Stable-Baselines3.

    Training is fully self-contained: creates the environment, trains,
    saves the model as ``.pt`` + config, and optionally collects GNN data.

    Args:
        ward_id: Ward identifier.
        project_root: Project root directory.
        algorithm: One of ``"ppo"``, ``"a2c"``, ``"dqn"``.
        total_timesteps: Total training steps.
        gui: Whether to show SUMO GUI during training.
        scenario_id: Traffic scenario for training.
        collect_gnn_data: If True, collect ward snapshots for GNN.
        results_dir: Where to save training results (defaults to ``results/training/``).

    Returns:
        Result dictionary with model path, training metrics, and GNN data path.
    """
    if not HAS_SB3:
        raise ImportError("stable-baselines3 is required for training")

    algo_cls = ALGORITHM_MAP.get(algorithm.lower())
    if algo_cls is None:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use: ppo, a2c, dqn")

    # Import adapter lazily to avoid circular imports
    from src.rl.sb3_ward_adapter import StableBaselinesWardEnv

    # Setup paths
    model_dir = project_root / "models" / algorithm / ward_id
    model_dir.mkdir(parents=True, exist_ok=True)
    if results_dir is None:
        results_dir = project_root / "results" / "training"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env_config = {
        "ward_id": ward_id,
        "project_root": str(project_root),
        "gui": gui,
        "scenario_id": scenario_id,
        "training_mode": True,
    }
    env = StableBaselinesWardEnv(env_config)

    # Setup callbacks
    callbacks = []
    gnn_collector = None
    if collect_gnn_data:
        gnn_collector = GNNDataCollector(collection_interval=30)
        callbacks.append(gnn_collector)

    # Create and train model
    logger.info(
        "Training %s on %s (%s, %d steps)",
        algorithm.upper(), ward_id, scenario_id, total_timesteps,
    )
    start_time = time.time()

    model = algo_cls("MlpPolicy", env, verbose=1, seed=42)
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    elapsed = time.time() - start_time
    logger.info("Training complete in %.1fs", elapsed)

    # Save model as PyTorch .pt state dict
    model_path = model_dir / "model.pt"
    if torch is not None:
        policy_state = model.policy.state_dict()
        torch.save(policy_state, model_path)
        logger.info("Model saved → %s", model_path)

    # Save config sidecar
    config = {
        "ward_id": ward_id,
        "algorithm": algorithm,
        "total_timesteps": total_timesteps,
        "scenario_id": scenario_id,
        "obs_dim": env.observation_space.shape[0],
        "action_dim": env.action_space.n,
        "hidden_dim": 64,
        "training_time_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    config_path = model_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    # Save GNN training data
    gnn_data_path = None
    if gnn_collector is not None:
        gnn_dataset = gnn_collector.get_dataset()
        if gnn_dataset and torch is not None:
            gnn_data_path = project_root / "models" / "gnn" / f"{ward_id}_gnn_data.pt"
            gnn_data_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(gnn_dataset, gnn_data_path)
            logger.info(
                "GNN data saved: %d samples → %s", len(gnn_dataset), gnn_data_path,
            )

    env.close()

    return {
        "ward_id": ward_id,
        "algorithm": algorithm,
        "model_path": str(model_path),
        "config_path": str(config_path),
        "gnn_data_path": str(gnn_data_path) if gnn_data_path else None,
        "training_time": elapsed,
        "total_timesteps": total_timesteps,
    }
