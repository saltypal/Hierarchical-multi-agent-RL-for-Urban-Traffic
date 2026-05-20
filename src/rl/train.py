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
    """SB3 callback that logs ward temporal traces for area training.

    Every ``collection_interval`` steps, captures the ward summary
    from the environment and stores it for later temporal area training.
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
            if hasattr(env, "unwrapped"):
                env = env.unwrapped

            if hasattr(env, "get_temporal_trace"):
                trace = env.get_temporal_trace()
                if trace:
                    self.data_buffer.append({
                        "step": self._step_count,
                        "trace": trace[-30:],
                    })
            elif hasattr(env, "get_gnn_snapshot"):
                snapshot = env.get_gnn_snapshot()
                if snapshot is not None:
                    self.data_buffer.append(snapshot)

        return True

    def get_dataset(self) -> list[dict[str, Any]]:
        """Return collected training samples with retroactive labels."""
        labeled: list[dict[str, Any]] = []

        for i in range(len(self.data_buffer) - 1):
            current = self.data_buffer[i]
            future = self.data_buffer[i + 1]

            if "trace" in current and current["trace"]:
                labeled.append({
                    "features": current["trace"],
                    "target": future.get("trace", [{}])[-1].get("congestion_score", 0.0) if future.get("trace") else 0.0,
                })
                continue

            # Fallback path: build correct 8-dimensional feature matrix
            # current["features"] is a 7-element array from build_ward_feature_frame:
            # [congestion, queue, avg_speed, inflow, outflow, incident_flag, ambulance_flag]
            curr_feats = current["features"]
            congestion = current.get("congestion", 0.0)
            queue = float(curr_feats[1])
            avg_speed = float(curr_feats[2])
            outflow = float(curr_feats[4])  # outflow represents throughput/outflow in GNN
            incident_flag = float(curr_feats[5])

            feats = np.zeros(8, dtype=np.float32)
            feats[0] = congestion
            feats[2] = queue
            feats[4] = avg_speed
            feats[5] = outflow
            feats[6] = incident_flag
            feats[7] = current.get("city_cap", 1.0)

            # Compute deltas using current and past state to avoid lookahead target leakage
            if i > 0:
                past = self.data_buffer[i - 1]
                past_feats = past["features"]
                feats[1] = congestion - past.get("congestion", 0.0)
                feats[3] = queue - float(past_feats[1])
            else:
                feats[1] = 0.0
                feats[3] = 0.0

            labeled.append({
                "features": feats,
                "target": future.get("congestion", 0.0),
            })

        return labeled

# ------------------------------------------------------------------
# Episode tracker callback
# ------------------------------------------------------------------

class EpisodeTrackerCallback(BaseCallback):
    """SB3 callback that tracks episodes, records rewards, and halts training."""

    def __init__(self, max_episodes: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0
        self.episode_rewards: list[float] = []
        self.current_reward = 0.0
        
        # Live Jupyter-friendly progress bar
        from tqdm.auto import tqdm
        self.pbar = tqdm(total=max_episodes, desc="RL Agent Training")

    def _on_step(self) -> bool:
        self.current_reward += self.locals["rewards"][0]
        
        # SB3 vec env returns a list/array of dones
        if self.locals["dones"][0]:
            self.episode_count += 1
            self.episode_rewards.append(float(self.current_reward))
            
            # Update progress bar
            self.pbar.update(1)
            
            # Set postfix stats on the bar
            last_rew = self.current_reward
            avg_10 = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
            self.pbar.set_postfix({
                "ep_reward": f"{last_rew:.1f}",
                "avg_10_ep": f"{avg_10:.1f}"
            })
            
            self.current_reward = 0.0

            if self.episode_count >= self.max_episodes:
                self.pbar.close()
                return False  # Halt training exactly at max_episodes
                
        return True


# ------------------------------------------------------------------
# Training pipeline
# ------------------------------------------------------------------

def train_global_agent(
    ward_ids: list[str],
    scenario_ids: list[str],
    project_root: Path,
    algorithm: str = "ppo",
    episodes: int = 400,
    gui: bool = False,
    collect_gnn_data: bool = True,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """Train a multi-ward, multi-scenario RL agent using Stable-Baselines3.

    Training is fully self-contained: creates the environment, trains,
    saves the model as ``.pt`` + config, and optionally collects GNN data.

    Args:
        ward_ids: List of Ward identifiers to randomize over.
        scenario_ids: List of Traffic scenarios to randomize over.
        project_root: Project root directory.
        algorithm: One of ``"ppo"``, ``"a2c"``, ``"dqn"``.
        episodes: Total training episodes.
        gui: Whether to show SUMO GUI during training.
        collect_gnn_data: If True, collect ward temporal traces for the area model.
        results_dir: Where to save training results (defaults to ``results/training/``).

    Returns:
        Result dictionary with model path, training metrics, rewards, and GNN data path.
    """
    if not HAS_SB3:
        raise ImportError("stable-baselines3 is required for training")

    algo_cls = ALGORITHM_MAP.get(algorithm.lower())
    if algo_cls is None:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use: ppo, a2c, dqn")

    # Import adapter lazily to avoid circular imports
    from src.rl.sb3_ward_adapter import StableBaselinesWardEnv

    # Setup paths
    model_dir = project_root / "models" / algorithm / "global_agent"
    model_dir.mkdir(parents=True, exist_ok=True)
    if results_dir is None:
        results_dir = project_root / "results" / "training"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env_config = {
        "ward_id": ward_ids,
        "project_root": str(project_root),
        "gui": gui,
        "scenario_id": scenario_ids,
        "training_mode": True,
        "decision_interval_steps": 30,
        "max_simulation_steps": 1200,
    }
    env = StableBaselinesWardEnv(env_config)

    # Setup callbacks
    callbacks = []
    
    episode_tracker = EpisodeTrackerCallback(max_episodes=episodes)
    callbacks.append(episode_tracker)
    
    gnn_collector = None
    if collect_gnn_data:
        gnn_collector = GNNDataCollector(collection_interval=1)
        callbacks.append(gnn_collector)

    # Create and train model
    logger.info(
        "Training %s on Multi-Ward Global Setup (%d wards, %d episodes)",
        algorithm.upper(), len(ward_ids), episodes,
    )
    start_time = time.time()

    # Use CPU for MLP policy networks to avoid PCIe host-to-device overhead
    model = algo_cls("MlpPolicy", env, verbose=0, seed=42, device="cpu")
    
    # Run for practically infinite timesteps, EpisodeTrackerCallback will halt it
    model.learn(total_timesteps=int(1e9), callback=callbacks)

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
        "ward_ids": ward_ids,
        "scenario_ids": scenario_ids,
        "algorithm": algorithm,
        "episodes_completed": int(episode_tracker.episode_count),
        "obs_dim": int(env.observation_space.shape[0]),
        "action_dim": int(env.action_space.n),
        "hidden_dim": 64,
        "temporal_window": 30,
        "training_time_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    config_path = model_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    # Save temporal ward training data
    gnn_data_path = None
    if gnn_collector is not None:
        gnn_dataset = gnn_collector.get_dataset()
        if gnn_dataset and torch is not None:
            temporal_data_path = project_root / "models" / "gnn" / "global_temporal_data.pt"
            temporal_data_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(gnn_dataset, temporal_data_path)
            gnn_data_path = temporal_data_path
            logger.info(
                "Temporal ward data saved: %d samples → %s", len(gnn_dataset), temporal_data_path,
            )

            legacy_path = project_root / "models" / "gnn" / "global_gnn_data.pt"
            torch.save(gnn_dataset, legacy_path)

    env.close()

    return {
        "ward_ids": ward_ids,
        "algorithm": algorithm,
        "model_path": str(model_path),
        "config_path": str(config_path),
        "gnn_data_path": str(gnn_data_path) if gnn_data_path else None,
        "training_time": elapsed,
        "episodes": episode_tracker.episode_count,
        "episode_rewards": episode_tracker.episode_rewards,
    }
