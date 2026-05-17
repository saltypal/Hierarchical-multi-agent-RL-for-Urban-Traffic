"""Training helper for Stable-Baselines3 PPO on the SUMO ward adapter."""

from __future__ import annotations

from pathlib import Path

from .sb3_ward_adapter import StableBaselinesWardEnv, WardAdapterConfig


def train_ward_ppo(
    sumo_config_path: str,
    model_output_path: str,
    total_timesteps: int = 10000,
    gui: bool = False,
) -> None:
    """Train PPO if Stable-Baselines3 is installed."""

    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "stable_baselines3 is not installed. Install it before PPO training. "
            "The existing DQN loop can still run without this dependency."
        ) from exc

    env = StableBaselinesWardEnv(
        WardAdapterConfig(
            sumo_config_path=sumo_config_path,
            gui=gui,
        )
    )
    try:
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=total_timesteps)
        output_path = Path(model_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output_path))
    finally:
        env.close()
