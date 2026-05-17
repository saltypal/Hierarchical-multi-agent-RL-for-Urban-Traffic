"""Area-level GNN traffic pressure forecaster.

A lightweight 2-layer Graph Convolutional Network (GCN) that predicts
future traffic pressure per ward within an area. The predictions are
used to shape ward RL observations and rewards.

Architecture:
    Input: X ∈ R^{N×7} (ward features) + c (city cap scalar)
    GCN Layer 1: ReLU(A_hat · W1 · X)
    GCN Layer 2: Sigmoid(A_hat · W2 · H1)
    Output: P ∈ R^N (pressure per ward ∈ [0, 1])

Training:
    Supervised learning on (X, Y) pairs collected during ward RL training.
    X = current ward state features
    Y = actual congestion per ward 30 seconds later
    Loss = MSE
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

logger = logging.getLogger(__name__)

# Node feature dimensions
NODE_FEATURES = 7  # congestion, Δcongestion, queue, Δqueue, avg_speed, throughput, incident
HIDDEN_DIM = 32


class WardPressureGNN(nn.Module):
    """2-layer Graph Convolutional Network for pressure prediction.

    Pure PyTorch — no dependency on ``torch_geometric``.
    """

    def __init__(
        self,
        in_features: int = NODE_FEATURES + 1,  # +1 for city cap
        hidden: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GCN.

        Args:
            x: Node feature matrix ``[N_wards, in_features]``.
            a_hat: Normalised adjacency matrix ``[N_wards, N_wards]``.

        Returns:
            Predicted pressure per ward ``[N_wards]`` ∈ [0, 1].
        """
        h = F.relu(a_hat @ self.fc1(x))
        out = torch.sigmoid(a_hat @ self.fc2(h))
        return out.squeeze(-1)


class AreaForecaster:
    """Area-level GNN forecaster with training and inference APIs.

    Operates every 30 simulation seconds:
        1. Collect ward summaries → node features
        2. Run GNN → predicted pressure per ward
        3. Pass predictions down to ward agents (obs + reward shaping)
    """

    def __init__(
        self,
        area_id: str,
        topology: Any,
        model_dir: Path | None = None,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for AreaForecaster")

        self.area_id = area_id
        self.topology = topology
        self.ward_ids = topology.get_area_wards(area_id)
        self.n_wards = len(self.ward_ids)

        # Set up dynamic device selection (GPU-accelerated GNN)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("AreaForecaster [%s] using device: %s", area_id, self.device)

        # Build normalised adjacency matrix and move to device
        adj_np = topology.get_adjacency_matrix(area_id)
        self.a_hat = torch.tensor(adj_np, dtype=torch.float32).to(self.device)

        # Initialise model and move to device
        self.model = WardPressureGNN(in_features=NODE_FEATURES + 1).to(self.device)
        self.model.eval()

        # Load pre-trained weights if available
        if model_dir is not None:
            self._load_model(model_dir)

        # State tracking for delta features
        self._prev_congestion = np.zeros(self.n_wards, dtype=np.float32)
        self._prev_queue = np.zeros(self.n_wards, dtype=np.float32)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        ward_summaries: dict[str, dict[str, float]],
        city_cap: float = 1.0,
    ) -> dict[str, float]:
        """Run GNN prediction on current ward states.

        Args:
            ward_summaries: Dict mapping ward_id → metric summary from SumoEnv.
            city_cap: City-level inflow capacity cap for this area.

        Returns:
            Dict mapping ward_id → predicted pressure ∈ [0, 1].
        """
        x = self._build_features(ward_summaries, city_cap)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions = self.model(x_tensor, self.a_hat)

        result = {}
        for i, wid in enumerate(self.ward_ids):
            result[wid] = float(predictions[i].item())

        return result

    def _build_features(
        self,
        ward_summaries: dict[str, dict[str, float]],
        city_cap: float,
    ) -> np.ndarray:
        """Build GNN node feature matrix from ward summaries.

        Features per node (8 dims):
            congestion, Δcongestion, queue, Δqueue,
            avg_speed, throughput, incident_flag, city_cap
        """
        x = np.zeros((self.n_wards, NODE_FEATURES + 1), dtype=np.float32)

        for i, wid in enumerate(self.ward_ids):
            summary = ward_summaries.get(wid, {})
            congestion = summary.get("congestion", 0.0)
            queue = summary.get("queue", 0.0)

            x[i, 0] = congestion
            x[i, 1] = congestion - self._prev_congestion[i]
            x[i, 2] = queue
            x[i, 3] = queue - self._prev_queue[i]
            x[i, 4] = summary.get("avg_speed", 0.0)
            x[i, 5] = summary.get("throughput", 0.0)
            x[i, 6] = summary.get("incident_flag", 0.0)
            x[i, 7] = city_cap

            self._prev_congestion[i] = congestion
            self._prev_queue[i] = queue

        return x

    # ------------------------------------------------------------------
    # Offline training
    # ------------------------------------------------------------------

    def train_offline(
        self,
        dataset_path: Path,
        epochs: int = 100,
        lr: float = 1e-3,
        save_dir: Path | None = None,
    ) -> list[float]:
        """Train the GNN on collected simulation data.

        Dataset format (list of dicts):
            ``[{"features": np.array[N,8], "target": np.array[N]}, ...]``

        Args:
            dataset_path: Path to ``.pt`` dataset file.
            epochs: Training epochs.
            lr: Learning rate.
            save_dir: Where to save the trained model.

        Returns:
            List of per-epoch losses.
        """
        dataset = torch.load(dataset_path, weights_only=False)
        if not dataset:
            raise ValueError("Empty training dataset")

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        losses: list[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for sample in dataset:
                x = torch.tensor(sample["features"], dtype=torch.float32).to(self.device)
                y = torch.tensor(sample["target"], dtype=torch.float32).to(self.device)

                pred = self.model(x, self.a_hat)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataset)
            losses.append(avg_loss)

            if (epoch + 1) % 20 == 0:
                logger.info("Epoch %d/%d — loss: %.6f", epoch + 1, epochs, avg_loss)

        self.model.eval()

        if save_dir is not None:
            self._save_model(save_dir)

        return losses

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def _save_model(self, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / "area_model.pt"
        torch.save(self.model.state_dict(), model_path)
        logger.info("GNN model saved → %s", model_path)

    def _load_model(self, model_dir: Path) -> None:
        model_path = model_dir / "area_model.pt"
        if model_path.exists():
            self.model.load_state_dict(
                torch.load(model_path, weights_only=True)
            )
            self.model.eval()
            logger.info("GNN model loaded ← %s", model_path)

    def reset(self) -> None:
        """Reset delta tracking between episodes."""
        self._prev_congestion[:] = 0.0
        self._prev_queue[:] = 0.0
