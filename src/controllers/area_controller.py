"""Area-level GNN traffic pressure forecaster.

Supports two model architectures:
    1. **GCN** (Graph Convolutional Network) — static spatial model
    2. **STGCN** (Spatio-Temporal GCN) — GCN + GRU temporal model

Architecture (GCN):
    Input: X ∈ R^{N×8} (ward features including city cap)
    GCN Layer 1: ReLU(A_hat · X · W1)
    GCN Layer 2: Sigmoid(A_hat · H1 · W2)
    Output: P ∈ R^N (pressure per ward ∈ [0, 1])

Architecture (STGCN):
    Input: X_seq ∈ R^{T×N×8} (temporal sequence of ward features)
    Spatial: GCN per timestep → H_t ∈ R^{N×hidden}
    Temporal: GRU across time → final hidden state
    Output: Sigmoid(Linear(h_T)) → P ∈ R^N (pressure ∈ [0, 1])

Training:
    Supervised learning on (X, Y) pairs collected during ward RL training.
    X = current ward state features
    Y = actual congestion per ward 30 seconds later
    Loss = MSE
"""

from __future__ import annotations

import logging
from collections import deque
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

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:  # pragma: no cover
    HAS_TQDM = False

    def tqdm(iterable, **kwargs):  # type: ignore[return-type]
        return iterable

logger = logging.getLogger(__name__)

# Node feature dimensions
NODE_FEATURES = 7  # congestion, Δcongestion, queue, Δqueue, avg_speed, throughput, incident
HIDDEN_DIM = 32
STGCN_SEQ_LEN = 10  # Temporal window for STGCN (covers 5 min at 30s intervals)


# ======================================================================
# Model Architectures
# ======================================================================

class WardPressureGCN(nn.Module):
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


class SpatioTemporalGCN(nn.Module):
    """Spatio-Temporal GCN (T-GCN) combining GCN spatial convolution
    with GRU temporal recurrence for traffic pressure forecasting.

    Reference: IEEE 9688532 — Traffic Forecasting using Graph Convolution Network.
    """

    def __init__(
        self,
        in_features: int = NODE_FEATURES + 1,
        hidden_dim: int = HIDDEN_DIM,
        seq_len: int = STGCN_SEQ_LEN,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # Spatial graph convolution projection
        self.gcn_fc = nn.Linear(in_features, hidden_dim)

        # Temporal GRU cell (processes each node's time-series)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Output prediction head
        self.out_fc = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Spatio-Temporal GCN.

        Args:
            x_seq: Temporal feature sequence ``[T, N, F]`` or ``[B, T, N, F]``.
            a_hat: Normalised adjacency matrix ``[N, N]``.

        Returns:
            Predicted pressure per ward ``[N]`` or ``[B, N]`` ∈ [0, 1].
        """
        if x_seq.dim() == 3:
            # Single sample: [T, N, F] → add batch dim
            x_seq = x_seq.unsqueeze(0)

        B, T, N, Feat = x_seq.shape

        # 1. Spatial GCN: apply graph convolution per timestep
        # Reshape: [B*T, N, F]
        x_flat = x_seq.reshape(B * T, N, Feat)
        # GCN: ReLU(A_hat @ X @ W)
        spatial = F.relu(a_hat @ self.gcn_fc(x_flat))  # [B*T, N, H]

        # 2. Reshape for GRU: group by node → [B*N, T, H]
        spatial = spatial.view(B, T, N, self.hidden_dim)
        spatial = spatial.permute(0, 2, 1, 3).contiguous()  # [B, N, T, H]
        spatial = spatial.view(B * N, T, self.hidden_dim)    # [B*N, T, H]

        # 3. Temporal GRU
        gru_out, _ = self.gru(spatial)  # [B*N, T, H]
        final_state = gru_out[:, -1, :]  # [B*N, H]

        # 4. Predict pressure
        out = torch.sigmoid(self.out_fc(final_state))  # [B*N, 1]
        return out.view(B, N).squeeze(0)  # [N] for single, [B, N] for batch


# ======================================================================
# Area Forecaster (Unified Training / Inference API)
# ======================================================================

class AreaForecaster:
    """Area-level GNN forecaster with training and inference APIs.

    Supports both GCN (static) and STGCN (spatio-temporal) architectures.

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
        model_type: str = "gcn",
    ) -> None:
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for AreaForecaster")

        self.area_id = area_id
        self.topology = topology
        self.ward_ids = topology.get_area_wards(area_id)
        self.n_wards = len(self.ward_ids)
        self.model_type = model_type.lower()

        # Set up dynamic device selection (GPU-accelerated GNN)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("AreaForecaster [%s] using device: %s, model: %s", area_id, self.device, self.model_type)

        # Build normalised adjacency matrix and move to device
        adj_np = topology.get_adjacency_matrix(area_id)
        self.a_hat = torch.tensor(adj_np, dtype=torch.float32).to(self.device)

        # Initialise the selected model
        if self.model_type == "stgcn":
            self.model: nn.Module = SpatioTemporalGCN(
                in_features=NODE_FEATURES + 1,
                hidden_dim=HIDDEN_DIM,
                seq_len=STGCN_SEQ_LEN,
            ).to(self.device)
        else:
            self.model = WardPressureGCN(in_features=NODE_FEATURES + 1).to(self.device)
        self.model.eval()

        # Load pre-trained weights if available
        if model_dir is not None:
            self._load_model(model_dir)

        # State tracking for delta features
        self._prev_congestion = np.zeros(self.n_wards, dtype=np.float32)
        self._prev_queue = np.zeros(self.n_wards, dtype=np.float32)

        # STGCN temporal history buffer
        self._history_buffer: deque[np.ndarray] = deque(maxlen=STGCN_SEQ_LEN)

        # Ingest state tracking for runtime orchestration
        self._last_summaries: dict[str, dict[str, float]] = {}
        self._last_city_cap: float = 1.0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def ingest_tick(
        self,
        ward_summaries: dict[str, dict[str, float]],
        ward_actions: dict[str, str] | None = None,
        city_cap: float = 1.0,
    ) -> None:
        """Ingest tick-level ward summaries and metadata from the simulator."""
        self._last_summaries = ward_summaries
        self._last_city_cap = city_cap

        # Build and buffer features for STGCN temporal window
        features = self._build_features(ward_summaries, city_cap)
        self._history_buffer.append(features)

    def predict(
        self,
        ward_summaries: dict[str, dict[str, float]] | None = None,
        city_cap: float = 1.0,
        ingest: bool = False,
    ) -> dict[str, float]:
        """Run model prediction on current ward states.

        Args:
            ward_summaries: Dict mapping ward_id → metric summary.
            city_cap: City-level inflow capacity cap.
            ingest: Unused flag kept for backward compatibility.

        Returns:
            Dict mapping ward_id → predicted pressure ∈ [0, 1].
        """
        if ward_summaries is None:
            ward_summaries = self._last_summaries
            city_cap = self._last_city_cap

        if self.model_type == "stgcn":
            return self._predict_stgcn(ward_summaries, city_cap)
        else:
            return self._predict_gcn(ward_summaries, city_cap)

    def _predict_gcn(
        self,
        ward_summaries: dict[str, dict[str, float]],
        city_cap: float,
    ) -> dict[str, float]:
        """GCN prediction: single-frame spatial convolution."""
        x = self._build_features(ward_summaries, city_cap)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions = self.model(x_tensor, self.a_hat)

        return {wid: float(predictions[i].item()) for i, wid in enumerate(self.ward_ids)}

    def _predict_stgcn(
        self,
        ward_summaries: dict[str, dict[str, float]],
        city_cap: float,
    ) -> dict[str, float]:
        """STGCN prediction: temporal sequence through GCN + GRU."""
        # Build current frame and add to buffer
        current = self._build_features(ward_summaries, city_cap)
        if len(self._history_buffer) == 0:
            self._history_buffer.append(current)

        # Pad history if insufficient
        frames = list(self._history_buffer)
        while len(frames) < STGCN_SEQ_LEN:
            frames.insert(0, frames[0].copy())

        # Stack into [T, N, F]
        x_seq = np.stack(frames[-STGCN_SEQ_LEN:], axis=0)
        x_tensor = torch.tensor(x_seq, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions = self.model(x_tensor, self.a_hat)

        return {wid: float(predictions[i].item()) for i, wid in enumerate(self.ward_ids)}

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
    ) -> dict[str, Any]:
        """Train the selected model on collected simulation data.

        Returns:
            Dict with training metrics: losses, final_mse, epochs, model_type.
        """
        dataset = torch.load(dataset_path, weights_only=False)
        if not dataset:
            raise ValueError("Empty training dataset")

        if self.model_type == "stgcn":
            return self._train_stgcn(dataset, epochs, lr, save_dir)
        else:
            return self._train_gcn(dataset, epochs, lr, save_dir)

    def _preprocess_sample(self, sample: dict) -> tuple[np.ndarray, np.ndarray] | None:
        """Convert a raw dataset sample into (features [N, 8], target [N])."""
        if isinstance(sample["features"], list) and len(sample["features"]) > 0 and isinstance(sample["features"][0], dict):
            # Temporal trace from ward env
            trace = sample["features"][-1]
            trace_start = sample["features"][0]
            delta_congestion = trace.get("congestion_score", 0.0) - trace_start.get("congestion_score", 0.0)
            delta_queue = trace.get("queue_length", 0.0) - trace_start.get("queue_length", 0.0)
            raw_feats = np.array([
                trace.get("congestion_score", 0.0),
                delta_congestion,
                trace.get("queue_length", 0.0),
                delta_queue,
                trace.get("avg_speed", 0.0),
                trace.get("outflow", 0.0),
                trace.get("incident_flag", 0.0),
                trace.get("city_directive", 1.0),
            ], dtype=np.float32)
        else:
            raw_feats = np.asarray(sample["features"], dtype=np.float32)

        raw_target = np.asarray(sample["target"], dtype=np.float32)

        # Shape alignment to [N, 8]
        if raw_feats.ndim == 1:
            if raw_feats.shape[0] == 7:
                raw_feats = np.append(raw_feats, 1.0)
            raw_feats = raw_feats.reshape(1, 8)
        elif raw_feats.ndim == 2:
            if raw_feats.shape[1] == 7:
                caps = np.ones((raw_feats.shape[0], 1), dtype=np.float32)
                raw_feats = np.hstack((raw_feats, caps))
        else:
            return None

        # Target alignment
        n_nodes = raw_feats.shape[0]
        if raw_target.ndim == 0 or raw_target.size == 1:
            raw_target = np.full(n_nodes, float(raw_target), dtype=np.float32)
        elif raw_target.ndim == 1 and raw_target.shape[0] != n_nodes:
            raw_target = np.resize(raw_target, n_nodes)

        return raw_feats, raw_target

    def _train_gcn(
        self,
        dataset: list,
        epochs: int,
        lr: float,
        save_dir: Path | None,
    ) -> dict[str, Any]:
        """Train GCN model on the dataset."""
        logger.info("Pre-processing %d samples for GCN training...", len(dataset))
        processed: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for sample in tqdm(dataset, desc=f"Preprocessing [{self.area_id}]", leave=False):
            result = self._preprocess_sample(sample)
            if result is None:
                continue
            raw_feats, raw_target = result
            n_nodes = raw_feats.shape[0]

            x = torch.tensor(raw_feats, dtype=torch.float32).to(self.device)
            y = torch.tensor(raw_target, dtype=torch.float32).to(self.device)

            if n_nodes == self.n_wards:
                a_hat_used = self.a_hat
            else:
                a_hat_used = torch.eye(n_nodes, dtype=torch.float32).to(self.device)

            processed.append((x, y, a_hat_used))

        logger.info("Training GCN on %d samples for %d epochs...", len(processed), epochs)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        losses: list[float] = []

        epoch_iter = tqdm(range(epochs), desc=f"GCN Training [{self.area_id}]")
        for epoch in epoch_iter:
            epoch_loss = 0.0
            for x, y, a_hat_used in processed:
                pred = self.model(x, a_hat_used)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(processed)
            losses.append(avg_loss)
            epoch_iter.set_postfix(mse=f"{avg_loss:.6f}")

        self.model.eval()
        if save_dir is not None:
            self._save_model(save_dir)

        return {
            "losses": losses,
            "final_mse": losses[-1],
            "epochs": epochs,
            "model_type": "gcn",
            "area_id": self.area_id,
            "samples": len(processed),
        }

    def _train_stgcn(
        self,
        dataset: list,
        epochs: int,
        lr: float,
        save_dir: Path | None,
    ) -> dict[str, Any]:
        """Train STGCN model on the dataset with temporal windowing."""
        logger.info("Pre-processing %d samples for STGCN temporal windowing...", len(dataset))

        # First pass: extract all (features, target) pairs
        all_samples: list[tuple[np.ndarray, np.ndarray]] = []
        for sample in tqdm(dataset, desc=f"Preprocessing [{self.area_id}]", leave=False):
            result = self._preprocess_sample(sample)
            if result is not None:
                all_samples.append(result)

        # Build temporal sequences: sliding window of seq_len consecutive samples
        seq_len = STGCN_SEQ_LEN
        sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for i in tqdm(range(len(all_samples) - seq_len), desc=f"Building sequences [{self.area_id}]", leave=False):
            window = all_samples[i:i + seq_len]
            target_feats, target_y = all_samples[i + seq_len - 1]

            # All samples in the window must have the same node count
            n_nodes = window[0][0].shape[0]
            if not all(w[0].shape[0] == n_nodes for w in window):
                continue

            # Stack into [T, N, F]
            x_seq = np.stack([w[0] for w in window], axis=0)
            x_tensor = torch.tensor(x_seq, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(target_y, dtype=torch.float32).to(self.device)

            if n_nodes == self.n_wards:
                a_hat_used = self.a_hat
            else:
                a_hat_used = torch.eye(n_nodes, dtype=torch.float32).to(self.device)

            sequences.append((x_tensor, y_tensor, a_hat_used))

        if not sequences:
            logger.warning("No valid temporal sequences built for STGCN. Need at least %d consecutive samples.", seq_len + 1)
            return {"losses": [], "final_mse": float("inf"), "epochs": 0, "model_type": "stgcn", "area_id": self.area_id, "samples": 0}

        logger.info("Training STGCN on %d sequences for %d epochs...", len(sequences), epochs)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        losses: list[float] = []

        epoch_iter = tqdm(range(epochs), desc=f"STGCN Training [{self.area_id}]")
        for epoch in epoch_iter:
            epoch_loss = 0.0
            for x_seq, y, a_hat_used in sequences:
                pred = self.model(x_seq, a_hat_used)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(sequences)
            losses.append(avg_loss)
            epoch_iter.set_postfix(mse=f"{avg_loss:.6f}")

        self.model.eval()
        if save_dir is not None:
            self._save_model(save_dir)

        return {
            "losses": losses,
            "final_mse": losses[-1],
            "epochs": epochs,
            "model_type": "stgcn",
            "area_id": self.area_id,
            "samples": len(sequences),
        }

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def _save_model(self, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        suffix = "stgcn" if self.model_type == "stgcn" else "gcn"
        model_path = save_dir / f"area_model_{suffix}.pt"
        torch.save(self.model.state_dict(), model_path)
        logger.info("Area %s model saved → %s", suffix.upper(), model_path)

        # Also save as area_model.pt for backward compatibility (GCN only)
        if self.model_type == "gcn":
            compat_path = save_dir / "area_model.pt"
            torch.save(self.model.state_dict(), compat_path)

    def _load_model(self, model_dir: Path) -> None:
        suffix = "stgcn" if self.model_type == "stgcn" else "gcn"
        model_path = model_dir / f"area_model_{suffix}.pt"

        # Fallback to legacy path for GCN
        if not model_path.exists() and self.model_type == "gcn":
            model_path = model_dir / "area_model.pt"

        if model_path.exists():
            self.model.load_state_dict(
                torch.load(model_path, weights_only=True, map_location=self.device)
            )
            self.model.eval()
            logger.info("Area %s model loaded ← %s", suffix.upper(), model_path)

    def reset(self) -> None:
        """Reset delta tracking and history buffer between episodes."""
        self._prev_congestion[:] = 0.0
        self._prev_queue[:] = 0.0
        self._history_buffer.clear()
