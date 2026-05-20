"""Optional temporal dataset export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


def export_temporal_dataset(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, str | None]:
    """Export flattened ward-tick records for future benchmarking."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for result in results:
        for record in result.get("ward_tick_records", []):
            enriched = dict(record)
            enriched.update({
                "use_rl": result["normalized_metrics"]["use_rl"],
                "use_area": result["normalized_metrics"]["use_area"],
                "use_city": result["normalized_metrics"]["use_city"],
                "algorithm": result["normalized_metrics"]["algorithm"],
            })
            records.append(enriched)

    json_path = output_dir / "temporal_dataset.json"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    torch_path: Path | None = None
    if HAS_TORCH:
        torch_path = output_dir / "temporal_dataset.pt"
        torch.save(records, torch_path)

    return {
        "json_path": str(json_path),
        "torch_path": str(torch_path) if torch_path is not None else None,
    }
