"""CSV/report helpers for evaluation runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def ensure_output_dirs(base_dir: Path) -> dict[str, Path]:
    plots_dir = base_dir / "plots"
    csv_dir = base_dir / "csv"
    reports_dir = base_dir / "reports"
    for path in (base_dir, plots_dir, csv_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "base": base_dir,
        "plots": plots_dir,
        "csv": csv_dir,
        "reports": reports_dir,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def write_text_report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
