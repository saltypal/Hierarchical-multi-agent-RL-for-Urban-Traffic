"""Real-time simulation dashboard server.

A lightweight Flask server that exposes simulation state as JSON via REST
endpoints, consumed by the HTML/JS dashboard in ``dashboard/index.html``.

Run alongside the simulation:
    python dashboard/server.py

Then open: http://localhost:5050
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

try:
    from flask import Flask, jsonify, send_from_directory
    from flask_cors import CORS
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

app = Flask(__name__, static_folder=".")
if HAS_FLASK:
    CORS(app)

# Shared state updated by the simulation runtime
_state_lock = threading.Lock()
_simulation_state: dict[str, Any] = {
    "tick": 0,
    "elapsed": 0.0,
    "total_arrived": 0,
    "scenario": "normal",
    "scope": "ward",
    "identifier": "ward_001",
    "city_caps": {},
    "area_predictions": {},
    "ward_states": {},
    "ward_actions": {},
    "ward_rewards": {},
    "metrics": {
        "avg_speed": 0.0,
        "total_vehicles": 0,
        "total_queue": 0,
        "throughput": 0,
    },
}


def update_state(new_state: dict[str, Any]) -> None:
    """Thread-safe update of simulation state (called by runtime)."""
    with _state_lock:
        _simulation_state.update(new_state)


def get_state() -> dict[str, Any]:
    """Thread-safe read of simulation state."""
    with _state_lock:
        return dict(_simulation_state)


# ------------------------------------------------------------------
# API Routes
# ------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(str(Path(__file__).parent), "index.html")

@app.route("/api/state")
def api_state():
    return jsonify(get_state())

@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "tick": get_state()["tick"]})


def run_server(host: str = "0.0.0.0", port: int = 5050, debug: bool = False) -> None:
    """Start the dashboard server in a background thread."""
    if not HAS_FLASK:
        print("[dashboard] Flask not installed. Run: pip install flask flask-cors")
        return
    print(f"[dashboard] Starting server at http://localhost:{port}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)


def start_background(port: int = 5050) -> threading.Thread:
    """Launch dashboard server as a daemon thread."""
    t = threading.Thread(target=run_server, kwargs={"port": port}, daemon=True)
    t.start()
    return t
