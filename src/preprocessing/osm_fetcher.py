"""Automated OSM data fetcher using the Overpass API with QL queries.

Downloads ward-level OpenStreetMap data by querying the actual BBMP
administrative ward boundary relation (admin_level=10) rather than
relying on hand-drawn bounding boxes.

Overpass QL query strategy (mirrors the Overpass Turbo approach):

    [out:xml][timeout:120];
    area["name"="Bengaluru"]->.blr;
    (
      relation(area.blr)
        ["boundary"="administrative"]
        ["admin_level"="10"]
        ["ward"="<N>"];
    )->.ward;
    (
      way(r.ward)["highway"];
    );
    (._;>;);
    out meta;

Ward numbers are extracted from the ward_id key (e.g. ``"ward_070"`` → 70).
A ``bbox`` fallback is used only when the registry entry explicitly sets
``"fetch_strategy": "bbox"`` or when the Overpass QL query fails.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OVERPASS_INTERPRETER_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_MAP_URL = "https://overpass-api.de/api/map"
REQUEST_DELAY_SECONDS = 2  # polite rate-limiting between wards
_USER_AGENT = "HMRL-Traffic/1.0 (Hierarchical Multi-Agent RL; contact: research)"


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def load_ward_registry(project_root: Path) -> dict[str, Any]:
    """Load the ward registry JSON."""
    path = project_root / "configs" / "hierarchy" / "ward_registry.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _ward_number_from_id(ward_id: str) -> int:
    """Extract the integer ward number from a ward_id string.

    Examples::

        "ward_001" → 1
        "ward_070" → 70
        "ward_100" → 100
    """
    numeric = ward_id.split("_")[-1]
    return int(numeric)


# ---------------------------------------------------------------------------
# Overpass QL query builder
# ---------------------------------------------------------------------------

def build_ward_overpass_query(ward_number: int, city: str = "Bengaluru") -> str:
    """Build a typed Overpass QL query for a BBMP ward boundary.

    The query follows the pattern proven in Overpass Turbo:

    1. Resolve the named city area.
    2. Find the administrative relation at admin_level=10 tagged with the
       ward number.
    3. Collect all highway ways that belong to that relation.
    4. Recurse down to include all member nodes.
    5. Output full metadata (way geometry, node coords, tags).

    Args:
        ward_number: Integer BBMP ward number (e.g. 70 for HSR Layout Ward 70).
        city: City area name as stored in OSM (default ``"Bengaluru"``).

    Returns:
        Overpass QL query string ready to POST to the interpreter.
    """
    return (
        f'[out:xml][timeout:120];\n'
        f'area["name"="{city}"]->.blr;\n'
        f'(\n'
        f'  relation(area.blr)\n'
        f'    ["boundary"="administrative"]\n'
        f'    ["admin_level"="10"]\n'
        f'    ["ward"="{ward_number}"];\n'
        f');\n'
        f'map_to_area->.ward_area;\n'
        f'(\n'
        f'  way(area.ward_area)["highway"];\n'
        f');\n'
        f'(._;>;);\n'
        f'out meta;\n'
    )


def build_relation_overpass_query(relation_id: int) -> str:
    """Build a precise Overpass QL query using the exact relation ID."""
    return (
        f'[out:xml][timeout:120];\n'
        f'(\n'
        f'  relation({relation_id});\n'
        f');\n'
        f'map_to_area->.ward_area;\n'
        f'(\n'
        f'  way(area.ward_area)["highway"];\n'
        f');\n'
        f'(._;>;);\n'
        f'out meta;\n'
    )



def build_ward_bbox_query(bbox: dict[str, float]) -> str:
    """Build a simple bounding-box Overpass QL query as a fallback.

    Uses the interpreter endpoint (not the deprecated /api/map endpoint)
    so the same HTTP POST machinery works for both strategies.

    Args:
        bbox: Dictionary with keys ``south``, ``west``, ``north``, ``east``.

    Returns:
        Overpass QL query string.
    """
    s, w, n, e = bbox["south"], bbox["west"], bbox["north"], bbox["east"]
    return (
        f'[out:xml][timeout:120];\n'
        f'(\n'
        f'  way["highway"]({s},{w},{n},{e});\n'
        f');\n'
        f'(._;>;);\n'
        f'out meta;\n'
    )


# ---------------------------------------------------------------------------
# HTTP fetch helpers
# ---------------------------------------------------------------------------

def _post_overpass(query: str) -> bytes:
    """POST a query to the Overpass interpreter and return raw bytes.

    Args:
        query: Overpass QL query string.

    Returns:
        Raw response bytes (OSM XML).

    Raises:
        urllib.error.URLError: On network or HTTP error.
    """
    encoded = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(
        OVERPASS_INTERPRETER_URL,
        data=encoded,
        method="POST",
        headers={
            "User-Agent": _USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read()


# ---------------------------------------------------------------------------
# Public fetch API
# ---------------------------------------------------------------------------

def fetch_ward_osm(
    ward_id: str,
    output_dir: Path,
    ward_number: int | None = None,
    osm_relation_id: int | None = None,
    bbox: dict[str, float] | None = None,
    city: str = "Bengaluru",
    force: bool = False,
) -> Path:
    """Download OSM XML for a single ward via an Overpass QL query.

    Strategy:
        1. If osm_relation_id is provided, fetch exact relation.
        2. Else build a ward-boundary query using the BBMP admin relation
           (``admin_level=10``, ``ward=<N>``).
        3. If no geometry is returned, fall back to a bounding-box query.

    Args:
        ward_id: Ward identifier (e.g. ``"ward_070"``).
        output_dir: Directory to save the ``.osm`` file (``maps/raw_osm/``).
        ward_number: Explicit BBMP ward number. Derived from ``ward_id``
            automatically if not provided.
        osm_relation_id: Explicit OSM relation ID.
        bbox: Bounding-box dict with ``south/west/north/east`` keys used as fallback.
        city: OSM area name for the city (default ``"Bengaluru"``).
        force: If *True*, re-download even if the file already exists.

    Returns:
        Path to the downloaded ``.osm`` file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    osm_path = output_dir / f"{ward_id}.osm"

    if osm_path.exists() and not force:
        logger.info("OSM file already exists, skipping: %s", osm_path)
        return osm_path

    if ward_number is None:
        ward_number = _ward_number_from_id(ward_id)

    # --- Attempt 1: precise relation or ward-boundary Overpass QL query -------------------------
    if osm_relation_id is not None:
        query = build_relation_overpass_query(osm_relation_id)
        logger.info(
            "Downloading OSM for %s via exact relation ID %d → %s",
            ward_id, osm_relation_id, OVERPASS_INTERPRETER_URL,
        )
    else:
        query = build_ward_overpass_query(ward_number, city=city)
        logger.info(
            "Downloading OSM for %s (ward=%d) via boundary query → %s",
            ward_id, ward_number, OVERPASS_INTERPRETER_URL,
        )
    logger.debug("Overpass QL query:\n%s", query)

    try:
        data = _post_overpass(query)
        if _osm_bytes_are_empty(data):
            logger.warning(
                "Ward boundary query returned no ways for ward_number=%d. "
                "The OSM relation may be unmapped. Trying fallback...",
                ward_number,
            )
            data = _try_bbox_fallback(ward_id, bbox)
        else:
            logger.info(
                "Ward boundary query succeeded for %s (%d bytes)",
                ward_id, len(data),
            )
    except Exception as exc:
        logger.warning(
            "Ward boundary query failed for %s: %s — trying bbox fallback",
            ward_id, exc,
        )
        data = _try_bbox_fallback(ward_id, bbox)

    osm_path.write_bytes(data)
    logger.info("Saved %d bytes → %s", len(data), osm_path)
    return osm_path


def _try_bbox_fallback(
    ward_id: str,
    bbox: dict[str, float] | None,
) -> bytes:
    """Run a bounding-box Overpass QL query as a fallback.

    Args:
        ward_id: Ward identifier (for logging).
        bbox: Bounding-box dict; if ``None``, raises ``RuntimeError``.

    Returns:
        Raw OSM XML bytes.

    Raises:
        RuntimeError: If no ``bbox`` is available.
    """
    if bbox is None:
        raise RuntimeError(
            f"Ward boundary query returned no data for {ward_id} and no "
            "bbox fallback is configured in ward_registry.json."
        )
    logger.info(
        "Using bbox fallback for %s: %s",
        ward_id, bbox,
    )
    query = build_ward_bbox_query(bbox)
    return _post_overpass(query)


def _osm_bytes_are_empty(data: bytes) -> bool:
    """Return True if the OSM XML response contains no way elements.

    The full response is checked because way elements appear after the
    node block and may not be within the first few kilobytes.
    """
    return b"<way" not in data


# ---------------------------------------------------------------------------
# Batch fetch
# ---------------------------------------------------------------------------

def fetch_all_wards(
    project_root: Path,
    force: bool = False,
    city: str = "Bengaluru",
) -> list[dict[str, Any]]:
    """Download OSM data for every ward in the registry.

    Each ward is downloaded using its BBMP administrative boundary relation
    query first. If that returns no road geometry, the ``bbox`` field in the
    registry entry is used as a fallback (when present).

    Args:
        project_root: Project root directory.
        force: Re-download even if the file already exists.
        city: OSM city area name (default ``"Bengaluru"``).

    Returns:
        List of result dicts with ``ward_id``, ``path``, ``status``,
        and ``strategy`` (``"boundary_query"`` or ``"bbox_fallback"``).
    """
    registry = load_ward_registry(project_root)
    output_dir = project_root / "maps" / "raw_osm"
    results: list[dict[str, Any]] = []

    for ward_id, meta in registry["wards"].items():
        bbox = meta.get("bbox")  # optional fallback
        ward_number = meta.get("ward_number")  # optional explicit override
        osm_relation_id = meta.get("osm_relation_id")  # explicit mapping

        try:
            path = fetch_ward_osm(
                ward_id,
                output_dir,
                ward_number=ward_number,
                osm_relation_id=osm_relation_id,
                bbox=bbox,
                city=city,
                force=force,
            )
            results.append({
                "ward_id": ward_id,
                "path": str(path),
                "status": "ok",
            })
        except Exception as exc:
            logger.error("Failed to download OSM for %s: %s", ward_id, exc)
            results.append({
                "ward_id": ward_id,
                "path": None,
                "status": f"error: {exc}",
            })

        time.sleep(REQUEST_DELAY_SECONDS)

    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_osm_file(osm_path: Path) -> bool:
    """Basic validity check: file exists and contains OSM XML with road data.

    Checks for both the ``<osm`` root element and at least one ``<way``
    element (a file with only nodes has no usable road geometry). The
    full file is scanned because way elements appear after the nodes
    block and may not be present in the first few kilobytes.
    """
    if not osm_path.exists():
        return False
    content = osm_path.read_bytes()
    return b"<osm" in content and b"<way" in content
