"""
Feature-set registry.
Each feature set is a JSON file in the feature_sets/ folder.
"""
from __future__ import annotations

import json
from pathlib import Path

FEATURE_SETS_DIR = Path(__file__).resolve().parent.parent / "feature_sets"


def list_feature_sets() -> list[dict]:
    """Return all feature-set definitions sorted by feature_set_id."""
    sets: list[dict] = []
    if FEATURE_SETS_DIR.exists():
        for f in sorted(FEATURE_SETS_DIR.glob("*.json")):
            sets.append(load_feature_set(f.stem))
    return sets


def load_feature_set(feature_set_id: str) -> dict:
    """Load a single feature-set definition by its ID (filename without .json)."""
    path = FEATURE_SETS_DIR / f"{feature_set_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Feature set '{feature_set_id}' not found at {path}")
    return json.loads(path.read_text())


def save_feature_set(definition: dict) -> Path:
    """Save a feature-set definition to the feature_sets/ folder."""
    fsid = definition["feature_set_id"]
    FEATURE_SETS_DIR.mkdir(parents=True, exist_ok=True)
    path = FEATURE_SETS_DIR / f"{fsid}.json"
    path.write_text(json.dumps(definition, indent=2))
    return path


def delete_feature_set(feature_set_id: str) -> None:
    """Delete a feature-set definition."""
    path = FEATURE_SETS_DIR / f"{feature_set_id}.json"
    if path.exists():
        path.unlink()
