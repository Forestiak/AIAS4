"""
Model registry.
Each model is stored as a JSON file somewhere under the models/ folder.
"""
from __future__ import annotations

import json
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _normalize_folder(folder: str) -> str:
    raw = folder.replace("\\", "/").strip().strip("/")
    if not raw:
        return ""

    parts = [part.strip() for part in raw.split("/") if part.strip()]
    if any(part in {".", ".."} for part in parts):
        raise ValueError("Folder paths cannot contain '.' or '..'.")
    return "/".join(parts)


def _iter_model_files() -> list[Path]:
    if not MODELS_DIR.exists():
        return []
    return sorted(
        MODELS_DIR.rglob("*.json"),
        key=lambda path: path.relative_to(MODELS_DIR).as_posix().lower(),
    )


def _find_model_paths(model_id: str) -> list[Path]:
    if not MODELS_DIR.exists():
        return []
    return sorted(MODELS_DIR.rglob(f"{model_id}.json"))


def _path_from_identifier(identifier: str) -> Path | None:
    """Resolve an identifier that may be a relative models/ path like best/foo.json."""
    normalized = identifier.replace("\\", "/").strip().strip("/")
    if not normalized.lower().endswith(".json"):
        return None

    candidate = (MODELS_DIR / normalized).resolve()
    try:
        candidate.relative_to(MODELS_DIR.resolve())
    except ValueError:
        return None

    if not candidate.exists() or not candidate.is_file():
        return None
    return candidate


def list_models() -> list[dict]:
    """Return all model definitions sorted by relative path."""
    return [json.loads(path.read_text()) for path in _iter_model_files()]


def load_model(model_id: str) -> dict:
    """Load a model definition by ID or relative path under models/."""
    path = _path_from_identifier(model_id)
    if path is not None:
        return json.loads(path.read_text())

    matches = _find_model_paths(model_id)
    if not matches:
        raise FileNotFoundError(f"Model '{model_id}' not found under {MODELS_DIR}")
    if len(matches) > 1:
        matches = sorted(
            matches,
            key=lambda p: (
                0 if p.relative_to(MODELS_DIR).as_posix().startswith("best/") else 1,
                p.relative_to(MODELS_DIR).as_posix().lower(),
            ),
        )
    return json.loads(matches[0].read_text())


def save_model(definition: dict, folder: str = "") -> Path:
    """Save a model definition to the models/ folder or one of its subfolders."""
    model_id = definition["model_id"]
    folder = _normalize_folder(folder)
    target_dir = MODELS_DIR / folder if folder else MODELS_DIR
    target_path = target_dir / f"{model_id}.json"

    existing_paths = _find_model_paths(model_id)
    if existing_paths and target_path not in existing_paths:
        existing = existing_paths[0].relative_to(MODELS_DIR).as_posix()
        raise FileExistsError(f"Model ID '{model_id}' already exists at {existing}")

    target_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(definition)
    if folder:
        payload["model_folder"] = folder
    else:
        payload.pop("model_folder", None)
    target_path.write_text(json.dumps(payload, indent=2))
    return target_path
