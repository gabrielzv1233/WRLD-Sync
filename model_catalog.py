from __future__ import annotations
from functools import lru_cache
import pathlib
import json

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
CATALOG_PATH = PROJECT_ROOT / "models_manifest.json"

@lru_cache(maxsize=1)
def load_catalog() -> dict:
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data.get("models"), list):
        raise RuntimeError("models_manifest.json must contain a models array")
    ids = [str(x.get("id", "")) for x in data["models"]]
    if not all(ids) or len(ids) != len(set(ids)):
        raise RuntimeError("models_manifest.json contains missing or duplicate model ids")
    return data

@lru_cache(maxsize=1)
def model_map() -> dict[str, dict]:
    return {str(item["id"]): item for item in load_catalog()["models"]}

def get_catalog_model(model_id: str) -> dict:
    try:
        return model_map()[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown catalog model: {model_id}") from exc

def models_for_task(task: str, *, include_advanced: bool = True) -> list[dict]:
    out = []
    for item in load_catalog()["models"]:
        if task not in item.get("tasks", []):
            continue
        if not include_advanced and item.get("advanced_only"):
            continue
        out.append(item)
    return out
