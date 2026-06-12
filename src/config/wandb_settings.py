"""Team Weights & Biases defaults for Colab and local training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

WANDB_ENTITY = "project-nn"
WANDB_PROJECT = "adm-project-tgnn"


@dataclass(frozen=True)
class WandbSettings:
    entity: str = WANDB_ENTITY
    project: str = WANDB_PROJECT


def expected_wandb_settings() -> WandbSettings:
    return WandbSettings()


def _read_field(obj: Any, *names: str) -> str | None:
    """Read a field from a wandb API object or dict."""
    for name in names:
        if isinstance(obj, dict):
            value = obj.get(name)
        else:
            value = getattr(obj, name, None)
        if value:
            return str(value)
    return None


def _collect_viewer_entities(viewer: Any) -> set[str]:
    """Best-effort list of entity names visible to the logged-in user."""
    entities: set[str] = set()

    for key in ("username", "entity", "id"):
        value = _read_field(viewer, key)
        if value:
            entities.add(value)

    teams_data = viewer.get("teams") if isinstance(viewer, dict) else getattr(viewer, "teams", None)
    if teams_data is None:
        return entities

    edges: list[Any]
    if isinstance(teams_data, dict):
        edges = teams_data.get("edges", [])
    elif isinstance(teams_data, (list, tuple)):
        edges = list(teams_data)
    else:
        edges = []

    for edge in edges:
        node = edge.get("node", edge) if isinstance(edge, dict) else edge
        name = _read_field(node, "name")
        if name:
            entities.add(name)

    return entities


def login_wandb(*, relogin: bool = False) -> bool:
    """Authenticate with Weights & Biases (interactive in Colab if needed)."""
    import wandb

    return bool(wandb.login(relogin=relogin))


def verify_wandb_access(settings: WandbSettings | None = None) -> dict[str, Any]:
    """Log in if needed and confirm the configured entity/project are reachable."""
    settings = settings or expected_wandb_settings()
    login_wandb()

    api = __import__("wandb").Api()
    viewer = api.viewer
    if viewer is None:
        raise RuntimeError("wandb login succeeded but viewer is unavailable.")

    username = _read_field(viewer, "username", "id")
    available_entities = _collect_viewer_entities(viewer)

    try:
        project = api.project(name=settings.project, entity=settings.entity)
        project_name = project.name
    except Exception as exc:  # noqa: BLE001 — wandb raises varied errors
        raise RuntimeError(
            f"Cannot access wandb project '{settings.entity}/{settings.project}'. "
            f"Logged in as {username or 'unknown'}. "
            f"Visible entities: {sorted(available_entities)}. "
            "Ask a team admin to invite you to the entity or create the project."
        ) from exc

    return {
        "entity": settings.entity,
        "project": project_name,
        "viewer": username,
        "project_path": f"{settings.entity}/{settings.project}",
        "available_entities": sorted(available_entities),
    }
