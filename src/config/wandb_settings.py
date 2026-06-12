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


def login_wandb(*, relogin: bool = False) -> bool:
    """Authenticate with Weights & Biases (interactive in Colab if needed)."""
    import wandb

    return bool(wandb.login(relogin=relogin))


def verify_wandb_access(settings: WandbSettings | None = None) -> dict[str, Any]:
    """Log in if needed and confirm the configured entity/project are reachable."""
    import wandb

    settings = settings or expected_wandb_settings()
    login_wandb()

    api = wandb.Api()
    viewer = api.viewer
    if viewer is None:
        raise RuntimeError("wandb login succeeded but viewer is unavailable.")

    username = viewer.get("username") or viewer.get("id")
    teams = {team.get("name") for team in viewer.get("teams", {}).get("edges", [])}
    teams.update({username} if username else set())

    if settings.entity not in teams:
        raise RuntimeError(
            f"wandb entity '{settings.entity}' is not available for this account. "
            f"Available: {sorted(t for t in teams if t)}"
        )

    try:
        project = api.project(name=settings.project, entity=settings.entity)
        project_name = project.name
    except Exception as exc:  # noqa: BLE001 — wandb raises varied errors
        raise RuntimeError(
            f"wandb project '{settings.entity}/{settings.project}' is not accessible."
        ) from exc

    return {
        "entity": settings.entity,
        "project": project_name,
        "viewer": username,
        "project_path": f"{settings.entity}/{settings.project}",
    }
