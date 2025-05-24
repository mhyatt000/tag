"""Helper utilities for the chase environment."""

from __future__ import annotations

from typing import Optional

import genesis as gs

from tag.gym.robots.go2 import Go2Config
from tag.gym.robots.multi import MultiRobot


def create_robots(scene: gs.Scene, cfg: Go2Config) -> MultiRobot:
    """Instantiate the pair of Go2 robots for the chase task."""
    return MultiRobot(scene, cfg, ["r1", "r2"])


def create_camera(scene: gs.Scene, enabled: bool) -> Optional[gs.Camera]:
    """Add a default camera if ``enabled`` is ``True``."""
    if not enabled:
        return None
    return scene.add_camera(
        res=(1280, 720),
        pos=(7, 0.0, 2.5),
        lookat=(0, 0, 0.5),
        fov=60,
        GUI=False,
    )
