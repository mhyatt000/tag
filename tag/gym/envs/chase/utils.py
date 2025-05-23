"""Helper utilities for the chase environment."""

from __future__ import annotations

from typing import Optional

import genesis as gs

from .chase_config import ChaseEnvConfig
from tag.gym.robots.multi import MultiRobot
from tag.gym.robots.go2 import Go2Config


def create_scene(cfg: ChaseEnvConfig, n_rendered: int) -> gs.Scene:
    """Return a Genesis scene configured from ``cfg``."""
    return gs.Scene(
        show_viewer=cfg.viewer.show_viewer,
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=cfg.solver.joint_limit,
            dt=cfg.solver.dt,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=cfg.vis.show_world_frame,
            n_rendered_envs=n_rendered,
        ),
    )


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
