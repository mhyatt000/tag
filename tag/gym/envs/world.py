from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import numpy as np
from gymnasium import spaces

from tag.utils import defaultcls, obs2space, spec

from .base import BaseEnv, BaseEnvConfig
from .mixins import CameraMixin, DomainRandMixin, TerrainMixin
from .mixins.cam import Cam
from .mixins.terrain import Terrain

from tag.gym.terrain.terrain import TerrainFactoryConfig


@dataclass
class WorldEnvConfig(BaseEnvConfig):
    terrain: Terrain = defaultcls(Terrain)
    cam: Cam = defaultcls(Cam)
    terrainfactory: TerrainFactoryConfig = defaultcls(TerrainFactoryConfig)


class WorldEnv(BaseEnv, TerrainMixin, CameraMixin, DomainRandMixin):
    """Environment without robots for sensor-only tasks."""

    def __init__(self, cfg: WorldEnvConfig):
        super().__init__(cfg)

        self._init_terrain(cfg.terrainfactory)  # TODO uncomment when terrain is ready
        self._setup_camera()

        # TODO inherit camera obs space

    @property
    def observation_space(self):
        obs = jax.tree.map( lambda x: np.zeros_like(x.shape), self.observe())  # torch 2 np
        space = obs2space(obs)
        return space

    @property
    def action_space(self):
        return spaces.Dict({})

    def reset(self) -> Tuple[dict, None]:
        return self.observe(), None

    def observe(self) -> Tuple[dict, dict]:
        if self.cfg.cam.enable:
            img, _, _, _ = self.cam.render()
            return {"imgs": img}
        return {}
