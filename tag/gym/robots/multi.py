from copy import deepcopy
from typing import Dict

import genesis as gs
from gymnasium import spaces

from tag.gym.robots.go2 import Go2Config, Go2Robot
from tag.gym.robots.robot import Robot

from .utils import tile_xyz


class MultiRobot(Robot):
    """Version 1"""

    def __init__(self, scene: gs.Scene, cfg: Go2Config, n: list[str], n_envs, colors: list):
        # FEATURE: Need settings for distance apart, where to spawn, etc.
        # NOTE(dle): Need to add something to define the distance between them properly

        init_pos_map = tile_xyz(n, cfg.init_state.pos[2])

        self.robots = {}
        for i in range(n):
            _cfg = deepcopy(cfg)
            _cfg.init_state.pos = init_pos_map[i]

            ### NOTE(dle): Placeholder Color System
            # NOTE(dle): Spaces Temp Fix
            robot = Go2Robot(scene, _cfg, n_envs, colors[i])
            self.robots[robot.name] = robot

        # NOTE(dle): If we don't tile, torch automatically tiles repeats, which is bad because we are getting different data
        # TODO: These need to be tiled n_envs x dofs

        self.observation_space = spaces.Dict({k: r.observation_space for k, r in self.robots.items()})
        self.action_space = spaces.Dict({k: r.action_space for k, r in self.robots.items()})

    def act(self, actions: Dict, mode: str = "position"):
        for k, robot in self.robots.items():
            robot.act(action=actions[k])

    def reset(self):
        pass

    def compute_observations(self) -> Dict:
        return {k: robot.observe_state() for k, robot in self.robots.items()}

    def __iter__(self):
        return iter(self.robots.values())
