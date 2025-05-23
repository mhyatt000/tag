from typing import Dict

import genesis as gs
from gymnasium import spaces

from tag.gym.robots.go2 import Go2Config, Go2Robot
from tag.gym.robots.robot import Robot

from .utils import tile_xyz


class MultiRobot(Robot):
    """Version 1"""

    def __init__(self, scene: gs.Scene, cfg: Go2Config, uid: list[str], n_envs: int, colors: list):
        # FEATURE: Need settings for distance apart, where to spawn, etc.
        # NOTE(dle): Need to add something to define the distance between them properly

        self.n_robots = len(uid)
        self.robots = {}
        init_pos_map = tile_xyz(self.n_robots, cfg.init_state.pos[2])
        for i in range(self.n_robots):
            cfg.init_state.pos = init_pos_map[i]  # NOTE(dle): Is this dumb? Probably, fix this
            ### NOTE(dle): Placeholder Color System
            self.robots.update({uid[i]: Go2Robot(scene, cfg, uid[i], n_envs, colors[i])})  # NOTE(dle): Spaces Temp Fix
            ###

        # TODO: These need to be tiled n_envs x dofs
        # NOTE(dle): If we don't tile, torch automatically tiles repeats, which is bad because we are getting different data
        # Temp fix for now is passing n_envs into constructor
        self.observation_space = spaces.Dict({uid: r.observation_space for uid, r in self.robots.items()})
        self.action_space = spaces.Dict({uid: r.action_space for uid, r in self.robots.items()})

    def act(self, actions: Dict, mode: str = "position"):
        for uid, robot in self.robots.items():
            robot.act(action=actions[uid])

    def reset(self):
        pass

    def compute_observations(self) -> Dict:
        return {uid: robot.observe_state() for uid, robot in self.robots.items()}

    def __iter__(self):
        return iter(self.robots.values())
