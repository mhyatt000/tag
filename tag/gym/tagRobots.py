from typing import Dict

import genesis as gs

from tag.gym.go2Robot import Go2Robot
from tag.gym.tagConfig import Go2Config


class TagRobots:
    """Version 1"""

    def __init__(
        self, scene: gs.Scene, cfg: Go2Config, uiv: list[str]
    ):  # Need settings for distance apart, where to spawn, etc.
        # We need to separate them via a config or a task input somewhere
        cfg.init_state.pos = [0.0, -0.5, 0.42]
        self.robot_1 = Go2Robot(scene, cfg, uiv)
        cfg.init_state.pos = [0.0, 0.5, 0.42]
        self.robot_2 = Go2Robot(scene, cfg, uiv)
        self.robots = [self.robot_1, self.robot_2]

    def act(self, actions: Dict, mode: str = "position"):
        self.robot_1.act(action=actions["r1"])
        self.robot_2.act(action=actions["r2"])

    def reset(self):
        pass

    def compute_observations(self) -> Dict:
        obs = {"r1": self.robot_1.observe_state(), "r2": self.robot_2.observe_state()}
        return obs

    def __iter__(self):
        return iter(self.robots)
