from typing import Dict

import genesis as gs

from gymnasium import spaces
from tag.gym.robots.go2 import Go2Config, Go2Robot
from tag.gym.robots.robot import Robot


class MultiRobot(Robot):
    """Version 1"""

    def __init__(self, scene: gs.Scene, cfg: Go2Config, uiv: list[str]):
        # Need settings for distance apart, where to spawn, etc.
        # We need to separate them via a config or a task input somewhere

        cfg.init_state.pos = [0.0, -0.5, 0.42]
        robot_1 = Go2Robot(scene, cfg, uiv[0])
        cfg.init_state.pos = [0.0, 0.5, 0.42]
        robot_2 = Go2Robot(scene, cfg, uiv[1])
        self.robots: Dict[str, Go2Robot] = {uiv[0]: robot_1, uiv[1]: robot_2}
        self.robot_1 = robot_1
        self.robot_2 = robot_2

        self.observation_space = spaces.Dict(
            {uid: r.observation_space for uid, r in self.robots.items()}
        )
        self.action_space = spaces.Dict(
            {uid: r.action_space for uid, r in self.robots.items()}
        )

    def act(self, actions: Dict, mode: str = "position"):
        for uid, robot in self.robots.items():
            robot.act(action=actions[uid])

    def reset(self):
        pass

    def compute_observations(self) -> Dict:
        return {uid: robot.observe_state() for uid, robot in self.robots.items()}

    def __iter__(self):
        return iter(self.robots.values())

    # TODO(codex) this needs observation and action space implemented
