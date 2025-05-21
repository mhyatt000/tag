# TODO: Fix File Structure
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseConfig import EnvConfig, RobotConfig
import genesis as gs
import torch
from vec_env import VecEnv


class BaseEnv(VecEnv):
    cfg: EnvConfig
    scene: gs.Scene
    episode_length: int

    def build(self) -> None: ...


class Robot:
    cfg: RobotConfig
    scene: gs.Scene

    def reset(self) -> None: ...

    def act(self, action: torch.Tensor, mode: str = "control") -> None: ...
