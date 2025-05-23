from gymnasium import spaces
import torch

from tag.gym.base.config import RobotConfig


class Robot:
    cfg: RobotConfig

    observation_space: spaces.Space
    action_space: spaces.Space

    def reset(self) -> None: ...

    def act(self, action: torch.Tensor, mode: str = "control") -> None: ...
