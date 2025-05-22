import genesis as gs
import torch

from tag.gym.base.config import RobotConfig


class Robot:
    cfg: RobotConfig

    def reset(self) -> None: ...

    def act(self, action: torch.Tensor, mode: str = "control") -> None: ...
