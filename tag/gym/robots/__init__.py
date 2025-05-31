from typing import Union

from .go2 import Go2Config, Go2Robot
from .h1 import H1Config, H1Robot
from .robot import Robot, RobotConfig

__all__ = [
    "Robot",
    "RobotConfig",
    "Go2Config",
    "Go2Robot",
    "H1Config",
    "H1Robot",
]

ops = (RobotConfig, Go2Config, H1Config)
RobotTyp = Union[*ops]
