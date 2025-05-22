import genesis as gs
import torch

from tag.gym.base.config import EnvConfig, RobotConfig

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

import torch

# TODO add somewhere
# num_privileged_obs: int
# max_episode_length: int | torch.Tensor
# episode len is in task cfg

class BaseEnv():

    def __init__(self, args=None, cfg: EnvConfig = EnvConfig()):
        pass

    def build(self) -> None: ...
        # TODO(dle) add build here

    def step(self, actions: torch.Tensor) -> None: ...

    def reset(self) -> None: ...

    @abstractmethod
    def get_observations(self) -> Tuple[torch.Tensor, Dict]: ...

    @abstractmethod
    def get_privileged_obs(self) -> Union[torch.Tensor, Dict]: ...

    def observation_space(self) -> Tuple[torch.Tensor, Dict]:

    def action_space(self) -> Tuple[torch.Tensor, Dict]:

