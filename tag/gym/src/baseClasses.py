# TODO: Fix File Structure
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import genesis as gs
from baseConfig import EnvConfig, RobotConfig
from typing import Any, Dict, Iterable, Protocol, Tuple

class BaseEnv:
    cfg: EnvConfig
    scene: gs.Scene
    episode_length: int
    observation_space : list # list-like: numpy, pytorch, etc...
    action_space: list # list-like

    def build(self) -> None: ...

    def reset(self) -> Tuple[Any, Dict[str, Any]]: ...

    def step(self, *actions: Any) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]: ... 
    '''
    Returns next_obs, reward, terminated, truncated, step_info
    '''

    def get_observations(self) -> Tuple[Any, Dict[str, Any]]: ...

    def get_priviledged_observations(self) -> Any:...

class Robot:
    cfg: RobotConfig
    scene: gs.Scene

    def reset(self) -> None: ...

    def act(self, act: Any | None = None, mode:str = 'control') -> None: ...


