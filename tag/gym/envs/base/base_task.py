from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class TaskConfig:
    render_fps: int = field(init=False)
    last_frame_time: int = field(init=False)
    num_envs: int = field(init=False)
    headless: bool = field(init=False)
    device: torch.device = field(init=False)

    num_build_envs: int = field(init=False)
    num_obs: int = field(init=False)
    num_privileged_obs: int = field(init=False)
    num_actions: int = field(init=False)


@dataclass
class Buffer:  # TODO(dle) init them
    obs: torch.Tensor
    rew: torch.Tensor
    reset: torch.Tensor
    episode_length: torch.Tensor
    time_out: torch.Tensor
    privileged_obs: Optional[torch.Tensor]


@dataclass
class BaseTask(ABC):
    extras: dict = field(init=False)

    def __init__(self, cfg, taskconfig, headless: bool = False):
        # TODO(dle) what type is cfg

        self.render_fps = 50
        self.last_frame_time = 0
        self.num_envs = 1 if cfg.env.num_envs == 0 else cfg.env.num_envs
        self.headless = headless

        cuda = torch.cuda.is_available()
        self.device = torch.device("gpu" if cuda else "cpu")

        # TODO(dle) cleanup.
        # num_obs num_actions is unique to env. cant be passed in
        self.num_build_envs = self.num_envs  # gs is natively vectorized
        # self.num_obs = cfg.env.num_observations
        # self.num_privileged_obs = cfg.env.num_privileged_obs
        # self.num_actions = cfg.env.num_actions

        # ignore privileged obs for now
        # TODO(dle) fix buffer class
        # p = self.num_privileged_obs
        self.buffer = Buffer()  # init buffer
        self.extras = dict()

        self.create_sim()

    @abstractmethod
    def get_observations(self, privileged: bool = False):
        pass

    # (mhyatt) personal preference to use one method for obs
    # @abstractmethod
    # def get_privledged_observations(self):
    # pass

    def reset_idx(self, env_ids):
        # (mhyatt) how is this different from reset?
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass

    @abstractmethod
    def create_sim(self):
        pass
