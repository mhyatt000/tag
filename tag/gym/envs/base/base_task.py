from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import genesis as gs
import torch


@dataclass
class BaseTask(ABC):
    render_fps: int = field(init=False)
    last_frame_time: int = field(init=False)
    num_envs: int = field(init=False)
    headless: bool = field(init=False)
    device: torch.device = field(init=False)

    num_build_envs: int = field(init=False)
    num_obs: int = field(init=False)
    num_privileged_obs: int = field(init=False)
    num_actions: int = field(init=False)

    # Buffers
    obs_buf: torch.Tensor = field(init=False)
    rew_buf: torch.Tensor = field(init=False)
    reset_buf: torch.Tensor = field(init=False)
    episode_length_buf: torch.Tensor = field(init=False)
    time_out_buf: torch.Tensor = field(init=False)
    privileged_obs_buf: Optional[torch.Tensor] = field(init=False)
    extras: dict = field(init=False)

    def __init__(self, cfg, sim_device, headless: bool = False):
        self.render_fps = 50
        self.last_frame_time = 0
        self.num_envs = 1 if cfg.env.num_envs == 0 else cfg.env.num_envs
        self.headless = headless
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            assert sim_device in ["cpu", "cuda"]
            self.device = torch.device("gpu")

        self.num_build_envs = self.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_int)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_int)

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(
                self.num_envs,
                self.num_privileged_obs,
                device=self.device,
                dtype=gs.tc_float,
            )
        else:
            self.privileged_obs_buf = None

        self.extras = dict()

        self.create_sim()

    @abstractmethod
    def get_observations(self):
        pass

    @abstractmethod
    def get_privledged_observations(self):
        pass

    def reset_idx(self, env_ids):
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
