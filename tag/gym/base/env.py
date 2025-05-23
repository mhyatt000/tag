from abc import abstractmethod
from typing import Dict, Tuple, Union

import genesis as gs
import torch

from tag.gym.base.config import EnvConfig, Task

# TODO add somewhere
# num_privileged_obs: int
# max_episode_length: int | torch.Tensor
# episode len is in task cfg


class BaseEnv:
    def __init__(self, args=None, cfg: EnvConfig = EnvConfig()):
        """Initialize common environment attributes."""

        self.cfg: EnvConfig = cfg
        self.device = gs.gpu
        self.n_envs = cfg.sim.num_envs
        self.n_rendered = cfg.vis.n_rendered_envs
        self.env_spacing = tuple(cfg.vis.env_spacing)

        task_cfg = getattr(cfg, "task", Task())
        self.num_obs = task_cfg.num_obs
        self.num_privileged_obs = task_cfg.num_privileged_obs
        self.max_episode_length = task_cfg.max_episode_length

        self._init_buffers()

    def build(self) -> None:
        ...
        # TODO(dle) add build here

    def step(self, actions: torch.Tensor) -> None: ...

    def reset(self) -> None: ...

    def record_data(self) -> None:
        """Finalize and save camera recordings, if any."""
        if getattr(self.cfg.vis, "visualized", False) and hasattr(self, "cam"):
            self.cam.stop_recording(
                save_to_filename="./tag/gym/mp4/tagV1_video.mp4", fps=60
            )

    def _init_buffers(self) -> None:
        """Allocate common buffers used for stepping the environment."""
        self.obs_buf = torch.zeros(
            (self.n_envs, self.num_obs), device=self.device, dtype=gs.tc_float
        )
        self.privileged_obs_buf = (
            None
            if self.num_privileged_obs is None
            else torch.zeros(
                (self.n_envs, self.num_privileged_obs),
                device=self.device,
                dtype=gs.tc_float,
            )
        )
        self.reset_buf = torch.ones(
            (self.n_envs,), device=self.device, dtype=gs.tc_int
        )
        self.rew_buf = torch.zeros(
            (self.n_envs,), device=self.device, dtype=gs.tc_float
        )
        self.episode_length_buf = torch.zeros(
            (self.n_envs,), device=self.device, dtype=gs.tc_int
        )

    def _update_buffers(self) -> None:
        """Placeholder buffer update used for testing."""
        self.episode_length_buf += 1
        self.obs_buf = torch.cat(
            [
                torch.cat(
                    [
                        torch.rand(4),
                        torch.rand(4),
                    ]
                )
            ]
        )
        self.rew_buf[:] = 0.0
        values = [i for i in range(len(self.rew_buf))]
        for i in range(len(values)):
            rew = values[i]
            self.rew_buf += rew

    @abstractmethod
    def get_observations(self) -> Tuple[torch.Tensor, Dict]: ...

    @abstractmethod
    def get_privileged_obs(self) -> Union[torch.Tensor, Dict]: ...

    def observation_space(self) -> Tuple[torch.Tensor, Dict]: ...

    def action_space(self) -> Tuple[torch.Tensor, Dict]: ...
