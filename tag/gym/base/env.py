from abc import abstractmethod
from typing import Dict, Tuple, Union

import genesis as gs
import torch

from tag.gym.base.config import EnvConfig
from tag.names import BASE

# TODO add somewhere
# num_privileged_obs: int
# max_episode_length: int | torch.Tensor
# episode len is in task cfg


class BaseEnv:
    def __init__(self, cfg: EnvConfig):
        """Initialize common environment attributes."""

        self.cfg: EnvConfig = cfg
        self.device = gs.gpu
        self.n_envs = cfg.sim.num_envs
        self.n_rendered = cfg.vis.n_rendered_envs
        self.env_spacing = cfg.vis.env_spacing

        # TODO task not defined
        # task_cfg = getattr(cfg, "task", Task())
        # self.num_obs = task_cfg.num_obs
        # self.num_privileged_obs = task_cfg.num_privileged_obs
        # self.max_episode_length = task_cfg.max_episode_length

        self._max_steps = int(1e3)
        # self._init_buffers()

    def __len__(self) -> int:
        """Return the number of environments."""
        return self._max_steps

    def _init_scene(self) -> gs.Scene:
        self.scene = gs.Scene(
            show_viewer=self.cfg.viewer.show_viewer,
            rigid_options=gs.options.RigidOptions(
                enable_joint_limit=self.cfg.solver.joint_limit,
                dt=self.cfg.solver.dt,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=self.cfg.vis.show_world_frame,
                n_rendered_envs=self.cfg.vis.n_rendered_envs,
            ),
        )

    def step(self, actions: torch.Tensor) -> None: ...

    def reset(self) -> None: ...

    @abstractmethod
    def get_observations(self) -> Tuple[torch.Tensor, Dict]: ...

    @abstractmethod
    def get_privileged_obs(self) -> Union[torch.Tensor, Dict]: ...

    def observation_space(self) -> Tuple[torch.Tensor, Dict]: ...

    def action_space(self) -> Tuple[torch.Tensor, Dict]: ...

    def record_visualization(self, fname: str = None) -> None:
        """Finalize and save camera recordings, if any."""
        if getattr(self.cfg.vis, "visualized", False) and hasattr(self, "cam"):
            dir = BASE / "mp4"
            dir.mkdir(parents=True, exist_ok=True)
            fname = dir / f'{fname if fname else "video"}.mp4'
            self.cam.stop_recording(save_to_filename=fname, fps=60)

    def _init_buffers(self) -> None:
        """Allocate common buffers used for stepping the environment."""
        self.obs_buf = torch.zeros((self.n_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.privileged_obs_buf = (
            None
            if self.num_privileged_obs is None
            else torch.zeros(
                (self.n_envs, self.num_privileged_obs),
                device=self.device,
                dtype=gs.tc_float,
            )
        )
        self.reset_buf = torch.ones((self.n_envs,), device=self.device, dtype=gs.tc_int)
        self.rew_buf = torch.zeros((self.n_envs,), device=self.device, dtype=gs.tc_float)
        self.episode_length_buf = torch.zeros((self.n_envs,), device=self.device, dtype=gs.tc_int)

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
