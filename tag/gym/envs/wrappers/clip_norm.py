from __future__ import annotations

from typing import Any, Tuple

import torch


class ClipNormWrapper:
    """Clip actions and observations using ``env.cfg.normalization``."""

    def __init__(self, env):
        self.env = env
        norm = getattr(env.cfg, "normalization", None)
        self.clip_actions = getattr(norm, "clip_actions", None)
        self.clip_obs = getattr(norm, "clip_observations", None)

    def step(self, actions: torch.Tensor) -> Tuple[Any, ...]:
        if self.clip_actions is not None:
            actions = torch.clip(actions, -self.clip_actions, self.clip_actions)
        obs, p_obs, rew, done, info = self.env.step(actions)
        if self.clip_obs is not None:
            obs = torch.clip(obs, -self.clip_obs, self.clip_obs)
            if p_obs is not None:
                p_obs = torch.clip(p_obs, -self.clip_obs, self.clip_obs)
        return obs, p_obs, rew, done, info

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def __getattr__(self, attr: str):
        return getattr(self.env, attr)
