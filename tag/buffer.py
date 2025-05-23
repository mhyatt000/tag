"""Simple replay buffer implementations for storing experience."""

from __future__ import annotations

from collections import deque
import random
from typing import Any, Deque, Dict, Iterable, Tuple


class ReplayBuffer:
    """Basic replay buffer for off-policy algorithms."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.buffer: Deque[Tuple[Any, Any, float, Any, bool]] = deque(maxlen=capacity)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.buffer)

    def add(self, obs: Any, action: Any, reward: float, next_obs: Any, done: bool) -> None:
        self.buffer.append((obs, action, float(reward), next_obs, bool(done)))

    def sample(self, batch_size: int) -> Dict[str, Iterable[Any]]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones,
        }


class HindsightExperienceReplayBuffer(ReplayBuffer):
    """Replay buffer with optional hindsight relabelling."""

    def __init__(self, capacity: int, her_ratio: float = 0.8) -> None:
        super().__init__(capacity)
        self.her_ratio = her_ratio

    def add(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        done: bool,
        achieved_goal: Any | None = None,
        desired_goal: Any | None = None,
    ) -> None:
        super().add(obs, action, reward, next_obs, done)
        if achieved_goal is not None and desired_goal is not None and random.random() < self.her_ratio:
            her_obs = obs
            her_next_obs = next_obs
            super().add(her_obs, action, 0.0, her_next_obs, done)
