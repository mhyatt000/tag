from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple


class League:
    """Manage a league of policies for curriculum training."""

    def __init__(self, policies: List[Any] | None = None) -> None:
        self.policies: List[Any] = list(policies) if policies is not None else []
        # Store results for matches as a dictionary mapping
        # (policy_idx_a, policy_idx_b) -> result
        self.match_history: Dict[Tuple[int, int], Any] = {}

    def add_policy(self, policy: Any) -> int:
        """Add a policy to the league and return its index."""
        self.policies.append(policy)
        return len(self.policies) - 1

    def remove_policy(self, index: int) -> None:
        """Remove a policy from the league by index."""
        if 0 <= index < len(self.policies):
            self.policies.pop(index)
            # Drop related history entries
            self.match_history = {k: v for k, v in self.match_history.items() if index not in k}

    def select_opponents(self, index: int, k: int = 1) -> List[int]:
        """Return ``k`` opponent indices for the given policy index."""
        pool = [i for i in range(len(self.policies)) if i != index]
        k = min(k, len(pool))
        return random.sample(pool, k)

    def record_result(self, idx_a: int, idx_b: int, result: Any) -> None:
        """Store a training result between two policies."""
        self.match_history[(idx_a, idx_b)] = result

    def curriculum_pairs(self) -> List[Tuple[int, int]]:
        """Generate policy pairs for one curriculum step."""
        pairs = []
        for a in range(len(self.policies)):
            for b in self.select_opponents(a):
                pairs.append((a, b))
        return pairs

    def run_curriculum_step(self) -> None:
        """Run one league training step, if policies define ``train_against``."""
        for a, b in self.curriculum_pairs():
            pa = self.policies[a]
            pb = self.policies[b]
            if hasattr(pa, "train_against"):
                result = pa.train_against(pb)
                self.record_result(a, b, result)
