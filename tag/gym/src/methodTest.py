import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, Union

import pytest
from tagEnv import TagEnv
import torch


@pytest.mark.parametrize("Tag V1")  # First Implementation of Tag Environment
def main():
    # Test Environment Setup
    env = TagEnv()
    env.build()

    reset_buf, info = env.reset()
    assert isinstance(reset_buf, torch.Tensor), "Reset did not return valid reset observation"
    assert isinstance(info, Dict) or info is None, "Info from step must be a Dict or None"

    # Sample action and call step()
    # TODO: Implement Actions
    obs_buf, priv_obs_buf, rew_buf, reset_buf, extras = env.step()

    # Validate types and values
    assert obs_buf is env.obs_buf, "Step did not return valid observation"
    assert isinstance(priv_obs_buf, torch.Tensor), "Privileged Observation must be a torch.Tensor"
    assert isinstance(rew_buf, Union[torch.Tensor, None]), "Reward Buffer must be a Union[torch.Tensor, None]"
    assert isinstance(reset_buf, torch.Tensor), "Reset Buffer must be a torch.Tensor"
    assert isinstance(extras, Dict) or extras is None, "Extras from step must be a Dict or None"

    print("Tests Completed!")


if __name__ == "__main__":
    main()
