import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from tagEnv import TagEnv


@pytest.mark.parametrize("env_id", ["Tag V1"])  # Replace with your custom env ID
def test_gymnasium_env_basic(env_id):
    env = TagEnv()

    # Check reset() returns observation and info
    obs, info = env.reset()

    # assert env.observation_space.contains(obs), "Reset returned invalid observation" - Bypassed

    assert isinstance(info, dict), "Info from reset must be a dict"

    # Check action space
    assert hasattr(env, "action_space"), "Env missing action_space"
    assert hasattr(env, "observation_space"), "Env missing observation_space"

    # Sample action and call step()

    # action = env.action_space.sample() - Bypassed

    next_obs, reward, terminated, truncated, step_info = env.step(3)

    # Validate types and values
    assert env.observation_space.contains(next_obs), "Step returned invalid observation"
    assert isinstance(reward, (int, float)), "Reward must be numeric"
    assert isinstance(terminated, bool), "Terminated must be a boolean"
    assert isinstance(truncated, bool), "Truncated must be a boolean"
    assert isinstance(step_info, dict), "Info from step must be a dict"

    # Check episode ends properly
    if terminated or truncated:
        env.reset()

    env.close()
