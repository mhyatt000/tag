import genesis as gs

from tag.gym.envs.chase.chase_env import Chase
from tag.policy.dummy import DummyPolicy

# import os
# os.environ["PYOPENGL_PLATFORM"] = "egl"


def main():
    gs.init(logging_level="info", backend=gs.gpu)

    env = Chase()
    policy = DummyPolicy(env.action_space)

    obs, _ = env.reset()
    for i in range(len(env)):
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i + 1} Action: {action}")
    env.record_visualization("4_Robot_Dummy_Policy_Test_With_Color")


if __name__ == "__main__":
    main()

# Test Environment With Dummy Policy
