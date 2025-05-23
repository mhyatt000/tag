from tag.gym.dummyPolicy import DummyPolicy
from tag.gym.tagEnv import TagEnv


def main():
    env = TagEnv()
    policy = DummyPolicy(env.action_space)

    obs, _ = env.reset()
    for i in range(1000):
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i + 1} Action: {action}")
    env.record_data()


if __name__ == "__main__":
    main()
