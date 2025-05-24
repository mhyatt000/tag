import genesis as gs
from rich.pretty import pprint
from tqdm import tqdm
import tyro

from tag.gym.envs.chase.chase_config import ChaseEnvConfig
from tag.gym.envs.chase.chase_env import Chase
from tag.policy.dummy import DummyPolicy
from tag.utils import spec


def main(cfg: ChaseEnvConfig):
    gs.init(logging_level="info", backend=gs.gpu)

    pprint(cfg)

    env = Chase(cfg)
    policy = DummyPolicy(env.action_space)

    env.build()
    obs, _ = env.reset()
    for i in tqdm(range(len(env))[:200], desc="Running Dummy Policy"):
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        pprint(spec(action))

    env.record_visualization("4_Robot_Dummy_Policy_Test_With_Color")


if __name__ == "__main__":
    main(tyro.cli(ChaseEnvConfig))
