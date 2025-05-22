import os
import sys

import genesis as gs

gym_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if gym_dir not in sys.path:
    sys.path.insert(0, gym_dir)

# Now you can import directly from modules within the gym directory
from gym.envs.go2.demo.go2_demo_config import Go2DemoCfg
from tag.gym.utils.utils import get_args
from tag.tag.gym.envs.go2.demo.go2_double_demo import Go2DoubleDemo


def play(args):
    gs.init(
        backend=gs.gpu,
        logging_level="warning",
    )
    env_cfg = Go2DemoCfg()
    env = Go2DoubleDemo(env_cfg, "gpu", True)
    scene = env.scene
    cam = env.floating_camera

    cam.start_recording()

    for i in range(int(env_cfg.env.episode_length_s * (1.0 / env_cfg.control.dt))):
        scene.step()
        cam.render()

    cam.stop_recording(save_to_filename="./video.mp4", fps=60)


if __name__ == "__main__":
    args = get_args()
    play(args)
