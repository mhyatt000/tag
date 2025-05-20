# Import and register environments
from gym.envs.base.base_task import BaseTask
from gym.envs.base.legged_robot import LeggedRobot
from gym.envs.go2.demo.go2_demo_config import Go2DemoCfg
from gym.envs.go2.demo.go2_double_demo import Go2DoubleDemo
from tag.gym import GYM_ENVS_DIR, GYM_ROOT_DIR

__all__ = ["BaseTask", "LeggedRobot", "Go2DemoCfg"]
