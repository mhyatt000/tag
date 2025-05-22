from dataclasses import dataclass, field

from rich.pretty import pprint

from tag.gym.base.config import (Asset, Control, EnvConfig, InitState,
                                 RobotConfig, Sim, Solver, Task, Terrain,
                                 Viewer, Vis, default)


@dataclass
class Go2Config(RobotConfig):
    control: Control = default(Control(kp=40.0, kd=2.0))
    asset: Asset = default(
        Asset(
            file="urdf/go2/urdf/go2.urdf",
            local_dofs=default([6, 8, 7, 9, 10, 12, 11, 13, 14, 16, 15, 17]),
        )
    )

    init_state: InitState = default(
        InitState(
            pos=default([0.0, 0.0, 0.42]),
            default_joint_angles=default(
                {
                    "FL_hip_joint": 0.1,
                    "RL_hip_joint": 0.1,
                    "FR_hip_joint": -0.1,
                    "RR_hip_joint": -0.1,
                    "FL_thigh_joint": 0.8,
                    "RL_thigh_joint": 1.0,
                    "FR_thigh_joint": 0.8,
                    "RR_thigh_joint": 1.0,
                    "FL_calf_joint": -1.5,
                    "RL_calf_joint": -1.5,
                    "FR_calf_joint": -1.5,
                    "RR_calf_joint": -1.5,
                }
            ),
        )
    )


@dataclass
class Go2EnvConfig(EnvConfig):
    terrain: Terrain = default(Terrain())
    viewer: Viewer = default(Viewer())
    vis: Vis = default(Vis(env_spacing=default([2.5, 2.5])))
    solver: Solver = default(Solver())
    sim: Sim = default(Sim())
    robotCfg: Go2Config = default(Go2Config())


# --- Task Environments ---

class TagConfig(Go2Config):
    task: Task = default(Task())


# IMPLEMENT: Configurations for Tasks/Rewards/Observations
