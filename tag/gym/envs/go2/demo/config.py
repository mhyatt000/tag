from dataclasses import dataclass

from tag.gym.envs.base.config import Asset, Control, Env, State, default


@dataclass
class Go2Asset(Asset):
    file: str = "urdf/go2/urdf/go2.urdf"
    dof_names: list[str] = [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]
    foot_name: list[str] = ["foot"]
    penalize_contacts_on: list[str] = ["thigh", "calf"]
    terminate_after_contacts_on: list[str] = ["base"]
    links_to_keep: list[str] = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    self_collisions: bool = True

    # TODO(dle) you can implement this as subclass of Asset
    # but tbh not sure if we need Assets or if it is just extra code
    # lets discuss

    # used to be...
    # class asset(LeggedRobotCfg.asset):


DEFAULT_JOINTS = {  # = target angles [rad] when action = 0.0
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


GO2_INIT: State = State(
    pos=[0.0, -0.5, 0.42],
    default_joint_angles=DEFAULT_JOINTS,
)


@dataclass
class Go2DemoEnv(Env):  # NOTE(dle) it inherits from Env
    init_state_1: State = default(GO2_INIT)
    init_state_2: State = default(GO2_INIT)

    control: Control = default(
        Control(
            kp={"joint": 20.0},  # stiffness
            kv={"joint": 0.5},  # dampening
            action_scale=0.25,
        )
    )
