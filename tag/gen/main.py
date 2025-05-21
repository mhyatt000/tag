from collections import OrderedDict
from dataclasses import dataclass
import sys

import genesis as gs
import numpy as np
from rich.pretty import pprint
import torch
from tqdm import tqdm
import tyro

from env import EnvConfig, Stack


@dataclass
class RunCN:
    show_viewer: bool = False
    b: int = 1  # batch = number of envs

    ## camera params
    gui: bool = False
    record: bool = True
    fov: float = 40.0  # field of view

    # gs.cpu gs.cuda, gs.vulkan or gs.metal
    # cuda if available, else metal if macos, else cpu
    backend: str = gs.metal if sys.platform == "darwin" else gs.cuda

    @property
    def macos(self):
        return sys.platform == "darwin"


"""
randomize domain attributes:
Please check this example
https://github.com/Genesis-Embodied-AI/Genesis/blob/main/examples/rigid/set_phys_attr.py
env_ids


"""


def main(cfg: RunCN):
    pprint(cfg)

    gs.init(backend=cfg.backend)

    env = Stack(EnvConfig(show_viewer=False))
    # env =  Stack(EnvConfig())

    dic = [
        (
            "robot0_joint_pos",
            np.array(
                [
                    0.02783962,
                    -0.01082598,
                    0.03008944,
                    1.22933797,
                    -0.01303133,
                    1.20581304,
                    0.01083638,
                ]
            ),
        ),
        ("robot0_joint_vel", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("robot0_eef_pos", np.array([-0.07882972, 0.02562581, 1.2204146])),
        (
            "robot0_eef_quat",
            np.array([0.02576225, -0.99950083, 0.0056943, -0.01737744]),
        ),
        (
            "robot0_eef_quat_site",
            np.array([-0.02576225, 0.9995009, -0.00569431, 0.01737745]),
        ),
        ("robot0_gripper_qpos", np.array([0.02, 0.0, 0.0, -0.02, 0.0, 0.0])),
        ("robot0_gripper_qvel", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("cubeA_pos", np.array([-0.0791929, 0.00976445, 0.83])),
        ("cubeA_quat", np.array([0.0, 0.0, 0.94906114, 0.31509197])),
        ("cubeB_pos", np.array([0.07915531, 0.02626393, 0.835])),
        ("cubeB_quat", np.array([0.0, 0.0, 0.90675999, -0.42164716])),
        ("cubeA_to_cubeB", np.array([0.15834821, 0.01649948, 0.005])),
        (
            "gripper_to_cubeA",
            np.array([-3.63177072e-04, -1.58613612e-02, -3.90414602e-01]),
        ),
        ("gripper_to_cubeB", np.array([0.15798503, 0.00063812, -0.3854146])),
        (
            "robot0_proprio-state",
            np.array(
                [
                    0.02783962,
                    -0.01082598,
                    0.03008944,
                    1.22933797,
                    -0.01303133,
                    1.20581304,
                    0.01083638,
                    0.9996125,
                    0.9999414,
                    0.99954735,
                    0.33486161,
                    0.99991509,
                    0.35693368,
                    0.99994129,
                    0.02783602,
                    -0.01082576,
                    0.0300849,
                    0.94226732,
                    -0.01303096,
                    0.93412973,
                    0.01083616,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.07882972,
                    0.02562581,
                    1.2204146,
                    0.02576225,
                    -0.99950083,
                    0.0056943,
                    -0.01737744,
                    -0.02576225,
                    0.99950087,
                    -0.00569431,
                    0.01737745,
                    0.02,
                    0.0,
                    0.0,
                    -0.02,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ),
        (
            "object-state",
            np.array(
                [
                    -7.91928968e-02,
                    9.76445098e-03,
                    8.30000000e-01,
                    0.00000000e00,
                    0.00000000e00,
                    9.49061140e-01,
                    3.15091975e-01,
                    7.91553134e-02,
                    2.62639337e-02,
                    8.35000000e-01,
                    0.00000000e00,
                    0.00000000e00,
                    9.06759986e-01,
                    -4.21647160e-01,
                    1.58348210e-01,
                    1.64994827e-02,
                    5.00000000e-03,
                    -3.63177072e-04,
                    -1.58613612e-02,
                    -3.90414602e-01,
                    1.57985033e-01,
                    6.38121459e-04,
                    -3.85414602e-01,
                ]
            ),
        ),
    ]
    dic = OrderedDict(dic)
    joints = dic["robot0_joint_pos"]

    
    # ############## add terrain ##############
    # size_x, size_y = 9, 9
    # step_height = 15.0
    # step_width = 3

    # height_field = np.zeros((size_x, size_y))

    # for i in range(0, size_x, step_width):
    #     height_field[i : i + step_width, :] = (i // step_width) * step_height

    # height_field[step_width::step_width, :] = (
    #     height_field[step_width::step_width, :] - step_height
    # )
    # env.scene.add_entity(
    #     gs.morphs.Terrain(height_field=height_field, pos=(-1.0, -10.0, 0.0)),
    # )


    # franka = scene.add_entity( gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),)

    """
    go2a = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=[0.5, 0.5, 0.5],
            # quat=self.base_init_quat.cpu().numpy(),
        ),
    )

    go2b = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=[-0.5, -0.5, 0.5],
            # quat=self.base_init_quat.cpu().numpy(),
        ),
    )

    # shadow = scene.add_entity( gs.morphs.URDF( file="urdf/shadow_hand/shadow_hand.urdf", pos=[1,1,1], # quat=self.base_init_quat.cpu().numpy(),),)

    # humanoid = scene.add_entity( gs.morphs.MJCF( file="urdf/humanoid/humanoid.urdf", pos=[-1,-1,1], # quat=self.base_init_quat.cpu().numpy(),),)

    """

    cam = env.scene.add_camera(
        res = (640, 480),
        pos = (0, 1, 0.75),
        lookat = (0.0, 0.0, 0.5),
        fov = 40,
        GUI = False
    )

    env.build()

    # if cfg.record:
    cam.start_recording()

    up_quat = (np.array([0.7071068, 0, 0, 0.7071068]),)

    # gripper open pos
    # qpos[-2:] = 0.04
    n = 50

    env.step()
    cam.render()

    def close(qpos):
        qpos = qpos.clone()
        qpos[-2:] = 1.0
        return qpos

    def open(qpos):
        qpos = qpos.clone()
        qpos[-2:] = 0.0
        return qpos

    for t in range(200): # 2 seconds
        env.step()
        cam.set_pose(
            pos    = (3.5 * np.sin(t / 60), 3.5 * np.cos(t / 60), 0.75),
            lookat = (0, 0, 0.5),
        )
        cam.render()
    # for _ in tqdm(range(50)):
    #     env.step()
    #     cam.render()

    cam.stop_recording(save_to_filename = "./blocktest.mp4", fps = 30)

    quit()

    # for i in tqdm(range(int(100))):
    #     things = env.step(None)
    #     env.render(names=["bird", "wrist_hi", "wrist_lo"])

        # cam.set_pose(
        # pos=(3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
        # lookat=(0, 0, 0.5),
        # )
        # cam.render()
        # rgb, depth, segmentation, normal = cam.render(depth=True, segmentation=True, normal=True)

    # if cfg.record:
    # cam.stop_recording(save_to_filename="video.mp4", fps=60)


if __name__ == "__main__":
    main(tyro.cli(RunCN))
