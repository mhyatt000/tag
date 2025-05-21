from abc import abstractmethod
from dataclasses import dataclass
import math

import genesis as gs
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat
import numpy as np
import torch

# import tiled_terrain as tt


@dataclass
class EnvConfig:
    num_envs: int = 0  # 1 is parallel... 0 non-parallel
    latency: bool = True  # there is a 1 step latency on real robot
    ep_len_s: int = 1000  # episode length in sec

    dt: float = 0.02  # control frequency on real robot is 50hz
    show_viewer: bool = True
    gui: bool = False

    def _ep_len(self):
        return math.ceil(self.ep_len_s / 0.02)


class Env:
    """
    Base interface for all environments.
    Defines core members and abstract methods.
    """

    def __init__(self, cfg: EnvConfig = EnvConfig()):
        self.cfg = cfg
        self.device = gs.device

        # self.obs_cfg = obs_cfg
        # self.reward_cfg = reward_cfg
        # self.command_cfg = command_cfg

        # episode horizon
        self.max_episode_length = math.ceil(self.cfg.ep_len_s / self.cfg.dt)

        # self.obs_scales = obs_cfg["obs_scales"]
        # self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.cfg.dt,
                substeps=2,
                gravity=(0, 0, -10.0),
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=60,
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=True,
                world_frame_size=1.0,
                show_link_frame=False,
                show_cameras=False,
                plane_reflection=False,
                ambient_light=(0.1, 0.1, 0.1),
            ),
            renderer=gs.renderers.Rasterizer(),
            rigid_options=gs.options.RigidOptions(
                dt=self.cfg.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=self.cfg.show_viewer,
        )
        self.plane = self.scene.add_entity(morph=gs.morphs.Plane(), surface=gs.surfaces.Default(color=(0.5, 0.5, 0.5)))

    def build(self):
        self.scene.build(n_envs=self.cfg.num_envs)

    @abstractmethod
    def step(self, *actions):
        """Advance the simulation by one step given actions."""
        self.scene.step()

    @abstractmethod
    def reset_idx(self, envs_idx):
        """Reset a subset of environments by index."""
        pass

    @abstractmethod
    def reset(self):
        """Reset all environments."""
        pass

    def get_observations(self):
        """Return current observations and extras dict."""
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        """Return privileged observations (if any)."""
        return None

    def reset(self):
        ############ Optional: set control gains ############
        # original gains
        kp0 = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])
        kv0 = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10])

        # to double ω_n: ω_n' = 2 ω_n  ⇒  Kp' = (2²) Kp  and  Kv' = (2) Kv
        scale = 4.0
        kp_fast = kp0 * scale**2  # 4× stiffer
        kv_fast = kv0 * scale  # 2× more damping

        self.robot.set_dofs_kp(
            kp=kp_fast,
            dofs_idx_local=self.dofs_idx,
        )
        self.robot.set_dofs_kv(
            kv=kv_fast,
            dofs_idx_local=self.dofs_idx,
        )
        # set force range for safety
        self.robot.set_dofs_force_range(
            lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
            dofs_idx_local=self.dofs_idx,
        )

    def __call__(self, act=None, idxs=None, mode="control"):
        """act shape 7+gripper"""

        assert act.shape[-1] in [2, 7, 9]
        idxs = self.dofs_idx if idxs is None else idxs

        if act.shape[-1] == 7:
            idxs = self.joint_idxs

        fns = {
            "set": self.robot.set_dofs_position,
            "control": self.robot.control_dofs_position,
        }
        fns[mode](position=act, dofs_idx_local=idxs)

    def act(self, act=None, mode="control"):
        return self(act, mode=mode)

class RobotEnv(Env):
    REGISTRY = {}

    def __init__(self, cfg: EnvConfig):
        super().__init__(cfg)

        self.cams = {}
        self.cams

        self.x = 0.5
        self.lookat = (self.x, 0, 0.5)  # center of task space
        self.dist = 1  # default distance
        self.fov = 40

        self.size = (224, 224)
        self.cams["bird"] = self.scene.add_camera(
            res=self.size,
            pos=(self.x, 0.0, self.dist * 2),
            lookat=self.lookat,
            fov=self.fov,
            GUI=self.cfg.gui,
        )

        self.cams["worm"] = self.scene.add_camera(
            res=self.size,
            pos=(self.dist, 0.0, 0.01),  # slightly above 0
            lookat=(0, 0, 0.05),
            fov=self.fov,
            GUI=self.cfg.gui,
        )
        self.cams["front"] = self.scene.add_camera(
            res=self.size,
            pos=(self.dist, 0, 1),
            lookat=self.lookat,
            fov=self.fov,
            GUI=self.cfg.gui,
        )

        self.cams["left"] = self.scene.add_camera(
            res=self.size,
            pos=(self.x, -1, 1),
            lookat=self.lookat,
            fov=self.fov,
            GUI=self.cfg.gui,
        )
        self.cams["right"] = self.scene.add_camera(
            res=self.size,
            pos=(self.x, 1, 1),
            lookat=self.lookat,
            fov=self.fov,
            GUI=self.cfg.gui,
        )

    def render(self, *args, names=None, **kwargs):
        outs = {k: v.render(*args, **kwargs) for k, v in self.cams.items() if k in names}
        return outs


class XArm7Env(RobotEnv):
    def __init__(self, cfg: EnvConfig):
        super().__init__(cfg)

        self.cams["wrist_hi"] = self.scene.add_camera(
            res=self.size,
            pos=(0, 0, 0),
            lookat=self.lookat,
            fov=80,
            GUI=self.cfg.gui,
        )
        self.cams["wrist_lo"] = self.scene.add_camera(
            res=self.size,
            pos=(0, 0, 0),
            lookat=self.lookat,
            fov=80,
            GUI=self.cfg.gui,
        )

    def build(self):
        super().build()

        deg = -np.deg2rad(35.0)
        R = lambda r: np.array(
            [
                [np.cos(r), 0, np.sin(r), 0],
                [0, 1, 0, 0],
                [-np.sin(r), 0, np.cos(r), 0],
                [0, 0, 0, 1],
            ]
        )

        dz = 0.15  # shift 15 cm
        off = lambda d: np.array([[1, 0, 0, d], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.0]])


class Stack(XArm7Env):
    def __init__(self, cfg: EnvConfig):
        super().__init__(cfg)

        # random uniform z=0
        def rpos():
            range = 0.5
            x = np.abs(np.random.uniform(range / 2, range, size=(3,)))
            x[1] *= np.random.choice([-1, 1], size=(1,))
            x[2] = 4
            return x

        self.cubea = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.04, 0.04, 0.04),
                pos=rpos(),
            ),
            surface=gs.surfaces.Default(color=(1, 0, 0)),
        )

        self.cubeb = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.04, 0.04, 0.04),
                pos=rpos(),
            ),
            surface=gs.surfaces.Default(color=(0, 1, 0)),
        )


        ############## add terrain ##############
        size_x, size_y = 9, 9
        step_height = 0.01
        step_width = 2

        height_field = np.zeros((size_x, size_y))

        for i in range(0, size_x, step_width):
            height_field[i : i + step_width, :] = (i // step_width) * step_height

        height_field[step_width::step_width, :] = (
            height_field[step_width::step_width, :] - step_height
        )
        
        self.scene.add_entity(
            gs.morphs.Terrain(height_field=height_field, pos=(-1, -1, 0.001), n_subterrains=(1,1), horizontal_scale=0.5, vertical_scale=2, subterrain_size = (5.0, 5.0))
        )
        self.scene.add_entity(
            gs.morphs.Terrain(height_field=height_field, pos=(-1, 1, 0.001), n_subterrains=(1,1), vertical_scale=1, subterrain_size = (5.0, 5.0))
        )

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower