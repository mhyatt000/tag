from abc import abstractmethod
from dataclasses import dataclass
import math

import genesis as gs
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat
import numpy as np
import torch


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
                rendered_envs_idx=list(range(1)),
                show_world_frame=True,
                world_frame_size=1.0,
                show_link_frame=False,
                show_cameras=False,
                plane_reflection=True,
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


class Robot:
    pass


class Go2(Robot):
    def __init__(self, scene: gs.Scene):
        self.scene = scene
        self.robot = scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=(0, 0, 0),
                euler=(0, 0, 0),
            ),
        )


class Franka(Robot):
    def __init__(self, scene: gs.Scene):
        self.scene = scene
        self.robot = scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        )


class XArm7(Robot):
    def __init__(self, scene: gs.Scene):
        self.scene = scene
        self.robot = scene.add_entity(
            gs.morphs.MJCF(file="xml/ufactory_xarm7/xarm7.xml"),
        )
        # self.robot.plan_path
        # self.robot.inverse_kinematics
        # self.robot.inverse_kinematics_multilink

        self.wrist = self.robot.get_link("xarm_gripper_base_link")
        self.tcps = [
            self.robot.get_link("left_finger"),
            self.robot.get_link("right_finger"),
        ]

        self.eef = self.wrist
        self.tcp_pos = (0, 0, 0.172)

        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.gripper_names = ["right_driver_joint", "left_driver_joint"]

        self.joint_idxs = [self.robot.get_joint(name).dof_idx_local for name in self.joint_names]
        self.gripper_idxs = [self.robot.get_joint(name).dof_idx_local for name in self.gripper_names]
        self.dofs_idx = self.joint_idxs + self.gripper_idxs

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

    def close(self):
        """Close the gripper."""
        self.robot.set_dofs_position(position=np.array([0.4, 0.4]), dofs_idx_local=self.gripper_idxs)

    def open(self):
        """Open the gripper."""
        self.robot.set_dofs_position(position=np.array([0.0, 0.0]), dofs_idx_local=self.gripper_idxs)


class Aloha(Robot):
    def __init__(self, scene: gs.Scene):
        self.scene = scene
        self.robot = scene.add_entity(
            gs.morphs.MJCF(file="xml/aloha/aloha.xml"),
        )


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


class FrankaEnv(RobotEnv):
    """
    When grasping the object, we used force control for the 2 gripper dofs, and applied a 0.5N grasping force.
    If everything goes right, you will see the object being grasped and lifted.
    """

    def __init__(self, cfg: EnvConfig):
        super().__init__(cfg)
        self.robot = Franka(self.scene)
        """
        self.cam = self.scene.add_camera(
            res=(640, 480),
            pos=(3.5, 0.0, 2.5),
            lookat=(0, 0, 0.5),
            fov=30,
            GUI=cfg.gui,
        )
        self.cam.attach(self.robot.get_link('hand'))
        """


class XArm7Env(RobotEnv):
    def __init__(self, cfg: EnvConfig):
        super().__init__(cfg)
        self.robot = XArm7(self.scene)

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

        self.cams["wrist_hi"].attach(self.robot.wrist, offset_T=off(dz) @ R(-deg))
        self.cams["wrist_lo"].attach(self.robot.wrist, offset_T=off(-dz) @ R(deg))


class Stack(XArm7Env):
    def __init__(self, cfg: EnvConfig):
        super().__init__(cfg)

        # random uniform z=0
        def rpos():
            range = 0.5
            x = np.abs(np.random.uniform(range / 2, range, size=(3,)))
            x[1] *= np.random.choice([-1, 1], size=(1,))
            x[2] = 0
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


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [
                self.obs_scales["lin_vel"],
                self.obs_scales["lin_vel"],
                self.obs_scales["ang_vel"],
            ],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), gs.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            ),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
