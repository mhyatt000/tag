import os
from typing import Optional

import genesis as gs
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
import numpy as np
import torch

from gym.envs.base.base_task import BaseTask
from tag.gym import GYM_ROOT_DIR

from .legged_robot_config import LeggedRobotCfg


class LeggedRobot(BaseTask):
    cfg: LeggedRobotCfg
    height_samples: Optional[torch.Tensor]
    debug_viz: bool
    init_done: bool

    def __init__(self, cfg: LeggedRobotCfg, sim_device, headless):
        self.cfg = cfg
        self.height_samples = None
        self.debug_viz = self.cfg.env.debug_viz
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_device, headless)

        self._init_buffers()
        # TODO: Add reward function
        self.init_done = True

    # TODO: Implement step() method in LeggedRobot
    def step(self, actions: torch.Tensor):
        pass

    # TODO: Implement post_physics_step() method in LeggedRobot
    def post_physics_step(self):
        pass

    # TODO: Implement check_base_pos_out_of_bounds() and check_termination() methods in LeggedRobot
    def check_base_pos_out_of_bounds(self):
        pass

    def check_termination(self):
        pass

    # TODO: Implement reset_idx() method in LeggedRobot
    def reset_idx(self, env_ids):
        pass

    # TODO: Implement compute_reward() and compute_observations() methods in LeggedRobot
    def compute_reward(self):
        pass

    def compute_observations(self):
        pass

    # TODO: Complete implementation of create_sim() method in LeggedRobot
    def create_sim(self):
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.sim_dt, substeps=self.sim_substeps
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / self.dt * self.cfg.control.decimation),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                n_rendered_envs=min(self.cfg.viewer.num_rendered_envs, self.num_envs)
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.sim_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=self.cfg.asset.self_collisions,
            ),
            show_viewer=self.cfg.viewer.show_viewer,
        )
        # query rigid solver
        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            self.rigid_solver = solver

        # add camera if needed
        if self.cfg.viewer.add_camera:
            self._setup_camera()

        # add terrain
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type == "plane":
            self.terrain = self.scene.add_entity(
                gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
            )
        # elif mesh_type=='heightfield':
        #    self.utils_terrain = Terrain(self.cfg.terrain)
        #    self._create_heightfield()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]"
            )
        self.terrain.set_friction(self.cfg.terrain.friction)
        # specify the boundary of the heightfield
        self.terrain_x_range = torch.zeros(2, device=self.device)
        self.terrain_y_range = torch.zeros(2, device=self.device)
        # if self.cfg.terrain.mesh_type=='heightfield':
        #    self.terrain_x_range[0] = -self.cfg.terrain.border_size + 1.0 # give a small margin(1.0m)
        #    self.terrain_x_range[1] = self.cfg.terrain.border_size + self.cfg.terrain.num_rows * self.cfg.terrain.terrain_length - 1.0
        #    self.terrain_y_range[0] = -self.cfg.terrain.border_size + 1.0
        #    self.terrain_y_range[1] = self.cfg.terrain.border_size + self.cfg.terrain.num_cols * self.cfg.terrain.terrain_width - 1.0
        # elif self.cfg.terrain.mesh_type=='plane': # the plane used has limited size,
        # and the origin of the world is at the center of the plane
        #    self.terrain_x_range[0] = -self.cfg.terrain.plane_length/2+1
        #    self.terrain_x_range[1] = self.cfg.terrain.plane_length/2-1
        #    self.terrain_y_range[0] = -self.cfg.terrain.plane_length/2+1 # the plane is a square
        #    self.terrain_y_range[1] = self.cfg.terrain.plane_length/2-1
        self._create_envs()

    # TODO: Implement set_camera() method in LeggedRobot
    def set_camera(self, pos, lookat):
        self.floating_camera.set_pose(pos=pos, lookat=lookat)

    # TODO: Implement callback methods in LeggedRobot: _post_physics_step_callback...
    def _setup_camera(self):
        self.floating_camera = self.scene.add_camera(
            res=self.cfg.cam_settings.res,
            pos=np.array(self.cfg.cam_settings.pos),
            lookat=np.array(self.cfg.cam_settings.lookat),
            fov=self.cfg.cam_settings.fov,
            GUI=self.cfg.cam_settings.GUI,
        )

        self._recording = False
        self._recorded_frames = []

    def _post_physics_set_callback(self):
        pass

    def _resample_commands(self, env_ids):
        pass

    def _compute_torques(self, actions):
        pass

    def _compute_target_dof_pos(self, actions):
        pass

    def _reset_dofs(self, envs_idx):
        pass

    def _reset_root_states(self, envs_idx):
        pass

    def _push_robots(self):
        pass

    def _update_terrain_curriculum(self, env_ids):
        pass

    def _update_command_curriculum(self, env_ids):
        pass

    def _get_noise_scale_vec(self, cfg):
        pass

    def _init_buffers(self):
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.forward_vec = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.forward_vec[:, 0] = 1.0
        self.base_init_pos = torch.tensor(self.cfg.init_state.pos, device=self.device)
        self.base_init_quat = torch.tensor(self.cfg.init_state.quat, device=self.device)
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.projected_gravity = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.global_gravity = torch.tensor(
            [0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float
        ).repeat(self.num_envs, 1)
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float
        )
        self.rew_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.reset_buf = torch.ones(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        # self.commands = torch.zeros((self.num_envs, self.cfg.commands.num_commands), device=self.device, dtype=gs.tc_float)
        # self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
        #    device=self.device,
        #    dtype=gs.tc_float,
        #    requires_grad=False,) # TODO change this
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float
        )
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float
        )
        self.feet_air_time = torch.zeros(
            (self.num_envs, len(self.feet_indices)),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.last_contacts = torch.zeros(
            (self.num_envs, len(self.feet_indices)), device=self.device, dtype=gs.tc_int
        )
        self.link_contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, 3),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.continuous_push = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.env_identities = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=gs.tc_int,
        )
        self.terrain_heights = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=gs.tc_float,
        )
        # if self.cfg.terrain.measure_heights:
        #    self.height_points = self._init_height_points()
        # self.measured_heights = 0

        self.default_dof_pos = torch.tensor(
            [
                self.cfg.init_state.default_joint_angles[name]
                for name in self.cfg.asset.dof_names
            ],
            device=self.device,
            dtype=gs.tc_float,
        )
        # PD control
        kp = self.cfg.control.kp
        kv = self.cfg.control.kv

        self.p_gains, self.d_gains = [], []
        for dof_name in self.cfg.asset.dof_names:
            for key in kp.keys():
                if key in dof_name:
                    self.p_gains.append(kp[key])
                    self.d_gains.append(kv[key])
        self.p_gains = torch.tensor(self.p_gains, device=self.device)
        self.d_gains = torch.tensor(self.d_gains, device=self.device)
        self.batched_p_gains = self.p_gains[None, :].repeat(self.num_envs, 1)
        self.batched_d_gains = self.d_gains[None, :].repeat(self.num_envs, 1)
        # PD control params
        self.robot.set_dofs_kp(self.p_gains, self.motor_dofs)
        self.robot.set_dofs_kv(self.d_gains, self.motor_dofs)

    def _prepare_reward_function(self):
        pass

    def _create_heightfield(self):
        pass

    def _create_envs(self):
        asset_path = self.cfg.asset.file.format(GYM_ROOT_DIR=GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=os.path.join(asset_root, asset_file),
                merge_fixed_links=True,  # if merge_fixed_links is True, then one link may have multiple geometries, which will cause error in set_friction_ratio
                links_to_keep=self.cfg.asset.links_to_keep,
                pos=np.array(self.cfg.init_state.pos),
                quat=np.array(self.cfg.init_state.quat),
                fixed=self.cfg.asset.fix_base_link,
            ),
            visualize_contact=self.debug,
        )

        # build
        self.scene.build(n_envs=self.num_envs)

        self._get_env_origins()

        # name to indices
        self.motor_dofs = [
            self.robot.get_joint(name).dof_idx_local for name in self.dof_names
        ]

        # find link indices, termination links, penalized links, and feet
        def find_link_indices(names):
            link_indices = list()
            for link in self.robot.links:
                flag = False
                for name in names:
                    if name in link.name:
                        flag = True
                if flag:
                    link_indices.append(link.idx - self.robot.link_start)
            return link_indices

        self.termination_indices = find_link_indices(
            self.cfg.asset.terminate_after_contacts_on
        )
        all_link_names = [link.name for link in self.robot.links]
        print(f"all link names: {all_link_names}")
        print("termination link indices:", self.termination_indices)
        self.penalized_indices = find_link_indices(self.cfg.asset.penalize_contacts_on)
        print(f"penalized link indices: {self.penalized_indices}")
        self.feet_indices = find_link_indices(self.cfg.asset.foot_name)
        print(f"feet link indices: {self.feet_indices}")
        assert len(self.termination_indices) > 0
        assert len(self.feet_indices) > 0
        self.feet_link_indices_world_frame = [i + 1 for i in self.feet_indices]

        # TODO: Complete _create_envs() method in LeggedRobot
        """
        # dof position limits
        self.dof_pos_limits = torch.stack(self.robot.get_dofs_limit(self.motor_dofs), dim=1)
        self.torque_limits = self.robot.get_dofs_force_range(self.motor_dofs)[1]
        for i in range(self.dof_pos_limits.shape[0]):
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = (
                m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            )
            self.dof_pos_limits[i, 1] = (
                m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        )

        # randomize friction
        if self.cfg.domain_rand.randomize_friction:
            self._randomize_friction(np.arange(self.num_envs))
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(np.arange(self.num_envs))
        # randomize COM displacement
        if self.cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(np.arange(self.num_envs))
        """

    def _randomize_friction(self, env_ids=None):
        pass

    def _randomize_base_mass(self, env_ids=None):
        pass

    def _randomize_com_displacement(self, env_ids):
        pass

    def _parse_cfg(self, cfg: LeggedRobotCfg):
        self.dt = self.cfg.control.dt
        if self.cfg.sim.use_implicit_controller:  # use embedded PD controller
            self.sim_dt = self.dt
            self.sim_substeps = self.cfg.control.decimation
        else:  # use explicit PD controller
            self.sim_dt = self.dt / self.cfg.control.decimation
            self.sim_substeps = 1
        # self.obs_scales = self.cfg.normalization.obs_scales
        # self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        # self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        # if self.cfg.terrain.mesh_type not in ['heightfield']:
        #    self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.push_interval_s = self.cfg.domain_rand.push_interval_s

        self.dof_names = self.cfg.asset.dof_names
        self.latency = self.cfg.domain_rand.latency
        self.debug = self.cfg.env.debug

    def _draw_debug_vis(self):
        pass

    def _get_env_origins(self):
        pass

    def _init_height_points(self):
        pass

    def _get_heights(self, env_ids=None):
        pass

    # TODO: Implement reward methods in LeggedRobot
