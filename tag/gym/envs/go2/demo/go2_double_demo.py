import os
from typing import Optional

import genesis as gs
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
import numpy as np
import torch

from tag.gym.envs.base.legged_robot import LeggedRobot

from .go2_demo_config import Go2DemoCfg


class Go2DoubleDemo(LeggedRobot):
    cfg: Go2DemoCfg
    height_samples: Optional[torch.Tensor]
    debug_viz: bool
    init_done: bool

    def __init__(self, cfg: Go2DemoCfg, sim_device, headless):
        self.cfg = cfg
        self.height_samples = None
        self.debug_viz = self.cfg.env.debug_viz
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_device, headless)

        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def create_sim(self):
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.sim_dt, substeps=self.sim_substeps),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / self.dt * self.cfg.control.decimation),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=min(self.cfg.viewer.num_rendered_envs, self.num_envs)),
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
            self.terrain = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self.terrain.set_friction(self.cfg.terrain.friction)
        self.terrain_x_range = torch.zeros(2, device=self.device)
        self.terrain_y_range = torch.zeros(2, device=self.device)
        self._create_envs()

    def _create_envs(self):
        self.robot_1 = self.scene.add_entity(
            gs.morphs.URDF(
                file=os.path.join("urdf/go2/urdf/go2.urdf"),
                merge_fixed_links=True,  # if merge_fixed_links is True, then one link may have multiple geometries, which will cause error in set_friction_ratio
                links_to_keep=self.cfg.asset.links_to_keep,
                pos=np.array(self.cfg.init_state_1.pos),
                quat=np.array(self.cfg.init_state_1.quat),
                fixed=self.cfg.asset.fix_base_link,
            ),
            visualize_contact=self.debug,
        )

        self.robot_2 = self.scene.add_entity(
            gs.morphs.URDF(
                file=os.path.join("urdf/go2/urdf/go2.urdf"),
                merge_fixed_links=True,  # if merge_fixed_links is True, then one link may have multiple geometries, which will cause error in set_friction_ratio
                links_to_keep=self.cfg.asset.links_to_keep,
                pos=np.array(self.cfg.init_state_2.pos),
                quat=np.array(self.cfg.init_state_2.quat),
                fixed=self.cfg.asset.fix_base_link,
            ),
            visualize_contact=self.debug,
        )

        # build
        self.scene.build(n_envs=self.num_envs)

        self._get_env_origins()

        # name to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.dof_names]

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

        self.termination_indices = find_link_indices(self.cfg.asset.terminate_after_contacts_on)
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

    def step(self, actions):
        self.scene.step()

    def get_observations(self):
        pass

    def get_privledged_observations(self):
        pass

    def reset(self):
        pass
