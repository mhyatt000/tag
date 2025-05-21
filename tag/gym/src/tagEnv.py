# TODO: Fix File Structure
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseClasses import BaseEnv
from tagConfig import TagConfig
import torch
import genesis as gs
from typing import Dict, Any, Tuple

class TagEnv(BaseEnv):
    '''
    Version 1
    '''

    def __init__(self, args, cfg: TagConfig = TagConfig()):
        self.cfg = cfg
        self.args = args
        self.device = gs.gpu # TODO: Fix
        # TODO: Figure out observation space
        self.observation_space = list()
        # TODO: Figure out action space
        self.action_space = list()

        # Init
        gs.init(logging_level = self.args.logging_level, backend = self.device)

        # Scene
        self.scene = gs.Scene(
            show_viewer = False, # False for now
            rigid_options = gs.options.RigidOptions(
                enable_joint_limit = cfg.solver.joint_limit,
                dt = cfg.solver.dt,
            ),
            vis_options = gs.options.VisOptions(
                show_world_frame = cfg.vis.show_world_frame,
                n_rendered_envs = self.args.n_rendered if self.args != None else 1,
			),
        )

        # Entites
        
        # Plane
        # TODO: Terrain Options/Implementation
        if cfg.terrain.mesh_type == 'plane':
            self.terrain = self.scene.add_entity(
                gs.morphs.Plane(),
            )
        
        # Go2
        # TODO: Implement Robot Class
        self.robot_1 = self.scene.add_entity(
            gs.morphs.URDF(
                file = cfg.robotCfg.asset.file,
                pos = torch.tensor(cfg.robotCfg.init_state.pos) + torch.tensor([0.0, -0.5, 0.0]),
            )
        )

        self.robot_2 = self.scene.add_entity(
            gs.morphs.URDF(
                file = cfg.robotCfg.asset.file,
                pos = torch.tensor(cfg.robotCfg.init_state.pos) + torch.tensor([0.0, 0.5, 0.0]),
            )
        )

        # TODO: Implement Camera Class
        if True:
            self.cam = self.scene.add_camera(
                res = (1280, 720),
                pos = (7, 0.0, 2.5),
                lookat = (0, 0, 0.5),
                fov = 60,
                GUI = False,
            )

    def build(self):
        self.scene.build(
            n_envs = self.args.n_envs if self.args != None else 4,
            env_spacing = (4, 4)
        )
        if True:
            self.cam.start_recording()

    # TODO: Implement
    def set_control_gains(self):
        pass

    # TODO: Implement Properly
    def step(self, *actions: Any) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
        self.scene.step()
        if True:
            self.cam.render()
    
    # TODO: What
    def end(self):
        if True:
            self.cam.stop_recording(save_to_filename = "./mp4/tagEnv_video.mp4", fps = 60)
