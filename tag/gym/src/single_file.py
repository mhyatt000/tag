import argparse
import sys

import genesis as gs
import torch


def main(args : argparse.Namespace):

	# Init
	gs.init(seed = 1, logging_level = args.logging_level, backend = gs.gpu)

	# Scene
	scene = gs.Scene(
			show_viewer = False,
			rigid_options = gs.options.RigidOptions(
                enable_joint_limit	= True,
                dt 					= 0.02, # 50hz
			),
			vis_options = gs.options.VisOptions(
                show_world_frame = True,
                world_frame_size = 1.0,
                show_link_frame = False,
                show_cameras = False,
                plane_reflection = True,
                ambient_light = (0.1, 0.1, 0.1),
                n_rendered_envs = args.n_rendered,
			),
	)

	# Entities

	# Plane
	plane = scene.add_entity(
		gs.morphs.Plane(), # Default Plane
	)
	# Go2 Robots
	robot_1 = scene.add_entity(
		gs.morphs.URDF(
			file = "urdf/go2/urdf/go2.urdf", 
			pos = (0.0, -0.5, 0.42),
		),
	)

	robot_2 = scene.add_entity(
		gs.morphs.URDF(
			file = "urdf/go2/urdf/go2.urdf",
			pos = (0.0, 0.5, 0.42),
		),
	)

	# Camera
	cam = scene.add_camera(
		res = (1280, 720),
		pos = (7, 0.0, 2.5),
		lookat = (0, 0, 0.5),
		fov = 60,
		GUI = False,
	)

	# Build
	scene.build(
		n_envs = args.n_envs,
		env_spacing = (4, 4)
	)

	# Settings dofs
	joint_names = [
		'FL_hip_joint',
		'RL_hip_joint',
		'FR_hip_joint',
		'RR_hip_joint',

		'FL_thigh_joint',
		'RL_thigh_joint',
		'FR_thigh_joint',
		'RR_thigh_joint',

		'FL_calf_joint',
		'RL_calf_joint',
		'FR_calf_joint',
		'RR_calf_joint'
	]

	local_dofs = [robot_1.get_joint(name).dof_idx_local for name in joint_names]

	# Activate Camera and Step Through Time
	cam.start_recording()

	for t in range(0, 600):
		# robot_1 will uncrouch repeatedly
		# robot_2 will crouch repeatedly
		if t % 60 == 0:
			if t != 0 and t % 120 != 0:
				robot_1.control_dofs_position(
					position = torch.tile(torch.tensor([0.1, 0.1, -0.1, -0.1, 0.6, 0.8, 0.6, 0.8, -1, -1, -1, -1], device = gs.device), (args.n_envs, 1)),
					dofs_idx_local = local_dofs,
				)
				robot_2.control_dofs_position(
					position = torch.tile(torch.tensor([0.1, 0.1, -0.1, -0.1, 1, 1.2, 1, 1.2, -1.7, -1.7, -1.7, -1.7], device = gs.device), (args.n_envs, 1)),
					dofs_idx_local = local_dofs,
				)
			else:
				robot_1.control_dofs_position(
					position = torch.tile(torch.tensor([0.1, 0.1, -0.1, -0.1, 0.8, 1.0, 0.8, 1.0, -1.5, -1.5, -1.5, -1.5], device = gs.device), (args.n_envs, 1)),
					dofs_idx_local = local_dofs,
				)
				robot_2.control_dofs_position(
					position = torch.tile(torch.tensor([0.1, 0.1, -0.1, -0.1, 0.8, 1.0, 0.8, 1.0, -1.5, -1.5, -1.5, -1.5], device = gs.device), (args.n_envs, 1)),
					dofs_idx_local = local_dofs,
				)
		robot_1.control_dofs_force( # Send instruction to only the first environment's robot_1
			force = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device = gs.device),
			dofs_idx_local = local_dofs,
			envs_idx = torch.tensor([0], device = gs.gpu)
		)
		scene.step()
		cam.render()

	# Save Recording
	cam.stop_recording(save_to_filename = "./mp4/single_file_video.mp4")


def get_args():
        '''
        Some basic arguments as an example
        Many are excluded
        '''
        parser = argparse.ArgumentParser()
        parser.add_argument('--logging_level',          type = str, default = 'warning')
        parser.add_argument('-B', '--n_envs',           type = int, default = 1)
        parser.add_argument('--n_rendered',       		type = int, default = 1)
        return parser.parse_args()

def check_args(args):

	logging_levels = [
			'debug',
			'info',
			'warning'
	]
	maximum_envs = 20000
	maximum_rendered_envs = 32

	# Check logging_level
	if args.logging_level not in logging_levels:
		sys.exit("ARGUMENT ERROR: LOGGING_LEVEL: Invalid logging_level parsed.")

	# Check n_envs
	if args.n_envs == 0: 
		args.n_envs = 1
	if args.n_envs > maximum_envs:
		sys.exit("ARGUMENT ERROR: N_ENVS: Exceeds maximum environments.")
	if args.n_rendered > maximum_rendered_envs:
		sys.exit("ARGUMENT ERROR: N_RENDERED: Exceeds maximum rendered environments.")
	if args.n_rendered > args.n_envs:
		sys.exit("ARGUMENT ERROR: N_RENDERED: Rendered environments exceeds number of environments.")

	return True

if __name__ == '__main__':
	args = get_args()
	check_args(args)
	main(args)
	