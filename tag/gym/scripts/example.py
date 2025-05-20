import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import genesis as gs
import numpy as np

### init
gs.init(seed=0, precision="32", logging_level="debug")

### scene
scene = gs.Scene(
    show_viewer=False,
    rigid_options=gs.options.RigidOptions(
        enable_joint_limit=False,
        enable_collision=False,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=1.0,
        show_link_frame=False,
        show_cameras=False,
        plane_reflection=True,
        ambient_light=(0.1, 0.1, 0.1),
    ),
)

### entities
scene.add_entity(
    gs.morphs.Plane(),
)
robot = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

### Target links for visualization
target_left = scene.add_entity(
    gs.morphs.Mesh(
        file="meshes/axis.obj",
        scale=0.1,
    ),
    surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
)
target_right = scene.add_entity(
    gs.morphs.Mesh(
        file="meshes/axis.obj",
        scale=0.1,
    ),
    surface=gs.surfaces.Default(color=(0.5, 1.0, 0.5, 1)),
)

### camera
cam = scene.add_camera(
    res=(640, 480),
    pos=(3.5, 0.0, 2.5),
    lookat=(0, 0, 0.5),
    fov=30,
    GUI=False,
)

### build

scene.build()

### target positions
# orient the fingers downwards
target_quat = np.array([0, 1, 0, 0])
# center of the circle to be traced
center = np.array([0.4, -0.2, 0.25])
# radius of the circle
r = 0.1

# Links
left_finger = robot.get_link("left_finger")
right_finger = robot.get_link("right_finger")

### activate camera
cam.start_recording()

### Movement
for i in range(0, 2000):
    target_pos_left = (
        center + np.array([np.cos(i / 360 * np.pi), np.sin(i / 360 * np.pi), 0]) * r
    )
    target_pos_right = target_pos_left + np.array([0.0, 0.03, 0])

    target_left.set_qpos(np.concatenate([target_pos_left, target_quat]))
    target_right.set_qpos(np.concatenate([target_pos_right, target_quat]))

    q = robot.inverse_kinematics_multilink(
        links=[left_finger, right_finger],
        poss=[target_pos_left, target_pos_right],
        quats=[target_quat, target_quat],
        rot_mask=[False, False, True],  # only restrict direction of z-axis
    )

    robot.set_dofs_position(q)
    scene.visualizer.update()
    cam.render()

### Saving Video
cam.stop_recording(save_to_filename="./mp4/user_guide/advanced_ik_video.mp4", fps=60)
