from dataclasses import dataclass, field
import genesis as gs
import numpy as np

def default(x):
    return field(default_factory=lambda: x)


gs.init(backend=gs.gpu)

scene = gs.Scene(
    show_viewer = False,
    viewer_options = gs.options.ViewerOptions(
        res           = (640, 480),
        camera_pos    = (3.5, 0.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = True,
        world_frame_size = 1.0,
        show_link_frame  = False,
        show_cameras     = False,
        plane_reflection = True,
        ambient_light    = (0.1, 0.1, 0.1),
    ),
    renderer=gs.renderers.Rasterizer(),
)

plane = scene.add_entity(
    gs.morphs.Plane()
)

class TerrainManager:
    def __init__(self, n:int, size:float, z_offset:float, horizontal_scale:float, vertical_scale:float, subterrain_types:list):
        self.n = n
        self.size = size
        self.z_offset = z_offset
        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale
        self.subterrain_types = subterrain_types

        pos = -1/2 * size * n
        self.pos = (pos, pos, z_offset)
        self.generate_terrain()

    def generate_terrain(self):
        st_types_2d_arr = [[None for _ in range(self.n)] for _ in range(self.n)]

        for i in range(self.n):
            for j in range(self.n):
                random_terrain = self.subterrain_types[np.random.randint(len(self.subterrain_types))] # grab a random subterrain from the list of subterrains

                st_types_2d_arr[i][j] = random_terrain

        self.result = gs.morphs.Terrain(
            n_subterrains = (self.n, self.n),
            subterrain_size = (self.size, self.size),
            horizontal_scale = self.horizontal_scale,
            vertical_scale = self.vertical_scale,
            pos = self.pos,
            subterrain_types = st_types_2d_arr
        )
        return self.result
    
    def terrain(self):
        return self.result

n, size, z_off = 2, 5.0, 10
# terrain = scene.add_entity(
#     gs.morphs.Terrain(
#         n_subterrains=(n,n),
#         subterrain_size=(size, size),
#         horizontal_scale=0.05,
#         vertical_scale=0.01,
#         pos=(-size, -size, z_off),
#         subterrain_types = [
#             ["flat_terrain", "random_uniform_terrain"],
#             ["pyramid_sloped_terrain", "discrete_obstacles_terrain"]
#         ]
#     )
# )

subterrain_types = ["flat_terrain", "random_uniform_terrain", "pyramid_sloped_terrain", "discrete_obstacles_terrain"]

test_terrain = TerrainManager(
    n, size, z_off, 0.05, 0.01, subterrain_types
)

terrain = scene.add_entity(
    test_terrain.terrain()
)

cam = scene.add_camera(
    res    = (640, 480),
    pos    = (3.5, 0.0, 2.5+z_off),
    lookat = (0, 0, 0.5+z_off),
    fov    = 30,
    GUI    = False,
)

scene.build()

# render rgb, depth, segmentation, and normal
# rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True)

cam.start_recording()
import numpy as np

for i in range(400):
    scene.step()
    cam.set_pose(
        pos    = (10.0 * np.sin(i / 60), 10.0 * np.cos(i / 60), 2+z_off),
        lookat = (0, 0, 0.5+z_off),
    )
    cam.render()
cam.stop_recording(save_to_filename='video.mp4', fps=30)