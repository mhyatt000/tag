import genesis as gs

# sample implementation for incorporating colors into robots

colors = {
    'blue':(0,0,1),
    'red':(1,0,0),
    'green':(0,1,0),
    'yellow':(1,1,0),
    'orange':(1,0.64,0),
    'purple':(1,0,1)
}

class ColoredSurface(gs.options.surfaces.Plastic):
    def __init__(self, color:str):
        super().__init__(diffuse_texture=gs.options.textures.ColorTexture(color=colors[color]))

def getTextures(n:int):
    colorwheel = [
        ColoredSurface('blue'),
        ColoredSurface('red'),
        ColoredSurface('green'),
        ColoredSurface('yellow'),
        ColoredSurface('orange'),
        ColoredSurface('purple')
    ]
    result = []

    for i in range(n):
        result.append(colorwheel[i%len(colorwheel)])

    return result



# sample code for implementing this

import numpy as np

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

z_off = 0

bot = scene.add_entity(
    gs.morphs.URDF(
        file="urdf/go2/urdf/go2.urdf",
        pos=(0.0, 0.0, z_off+0.5)
    ),
    surface=getTextures(6)[np.random.randint(6)] # get a randomly-colored texture -- definitely not how it will actually be implemented, just showcasing the getTextures method
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
        pos    = (10.0 * np.sin(i / 100), 10.0 * np.cos(i / 100), 2+z_off),
        lookat = (0, 0, 0.5+z_off),
    )
    cam.render()
cam.stop_recording(save_to_filename='video_ct.mp4', fps=60)