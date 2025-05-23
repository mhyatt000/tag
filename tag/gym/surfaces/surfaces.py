import genesis as gs

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