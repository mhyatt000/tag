import genesis as gs


# TODO(mbt): Implement Class
# NOTE(dle): Placeholder Class until terrain is implemented
class Terrain:
    def __init__(self, scene: gs.Scene):
        self.terrain = scene.add_entity(gs.morphs.Plane())


# NOTE(dle): I believe this class should have the Obstacle object, we should discuss
