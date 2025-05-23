import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Provide a stub genesis module so imports succeed without the heavy dependency.
sys.modules.setdefault(
    "genesis",
    SimpleNamespace(
        morphs=SimpleNamespace(Plane=object, Terrain=object),
        options=SimpleNamespace(RigidOptions=object, VisOptions=object),
        Scene=object,
    ),
)
sys.modules.setdefault("numpy", SimpleNamespace())
sys.modules.setdefault("torch", SimpleNamespace())
sys.modules.setdefault("tag.gym.robots.go2", SimpleNamespace(Go2Config=object, Go2Robot=object))

from tag.gym.envs import terrain_mixin
from tag.gym.envs.chase.chase_config import ChaseEnvConfig
from tag.gym.envs.chase.utils import create_scene
from tag.gym.envs.terrain_mixin import TerrainEnvMixin


class DummyScene:
    def __init__(self):
        self.entities = []

    def add_entity(self, entity):
        self.entities.append(entity)
        return entity


class DummyEnv(TerrainEnvMixin):
    def __init__(self, mesh_type: str):
        self.scene = DummyScene()
        self.cfg = SimpleNamespace(terrain=SimpleNamespace(mesh_type=mesh_type))


def test_build_terrain_plane(monkeypatch):
    """Ensure the mixin adds a plane when requested."""
    Plane = type("Plane", (), {})
    Terrain = type("Terrain", (), {})
    stub_gs = SimpleNamespace(morphs=SimpleNamespace(Plane=Plane, Terrain=Terrain))
    monkeypatch.setattr(terrain_mixin, "gs", stub_gs)

    env = DummyEnv("plane")
    env.build_terrain()

    assert isinstance(env.scene.entities[0], Plane)


def test_create_scene(monkeypatch):
    """create_scene should forward options to the stub Scene class."""
    calls = {}

    class Scene:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    class RigidOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class VisOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    stub_gs = SimpleNamespace(
        Scene=Scene,
        options=SimpleNamespace(RigidOptions=RigidOptions, VisOptions=VisOptions),
    )
    import tag.gym.envs.chase.utils as utils

    monkeypatch.setattr(utils, "gs", stub_gs)

    cfg = ChaseEnvConfig()
    scene = create_scene(cfg, 3)

    assert isinstance(scene, Scene)
    assert calls["vis_options"].kwargs["n_rendered_envs"] == 3
