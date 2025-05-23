"""Utility mixin for adding terrain support to environments."""

import genesis as gs


class TerrainEnvMixin:
    """Mixin to add terrain creation helpers for environments."""

    def _init_terrain(self) -> None:
        """Create terrain entity based on ``self.cfg.terrain``.

        Subclasses are expected to define ``self.scene`` and ``self.cfg`` with a
        ``terrain`` attribute specifying the mesh type.
        """
        mesh_type = getattr(self.cfg.terrain, "mesh_type", "plane")
        if mesh_type == "plane":
            self.terrain = self.scene.add_entity(gs.morphs.Plane())
        elif mesh_type == "heightfield":
            self.terrain = self.scene.add_entity(gs.morphs.Terrain())
        else:
            raise ValueError(f"Unsupported terrain mesh type: {mesh_type}")
