"""Free-form deformable-mesh bubble surfaces (alternative to spherical harmonics).

See DESIGN.md.  Public API:

    from visual_hull.mesh_surface import fit_mesh_surface, MeshSurfaceConfig, MeshSurface
"""

from __future__ import annotations

from .fit import MeshSurfaceConfig, MeshSurface, fit_mesh_surface
from .icosphere import icosphere
from .sdf import build_hull_sdf, SignedDistanceField

__all__ = [
    "fit_mesh_surface", "MeshSurfaceConfig", "MeshSurface",
    "icosphere", "build_hull_sdf", "SignedDistanceField",
]
