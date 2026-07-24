"""High-level API: round a bubble's hull, then extract a clean mesh surface.

    from visual_hull.virtual_camera import round_bubble_surface
    rb = round_bubble_surface(bubble_hull_voxels, voxel_size)
    rb.mesh.vertices, rb.mesh.faces   # fixed-topology icosphere surface
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .carve import VirtualCarveConfig, round_hull_occupancy
from ..mesh_surface import fit_mesh_surface, MeshSurfaceConfig, MeshSurface


@dataclass(slots=True)
class RoundedBubble:
    mesh: MeshSurface
    rounded_points: np.ndarray
    hull_volume: float
    rounded_volume: float
    info: dict


def round_bubble_surface(hull_points: np.ndarray, voxel_size,
                         carve: VirtualCarveConfig | None = None,
                         mesh: MeshSurfaceConfig | None = None,
                         volume_match: bool = True) -> RoundedBubble | None:
    """Virtual-camera-round a bubble's 4-view hull, then fit an icosphere mesh.

    ``hull_points`` — the bubble's real-hull voxel centres (world mm).
    ``voxel_size``  — reconstruction voxel size (mm); sets the carve resolution.
    """
    pts = np.asarray(hull_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 8:
        return None
    vsz = float(np.min(np.asarray(voxel_size, dtype=np.float64)))

    cc = carve or VirtualCarveConfig(spacing=max(vsz / 3.0, 1e-3), input_pitch=vsz)
    if cc.input_pitch is None:
        cc.input_pitch = vsz
    rounded, info = round_hull_occupancy(pts, cc)
    if rounded.shape[0] < 8:
        return None

    # the rounded occupancy is already smooth → let the mesh conform tightly
    mc = mesh or MeshSurfaceConfig(
        subdivisions=3, iterations=150, w_smooth=0.12, w_convex=0.08,
        sdf_smooth_mm=cc.spacing, sdf_close_mm=2.0 * cc.spacing,
    )
    surface = fit_mesh_surface(rounded, voxel_size=cc.spacing, config=mc)
    if surface is None:
        return None

    hull_vol = pts.shape[0] * (vsz ** 3)
    rounded_vol = rounded.shape[0] * (cc.spacing ** 3)

    # The carve carries the SIZE (volume-preserving open/close); the mesh carries
    # the SHAPE but its Laplacian smoothing shrinks it ~10%.  Rescale the mesh
    # about its centre so its volume matches the rounded occupancy.
    if volume_match and surface.volume > 0:
        scale = (rounded_vol / surface.volume) ** (1.0 / 3.0)
        surface.vertices = surface.center + (surface.vertices - surface.center) * scale
        surface.volume = rounded_vol
        surface.equiv_diameter = 2.0 * (3.0 * rounded_vol / (4.0 * np.pi)) ** (1.0 / 3.0)
        surface.roughness = surface.roughness / scale   # curvature scales as 1/length
    return RoundedBubble(mesh=surface, rounded_points=rounded,
                         hull_volume=hull_vol, rounded_volume=rounded_vol, info=info)


__all__ = ["RoundedBubble", "round_bubble_surface"]
