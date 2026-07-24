"""High-level API: fit a free-form mesh surface to a bubble's voxels.

    from visual_hull.mesh_surface import fit_mesh_surface, MeshSurfaceConfig
    mesh = fit_mesh_surface(bubble_voxels, voxel_size)   # -> MeshSurface

The mesh has fixed icosphere topology (shared across bubbles/frames), so the
resulting ``vertices`` arrays are directly comparable frame to frame for
temporal filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .icosphere import icosphere
from .sdf import build_hull_sdf
from .optimize import evolve_mesh
from .mesh_ops import build_adjacency, mean_curvature, mesh_volume


@dataclass(slots=True)
class MeshSurfaceConfig:
    subdivisions: int = 3          # icosphere level (642 verts @3, 2562 @4)
    sdf_spacing: float | None = None  # mm; default = min(voxel_size)/2
    iterations: int = 150
    lr: float = 0.5
    w_data: float = 1.0
    w_smooth: float = 0.15
    w_convex: float = 0.10
    smooth_anneal: tuple[float, float] = (3.0, 1.0)
    target_sdf: float = 0.0        # >0 shrinks inward (bias correction)
    sdf_smooth_mm: float | None = None  # Gaussian SDF smoothing; default = spacing
    sdf_close_mm: float | None = None   # morphological close; default = 2*voxel_size


@dataclass(slots=True)
class MeshSurface:
    vertices: np.ndarray
    faces: np.ndarray
    center: np.ndarray
    volume: float
    equiv_diameter: float
    concavity_fraction: float
    roughness: float
    sdf_rmse: float
    iterations: int
    history: list = field(default_factory=list)


def fit_mesh_surface(interior_points: np.ndarray, voxel_size,
                     config: MeshSurfaceConfig | None = None,
                     record: bool = False) -> MeshSurface | None:
    """Fit a deformable icosphere mesh to a bubble's refined voxels.

    ``interior_points`` — the bubble's refined surface/interior voxels (world mm).
    ``voxel_size`` — reconstruction voxel size (mm), scalar or (3,).
    """
    cfg = config or MeshSurfaceConfig()
    pts = np.asarray(interior_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 8:
        return None
    vsz = float(np.min(np.asarray(voxel_size, dtype=np.float64)))
    spacing = cfg.sdf_spacing if cfg.sdf_spacing else max(vsz * 0.5, 1e-3)
    smooth_mm = cfg.sdf_smooth_mm if cfg.sdf_smooth_mm is not None else spacing
    close_mm = cfg.sdf_close_mm if cfg.sdf_close_mm is not None else 2.0 * vsz

    sdf = build_hull_sdf(pts, spacing, smooth_mm=smooth_mm, close_mm=close_mm)
    center = pts.mean(axis=0)
    # Initialize the icosphere at the mean surface radius (safely inside/near).
    v0, faces = icosphere(cfg.subdivisions)
    r_init = float(np.median(np.linalg.norm(pts - center, axis=1)))
    vertices = center + v0 * max(r_init, 2.0 * spacing)

    vertices, history = evolve_mesh(
        vertices, faces, sdf,
        iterations=cfg.iterations, lr=cfg.lr,
        w_data=cfg.w_data, w_smooth=cfg.w_smooth, w_convex=cfg.w_convex,
        target_sdf=cfg.target_sdf, smooth_anneal=cfg.smooth_anneal,
        record=record,
    )

    vol = mesh_volume(vertices, faces)
    deq = 2.0 * (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
    adj = build_adjacency(faces, vertices.shape[0])
    kappa = mean_curvature(vertices, faces, adj)
    concavity = float(np.mean(kappa < -1e-6 * max(np.abs(kappa).max(), 1e-9)))
    roughness = float(np.sqrt(np.mean(kappa ** 2)))
    sdf_rmse = float(np.sqrt(np.mean(sdf.sample(vertices) ** 2)))

    return MeshSurface(
        vertices=vertices, faces=faces, center=center,
        volume=vol, equiv_diameter=deq,
        concavity_fraction=concavity, roughness=roughness,
        sdf_rmse=sdf_rmse, iterations=cfg.iterations, history=history,
    )


__all__ = ["MeshSurfaceConfig", "MeshSurface", "fit_mesh_surface"]
