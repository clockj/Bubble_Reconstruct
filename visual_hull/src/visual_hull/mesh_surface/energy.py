"""Energy / force terms for mesh evolution.

Each term returns a per-vertex **force** (the descent direction of its energy)
plus a scalar energy for monitoring.  The optimizer sums the forces and steps
the vertices — a standard active-surface ("snake") formulation that is robust
and avoids building sparse operators.

Sign convention: forces are added to the vertices (``v += lr * F``).
"""

from __future__ import annotations

import numpy as np

from .mesh_ops import uniform_laplacian, vertex_normals, mean_curvature


def data_force(sdf, vertices: np.ndarray, target: float = 0.0):
    """Pull each vertex onto the ``sdf == target`` isosurface (the hull boundary).

    F = -(sdf(v) - target) * ∇sdf   (inward if the vertex is beyond the surface,
    outward if it is inside).  ∇sdf is unit-ish, pointing outward.
    """
    s = sdf.sample(vertices)
    g = sdf.gradient(vertices)
    gn = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
    force = -(s - target)[:, None] * gn
    energy = 0.5 * float(np.mean((s - target) ** 2))
    return force, energy


def smooth_force(vertices: np.ndarray, adjacency):
    """Umbrella smoothing force L v (toward the local neighbor centroid).

    Descent direction of the bending/roughness energy; ‖L v‖ is a curvature
    proxy.  This is *local* — it smooths only where the surface is rough, unlike
    a global spectral penalty.
    """
    lap = uniform_laplacian(vertices, adjacency)
    energy = 0.5 * float(np.mean(np.sum(lap ** 2, axis=1)))
    return lap, energy


def convex_force(vertices: np.ndarray, faces: np.ndarray, adjacency):
    """Push concave vertices outward along the normal to fill dents (P4 prior).

    Concave vertex -> mean_curvature < 0 -> force = |κ| * outward_normal.
    """
    kappa = mean_curvature(vertices, faces, adjacency)
    nrm = vertex_normals(vertices, faces)
    concave = np.maximum(-kappa, 0.0)
    energy = float(np.mean(concave ** 2))
    return concave[:, None] * nrm, energy


__all__ = ["data_force", "smooth_force", "convex_force"]
