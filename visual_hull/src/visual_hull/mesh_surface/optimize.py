"""Explicit mesh evolution: sum the forces, clamp the step, iterate.

A deliberately simple, dependency-light integrator.  Smoothness weight is
annealed from strong to weak so the surface first finds the coarse shape
(avoiding the local minima / "mode trapping" that hurt spherical harmonics),
then relaxes onto the data.
"""

from __future__ import annotations

import numpy as np

from .energy import data_force, smooth_force, convex_force
from .mesh_ops import build_adjacency


def evolve_mesh(vertices, faces, sdf, *, iterations=150, lr=0.5,
                w_data=1.0, w_smooth=0.15, w_convex=0.1, target_sdf=0.0,
                smooth_anneal=(3.0, 1.0), step_clamp=None, record=False):
    """Evolve ``vertices`` under data + smoothness + convexity forces.

    ``smooth_anneal`` multiplies ``w_smooth`` linearly from anneal[0] -> anneal[1]
    across the iterations.  ``step_clamp`` caps per-vertex movement per step
    (defaults to the SDF spacing, for stability).
    """
    V = np.asarray(vertices, dtype=np.float64).copy()
    adj = build_adjacency(faces, V.shape[0])
    clamp = float(sdf.spacing if step_clamp is None else step_clamp)
    history = []

    a0, a1 = smooth_anneal
    for it in range(int(iterations)):
        frac = it / max(int(iterations) - 1, 1)
        smooth_scale = a0 + (a1 - a0) * frac

        f_data, e_data = data_force(sdf, V, target_sdf)
        f_smooth, e_smooth = smooth_force(V, adj)
        f_convex, e_convex = convex_force(V, faces, adj)

        F = (w_data * f_data
             + w_smooth * smooth_scale * f_smooth
             + w_convex * f_convex)
        step = lr * F
        # clamp per-vertex step length for stability
        mag = np.linalg.norm(step, axis=1, keepdims=True)
        too_big = (mag > clamp).ravel()
        step[too_big] *= (clamp / mag[too_big])
        V += step

        if record:
            history.append({"iter": it, "e_data": e_data,
                            "e_smooth": e_smooth, "e_convex": e_convex,
                            "max_step": float(np.max(np.linalg.norm(step, axis=1)))})
    return V, history


__all__ = ["evolve_mesh"]
