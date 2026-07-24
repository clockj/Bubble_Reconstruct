"""Synthetic virtual viewpoints for silhouette rounding.

These are *orthographic* directions used only to impose the smoothness/convexity
prior (round each self-silhouette and re-carve).  They are NOT real cameras and
do NOT use OpenLPT — the real refractive cameras already produced the input hull.
"""

from __future__ import annotations

import numpy as np


def virtual_directions(n: int) -> np.ndarray:
    """``n`` ~uniform unit view directions on the sphere (Fibonacci lattice)."""
    i = np.arange(int(n), dtype=np.float64) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / int(n))
    golden = np.pi * (1.0 + 5.0 ** 0.5)
    az = golden * i
    return np.column_stack((np.sin(polar) * np.cos(az),
                            np.sin(polar) * np.sin(az),
                            np.cos(polar)))


def viewing_directions(cameras, center: np.ndarray, eps: float = 0.05) -> np.ndarray:
    """Estimate each real camera's optical (depth) axis at ``center`` (N,3 unit).

    Uses only OpenLPT projection (the authoritative model): the depth direction is
    the world direction that moves the 3-D point but barely moves the pixel — i.e.
    the right singular vector of the 2x3 projection Jacobian with the smallest
    singular value.  Sign is arbitrary (callers use |n.d|).
    """
    X0 = np.asarray(center, dtype=np.float64)
    out = []
    for c in range(cameras.count):
        J = np.zeros((2, 3))
        for i in range(3):
            off = np.zeros(3); off[i] = eps
            p1 = cameras.project_points(c, (X0 + off)[None]).pixels[0]
            p0 = cameras.project_points(c, (X0 - off)[None]).pixels[0]
            J[:, i] = (p1 - p0) / (2.0 * eps)
        _, _, Vt = np.linalg.svd(J)
        axis = Vt[-1]
        out.append(axis / max(np.linalg.norm(axis), 1e-12))
    return np.asarray(out)


def orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors (u, v) spanning the plane perpendicular to ``direction``."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / max(np.linalg.norm(d), 1e-12)
    # pick the world axis least aligned with d to avoid degeneracy
    helper = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(d, helper); u /= max(np.linalg.norm(u), 1e-12)
    v = np.cross(d, u)
    return u, v


__all__ = ["virtual_directions", "orthonormal_basis"]
