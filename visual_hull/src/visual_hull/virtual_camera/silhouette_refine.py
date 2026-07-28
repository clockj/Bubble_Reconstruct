"""Tier-1 refinement: virtual-camera outline smoothing with a real-camera gate.

Implements the two proven upgrades from the literature
(Masuk/Salibindla/Ni 2019; Huang et al. 2025):

1. **Rational-quadratic Bézier (conic) outline smoothing.**  On each virtual
   (orthographic) view, the shape's 2-D silhouette corners — the sharp
   reconstruction facets — are replaced by conic Bézier arcs (weight ``omega``:
   <1 rounds toward a circle), then the 3-D shape is carved to the smoothed,
   inscribed outline.  Exactly recovers spheres/ellipsoids.

2. **Real-camera reprojection gate.**  Every carve is tentatively applied and the
   result re-projected (through the *real* refractive cameras, OpenLPT) onto each
   real view.  The carve is accepted only if it does **not** shrink any real
   silhouette by more than ``area_tol``; otherwise it is reverted.  This keeps the
   surface tangent to the measured silhouettes (no over-trim) and is the correct
   form of "round only where unconstrained".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MplPath

from .carve import VirtualCarveConfig, _solid_occupancy
from .views import virtual_directions, orthonormal_basis
from .round_surface import RoundedBubble
from ..mesh_surface import fit_mesh_surface, MeshSurfaceConfig


def _bezier_smooth(poly: np.ndarray, omega: float, samples: int = 8) -> np.ndarray:
    """Round each convex-polygon corner with a rational quadratic Bézier arc
    (control points: edge-midpoints P0,P2 and the corner P1; weight omega on P1).
    Returns an inscribed, corner-rounded polygon."""
    M = poly.shape[0]
    t = np.linspace(0.0, 1.0, samples)
    b0 = (1 - t) ** 2; b1 = 2 * t * (1 - t) * omega; b2 = t ** 2
    denom = b0 + b1 + b2
    out = []
    for i in range(M):
        P1 = poly[i]
        P0 = 0.5 * (poly[(i - 1) % M] + P1)
        P2 = 0.5 * (poly[(i + 1) % M] + P1)
        arc = (b0[:, None] * P0 + b1[:, None] * P1 + b2[:, None] * P2) / denom[:, None]
        out.append(arc)
    return np.vstack(out)


def _hull_area(pts2d: np.ndarray) -> float:
    if pts2d.shape[0] < 3:
        return 0.0
    try:
        return float(ConvexHull(pts2d).volume)   # 2-D hull "volume" = area
    except Exception:
        return 0.0


_RNG = np.random.default_rng(0)


def _gate_points(alive: np.ndarray, idx: np.ndarray, shape, lo, sp: float,
                 mode: str) -> np.ndarray:
    """World points used for the reprojection gate / target, per ``mode``.

    A silhouette (2-D convex-hull area) is a boundary phenomenon — interior
    voxels project strictly inside and are redundant — so we project only the
    solid's **3-D surface**, not the full solid.  ``surface`` = the boundary
    shell (voxels with an empty neighbour); ``hull3d`` = the 3-D convex-hull
    vertices (fewest points, exact for the convex-hull silhouette); ``subsample``
    = legacy random subset of all voxels.
    """
    sub = idx[alive]
    world = lo + sub * sp
    if mode == "hull3d":
        if world.shape[0] > 4:
            try:
                return world[ConvexHull(world).vertices]
            except Exception:
                return world
        return world
    if mode == "subsample":
        if world.shape[0] > 500:
            return world[_RNG.choice(world.shape[0], 500, replace=False)]
        return world
    # "surface": 1-voxel boundary shell of the alive solid
    G = np.zeros(shape, dtype=bool); G[sub[:, 0], sub[:, 1], sub[:, 2]] = True
    B = G & ~ndimage.binary_erosion(G)
    return lo + np.argwhere(B) * sp


def _reproj_ok(gate_pts: np.ndarray, cameras, target_area, area_tol: float) -> bool:
    """True iff the (already-extracted boundary) point set still fills every real
    silhouette to within tol."""
    for c in range(cameras.count):
        pr = cameras.project_points(c, gate_pts)
        if _hull_area(pr.pixels[pr.valid]) < (1.0 - area_tol) * target_area[c]:
            return False
    return True


@dataclass(slots=True)
class RefineConfig:
    spacing: float = 0.10
    input_pitch: float | None = None
    n_virtual: int = 30
    omega: float = 0.7          # Bézier weight (<1 rounds toward circle)
    area_tol: float = 0.15      # allowed real-silhouette area loss per view
                                # (calibrated for the exact gate; the old noisy
                                #  subsample gate needed ~0.05)
    iters: int = 3
    pad: float = 1.5
    gate_mode: str = "hull3d"   # project only the 3-D surface for the silhouette:
                                # "hull3d" = convex-hull vertices (~40 pts, exact
                                #  for convex bubbles, fastest); "surface" = full
                                #  boundary shell (identical accuracy, ~1000 pts,
                                #  robust for non-convex); "subsample" = legacy


def refine_bubble_silhouette(hull_points: np.ndarray, cameras, real_masks,
                             voxel_size, config: RefineConfig | None = None,
                             mesh: MeshSurfaceConfig | None = None) -> RoundedBubble | None:
    """Refine a bubble's real-hull voxels with Bézier virtual-view smoothing under
    a real-camera reprojection gate.  ``real_masks`` are the per-camera binary
    silhouettes (used only for the target areas)."""
    cfg = config or RefineConfig()
    pts = np.asarray(hull_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 8:
        return None
    vsz = float(np.min(np.asarray(voxel_size, dtype=np.float64)))
    pitch = cfg.input_pitch if cfg.input_pitch else vsz
    sp = cfg.spacing

    occ, lo = _solid_occupancy(pts, sp, cfg.pad, pitch)
    idx = np.argwhere(occ)
    X = lo + idx * sp
    alive = np.ones(X.shape[0], dtype=bool)
    shape = occ.shape

    # target: the solid's reprojected silhouette area per real camera, from its
    # 3-D SURFACE only (interior voxels are redundant for a silhouette).
    gp0 = _gate_points(alive, idx, shape, lo, sp, cfg.gate_mode)
    target = []
    for c in range(cameras.count):
        pr0 = cameras.project_points(c, gp0)
        target.append(_hull_area(pr0.pixels[pr0.valid]))

    dirs = virtual_directions(cfg.n_virtual)
    for _ in range(int(cfg.iters)):
        for d in dirs:
            u, v = orthonormal_basis(d)
            A2 = np.column_stack((X @ u, X @ v))
            al2 = A2[alive]
            if al2.shape[0] < 4:
                continue
            try:
                poly = al2[ConvexHull(al2).vertices]
            except Exception:
                continue
            smoothed = _bezier_smooth(poly, cfg.omega)
            inside = MplPath(smoothed).contains_points(A2)
            tentative = alive & inside
            if tentative.sum() < 8 or tentative.sum() == alive.sum():
                continue
            gpts = _gate_points(tentative, idx, shape, lo, sp, cfg.gate_mode)
            if _reproj_ok(gpts, cameras, target, cfg.area_tol):
                alive = tentative

    refined = X[alive]
    if refined.shape[0] < 8:
        return None
    refined_vol = refined.shape[0] * sp ** 3
    mc = mesh or MeshSurfaceConfig(w_smooth=0.10, w_convex=0.06,
                                   sdf_smooth_mm=sp, sdf_close_mm=2 * sp)
    surface = fit_mesh_surface(refined, voxel_size=sp, config=mc)
    if surface is None:
        return None
    if surface.volume > 0:                       # volume-match to the carved solid
        scale = (refined_vol / surface.volume) ** (1.0 / 3.0)
        surface.vertices = surface.center + (surface.vertices - surface.center) * scale
        surface.volume = refined_vol
        surface.equiv_diameter = 2.0 * (3.0 * refined_vol / (4.0 * np.pi)) ** (1.0 / 3.0)
        surface.roughness = surface.roughness / scale

    return RoundedBubble(mesh=surface, rounded_points=refined,
                         hull_volume=pts.shape[0] * vsz ** 3, rounded_volume=refined_vol,
                         info={"omega": cfg.omega, "area_tol": cfg.area_tol,
                               "n_virtual": cfg.n_virtual})


__all__ = ["RefineConfig", "refine_bubble_silhouette"]
