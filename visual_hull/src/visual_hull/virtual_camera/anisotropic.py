"""Anisotropic (camera-aware) rounding.

The isotropic ball-opening rounds equally in every direction, including the
directions your real cameras actually measure — so it slightly over-trims where
you have data.  This variant rounds **only where there is no measurement**:

1. Compute the isotropic ρ-opening of the hull solid (rounds every facet corner).
2. The corner voxels the opening *removed* are the candidates.  Restore the ones
   whose direction from the centre lies **near a real camera's occluding
   contour** (its surface normal ``n`` is perpendicular to some viewing
   direction ``d_c``, i.e. ``min_c |n.d_c|`` small) — there the silhouette is
   *measured*, so keep the hull material (stay tangent, preserve size).  Leave
   removed the corners pointing into the **gaps between all cameras**
   (``min_c |n.d_c|`` large) — those are the unconstrained, over-estimating box
   corners.

Done on the occupancy (not the mesh), so there is no smoothing-shrink artifact;
the mesh is then fitted and volume-matched exactly like the isotropic path.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import ball

from .carve import VirtualCarveConfig, _solid_occupancy
from .round_surface import RoundedBubble
from ..mesh_surface import fit_mesh_surface, MeshSurfaceConfig


def round_bubble_surface_anisotropic(
        hull_points: np.ndarray, voxel_size, view_dirs: np.ndarray,
        carve: VirtualCarveConfig | None = None,
        mesh: MeshSurfaceConfig | None = None,
        g_thresh: float = 0.30, volume_match: bool = True) -> RoundedBubble | None:
    """Camera-aware rounding: round the gap corners, keep the corners that lie on
    a real camera contour.  ``view_dirs`` = (N,3) real camera axes (see
    ``views.viewing_directions``)."""
    pts = np.asarray(hull_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 8:
        return None
    vsz = float(np.min(np.asarray(voxel_size, dtype=np.float64)))
    cc = carve or VirtualCarveConfig(spacing=max(vsz / 3.0, 1e-3), input_pitch=vsz)
    pitch = float(cc.input_pitch) if cc.input_pitch else vsz
    sp = float(cc.spacing)
    D = np.asarray(view_dirs, dtype=np.float64)
    D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)

    occ, lo = _solid_occupancy(pts, sp, cc.pad, pitch)
    vol = int(occ.sum()) * sp ** 3
    radius = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0) if vol > 0 else 0.0
    rho = cc.rho_mm if cc.rho_mm is not None else cc.rho_frac * radius
    r_open = max(int(round(rho / sp)), 1)
    r_close = max(int(round(cc.close_frac * rho / sp)), 0)

    rounded = ndimage.binary_opening(occ, ball(r_open))
    if not rounded.any():
        rounded = occ.copy()
    removed = occ & ~rounded                      # the shaved corner shell

    center = pts.mean(axis=0)
    ridx = np.argwhere(removed)
    if ridx.size:
        world = lo + ridx * sp
        n = world - center
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        g = np.min(np.abs(n @ D.T), axis=1)       # small = near a camera contour
        restore = g < g_thresh
        rr = ridx[restore]
        rounded[rr[:, 0], rr[:, 1], rr[:, 2]] = True
    final = ndimage.binary_fill_holes(rounded)
    if r_close >= 1:
        final = ndimage.binary_fill_holes(ndimage.binary_closing(final, ball(r_close)))

    fidx = np.argwhere(final)
    if fidx.shape[0] < 8:
        return None
    rounded_pts = lo + fidx * sp
    final_vol = fidx.shape[0] * sp ** 3

    mc = mesh or MeshSurfaceConfig(w_smooth=0.12, w_convex=0.08,
                                   sdf_smooth_mm=sp, sdf_close_mm=2 * sp)
    surface = fit_mesh_surface(rounded_pts, voxel_size=sp, config=mc)
    if surface is None:
        return None
    if volume_match and surface.volume > 0:
        scale = (final_vol / surface.volume) ** (1.0 / 3.0)
        surface.vertices = surface.center + (surface.vertices - surface.center) * scale
        surface.volume = final_vol
        surface.equiv_diameter = 2.0 * (3.0 * final_vol / (4.0 * np.pi)) ** (1.0 / 3.0)
        surface.roughness = surface.roughness / scale

    info = {"rho_mm": float(rho), "g_thresh": g_thresh,
            "restored_frac": float(restore.mean()) if ridx.size else 0.0}
    return RoundedBubble(mesh=surface, rounded_points=rounded_pts,
                         hull_volume=pts.shape[0] * vsz ** 3,
                         rounded_volume=final_vol, info=info)


__all__ = ["round_bubble_surface_anisotropic"]
