"""Virtual-camera silhouette rounding of a visual hull (the core method).

Given the bubble's real 4-view hull voxels, we round off the reconstruction
facets so the boxy, over-sized hull becomes a smooth bubble.

**Duality used here.**  Rounding every virtual-camera silhouette (open each
orthographic silhouette by a disk of radius ``rho``, then re-carve) is, for a
convex body, equivalent to a single **3-D morphological opening of the solid by a
ball of radius ``rho``**: "roll a ball of radius ``rho`` around the inside of the
surface — wherever it cannot reach (sharp convex facet corners) is shaved off".
Opening is *erode-then-dilate*, so flat/smooth faces return to their original
position (no net shrink there) while only corners/edges sharper than ``rho`` are
rounded.  We implement the robust volumetric form; ``views.py`` documents the
equivalent silhouette construction.

A following **closing** by ``rho`` rounds any concave notches (e.g. left by
bubble separation), giving a full open-then-close morphological smoothing.

``rho`` is the only shape knob (smallest surface-feature radius kept).  Because
opening only removes the over-hanging corners, it also pulls the hull's
+6 %/+19 % over-estimate down toward truth; ``rho`` is calibrated once on a
synthetic sphere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.morphology import ball


@dataclass(slots=True)
class VirtualCarveConfig:
    spacing: float = 0.16          # mm; carve resolution (≈ voxel/3 @0.5)
    input_pitch: float | None = None  # mm; source hull voxel pitch (for solid fill)
    rho_frac: float = 0.30         # rounding radius as a fraction of bubble radius
    rho_mm: float | None = None    # absolute rounding radius (overrides rho_frac)
    close_frac: float = 0.6        # concave-notch rounding = close_frac * rho
    pad: float = 3.0               # mm margin around the point cloud
    # legacy silhouette-carve knobs (kept for API compatibility; unused here)
    n_views: int = 0
    iters: int = 0


def _solid_occupancy(points: np.ndarray, spacing: float, pad: float,
                     input_pitch: float):
    """Rasterize each hull voxel as its actual cube (side = source pitch) so the
    cubes tile exactly into a dense, **volume-preserving** solid, then fill any
    interior gaps.  Returns (solid grid, origin)."""
    lo = points.min(axis=0) - pad
    hi = points.max(axis=0) + pad
    dims = np.ceil((hi - lo) / spacing).astype(int) + 1
    occ = np.zeros(tuple(dims), dtype=bool)
    ijk = np.clip(np.round((points - lo) / spacing).astype(int), 0, np.array(dims) - 1)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    # expand each point to a cube of side k = pitch/spacing → tiles with no gaps
    k = max(int(round(input_pitch / spacing)), 1)
    if k > 1:
        occ = ndimage.binary_dilation(occ, np.ones((k, k, k), dtype=bool))
    occ = ndimage.binary_fill_holes(occ)
    return occ, lo


def round_hull_occupancy(points: np.ndarray, config: VirtualCarveConfig | None = None):
    """Round a bubble's real-hull voxels via morphological open-then-close
    (the volumetric form of virtual-camera silhouette rounding).

    ``points`` — real 4-view hull voxel centres (world mm).
    Returns ``(rounded_points (M,3), info)``.
    """
    cfg = config or VirtualCarveConfig()
    pts = np.asarray(points, dtype=np.float64)
    sp = float(cfg.spacing)
    pitch = float(cfg.input_pitch) if cfg.input_pitch else max(2.0 * sp, sp)

    occ, lo = _solid_occupancy(pts, sp, cfg.pad, pitch)
    solid_before = int(occ.sum())
    vol = solid_before * sp ** 3
    radius = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0) if vol > 0 else 0.0
    rho = cfg.rho_mm if cfg.rho_mm is not None else cfg.rho_frac * radius
    r_open = max(int(round(rho / sp)), 1)
    r_close = max(int(round(cfg.close_frac * rho / sp)), 0)

    # open → shave sharp convex facet corners (roll a ball of radius rho inside)
    rounded = ndimage.binary_opening(occ, ball(r_open))
    if not rounded.any():                     # guard against too-large rho
        rounded = occ
    # close → round concave notches (e.g. from separation)
    if r_close >= 1:
        rounded = ndimage.binary_closing(rounded, ball(r_close))
        rounded = ndimage.binary_fill_holes(rounded)

    idx = np.argwhere(rounded)
    X = lo + idx * sp
    info = {"rho_mm": float(rho), "r_open_px": r_open, "r_close_px": r_close,
            "spacing": sp, "hull_radius_mm": float(radius),
            "solid_voxels": solid_before, "rounded_voxels": int(idx.shape[0])}
    return X, info


__all__ = ["VirtualCarveConfig", "round_hull_occupancy"]
