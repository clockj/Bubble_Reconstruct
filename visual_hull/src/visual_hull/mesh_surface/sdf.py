"""Signed distance field of a bubble's visual hull, for the data term.

The visual hull is the intersection of all camera silhouette cones, so its
boundary already encodes the full multi-view constraint.  We rasterize the
bubble's (refined) surface/interior voxels into a local grid and build a signed
distance field: negative inside, positive outside.  The mesh optimizer then just
reads ``sdf(vertex)`` and its gradient to pull vertices onto the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(slots=True)
class SignedDistanceField:
    values: np.ndarray          # (nx, ny, nz), mm; <0 inside, >0 outside
    origin: np.ndarray          # world coord of grid index (0,0,0)
    spacing: float              # mm per voxel

    def sample(self, points: np.ndarray) -> np.ndarray:
        """Trilinearly interpolated SDF at world points (P,)."""
        idx = (np.asarray(points, dtype=np.float64) - self.origin) / self.spacing
        return _trilinear(self.values, idx)

    def gradient(self, points: np.ndarray, eps: float | None = None) -> np.ndarray:
        """Central-difference SDF gradient (world units) at points (P,3)."""
        h = self.spacing * 0.5 if eps is None else eps
        g = np.empty((points.shape[0], 3), dtype=np.float64)
        for d in range(3):
            off = np.zeros(3); off[d] = h
            g[:, d] = (self.sample(points + off) - self.sample(points - off)) / (2.0 * h)
        return g


def build_hull_sdf(interior_points: np.ndarray, spacing: float,
                   pad: float = 4.0, smooth_mm: float = 0.0,
                   close_mm: float = 0.0) -> SignedDistanceField:
    """Build an SDF from a point set treated as occupied samples.

    ``interior_points`` are the bubble's refined surface/interior voxels (world
    mm).  ``spacing`` is the SDF grid resolution (mm) — can be finer than the
    reconstruction voxel size for sub-voxel accuracy.  ``pad`` mm of margin is
    added around the point cloud.  ``close_mm`` morphologically closes gaps
    between sparse points before filling (so a sparse shell becomes solid);
    ``smooth_mm`` Gaussian-smooths the final SDF to remove voxel blockiness (the
    surface then conforms to a smooth boundary rather than a staircase).
    """
    pts = np.asarray(interior_points, dtype=np.float64)
    lo = pts.min(axis=0) - pad
    hi = pts.max(axis=0) + pad
    dims = np.ceil((hi - lo) / spacing).astype(int) + 1
    occ = np.zeros(tuple(dims), dtype=bool)
    ijk = np.round((pts - lo) / spacing).astype(int)
    ijk = np.clip(ijk, 0, np.array(dims) - 1)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    # Close gaps between sparse points, then fill the interior solid.
    close_iter = max(int(round(close_mm / spacing)), 1)
    occ = ndimage.binary_closing(occ, iterations=close_iter)
    occ = ndimage.binary_fill_holes(occ)

    dist_out = ndimage.distance_transform_edt(~occ) * spacing
    dist_in = ndimage.distance_transform_edt(occ) * spacing
    sdf = (dist_out - dist_in).astype(np.float64)
    if smooth_mm > 0.0:
        sdf = ndimage.gaussian_filter(sdf, sigma=smooth_mm / spacing)
    return SignedDistanceField(values=sdf, origin=lo, spacing=float(spacing))


def _trilinear(grid: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Trilinear sample of a 3D grid at fractional indices idx (P,3)."""
    nx, ny, nz = grid.shape
    x = np.clip(idx[:, 0], 0, nx - 1.001)
    y = np.clip(idx[:, 1], 0, ny - 1.001)
    z = np.clip(idx[:, 2], 0, nz - 1.001)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); z0 = np.floor(z).astype(int)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    fx, fy, fz = x - x0, y - y0, z - z0
    out = np.zeros(idx.shape[0], dtype=np.float64)
    for dx, wx in ((0, 1 - fx), (1, fx)):
        for dy, wy in ((0, 1 - fy), (1, fy)):
            for dz, wz in ((0, 1 - fz), (1, fz)):
                out += wx * wy * wz * grid[
                    np.where(dx, x1, x0), np.where(dy, y1, y0), np.where(dz, z1, z0)]
    return out


__all__ = ["SignedDistanceField", "build_hull_sdf"]
