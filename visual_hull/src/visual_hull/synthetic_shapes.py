"""Synthetic known-answer shapes for benchmarking the SH reconstruction.

Every shape here is *star-convex* about a known centre and described by a
radial function ``r(theta, phi)`` (mm), so we always know the ground-truth
volume, principal axes and an exact inside/outside test.  Shapes are rendered
into the real 4-camera refractive (PINPLATE) model so the reconstruction
pipeline consumes them exactly like real bubble masks.

Provided:
    - sphere(radius)
    - ellipsoid(a, b, c, rotation)          -> known axes / aspect ratio
    - convex_sh(base_radius, deformations)   -> known low-order c_lm, convex
    - render_masks_through_cameras(...)      -> per-camera binary silhouettes
    - corrupt_mask(...)                      -> holes / speckle (Phase 2 tests)
    - radial_inside(...)                     -> exact inside test (for 3D IoU)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import Callable

import numpy as np


# ── Direction sampling ────────────────────────────────────────────────────────
def fibonacci_directions(n: int) -> np.ndarray:
    """~Uniform unit vectors on the sphere, shape (n, 3)."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi_pol = np.arccos(1.0 - 2.0 * i / n)
    golden = pi * (1.0 + 5.0 ** 0.5)
    az = golden * i
    return np.column_stack(
        (np.sin(phi_pol) * np.cos(az), np.sin(phi_pol) * np.sin(az), np.cos(phi_pol))
    )


def directions_to_angles(dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(theta in [0,pi], phi in [0,2pi)) for unit directions."""
    d = np.asarray(dirs, dtype=np.float64)
    theta = np.arccos(np.clip(d[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(d[:, 1], d[:, 0]), 2.0 * pi)
    return theta, phi


# ── Shape definition ──────────────────────────────────────────────────────────
@dataclass(slots=True)
class SyntheticShape:
    name: str
    center: np.ndarray
    radial_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]  # (theta, phi) -> r mm
    true_volume: float
    true_axes: np.ndarray  # sorted-descending principal semi-axis lengths (mm)
    meta: dict = field(default_factory=dict)

    def surface_points(self, n: int = 8000) -> np.ndarray:
        dirs = fibonacci_directions(n)
        theta, phi = directions_to_angles(dirs)
        r = self.radial_fn(theta, phi)
        return self.center + dirs * r[:, None]

    @property
    def true_diameter(self) -> float:
        """Equal-volume-sphere diameter (mm) — matches pipeline's D_eq metric."""
        return 2.0 * (3.0 * self.true_volume / (4.0 * pi)) ** (1.0 / 3.0)

    @property
    def true_aspect_ratio(self) -> float:
        return float(self.true_axes[0] / self.true_axes[-1])


def _radial_volume(radial_fn, n: int = 40000) -> float:
    """Monte-Carlo-free volume via V = (1/3) mean(r^3) * 4pi over uniform dirs."""
    dirs = fibonacci_directions(n)
    theta, phi = directions_to_angles(dirs)
    r = radial_fn(theta, phi)
    return float((4.0 * pi / 3.0) * np.mean(r ** 3))


def _principal_axes(radial_fn, n: int = 20000) -> np.ndarray:
    """Descending semi-axis lengths from the covariance of surface points."""
    dirs = fibonacci_directions(n)
    theta, phi = directions_to_angles(dirs)
    pts = dirs * radial_fn(theta, phi)[:, None]
    # For a surface, semi-axes ~ sqrt(5 * eigenvalue) (uniform shell approx);
    # we only use the *ratio*, so exact scaling is irrelevant.
    cov = np.cov(pts.T)
    eig = np.sqrt(np.maximum(np.linalg.eigvalsh(cov), 0.0))
    return np.sort(eig)[::-1]


def sphere(center, radius: float) -> SyntheticShape:
    center = np.asarray(center, dtype=np.float64)
    r = float(radius)
    fn = lambda th, ph: np.full_like(np.asarray(th, dtype=np.float64), r)
    vol = (4.0 / 3.0) * pi * r ** 3
    return SyntheticShape("sphere", center, fn, vol, np.array([r, r, r]),
                          {"radius": r})


def ellipsoid(center, a: float, b: float, c: float,
              rotation: np.ndarray | None = None) -> SyntheticShape:
    """Ellipsoid with semi-axes (a,b,c) mm, optional 3x3 rotation."""
    center = np.asarray(center, dtype=np.float64)
    axes = np.array([float(a), float(b), float(c)])
    R = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)

    def fn(th, ph):
        th = np.asarray(th, dtype=np.float64)
        ph = np.asarray(ph, dtype=np.float64)
        d = np.column_stack((np.sin(th) * np.cos(ph),
                             np.sin(th) * np.sin(ph),
                             np.cos(th)))
        dl = d @ R  # rotate direction into ellipsoid-aligned frame
        denom = np.sqrt((dl[:, 0] / a) ** 2 + (dl[:, 1] / b) ** 2 + (dl[:, 2] / c) ** 2)
        return 1.0 / denom

    vol = (4.0 / 3.0) * pi * a * b * c
    return SyntheticShape("ellipsoid", center, fn, vol,
                          np.sort(axes)[::-1], {"axes": axes.tolist()})


def convex_sh(center, base_radius: float,
              deformations: dict[tuple[int, int], float]) -> SyntheticShape:
    """Radial shape r = base * (1 + sum a_lm * Y_lm), kept convex by small a_lm.

    Deformations are given as {(l, m): amplitude} with amplitude a *fraction*
    of the base radius.  Keeping |sum| well below 1 guarantees r>0 and (for
    l<=2, small amplitudes) a convex shape.
    """
    from .improved.spherical_harmonics.surface import _real_spherical_harmonic

    center = np.asarray(center, dtype=np.float64)
    base = float(base_radius)
    items = list(deformations.items())

    def fn(th, ph):
        th = np.asarray(th, dtype=np.float64)
        ph = np.asarray(ph, dtype=np.float64)
        defo = np.zeros_like(th)
        for (l, m), amp in items:
            defo = defo + float(amp) * _real_spherical_harmonic(int(l), int(m), th, ph)
        return base * (1.0 + defo)

    vol = _radial_volume(fn)
    axes = _principal_axes(fn)
    return SyntheticShape("convex_sh", center, fn, vol, axes,
                          {"base_radius": base, "deformations":
                           {f"{l},{m}": a for (l, m), a in deformations.items()}})


def radial_inside(points: np.ndarray, center: np.ndarray, radial_fn) -> np.ndarray:
    """Boolean inside test for a star-convex radial shape."""
    p = np.asarray(points, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    dist = np.linalg.norm(p, axis=1)
    safe = np.maximum(dist, 1e-12)
    theta = np.arccos(np.clip(p[:, 2] / safe, -1.0, 1.0))
    phi = np.mod(np.arctan2(p[:, 1], p[:, 0]), 2.0 * pi)
    return dist <= radial_fn(theta, phi)


# ── Rendering into the real cameras ───────────────────────────────────────────
def render_masks_through_cameras(shape: SyntheticShape, cameras, hw: tuple[int, int],
                                 n_surface: int = 9000) -> list[np.ndarray] | None:
    """Exact silhouette of a convex shape = 2D convex hull of projected surface."""
    from scipy.spatial import ConvexHull
    from skimage.draw import polygon as sk_polygon

    height, width = hw
    surface = shape.surface_points(n_surface)
    masks: list[np.ndarray] = []
    for cam in range(cameras.count):
        proj = cameras.project_points(cam, surface)
        uv = proj.pixels[proj.valid]
        if uv.shape[0] < 3:
            return None
        try:
            hull = ConvexHull(uv)
        except Exception:
            return None
        poly = uv[hull.vertices]
        rr, cc = sk_polygon(poly[:, 1], poly[:, 0], shape=(height, width))
        mask = np.zeros((height, width), dtype=bool)
        mask[rr, cc] = True
        if not mask.any():
            return None
        masks.append(mask)
    return masks


def corrupt_mask(mask: np.ndarray, hole_frac: float = 0.0, speckle_frac: float = 0.0,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Inject a mask hole (erode a random sub-region) and/or speckle noise."""
    rng = rng or np.random.default_rng(0)
    out = np.asarray(mask, dtype=bool).copy()
    ys, xs = np.where(out)
    if ys.size and hole_frac > 0.0:
        # Punch a rectangular hole covering ~hole_frac of the bbox.
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        h = max(1, int((y1 - y0 + 1) * hole_frac ** 0.5))
        w = max(1, int((x1 - x0 + 1) * hole_frac ** 0.5))
        cy = rng.integers(y0, max(y0 + 1, y1 - h + 1))
        cx = rng.integers(x0, max(x0 + 1, x1 - w + 1))
        out[cy:cy + h, cx:cx + w] = False
    if speckle_frac > 0.0:
        noise = rng.random(out.shape) < speckle_frac
        out |= noise  # add extra foreground specks outside the shape
    return out


__all__ = [
    "SyntheticShape", "sphere", "ellipsoid", "convex_sh",
    "fibonacci_directions", "directions_to_angles", "radial_inside",
    "render_masks_through_cameras", "corrupt_mask",
]
