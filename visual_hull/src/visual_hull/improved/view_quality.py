"""Per-camera view-quality / overlap analysis for robust silhouette fitting.

Implements Steps 1.4-1.5 of the reconstruction plan.  For one bubble we
re-project its own visual-hull voxels (and, optionally, the other bubbles')
into each camera and compare against the *observed* mask to decide how much
to trust that camera when fitting the surface:

    - overlap_ratio  : how much of this bubble's projection is shared with
                       other bubbles (occlusion / ambiguity),
    - quality_score  : IoU-style agreement between the re-projected hull
                       silhouette and the observed mask (mask holes / noise),
    - view_weight    : combined trust in [0, 1], per the plan's rule.

The re-projected hull silhouette is used as the reference (overlap-free,
independent of raw-mask overlaps), matching the existing --sh-silhouette
target.
"""

from __future__ import annotations

import numpy as np


def _fill_hull_silhouette(pixels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Filled convex-hull silhouette of projected points (bubbles are convex)."""
    from scipy.spatial import ConvexHull
    from skimage.draw import polygon as sk_polygon

    mask = np.zeros(shape, dtype=bool)
    if pixels.shape[0] < 3:
        return mask
    try:
        hull = ConvexHull(pixels)
    except Exception:
        return mask
    poly = pixels[hull.vertices]
    rows, cols = sk_polygon(poly[:, 1], poly[:, 0], shape=shape)
    mask[rows, cols] = True
    return mask


def _camera_fill(points_3d, cameras, camera_index, shape, scale) -> np.ndarray:
    projection = cameras.project_points(camera_index, points_3d)
    pixels = projection.pixels[projection.valid] / float(scale)
    return _fill_hull_silhouette(pixels, shape)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return 1.0 if union == 0 else inter / union


def analyze_bubble_views(
    bubble_voxels: np.ndarray,
    other_voxels: list[np.ndarray] | None,
    observed_masks: list[np.ndarray],
    cameras,
    scale: int = 4,
) -> list[dict]:
    """Return a per-camera dict of overlap / quality / weight for one bubble.

    ``observed_masks`` are full-resolution binary masks; they are downsampled
    by ``scale`` to match the re-projected silhouette resolution.
    """
    n_cam = cameras.count
    per_camera: list[dict] = []
    for c in range(n_cam):
        obs_full = np.asarray(observed_masks[c], dtype=bool)
        ds_shape = (obs_full.shape[0] // scale, obs_full.shape[1] // scale)
        obs = obs_full[: ds_shape[0] * scale, : ds_shape[1] * scale]
        obs = obs.reshape(ds_shape[0], scale, ds_shape[1], scale).any(axis=(1, 3))

        this_proj = _camera_fill(bubble_voxels, cameras, c, ds_shape, scale)
        this_area = int(np.count_nonzero(this_proj))

        # Overlap with other bubbles' projections.
        overlap_ratio = 0.0
        if other_voxels:
            others = np.zeros(ds_shape, dtype=bool)
            for ov in other_voxels:
                if ov.shape[0] >= 3:
                    others |= _camera_fill(ov, cameras, c, ds_shape, scale)
            if this_area > 0:
                overlap_ratio = float(np.count_nonzero(this_proj & others)) / this_area

        # Mask quality: agreement of re-projected hull with observed mask,
        # restricted to where the bubble is expected (this_proj).
        if this_area > 0:
            missing = float(np.count_nonzero(this_proj & ~obs)) / this_area
            quality = _iou(this_proj, obs)
        else:
            missing, quality = 1.0, 0.0

        # Plan's weighting rule (Step 1.5).
        if overlap_ratio > 0.5:
            weight = 0.2
        elif overlap_ratio > 0.1:
            weight = 0.5 * quality
        else:
            weight = quality

        per_camera.append({
            "camera": c,
            "overlap_ratio": overlap_ratio,
            "quality_score": float(quality),
            "missing_ratio": float(missing),
            "view_weight": float(max(0.0, min(1.0, weight))),
        })
    return per_camera


def camera_weights(analysis: list[dict], floor: float = 0.05) -> np.ndarray:
    """Normalized non-negative per-camera weights (sum to 1), with a floor."""
    w = np.array([max(a["view_weight"], floor) for a in analysis], dtype=np.float64)
    total = float(w.sum())
    return w / total if total > 0 else np.full(len(analysis), 1.0 / len(analysis))


__all__ = ["analyze_bubble_views", "camera_weights"]
