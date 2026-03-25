from __future__ import annotations

import numpy as np
from scipy import ndimage

from .camera import OpenLPTCameraSet
from .hull import vote_visual_hull_for_points


def _points_from_mask(grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    flat_mask = np.ravel(mask, order="F")
    return np.column_stack(
        (
            np.ravel(grid_x, order="F")[flat_mask],
            np.ravel(grid_y, order="F")[flat_mask],
            np.ravel(grid_z, order="F")[flat_mask],
        )
    ).astype(np.float64, copy=False)


def find_surface_components(
    voxel_volume: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
) -> list[np.ndarray]:
    structure_6 = ndimage.generate_binary_structure(3, 1)
    structure_26 = np.ones((3, 3, 3), dtype=bool)
    eroded = ndimage.binary_erosion(voxel_volume, structure=structure_6, border_value=0)
    surface = voxel_volume & ~eroded
    labeled, num_features = ndimage.label(surface, structure=structure_26)

    components: list[np.ndarray] = []
    for label_id in range(1, num_features + 1):
        mask = labeled == label_id
        if not np.any(mask):
            continue
        components.append(_points_from_mask(grid_x, grid_y, grid_z, mask))
    return components


def refine_surface_points(
    surface_points: np.ndarray,
    *,
    coarse_voxel_size: np.ndarray,
    masks: list[np.ndarray],
    cameras: OpenLPTCameraSet,
    mv: int = 2,
    res_inc: int = 3,
) -> np.ndarray:
    points = np.asarray(surface_points, dtype=np.float64)
    coarse_size = np.asarray(coarse_voxel_size, dtype=np.float64)
    fine_size = coarse_size / float(res_inc)

    offset_x = np.arange(-mv * coarse_size[0], mv * coarse_size[0] + fine_size[0] * 0.5, fine_size[0], dtype=np.float64)
    offset_y = np.arange(-mv * coarse_size[1], mv * coarse_size[1] + fine_size[1] * 0.5, fine_size[1], dtype=np.float64)
    offset_z = np.arange(-mv * coarse_size[2], mv * coarse_size[2] + fine_size[2] * 0.5, fine_size[2], dtype=np.float64)
    mesh = np.meshgrid(offset_x, offset_y, offset_z, indexing="xy")
    offsets = np.column_stack([axis.reshape(-1) for axis in mesh])

    candidate_points = (points[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    candidate_points = np.unique(np.round(candidate_points, decimals=10), axis=0)

    voted = vote_visual_hull_for_points(masks, candidate_points, cameras)
    kept = voted[voted[:, 3] >= float(cameras.count), :3]
    return kept.astype(np.float64, copy=False)
