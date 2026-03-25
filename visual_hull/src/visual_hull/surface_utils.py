from __future__ import annotations

import numpy as np
from skimage.measure import marching_cubes

from .voxel_grid import convert_voxel_list_to_volume


def points_from_mask(grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    flat_mask = np.ravel(mask, order="F")
    return np.column_stack(
        (
            np.ravel(grid_x, order="F")[flat_mask],
            np.ravel(grid_y, order="F")[flat_mask],
            np.ravel(grid_z, order="F")[flat_mask],
        )
    ).astype(np.float64, copy=False)


def surface_mesh_from_voxels(voxels: np.ndarray, voxel_size: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    voxel_array = np.asarray(voxels, dtype=np.float64)
    size = np.asarray(voxel_size, dtype=np.float64)
    if voxel_array.size == 0:
        return None

    grid_x, grid_y, grid_z, volume = convert_voxel_list_to_volume(voxel_array, size)
    if np.count_nonzero(volume) == 0:
        return None

    origin = np.array([float(np.min(grid_x)), float(np.min(grid_y)), float(np.min(grid_z))], dtype=np.float64)
    verts, faces, _, _ = marching_cubes(
        volume.astype(np.float32),
        level=0.5,
        spacing=(float(size[1]), float(size[0]), float(size[2])),
    )
    world_verts = np.column_stack(
        (
            origin[0] + verts[:, 1],
            origin[1] + verts[:, 0],
            origin[2] + verts[:, 2],
        )
    )
    return world_verts.astype(np.float64, copy=False), faces.astype(np.int64, copy=False)


__all__ = ["points_from_mask", "surface_mesh_from_voxels"]