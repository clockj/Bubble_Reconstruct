from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class VoxelGrid:
    voxels: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    grid_z: np.ndarray
    voxel_counts: np.ndarray


def _axis_values(lower: float, upper: float, step: float) -> np.ndarray:
    count = int(round(abs(upper - lower) / step)) + 1
    direction = step if upper >= lower else -step
    return lower + np.arange(count, dtype=np.float64) * direction


def initialize_voxels(limits: np.ndarray | list[float], voxel_size: np.ndarray | list[float]) -> VoxelGrid:
    limits_array = np.asarray(limits, dtype=np.float64)
    voxel_array = np.asarray(voxel_size, dtype=np.float64)

    x_values = _axis_values(limits_array[0], limits_array[1], voxel_array[0])
    y_values = _axis_values(limits_array[2], limits_array[3], voxel_array[1])
    z_values = _axis_values(limits_array[4], limits_array[5], voxel_array[2])

    grid_x, grid_y, grid_z = np.meshgrid(x_values, y_values, z_values, indexing="xy")

    voxels = np.array(
        [[x_coord, y_coord, z_coord] for z_coord in z_values for x_coord in x_values for y_coord in y_values],
        dtype=np.float64,
    )

    voxel_counts = np.array([len(x_values) - 1, len(y_values) - 1, len(z_values) - 1], dtype=np.int64)
    return VoxelGrid(voxels=voxels, grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, voxel_counts=voxel_counts)


def convert_voxel_list_to_volume(
    voxel_list: np.ndarray,
    voxel_size: np.ndarray | list[float],
    grid_x: np.ndarray | None = None,
    grid_y: np.ndarray | None = None,
    grid_z: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(voxel_list, dtype=np.float64)
    voxel_array = np.asarray(voxel_size, dtype=np.float64)

    if points.size == 0:
        if grid_x is None or grid_y is None or grid_z is None:
            grid_x, grid_y, grid_z = np.meshgrid([0.0], [0.0], [0.0], indexing="xy")
        volume = np.zeros_like(grid_x, dtype=bool)
        return grid_x, grid_y, grid_z, volume

    if grid_x is None or grid_y is None or grid_z is None:
        x_values = _axis_values(points[:, 0].min() - voxel_array[0], points[:, 0].max() + voxel_array[0], voxel_array[0])
        y_values = _axis_values(points[:, 1].min() - voxel_array[1], points[:, 1].max() + voxel_array[1], voxel_array[1])
        z_values = _axis_values(points[:, 2].min() - voxel_array[2], points[:, 2].max() + voxel_array[2], voxel_array[2])
        grid_x, grid_y, grid_z = np.meshgrid(x_values, y_values, z_values, indexing="xy")
        x_min = x_values.min()
        y_min = y_values.min()
        z_min = z_values.min()
    else:
        x_min = float(np.min(grid_x))
        y_min = float(np.min(grid_y))
        z_min = float(np.min(grid_z))

    subscripts = np.rint((points - np.array([x_min, y_min, z_min], dtype=np.float64)) / voxel_array).astype(np.int64)
    volume = np.zeros(grid_x.shape, dtype=bool)
    volume[subscripts[:, 1], subscripts[:, 0], subscripts[:, 2]] = True
    return grid_x, grid_y, grid_z, volume
