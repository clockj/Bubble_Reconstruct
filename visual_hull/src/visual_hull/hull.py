from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import OpenLPTCameraSet, floor_pixels_for_mask
from .voxel_grid import VoxelGrid, convert_voxel_list_to_volume, initialize_voxels


@dataclass(slots=True)
class VisualHullResult:
    voxels_voted: np.ndarray
    voxel_volume: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    grid_z: np.ndarray
    kept_voxels: np.ndarray


def _sample_mask(mask: np.ndarray, pixels: np.ndarray, valid_projection: np.ndarray) -> np.ndarray:
    mask_array = np.asarray(mask, dtype=bool)
    pixel_indices = floor_pixels_for_mask(pixels)

    rows = pixel_indices[:, 1]
    cols = pixel_indices[:, 0]
    in_bounds = (
        valid_projection
        & (rows >= 0)
        & (rows < mask_array.shape[0])
        & (cols >= 0)
        & (cols < mask_array.shape[1])
    )

    visible = np.zeros(len(valid_projection), dtype=bool)
    visible[in_bounds] = mask_array[rows[in_bounds], cols[in_bounds]]
    return visible


def vote_visual_hull(
    masks: list[np.ndarray],
    voxel_grid: VoxelGrid,
    cameras: OpenLPTCameraSet,
) -> np.ndarray:
    if len(masks) != cameras.count:
        raise ValueError("The number of masks must match the number of cameras.")

    votes = np.zeros(voxel_grid.voxels.shape[0], dtype=np.int32)
    for camera_index, mask in enumerate(masks):
        projection = cameras.project_points(camera_index, voxel_grid.voxels)
        votes += _sample_mask(mask, projection.pixels, projection.valid).astype(np.int32)

    return np.column_stack((voxel_grid.voxels, votes.astype(np.float64)))


def vote_visual_hull_for_points(
    masks: list[np.ndarray],
    points: np.ndarray,
    cameras: OpenLPTCameraSet,
) -> np.ndarray:
    point_array = np.asarray(points, dtype=np.float64)
    if len(masks) != cameras.count:
        raise ValueError("The number of masks must match the number of cameras.")

    votes = np.zeros(point_array.shape[0], dtype=np.int32)
    for camera_index, mask in enumerate(masks):
        projection = cameras.project_points(camera_index, point_array)
        votes += _sample_mask(mask, projection.pixels, projection.valid).astype(np.int32)

    return np.column_stack((point_array, votes.astype(np.float64)))


def create_visual_hull(
    masks: list[np.ndarray],
    cameras: OpenLPTCameraSet,
    voxel_size: np.ndarray | list[float],
    limits: np.ndarray | list[float],
    min_votes: int | None = None,
) -> VisualHullResult:
    voxel_grid = initialize_voxels(limits, voxel_size)
    voxels_voted = vote_visual_hull(masks, voxel_grid, cameras)

    required_votes = cameras.count if min_votes is None else min_votes
    kept_voxels = voxels_voted[voxels_voted[:, 3] >= float(required_votes), :3]
    grid_x, grid_y, grid_z, voxel_volume = convert_voxel_list_to_volume(
        kept_voxels,
        voxel_size,
        voxel_grid.grid_x,
        voxel_grid.grid_y,
        voxel_grid.grid_z,
    )

    return VisualHullResult(
        voxels_voted=voxels_voted,
        voxel_volume=voxel_volume,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        kept_voxels=kept_voxels,
    )
