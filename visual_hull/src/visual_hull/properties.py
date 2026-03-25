from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes

from .camera import OpenLPTCameraSet
from .voxel_grid import convert_voxel_list_to_volume


def _points_from_mask(grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    flat_mask = np.ravel(mask, order="F")
    return np.column_stack(
        (
            np.ravel(grid_x, order="F")[flat_mask],
            np.ravel(grid_y, order="F")[flat_mask],
            np.ravel(grid_z, order="F")[flat_mask],
        )
    ).astype(np.float64, copy=False)


def _world_vertices_from_volume(volume: np.ndarray, voxel_size: np.ndarray, origin: np.ndarray) -> np.ndarray:
    verts, _, _, _ = marching_cubes(
        volume.astype(np.float32),
        level=0.5,
        spacing=(float(voxel_size[1]), float(voxel_size[0]), float(voxel_size[2])),
    )
    world = np.column_stack(
        (
            origin[0] + verts[:, 1],
            origin[1] + verts[:, 0],
            origin[2] + verts[:, 2],
        )
    )
    return world


def _is_in_boundary(points: np.ndarray, limits: np.ndarray) -> bool:
    return bool(np.all(np.max(points, axis=0) < limits[1::2]) and np.all(np.min(points, axis=0) > limits[0::2]))


def get_bubble_props(
    voxel_list: np.ndarray,
    *,
    voxel_size: np.ndarray,
    image_resolution: np.ndarray,
    num_cameras: int,
    limits: np.ndarray,
    cameras: OpenLPTCameraSet,
    voxels_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(voxel_list, dtype=np.float64)
    size = np.asarray(voxel_size, dtype=np.float64)
    resolution = np.asarray(image_resolution, dtype=np.float64)
    bubble_center = np.asarray(voxels_center, dtype=np.float64)

    grid_x, grid_y, grid_z, volume = convert_voxel_list_to_volume(points, size)
    volume = ndimage.binary_fill_holes(volume)

    labeled, num_features = ndimage.label(volume, structure=np.ones((3, 3, 3), dtype=bool))
    if num_features > 1:
        distances = np.zeros(num_features, dtype=np.float64)
        for label_id in range(1, num_features + 1):
            mask = labeled == label_id
            component_points = _points_from_mask(grid_x, grid_y, grid_z, mask)
            distances[label_id - 1] = np.linalg.norm(np.mean(component_points, axis=0) - bubble_center)
        keep_label = int(np.argmin(distances) + 1)
        volume = labeled == keep_label
    elif num_features == 1:
        volume = labeled == 1

    voxel_points = _points_from_mask(grid_x, grid_y, grid_z, volume)
    centroid = np.mean(voxel_points, axis=0)
    voxel_volume = float(np.sum(volume) * np.prod(size))
    radius = float((3.0 * voxel_volume / (4.0 * np.pi)) ** (1.0 / 3.0))

    origin = np.array([float(np.min(grid_x)), float(np.min(grid_y)), float(np.min(grid_z))], dtype=np.float64)
    verts = _world_vertices_from_volume(volume, size, origin)
    distances = np.linalg.norm(verts - centroid, axis=1)
    major_mag = float(np.max(distances))
    minor_mag = float(np.min(distances))
    aspect_ratio = float(major_mag / minor_mag) if minor_mag > 0 else np.inf

    major_axis = verts[int(np.argmax(distances))] - centroid
    minor_axis = verts[int(np.argmin(distances))] - centroid

    padded_limits = np.array(
        [
            limits[0] + size[0],
            limits[1] - size[0],
            limits[2] + size[1],
            limits[3] - size[1],
            limits[4] + size[2],
            limits[5] - size[2],
        ],
        dtype=np.float64,
    )
    in_boundary = _is_in_boundary(verts, padded_limits)

    if in_boundary:
        pixel_limits = np.array([2.0, resolution[0] - 2.0, 2.0, resolution[1] - 2.0], dtype=np.float64)
        for camera_index in range(num_cameras):
            projection = cameras.project_points(camera_index, verts)
            if not np.all(projection.valid):
                in_boundary = False
                break
            if not _is_in_boundary(projection.pixels, pixel_limits):
                in_boundary = False
                break

    props = np.concatenate(
        (
            centroid,
            np.array([radius, voxel_volume, aspect_ratio, float(in_boundary)], dtype=np.float64),
            major_axis,
            minor_axis,
            np.array([major_mag, minor_mag], dtype=np.float64),
        )
    )
    return voxel_points, props
