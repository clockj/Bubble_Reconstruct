from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..camera import OpenLPTCameraSet
from ..hull import VisualHullResult
from ..voxel_grid import VoxelGrid, convert_voxel_list_to_volume, initialize_voxels


@dataclass(slots=True)
class SoftVisualHullConfig:
    band_width_px: float = 0.75
    min_score: float | None = None
    min_score_ratio: float = 0.85
    min_camera_score: float = 0.35
    max_outside_distance_px: float = 0.25

    def to_dict(self) -> dict[str, float | None]:
        return {
            "band_width_px": float(self.band_width_px),
            "min_score": None if self.min_score is None else float(self.min_score),
            "min_score_ratio": float(self.min_score_ratio),
            "min_camera_score": float(self.min_camera_score),
            "max_outside_distance_px": float(self.max_outside_distance_px),
        }


def _signed_distance_map(mask: np.ndarray) -> np.ndarray:
    mask_array = np.asarray(mask, dtype=bool)
    inside = ndimage.distance_transform_edt(mask_array)
    outside = ndimage.distance_transform_edt(~mask_array)
    return inside - outside


def _confidence_map(mask: np.ndarray, band_width_px: float) -> np.ndarray:
    signed_distance = _signed_distance_map(mask)
    scale = max(float(band_width_px), 1e-6)
    return 0.5 * (1.0 + np.tanh(signed_distance / scale))


def _bilinear_sample(image: np.ndarray, pixels: np.ndarray, valid_projection: np.ndarray) -> np.ndarray:
    image_array = np.asarray(image, dtype=np.float64)
    coords = np.asarray(pixels, dtype=np.float64)

    x = coords[:, 0]
    y = coords[:, 1]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    in_bounds = (
        valid_projection
        & np.isfinite(x)
        & np.isfinite(y)
        & (x0 >= 0)
        & (y0 >= 0)
        & (x1 < image_array.shape[1])
        & (y1 < image_array.shape[0])
    )

    samples = np.zeros(coords.shape[0], dtype=np.float64)
    if not np.any(in_bounds):
        return samples

    x_valid = x[in_bounds]
    y_valid = y[in_bounds]
    x0_valid = x0[in_bounds]
    y0_valid = y0[in_bounds]
    x1_valid = x1[in_bounds]
    y1_valid = y1[in_bounds]

    dx = x_valid - x0_valid
    dy = y_valid - y0_valid

    top_left = image_array[y0_valid, x0_valid]
    top_right = image_array[y0_valid, x1_valid]
    bottom_left = image_array[y1_valid, x0_valid]
    bottom_right = image_array[y1_valid, x1_valid]

    samples[in_bounds] = (
        (1.0 - dx) * (1.0 - dy) * top_left
        + dx * (1.0 - dy) * top_right
        + (1.0 - dx) * dy * bottom_left
        + dx * dy * bottom_right
    )
    return samples


def _sample_camera_support(
    masks: list[np.ndarray],
    points: np.ndarray,
    cameras: OpenLPTCameraSet,
    config: SoftVisualHullConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    confidence_maps = [_confidence_map(mask, config.band_width_px) for mask in masks]
    signed_distance_maps = [_signed_distance_map(mask) for mask in masks]

    confidence_samples = np.zeros((points.shape[0], cameras.count), dtype=np.float64)
    signed_distance_samples = np.full((points.shape[0], cameras.count), -np.inf, dtype=np.float64)
    valid_samples = np.zeros((points.shape[0], cameras.count), dtype=bool)

    for camera_index, (confidence_map, signed_distance_map) in enumerate(zip(confidence_maps, signed_distance_maps)):
        projection = cameras.project_points(camera_index, points)
        confidence_samples[:, camera_index] = _bilinear_sample(confidence_map, projection.pixels, projection.valid)
        signed_distance_samples[:, camera_index] = _bilinear_sample(
            signed_distance_map,
            projection.pixels,
            projection.valid,
        )
        valid_samples[:, camera_index] = projection.valid

    return confidence_samples, signed_distance_samples, valid_samples


def _keep_mask_from_support(
    confidence_samples: np.ndarray,
    signed_distance_samples: np.ndarray,
    valid_samples: np.ndarray,
    config: SoftVisualHullConfig,
) -> tuple[np.ndarray, np.ndarray]:
    total_score = np.sum(confidence_samples, axis=1)
    mean_score = np.mean(confidence_samples, axis=1)
    min_score = np.min(confidence_samples, axis=1)
    all_valid = np.all(valid_samples, axis=1)
    outside_ok = np.all(signed_distance_samples >= -float(config.max_outside_distance_px), axis=1)

    required_total_score = config.min_score
    if required_total_score is None:
        required_total_score = float(confidence_samples.shape[1]) * float(config.min_score_ratio)

    keep = (
        all_valid
        & outside_ok
        & (total_score >= float(required_total_score))
        & (mean_score >= float(config.min_score_ratio))
        & (min_score >= float(config.min_camera_score))
    )
    return total_score, keep


def vote_visual_hull_soft(
    masks: list[np.ndarray],
    voxel_grid: VoxelGrid,
    cameras: OpenLPTCameraSet,
    *,
    config: SoftVisualHullConfig | None = None,
) -> np.ndarray:
    if len(masks) != cameras.count:
        raise ValueError("The number of masks must match the number of cameras.")

    hull_config = config or SoftVisualHullConfig()
    confidence_samples, signed_distance_samples, valid_samples = _sample_camera_support(
        masks,
        voxel_grid.voxels,
        cameras,
        hull_config,
    )
    scores, _ = _keep_mask_from_support(confidence_samples, signed_distance_samples, valid_samples, hull_config)

    return np.column_stack((voxel_grid.voxels, scores))


def vote_visual_hull_for_points_soft(
    masks: list[np.ndarray],
    points: np.ndarray,
    cameras: OpenLPTCameraSet,
    *,
    config: SoftVisualHullConfig | None = None,
) -> np.ndarray:
    point_array = np.asarray(points, dtype=np.float64)
    if len(masks) != cameras.count:
        raise ValueError("The number of masks must match the number of cameras.")

    hull_config = config or SoftVisualHullConfig()
    confidence_samples, signed_distance_samples, valid_samples = _sample_camera_support(
        masks,
        point_array,
        cameras,
        hull_config,
    )
    scores, _ = _keep_mask_from_support(confidence_samples, signed_distance_samples, valid_samples, hull_config)

    return np.column_stack((point_array, scores))


def create_visual_hull_soft(
    masks: list[np.ndarray],
    cameras: OpenLPTCameraSet,
    voxel_size: np.ndarray | list[float],
    limits: np.ndarray | list[float],
    *,
    config: SoftVisualHullConfig | None = None,
) -> VisualHullResult:
    hull_config = config or SoftVisualHullConfig()
    voxel_grid = initialize_voxels(limits, voxel_size)
    confidence_samples, signed_distance_samples, valid_samples = _sample_camera_support(
        masks,
        voxel_grid.voxels,
        cameras,
        hull_config,
    )
    scores, keep_mask = _keep_mask_from_support(confidence_samples, signed_distance_samples, valid_samples, hull_config)
    voxels_scored = np.column_stack((voxel_grid.voxels, scores))
    kept_voxels = voxel_grid.voxels[keep_mask]
    grid_x, grid_y, grid_z, voxel_volume = convert_voxel_list_to_volume(
        kept_voxels,
        voxel_size,
        voxel_grid.grid_x,
        voxel_grid.grid_y,
        voxel_grid.grid_z,
    )

    return VisualHullResult(
        voxels_voted=voxels_scored,
        voxel_volume=voxel_volume,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        kept_voxels=kept_voxels,
    )