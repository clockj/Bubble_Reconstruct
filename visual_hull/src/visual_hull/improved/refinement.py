from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..camera import OpenLPTCameraSet
from ..refinement import find_surface_components
from ..surface_utils import points_from_mask
from .hull import SoftVisualHullConfig, create_visual_hull_soft


@dataclass(slots=True)
class ImprovedRefinementConfig:
    margin_voxels: int = 1
    resolution_factor: int = 3

    def to_dict(self) -> dict[str, int]:
        return {
            "margin_voxels": int(self.margin_voxels),
            "resolution_factor": int(self.resolution_factor),
        }


def _local_limits(points: np.ndarray, coarse_voxel_size: np.ndarray, margin_voxels: int) -> np.ndarray:
    margin = np.asarray(coarse_voxel_size, dtype=np.float64) * float(margin_voxels)
    mins = np.min(points, axis=0) - margin
    maxs = np.max(points, axis=0) + margin
    return np.array([mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]], dtype=np.float64)


def _nearest_component_points(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    volume: np.ndarray,
    coarse_center: np.ndarray,
) -> np.ndarray:
    labeled, num_features = ndimage.label(volume, structure=np.ones((3, 3, 3), dtype=bool))
    if num_features <= 1:
        return points_from_mask(grid_x, grid_y, grid_z, volume)

    best_points = np.empty((0, 3), dtype=np.float64)
    best_distance = np.inf
    for label_id in range(1, num_features + 1):
        mask = labeled == label_id
        if not np.any(mask):
            continue
        component_points = points_from_mask(grid_x, grid_y, grid_z, mask)
        if component_points.size == 0:
            continue
        distance = float(np.linalg.norm(np.mean(component_points, axis=0) - coarse_center))
        if distance < best_distance:
            best_distance = distance
            best_points = component_points
    return best_points


def refine_surface_points_improved(
    surface_points: np.ndarray,
    *,
    coarse_voxel_size: np.ndarray,
    masks: list[np.ndarray],
    cameras: OpenLPTCameraSet,
    hull_config: SoftVisualHullConfig | None = None,
    refinement_config: ImprovedRefinementConfig | None = None,
) -> np.ndarray:
    points = np.asarray(surface_points, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    coarse_size = np.asarray(coarse_voxel_size, dtype=np.float64)
    settings = refinement_config or ImprovedRefinementConfig()
    fine_size = coarse_size / float(settings.resolution_factor)

    local_result = create_visual_hull_soft(
        masks,
        cameras,
        voxel_size=fine_size,
        limits=_local_limits(points, coarse_size, settings.margin_voxels),
        config=hull_config,
    )
    if local_result.kept_voxels.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    return _nearest_component_points(
        local_result.grid_x,
        local_result.grid_y,
        local_result.grid_z,
        local_result.voxel_volume,
        np.mean(points, axis=0),
    )


__all__ = [
    "ImprovedRefinementConfig",
    "find_surface_components",
    "refine_surface_points_improved",
]