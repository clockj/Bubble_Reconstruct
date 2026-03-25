from __future__ import annotations

from pathlib import Path

import numpy as np

from .camera import OpenLPTCameraSet
from .hull import VisualHullResult, create_visual_hull
from .io import discover_camera_files, load_camera_masks, stack_boolean_images
from .models import FullReconstructionResult, ReconstructionInputs
from .properties import get_bubble_props
from .refinement import find_surface_components, refine_surface_points


def build_inputs(
    data_dir: str | Path,
    calibration_dir: str | Path,
    frame: int,
    num_cameras: int,
    voxel_size: list[float] | np.ndarray,
    limits: list[float] | np.ndarray,
    resolution: list[float] | np.ndarray | None = None,
) -> ReconstructionInputs:
    data_path = Path(data_dir).resolve()
    calibration_path = Path(calibration_dir).resolve()
    return ReconstructionInputs(
        data_dir=data_path,
        calibration_dir=calibration_path,
        frame=int(frame),
        num_cameras=int(num_cameras),
        voxel_size=np.asarray(voxel_size, dtype=np.float64),
        limits=np.asarray(limits, dtype=np.float64),
        resolution=None if resolution is None else np.asarray(resolution, dtype=np.float64),
    )


def run_coarse_reconstruction(inputs: ReconstructionInputs) -> VisualHullResult:
    masks = load_camera_masks(inputs.data_dir, inputs.frame, inputs.num_cameras)
    camera_files = discover_camera_files(inputs.calibration_dir)
    if len(camera_files) < inputs.num_cameras:
        raise FileNotFoundError(
            f"Expected at least {inputs.num_cameras} camera files in {inputs.calibration_dir}, found {len(camera_files)}."
        )

    cameras = OpenLPTCameraSet.from_camera_files(camera_files[: inputs.num_cameras])
    return create_visual_hull(
        masks=masks,
        cameras=cameras,
        voxel_size=inputs.voxel_size,
        limits=inputs.limits,
    )


def run_full_reconstruction(inputs: ReconstructionInputs) -> FullReconstructionResult:
    masks = load_camera_masks(inputs.data_dir, inputs.frame, inputs.num_cameras)
    camera_files = discover_camera_files(inputs.calibration_dir)
    if len(camera_files) < inputs.num_cameras:
        raise FileNotFoundError(
            f"Expected at least {inputs.num_cameras} camera files in {inputs.calibration_dir}, found {len(camera_files)}."
        )

    cameras = OpenLPTCameraSet.from_camera_files(camera_files[: inputs.num_cameras])
    coarse_result = create_visual_hull(
        masks=masks,
        cameras=cameras,
        voxel_size=inputs.voxel_size,
        limits=inputs.limits,
    )
    real_images = stack_boolean_images(masks)
    fine_voxel_size = inputs.voxel_size / 3.0

    if int(np.sum(coarse_result.voxel_volume)) <= 0:
        return FullReconstructionResult(
            voxel_size=inputs.voxel_size,
            voxel_size_2=fine_voxel_size,
            limits=inputs.limits,
            real_images=real_images,
            voxels=np.empty((0, 3), dtype=np.float64),
            bubbles=np.empty((2, 0), dtype=np.int64),
            properties=np.empty((0, 15), dtype=np.float64),
            completed=True,
            coarse_result=coarse_result,
        )

    surface_components = find_surface_components(
        coarse_result.voxel_volume,
        coarse_result.grid_x,
        coarse_result.grid_y,
        coarse_result.grid_z,
    )

    all_voxels: list[np.ndarray] = []
    bubbles: list[tuple[int, int]] = []
    properties: list[np.ndarray] = []
    count = 0

    image_resolution = inputs.resolution
    if image_resolution is None:
        image_resolution = np.array([real_images.shape[1], real_images.shape[0]], dtype=np.float64)

    for surface_points in surface_components:
        refined_points = refine_surface_points(
            surface_points,
            coarse_voxel_size=inputs.voxel_size,
            masks=masks,
            cameras=cameras,
            mv=2,
            res_inc=3,
        )
        voxel_list, props = get_bubble_props(
            refined_points,
            voxel_size=fine_voxel_size,
            image_resolution=image_resolution,
            num_cameras=inputs.num_cameras,
            limits=inputs.limits,
            cameras=cameras,
            voxels_center=np.mean(surface_points, axis=0),
        )
        all_voxels.append(voxel_list)
        bubbles.append((count + 1, count + voxel_list.shape[0]))
        properties.append(props)
        count += voxel_list.shape[0]

    voxels = np.vstack(all_voxels) if all_voxels else np.empty((0, 3), dtype=np.float64)
    bubble_array = np.array(bubbles, dtype=np.int64).T if bubbles else np.empty((2, 0), dtype=np.int64)
    props_array = np.vstack(properties) if properties else np.empty((0, 15), dtype=np.float64)

    return FullReconstructionResult(
        voxel_size=inputs.voxel_size,
        voxel_size_2=fine_voxel_size,
        limits=inputs.limits,
        real_images=real_images,
        voxels=voxels,
        bubbles=bubble_array,
        properties=props_array,
        completed=True,
        coarse_result=coarse_result,
    )
