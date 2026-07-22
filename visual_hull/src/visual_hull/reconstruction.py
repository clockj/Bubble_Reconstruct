from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
from joblib import Parallel, cpu_count, delayed

from .camera import OpenLPTCameraSet
from .hull import VisualHullResult, create_visual_hull
from .io import discover_camera_files, load_camera_masks, stack_boolean_images
from .models import FrameExportResult, FullReconstructionResult, ReconstructionInputs
from .properties import get_bubble_props
from .refinement import find_surface_components, refine_surface_points
from .writers import ExportFormat, write_reconstruction


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


def run_full_reconstruction_from_data(
    masks: list[np.ndarray],
    cameras: OpenLPTCameraSet,
    voxel_size: np.ndarray | list[float],
    limits: np.ndarray | list[float],
    *,
    num_cameras: int | None = None,
    resolution: np.ndarray | None = None,
) -> FullReconstructionResult:
    """Run the full reconstruction pipeline with pre-loaded masks and cameras.

    This is the same as :func:`run_full_reconstruction` but accepts data
    directly instead of loading from disk.  Useful when masks come from
    TIFF files or other non-MATLAB sources.
    """
    _voxel_size = np.asarray(voxel_size, dtype=np.float64)
    _limits = np.asarray(limits, dtype=np.float64)
    _num_cameras = num_cameras if num_cameras is not None else cameras.count

    coarse_result = create_visual_hull(
        masks=masks,
        cameras=cameras,
        voxel_size=_voxel_size,
        limits=_limits,
    )
    real_images = stack_boolean_images(masks)
    fine_voxel_size = _voxel_size / 3.0

    if int(np.sum(coarse_result.voxel_volume)) <= 0:
        return FullReconstructionResult(
            voxel_size=_voxel_size,
            voxel_size_2=fine_voxel_size,
            limits=_limits,
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

    image_resolution = resolution
    if image_resolution is None:
        image_resolution = np.array([real_images.shape[1], real_images.shape[0]], dtype=np.float64)

    for surface_points in surface_components:
        refined_points = refine_surface_points(
            surface_points,
            coarse_voxel_size=_voxel_size,
            masks=masks,
            cameras=cameras,
            mv=2,
            res_inc=3,
        )
        voxel_list, props = get_bubble_props(
            refined_points,
            voxel_size=fine_voxel_size,
            image_resolution=image_resolution,
            num_cameras=_num_cameras,
            limits=_limits,
            cameras=cameras,
            voxels_center=np.mean(surface_points, axis=0),
        )
        all_voxels.append(voxel_list)
        bubbles.append((count + 1, count + voxel_list.shape[0]))
        properties.append(props)
        count += voxel_list.shape[0]

    final_voxels = np.vstack(all_voxels) if all_voxels else np.empty((0, 3), dtype=np.float64)
    bubble_array = np.array(bubbles, dtype=np.int64).T if bubbles else np.empty((2, 0), dtype=np.int64)
    props_array = np.vstack(properties) if properties else np.empty((0, 15), dtype=np.float64)

    return FullReconstructionResult(
        voxel_size=_voxel_size,
        voxel_size_2=fine_voxel_size,
        limits=_limits,
        real_images=real_images,
        voxels=final_voxels,
        bubbles=bubble_array,
        properties=props_array,
        completed=True,
        coarse_result=coarse_result,
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


def _frame_output_suffix(export_format: ExportFormat) -> str:
    if export_format == "mat":
        return ".mat"
    if export_format in {"h5", "hdf5"}:
        return ".h5"
    raise ValueError("Batch frame reconstruction requires an explicit export format of 'mat', 'h5', or 'hdf5'.")


def _run_and_write_frame(
    frame: int,
    inputs: ReconstructionInputs,
    output_dir: Path,
    export_format: ExportFormat,
    compression: str | None,
) -> FrameExportResult:
    frame_inputs = replace(inputs, frame=int(frame))
    result = run_full_reconstruction(frame_inputs)
    output_path = output_dir / f"Bubble_Frame_{frame:06d}{_frame_output_suffix(export_format)}"
    write_reconstruction(result, output_path, export_format=export_format, compression=compression)
    return FrameExportResult(
        frame=int(frame),
        output_path=output_path,
        voxel_count=int(result.voxels.shape[0]),
        bubble_count=int(result.bubbles.shape[1]) if result.bubbles.ndim == 2 else 0,
        completed=bool(result.completed),
    )


def run_reconstruction_frames_parallel(
    inputs: ReconstructionInputs,
    frames: Sequence[int],
    output_dir: str | Path,
    *,
    export_format: ExportFormat = "mat",
    max_workers: int | None = None,
    compression: str | None = "gzip",
) -> list[FrameExportResult]:
    frame_list = [int(frame) for frame in frames]
    if not frame_list:
        return []
    if export_format == "auto":
        raise ValueError("run_reconstruction_frames_parallel requires an explicit export format.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    worker_count = max_workers
    if worker_count is None:
        worker_count = min(len(frame_list), cpu_count())
    worker_count = max(1, min(int(worker_count), len(frame_list)))

    if worker_count == 1:
        return [
            _run_and_write_frame(frame, inputs, destination, export_format, compression)
            for frame in frame_list
        ]

    results = Parallel(n_jobs=worker_count, prefer="processes")(
        delayed(_run_and_write_frame)(frame, inputs, destination, export_format, compression)
        for frame in frame_list
    )

    return sorted(results, key=lambda item: item.frame)
