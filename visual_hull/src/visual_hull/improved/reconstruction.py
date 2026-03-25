from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np
from joblib import Parallel, cpu_count, delayed

from ..camera import OpenLPTCameraSet
from ..io import discover_camera_files, load_camera_masks, stack_boolean_images
from ..models import FrameExportResult, FullReconstructionResult, ReconstructionInputs
from ..properties import get_bubble_props
from ..writers import ExportFormat, write_reconstruction
from .hull import SoftVisualHullConfig, VisualHullResult, create_visual_hull_soft
from .refinement import ImprovedRefinementConfig, find_surface_components, refine_surface_points_improved


@dataclass(slots=True)
class ImprovedReconstructionConfig:
    hull: SoftVisualHullConfig = field(default_factory=SoftVisualHullConfig)
    refinement: ImprovedRefinementConfig = field(default_factory=ImprovedRefinementConfig)

    def to_dict(self) -> dict[str, dict[str, float | int | bool | None]]:
        return {
            "hull": self.hull.to_dict(),
            "refinement": self.refinement.to_dict(),
        }


def run_coarse_reconstruction_improved(
    inputs: ReconstructionInputs,
    *,
    config: ImprovedReconstructionConfig | None = None,
) -> VisualHullResult:
    settings = config or ImprovedReconstructionConfig()
    masks = load_camera_masks(inputs.data_dir, inputs.frame, inputs.num_cameras)
    camera_files = discover_camera_files(inputs.calibration_dir)
    if len(camera_files) < inputs.num_cameras:
        raise FileNotFoundError(
            f"Expected at least {inputs.num_cameras} camera files in {inputs.calibration_dir}, found {len(camera_files)}."
        )

    cameras = OpenLPTCameraSet.from_camera_files(camera_files[: inputs.num_cameras])
    return create_visual_hull_soft(
        masks=masks,
        cameras=cameras,
        voxel_size=inputs.voxel_size,
        limits=inputs.limits,
        config=settings.hull,
    )


def run_full_reconstruction_improved(
    inputs: ReconstructionInputs,
    *,
    config: ImprovedReconstructionConfig | None = None,
) -> FullReconstructionResult:
    settings = config or ImprovedReconstructionConfig()
    masks = load_camera_masks(inputs.data_dir, inputs.frame, inputs.num_cameras)
    camera_files = discover_camera_files(inputs.calibration_dir)
    if len(camera_files) < inputs.num_cameras:
        raise FileNotFoundError(
            f"Expected at least {inputs.num_cameras} camera files in {inputs.calibration_dir}, found {len(camera_files)}."
        )

    cameras = OpenLPTCameraSet.from_camera_files(camera_files[: inputs.num_cameras])
    coarse_result = create_visual_hull_soft(
        masks=masks,
        cameras=cameras,
        voxel_size=inputs.voxel_size,
        limits=inputs.limits,
        config=settings.hull,
    )
    real_images = stack_boolean_images(masks)
    fine_voxel_size = inputs.voxel_size / float(settings.refinement.resolution_factor)

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
        refined_points = refine_surface_points_improved(
            surface_points,
            coarse_voxel_size=inputs.voxel_size,
            masks=masks,
            cameras=cameras,
            hull_config=settings.hull,
            refinement_config=settings.refinement,
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


def _run_and_write_frame_improved(
    frame: int,
    inputs: ReconstructionInputs,
    output_dir: Path,
    export_format: ExportFormat,
    compression: str | None,
    config: ImprovedReconstructionConfig,
) -> FrameExportResult:
    frame_inputs = replace(inputs, frame=int(frame))
    result = run_full_reconstruction_improved(frame_inputs, config=config)
    output_path = output_dir / f"Bubble_Frame_{frame:06d}{_frame_output_suffix(export_format)}"
    write_reconstruction(result, output_path, export_format=export_format, compression=compression)
    return FrameExportResult(
        frame=int(frame),
        output_path=output_path,
        voxel_count=int(result.voxels.shape[0]),
        bubble_count=int(result.bubbles.shape[1]) if result.bubbles.ndim == 2 else 0,
        completed=bool(result.completed),
    )


def run_reconstruction_frames_parallel_improved(
    inputs: ReconstructionInputs,
    frames: Sequence[int],
    output_dir: str | Path,
    *,
    config: ImprovedReconstructionConfig | None = None,
    export_format: ExportFormat = "mat",
    max_workers: int | None = None,
    compression: str | None = "gzip",
) -> list[FrameExportResult]:
    settings = config or ImprovedReconstructionConfig()
    frame_list = [int(frame) for frame in frames]
    if not frame_list:
        return []
    if export_format == "auto":
        raise ValueError("run_reconstruction_frames_parallel_improved requires an explicit export format.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    worker_count = max_workers
    if worker_count is None:
        worker_count = min(len(frame_list), cpu_count())
    worker_count = max(1, min(int(worker_count), len(frame_list)))

    if worker_count == 1:
        return [
            _run_and_write_frame_improved(frame, inputs, destination, export_format, compression, settings)
            for frame in frame_list
        ]

    results = Parallel(n_jobs=worker_count, prefer="processes")(
        delayed(_run_and_write_frame_improved)(frame, inputs, destination, export_format, compression, settings)
        for frame in frame_list
    )

    return sorted(results, key=lambda item: item.frame)


__all__ = [
    "ImprovedReconstructionConfig",
    "run_coarse_reconstruction_improved",
    "run_full_reconstruction_improved",
    "run_reconstruction_frames_parallel_improved",
]