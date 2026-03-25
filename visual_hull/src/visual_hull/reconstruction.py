from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .camera import OpenLPTCameraSet
from .hull import VisualHullResult, create_visual_hull
from .io import discover_camera_files, load_camera_masks


@dataclass(slots=True)
class ReconstructionInputs:
    data_dir: Path
    calibration_dir: Path
    frame: int
    num_cameras: int
    voxel_size: np.ndarray
    limits: np.ndarray


def build_inputs(
    data_dir: str | Path,
    calibration_dir: str | Path,
    frame: int,
    num_cameras: int,
    voxel_size: list[float] | np.ndarray,
    limits: list[float] | np.ndarray,
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
