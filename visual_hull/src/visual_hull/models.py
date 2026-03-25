from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hull import VisualHullResult


@dataclass(slots=True)
class ReconstructionInputs:
    data_dir: Path
    calibration_dir: Path
    frame: int
    num_cameras: int
    voxel_size: np.ndarray
    limits: np.ndarray
    resolution: np.ndarray | None


@dataclass(slots=True)
class FullReconstructionResult:
    voxel_size: np.ndarray
    voxel_size_2: np.ndarray
    limits: np.ndarray
    real_images: np.ndarray
    voxels: np.ndarray
    bubbles: np.ndarray
    properties: np.ndarray
    completed: bool
    coarse_result: VisualHullResult

    def to_matlab_payload(self) -> dict[str, np.ndarray | bool]:
        return {
            "voxel_size": np.asarray(self.voxel_size, dtype=np.float64),
            "voxel_size_2": np.asarray(self.voxel_size_2, dtype=np.float64),
            "limits": np.asarray(self.limits, dtype=np.float64),
            "real_images": np.asarray(self.real_images, dtype=bool),
            "voxels": np.asarray(self.voxels, dtype=np.float64),
            "bubbles": np.asarray(self.bubbles, dtype=np.int64),
            "properties": np.asarray(self.properties, dtype=np.float64),
            "completed": bool(self.completed),
        }

    def to_hdf5_payload(self) -> dict[str, np.ndarray | bool]:
        return self.to_matlab_payload()


@dataclass(slots=True)
class FrameExportResult:
    frame: int
    output_path: Path
    voxel_count: int
    bubble_count: int
    completed: bool
