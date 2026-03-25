from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pyopenlpt as lpt


@dataclass(slots=True)
class ProjectionResult:
    pixels: np.ndarray
    valid: np.ndarray
    errors: list[str]


class OpenLPTCameraSet:
    def __init__(self, cameras: Sequence[lpt.Camera], camera_files: Sequence[Path]):
        if len(cameras) == 0:
            raise ValueError("At least one camera is required.")
        self._cameras = tuple(cameras)
        self._camera_files = tuple(Path(path) for path in camera_files)

    @classmethod
    def from_camera_files(cls, camera_files: Iterable[str | Path]) -> "OpenLPTCameraSet":
        paths = [Path(path) for path in camera_files]
        cameras = [lpt.Camera(str(path)) for path in paths]
        return cls(cameras, paths)

    @classmethod
    def from_folder(cls, folder: str | Path, pattern: str = "C*P.txt") -> "OpenLPTCameraSet":
        folder_path = Path(folder)
        camera_files = sorted(folder_path.glob(pattern))
        if not camera_files:
            raise FileNotFoundError(f"No camera files matching {pattern!r} were found in {folder_path}.")
        return cls.from_camera_files(camera_files)

    @property
    def camera_files(self) -> tuple[Path, ...]:
        return self._camera_files

    @property
    def count(self) -> int:
        return len(self._cameras)

    def project_points(
        self,
        camera_index: int,
        points_world: np.ndarray,
        *,
        batch_size: int = 50000,
        is_print_detail: bool = False,
    ) -> ProjectionResult:
        points = np.asarray(points_world, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_world must have shape (n_points, 3).")

        camera = self._cameras[camera_index]
        pixels = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
        valid = np.zeros(points.shape[0], dtype=bool)
        errors = [""] * points.shape[0]

        if hasattr(camera, "projectBatchStatus"):
            for start in range(0, points.shape[0], batch_size):
                stop = min(start + batch_size, points.shape[0])
                batch = [lpt.Pt3D(float(x), float(y), float(z)) for x, y, z in points[start:stop]]
                statuses = camera.projectBatchStatus(batch, is_print_detail)
                for offset, status in enumerate(statuses):
                    ok, pt2d, err = status
                    idx = start + offset
                    if ok:
                        pixels[idx, 0] = float(pt2d[0])
                        pixels[idx, 1] = float(pt2d[1])
                        valid[idx] = True
                    else:
                        errors[idx] = str(err)
            return ProjectionResult(pixels=pixels, valid=valid, errors=errors)

        for idx, (x_coord, y_coord, z_coord) in enumerate(points):
            try:
                pt2d = camera.project(lpt.Pt3D(float(x_coord), float(y_coord), float(z_coord)), is_print_detail)
                pixels[idx, 0] = float(pt2d[0])
                pixels[idx, 1] = float(pt2d[1])
                valid[idx] = True
            except Exception as exc:
                errors[idx] = str(exc)

        return ProjectionResult(pixels=pixels, valid=valid, errors=errors)


def floor_pixels_for_mask(pixels: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(pixels, dtype=np.float64)).astype(np.int64)
