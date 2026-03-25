from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import numpy as np
from scipy.io import loadmat


_FRAME_KEY_PATTERN = re.compile(r"^Image(\d{6})$")


def frame_key(frame: int) -> str:
    return f"Image{frame:06d}"


def load_camera_mask(camera_mat_path: str | Path, frame: int) -> np.ndarray:
    mat_path = Path(camera_mat_path)
    data = loadmat(mat_path)
    key = frame_key(frame)
    if key not in data:
        raise KeyError(f"Frame key {key!r} was not found in {mat_path}.")
    return np.asarray(data[key]).astype(bool)


def load_camera_masks(data_dir: str | Path, frame: int, num_cameras: int) -> list[np.ndarray]:
    directory = Path(data_dir)
    return [load_camera_mask(directory / f"Cam{camera_index}.mat", frame) for camera_index in range(1, num_cameras + 1)]


def list_camera_frames(camera_mat_path: str | Path) -> list[int]:
    mat_path = Path(camera_mat_path)
    data = loadmat(mat_path)
    frames: list[int] = []
    for key in data:
        match = _FRAME_KEY_PATTERN.match(key)
        if match is not None:
            frames.append(int(match.group(1)))
    return sorted(frames)


def list_available_frames(data_dir: str | Path, num_cameras: int) -> list[int]:
    directory = Path(data_dir)
    available: set[int] | None = None
    for camera_index in range(1, num_cameras + 1):
        frames = set(list_camera_frames(directory / f"Cam{camera_index}.mat"))
        available = frames if available is None else available & frames
    return sorted(available or set())


def discover_camera_files(folder: str | Path, pattern: str = "C*P.txt") -> list[Path]:
    return sorted(Path(folder).glob(pattern))


def stack_boolean_images(images: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(image, dtype=bool) for image in images]
    if not arrays:
        raise ValueError("At least one image is required.")
    return np.stack(arrays, axis=2)
