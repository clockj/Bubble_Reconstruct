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


def load_tiff_mask(tiff_path: str | Path) -> np.ndarray:
    """Load a single TIFF file and convert to a 2D boolean mask.

    Handles RGBA, RGB, and grayscale inputs.  Any non-zero pixel is
    treated as foreground.
    """
    import matplotlib.pyplot as plt

    img = plt.imread(str(tiff_path))
    if img.ndim == 3 and img.shape[2] >= 3:
        # RGBA / RGB → grayscale via luminance
        if img.dtype == np.uint8:
            img = img.astype(np.float64) / 255.0
        gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    elif img.ndim == 3 and img.shape[2] == 1:
        gray = img[:, :, 0]
    else:
        gray = img
    return gray > 0.0


def load_tiff_masks(
    mask_dir: str | Path,
    frame: int,
    num_cameras: int,
    *,
    camera_base: int = 0,
    name_template: str = "img{frame:06d}.tif",
    subdir_template: str = "cam{camera}",
) -> list[np.ndarray]:
    """Load a single frame of TIFF masks for all cameras.

    Parameters
    ----------
    mask_dir:
        Root directory containing per-camera subdirectories.
    frame:
        Frame number to load.
    num_cameras:
        Number of cameras.
    camera_base:
        Starting camera index (0 for cam0, 1 for Cam1).
    name_template:
        Python format string for the TIFF filename.  Receives ``frame``.
    subdir_template:
        Python format string for per-camera subdir.  Receives ``camera``.

    Returns
    -------
    list of 2D boolean ndarray, one per camera, in camera-index order.
    """
    root = Path(mask_dir)
    masks: list[np.ndarray] = []
    for cam_idx in range(camera_base, camera_base + num_cameras):
        subdir = subdir_template.format(camera=cam_idx)
        fname = name_template.format(frame=frame)
        path = root / subdir / fname
        if not path.is_file():
            raise FileNotFoundError(f"Mask file not found: {path}")
        masks.append(load_tiff_mask(path))
    return masks


def stack_boolean_images(images: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(image, dtype=bool) for image in images]
    if not arrays:
        raise ValueError("At least one image is required.")
    return np.stack(arrays, axis=2)
