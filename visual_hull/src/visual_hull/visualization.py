from __future__ import annotations

import os
from typing import Iterable

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from .models import FullReconstructionResult
from .voxel_grid import convert_voxel_list_to_volume


def _ensure_qt_backend() -> None:
    os.environ.setdefault("QT_API", "pyside6")

    import matplotlib

    backend = matplotlib.get_backend().lower()
    if "qt" not in backend:
        matplotlib.use("qtagg", force=True)


def _bubble_ranges(bubbles: np.ndarray) -> list[tuple[int, int]]:
    bubble_array = np.asarray(bubbles)
    if bubble_array.size == 0:
        return []
    return [(int(start) - 1, int(stop)) for start, stop in bubble_array.T]


def _axis_limits(points: np.ndarray, padding: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    mins = np.min(points, axis=0) - padding
    maxs = np.max(points, axis=0) + padding
    return (mins[0], maxs[0]), (mins[1], maxs[1]), (mins[2], maxs[2])


def _set_equal_axes(ax, points: np.ndarray, padding: np.ndarray) -> None:
    x_limits, y_limits, z_limits = _axis_limits(points, padding)
    x_mid = 0.5 * (x_limits[0] + x_limits[1])
    y_mid = 0.5 * (y_limits[0] + y_limits[1])
    z_mid = 0.5 * (z_limits[0] + z_limits[1])
    radius = max(
        0.5 * (x_limits[1] - x_limits[0]),
        0.5 * (y_limits[1] - y_limits[0]),
        0.5 * (z_limits[1] - z_limits[0]),
    )
    ax.set_xlim(x_mid - radius, x_mid + radius)
    ax.set_ylim(y_mid - radius, y_mid + radius)
    ax.set_zlim(z_mid - radius, z_mid + radius)


def _surface_mesh(voxels: np.ndarray, voxel_size: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    grid_x, grid_y, grid_z, volume = convert_voxel_list_to_volume(voxels, voxel_size)
    if np.count_nonzero(volume) == 0:
        return None

    origin = np.array([float(np.min(grid_x)), float(np.min(grid_y)), float(np.min(grid_z))], dtype=np.float64)
    verts, faces, _, _ = marching_cubes(
        volume.astype(np.float32),
        level=0.5,
        spacing=(float(voxel_size[1]), float(voxel_size[0]), float(voxel_size[2])),
    )
    world_verts = np.column_stack(
        (
            origin[0] + verts[:, 1],
            origin[1] + verts[:, 0],
            origin[2] + verts[:, 2],
        )
    )
    return world_verts, faces.astype(np.int64, copy=False)


def _iter_bubble_voxels(result: FullReconstructionResult) -> Iterable[np.ndarray]:
    for start, stop in _bubble_ranges(result.bubbles):
        yield result.voxels[start:stop]


def show_reconstruction_interactive(
    result: FullReconstructionResult,
    *,
    mode: str = "surface",
    point_size: float = 8.0,
    alpha: float = 0.7,
    title: str | None = None,
) -> None:
    _ensure_qt_backend()

    import matplotlib.pyplot as plt

    if result.voxels.size == 0:
        raise ValueError("The reconstruction result does not contain any voxels to visualize.")

    mode_name = mode.lower()
    if mode_name not in {"surface", "scatter"}:
        raise ValueError(f"Unsupported visualization mode: {mode}. Expected 'surface' or 'scatter'.")

    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111, projection="3d")
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(result.bubbles.shape[1], 1), endpoint=False))

    for bubble_index, bubble_voxels in enumerate(_iter_bubble_voxels(result)):
        color = colors[bubble_index % len(colors)]
        if mode_name == "scatter" or bubble_voxels.shape[0] < 4:
            axis.scatter(
                bubble_voxels[:, 0],
                bubble_voxels[:, 1],
                bubble_voxels[:, 2],
                s=point_size,
                alpha=alpha,
                color=color,
                depthshade=True,
                label=f"Bubble {bubble_index + 1}",
            )
            continue

        mesh = _surface_mesh(bubble_voxels, result.voxel_size_2)
        if mesh is None:
            continue

        vertices, faces = mesh
        tris = vertices[faces]
        collection = Poly3DCollection(tris, alpha=alpha, facecolor=color, edgecolor="none")
        axis.add_collection3d(collection)
        axis.plot([], [], color=color, label=f"Bubble {bubble_index + 1}")

    _set_equal_axes(axis, result.voxels, np.asarray(result.voxel_size_2, dtype=np.float64) * 2.0)
    axis.set_xlabel("X [mm]")
    axis.set_ylabel("Y [mm]")
    axis.set_zlabel("Z [mm]")
    axis.set_title(title or "Reconstructed Bubble Shape")
    axis.legend(loc="upper right")
    figure.tight_layout()
    plt.show()