from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .models import FullReconstructionResult
from .surface_utils import surface_mesh_from_voxels


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


def _iter_bubble_voxels(result: FullReconstructionResult) -> Iterable[np.ndarray]:
    for start, stop in _bubble_ranges(result.bubbles):
        yield result.voxels[start:stop]


def _draw_reconstruction(
    axis,
    result: FullReconstructionResult,
    *,
    mode: str,
    point_size: float,
    alpha: float,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(result.bubbles.shape[1], 1), endpoint=False))

    for bubble_index, bubble_voxels in enumerate(_iter_bubble_voxels(result)):
        color = colors[bubble_index % len(colors)]
        if mode == "scatter" or bubble_voxels.shape[0] < 4:
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

        mesh = surface_mesh_from_voxels(bubble_voxels, result.voxel_size_2)
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
    axis.set_title(title)
    axis.legend(loc="upper right")


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
    _draw_reconstruction(
        axis,
        result,
        mode=mode_name,
        point_size=point_size,
        alpha=alpha,
        title=title or "Reconstructed Bubble Shape",
    )
    figure.tight_layout()
    plt.show()


def show_reconstruction_comparison_interactive(
    results: Sequence[tuple[str, FullReconstructionResult]],
    *,
    mode: str = "surface",
    point_size: float = 8.0,
    alpha: float = 0.7,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    _ensure_qt_backend()

    import matplotlib.pyplot as plt

    result_items = list(results)
    if len(result_items) < 2:
        raise ValueError("At least two reconstruction results are required for comparison visualization.")

    mode_name = mode.lower()
    if mode_name not in {"surface", "scatter"}:
        raise ValueError(f"Unsupported visualization mode: {mode}. Expected 'surface' or 'scatter'.")

    for label, result in result_items:
        if result.voxels.size == 0:
            raise ValueError(f"The reconstruction result {label!r} does not contain any voxels to visualize.")

    figure = plt.figure(figsize=(8.0 * len(result_items), 8))
    if title is not None:
        figure.suptitle(title)

    for index, (label, result) in enumerate(result_items, start=1):
        axis = figure.add_subplot(1, len(result_items), index, projection="3d")
        _draw_reconstruction(
            axis,
            result,
            mode=mode_name,
            point_size=point_size,
            alpha=alpha,
            title=label,
        )

    figure.tight_layout()
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)