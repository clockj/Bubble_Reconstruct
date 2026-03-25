from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_hull import FullReconstructionResult, build_inputs, run_full_reconstruction  # type: ignore[import-not-found]
from visual_hull.improved import (  # type: ignore[import-not-found]
    ImprovedReconstructionConfig,
    MeshSmoothingConfig,
    SoftVisualHullConfig,
    run_full_reconstruction_improved,
    surface_mesh_from_voxels,
    taubin_smooth_mesh,
)
from visual_hull.test_runs import create_test_run, write_report_markdown  # type: ignore[import-not-found]


def _bubble_ranges(bubbles: np.ndarray) -> list[tuple[int, int]]:
    bubble_array = np.asarray(bubbles)
    if bubble_array.size == 0:
        return []
    return [(int(start) - 1, int(stop)) for start, stop in bubble_array.T]


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tris = vertices[faces]
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tris = vertices[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1e-12)


def _adjacent_face_pairs(faces: np.ndarray) -> np.ndarray:
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for start, stop in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(face[start]), int(face[stop]))))
            edge_to_faces.setdefault(edge, []).append(face_index)

    pairs: list[tuple[int, int]] = []
    for face_indices in edge_to_faces.values():
        if len(face_indices) == 2:
            pairs.append((face_indices[0], face_indices[1]))
    return np.asarray(pairs, dtype=np.int64) if pairs else np.empty((0, 2), dtype=np.int64)


def _mesh_metrics(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float]:
    areas = _triangle_areas(vertices, faces)
    area = float(np.sum(areas))

    tris = vertices[faces]
    signed_volume = np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])) / 6.0
    volume = float(abs(np.sum(signed_volume)))

    normals = _face_normals(vertices, faces)
    face_pairs = _adjacent_face_pairs(faces)
    if face_pairs.size > 0:
        dot = np.sum(normals[face_pairs[:, 0]] * normals[face_pairs[:, 1]], axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        dihedral = np.degrees(np.arccos(dot))
        mean_dihedral = float(np.mean(dihedral))
        std_dihedral = float(np.std(dihedral))
    else:
        mean_dihedral = 0.0
        std_dihedral = 0.0

    sphere_area = (math.pi ** (1.0 / 3.0)) * ((6.0 * max(volume, 1e-12)) ** (2.0 / 3.0))
    sphericity = float(sphere_area / max(area, 1e-12))

    return {
        "vertex_count": float(vertices.shape[0]),
        "face_count": float(faces.shape[0]),
        "surface_area": area,
        "mesh_volume": volume,
        "mean_dihedral_deg": mean_dihedral,
        "std_dihedral_deg": std_dihedral,
        "sphericity": sphericity,
    }


def _mesh_bundle_from_result(
    result: FullReconstructionResult,
    *,
    smoothing: MeshSmoothingConfig | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    meshes: list[tuple[np.ndarray, np.ndarray]] = []
    for start, stop in _bubble_ranges(result.bubbles):
        mesh = surface_mesh_from_voxels(result.voxels[start:stop], result.voxel_size_2)
        if mesh is None:
            continue
        vertices, faces = mesh
        if smoothing is not None:
            vertices, faces = taubin_smooth_mesh(vertices, faces, config=smoothing)
        meshes.append((vertices, faces))
    return meshes


def _bundle_metrics(meshes: list[tuple[np.ndarray, np.ndarray]], voxel_count: int, bubble_count: int) -> dict[str, object]:
    bubble_metrics: list[dict[str, float]] = []
    all_vertices: list[np.ndarray] = []
    total_area = 0.0
    total_mesh_volume = 0.0
    weighted_dihedral_sum = 0.0
    weighted_dihedral_count = 0.0

    for vertices, faces in meshes:
        metrics = _mesh_metrics(vertices, faces)
        bubble_metrics.append(metrics)
        all_vertices.append(vertices)
        total_area += metrics["surface_area"]
        total_mesh_volume += metrics["mesh_volume"]
        weighted_dihedral_sum += metrics["mean_dihedral_deg"] * metrics["face_count"]
        weighted_dihedral_count += metrics["face_count"]

    overall = {
        "bubble_count": int(bubble_count),
        "voxel_count": int(voxel_count),
        "surface_area": float(total_area),
        "mesh_volume": float(total_mesh_volume),
        "mean_dihedral_deg": float(weighted_dihedral_sum / max(weighted_dihedral_count, 1.0)),
        "mean_sphericity": float(np.mean([item["sphericity"] for item in bubble_metrics])) if bubble_metrics else 0.0,
    }
    if all_vertices:
        points = np.vstack(all_vertices)
        overall["bbox_min"] = [float(value) for value in np.min(points, axis=0)]
        overall["bbox_max"] = [float(value) for value in np.max(points, axis=0)]
    else:
        overall["bbox_min"] = None
        overall["bbox_max"] = None
    return {"overall": overall, "bubbles": bubble_metrics}


def _distance_to_baseline(baseline_meshes: list[tuple[np.ndarray, np.ndarray]], other_meshes: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    if not baseline_meshes or not other_meshes:
        return {"mean_nearest_distance": 0.0, "p95_nearest_distance": 0.0, "max_nearest_distance": 0.0}
    baseline_vertices = np.vstack([vertices for vertices, _ in baseline_meshes])
    other_vertices = np.vstack([vertices for vertices, _ in other_meshes])
    tree = cKDTree(baseline_vertices)
    distances, _ = tree.query(other_vertices, k=1)
    return {
        "mean_nearest_distance": float(np.mean(distances)),
        "p95_nearest_distance": float(np.percentile(distances, 95.0)),
        "max_nearest_distance": float(np.max(distances)),
    }


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


def _draw_meshes(ax, meshes: list[tuple[np.ndarray, np.ndarray]], title: str) -> None:
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(meshes), 1), endpoint=False))
    points: list[np.ndarray] = []
    for index, (vertices, faces) in enumerate(meshes):
        tris = vertices[faces]
        collection = Poly3DCollection(tris, alpha=0.8, facecolor=colors[index % len(colors)], edgecolor="none")
        ax.add_collection3d(collection)
        points.append(vertices)
    if points:
        stacked = np.vstack(points)
        _set_equal_axes(ax, stacked, np.array([0.6, 0.6, 0.6], dtype=np.float64))
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title(title)


def _save_visualization(
    run_dir: Path,
    baseline_meshes: list[tuple[np.ndarray, np.ndarray]],
    improved_meshes: list[tuple[np.ndarray, np.ndarray]],
    smoothed_meshes: list[tuple[np.ndarray, np.ndarray]],
) -> Path:
    figure = plt.figure(figsize=(24, 8))
    axis_left = figure.add_subplot(131, projection="3d")
    axis_mid = figure.add_subplot(132, projection="3d")
    axis_right = figure.add_subplot(133, projection="3d")
    _draw_meshes(axis_left, baseline_meshes, "Baseline Surface")
    _draw_meshes(axis_mid, improved_meshes, "Improved Surface")
    _draw_meshes(axis_right, smoothed_meshes, "Smoothed Improved Surface")
    figure.tight_layout()
    output_path = run_dir / "smoothed_surface_comparison.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    run = create_test_run(PROJECT_ROOT, "smoothed-surface-frame0")
    inputs = build_inputs(
        data_dir=WORKSPACE_ROOT / "Islam_0207",
        calibration_dir=WORKSPACE_ROOT / "Islam_0207",
        frame=0,
        num_cameras=3,
        voxel_size=[0.3, 0.3, 0.3],
        limits=[10, 30, -5, 5, -5, 5],
    )

    baseline = run_full_reconstruction(inputs)
    improved = run_full_reconstruction_improved(
        inputs,
        config=ImprovedReconstructionConfig(hull=SoftVisualHullConfig()),
    )

    smoothing = MeshSmoothingConfig()
    baseline_meshes = _mesh_bundle_from_result(baseline)
    improved_meshes = _mesh_bundle_from_result(improved)
    smoothed_meshes = _mesh_bundle_from_result(improved, smoothing=smoothing)

    baseline_metrics = _bundle_metrics(
        baseline_meshes,
        voxel_count=int(baseline.voxels.shape[0]),
        bubble_count=int(baseline.bubbles.shape[1]) if baseline.bubbles.ndim == 2 else 0,
    )
    improved_metrics = _bundle_metrics(
        improved_meshes,
        voxel_count=int(improved.voxels.shape[0]),
        bubble_count=int(improved.bubbles.shape[1]) if improved.bubbles.ndim == 2 else 0,
    )
    smoothed_metrics = _bundle_metrics(
        smoothed_meshes,
        voxel_count=int(improved.voxels.shape[0]),
        bubble_count=int(improved.bubbles.shape[1]) if improved.bubbles.ndim == 2 else 0,
    )
    improved_distance = _distance_to_baseline(baseline_meshes, improved_meshes)
    smoothed_distance = _distance_to_baseline(baseline_meshes, smoothed_meshes)
    image_path = _save_visualization(run.root, baseline_meshes, improved_meshes, smoothed_meshes)

    payload = {
        "smoothing_config": {
            "iterations": smoothing.iterations,
            "lambda_factor": smoothing.lambda_factor,
            "mu_factor": smoothing.mu_factor,
        },
        "baseline": baseline_metrics,
        "improved": improved_metrics,
        "smoothed_improved": smoothed_metrics,
        "surface_distance_vs_baseline": {
            "improved": improved_distance,
            "smoothed_improved": smoothed_distance,
        },
        "comparison": {
            "improved_delta_mean_dihedral_deg": improved_metrics["overall"]["mean_dihedral_deg"] - baseline_metrics["overall"]["mean_dihedral_deg"],
            "smoothed_delta_mean_dihedral_deg": smoothed_metrics["overall"]["mean_dihedral_deg"] - baseline_metrics["overall"]["mean_dihedral_deg"],
            "improved_delta_mean_sphericity": improved_metrics["overall"]["mean_sphericity"] - baseline_metrics["overall"]["mean_sphericity"],
            "smoothed_delta_mean_sphericity": smoothed_metrics["overall"]["mean_sphericity"] - baseline_metrics["overall"]["mean_sphericity"],
            "improved_delta_surface_area": improved_metrics["overall"]["surface_area"] - baseline_metrics["overall"]["surface_area"],
            "smoothed_delta_surface_area": smoothed_metrics["overall"]["surface_area"] - baseline_metrics["overall"]["surface_area"],
        },
        "artifacts": {
            "smoothed_surface_comparison_png": str(image_path),
        },
    }

    run.write_json("smoothed_surface_comparison.json", payload)
    run.write_json("report.json", payload)
    write_report_markdown(
        run,
        "Smoothed Surface Comparison - Frame 0",
        [
            f"Baseline mean dihedral: {baseline_metrics['overall']['mean_dihedral_deg']:.3f} deg",
            f"Improved mean dihedral: {improved_metrics['overall']['mean_dihedral_deg']:.3f} deg",
            f"Smoothed mean dihedral: {smoothed_metrics['overall']['mean_dihedral_deg']:.3f} deg",
            f"Baseline mean sphericity: {baseline_metrics['overall']['mean_sphericity']:.6f}",
            f"Improved mean sphericity: {improved_metrics['overall']['mean_sphericity']:.6f}",
            f"Smoothed mean sphericity: {smoothed_metrics['overall']['mean_sphericity']:.6f}",
            f"Improved mean nearest distance vs baseline: {improved_distance['mean_nearest_distance']:.4f} mm",
            f"Smoothed mean nearest distance vs baseline: {smoothed_distance['mean_nearest_distance']:.4f} mm",
            f"Visualization: {image_path.name}",
        ],
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()