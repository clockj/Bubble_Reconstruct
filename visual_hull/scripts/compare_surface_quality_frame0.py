from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

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
    LaplacianSmoothingConfig,
    SoftVisualHullConfig,
    laplacian_smooth_mesh,
    run_full_reconstruction_improved,
)
from visual_hull.test_runs import create_test_run, write_report_markdown  # type: ignore[import-not-found]
from visual_hull.voxel_grid import convert_voxel_list_to_volume  # type: ignore[import-not-found]


def _bubble_ranges(bubbles: np.ndarray) -> list[tuple[int, int]]:
    bubble_array = np.asarray(bubbles)
    if bubble_array.size == 0:
        return []
    return [(int(start) - 1, int(stop)) for start, stop in bubble_array.T]


def _surface_mesh(voxels: np.ndarray, voxel_size: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if voxels.size == 0:
        return None
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


def _mesh_payload(vertices: np.ndarray, faces: np.ndarray) -> dict[str, object]:
    return {
        "vertices": np.asarray(vertices, dtype=np.float64),
        "faces": np.asarray(faces, dtype=np.int64),
        "metrics": _mesh_metrics(vertices, faces),
    }


def _result_metrics(result: FullReconstructionResult) -> dict[str, object]:
    bubble_metrics: list[dict[str, float]] = []
    all_vertices: list[np.ndarray] = []
    total_area = 0.0
    total_mesh_volume = 0.0
    weighted_dihedral_sum = 0.0
    weighted_dihedral_count = 0.0

    for start, stop in _bubble_ranges(result.bubbles):
        bubble_voxels = result.voxels[start:stop]
        mesh = _surface_mesh(bubble_voxels, result.voxel_size_2)
        if mesh is None:
            continue
        vertices, faces = mesh
        metrics = _mesh_metrics(vertices, faces)
        bubble_metrics.append(metrics)
        all_vertices.append(vertices)
        total_area += metrics["surface_area"]
        total_mesh_volume += metrics["mesh_volume"]
        weighted_dihedral_sum += metrics["mean_dihedral_deg"] * metrics["face_count"]
        weighted_dihedral_count += metrics["face_count"]

    overall = {
        "bubble_count": int(result.bubbles.shape[1]) if result.bubbles.ndim == 2 else 0,
        "voxel_count": int(result.voxels.shape[0]),
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

    return {
        "overall": overall,
        "bubbles": bubble_metrics,
    }


def _result_meshes(result: FullReconstructionResult) -> list[dict[str, object]]:
    meshes: list[dict[str, object]] = []
    for start, stop in _bubble_ranges(result.bubbles):
        bubble_voxels = result.voxels[start:stop]
        mesh = _surface_mesh(bubble_voxels, result.voxel_size_2)
        if mesh is None:
            continue
        meshes.append(_mesh_payload(mesh[0], mesh[1]))
    return meshes


def _aggregate_mesh_payloads(meshes: list[dict[str, object]]) -> dict[str, object]:
    bubble_metrics = [item["metrics"] for item in meshes]
    total_area = float(sum(item["surface_area"] for item in bubble_metrics)) if bubble_metrics else 0.0
    total_volume = float(sum(item["mesh_volume"] for item in bubble_metrics)) if bubble_metrics else 0.0
    weighted_dihedral_sum = float(sum(item["mean_dihedral_deg"] * item["face_count"] for item in bubble_metrics)) if bubble_metrics else 0.0
    weighted_dihedral_count = float(sum(item["face_count"] for item in bubble_metrics)) if bubble_metrics else 0.0

    overall = {
        "bubble_count": len(meshes),
        "surface_area": total_area,
        "mesh_volume": total_volume,
        "mean_dihedral_deg": float(weighted_dihedral_sum / max(weighted_dihedral_count, 1.0)),
        "mean_sphericity": float(np.mean([item["sphericity"] for item in bubble_metrics])) if bubble_metrics else 0.0,
    }

    if meshes:
        points = np.vstack([item["vertices"] for item in meshes])
        overall["bbox_min"] = [float(value) for value in np.min(points, axis=0)]
        overall["bbox_max"] = [float(value) for value in np.max(points, axis=0)]
    else:
        overall["bbox_min"] = None
        overall["bbox_max"] = None

    return {"overall": overall, "bubbles": bubble_metrics}


def _surface_distance_stats(baseline: FullReconstructionResult, improved: FullReconstructionResult) -> dict[str, float]:
    baseline_points: list[np.ndarray] = []
    improved_points: list[np.ndarray] = []

    for start, stop in _bubble_ranges(baseline.bubbles):
        mesh = _surface_mesh(baseline.voxels[start:stop], baseline.voxel_size_2)
        if mesh is not None:
            baseline_points.append(mesh[0])

    for start, stop in _bubble_ranges(improved.bubbles):
        mesh = _surface_mesh(improved.voxels[start:stop], improved.voxel_size_2)
        if mesh is not None:
            improved_points.append(mesh[0])

    if not baseline_points or not improved_points:
        return {"mean_nearest_distance": 0.0, "p95_nearest_distance": 0.0, "max_nearest_distance": 0.0}

    baseline_vertices = np.vstack(baseline_points)
    improved_vertices = np.vstack(improved_points)
    tree = cKDTree(baseline_vertices)
    distances, _ = tree.query(improved_vertices, k=1)
    return {
        "mean_nearest_distance": float(np.mean(distances)),
        "p95_nearest_distance": float(np.percentile(distances, 95.0)),
        "max_nearest_distance": float(np.max(distances)),
    }


def _surface_distance_from_meshes(
    baseline_meshes: list[dict[str, object]],
    comparison_meshes: list[dict[str, object]],
) -> dict[str, float]:
    if not baseline_meshes or not comparison_meshes:
        return {"mean_nearest_distance": 0.0, "p95_nearest_distance": 0.0, "max_nearest_distance": 0.0}

    baseline_vertices = np.vstack([item["vertices"] for item in baseline_meshes])
    comparison_vertices = np.vstack([item["vertices"] for item in comparison_meshes])
    tree = cKDTree(baseline_vertices)
    distances, _ = tree.query(comparison_vertices, k=1)
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


def _draw_meshes(ax, meshes: list[dict[str, object]], title: str) -> None:
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(meshes), 1), endpoint=False))
    all_points: list[np.ndarray] = []
    for bubble_index, mesh in enumerate(meshes):
        vertices = np.asarray(mesh["vertices"], dtype=np.float64)
        faces = np.asarray(mesh["faces"], dtype=np.int64)
        tris = vertices[faces]
        collection = Poly3DCollection(tris, alpha=0.8, facecolor=colors[bubble_index % len(colors)], edgecolor="none")
        ax.add_collection3d(collection)
        all_points.append(vertices)
    if all_points:
        points = np.vstack(all_points)
        bbox = np.max(points, axis=0) - np.min(points, axis=0)
        padding = np.maximum(bbox * 0.05, 0.1)
        _set_equal_axes(ax, points, padding)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title(title)


def _save_visualization(
    run_dir: Path,
    baseline_meshes: list[dict[str, object]],
    improved_meshes: list[dict[str, object]],
    smoothed_meshes: list[dict[str, object]],
) -> Path:
    figure = plt.figure(figsize=(24, 8))
    axis_left = figure.add_subplot(131, projection="3d")
    axis_mid = figure.add_subplot(132, projection="3d")
    axis_right = figure.add_subplot(133, projection="3d")
    _draw_meshes(axis_left, baseline_meshes, "Baseline Surface")
    _draw_meshes(axis_mid, improved_meshes, "Improved Surface")
    _draw_meshes(axis_right, smoothed_meshes, "Improved + Laplacian")
    figure.tight_layout()
    output_path = run_dir / "surface_comparison.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    run = create_test_run(PROJECT_ROOT, "surface-quality-frame0")
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

    baseline_meshes = _result_meshes(baseline)
    improved_meshes = _result_meshes(improved)
    smoothed_meshes = [
        _mesh_payload(*laplacian_smooth_mesh(mesh["vertices"], mesh["faces"], config=LaplacianSmoothingConfig()))
        for mesh in improved_meshes
    ]

    baseline_metrics = _result_metrics(baseline)
    improved_metrics = _result_metrics(improved)
    smoothed_metrics = _aggregate_mesh_payloads(smoothed_meshes)
    distance_metrics = _surface_distance_stats(baseline, improved)
    smoothed_distance_metrics = _surface_distance_from_meshes(baseline_meshes, smoothed_meshes)
    image_path = _save_visualization(run.root, baseline_meshes, improved_meshes, smoothed_meshes)

    payload = {
        "baseline": baseline_metrics,
        "improved": improved_metrics,
        "improved_laplacian": smoothed_metrics,
        "surface_distance_vs_baseline": distance_metrics,
        "surface_distance_smoothed_vs_baseline": smoothed_distance_metrics,
        "comparison": {
            "delta_voxel_count": improved_metrics["overall"]["voxel_count"] - baseline_metrics["overall"]["voxel_count"],
            "delta_surface_area": improved_metrics["overall"]["surface_area"] - baseline_metrics["overall"]["surface_area"],
            "delta_mesh_volume": improved_metrics["overall"]["mesh_volume"] - baseline_metrics["overall"]["mesh_volume"],
            "delta_mean_dihedral_deg": improved_metrics["overall"]["mean_dihedral_deg"] - baseline_metrics["overall"]["mean_dihedral_deg"],
            "delta_mean_sphericity": improved_metrics["overall"]["mean_sphericity"] - baseline_metrics["overall"]["mean_sphericity"],
            "delta_smoothed_surface_area": smoothed_metrics["overall"]["surface_area"] - baseline_metrics["overall"]["surface_area"],
            "delta_smoothed_mesh_volume": smoothed_metrics["overall"]["mesh_volume"] - baseline_metrics["overall"]["mesh_volume"],
            "delta_smoothed_mean_dihedral_deg": smoothed_metrics["overall"]["mean_dihedral_deg"] - baseline_metrics["overall"]["mean_dihedral_deg"],
            "delta_smoothed_mean_sphericity": smoothed_metrics["overall"]["mean_sphericity"] - baseline_metrics["overall"]["mean_sphericity"],
        },
        "artifacts": {
            "surface_comparison_png": str(image_path),
        },
    }

    run.write_json("surface_quality_comparison.json", payload)
    run.write_json("report.json", payload)
    write_report_markdown(
        run,
        "Surface Quality Comparison - Frame 0",
        [
            f"Baseline voxel count: {baseline_metrics['overall']['voxel_count']}",
            f"Improved voxel count: {improved_metrics['overall']['voxel_count']}",
            f"Baseline mean dihedral: {baseline_metrics['overall']['mean_dihedral_deg']:.3f} deg",
            f"Improved mean dihedral: {improved_metrics['overall']['mean_dihedral_deg']:.3f} deg",
            f"Smoothed mean dihedral: {smoothed_metrics['overall']['mean_dihedral_deg']:.3f} deg",
            f"Baseline mean sphericity: {baseline_metrics['overall']['mean_sphericity']:.6f}",
            f"Improved mean sphericity: {improved_metrics['overall']['mean_sphericity']:.6f}",
            f"Smoothed mean sphericity: {smoothed_metrics['overall']['mean_sphericity']:.6f}",
            f"Mean nearest surface distance vs baseline: {distance_metrics['mean_nearest_distance']:.4f} mm",
            f"Mean nearest smoothed surface distance vs baseline: {smoothed_distance_metrics['mean_nearest_distance']:.4f} mm",
            f"Visualization: {image_path.name}",
        ],
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()