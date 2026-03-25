from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree
from scipy.io import loadmat

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_hull import OpenLPTCameraSet, build_inputs, run_full_reconstruction  # type: ignore[import-not-found]
from visual_hull.improved import (  # type: ignore[import-not-found]
    ImprovedReconstructionConfig,
    SoftVisualHullConfig,
    SphericalHarmonicFitConfig,
    fit_spherical_harmonic_surface_from_voxels,
    run_full_reconstruction_improved,
    surface_mesh_from_voxels,
)
from visual_hull.io import discover_camera_files, load_camera_masks  # type: ignore[import-not-found]
from visual_hull.silhouette_metrics import (  # type: ignore[import-not-found]
    project_meshes_to_camera_masks,
    summarize_mask_overlap,
)
from visual_hull.test_runs import create_test_run, write_report_markdown  # type: ignore[import-not-found]

REFERENCE_FILE = WORKSPACE_ROOT / "Islam_0207" / "Reconstruction" / "Bubble_Frame_000000.mat"


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


def _result_meshes(result) -> list[tuple[np.ndarray, np.ndarray]]:
    meshes: list[tuple[np.ndarray, np.ndarray]] = []
    for start, stop in _bubble_ranges(result.bubbles):
        mesh = surface_mesh_from_voxels(result.voxels[start:stop], result.voxel_size_2)
        if mesh is not None:
            meshes.append(mesh)
    return meshes


def _reference_payload(reference_file: Path) -> dict[str, np.ndarray | str]:
    try:
        reference = loadmat(reference_file)
        return {
            "voxels": np.asarray(reference.get("voxels", np.empty((0, 3))), dtype=np.float64),
            "bubbles": np.asarray(reference.get("bubbles", np.empty((2, 0))), dtype=np.int64),
            "loader": "scipy.io.loadmat",
        }
    except NotImplementedError:
        with h5py.File(reference_file, "r") as reference:
            return {
                "voxels": np.array(reference["voxels"], dtype=np.float64).T,
                "bubbles": np.array(reference["bubbles"], dtype=np.int64).T,
                "loader": "h5py",
            }


def _reference_meshes(reference_file: Path, voxel_size: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    payload = _reference_payload(reference_file)
    ref_voxels = np.asarray(payload["voxels"], dtype=np.float64)
    ref_bubbles = np.asarray(payload["bubbles"], dtype=np.int64)
    loader = str(payload["loader"])
    meshes: list[tuple[np.ndarray, np.ndarray]] = []
    if ref_bubbles.size == 0:
        mesh = surface_mesh_from_voxels(ref_voxels, voxel_size)
        if mesh is not None:
            meshes.append(mesh)
        return meshes, loader

    for start, stop in _bubble_ranges(ref_bubbles):
        mesh = surface_mesh_from_voxels(ref_voxels[start:stop], voxel_size)
        if mesh is not None:
            meshes.append(mesh)
    return meshes, loader


def _sh_meshes(result, config: SphericalHarmonicFitConfig) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[float]]:
    meshes: list[tuple[np.ndarray, np.ndarray]] = []
    rmses: list[float] = []
    for start, stop in _bubble_ranges(result.bubbles):
        fitted = fit_spherical_harmonic_surface_from_voxels(result.voxels[start:stop], result.voxel_size_2, config=config)
        if fitted is None:
            continue
        meshes.append((fitted.vertices, fitted.faces))
        rmses.append(float(fitted.fit_rmse))
    return meshes, rmses


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


def _set_equal_axes(ax, points: np.ndarray, padding: np.ndarray) -> None:
    mins = np.min(points, axis=0) - padding
    maxs = np.max(points, axis=0) + padding
    mid = 0.5 * (mins + maxs)
    radius = float(np.max(maxs - mins) * 0.5)
    ax.set_xlim(mid[0] - radius, mid[0] + radius)
    ax.set_ylim(mid[1] - radius, mid[1] + radius)
    ax.set_zlim(mid[2] - radius, mid[2] + radius)


def _draw_meshes(ax, meshes: list[tuple[np.ndarray, np.ndarray]], title: str) -> None:
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(meshes), 1), endpoint=False))
    points: list[np.ndarray] = []
    for index, (vertices, faces) in enumerate(meshes):
        tris = vertices[faces]
        collection = Poly3DCollection(tris, alpha=0.82, facecolor=colors[index % len(colors)], edgecolor="none")
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
    sh_meshes: list[tuple[np.ndarray, np.ndarray]],
) -> Path:
    figure = plt.figure(figsize=(24, 8))
    axis_left = figure.add_subplot(131, projection="3d")
    axis_mid = figure.add_subplot(132, projection="3d")
    axis_right = figure.add_subplot(133, projection="3d")
    _draw_meshes(axis_left, baseline_meshes, "Baseline")
    _draw_meshes(axis_mid, improved_meshes, "Improved Raw")
    _draw_meshes(axis_right, sh_meshes, "Improved + SH")
    figure.tight_layout()
    output_path = run_dir / "spherical_harmonic_comparison.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _mask_delta_rgb(reference_mask: np.ndarray, predicted_mask: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference_mask, dtype=bool)
    predicted = np.asarray(predicted_mask, dtype=bool)
    rgb = np.zeros(reference.shape + (3,), dtype=np.float64)
    overlap = reference & predicted
    missed = reference & ~predicted
    extra = ~reference & predicted
    rgb[overlap] = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    rgb[missed] = np.array([0.92, 0.25, 0.25], dtype=np.float64)
    rgb[extra] = np.array([0.25, 0.55, 1.0], dtype=np.float64)
    return rgb


def _save_mask_comparison_visualization(
    run_dir: Path,
    masks: list[np.ndarray],
    comparisons: list[tuple[str, list[np.ndarray], dict[str, object]]],
) -> Path:
    column_count = len(comparisons) + 1
    figure, axes = plt.subplots(len(masks), column_count, figsize=(4.5 * column_count, 4.0 * len(masks)), squeeze=False)

    for row_index, mask in enumerate(masks):
        ground_truth_axis = axes[row_index, 0]
        ground_truth_axis.imshow(mask, cmap="gray", interpolation="nearest")
        ground_truth_axis.set_title(f"Camera {row_index + 1}\nGround Truth")
        ground_truth_axis.set_axis_off()

        for column_index, (label, predicted_masks, summary) in enumerate(comparisons, start=1):
            metrics = summary["per_camera"][row_index]
            axis = axes[row_index, column_index]
            axis.imshow(_mask_delta_rgb(mask, predicted_masks[row_index]), interpolation="nearest")
            axis.set_title(
                f"{label}\nIoU {float(metrics['iou']):.3f} | Dice {float(metrics['dice']):.3f}"
            )
            axis.set_axis_off()

    figure.tight_layout()
    output_path = run_dir / "spherical_harmonic_mask_comparison.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    run = create_test_run(PROJECT_ROOT, "spherical-harmonic-frame0")
    inputs = build_inputs(
        data_dir=WORKSPACE_ROOT / "Islam_0207",
        calibration_dir=WORKSPACE_ROOT / "Islam_0207",
        frame=0,
        num_cameras=3,
        voxel_size=[0.3, 0.3, 0.3],
        limits=[10, 30, -5, 5, -5, 5],
    )

    baseline = run_full_reconstruction(inputs)
    improved = run_full_reconstruction_improved(inputs, config=ImprovedReconstructionConfig(hull=SoftVisualHullConfig()))
    sh_config = SphericalHarmonicFitConfig()
    masks = load_camera_masks(inputs.data_dir, inputs.frame, inputs.num_cameras)
    camera_files = discover_camera_files(inputs.calibration_dir)
    cameras = OpenLPTCameraSet.from_camera_files(camera_files[: inputs.num_cameras])

    baseline_meshes = _result_meshes(baseline)
    improved_meshes = _result_meshes(improved)
    sh_meshes, fit_rmses = _sh_meshes(improved, sh_config)
    reference_meshes, reference_loader = _reference_meshes(REFERENCE_FILE, baseline.voxel_size_2)

    baseline_projected_masks = project_meshes_to_camera_masks(baseline_meshes, masks, cameras)
    improved_projected_masks = project_meshes_to_camera_masks(improved_meshes, masks, cameras)
    sh_projected_masks = project_meshes_to_camera_masks(sh_meshes, masks, cameras)
    baseline_mask_metrics = summarize_mask_overlap(baseline_projected_masks, masks)
    improved_mask_metrics = summarize_mask_overlap(improved_projected_masks, masks)
    sh_mask_metrics = summarize_mask_overlap(sh_projected_masks, masks)

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
    sh_metrics = _bundle_metrics(
        sh_meshes,
        voxel_count=int(improved.voxels.shape[0]),
        bubble_count=len(sh_meshes),
    )
    reference_metrics = _bundle_metrics(
        reference_meshes,
        voxel_count=0,
        bubble_count=len(reference_meshes),
    )
    improved_distance = _distance_to_baseline(baseline_meshes, improved_meshes)
    sh_distance = _distance_to_baseline(baseline_meshes, sh_meshes)
    improved_distance_to_reference = _distance_to_baseline(reference_meshes, improved_meshes)
    sh_distance_to_reference = _distance_to_baseline(reference_meshes, sh_meshes)
    baseline_distance_to_reference = _distance_to_baseline(reference_meshes, baseline_meshes)
    image_path = _save_visualization(run.root, baseline_meshes, improved_meshes, sh_meshes)
    mask_image_path = _save_mask_comparison_visualization(
        run.root,
        masks,
        [
            ("Baseline", baseline_projected_masks, baseline_mask_metrics),
            ("Improved Raw", improved_projected_masks, improved_mask_metrics),
            ("Improved + SH", sh_projected_masks, sh_mask_metrics),
        ],
    )

    payload = {
        "spherical_harmonic_config": sh_config.to_dict(),
        "baseline": baseline_metrics,
        "improved": improved_metrics,
        "spherical_harmonic": sh_metrics,
        "silhouette_overlap_vs_masks": {
            "baseline": baseline_mask_metrics,
            "improved": improved_mask_metrics,
            "spherical_harmonic": sh_mask_metrics,
        },
        "matlab_reference": {
            "loader": reference_loader,
            "metrics": reference_metrics,
        },
        "fit_rmse_mm": [float(value) for value in fit_rmses],
        "surface_distance_vs_baseline": {
            "improved": improved_distance,
            "spherical_harmonic": sh_distance,
        },
        "surface_distance_vs_matlab_reference": {
            "baseline": baseline_distance_to_reference,
            "improved": improved_distance_to_reference,
            "spherical_harmonic": sh_distance_to_reference,
        },
        "comparison": {
            "improved_delta_mean_dihedral_deg": improved_metrics["overall"]["mean_dihedral_deg"] - baseline_metrics["overall"]["mean_dihedral_deg"],
            "spherical_harmonic_delta_mean_dihedral_deg": sh_metrics["overall"]["mean_dihedral_deg"] - baseline_metrics["overall"]["mean_dihedral_deg"],
            "improved_delta_mean_sphericity": improved_metrics["overall"]["mean_sphericity"] - baseline_metrics["overall"]["mean_sphericity"],
            "spherical_harmonic_delta_mean_sphericity": sh_metrics["overall"]["mean_sphericity"] - baseline_metrics["overall"]["mean_sphericity"],
            "improved_delta_surface_area": improved_metrics["overall"]["surface_area"] - baseline_metrics["overall"]["surface_area"],
            "spherical_harmonic_delta_surface_area": sh_metrics["overall"]["surface_area"] - baseline_metrics["overall"]["surface_area"],
            "improved_delta_mask_iou": improved_mask_metrics["overall"]["iou"] - baseline_mask_metrics["overall"]["iou"],
            "spherical_harmonic_delta_mask_iou": sh_mask_metrics["overall"]["iou"] - baseline_mask_metrics["overall"]["iou"],
            "improved_reference_distance_delta_mm": improved_distance_to_reference["mean_nearest_distance"] - baseline_distance_to_reference["mean_nearest_distance"],
            "spherical_harmonic_reference_distance_delta_mm": sh_distance_to_reference["mean_nearest_distance"] - baseline_distance_to_reference["mean_nearest_distance"],
        },
        "artifacts": {
            "spherical_harmonic_comparison_png": str(image_path),
            "spherical_harmonic_mask_comparison_png": str(mask_image_path),
        },
    }

    run.write_json("report.json", payload)
    run.write_json("spherical_harmonic_comparison.json", payload)
    write_report_markdown(
        run,
        "Spherical Harmonic Comparison - Frame 0",
        [
            f"Baseline mean dihedral: {baseline_metrics['overall']['mean_dihedral_deg']:.3f} deg",
            f"Improved mean dihedral: {improved_metrics['overall']['mean_dihedral_deg']:.3f} deg",
            f"Spherical harmonic mean dihedral: {sh_metrics['overall']['mean_dihedral_deg']:.3f} deg",
            f"Baseline mean sphericity: {baseline_metrics['overall']['mean_sphericity']:.6f}",
            f"Improved mean sphericity: {improved_metrics['overall']['mean_sphericity']:.6f}",
            f"Spherical harmonic mean sphericity: {sh_metrics['overall']['mean_sphericity']:.6f}",
            f"Improved mean nearest distance vs baseline: {improved_distance['mean_nearest_distance']:.4f} mm",
            f"Spherical harmonic mean nearest distance vs baseline: {sh_distance['mean_nearest_distance']:.4f} mm",
            f"Baseline mask IoU: {float(baseline_mask_metrics['overall']['iou']):.4f}",
            f"Improved mask IoU: {float(improved_mask_metrics['overall']['iou']):.4f}",
            f"Spherical harmonic mask IoU: {float(sh_mask_metrics['overall']['iou']):.4f}",
            f"Baseline mean nearest distance vs MATLAB reference: {baseline_distance_to_reference['mean_nearest_distance']:.4f} mm",
            f"Improved mean nearest distance vs MATLAB reference: {improved_distance_to_reference['mean_nearest_distance']:.4f} mm",
            f"Spherical harmonic mean nearest distance vs MATLAB reference: {sh_distance_to_reference['mean_nearest_distance']:.4f} mm",
            f"Mean SH fit RMSE: {float(np.mean(fit_rmses)) if fit_rmses else 0.0:.4f} mm",
            f"Visualization: {image_path.name}",
            f"Mask comparison: {mask_image_path.name}",
        ],
    )
    print(json.dumps(payload, indent=2))



if __name__ == "__main__":
    main()