from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
REFERENCE_FILE = WORKSPACE_ROOT / "Islam_0207" / "Reconstruction" / "Bubble_Frame_000000.mat"

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


def _bubble_ranges(bubbles: np.ndarray) -> list[tuple[int, int]]:
    bubble_array = np.asarray(bubbles)
    if bubble_array.size == 0:
        return []
    return [(int(start) - 1, int(stop)) for start, stop in bubble_array.T]


def _reference_meshes(voxel_size: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    try:
        reference = loadmat(REFERENCE_FILE)
        ref_voxels = np.asarray(reference.get("voxels", np.empty((0, 3))), dtype=np.float64)
        ref_bubbles = np.asarray(reference.get("bubbles", np.empty((2, 0))), dtype=np.int64)
    except NotImplementedError:
        with h5py.File(REFERENCE_FILE, "r") as reference:
            ref_voxels = np.array(reference["voxels"], dtype=np.float64).T
            ref_bubbles = np.array(reference["bubbles"], dtype=np.int64).T

    if ref_bubbles.size == 0:
        mesh = surface_mesh_from_voxels(ref_voxels, voxel_size)
        return [mesh] if mesh is not None else []

    meshes: list[tuple[np.ndarray, np.ndarray]] = []
    for start, stop in _bubble_ranges(ref_bubbles):
        mesh = surface_mesh_from_voxels(ref_voxels[start:stop], voxel_size)
        if mesh is not None:
            meshes.append(mesh)
    return meshes


def _distance(reference_meshes: list[tuple[np.ndarray, np.ndarray]], other_meshes: list[tuple[np.ndarray, np.ndarray]]) -> float:
    if not reference_meshes or not other_meshes:
        return float("inf")
    ref_vertices = np.vstack([vertices for vertices, _ in reference_meshes])
    other_vertices = np.vstack([vertices for vertices, _ in other_meshes])
    distances, _ = cKDTree(ref_vertices).query(other_vertices, k=1)
    return float(np.mean(distances))


def main() -> None:
    run = create_test_run(PROJECT_ROOT, "spherical-harmonic-sweep-frame0")
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
    reference_meshes = _reference_meshes(baseline.voxel_size_2)
    masks = load_camera_masks(inputs.data_dir, inputs.frame, inputs.num_cameras)
    camera_files = discover_camera_files(inputs.calibration_dir)
    cameras = OpenLPTCameraSet.from_camera_files(camera_files[: inputs.num_cameras])

    baseline_ranges = _bubble_ranges(baseline.bubbles)
    baseline_meshes = [
        mesh
        for start, stop in baseline_ranges
        for mesh in [surface_mesh_from_voxels(baseline.voxels[start:stop], baseline.voxel_size_2)]
        if mesh is not None
    ]
    improved_ranges = _bubble_ranges(improved.bubbles)
    improved_meshes = [
        mesh
        for start, stop in improved_ranges
        for mesh in [surface_mesh_from_voxels(improved.voxels[start:stop], improved.voxel_size_2)]
        if mesh is not None
    ]
    baseline_mask_metrics = summarize_mask_overlap(project_meshes_to_camera_masks(baseline_meshes, masks, cameras), masks)
    improved_mask_metrics = summarize_mask_overlap(project_meshes_to_camera_masks(improved_meshes, masks, cameras), masks)

    degree_values = [2, 3, 4, 5, 6]
    regularization_values = [1e-4, 1e-3, 1e-2, 1e-1]
    records: list[dict[str, float | int]] = []

    if not improved_ranges:
        raise ValueError("Improved reconstruction does not contain any bubbles to fit.")

    for degree, regularization in product(degree_values, regularization_values):
        sh_config = SphericalHarmonicFitConfig(max_degree=degree, regularization=regularization)
        sh_meshes: list[tuple[np.ndarray, np.ndarray]] = []
        fit_rmses: list[float] = []
        total_vertex_count = 0
        total_face_count = 0
        for start, stop in improved_ranges:
            fitted = fit_spherical_harmonic_surface_from_voxels(
                improved.voxels[start:stop],
                improved.voxel_size_2,
                config=sh_config,
            )
            if fitted is None:
                continue
            sh_meshes.append((fitted.vertices, fitted.faces))
            fit_rmses.append(float(fitted.fit_rmse))
            total_vertex_count += int(fitted.vertices.shape[0])
            total_face_count += int(fitted.faces.shape[0])
        if not sh_meshes:
            continue
        mask_metrics = summarize_mask_overlap(project_meshes_to_camera_masks(sh_meshes, masks, cameras), masks)
        records.append(
            {
                "max_degree": int(degree),
                "regularization": float(regularization),
                "fit_bubble_count": len(sh_meshes),
                "fit_rmse_mm": float(np.mean(fit_rmses)),
                "mean_distance_to_matlab_reference_mm": _distance(reference_meshes, sh_meshes),
                "mean_mask_iou": float(mask_metrics["overall"]["iou"]),
                "mean_mask_dice": float(mask_metrics["overall"]["dice"]),
                "vertex_count": total_vertex_count,
                "face_count": total_face_count,
            }
        )

    records.sort(
        key=lambda item: (
            -float(item["mean_mask_iou"]),
            float(item["mean_distance_to_matlab_reference_mm"]),
            float(item["fit_rmse_mm"]),
        )
    )
    payload = {
        "reference_file": str(REFERENCE_FILE),
        "degree_values": degree_values,
        "regularization_values": regularization_values,
        "baseline_mask_metrics": baseline_mask_metrics,
        "improved_mask_metrics": improved_mask_metrics,
        "best": records[0] if records else None,
        "results": records,
    }
    run.write_json("report.json", payload)
    run.write_json("spherical_harmonic_sweep.json", payload)
    best = payload["best"]
    lines = [f"Evaluated configs: {len(records)}"]
    if best is not None:
        lines.extend(
            [
                f"Best degree: {best['max_degree']}",
                f"Best regularization: {best['regularization']}",
                f"Best mask IoU: {best['mean_mask_iou']:.4f}",
                f"Best mean distance to MATLAB reference: {best['mean_distance_to_matlab_reference_mm']:.4f} mm",
                f"Best fit RMSE: {best['fit_rmse_mm']:.4f} mm",
            ]
        )
    lines.extend(
        [
            f"Baseline mask IoU: {float(baseline_mask_metrics['overall']['iou']):.4f}",
            f"Improved mask IoU: {float(improved_mask_metrics['overall']['iou']):.4f}",
        ]
    )
    write_report_markdown(run, "Spherical Harmonic Sweep - Frame 0", lines)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()