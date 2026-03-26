from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_hull import OpenLPTCameraSet, build_inputs  # type: ignore[import-not-found]
from visual_hull.improved import (  # type: ignore[import-not-found]
    ImprovedReconstructionConfig,
    SoftVisualHullConfig,
    SphericalHarmonicFitConfig,
    fit_spherical_harmonic_surface_from_voxels,
    run_full_reconstruction_improved,
)
from visual_hull.io import discover_camera_files, load_camera_masks  # type: ignore[import-not-found]
from visual_hull.silhouette_metrics import project_meshes_to_camera_masks, summarize_mask_overlap  # type: ignore[import-not-found]
from visual_hull.test_runs import create_test_run, write_report_markdown  # type: ignore[import-not-found]


def _bubble_ranges(bubbles: np.ndarray) -> list[tuple[int, int]]:
    bubble_array = np.asarray(bubbles)
    if bubble_array.size == 0:
        return []
    return [(int(start) - 1, int(stop)) for start, stop in bubble_array.T]


def _mask_iou(mesh_vertices: np.ndarray, mesh_faces: np.ndarray, masks: list[np.ndarray], cameras: OpenLPTCameraSet) -> float:
    predicted_masks = project_meshes_to_camera_masks([(mesh_vertices, mesh_faces)], masks, cameras)
    return float(summarize_mask_overlap(predicted_masks, masks)["overall"]["iou"])


def main() -> None:
    run = create_test_run(PROJECT_ROOT, "benchmark-sh-degree-scaling-frame0")
    inputs = build_inputs(
        data_dir=WORKSPACE_ROOT / "Islam_0207",
        calibration_dir=WORKSPACE_ROOT / "Islam_0207",
        frame=0,
        num_cameras=3,
        voxel_size=[0.3, 0.3, 0.3],
        limits=[10, 30, -5, 5, -5, 5],
    )
    improved = run_full_reconstruction_improved(inputs, config=ImprovedReconstructionConfig(hull=SoftVisualHullConfig()))
    masks = load_camera_masks(inputs.data_dir, inputs.frame, inputs.num_cameras)
    camera_files = discover_camera_files(inputs.calibration_dir)
    cameras = OpenLPTCameraSet.from_camera_files(camera_files[: inputs.num_cameras])

    improved_ranges = _bubble_ranges(improved.bubbles)
    if not improved_ranges:
        raise ValueError("Improved reconstruction does not contain any bubbles to fit.")
    start, stop = improved_ranges[0]
    bubble_voxels = improved.voxels[start:stop]

    degrees = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28]
    silhouette_degrees = {4, 6, 8, 10}
    raw_regularization = 1e-3
    silhouette_regularization = 1e-1
    records: list[dict[str, float | int | None]] = []

    for degree in degrees:
        coefficient_count = (degree + 1) ** 2

        raw_config = SphericalHarmonicFitConfig(max_degree=degree, regularization=raw_regularization)
        start_time = perf_counter()
        raw_fit = fit_spherical_harmonic_surface_from_voxels(
            bubble_voxels,
            improved.voxel_size_2,
            config=raw_config,
        )
        raw_time = perf_counter() - start_time
        if raw_fit is None:
            continue

        silhouette_fit = None
        silhouette_time = None
        if degree in silhouette_degrees:
            silhouette_config = SphericalHarmonicFitConfig(
                max_degree=degree,
                regularization=silhouette_regularization,
                silhouette_enabled=True,
                silhouette_weight=0.25,
                silhouette_max_passes=2,
                silhouette_step_scale=0.05,
                silhouette_top_k=min(12, coefficient_count),
                coefficient_drift_weight=0.1,
            )
            start_time = perf_counter()
            silhouette_fit = fit_spherical_harmonic_surface_from_voxels(
                bubble_voxels,
                improved.voxel_size_2,
                config=silhouette_config,
                masks=masks,
                cameras=cameras,
            )
            silhouette_time = perf_counter() - start_time

        records.append(
            {
                "max_degree": degree,
                "coefficient_count": coefficient_count,
                "raw_time_seconds": raw_time,
                "raw_fit_rmse_mm": float(raw_fit.fit_rmse),
                "raw_mask_iou": _mask_iou(raw_fit.vertices, raw_fit.faces, masks, cameras),
                "silhouette_time_seconds": silhouette_time,
                "silhouette_fit_rmse_mm": float(silhouette_fit.fit_rmse) if silhouette_fit is not None else None,
                "silhouette_mask_iou": float(silhouette_fit.silhouette_iou) if silhouette_fit is not None and silhouette_fit.silhouette_iou is not None else None,
                "silhouette_objective": float(silhouette_fit.objective_value) if silhouette_fit is not None and silhouette_fit.objective_value is not None else None,
                "silhouette_evaluations": int(silhouette_fit.evaluation_count) if silhouette_fit is not None else 0,
            }
        )

    saturation_degree_delta_1e3 = None
    saturation_degree_delta_5e4 = None
    for previous, current in zip(records, records[1:]):
        delta_iou = float(current["raw_mask_iou"]) - float(previous["raw_mask_iou"])
        current["raw_mask_iou_delta_from_previous"] = delta_iou
        if saturation_degree_delta_1e3 is None and delta_iou < 1e-3:
            saturation_degree_delta_1e3 = int(current["max_degree"])
        if saturation_degree_delta_5e4 is None and delta_iou < 5e-4:
            saturation_degree_delta_5e4 = int(current["max_degree"])

    best_raw = max(records, key=lambda item: float(item["raw_mask_iou"])) if records else None
    payload = {
        "degrees": degrees,
        "silhouette_degrees": sorted(silhouette_degrees),
        "raw_regularization": raw_regularization,
        "silhouette_regularization": silhouette_regularization,
        "best_raw": best_raw,
        "saturation_degree_delta_iou_lt_1e3": saturation_degree_delta_1e3,
        "saturation_degree_delta_iou_lt_5e4": saturation_degree_delta_5e4,
        "results": records,
    }
    run.write_json("report.json", payload)
    run.write_json("benchmark_spherical_harmonic_degree_scaling.json", payload)
    lines = []
    if best_raw is not None:
        lines.append(
            f"Best raw degree by mask IoU: {best_raw['max_degree']} (IoU {float(best_raw['raw_mask_iou']):.4f})"
        )
    lines.append(f"Saturation degree for delta IoU < 1e-3: {saturation_degree_delta_1e3}")
    lines.append(f"Saturation degree for delta IoU < 5e-4: {saturation_degree_delta_5e4}")
    for record in records:
        lines.append(
            "Degree {degree}: coeffs={coeffs}, raw={raw:.3f}s, raw IoU={raw_iou:.4f}, delta={delta:+.4f}, silhouette={sil:.3f}s, silhouette IoU={sil_iou:.4f}".format(
                degree=record["max_degree"],
                coeffs=record["coefficient_count"],
                raw=record["raw_time_seconds"],
                raw_iou=record["raw_mask_iou"],
                delta=record.get("raw_mask_iou_delta_from_previous", float("nan")),
                sil=record["silhouette_time_seconds"] if record["silhouette_time_seconds"] is not None else float("nan"),
                sil_iou=record["silhouette_mask_iou"] if record["silhouette_mask_iou"] is not None else float("nan"),
            )
        )
    write_report_markdown(run, "Benchmark SH Degree Scaling - Frame 0", lines)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()