from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_hull import build_inputs, run_full_reconstruction  # type: ignore[import-not-found]
from visual_hull.improved import (  # type: ignore[import-not-found]
    ImprovedReconstructionConfig,
    LaplacianSmoothingConfig,
    SoftVisualHullConfig,
    laplacian_smooth_mesh,
    run_full_reconstruction_improved,
    surface_mesh_from_voxels,
)
from visual_hull.io import list_available_frames  # type: ignore[import-not-found]
from visual_hull.test_runs import create_test_run, write_report_markdown  # type: ignore[import-not-found]


def _parse_frames(raw_frames: str | None, available_frames: list[int], max_frames: int) -> list[int]:
    if raw_frames:
        requested = [int(value.strip()) for value in raw_frames.split(",") if value.strip()]
        missing = [frame for frame in requested if frame not in available_frames]
        if missing:
            raise ValueError(f"Requested frames are not available in every camera file: {missing}")
        return requested
    return available_frames[: max(int(max_frames), 1)]


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


def _mesh_bundle(result, smoothing: LaplacianSmoothingConfig | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    meshes: list[tuple[np.ndarray, np.ndarray]] = []
    for start, stop in _bubble_ranges(result.bubbles):
        mesh = surface_mesh_from_voxels(result.voxels[start:stop], result.voxel_size_2)
        if mesh is None:
            continue
        vertices, faces = mesh
        if smoothing is not None:
            vertices, faces = laplacian_smooth_mesh(vertices, faces, config=smoothing)
        meshes.append((vertices, faces))
    return meshes


def _bundle_metrics(meshes: list[tuple[np.ndarray, np.ndarray]], voxel_count: int, bubble_count: int) -> dict[str, float | int | list[float] | None]:
    bubble_metrics = [_mesh_metrics(vertices, faces) for vertices, faces in meshes]
    total_area = float(sum(item["surface_area"] for item in bubble_metrics)) if bubble_metrics else 0.0
    total_volume = float(sum(item["mesh_volume"] for item in bubble_metrics)) if bubble_metrics else 0.0
    weighted_dihedral_sum = float(sum(item["mean_dihedral_deg"] * item["face_count"] for item in bubble_metrics)) if bubble_metrics else 0.0
    weighted_dihedral_count = float(sum(item["face_count"] for item in bubble_metrics)) if bubble_metrics else 0.0

    payload: dict[str, float | int | list[float] | None] = {
        "bubble_count": int(bubble_count),
        "voxel_count": int(voxel_count),
        "surface_area": total_area,
        "mesh_volume": total_volume,
        "mean_dihedral_deg": float(weighted_dihedral_sum / max(weighted_dihedral_count, 1.0)),
        "mean_sphericity": float(np.mean([item["sphericity"] for item in bubble_metrics])) if bubble_metrics else 0.0,
        "bbox_min": None,
        "bbox_max": None,
    }
    if meshes:
        points = np.vstack([vertices for vertices, _ in meshes])
        payload["bbox_min"] = [float(value) for value in np.min(points, axis=0)]
        payload["bbox_max"] = [float(value) for value in np.max(points, axis=0)]
    return payload


def _distance_to_baseline(
    baseline_meshes: list[tuple[np.ndarray, np.ndarray]],
    other_meshes: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    if not baseline_meshes or not other_meshes:
        return {"mean_nearest_distance": 0.0, "p95_nearest_distance": 0.0, "max_nearest_distance": 0.0}

    baseline_vertices = np.vstack([vertices for vertices, _ in baseline_meshes])
    other_vertices = np.vstack([vertices for vertices, _ in other_meshes])
    distances, _ = cKDTree(baseline_vertices).query(other_vertices, k=1)
    return {
        "mean_nearest_distance": float(np.mean(distances)),
        "p95_nearest_distance": float(np.percentile(distances, 95.0)),
        "max_nearest_distance": float(np.max(distances)),
    }


def _frame_payload(frame: int, baseline, improved, smoothing: LaplacianSmoothingConfig) -> dict[str, object]:
    baseline_meshes = _mesh_bundle(baseline)
    improved_meshes = _mesh_bundle(improved)
    smoothed_meshes = _mesh_bundle(improved, smoothing=smoothing)

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

    return {
        "frame": int(frame),
        "baseline": baseline_metrics,
        "improved": improved_metrics,
        "smoothed_improved": smoothed_metrics,
        "surface_distance_vs_baseline": {
            "improved": _distance_to_baseline(baseline_meshes, improved_meshes),
            "smoothed_improved": _distance_to_baseline(baseline_meshes, smoothed_meshes),
        },
        "comparison": {
            "improved_delta_mean_dihedral_deg": float(improved_metrics["mean_dihedral_deg"] - baseline_metrics["mean_dihedral_deg"]),
            "smoothed_delta_mean_dihedral_deg": float(smoothed_metrics["mean_dihedral_deg"] - baseline_metrics["mean_dihedral_deg"]),
            "improved_delta_mean_sphericity": float(improved_metrics["mean_sphericity"] - baseline_metrics["mean_sphericity"]),
            "smoothed_delta_mean_sphericity": float(smoothed_metrics["mean_sphericity"] - baseline_metrics["mean_sphericity"]),
        },
    }


def _aggregate_frames(frame_results: list[dict[str, object]]) -> dict[str, object]:
    if not frame_results:
        return {
            "frame_count": 0,
            "mean_improved_delta_mean_dihedral_deg": 0.0,
            "mean_smoothed_delta_mean_dihedral_deg": 0.0,
            "mean_improved_delta_mean_sphericity": 0.0,
            "mean_smoothed_delta_mean_sphericity": 0.0,
        }

    improved_dihedral = [float(item["comparison"]["improved_delta_mean_dihedral_deg"]) for item in frame_results]
    smoothed_dihedral = [float(item["comparison"]["smoothed_delta_mean_dihedral_deg"]) for item in frame_results]
    improved_sphericity = [float(item["comparison"]["improved_delta_mean_sphericity"]) for item in frame_results]
    smoothed_sphericity = [float(item["comparison"]["smoothed_delta_mean_sphericity"]) for item in frame_results]

    return {
        "frame_count": len(frame_results),
        "mean_improved_delta_mean_dihedral_deg": float(np.mean(improved_dihedral)),
        "mean_smoothed_delta_mean_dihedral_deg": float(np.mean(smoothed_dihedral)),
        "mean_improved_delta_mean_sphericity": float(np.mean(improved_sphericity)),
        "mean_smoothed_delta_mean_sphericity": float(np.mean(smoothed_sphericity)),
        "improved_better_dihedral_frames": int(sum(value < 0.0 for value in improved_dihedral)),
        "smoothed_better_dihedral_frames": int(sum(value < 0.0 for value in smoothed_dihedral)),
        "improved_better_sphericity_frames": int(sum(value > 0.0 for value in improved_sphericity)),
        "smoothed_better_sphericity_frames": int(sum(value > 0.0 for value in smoothed_sphericity)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark baseline vs improved visual-hull surface quality across frames.")
    parser.add_argument("--frames", type=str, default=None, help="Comma-separated frame numbers. Defaults to the first available frames shared by all camera masks.")
    parser.add_argument("--max-frames", type=int, default=5, help="Maximum number of discovered frames to benchmark when --frames is omitted.")
    parser.add_argument("--num-cameras", type=int, default=3, help="Number of calibrated cameras to use.")
    args = parser.parse_args()

    data_dir = WORKSPACE_ROOT / "Islam_0207"
    available_frames = list_available_frames(data_dir, args.num_cameras)
    frames = _parse_frames(args.frames, available_frames, args.max_frames)
    if not frames:
        raise ValueError("No common frames were found across the requested camera mask files.")

    improved_config = ImprovedReconstructionConfig(hull=SoftVisualHullConfig())
    smoothing_config = LaplacianSmoothingConfig()
    run = create_test_run(PROJECT_ROOT, "surface-quality-benchmark")

    frame_results: list[dict[str, object]] = []
    for frame in frames:
        inputs = build_inputs(
            data_dir=data_dir,
            calibration_dir=data_dir,
            frame=frame,
            num_cameras=args.num_cameras,
            voxel_size=[0.3, 0.3, 0.3],
            limits=[10, 30, -5, 5, -5, 5],
        )
        baseline = run_full_reconstruction(inputs)
        improved = run_full_reconstruction_improved(inputs, config=improved_config)
        frame_results.append(_frame_payload(frame, baseline, improved, smoothing_config))

    payload = {
        "frames": frames,
        "available_frames": available_frames,
        "improved_config": improved_config.to_dict(),
        "laplacian_smoothing_config": smoothing_config.to_dict(),
        "summary": _aggregate_frames(frame_results),
        "results": frame_results,
    }

    run.write_json("surface_quality_benchmark.json", payload)
    run.write_json("report.json", payload)
    summary = payload["summary"]
    write_report_markdown(
        run,
        "Surface Quality Benchmark",
        [
            f"Frames benchmarked: {', '.join(str(frame) for frame in frames)}",
            f"Mean improved dihedral delta: {summary['mean_improved_delta_mean_dihedral_deg']:.4f} deg",
            f"Mean smoothed dihedral delta: {summary['mean_smoothed_delta_mean_dihedral_deg']:.4f} deg",
            f"Mean improved sphericity delta: {summary['mean_improved_delta_mean_sphericity']:.6f}",
            f"Mean smoothed sphericity delta: {summary['mean_smoothed_delta_mean_sphericity']:.6f}",
            f"Improved better-dihedral frames: {summary['improved_better_dihedral_frames']} / {summary['frame_count']}",
            f"Smoothed better-dihedral frames: {summary['smoothed_better_dihedral_frames']} / {summary['frame_count']}",
        ],
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()