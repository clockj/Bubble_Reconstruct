from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_hull import build_inputs, run_full_reconstruction, show_mesh_comparison_interactive  # type: ignore[import-not-found]
from visual_hull.improved import (  # type: ignore[import-not-found]
    ImprovedReconstructionConfig,
    SoftVisualHullConfig,
    SphericalHarmonicFitConfig,
    fit_spherical_harmonic_surface_from_voxels,
    run_full_reconstruction_improved,
    surface_mesh_from_voxels,
)
from visual_hull.test_runs import create_test_run, write_report_markdown  # type: ignore[import-not-found]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open an interactive 3D viewer for baseline, improved, and spherical-harmonic surfaces.")
    parser.add_argument("--frame", type=int, default=0, help="Frame number to reconstruct and visualize.")
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE_ROOT / "Islam_0207", help="Directory containing the camera mask MAT files.")
    parser.add_argument("--calibration-dir", type=Path, default=WORKSPACE_ROOT / "Islam_0207", help="Directory containing the OpenLPT camera files.")
    parser.add_argument("--num-cameras", type=int, default=3, help="Number of cameras to use in the reconstruction.")
    parser.add_argument("--voxel-size", type=float, nargs=3, default=[0.3, 0.3, 0.3], metavar=("DX", "DY", "DZ"), help="Coarse reconstruction voxel size in millimeters.")
    parser.add_argument("--limits", type=float, nargs=6, default=[10.0, 30.0, -5.0, 5.0, -5.0, 5.0], metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"), help="Reconstruction volume limits in millimeters.")
    parser.add_argument("--sh-degree", type=int, default=4, help="Maximum spherical harmonic degree.")
    parser.add_argument("--sh-regularization", type=float, default=1e-3, help="Tikhonov regularization strength for the SH fit.")
    parser.add_argument("--save", action="store_true", help="Save a comparison snapshot and report under visual_hull/test/.")
    parser.add_argument("--no-show", action="store_true", help="Do not open the interactive window. Useful when you only want saved outputs.")
    return parser.parse_args()


def _bubble_ranges(bubbles) -> list[tuple[int, int]]:
    bubble_array = np.asarray(bubbles)
    if bubble_array.size == 0:
        return []
    return [(int(start) - 1, int(stop)) for start, stop in bubble_array.T]


import numpy as np


def _result_meshes(result) -> list[tuple[np.ndarray, np.ndarray]]:
    meshes: list[tuple[np.ndarray, np.ndarray]] = []
    for start, stop in _bubble_ranges(result.bubbles):
        mesh = surface_mesh_from_voxels(result.voxels[start:stop], result.voxel_size_2)
        if mesh is not None:
            meshes.append(mesh)
    return meshes


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


def main() -> None:
    args = parse_args()
    inputs = build_inputs(
        data_dir=args.data_dir.resolve(),
        calibration_dir=args.calibration_dir.resolve(),
        frame=args.frame,
        num_cameras=args.num_cameras,
        voxel_size=args.voxel_size,
        limits=args.limits,
    )
    baseline = run_full_reconstruction(inputs)
    improved = run_full_reconstruction_improved(inputs, config=ImprovedReconstructionConfig(hull=SoftVisualHullConfig()))
    sh_config = SphericalHarmonicFitConfig(max_degree=args.sh_degree, regularization=args.sh_regularization)
    baseline_meshes = _result_meshes(baseline)
    improved_meshes = _result_meshes(improved)
    sh_meshes, fit_rmses = _sh_meshes(improved, sh_config)

    save_path: Path | None = None
    run = None
    if args.save:
        run = create_test_run(PROJECT_ROOT, f"visualize-sh-compare-frame{args.frame:06d}")
        save_path = run.path(f"compare_sh_frame_{args.frame:06d}.png")

    show_mesh_comparison_interactive(
        [
            (f"Baseline - Frame {args.frame:06d}", baseline_meshes),
            (f"Improved Raw - Frame {args.frame:06d}", improved_meshes),
            (f"Improved + SH - Frame {args.frame:06d}", sh_meshes),
        ],
        title=f"Visual Hull SH Comparison - Frame {args.frame:06d}",
        save_path=save_path,
        show=not args.no_show,
    )

    if run is not None and save_path is not None:
        payload = {
            "frame": int(args.frame),
            "sh_config": sh_config.to_dict(),
            "fit_rmse_mm": [float(value) for value in fit_rmses],
            "artifacts": {"comparison_png": str(save_path)},
            "baseline_mesh_count": len(baseline_meshes),
            "improved_mesh_count": len(improved_meshes),
            "spherical_harmonic_mesh_count": len(sh_meshes),
        }
        run.write_json("report.json", payload)
        run.write_json("visualize_spherical_harmonic_compare_frame.json", payload)
        write_report_markdown(
            run,
            f"Visual SH Comparison - Frame {args.frame:06d}",
            [
                f"SH degree: {args.sh_degree}",
                f"SH regularization: {args.sh_regularization}",
                f"Mean SH fit RMSE: {float(np.mean(fit_rmses)) if fit_rmses else 0.0:.4f} mm",
                f"Saved image: {save_path.name}",
            ],
        )
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()