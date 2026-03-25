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

from visual_hull import (  # type: ignore[import-not-found]
    build_inputs,
    run_full_reconstruction,
    run_full_reconstruction_improved,
    show_reconstruction_comparison_interactive,
)
from visual_hull.improved import ImprovedReconstructionConfig, SoftVisualHullConfig  # type: ignore[import-not-found]
from visual_hull.test_runs import create_test_run, write_report_markdown  # type: ignore[import-not-found]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open an interactive 3D viewer for baseline vs improved reconstructed bubble surfaces.")
    parser.add_argument("--frame", type=int, default=0, help="Frame number to reconstruct and visualize.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=WORKSPACE_ROOT / "Islam_0207",
        help="Directory containing the camera mask MAT files.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=WORKSPACE_ROOT / "Islam_0207",
        help="Directory containing the OpenLPT camera files.",
    )
    parser.add_argument("--num-cameras", type=int, default=3, help="Number of cameras to use in the reconstruction.")
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        default=[0.3, 0.3, 0.3],
        metavar=("DX", "DY", "DZ"),
        help="Coarse reconstruction voxel size in millimeters.",
    )
    parser.add_argument(
        "--limits",
        type=float,
        nargs=6,
        default=[10.0, 30.0, -5.0, 5.0, -5.0, 5.0],
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Reconstruction volume limits in millimeters.",
    )
    parser.add_argument(
        "--mode",
        choices=["surface", "scatter"],
        default="surface",
        help="Visualization mode: triangulated surface or raw voxel scatter.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save a comparison snapshot and report under visual_hull/test/.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive window. Useful when you only want saved outputs.",
    )
    return parser.parse_args()


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
    improved = run_full_reconstruction_improved(
        inputs,
        config=ImprovedReconstructionConfig(hull=SoftVisualHullConfig()),
    )

    save_path: Path | None = None
    run = None
    if args.save:
        run = create_test_run(PROJECT_ROOT, f"visualize-compare-frame{args.frame:06d}")
        save_path = run.path(f"compare_frame_{args.frame:06d}_{args.mode}.png")

    show_reconstruction_comparison_interactive(
        [
            (f"Baseline - Frame {args.frame:06d}", baseline),
            (f"Improved - Frame {args.frame:06d}", improved),
        ],
        mode=args.mode,
        title=f"Visual Hull Comparison - Frame {args.frame:06d}",
        save_path=save_path,
        show=not args.no_show,
    )

    if run is not None and save_path is not None:
        payload = {
            "frame": int(args.frame),
            "mode": args.mode,
            "data_dir": str(args.data_dir.resolve()),
            "calibration_dir": str(args.calibration_dir.resolve()),
            "num_cameras": int(args.num_cameras),
            "voxel_size": [float(value) for value in args.voxel_size],
            "limits": [float(value) for value in args.limits],
            "artifacts": {
                "comparison_png": str(save_path),
            },
            "baseline": {
                "voxel_count": int(baseline.voxels.shape[0]),
                "bubble_count": int(baseline.bubbles.shape[1]) if baseline.bubbles.ndim == 2 else 0,
            },
            "improved": {
                "voxel_count": int(improved.voxels.shape[0]),
                "bubble_count": int(improved.bubbles.shape[1]) if improved.bubbles.ndim == 2 else 0,
            },
        }
        run.write_json("report.json", payload)
        run.write_json("visualize_compare_frame.json", payload)
        write_report_markdown(
            run,
            f"Visual Comparison - Frame {args.frame:06d}",
            [
                f"Mode: {args.mode}",
                f"Baseline voxel count: {payload['baseline']['voxel_count']}",
                f"Improved voxel count: {payload['improved']['voxel_count']}",
                f"Saved image: {save_path.name}",
            ],
        )
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()