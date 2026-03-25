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

from visual_hull import build_inputs, run_reconstruction_frames_parallel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frame-level parallel bubble reconstruction and write one file per frame.")
    parser.add_argument("--frames", type=int, nargs="+", required=True, help="Frame numbers to reconstruct.")
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
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory where one output file per frame will be written.",
    )
    parser.add_argument(
        "--format",
        choices=["mat", "h5", "hdf5"],
        default="mat",
        help="Per-frame export format.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker processes. Defaults to min(number of frames, CPU count).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = build_inputs(
        data_dir=args.data_dir.resolve(),
        calibration_dir=args.calibration_dir.resolve(),
        frame=args.frames[0],
        num_cameras=args.num_cameras,
        voxel_size=args.voxel_size,
        limits=args.limits,
    )
    results = run_reconstruction_frames_parallel(
        inputs,
        frames=args.frames,
        output_dir=args.output_dir.resolve(),
        export_format=args.format,
        max_workers=args.max_workers,
    )
    payload = [
        {
            "frame": item.frame,
            "output_path": str(item.output_path),
            "voxel_count": item.voxel_count,
            "bubble_count": item.bubble_count,
            "completed": item.completed,
        }
        for item in results
    ]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()