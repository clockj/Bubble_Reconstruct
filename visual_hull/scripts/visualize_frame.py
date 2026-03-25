from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_hull import build_inputs, run_full_reconstruction, show_reconstruction_interactive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open an interactive 3D viewer for a reconstructed bubble frame.")
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
    result = run_full_reconstruction(inputs)
    show_reconstruction_interactive(
        result,
        mode=args.mode,
        title=f"Reconstructed Bubble Shape - Frame {args.frame:06d}",
    )


if __name__ == "__main__":
    main()