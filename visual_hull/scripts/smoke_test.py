from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_hull.reconstruction import build_inputs, run_coarse_reconstruction


def main() -> None:
    inputs = build_inputs(
        data_dir=WORKSPACE_ROOT / "Islam_0207",
        calibration_dir=WORKSPACE_ROOT / "Islam_0207",
        frame=0,
        num_cameras=3,
        voxel_size=[0.3, 0.3, 0.3],
        limits=[10, 20, -2, 2, -2, 2],
    )
    result = run_coarse_reconstruction(inputs)
    print(
        {
            "kept_voxels": int(result.kept_voxels.shape[0]),
            "grid_shape": tuple(int(value) for value in result.voxel_volume.shape),
        }
    )


if __name__ == "__main__":
    main()
