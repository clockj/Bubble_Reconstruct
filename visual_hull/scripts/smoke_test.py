from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_hull.reconstruction import build_inputs, run_full_reconstruction
from visual_hull.test_runs import create_test_run, write_report_markdown


def main() -> None:
    run = create_test_run(PROJECT_ROOT, "smoke-test")
    inputs = build_inputs(
        data_dir=WORKSPACE_ROOT / "Islam_0207",
        calibration_dir=WORKSPACE_ROOT / "Islam_0207",
        frame=0,
        num_cameras=3,
        voxel_size=[0.3, 0.3, 0.3],
        limits=[10, 20, -2, 2, -2, 2],
    )
    result = run_full_reconstruction(inputs)
    payload = {
        "voxel_count": int(result.voxels.shape[0]),
        "bubble_count": int(result.bubbles.shape[1]) if result.bubbles.ndim == 2 else 0,
        "properties_shape": tuple(int(value) for value in result.properties.shape),
        "grid_shape": tuple(int(value) for value in result.coarse_result.voxel_volume.shape),
    }
    run.write_json("report.json", payload)
    write_report_markdown(
        run,
        "Smoke Test",
        [
            f"Voxel count: {payload['voxel_count']}",
            f"Bubble count: {payload['bubble_count']}",
            f"Properties shape: {payload['properties_shape']}",
            f"Grid shape: {payload['grid_shape']}",
        ],
    )
    print(payload)


if __name__ == "__main__":
    main()
