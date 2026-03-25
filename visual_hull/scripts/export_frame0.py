from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_hull.reconstruction import build_inputs, run_full_reconstruction  # type: ignore[import-not-found]
from visual_hull.test_runs import create_test_run, write_report_markdown  # type: ignore[import-not-found]
from visual_hull.writers import write_reconstruction  # type: ignore[import-not-found]


def main() -> None:
    run = create_test_run(PROJECT_ROOT, "export-frame0")
    inputs = build_inputs(
        data_dir=WORKSPACE_ROOT / "Islam_0207",
        calibration_dir=WORKSPACE_ROOT / "Islam_0207",
        frame=0,
        num_cameras=3,
        voxel_size=[0.3, 0.3, 0.3],
        limits=[10, 30, -5, 5, -5, 5],
    )
    result = run_full_reconstruction(inputs)
    mat_path = write_reconstruction(result, run.path("Bubble_Frame_000000_python.mat"))
    h5_path = write_reconstruction(result, run.path("Bubble_Frame_000000_python.h5"))
    payload = {"mat": str(mat_path), "h5": str(h5_path)}
    run.write_json("report.json", payload)
    write_report_markdown(
        run,
        "Frame 0 Export",
        [
            f"MAT export: {mat_path.name}",
            f"HDF5 export: {h5_path.name}",
        ],
    )
    print(payload)


if __name__ == "__main__":
    main()
