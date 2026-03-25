from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
REFERENCE_FILE = WORKSPACE_ROOT / "Islam_0207" / "Reconstruction" / "Bubble_Frame_000000.mat"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_hull.reconstruction import build_inputs, run_full_reconstruction  # type: ignore[import-not-found]
from visual_hull.test_runs import create_test_run, write_report_markdown  # type: ignore[import-not-found]


def main() -> None:
    run = create_test_run(PROJECT_ROOT, "strict-compare-frame0")
    inputs = build_inputs(
        data_dir=WORKSPACE_ROOT / "Islam_0207",
        calibration_dir=WORKSPACE_ROOT / "Islam_0207",
        frame=0,
        num_cameras=3,
        voxel_size=[0.3, 0.3, 0.3],
        limits=[10, 30, -5, 5, -5, 5],
    )
    result = run_full_reconstruction(inputs)

    with h5py.File(REFERENCE_FILE, "r") as reference:
        ref_voxels = np.array(reference["voxels"], dtype=np.float64).T
        ref_bubbles = np.array(reference["bubbles"], dtype=np.int64).T
        ref_properties = np.array(reference["properties"], dtype=np.float64).T

    payload = {
        "voxels_exact_match": bool(np.array_equal(result.voxels, ref_voxels)),
        "bubbles_exact_match": bool(np.array_equal(result.bubbles, ref_bubbles)),
        "properties_allclose_atol_1e-9": bool(np.allclose(result.properties, ref_properties, rtol=0.0, atol=1e-9)),
        "properties_max_abs_diff": float(np.max(np.abs(result.properties - ref_properties))) if result.properties.size else 0.0,
    }
    run.write_json("strict_compare_frame0.json", payload)
    run.write_json("report.json", payload)
    write_report_markdown(
        run,
        "Strict Frame 0 Comparison",
        [
            f"Voxels exact match: {payload['voxels_exact_match']}",
            f"Bubbles exact match: {payload['bubbles_exact_match']}",
            f"Properties allclose @ 1e-9: {payload['properties_allclose_atol_1e-9']}",
            f"Properties max abs diff: {payload['properties_max_abs_diff']}",
        ],
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
