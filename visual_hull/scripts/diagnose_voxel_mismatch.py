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
    run = create_test_run(PROJECT_ROOT, "diagnose-voxel-mismatch")
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

    py_voxels = result.voxels
    sort_idx_py = np.lexsort((py_voxels[:, 2], py_voxels[:, 1], py_voxels[:, 0]))
    sort_idx_ref = np.lexsort((ref_voxels[:, 2], ref_voxels[:, 1], ref_voxels[:, 0]))
    py_sorted = py_voxels[sort_idx_py]
    ref_sorted = ref_voxels[sort_idx_ref]

    payload = {
        "same_shape": py_voxels.shape == ref_voxels.shape,
        "sorted_exact_match": bool(np.array_equal(py_sorted, ref_sorted)),
        "sorted_allclose_1e-12": bool(np.allclose(py_sorted, ref_sorted, rtol=0.0, atol=1e-12)),
        "sorted_max_abs_diff": float(np.max(np.abs(py_sorted - ref_sorted))),
        "first_py_rows": py_voxels[:5].tolist(),
        "first_ref_rows": ref_voxels[:5].tolist(),
        "first_sorted_py_rows": py_sorted[:5].tolist(),
        "first_sorted_ref_rows": ref_sorted[:5].tolist(),
    }
    run.write_json("diagnose_voxel_mismatch.json", payload)
    run.write_json("report.json", payload)
    write_report_markdown(
        run,
        "Voxel Mismatch Diagnosis",
        [
            f"Same shape: {payload['same_shape']}",
            f"Sorted exact match: {payload['sorted_exact_match']}",
            f"Sorted allclose @ 1e-12: {payload['sorted_allclose_1e-12']}",
            f"Sorted max abs diff: {payload['sorted_max_abs_diff']}",
        ],
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
