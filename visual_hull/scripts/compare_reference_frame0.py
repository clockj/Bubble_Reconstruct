from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
REFERENCE_FILE = WORKSPACE_ROOT / "Islam_0207" / "Reconstruction" / "Bubble_Frame_000000.mat"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_hull.reconstruction import build_inputs, run_full_reconstruction  # type: ignore[import-not-found]
from visual_hull.test_runs import create_test_run, write_report_markdown  # type: ignore[import-not-found]


def _float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def _bbox(points: np.ndarray) -> dict[str, list[float] | None]:
    if points.size == 0:
        return {"min": None, "max": None}
    return {
        "min": _float_list(np.min(points, axis=0)),
        "max": _float_list(np.max(points, axis=0)),
    }


def _load_reference_payload(reference_file: Path) -> dict[str, np.ndarray | bool | None]:
    try:
        reference = loadmat(reference_file)
        completed_raw = reference.get("completed")
        completed = bool(np.asarray(completed_raw).squeeze()) if completed_raw is not None else None
        return {
            "voxels": np.asarray(reference.get("voxels", np.empty((0, 3))), dtype=np.float64),
            "bubbles": np.asarray(reference.get("bubbles", np.empty((2, 0))), dtype=np.float64),
            "properties": np.asarray(reference.get("properties", np.empty((0, 0))), dtype=np.float64),
            "completed": completed,
            "loader": "scipy.io.loadmat",
        }
    except NotImplementedError:
        with h5py.File(reference_file, "r") as reference:
            voxels = np.array(reference["voxels"], dtype=np.float64).T
            bubbles = np.array(reference["bubbles"], dtype=np.float64).T
            properties = np.array(reference["properties"], dtype=np.float64).T
            completed = bool(np.array(reference["completed"]).squeeze()) if "completed" in reference else None
        return {
            "voxels": voxels,
            "bubbles": bubbles,
            "properties": properties,
            "completed": completed,
            "loader": "h5py",
        }


def main() -> None:
    run = create_test_run(PROJECT_ROOT, "compare-frame0")
    inputs = build_inputs(
        data_dir=WORKSPACE_ROOT / "Islam_0207",
        calibration_dir=WORKSPACE_ROOT / "Islam_0207",
        frame=0,
        num_cameras=3,
        voxel_size=[0.3, 0.3, 0.3],
        limits=[10, 30, -5, 5, -5, 5],
    )
    python_result = run_full_reconstruction(inputs)

    reference = _load_reference_payload(REFERENCE_FILE)
    ref_voxels = np.asarray(reference["voxels"], dtype=np.float64)
    ref_bubbles = np.asarray(reference["bubbles"], dtype=np.float64)
    ref_properties = np.asarray(reference["properties"], dtype=np.float64)
    ref_completed = reference["completed"]

    comparison = {
        "reference_file": str(REFERENCE_FILE),
        "python": {
            "kind": "full_reconstruction",
            "voxel_count": int(python_result.voxels.shape[0]),
            "bubble_count": int(python_result.bubbles.shape[1]) if python_result.bubbles.ndim == 2 else 0,
            "properties_shape": [int(value) for value in python_result.properties.shape],
            "bbox": _bbox(python_result.voxels),
            "grid_shape": [int(value) for value in python_result.coarse_result.voxel_volume.shape],
        },
        "matlab_reference": {
            "loader": reference["loader"],
            "completed": ref_completed,
            "voxel_count": int(ref_voxels.shape[0]),
            "bubble_count": int(ref_bubbles.shape[1]) if ref_bubbles.ndim == 2 else 0,
            "properties_shape": [int(value) for value in ref_properties.shape],
            "bbox": _bbox(ref_voxels),
        },
        "difference": {
            "voxel_count_delta": int(python_result.voxels.shape[0] - ref_voxels.shape[0]),
            "matches_reference_voxel_count": bool(python_result.voxels.shape[0] == ref_voxels.shape[0]),
            "bubble_count_delta": int((python_result.bubbles.shape[1] if python_result.bubbles.ndim == 2 else 0) - (ref_bubbles.shape[1] if ref_bubbles.ndim == 2 else 0)),
            "expected_to_match_exactly": False,
            "reason": "The Python port now includes the MATLAB refinement stages, but numeric differences can still remain because the projection engine is OpenLPT-backed and the marching-cubes/property extraction path is a NumPy/SciPy translation of the MATLAB code.",
        },
    }

    run.write_json("comparison_frame0.json", comparison)
    run.write_json("report.json", comparison)
    write_report_markdown(
        run,
        "Frame 0 Comparison",
        [
            f"Python voxel count: {comparison['python']['voxel_count']}",
            f"MATLAB voxel count: {comparison['matlab_reference']['voxel_count']}",
            f"Bubble count delta: {comparison['difference']['bubble_count_delta']}",
            f"Reference loader: {comparison['matlab_reference']['loader']}",
        ],
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
