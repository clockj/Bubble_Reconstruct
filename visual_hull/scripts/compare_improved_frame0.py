from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_hull import build_inputs, run_full_reconstruction  # type: ignore[import-not-found]
from visual_hull.improved import (  # type: ignore[import-not-found]
    ImprovedReconstructionConfig,
    SoftVisualHullConfig,
    run_full_reconstruction_improved,
)
from visual_hull.test_runs import create_test_run, write_report_markdown  # type: ignore[import-not-found]


def _bbox(points: np.ndarray) -> dict[str, list[float] | None]:
    if points.size == 0:
        return {"min": None, "max": None}
    return {
        "min": [float(value) for value in np.min(points, axis=0)],
        "max": [float(value) for value in np.max(points, axis=0)],
    }


def main() -> None:
    run = create_test_run(PROJECT_ROOT, "compare-improved-frame0")
    inputs = build_inputs(
        data_dir=WORKSPACE_ROOT / "Islam_0207",
        calibration_dir=WORKSPACE_ROOT / "Islam_0207",
        frame=0,
        num_cameras=3,
        voxel_size=[0.3, 0.3, 0.3],
        limits=[10, 30, -5, 5, -5, 5],
    )

    baseline = run_full_reconstruction(inputs)
    improved = run_full_reconstruction_improved(
        inputs,
        config=ImprovedReconstructionConfig(hull=SoftVisualHullConfig()),
    )

    payload = {
        "baseline": {
            "voxel_count": int(baseline.voxels.shape[0]),
            "bubble_count": int(baseline.bubbles.shape[1]) if baseline.bubbles.ndim == 2 else 0,
            "bbox": _bbox(baseline.voxels),
        },
        "improved": {
            "voxel_count": int(improved.voxels.shape[0]),
            "bubble_count": int(improved.bubbles.shape[1]) if improved.bubbles.ndim == 2 else 0,
            "bbox": _bbox(improved.voxels),
        },
        "delta": {
            "voxel_count": int(improved.voxels.shape[0] - baseline.voxels.shape[0]),
            "bubble_count": int(
                (improved.bubbles.shape[1] if improved.bubbles.ndim == 2 else 0)
                - (baseline.bubbles.shape[1] if baseline.bubbles.ndim == 2 else 0)
            ),
        },
    }

    run.write_json("compare_improved_frame0.json", payload)
    run.write_json("report.json", payload)
    write_report_markdown(
        run,
        "Improved vs Baseline Frame 0",
        [
            f"Baseline voxel count: {payload['baseline']['voxel_count']}",
            f"Improved voxel count: {payload['improved']['voxel_count']}",
            f"Voxel delta: {payload['delta']['voxel_count']}",
            f"Baseline bubble count: {payload['baseline']['bubble_count']}",
            f"Improved bubble count: {payload['improved']['bubble_count']}",
        ],
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()