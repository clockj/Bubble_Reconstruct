# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python port of a MATLAB pipeline for **3D bubble surface reconstruction from multi-view cameras** with refractive interfaces (e.g., water tanks with glass/acrylic walls). The core algorithm is **visual hull** reconstruction — intersecting viewing cones from calibrated cameras to carve out 3D voxel grids, then refining surface points, extracting per-bubble properties, and optionally fitting spherical harmonic surfaces.

The pipeline is defined in `Plan.md` (a **future** 7-step design spec with PyTorch optimization) and implemented in `visual_hull/` (the **current** MATLAB port — steps 2-3 and 7 only). The `improved/` subpackage adds soft voting, mesh smoothing, and spherical harmonics fitting but does not yet implement the full Plan.md optimization framework.

### Production pipeline (2026-07, current)

The live workflow runs on the 4-camera refractive dataset at
`X:\Shijie Zhong\Bubble Shear Project\Processed\20260710\20Hz_r_b_1_lpt\` (`X:` → `\\ruisrv5.wse.jhu.edu\data`) via the `bubble-visual-hull` conda env
(`C:\Users\zcloc\.conda\envs\bubble-visual-hull\python.exe`; invoke through PowerShell). Three stages:

- **A. Per-frame reconstruction** — `scripts/reconstruct_tiff_data.py`: TIFF masks → visual hull →
  surface refinement → per-bubble properties → optional SH surface fit. Writes
  `Results/<name>/Bubble_Frame_*.mat` (+ `*_sh.mat`, `reconstruction_summary.json`).
- **B. Tracking + temporal smoothing** — `scripts/temporal_smooth_sh.py`: nearest-neighbour +
  diameter-gated matcher, Gaussian smoothing along each track, and a stable **`sh_track_id`** per
  bubble (same physical bubble keeps its ID/color across frames).
- **C. Visualization** — `scripts/visualize_raw_html.py` (raw voxels) and
  `scripts/visualize_smoothed_html.py` (smoothed SH, colored by `sh_track_id`): self-contained
  interactive Plotly HTML with a frame slider, Plotly loaded via CDN (needs internet).

**Reference run (50 frames):**
`reconstruct_tiff_data.py --frames 0 49 --voxel-size 0.5 0.5 0.5 --sh-degree 4 --sh-inscribed --sh-silhouette --max-aspect-ratio 5`.

**SH surface fit** (`improved/spherical_harmonics/surface.py`) has three safeguards, all opt-in via
flags on `reconstruct_tiff_data.py`:
- `--sh-min-points-per-coeff` (default 3) — **flower fix**: caps each bubble's degree so an
  undersampled fit can't oscillate into negative-radius "petals" (root cause: pts/coeff < 1).
- `--sh-inscribed` — fits the hull's *outer envelope* with a one-sided overshoot penalty (IRLS),
  so the surface hugs the inside of the hull boundary instead of bulging past it.
- `--sh-silhouette` — coordinate-descends the SH coeffs to maximize IoU vs each bubble's *own
  re-projected* hull silhouette in every camera. Raised mean IoU 0.60 → 0.86 (frame 16).

**Filtering** on `reconstruct_tiff_data.py`: `--size-range MIN MAX` (equivalent diameter mm,
`D_eq = 2·(3V/4π)^(1/3)`) and `--max-aspect-ratio` (drop elongated **visual-hull ghost** slivers;
real bubbles ~1.5–3, phantoms >6).

**Known bias:** the 4-view visual hull systematically **over-estimates ~+6% diameter / +19% volume**
(synthetic-sphere study, `test/20260722-142124-synthetic-sphere-bias/`), independent of
size/location/resolution. Temporal smoothing improves **precision** (jitter ~×3) but does **not**
remove this **accuracy** bias — that needs calibration or the silhouette-consistent fit. The two are
complementary. The latest report is `report/reconstruction_report_20260723.md`.

## Package Structure

```
visual_hull/
├── pyproject.toml                  # Build config (setuptools), deps
├── src/visual_hull/                # Main package
│   ├── camera.py                   # OpenLPTCameraSet wrapping OpenLPT Camera objects
│   ├── voxel_grid.py               # VoxelGrid dataclass + grid init helpers
│   ├── hull.py                     # Binary hard-voting visual hull
│   ├── io.py                       # Load masks from .mat files (Cam1.mat, ...)
│   ├── models.py                   # ReconstructionInputs / FullReconstructionResult / FrameExportResult
│   ├── reconstruction.py           # High-level pipeline entry points + parallel frame batching
│   ├── refinement.py               # Surface component extraction + fine-resolution point refinement
│   ├── properties.py               # Bubble property extraction (volume, centroid, axes, aspect ratio)
│   ├── surface_utils.py            # Marching cubes mesh construction
│   ├── silhouette_metrics.py       # Silhouette overlap/IoU metrics
│   ├── writers.py                  # Export to .mat (scipy) or .h5 (h5py)
│   ├── visualization.py            # Matplotlib 3D interactive plots (surface/scatter/mesh compare)
│   ├── test_runs.py                # Test scaffolding: timestamped folders, JSON/text reports
│   └── improved/                   # Accuracy improvements over the baseline MATLAB port
│       ├── hull.py                 # Soft visual hull (signed distance transform, bilinear sampling)
│       ├── reconstruction.py       # Improved pipeline config + orchestration
│       ├── refinement.py           # Improved refinement
│       ├── surface.py              # Laplacian/Taubin mesh smoothing + surface mesh from voxels
│       └── spherical_harmonics/    # SH surface representation + fitting
└── scripts/                        # Runnable entry points
    ├── reconstruct_tiff_data.py    # PRODUCTION Stage A: TIFF masks -> voxels + SH (main entry)
    ├── temporal_smooth_sh.py       # PRODUCTION Stage B: tracking + smoothing + sh_track_id
    ├── visualize_raw_html.py       # PRODUCTION Stage C: raw voxel interactive HTML viewer
    ├── visualize_smoothed_html.py  # PRODUCTION Stage C: smoothed, track-colored HTML viewer
    ├── smoke_test.py               # Quick end-to-end validation
    ├── export_frame0.py            # Single-frame reconstruction export
    ├── reconstruct_frames_parallel.py  # Multi-frame batch reconstruction
    ├── visualize_frame.py          # 3D viz of a single reconstruction
    ├── visualize_compare_frame.py  # Side-by-side comparison viz
    ├── compare_reference_frame0.py # Validate against MATLAB reference output
    ├── compare_improved_frame0.py  # Validate improved pipeline vs baseline
    ├── strict_compare_frame0.py    # Strict voxel-level comparison
    ├── diagnose_voxel_mismatch.py  # Debug voxel differences
    ├── compare_smoothed_surface_frame0.py
    ├── compare_surface_quality_frame0.py
    ├── compare_spherical_harmonic_frame0.py
    ├── visualize_spherical_harmonic_compare_frame.py
    ├── sweep_spherical_harmonic_frame0.py
    ├── benchmark_surface_quality.py
    └── benchmark_spherical_harmonic_degree_scaling.py
```

The `build/` directory contains a copy of the installed package — always edit under `src/`, never `build/`.

## Environment & Build

A dedicated **conda environment** is required (spec in `visual_hull/environment.yml`). The key dependency is **OpenLPT** (`openlpt>=2.2.0`), which provides the refractive camera projection model through `pyopenlpt`. This is a native C++/Python binding — it is pip-installed after conda packages.

```bash
# Create the conda environment (one-time)
conda env create -f visual_hull/environment.yml
conda activate bubble-visual-hull

# Install the package in development mode
cd visual_hull
pip install -e .
```

Python >= 3.11 required. Core dependencies: `numpy`, `scipy`, `scikit-image`, `joblib`, `matplotlib`, `pyside6`, `h5py`.

## Running the Pipeline

### Smoke test (quick validation)
```bash
cd visual_hull
python scripts/smoke_test.py
```
Uses data from `Islam_0207/` (3 cameras, frame 0, small voxel grid) and writes results to `visual_hull/test/<timestamp>-smoke-test/`.

### Single-frame reconstruction
```bash
cd visual_hull
python scripts/export_frame0.py
```

### Multi-frame batch reconstruction
```bash
cd visual_hull
python scripts/reconstruct_frames_parallel.py
```
Uses `joblib` for process-based parallelism. Relies on `build_inputs()` with a `data_dir` containing `Cam1.mat`, `Cam2.mat`, ... `CamN.mat` and a `calibration_dir` with `C*P.txt` camera files.

### Programmatic usage
```python
from visual_hull.reconstruction import build_inputs, run_full_reconstruction
from visual_hull.writers import write_reconstruction

inputs = build_inputs(
    data_dir="path/to/data",
    calibration_dir="path/to/cameras",
    frame=0,
    num_cameras=3,
    voxel_size=[0.3, 0.3, 0.3],
    limits=[10, 20, -2, 2, -2, 2],
)
result = run_full_reconstruction(inputs)
write_reconstruction(result, "output.mat")
```

For the improved (soft hull + smoothing + spherical harmonics) pipeline, use `visual_hull.improved`:
```python
from visual_hull.improved import (
    ImprovedReconstructionConfig,
    run_full_reconstruction_improved,
)
```

### 3D visualization
```python
from visual_hull.visualization import show_reconstruction_interactive
show_reconstruction_interactive(result, mode="surface")  # or "scatter"
```

## Key Design Decisions

- **OpenLPT is the authoritative projection model**. All camera/projection logic goes through `OpenLPTCameraSet.project_points()`, which uses `lpt.Camera` objects for refractive projection. Do not reimplement projection math.
- **Hard-vote vs soft-vote visual hull**: `visual_hull.hull` uses binary all-camera consensus (`min_votes == camera_count`). `visual_hull.improved.hull` uses signed distance transforms with bilinear sampling for sub-pixel accuracy.
- **Two-pass refinement**: Coarse visual hull → surface component erosion → fine-resolution local re-voting around each surface component (see `refinement.py`).
- **Bubble separation**: Uses 3D connected components on the eroded surface shell, not on the filled volume. Each surface component is refined independently.
- **MATLAB compatibility**: The output format (`FullReconstructionResult.to_matlab_payload()`) matches the original MATLAB struct layout. `scripts/compare_reference_frame0.py` validates against reference MATLAB outputs.

## Test Artifacts

Per `.github/copilot-instructions.md`: All test results go under `visual_hull/test/<YYYYMMDD-HHMMSS>-<name>/`. Each run gets a unique subfolder. Use `test_runs.create_test_run()` to scaffold this automatically.

## Validation & Comparison

Reference data lives in `Islam_0207/` (at the workspace root, above the project dir). Key validation scripts:

| Script | Purpose |
|--------|---------|
| `scripts/compare_reference_frame0.py` | Python vs MATLAB reference output comparison |
| `scripts/strict_compare_frame0.py` | Strict voxel-exact comparison with tolerance checks |
| `scripts/compare_improved_frame0.py` | Improved pipeline vs baseline comparison |
| `scripts/diagnose_voxel_mismatch.py` | Debug coordinate-level differences |

Current status: frame-0 voxel count, bubble count, property shape, and bubble properties match the MATLAB reference; voxel coordinates match within floating-point tolerance.

## Project Skills

This repo includes custom skills in `.claude/skills/`:

| Skill | Scope | Purpose |
|-------|-------|---------|
| `test-workflow` | `visual_hull/` | Test output conventions — timestamped subfolders under `test/`, no overwrites, `report.md`/`report.json` per run |
| `karpathy-guidelines` | global | Coding discipline — think before coding, simplicity first, surgical changes, goal-driven execution |
| `debug-workflow` | global | Debug workflow — plan first, get approval, save results alongside test code, explain changes before implementing |
| `rockfish-workflow` | on-demand | HPC cluster workflow — only apply when user mentions "rockfish" or remote server |

Invoke via `/skill-name` (e.g., `/test-workflow`). The `test-workflow` skill auto-activates when working on files under `visual_hull/`.

## Remote / HPC (Rockfish)

When running on the Rockfish cluster:

```bash
ml anaconda            # Load Anaconda module
conda activate OED     # Activate the OED environment
```

- Never use `sudo` or `systemctl`; never run heavy computation locally
- Use `ml <name>` to load software, `ml spider <name>` to search
- C++ build: `conda deactivate && ./command.rockfish`
- Submit heavy jobs to the scheduler — check existing job scripts for templates

## Development Conventions

From `karpathy-guidelines` and `debug-workflow`:

- **Plan first**: For non-trivial changes, outline the approach and get approval before implementing. Save the plan to the relevant test/output folder.
- **Surgical edits**: Touch only what's needed. Don't refactor adjacent code, fix unrelated formatting, or delete pre-existing dead code unless asked.
- **Simplicity**: No abstractions for single-use code, no unrequested features, no error handling for impossible scenarios.
- **Verifiable goals**: Define success criteria upfront (e.g., "write a test that reproduces the bug, then fix it"). Loop until verified.
- **Keep results with code**: Debug artifacts, test outputs, and reports live alongside the test scripts that generated them. Use `test_runs.create_test_run()` under `visual_hull/`.

## Claude Code Working Rules

You may work autonomously inside this repository.

You may read, search, edit existing files, and create new files or folders inside this repo when relevant to the task.

You may run safe local Python/C++/MATLAB build, lint, format, and test commands.

Ask before:
- deleting files or folders
- overwriting important existing files
- running any git command
- editting or writing outside this repo
- installing packages or changing environments
- modifying lockfiles
- running sudo/chmod/chown
- using network commands
- modifying secrets/env/license/key files
- creating large generated outputs
- starting long-running background processes

Use a loop: inspect, plan briefly, edit/create files as needed, run relevant checks, fix failures, repeat, then summarize.

