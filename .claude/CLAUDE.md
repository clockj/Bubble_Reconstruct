# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python port of a MATLAB pipeline for **3D bubble surface reconstruction from multi-view cameras** with refractive interfaces (e.g., water tanks with glass/acrylic walls). The core algorithm is **visual hull** reconstruction — intersecting viewing cones from calibrated cameras to carve out 3D voxel grids, then refining surface points, extracting per-bubble properties, and optionally fitting spherical harmonic surfaces.

The pipeline is defined in `Plan.md` (a **future** 7-step design spec with PyTorch optimization) and implemented in `visual_hull/` (the **current** MATLAB port — steps 2-3 and 7 only). The `improved/` subpackage adds soft voting, mesh smoothing, and spherical harmonics fitting but does not yet implement the full Plan.md optimization framework.

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

