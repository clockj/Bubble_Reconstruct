# Visual Hull Python Port

This folder contains the Python implementation of the MATLAB bubble-reconstruction workflow used in this repository.

## Environment

A dedicated conda environment was created for this port. The reproducible spec is stored in `environment.yml`.

To recreate it manually:

```powershell
conda env create -f environment.yml
conda activate bubble-visual-hull
```

If you need to install the package into the active environment:

```powershell
python -m pip install -e .
```

## Current Status

The current implementation includes:

- an OpenLPT-backed camera adapter built on `pyopenlpt`
- MATLAB-style voxel-grid generation and sparse-to-dense conversion
- coarse visual-hull voting against binary silhouettes stored in MATLAB `.mat` files
- surface extraction, local fine-grid refinement, and 3D bubble property extraction
- a full reconstruction entry point for loading masks and generating MATLAB-comparable outputs
- frame-0 validation scripts against `Islam_0207/Reconstruction/Bubble_Frame_000000.mat`

## Source Mapping

- `camera.py` maps OpenLPT camera files to Python projection helpers
- `io.py` loads MATLAB camera-mask files and frame keys
- `models.py` defines the shared reconstruction input/output dataclasses
- `voxel_grid.py` ports `InitializeVoxels.m` and `ConvertVoxelListTo3D.m`
- `hull.py` ports the coarse voting logic from `CreateVisualHull_2.m` and `VisualHull.m`
- `refinement.py` ports the surface-component separation and local refinement flow from `Reconstruction.m` and `mesh_expand.m`
- `properties.py` ports `GetBubbleProps.m`
- `reconstruction.py` orchestrates the full translated pipeline
- `writers.py` exports reconstruction results to MATLAB `.mat` or HDF5 `.h5/.hdf5`

## Export

Use `write_reconstruction(...)` to export a `FullReconstructionResult`.

Supported formats:

- `.mat`: best when you need drop-in interoperability with the existing MATLAB workflow
- `.h5` or `.hdf5`: best for Python-native analysis, larger datasets, compression, and schema evolution

Recommendation:

- Use `.mat` if the immediate consumer is MATLAB or if you want exact compatibility with the existing `Bubble_Frame_*.mat` workflow.
- Use `.h5` for new Python-first pipelines, multi-frame datasets, or larger exports where compression and explicit metadata matter.
- If you need both ecosystems, write both formats from the same result object; the package now supports that directly.

## Validation

- `scripts/compare_reference_frame0.py` compares the Python frame-0 result with the MATLAB reference output.
- `scripts/strict_compare_frame0.py` checks voxel, bubble, and property agreement at strict or tolerance-based levels.
- `scripts/export_frame0.py` writes both `.mat` and `.h5` exports for frame 0.
- All generated test outputs are now written under `test/YYYYMMDD-HHMMSS-<name>/` with `report.json` and `report.md`, following `.github/copilot-instructions.md`.
- Current frame-0 status: voxel count, bubble count, property shape, and bubble properties match the MATLAB reference; voxel coordinates match within floating-point tolerance.
