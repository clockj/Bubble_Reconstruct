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

The initial implementation includes:

- an OpenLPT-backed camera adapter built on `pyopenlpt`
- MATLAB-style voxel-grid generation and sparse-to-dense conversion
- coarse visual-hull voting against binary silhouettes stored in MATLAB `.mat` files
- a reconstruction entry point for loading masks and building the first-pass hull

## Source Mapping

- `camera.py` maps OpenLPT camera files to Python projection helpers
- `io.py` loads MATLAB camera-mask files and frame keys
- `voxel_grid.py` ports `InitializeVoxels.m` and `ConvertVoxelListTo3D.m`
- `hull.py` ports the coarse voting logic from `CreateVisualHull_2.m` and `VisualHull.m`
- `reconstruction.py` is the first Python orchestration layer

## Next Implementation Steps

1. Port bubble separation and local refinement from `Reconstruction.m` and `mesh_expand.m`.
2. Port `GetBubbleProps.m` for 3D bubble metrics.
3. Add MATLAB-vs-Python validation scripts for `Islam_0207`.
