# Visual Hull Python Port Plan

## Goal
Port the existing MATLAB bubble-reconstruction workflow into Python under this folder, using a dedicated conda environment and OpenLPT camera objects as the primary projection path.

## Steps
1. Create and use a dedicated conda environment with OpenLPT, NumPy, SciPy, scikit-image, tifffile or imageio, matplotlib, and h5py.
2. Build a thin OpenLPT camera adapter that owns camera loading and projection semantics.
3. Port MATLAB-compatible data loading for masks, camera assets, and output naming.
4. Port the coarse visual hull stage from `InitializeVoxels.m`, `CreateVisualHull_2.m`, `ConvertVoxelListTo3D.m`, and `VisualHull.m`.
5. Port bubble separation and local refinement from `Reconstruction.m` and `mesh_expand.m`.
6. Port bubble property extraction from `GetBubbleProps.m`.
7. Implement a Python orchestration entry point matching the current MATLAB workflow.
8. Validate against the `Islam_0207` example outputs.
9. Profile and optimize projection-heavy sections without replacing OpenLPT as the camera model.

## Key Decisions
- Target folder: `d:/Bubble_Reconstruct/Bubble_Reconstruct/visual_hull`
- Environment: dedicated conda environment for this port
- Projection: OpenLPT camera objects are the authoritative implementation
- First-pass scope: match current MATLAB behavior before algorithm redesign

## Relevant Source Files
- `imgProcess.m`
- `code/Reconstruction.m`
- `code/InitializeVoxels.m`
- `code/CreateVisualHull_2.m`
- `code/VisualHull.m`
- `code/ConvertVoxelListTo3D.m`
- `code/mesh_expand.m`
- `code/GetBubbleProps.m`
- `code/calibProj_Pinhole.m`
- `convertCamFile.py`
