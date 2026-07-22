# Bubble Reconstruction Report

**Date:** 2026-07-22  
**Project:** 3D Bubble Surface Reconstruction from Multi-View Cameras  
**Working directory:** `X:\Shijie Zhong\Bubble Shear Project\Processed\20260710\20Hz_r_b_1_lpt`

---

## 1. Data Specifications

### Working folder layout

```
20Hz_r_b_1_lpt/
├── imgFile_bb/                  # Binary bubble masks (TIFF)
│   ├── cam0/img000000.tif ... img012604.tif    (12,605 files)
│   ├── cam1/img000000.tif ... img012604.tif
│   ├── cam2/img000000.tif ... img012604.tif
│   └── cam3/img000000.tif ... img012604.tif
├── camFile_VSC/                 # OpenLPT camera calibration files
│   ├── vsc_cam0.txt             # PINPLATE model (refractive)
│   ├── vsc_cam1.txt
│   ├── vsc_cam2.txt
│   └── vsc_cam3.txt
├── imgFile/                     # Original camera images (for 2D projection viz)
│   ├── cam0/cam0frame000000.tif ...
│   └── ...
├── config.txt                   # VSC tracking configuration
├── Results/
│   ├── recon/                   # 1mm coarse + SH4 results (100 frames)
│   ├── recon_0.5mm_sh8/         # 0.5mm + fixed SH8 (100 frames)
│   ├── recon_0.5mm_sh8_v2/      # 0.5mm + adaptive SH8 v1 (100 frames)
│   ├── recon_parsimonious_v2/   # 0.5mm + parsimonious SH (1 frame)
│   ├── recon_temporal/          # 0.5mm + parsimonious SH (10 frames)
│   ├── recon_temporal_smoothed/ # Post-processed: temporally smoothed SH
│   ├── recon_1mm_refine0.05/    # 1mm → 0.05mm multi-level (1 frame)
│   ├── recon_0.5mm_refine0.05/  # 0.5mm → 0.05mm multi-level (5 frames)
│   ├── viz_sh/                  # PNG visualizations
│   ├── viz_html/                # Interactive HTML visualizations
│   ├── viz_temporal/            # Temporal comparison visualizations
│   └── viz_*/                   # Other visualization outputs
└── config_recon/                # Run configuration JSONs (timestamped)
```

### Camera parameters

| Property | Value |
|---|---|
| Model | PINPLATE (pinhole + refractive plate) |
| Cameras | 4 (0-indexed: cam0–cam3) |
| Image size | 1952 × 2048 px |
| Focal length | 13,346 px |
| Refractive indices | 1.33 (water), 1.49 (acrylic), 1.00 (air) |
| Plate thickness | 12.7 mm |
| Pixel footprint | ~0.05 mm/px at volume center |

### Mask images

| Property | Value |
|---|---|
| Format | TIFF, RGBA uint8 |
| Size | 1952 × 2048 px |
| Foreground | Any non-zero pixel (thresholded to boolean) |
| Naming | `img{frame:06d}.tif` |
| Frames total | 12,605 (0–12604) |

### Reconstruction volume

| Parameter | Value |
|---|---|
| X range | -85.0 to 45.0 mm |
| Y range | -60.0 to 50.0 mm |
| Z range | -40.0 to 70.0 mm |
| Volume size | 130 × 110 × 110 mm |

---

## 2. Pipeline Architecture

```
Multi-view TIFF masks (4 cameras)
       │
       ▼
┌──────────────────────────────────┐
│ Step 1: Coarse Visual Hull       │  create_visual_hull()
│ - Initialize voxel grid          │  Voxel centers projected through
│ - Project through 4 cameras      │  refractive camera model (pyopenlpt)
│ - Unanimous-vote intersection    │  → 3D binary voxel volume
└──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ Step 2: Surface Component Find   │  find_surface_components()
│ - 6-connected erosion → shell    │
│ - 26-connected labeling          │  → Per-bubble surface point sets
└──────────────────────────────────┘
       │
       ├── Default: refine_surface_points(res_inc=3) → voxel_size/3
       │
       ├── Multi-level (--refine-to 0.05): recursive refinement
       │   1mm → 0.25mm → 0.0625mm → ~0.05mm
       │
       ▼
┌──────────────────────────────────┐
│ Step 3: Property Extraction      │  get_bubble_props()
│ - Volume, centroid, axes         │
│ - Aspect ratio, boundary check   │  → Per-bubble property vector (15)
└──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ Step 4: Spherical Harmonics Fit  │  fit_spherical_harmonic_surface()
│ - r(θ,φ) = Σ c_lm · Y_l^m(θ,φ)  │
│ - Parsimonious degree selection  │  → SH coefficients + mesh per bubble
│ - Tikhonov regularization        │
└──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ Step 5: Temporal Post-Processing │  temporal_smooth_sh.py
│ - Hungarian matching (centroid   │
│   distance + volume similarity)  │
│ - Gaussian temporal smoothing    │  → Smoothed SH coefficients
│ - Volume conservation            │
└──────────────────────────────────┘
```

### Voxel refinement strategies

| Strategy | Coarse (mm) | Fine (mm) | Voxels/frame | Bubbles | Use case |
|---|---|---|---|---|---|
| Default | 1.0 | 0.33 | ~393 | 6 | Quick test |
| Default | 0.5 | 0.17 | ~3,666 | 14 | Standard |
| Multi-level | 1.0 | 0.05 | ~120,000 | 6 | High-res surfaces |
| Multi-level | 0.5 | 0.05 | ~140,000 | 14 | Best quality |

---

## 3. Spherical Harmonics Configuration

### Degree selection

| Method | Description | Option |
|---|---|---|
| **Parsimonious** (default) | Fit increasing degrees (2→4→6→8), stop when improvement < 5% | `--sh-degree 8` |
| Fixed | All bubbles use same degree | `--sh-degree 8 --sh-no-adaptive` |
| Voxel-count heuristic | Degree based on N_voxels / 10 per coefficient | (legacy fallback) |

### Parsimonious degree criteria

1. Try degree 2, 4, 6, 8 sequentially
2. Stop if RMSE < 0.01 mm (below pixel resolution)
3. Stop if relative RMSE improvement < 5%
4. Fall back to voxel-count heuristic for < 20 voxels

### Regularization

| Parameter | Default | Effect |
|---|---|---|
| `--sh-regularization` | 1e-2 | Higher → smoother shapes, prevents overfitting |

### Output per bubble

| Field | Description |
|---|---|
| `center` | (3,) centroid in world mm |
| `coefficients` | (K,) SH coefficients c_lm, K = (L_max+1)² |
| `basis_terms` | [(l,m)] pairs for each coefficient |
| `vertices` | (V, 3) surface mesh vertices |
| `faces` | (F, 3) triangular faces (1-indexed) |
| `fit_rmse` | RMS error of SH radius vs voxel radii (mm) |
| `degree_used` | Actual degree used for this bubble |

---

## 4. Temporal Tracking & Smoothing

### Algorithm

```
Frame 0 SH ─┐
Frame 1 SH ─┤  Hungarian matching    ──→  Trajectories
Frame 2 SH ─┤  cost = d_center +      (per-bubble)
   ...     ─┘  0.2·Δvol·threshold
                     │
                     ▼
            Gaussian temporal smoothing
            kernel width = sigma frames
                     │
                     ▼
            Volume conservation:
            rescale c_00 to mean volume
                     │
                     ▼
            Smoothed SH coefficients
            + regenerated meshes
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--match-dist-threshold` | 15 mm | Max centroid distance for matching |
| `--smooth-sigma` | 1.5 frames | Gaussian kernel width |

### Current results (10 frames @ 20Hz)

| Metric | Value |
|---|---|
| Trajectories found | 26 |
| Mean trajectory length | 1.7 frames |
| Max trajectory length | 5 frames |
| Smoothed frames | 8/10 |

*Note: Short trajectories are expected for 10 frames at 20 Hz — bubbles rapidly enter/exit the 130×110×110 mm volume. With 100+ frames, trajectories will lengthen naturally.*

---

## 5. Package Changes

### New scripts created in `visual_hull/scripts/`

| Script | Lines | Purpose |
|---|---|---|
| `reconstruct_tiff_data.py` | ~700 | Main reconstruction: TIFF masks → voxels → SH → .mat |
| `visualize_sh_bubbles.py` | ~330 | 3D visualization: PNG (matplotlib) or HTML (plotly) |
| `temporal_smooth_sh.py` | ~600 | Temporal tracking + smoothing + 3D/2D comparison viz |

### Modified package files

| File | Change |
|---|---|
| `src/visual_hull/io.py` | Added `load_tiff_mask()` + `load_tiff_masks()` for TIFF loading |
| `src/visual_hull/reconstruction.py` | Added `run_full_reconstruction_from_data()` for pre-loaded masks/cameras |

---

## 6. Commands Reference

### Environment setup

```powershell
# Activate conda environment
$env:Path = "d:\Bubble_Reconstruct\Bubble_Reconstruct.conda-visual-hull\Library\bin;" +
            "d:\Bubble_Reconstruct\Bubble_Reconstruct.conda-visual-hull;" +
            "d:\Bubble_Reconstruct\Bubble_Reconstruct.conda-visual-hull\Scripts;" +
            $env:Path
```

### Reconstruction

```powershell
# Quick test (1 frame, 1mm coarse)
python scripts/reconstruct_tiff_data.py --frames 0 --dry-run

# Standard quality (0.5mm coarse, parsimonious SH8, 8 workers)
python scripts/reconstruct_tiff_data.py \
    --frames 0 99 \
    --voxel-size 0.5 0.5 0.5 \
    --sh-degree 8 \
    --max-workers 8 \
    --output-dir .../Results/recon_standard

# High-res surfaces (0.5mm → 0.05mm multi-level refine)
python scripts/reconstruct_tiff_data.py \
    --frames 0 99 \
    --voxel-size 0.5 0.5 0.5 \
    --refine-to 0.05 \
    --sh-degree 8 \
    --max-workers 8 \
    --output-dir .../Results/recon_hires

# Fixed SH degree (no adaptive)
python scripts/reconstruct_tiff_data.py \
    --frames 0 99 --sh-degree 8 --sh-no-adaptive ...

# Stronger/weaker regularization
python scripts/reconstruct_tiff_data.py \
    --frames 0 99 --sh-degree 8 --sh-regularization 0.05 ...
```

### CLI reference (`reconstruct_tiff_data.py`)

| Option | Default | Description |
|---|---|---|
| `--working-dir` | `X:\...\20Hz_r_b_1_lpt` | Root working directory |
| `--frames N M` | `0` | Frame range [N, M] inclusive |
| `--voxel-size DX DY DZ` | `1.0 1.0 1.0` | Coarse voxel size (mm) |
| `--refine-to MM` | `None` | Multi-level refine target (mm) |
| `--sh-degree N` | `0` | SH max degree (0 = disabled) |
| `--sh-no-adaptive` | `False` | Disable parsimonious degree |
| `--sh-regularization F` | `0.01` | Tikhonov regularization |
| `--max-workers N` | `-1` (80% CPU) | Parallel workers |
| `--dry-run` | — | Print config, don't run |
| `--output-dir PATH` | `.../Results/recon` | Output directory |
| `--format mat/h5` | `mat` | Export format |

### Visualization

```powershell
# Static PNG (matplotlib)
python scripts/visualize_sh_bubbles.py \
    --frames 0 10 25 50 99 \
    --recon-dir .../Results/recon_temporal \
    --show-voxels

# Interactive HTML (plotly — zoom, rotate, pan)
python scripts/visualize_sh_bubbles.py \
    --frame 0 \
    --recon-dir .../Results/recon_temporal \
    --format html --show-voxels
```

### CLI reference (`visualize_sh_bubbles.py`)

| Option | Default | Description |
|---|---|---|
| `--frames N ...` | `0 5 10 50 99` | Frames to visualize |
| `--frame N` | — | Single frame shortcut |
| `--recon-dir PATH` | auto | Directory with SH .mat files |
| `--format png/html` | `png` | Output format |
| `--show-voxels` | `False` | Overlay raw voxels |
| `--interactive` | `False` | Open matplotlib Qt window |
| `--dpi N` | `150` | PNG resolution |

### Temporal post-processing

```powershell
# Smooth + generate comparison PNGs
python scripts/temporal_smooth_sh.py \
    --frames 0 99 \
    --recon-dir .../Results/recon_standard \
    --visualize --format png

# Smooth + interactive HTML comparison
python scripts/temporal_smooth_sh.py \
    --frames 0 99 \
    --recon-dir .../Results/recon_standard \
    --visualize --format html \
    --match-dist-threshold 30 --smooth-sigma 2.0
```

### CLI reference (`temporal_smooth_sh.py`)

| Option | Default | Description |
|---|---|---|
| `--frames N M` | `0 9` | Frame range to process |
| `--recon-dir PATH` | auto | Directory with SH .mat files |
| `--output-dir PATH` | auto | Smoothed output directory |
| `--visualize` | `False` | Generate comparison figures |
| `--format png/html` | `png` | Visualization format |
| `--match-dist-threshold F` | `15.0` | Max centroid distance (mm) |
| `--smooth-sigma F` | `1.5` | Gaussian kernel sigma (frames) |

---

## 7. Output File Formats

### `Bubble_Frame_######.mat` (baseline voxel reconstruction)

| Field | Shape | Description |
|---|---|---|
| `voxels` | (N, 3) | Refined surface point coordinates (mm) |
| `bubbles` | (2, B) | [start_idx, end_idx] per bubble (1-indexed) |
| `properties` | (B, 15) | Per-bubble: centroid, radius, volume, aspect_ratio, axes |
| `voxel_size` | (3,) | Coarse voxel size (mm) |
| `voxel_size_2` | (3,) | Fine voxel size (mm) |
| `limits` | (6,) | Volume limits [xmin, xmax, ymin, ymax, zmin, zmax] |
| `real_images` | (H, W, N_cam) | Stacked boolean masks |
| `completed` | bool | Reconstruction success flag |

### `Bubble_Frame_######_sh.mat` (spherical harmonics)

| Field | Shape | Description |
|---|---|---|
| `sh_max_degree` | (1, 1) | Global maximum SH degree |
| `sh_num_bubbles` | (1, 1) | Number of bubbles fitted |
| `sh_centers` | (B, 3) | Bubble centers (mm) |
| `sh_coefficients` | (B, K) | SH coefficients (padded, K = (L_max+1)²) |
| `sh_basis_l` | (B, K) | Degree l for each coefficient |
| `sh_basis_m` | (B, K) | Order m for each coefficient |
| `sh_vertices` | (B, V, 3) | Surface mesh vertices |
| `sh_faces` | (B, F, 3) | Triangular faces (1-indexed) |
| `sh_fit_rmse` | (1, B) | Per-bubble fit RMSE (mm) |
| `sh_degree_used` | (1, B) | Actual degree used per bubble |

### `reconstruction_summary.json`

```json
[{
  "frame": 0,
  "output": ".../Bubble_Frame_000000.mat",
  "voxel_count": 3666,
  "bubble_count": 14,
  "completed": true,
  "sh_saved": true,
  "sh_output": ".../Bubble_Frame_000000_sh.mat",
  "sh_num_bubbles": 14,
  "sh_fit_rmse_mean": 0.092
}]
```

---

## 8. Results Summary

### 100-frame runs compared

| Metric | 1mm + SH4 | 0.5mm + SH8 fixed | 0.5mm + SH8 adaptive | 0.5mm + SH8 parsimonious |
|---|---|---|---|---|
| Voxels (total) | 35,788 | 314,076 | 314,076 | 35,582 (10 fr) |
| Voxels/frame | ~358 | ~3,141 | ~3,141 | ~3,558 |
| Bubbles (total) | 646 | 1,172 | 1,172 | 134 (10 fr) |
| Bubbles/frame | 6.5 | 11.7 | 11.7 | 13.4 |
| Mean SH degree | 4.0 | 8.0 | ~4.8 | **~3.3** |
| Mean SH RMSE | 0.145 mm | 0.076 mm | 0.107 mm | 0.092 mm |
| Flower petals | N/A (too simple) | ❌ on small | ✅ none | ✅ none |
| Overfitting | No | Yes (small bubbles) | Minimal | **Minimal** |

### Multi-level refinement (5 frames)

| Metric | 1mm → 0.05mm | 0.5mm → 0.05mm |
|---|---|---|
| Voxels/frame | 120,146 | ~140,000 |
| Bubbles/frame | 6 | 14 |
| SH RMSE | 0.161 mm | 0.114 mm |
| Min feature resolved | 0.05 mm | 0.05 mm |

---

## 9. Key Design Decisions

1. **OpenLPT is the authoritative projection model** — all camera projection uses `pyopenlpt` with refractive PINPLATE model
2. **Parsimonious SH** — minimum modes needed, no overfitting. Degree selected per-bubble by sequential fitting with 5% improvement threshold and 0.01mm RMSE floor
3. **Multi-level refinement** — hierarchical voxel refinement (4-5× per level) down to pixel resolution (~0.05 mm). Sub-pixel refinement adds no new visual-hull information but SH surface remains continuous
4. **Temporal smoothing** — Hungarian matching + Gaussian kernel smoothing + volume conservation. Post-processing step that doesn't modify raw reconstruction
5. **Surgical code changes** — only 2 package files modified (io.py, reconstruction.py), each with minimal new functions. All scripts are standalone additions

---

## 10. Limitations & Future Work

- **0.3mm direct voxel grid** is infeasible with current full-volume limits (58M voxels, ~1.4 GB). Use multi-level refinement (`--refine-to`) or reduce limits
- **Short trajectories** in 10-frame window — natural at 20 Hz. Run on 100+ frames for meaningful temporal statistics
- **No Kalman filtering** yet — current smoothing is Gaussian-weighted moving average. Kalman filter would better handle varying trajectory lengths
- **No bubble splitting/merging** in temporal tracking — each bubble treated independently
- **SH assumes star-shaped surfaces** — cannot represent concave or toroidal bubble geometries
