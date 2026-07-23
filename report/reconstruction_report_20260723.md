# Bubble Reconstruction Report

**Date:** 2026-07-23
**Project:** 3D Bubble Surface Reconstruction from Multi-View Cameras (Python port of MATLAB visual-hull)
**Author:** Claude Code session

---

## 1. Summary of the Current Algorithm

The pipeline reconstructs 3D bubble surfaces from **4 refractive multi-view cameras** and now
runs end-to-end from raw TIFF masks through a temporally-smoothed, track-labelled surface set.

### Stage A — Per-frame reconstruction (`scripts/reconstruct_tiff_data.py`)

1. **Load masks** — binary bubble masks per camera (`imgFile_bb/cam{0..3}/img{frame:06d}.tif`).
   Optional `--clean-mask-border` zeros saturated edge bands (no-op on the current dataset).
2. **Visual hull** — intersect the 4 refractive viewing cones (OpenLPT PINPLATE model) into a
   coarse voxel grid (default 1.0 mm; the 50-frame study used 0.5 mm). Soft/hard voting via
   `visual_hull.improved.hull`.
3. **Surface refinement** — erode to a surface shell, split into connected components
   (one per bubble), and locally re-vote at finer resolution around each component.
4. **Per-bubble properties** — centroid, equal-volume radius, voxel volume, aspect ratio
   (major/minor axis magnitude), principal axes. Stored as a 15-column `properties` array.
5. **Filtering** (both optional, applied before SH):
   - `--size-range MIN MAX` — keep only bubbles with equivalent diameter
     `D_eq = 2·(3V/4π)^(1/3)` in `[MIN, MAX]` mm.
   - `--max-aspect-ratio` — drop elongated **visual-hull ghost** slivers (aspect > ~5;
     real bubbles are ~1.5–3).
6. **Spherical-harmonic surface fit** (`--sh-degree`, `improved/spherical_harmonics/surface.py`):
   - **Flower fix / degree cap** — `--sh-min-points-per-coeff` (default 3) caps each bubble's
     degree so an undersampled fit cannot oscillate into negative-radius "flower petals".
     Root cause was points-per-coefficient < 1.
   - **Inscribed fit** (`--sh-inscribed`) — fit the hull's *outer envelope* (max radius per
     angular bin) with a one-sided IRLS penalty on outward overshoot, so the surface hugs the
     inside of the hull boundary instead of bulging past it.
   - **Silhouette optimization** (`--sh-silhouette`) — coordinate-descent the SH coefficients to
     maximize IoU against each bubble's *own re-projected* visual-hull silhouette in every
     camera (an overlap-free target), at a downsampled resolution.
7. **Export** — per-frame `Bubble_Frame_*.mat` (voxels + properties) and `*_sh.mat`
   (SH coefficients, degree used, mesh) plus a `reconstruction_summary.json`.

### Stage B — Temporal tracking + smoothing (`scripts/temporal_smooth_sh.py`)

- **Matcher** (`match_bubbles`) — pure nearest-neighbour centroid distance, gated by
  `--match-dist-threshold` (mm) **and** `--max-diameter-ratio` (default 1.5). Distance is the
  primary criterion; size similarity is a lenient guard against identity switches.
- **Trajectories** (`build_trajectories`) — start a track on any match, carry it as long as it
  keeps matching, and emit singletons so **every** bubble is covered (untracked bubbles pass
  through unsmoothed).
- **Smoothing** (`apply_temporal_smoothing`) — Gaussian filter (σ frames) over each track's
  centroid and SH coefficients; reduces frame-to-frame jitter while preserving true trends.
- **Track identity** — each output bubble carries a stable **`sh_track_id`**; the same physical
  bubble keeps its ID (and therefore its color in the visualizations) across all frames.

### Stage C — Visualization

- `scripts/visualize_raw_html.py` — raw voxel point cloud (all bubbles, colored per-frame
  index) + camera-projection overlay. Interactive Plotly HTML, frame slider, Plotly via CDN.
- `scripts/visualize_smoothed_html.py` — smoothed SH surfaces, **colored by `sh_track_id`**
  (golden-ratio hue spacing) in both 3D and the 4-camera projection views.

### Key findings baked into the design

- **Visual-hull bias:** 4-view hull systematically **over-estimates ~+6% diameter / +19% volume**
  (from a synthetic-sphere study). Independent of size/location/resolution.
- **Temporal filtering fixes precision, not accuracy:** Gaussian smoothing cuts volume jitter
  ~×2.7–3.4 but does **not** remove the systematic bias. Bias needs calibration or the
  silhouette-consistent fit; smoothing is complementary.
- **SH quality:** L≤4 + inscribed + silhouette-opt raised mean projection IoU vs the hull
  silhouette from **0.60 → 0.86** (frame 16), at lower degree than the old deg-8 fit.

---

## 2. Input / Output

### Inputs

| Input | Description |
|-------|-------------|
| Masks | `imgFile_bb/cam{0..3}/img{frame:06d}.tif` — binary bubble segmentation, one folder per camera (`--mask-dir`, `--mask-template`) |
| Calibration | `camFile_VSC/vsc_cam{0..3}.txt` — OpenLPT PINPLATE refractive camera files (`--camera-dir`, `--camera-template`) |
| Volume | `--limits XMIN XMAX YMIN YMAX ZMIN ZMAX` (mm), `--voxel-size DX DY DZ` (mm) |
| Frames | `--frames START END` (inclusive range) |

### Outputs (`<working-dir>/Results/<name>/`)

| Output | Content |
|--------|---------|
| `Bubble_Frame_{n}.mat` | Voxel coordinates + 15-col `properties` (centroid, radius, volume, aspect, axes) |
| `Bubble_Frame_{n}_sh.mat` | `sh_coefficients`, `sh_degree_used`, mesh vertices/faces per bubble |
| `reconstruction_summary.json` | Per-frame counts, totals, config |
| smoothed `.mat` (Stage B) | Same layout + **`sh_track_id`** per bubble |
| `*.html` (Stage C) | Interactive 3D + camera-projection viewers, frame slider |

### Properties array (15 columns)

`[0:3]` centroid · `[3]` equal-volume radius · `[4]` voxel volume · `[5]` aspect ratio ·
`[6]` in-boundary flag · `[7:10]` major axis · `[10:13]` minor axis · `[13]` major mag · `[14]` minor mag.

---

## 3. Working Folder

**Data root** (network share, `X:` → `\\ruisrv5.wse.jhu.edu\data`):
`X:\Shijie Zhong\Bubble Shear Project\Processed\20260710\20Hz_r_b_1_lpt\`

```
20Hz_r_b_1_lpt/
├── imgFile_bb/cam{0..3}/img000000.tif … img012604.tif   # 12,605 frames × 4 cams (binary masks)
├── camFile_VSC/vsc_cam{0..3}.txt                          # PINPLATE refractive calibration
├── imgFile/cam{0..3}/…                                    # original images (for projection overlay)
├── config.txt
└── Results/
    ├── recon_test50_v2/            # 50-frame v2 run (0.5 mm, deg≤4 + inscribed + silhouette + aspect filter)
    └── recon_test50_v2_smoothed/   # temporally smoothed + sh_track_id
```

**Code root:** `D:\Bubble_Reconstruct\Bubble_Reconstruct\visual_hull\`
**Environment:** conda env `bubble-visual-hull`
(`C:\Users\zcloc\.conda\envs\bubble-visual-hull\python.exe`; numpy 2.4, scipy, scikit-image, pyopenlpt).
Invoke scripts via PowerShell (`& '…python.exe' '…script.py' …`).

---

## 4. Test Status

| Study | Location | Result |
|-------|----------|--------|
| Flower fix + size filter | `test/20260722-132445-flower-sizefilter-frame2/` | Degree cap eliminates all flower artifacts; size filter drops out-of-range bubbles. ✅ |
| Synthetic-sphere bias | `test/20260722-142124-synthetic-sphere-bias/` | Quantified **+6% D / +19% V** over-estimate; size/location/resolution-independent. ✅ |
| 50-frame v2 pipeline | `test/20260722-150800-track50-eval/` | 50/50 frames, 547 bubbles; deg≤4 + inscribed + silhouette-opt + aspect filter. ✅ |
| SH quality (frame 16) | same | Projection IoU vs hull silhouette **0.60 → 0.86**; overshoot 27.6% → 16.9%. ✅ |
| Aspect filter (frame 16) | same | 14 → 12 bubbles; phantoms (aspect 12.9, 6.1) removed, all remaining ≤3.1. ✅ |
| Tracker fix | same, `tracking-and-artifacts` memory | `build_trajectories` bug fixed: mean track length 1.8 → 7.1, max 33 → 50; smoothed bubbles/frame 3.1 → 10.8 (== raw). Large ~2 mm bubble now a full 50-frame track. ✅ |
| Track-ID coloring | `test/20260722-150800-track50-eval/viz_v2_trackcolor/` | `sh_track_id` propagated; track 43 (large bubble) verified same color across frames 0/10/25/49. ✅ |

**Run command (v2, 50 frames):**
```
reconstruct_tiff_data.py --frames 0 49 --voxel-size 0.5 0.5 0.5 \
    --sh-degree 4 --sh-inscribed --sh-silhouette --max-aspect-ratio 5
```

**Not yet run:** full 12,605-frame dataset; velocity-gated tracker; bias-correction calibration
applied to production output.

---

## 5. Code Structure

```
visual_hull/
├── src/visual_hull/
│   ├── camera.py                  # OpenLPTCameraSet — refractive projection (authoritative)
│   ├── hull.py / improved/hull.py # hard-vote / soft-vote (SDT + bilinear) visual hull
│   ├── refinement.py, improved/refinement.py   # surface shell extraction + fine re-voting
│   ├── properties.py              # per-bubble properties (15-col array)
│   ├── io.py                      # mask loading; clean_mask_border()
│   ├── improved/surface.py        # Laplacian/Taubin mesh smoothing
│   ├── improved/spherical_harmonics/surface.py  # SH fit: degree cap, inscribed IRLS, envelope
│   ├── silhouette_metrics.py      # IoU / overlap metrics
│   ├── writers.py, models.py, voxel_grid.py, test_runs.py, visualization.py
├── scripts/
│   ├── reconstruct_tiff_data.py   # Stage A driver (TIFF → voxels + SH)  ← main entry
│   ├── temporal_smooth_sh.py      # Stage B tracking + smoothing + sh_track_id
│   ├── visualize_raw_html.py      # Stage C raw voxel viewer
│   ├── visualize_smoothed_html.py # Stage C smoothed, track-colored viewer
│   └── (legacy compare_/benchmark_/smoke_test scripts)
├── test/<YYYYMMDD-HHMMSS>-<name>/  # all test outputs (never overwritten)
└── environment.yml, pyproject.toml
```

### Key CLI flags — `reconstruct_tiff_data.py`

`--frames`, `--voxel-size`, `--limits`, `--refine-to`, `--num-cameras`, `--max-workers` ·
`--sh-degree`, `--sh-regularization`, `--sh-min-points-per-coeff` (flower cap),
`--sh-inscribed` (+`--sh-overshoot-weight`), `--sh-silhouette` (+`--sh-silhouette-scale`,
`--sh-silhouette-passes`) · `--size-range MIN MAX`, `--max-aspect-ratio`, `--clean-mask-border`.

### Key CLI flags — `temporal_smooth_sh.py`

`--frames`, `--recon-dir`, `--match-dist-threshold`, `--max-diameter-ratio` (default 1.5),
`--smooth-sigma`, `--format {png,html}`.

---

## 6. Recommended Next Steps

1. **Bias correction** — apply the +6% D / +19% V calibration to production sizes (accuracy).
2. **Velocity-gated tracker** — long tracks are rare at 20 Hz; a motion model would earn more
   correct tracks and make temporal smoothing pay off for more bubbles.
3. **Scale-up** — apply the v2 pipeline to a larger frame range / the full 12,605-frame dataset.
4. **3D trajectory view** — draw each track's centroid path over all frames (offered, not built).
```
