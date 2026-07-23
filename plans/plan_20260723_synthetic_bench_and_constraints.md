# Plan — Synthetic-Shape Benchmark + SH Constraint Improvements (2026-07-23)

Derived from a critical review of `plans/plan_20260723.md` against the current
codebase and the synthetic-sphere / 50-frame results. Builds a known-answer test
harness, then adds three constraint groups, measuring each against the harness.

## Context / rationale
- ~60% of `plan_20260723.md` is already implemented (visual hull, 3D CC separation,
  per-bubble back-projection, `--sh-inscribed` = H1, `--sh-silhouette` = D1,
  `--sh-min-points-per-coeff` flower cap = H2-ish, `--max-aspect-ratio` = P5,
  Tikhonov = R1, Gaussian temporal = R2).
- Key known result: 4-view visual hull over-estimates ~+6% D / +19% V (systematic).
  Constraining the surface *inside* the hull cannot fix this; P1 volume-conservation
  to VH volume would *lock in* the bias, so it is excluded.
- User-selected scope: Phase 1 (fit init + spectral), Phase 2 (mask-quality
  view-weighting), Phase 3 (physical priors P2-P4), plus a synthetic-shape harness
  (spheres/ellipsoids/convex SH shapes) to validate the algorithm — Phase 0.

## Phase 0 — Synthetic-shape test harness (foundation)
- Extend the proven `scripts/validate_synthetic_spheres.py` approach (renders exact
  silhouettes through the real 4-camera PINPLATE model via OpenLPT).
- `src/visual_hull/synthetic_shapes.py`: generators for
  - sphere(R), ellipsoid(a,b,c, orientation) — analytic volume / axes,
  - convex SH shape from prescribed low-order c_lm (star-convex, r>0),
  - optional mask corruption (holes/speckle) and two-bubble overlap scenes.
- `scripts/validate_synthetic_shapes.py`: run recon, report per shape volume/diameter
  bias, SH-coefficient error, 3D IoU, projection IoU, aspect ratio, curvature stats.
- Output: `test/<ts>-synthetic-shape-bench/`.
- Success: reproduces +6%D/+19%V on a clean sphere; recovers ellipsoid axis ratios
  within tolerance.

## Phase 1 — Fit initialization + spectral penalty (cheap)
- `improved/spherical_harmonics/surface.py`: `(l-2)^2` spectral penalty (l=0,1,2 free);
  ellipsoid init (PCA on hull voxels -> l=0,2), optional previous-frame init.
- Flags on `reconstruct_tiff_data.py`: `--sh-spectral-weight`, `--sh-init`.
- Measure: SH-coeff error + flower suppression at higher degree.

## Phase 2 — Mask-quality + view-weighting (Steps 1.4-1.5, best new idea)
- `improved/view_quality.py`: per bubble x camera overlap ratio, mask quality
  (missing/extra vs observed), boundary classification (reliable/overlapped/mask_error).
- Feed per-camera weights + classification into the silhouette optimizer: weight each
  camera IoU; trust observed boundary where reliable, upper-bound where overlapped,
  projected-hull where mask_error.
- Flag: `--sh-view-weighting`.
- Measure: corrupted-mask + overlap scenes vs equal-weight silhouette-opt.

## Phase 3 — Physical priors P2-P4 (implement, benchmark judges)
- `surface.py`: analytic mean/Gaussian curvature from r(theta,phi); soft penalties
  Var(H), max(|k|-k_max,0)^2, max(-K,0)^2.
- Flags: `--sh-curv-weight`, `--sh-maxcurv-weight`, `--sh-convex-weight`.
- Measure: convex synthetic shapes — remove spurious concavities without biasing
  volume or flattening real deformation.

## Excluded
- P1 volume-conservation-to-hull-volume (locks in +19% bias).
- Full unified L-BFGS multi-constraint optimizer (defer; tuning burden).
- Explicit bias correction (not selected; harness will still report residual bias).

## Working rules
- All outputs under `test/<YYYYMMDD-HHMMSS>-synthetic-shape-bench/`.
- Edit under `src/`, never `build/`. OpenLPT is the only projection path.
- Measure each phase before the next; final combined report at the end.
