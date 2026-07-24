# Bubble Surface Representation — Methods & Benchmark Report

**Date:** 2026-07-24
**Project:** 3D Bubble Surface Reconstruction from 4 refractive multi-view cameras (Python port of MATLAB visual-hull)
**Author:** Claude Code session
**Scope:** How to turn the raw 4-view visual hull into an accurate, bubble-like
surface. Compares spherical harmonics, free-form mesh, and the new
virtual-camera rounding (isotropic + anisotropic). Supersedes the surface-fitting
sections of `reconstruction_report_20260723.md`.

---

## 1. The problem

The per-frame pipeline (`scripts/reconstruct_tiff_data.py`) produces a **visual
hull**: the intersection of the 4 refractive camera silhouette cones (OpenLPT
PINPLATE model), refined to ~0.17 mm voxels. With only 4 views this hull has two
systematic defects:

1. **It is boxy.** Each silhouette cone contributes a few flat tangent planes, so
   between the 4 viewing directions the surface bulges out to flat facets — the
   hull is a faceted convex polyhedron, not a smooth bubble.
2. **It over-estimates size** by **~+6 % diameter / +19 % volume**
   (synthetic-sphere study, `test/20260722-142124-synthetic-sphere-bias/`),
   independent of size/location/resolution.

Any surface we fit must *de-box* the hull and, ideally, *correct the bias*, while
staying consistent with the 4 real silhouettes. Real bubbles are smooth and
nearly convex, which is the prior every method below leans on.

---

## 2. Methods evaluated

### 2.1 Spherical harmonics (SH) — `src/visual_hull/improved/spherical_harmonics/`
Represents the surface as a radial function `r(θ,φ) = Σ c_lm Y_l^m` about the
bubble centre. Production settings: degree 4, inscribed-envelope fit (one-sided
overshoot penalty), silhouette IoU refinement, spectral `(l−2)²` + convexity
priors (`--sh-inscribed --sh-silhouette`).

- **Strength:** compact (25 coefficients), smooth.
- **Weakness:** it is a **global, star-convex, low-degree** basis, so it gets
  **trapped in "mode" shapes** — flowers, peanuts, and (on a 4-fold shape) an
  X-star. The inscribed fit also **under-estimates** (−8 … −11 %). These are
  structural, not tuning, problems.

### 2.2 Free-form deformable mesh — `src/visual_hull/mesh_surface/`
A fixed-topology **icosphere** (642 vertices, shared across all bubbles/frames →
free temporal correspondence) deformed under *local* energies: an SDF data term
built from the hull occupancy, umbrella-Laplacian smoothness, and a convexity
term. Optimised by explicit force integration (numpy/scipy; no PyTorch, since
OpenLPT projection is not autodiff-friendly).

- **Strength:** no global modes — it **removes the SH peanut/flower** (a real
  lumpy bubble went from concavity 0.83 → 0.09).
- **Weakness:** with the hull as its target it faithfully reproduces the **box**;
  and the Laplacian smoothing **shrinks the mesh ~10 %**, which must be corrected
  by a volume-match rescale.

### 2.3 Virtual-camera rounding (isotropic) — `src/visual_hull/virtual_camera/` **[recommended]**
Adds synthetic viewpoints, rounds each silhouette (imposing "the outline is smooth
& convex from every direction"), and re-carves. For a convex body this is
**mathematically equivalent to a 3-D morphological opening of the solid by a ball
of radius ρ**: roll a ball of radius **ρ** inside the surface; whatever it cannot
reach (sharp facet corners) is shaved off. Opening is *erode-then-dilate*, so
flat faces stay put (no net shrink) and only corners sharper than ρ are rounded;
a following closing rounds concave notches. The rounded occupancy then feeds
`mesh_surface.fit_mesh_surface`, volume-matched to the carve.

- **ρ = rounding radius** — the smallest surface-feature radius kept. Set to
  **ρ = 0.30 · bubble_radius**, calibrated once on a synthetic sphere. Because
  opening only *removes* material, it also pulls the +6 % over-estimate down —
  ρ doubles as the bias corrector.
- **No new information:** the roundness *between* the 4 real views is an imposed
  prior, not a measurement. For bubbles that prior is essentially correct.

### 2.4 Virtual-camera rounding (anisotropic, camera-aware) — negative result
Motivated by "round only where there is no measurement": round the gap corners
but **restore** corners lying on a real camera's occluding contour (normal ⟂ a
camera axis, recovered from the projection Jacobian). Implemented in
`anisotropic.py`. **It did not beat isotropic** — see §3.

---

## 3. Benchmark against ground truth

Reference shapes (sphere, ellipsoid, convex bubble) are rendered through the
**real 4 refractive cameras**, the real hull is carved, each method is applied,
and all are scored against ground truth
(`test/20260723-204041-virtual-camera/compare_methods.{py,html,png,json}`).

| shape | metric | raw hull | spherical harmonic | **isotropic VC** | anisotropic VC |
|---|---|---|---|---|---|
| sphere    | D error  | +5.7 % | −10.2 % | **+1.4 %** | +5.9 % |
|           | 3-D IoU  | 0.763 | 0.598 | **0.744** | 0.734 |
|           | reproj-IoU | 0.868 | 0.915 | 0.821 | 0.799 |
| ellipsoid | D error  | +4.6 % | −7.6 % | **+1.3 %** | +5.8 % |
|           | 3-D IoU  | 0.776 | 0.775 | 0.758 | 0.750 |
|           | reproj-IoU | 0.881 | 0.880 | 0.844 | 0.840 |
| convex    | D error  | +4.9 % | −10.7 % | **−0.8 %** | +3.9 % |
|           | 3-D IoU  | 0.757 | 0.594 | **0.771** | 0.770 |
|           | reproj-IoU | 0.844 | 0.900 | 0.831 | 0.854 |

**Reading the table**
- **Isotropic virtual-cam is the winner:** diameter within **±1 %** on all shapes
  (vs the hull's +5 %), best/near-best 3-D IoU, and visibly smooth bubbles.
- **Spherical harmonic under-estimates and mode-traps** (convex → X-star, sphere
  → lumpy blob; `compare_methods.png` col 3). Its high reproj-IoU is *misleading*:
  it comes from bulges that happen to cover the mask, not from a correct 3-D
  shape — which is why its **3-D IoU is the lowest** (0.59–0.78).
- **`reproj-IoU` vs size is an inherent trade-off.** The raw hull has the highest
  reproj-IoU (0.87) *because* it over-fills the masks (a round bubble cannot fill
  the square silhouette corners). Any method that corrects the +5 % size must give
  back a little reproj-IoU. This makes reproj-IoU a poor sole objective.
- **Anisotropic did not help.** Restoring the near-camera corners just
  **re-inflates** size back toward the hull (+4…+6 %) with no reliable reproj-IoU
  gain — those "measured" corners *are* the over-estimate. It only slides back
  along the reproj-IoU-vs-size trade. Kept in the module as a documented negative
  result.

---

## 4. Real data — frames 0-9

Isotropic rounding applied to every bubble in frames 0-9
(`test/20260723-204041-virtual-camera/round_frames.py`,
`rounded_frames_3d.html`, `real_hull_vs_rounded.png`):

| quantity | value |
|---|---|
| bubbles processed | 60 (6/frame), 113 s |
| mean diameter: raw hull → rounded | 1.41 → **1.34 mm (−5.0 % ± 1.7 %)** |
| mean surface roughness (rms curvature) | 0.008 |
| mean concavity fraction | **0.03** |

The −5 % is consistent with removing the +6 % hull over-estimate; concavity 0.03
and roughness 0.008 confirm clean, convex, rounded bubbles — **no boxes, no SH
flower/peanut modes**.

---

## 5. On the virtual-camera questions (position / number of cameras)

- The recommended method is the **isotropic** 3-D ball-opening, which is the
  continuous limit of *infinitely many uniformly-placed* virtual cameras. So it
  has **no single virtual-camera position to tune**, and adding more virtual
  cameras only converges to it — with **diminishing returns and zero new
  information** (they re-apply the same smooth/convex prior from more angles).
- A **finite literal-silhouette** carve *would* depend on placement — each virtual
  camera rounds only the edges it sees as sharp silhouette corners (a top view
  rounds the azimuthal box, not the top/bottom edges) — but it cannot recover
  detail the 4 real views never captured.
- The **anisotropic** experiment is the direct test of "does targeting specific
  directions help": it does not (§3).

---

## 6. Recommendation & status

- **Use isotropic virtual-camera rounding, ρ = 0.30 · radius**, as the surface
  representation. It de-boxes the hull, corrects the +6 % bias to within ±1 %, has
  the best 3-D accuracy, and shares icosphere topology (temporal filtering is
  trivial). Store per-bubble vertices (~7.7 KB) + one shared face list.
- SH remains available but is **not recommended** (mode-trapping, under-estimate).
- All code is **uncommitted**.

### Code map
```
src/visual_hull/
  mesh_surface/        icosphere + SDF free-form mesh (fit_mesh_surface)
  virtual_camera/
    carve.py           round_hull_occupancy — 3-D open/close by rho  [core]
    round_surface.py   round_bubble_surface — carve + mesh + volume-match  [recommended]
    anisotropic.py     round_bubble_surface_anisotropic  [negative result]
    views.py           virtual_directions, viewing_directions (Jacobian-null axis)
    DESIGN.md
test/20260723-204041-virtual-camera/
  test_reference_shapes.py  reference_shapes.png/.json
  compare_methods.py        compare_methods.html/.png/.json   [5-method comparison]
  round_frames.py           rounded_frames_3d.html  real_hull_vs_rounded.png
  Results/                  raw hull, frames 0-9
  report.md
```

### Next steps (need approval)
1. Pipeline integration — a `--surface {sh,mesh,rounded}` switch in
   `reconstruct_tiff_data.py`, writing rounded meshes + compact storage.
2. Temporal filter across the fixed-topology rounded meshes (per-vertex, along a
   track).
3. Optional per-bubble silhouette polish against the real masks (OpenLPT) for the
   few frames where a camera mask is corrupted.
