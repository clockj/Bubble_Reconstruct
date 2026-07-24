# `mesh_surface` — free-form deformable-mesh bubble surfaces

A replacement for the spherical-harmonic surface representation, motivated by
the observation that SH `r(θ,φ)` is a **global, star-convex, low-degree basis**
that gets trapped in "mode" shapes (flowers, peanuts, lumps) because the basis
cannot locally conform to the data. A triangle mesh with free vertex positions,
optimized under local energies, has no such basis limitation.

## Why a mesh beats spherical harmonics here

| issue | spherical harmonics | free-form mesh |
|---|---|---|
| shape space | star-convex `r(θ,φ)`, one radius/ray | any genus-0 surface |
| truncation | degree ≤4 → forced smooth modes ("flower"/"peanut") | resolution set by #vertices, local |
| smoothing | global spectral penalty (affects whole shape) | **local** Laplacian (smooth only where needed) |
| storage | (L+1)² coeffs, but needs voxels for detail | vertices only (~7 KB @642 verts), sub-voxel |
| temporal | average coeffs (modes may not correspond) | fixed topology → **per-vertex** correspondence, trivial temporal filter |
| accuracy | limited by degree | limited by #vertices + SDF interpolation (decoupled from voxel grid) |

The user's requested criteria all map to mesh energies: **smoothness (curvature)**,
**convexity**, **camera-projection consistency**, **temporal filtering**.

## Representation

- **Fixed-topology icosphere** (subdivided icosahedron, projected to the sphere).
  Subdivision level L → 20·4^L faces, 10·4^L+2 vertices (L3 = 1280 F / 642 V,
  L4 = 5120 / 2562). Genus-0 (correct for single bubbles), uniform triangles,
  **identical topology across all bubbles and frames** → free temporal
  correspondence. Faces are shared/constant; only vertices are stored.
- `MeshSurface`: `vertices (N,3)`, `faces (M,3)`, `center`, plus fit diagnostics.

## Data term — the multi-view constraint, dependency-light

The visual hull already IS the intersection of all 4 camera silhouette cones, so
it encodes the full multi-view constraint. We turn it into a **signed distance
field (SDF)** once per bubble:

1. Rasterize the bubble's (refined) voxels into a local occupancy grid.
2. `sdf = EDT(outside) − EDT(inside)` (scipy `distance_transform_edt`), sub-voxel
   via trilinear interpolation; analytic gradient = normalized finite differences.
3. **Data energy** pulls each vertex to the boundary from the inside:
   `E_data = Σ ρ(sdf(vᵢ))`, with `ρ` a one-sided penalty (hard outside `sdf>0`,
   soft far-inside) so the mesh shrink-wraps the hull surface without bulging out.
   The SDF gradient gives the exact move direction — clean gradient descent, no
   per-vertex reprojection in the inner loop.
4. **Bias handling (bonus):** target the isosurface `sdf = +δ` (δ from the
   synthetic-sphere calibration, ~ the +6% radius bias) to get a *debiased*
   surface directly. Optional.

Optional **silhouette polish** (uses the real cameras/masks via OpenLPT): after
SDF fit, a few passes nudging only silhouette-boundary vertices to maximize IoU
vs each camera's re-projected hull silhouette (reuse existing machinery, finite-
difference gradient on boundary vertices only). This adds true camera-projection
comparison at sub-voxel accuracy without a differentiable renderer.

## Regularization energies

- **Smoothness (curvature):** `E_smooth = Σ ‖L vᵢ‖²`, L = uniform (umbrella) or
  cotangent Laplacian. Penalizes roughness **locally** (the SH-mode fix). Gradient
  = `2 LᵀL v`. Cotangent Laplacian ≈ mean-curvature flow.
- **Convexity:** `E_convex = Σ max(−κ_meanᵢ, 0)²` (penalize concave vertices) or a
  soft pull of concave vertices toward the mesh's convex hull. Tunable weight;
  bubbles are near-convex at low Weber number.
- **Temporal:** with fixed topology, after rigid (Procrustes) alignment of a
  track's consecutive meshes, `E_temporal = Σᵢ ‖vᵢ(t) − v̄ᵢ‖²` (1st-order) or a
  discrete acceleration penalty (2nd-order). Trivially well-defined because vertex
  i means the same surface point every frame — unlike SH coefficient averaging.

## Optimization

Gradient-based evolution of all vertex positions (3N params):
- Analytic gradients for data(SDF), smoothness, temporal; finite-diff/soft for the
  optional silhouette polish.
- **Weight annealing** (strong smoothness → weak) and **coarse-to-fine**
  (optimize at L2, subdivide to L3/L4, continue) — the mesh analog of "progressive
  fitting", which avoids the local-minima / mode-trapping that hurt SH.
- Explicit gradient descent with backtracking line search (numpy) or scipy
  L-BFGS-B. Dependency-light; **no PyTorch / differentiable renderer required**
  (OpenLPT's refractive projection isn't autodiff-friendly, so the SDF surrogate
  is the right call). PyTorch acceleration is a possible future swap.

## Storage

Per bubble/frame: `vertices` float32 (N,3) + shared faces once per run. L3 ≈ 7.7 KB
vs thousands of voxels. Sub-voxel accuracy from the SDF. Compact and mesh-ready.

## Module layout (`src/visual_hull/mesh_surface/`)

| file | contents | status |
|---|---|---|
| `icosphere.py` | subdivided icosphere generation, edges | implement now |
| `mesh_ops.py` | adjacency, uniform/cotangent Laplacian, vertex normals, mean curvature | implement now |
| `sdf.py` | build hull SDF from voxels, trilinear sample + gradient | implement now |
| `energy.py` | data/smoothness/convexity energies + gradients | implement now |
| `optimize.py` | mesh evolution (anneal + coarse-to-fine) | core now |
| `fit.py` | `MeshSurfaceConfig`, `MeshSurface`, `fit_mesh_surface(...)` | core now |
| `silhouette.py` | optional camera-projection polish (OpenLPT) | after approval |
| `temporal.py` | fixed-topology temporal mesh filter | after approval |
| `__init__.py` | public API | now |

## Validation plan (test/<ts>-mesh-surface/)

1. **Synthetic** (reuse `synthetic_shapes`): sphere/ellipsoid/convex — compare
   mesh vs SH-recommended on 3D IoU, diameter bias, and a "lumpiness" metric.
   Success: mesh ≥ SH IoU with lower roughness, no mode artifacts.
2. **Real lumpy bubble** (SH bubble 5 @ frame 1, the reported case): show the mesh
   yields a clean smooth bubble where SH was lumpy, at comparable/better silhouette
   IoU.
3. **Temporal** (after approval): a tracked bubble over frames — show per-vertex
   temporal filtering removes jitter while preserving real deformation.

## Risks / honest caveats

- **Genus-0 only** (single simply-connected bubble). Merging/splitting bubbles need
  topology change — out of scope initially (the separation step handles splitting
  upstream).
- **Garbage-in:** if the hull/masks are noisy, the SDF is noisy; the smoothness +
  temporal priors suppress this far better than SH, but can't invent data.
- **Self-intersection** during aggressive evolution — mitigated by smoothness,
  modest steps, and coarse-to-fine; optional intersection check.
- Not a magic accuracy fix for the hull's +6%/+19% bias — but it can target a
  debiased isosurface (δ offset), which SH could not do cleanly.
