# Virtual-camera silhouette rounding

## The problem it solves

With only **4 refractive cameras**, the visual hull is the intersection of 4
silhouette cones.  Each cone contributes a handful of flat tangent planes, so
between the 4 real viewing directions the hull bulges out to those planes: the
reconstructed surface is a **faceted convex polyhedron** ("boxy"), and it
systematically **over-estimates** size (~+6 % diameter / +19 % volume on the
synthetic-sphere study).  Fitting any surface — SH or a free-form mesh — to that
hull faithfully reproduces the box, because the hull genuinely *is* boxy where
there are no views.

A real bubble is smooth and (nearly) convex.  The facets are an artifact of the
missing views, not a measured feature.

## The idea

The classic **virtual-camera** trick adds extra viewpoints the rig doesn't have.
You have no image from a virtual camera, so you *synthesize* its silhouette by
projecting the current shape and then **rounding that silhouette** (assuming the
outline from every direction is smooth), and re-carve with the rounded
silhouette.  Iterating over many virtual directions shaves the sharp
intersection edges and relaxes the faceted polyhedron into a smooth rounded body.

A virtual camera adds **no new measurement** — it injects the *prior* used to
round its silhouette.  For bubbles ("smooth, convex") that prior is essentially
correct, so it works; the roundness between the 4 real views is *assumed*, not
measured.  This is the honest best-case for a 4-view rig.

## How this module does it (mesh-free, OpenLPT-free carving)

Virtual cameras are **synthetic orthographic** viewpoints — appropriate because
we are imposing a prior, not reading pixels, so refractive projection is
irrelevant here (OpenLPT stays the authority for the *real* cameras that produced
the input hull).

1. Rasterize the bubble's real-hull voxels into a fine local occupancy grid
   (`spacing` ≪ reconstruction voxel size, e.g. voxel/3), close + fill solid.
2. Generate `n_views` ~uniform directions on the sphere (Fibonacci).
3. For each direction `d`:
   - project the occupied voxel centres onto the plane ⟂ `d` → a 2D silhouette,
   - **morphological close then open** with a disk of radius `rho` — closing
     rounds concave 2D corners, opening rounds convex 2D corners *sharper than
     `rho`* while leaving smooth boundary (curvature radius > `rho`) untouched,
   - a voxel **survives** this view iff its projection lands inside the rounded
     silhouette.
4. Keep voxels surviving **all** views (silhouette-cone intersection). Repeat for
   a few `iters` so freshly exposed corners round too.
5. Feed the rounded occupancy to `mesh_surface.fit_mesh_surface` → a clean
   fixed-topology icosphere mesh (sub-voxel, temporally comparable).

`rho` is the **only** shape knob: it is the smallest surface-feature radius kept.
Because opening only *removes* material, rounding also pulls the hull's +6 %/+19 %
over-estimate downward — so `rho` doubles as a bias corrector.  It is calibrated
once on a synthetic sphere (minimise diameter error) and reused.

## Why not just Laplacian-smooth the mesh?

Umbrella/Laplacian smoothing rounds *and* shrinks everywhere, losing size where
the real cameras actually constrain it.  Virtual-camera carving is
**silhouette-consistent**: it only removes the corner material that no real
silhouette requires, staying tangent to the 4 real silhouettes on their smooth
edges.  Same rounding, without the uniform shrink.

## Module layout

- `views.py`  — virtual directions + per-direction orthonormal projection basis.
- `carve.py`  — `round_hull_occupancy(...)`: the rounding carve (the core).
- `round_surface.py` — `round_bubble_surface(...)`: carve → `fit_mesh_surface`,
  returns a `RoundedBubble` with metrics.
- `__init__.py` — public API.

## Validation

- **Reference shapes** (`test/*-virtual-camera/test_reference_shapes.py`): sphere
  / ellipsoid / convex, rendered through the *real* 4 cameras → carve real hull →
  round → compare hull vs rounded vs **ground truth** (3D IoU, D error, roughness,
  reprojection IoU vs the real masks).  Answers "does it help with 4 views".
- **Real frames 0–9**: round every bubble, report roughness/concavity/size change
  vs the raw hull, and interactive 3D visualisation.
