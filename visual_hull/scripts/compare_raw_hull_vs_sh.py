"""Raw visual hull vs SH-recommended fit vs ground truth, on synthetic shapes.

Answers two questions with ground-truth numbers:
  1. How does the SH-recommended fit (--sh-inscribed --sh-convexity-weight 0.5
     --sh-spectral-weight 1.0) compare with the *raw visual hull* itself?
  2. Does finer surface refinement (more re-carving levels) help?

Uses the exact hull membership test (a point is in the visual hull iff it
projects inside the mask in ALL cameras), so the "raw hull" numbers are the
true 4-view carving, free of any voxelization choice.

Output: visual_hull/test/<ts>-raw-vs-sh/report.md, results.json, compare.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"
for p in (str(SRC_ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from visual_hull import synthetic_shapes as ss
from visual_hull.camera import OpenLPTCameraSet
from visual_hull.hull import create_visual_hull
from visual_hull.io import load_tiff_mask
from visual_hull.refinement import find_surface_components, refine_surface_points
from visual_hull.test_runs import create_test_run
from visual_hull.improved.spherical_harmonics.surface import (
    SphericalHarmonicFitConfig, fit_spherical_harmonic_surface,
    _basis_terms, _design_matrix, _cartesian_to_spherical,
)

WORKING_DIR = Path(r"X:\Shijie Zhong\Bubble Shear Project\Processed\20260710\20Hz_r_b_1_lpt")
CAMERA_DIR = WORKING_DIR / "camFile_VSC"
NUM_CAMERAS = 4
COARSE = 0.5
RECO = dict(max_degree=4, regularization=1e-2, inscribed=True,
            spectral_weight=1.0, convexity_weight=0.5,
            theta_samples=32, phi_samples=64)


def hull_inside(points, masks, cameras):
    """Exact 4-view visual-hull membership: inside every camera's mask."""
    inside = np.ones(points.shape[0], dtype=bool)
    for c in range(cameras.count):
        proj = cameras.project_points(c, points)
        px = proj.pixels
        h, w = masks[c].shape
        cols = np.round(px[:, 0]).astype(np.int64)
        rows = np.round(px[:, 1]).astype(np.int64)
        ok = proj.valid & (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        inh = np.zeros(points.shape[0], dtype=bool)
        inh[ok] = masks[c][rows[ok], cols[ok]]
        inside &= inh
    return inside


def _grid(shape, cameras, masks, n=64, pad=0.4):
    surf = shape.surface_points(3000)
    lo, hi = surf.min(0) - pad, surf.max(0) + pad
    axes = [np.linspace(lo[i], hi[i], n) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    box_vol = float(np.prod(hi - lo))
    return grid, box_vol


def raw_hull_metrics(shape, cameras, masks, n=64):
    grid, box_vol = _grid(shape, cameras, masks, n)
    in_gt = ss.radial_inside(grid, shape.center, shape.radial_fn)
    in_hull = hull_inside(grid, masks, cameras)
    inter = int(np.count_nonzero(in_gt & in_hull))
    union = int(np.count_nonzero(in_gt | in_hull))
    hull_vol = box_vol * float(np.mean(in_hull))
    hull_d = 2.0 * (3.0 * hull_vol / (4.0 * np.pi)) ** (1.0 / 3.0)
    return {
        "iou_3d": 0.0 if union == 0 else inter / union,
        "diameter_bias_pct": 100.0 * (hull_d - shape.true_diameter) / shape.true_diameter,
        "volume_bias_pct": 100.0 * (hull_vol - shape.true_volume) / shape.true_volume,
    }


def reconstruct_points(masks, cameras, center, extent, levels=1, res_inc=3):
    margin = 0.8 * extent + 2.0 * COARSE
    limits = np.array([center[0]-extent-margin, center[0]+extent+margin,
                       center[1]-extent-margin, center[1]+extent+margin,
                       center[2]-extent-margin, center[2]+extent+margin])
    voxel = np.full(3, COARSE)
    hull = create_visual_hull(masks, cameras, voxel, limits)
    if int(np.sum(hull.voxel_volume)) <= 0:
        return None
    comps = find_surface_components(hull.voxel_volume, hull.grid_x, hull.grid_y, hull.grid_z)
    if not comps:
        return None
    pts = max(comps, key=lambda c: c.shape[0])
    size = voxel
    for lvl in range(levels):
        pts = refine_surface_points(pts, coarse_voxel_size=size, masks=masks,
                                    cameras=cameras, mv=(2 if lvl == 0 else 1), res_inc=res_inc)
        size = size / float(res_inc)
        if pts.shape[0] < 8:
            return None
    return pts


def sh_metrics(shape, pts, cfg):
    sh = fit_spherical_harmonic_surface(pts, config=cfg)
    terms = _basis_terms(cfg.max_degree)
    min_r = float(cfg.minimum_radius)
    # volume + IoU3D
    dirs = ss.fibonacci_directions(20000)
    th, ph = ss.directions_to_angles(dirs)
    r = np.maximum(_design_matrix(th, ph, terms) @ sh.coefficients, min_r)
    vol = float((4*np.pi/3) * np.mean(r**3))
    d = 2.0 * (3.0*vol/(4*np.pi))**(1/3)
    # IoU on a grid
    verts = sh.vertices
    lo = np.minimum(shape.surface_points(2000).min(0), verts.min(0)) - 0.1
    hi = np.maximum(shape.surface_points(2000).max(0), verts.max(0)) + 0.1
    ax = [np.linspace(lo[i], hi[i], 60) for i in range(3)]
    grid = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    in_gt = ss.radial_inside(grid, shape.center, shape.radial_fn)
    _, gth, gph = _cartesian_to_spherical(grid, sh.center)
    rr = np.maximum(_design_matrix(gth, gph, terms) @ sh.coefficients, min_r)
    in_re = np.linalg.norm(grid - sh.center, axis=1) <= rr
    inter = int(np.count_nonzero(in_gt & in_re)); union = int(np.count_nonzero(in_gt | in_re))
    return {"diameter_bias_pct": 100.0*(d-shape.true_diameter)/shape.true_diameter,
            "iou_3d": 0.0 if union == 0 else inter/union,
            "n_points": int(pts.shape[0])}


def main():
    run = create_test_run(PROJECT_ROOT, "raw-vs-sh")
    print(f"Output -> {run.root}")
    cams = OpenLPTCameraSet.from_camera_files([CAMERA_DIR / f"vsc_cam{i}.txt" for i in range(NUM_CAMERAS)])
    real = load_tiff_mask(WORKING_DIR / "imgFile_bb" / "cam0" / "img000000.tif")
    hw = (real.shape[0], real.shape[1])

    ctr = np.array([-20.0, -8.0, 0.0])
    def rz(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
    def ry(a): c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    R = rz(0.6) @ ry(0.4)
    shapes = [
        ss.sphere(ctr, 0.75), ss.sphere(ctr, 1.25),
        ss.ellipsoid(ctr, 1.6, 1.0, 0.9, R), ss.ellipsoid(ctr, 2.0, 1.1, 0.8, R),
        ss.convex_sh(ctr, 1.2, {(2,0):0.12,(2,2):0.08}),
        ss.convex_sh(ctr, 1.0, {(2,-1):0.10,(2,1):0.10,(4,0):0.05}),
    ]
    cfg = SphericalHarmonicFitConfig(**RECO)

    rows = []
    print("\n== raw hull vs SH-recommended ==")
    for shp in shapes:
        masks = ss.render_masks_through_cameras(shp, cams, hw)
        if masks is None:
            continue
        raw = raw_hull_metrics(shp, cams, masks)
        pts = reconstruct_points(masks, cams, shp.center, float(np.max(shp.true_axes)))
        if pts is None:
            continue
        shm = sh_metrics(shp, pts, cfg)
        rows.append({"shape": shp.name, "D_true": shp.true_diameter,
                     "raw": raw, "sh": shm})
        print(f"  {shp.name:10s} D={shp.true_diameter:.2f}  RAW: IoU={raw['iou_3d']:.3f} "
              f"Dbias={raw['diameter_bias_pct']:+5.1f}% Vbias={raw['volume_bias_pct']:+5.1f}%   "
              f"SH: IoU={shm['iou_3d']:.3f} Dbias={shm['diameter_bias_pct']:+5.1f}%")

    # Refinement test on the small + a mid sphere.
    print("\n== refinement-level test (SH-recommended) ==")
    refine_rows = []
    for shp in (shapes[0], shapes[2]):
        masks = ss.render_masks_through_cameras(shp, cams, hw)
        if masks is None:
            continue
        for label, levels, res in [("coarse/3 (1 lvl)", 1, 3),
                                   ("coarse/9 (2 lvl)", 2, 3),
                                   ("coarse/15 (2 lvl x5)", 2, 5)]:
            pts = reconstruct_points(masks, cams, shp.center, float(np.max(shp.true_axes)),
                                     levels=levels, res_inc=res)
            if pts is None:
                continue
            m = sh_metrics(shp, pts, cfg)
            refine_rows.append({"shape": shp.name, "D_true": shp.true_diameter,
                                "refine": label, **m})
            print(f"  {shp.name:10s} {label:20s} pts={m['n_points']:5d}  "
                  f"IoU={m['iou_3d']:.3f}  Dbias={m['diameter_bias_pct']:+5.1f}%")

    run.write_json("results.json", {"compare": rows, "refine": refine_rows})
    _plot(rows, run)
    _report(rows, refine_rows, run)
    print(f"\nDone. Report: {run.path('report.md')}")


def _plot(rows, run):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"{r['shape']}\nD={r['D_true']:.1f}" for r in rows]
    x = np.arange(len(rows)); w = 0.35
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.bar(x-w/2, [r['raw']['iou_3d'] for r in rows], w, label="raw visual hull", color="#8064a2")
    a1.bar(x+w/2, [r['sh']['iou_3d'] for r in rows], w, label="SH recommended", color="#4bacc6")
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=8); a1.set_ylabel("3D IoU vs truth")
    a1.set_title("Shape accuracy: raw hull vs SH-recommended"); a1.legend(); a1.grid(True, alpha=0.3)
    a2.bar(x-w/2, [r['raw']['diameter_bias_pct'] for r in rows], w, label="raw visual hull", color="#8064a2")
    a2.bar(x+w/2, [r['sh']['diameter_bias_pct'] for r in rows], w, label="SH recommended", color="#4bacc6")
    a2.axhline(0, color="k", lw=0.8, ls="--")
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=8); a2.set_ylabel("diameter bias (%)")
    a2.set_title("Size bias: raw hull vs SH-recommended"); a2.legend(); a2.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(run.path("compare.png"), dpi=150); plt.close(fig)


def _report(rows, refine_rows, run):
    L = ["# Raw visual hull vs SH-recommended (synthetic ground truth)", "",
         "SH-recommended = `--sh-inscribed --sh-convexity-weight 0.5 --sh-spectral-weight 1.0`, degree 4.",
         "Raw hull = exact 4-view carving (point in all camera masks).", "",
         "| shape | D_true | RAW IoU | RAW Dbias% | RAW Vbias% | SH IoU | SH Dbias% |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['shape']} | {r['D_true']:.2f} | {r['raw']['iou_3d']:.3f} | "
                 f"{r['raw']['diameter_bias_pct']:+.1f} | {r['raw']['volume_bias_pct']:+.1f} | "
                 f"{r['sh']['iou_3d']:.3f} | {r['sh']['diameter_bias_pct']:+.1f} |")
    if rows:
        L += ["", "**Means:** "
              f"raw IoU {np.mean([r['raw']['iou_3d'] for r in rows]):.3f} / "
              f"SH IoU {np.mean([r['sh']['iou_3d'] for r in rows]):.3f}; "
              f"raw |Dbias| {np.mean([abs(r['raw']['diameter_bias_pct']) for r in rows]):.1f}% / "
              f"SH |Dbias| {np.mean([abs(r['sh']['diameter_bias_pct']) for r in rows]):.1f}%."]
    L += ["", "## Refinement-level test (SH-recommended)", "",
          "Does finer re-carving help? Coarse hull is 0.5 mm; each level re-votes finer.", "",
          "| shape | D_true | refine | surf pts | IoU | Dbias% |", "|---|---|---|---|---|---|"]
    for r in refine_rows:
        L.append(f"| {r['shape']} | {r['D_true']:.2f} | {r['refine']} | {r['n_points']} | "
                 f"{r['iou_3d']:.3f} | {r['diameter_bias_pct']:+.1f} |")
    L += ["", "See `compare.png`."]
    run.write_text("report.md", "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
