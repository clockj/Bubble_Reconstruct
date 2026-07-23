"""Synthetic-shape benchmark for the SH reconstruction improvements.

Renders known spheres / ellipsoids / convex-SH shapes through the real
4-camera refractive model, reconstructs them with the production pipeline
(visual hull -> surface components -> refine), then fits spherical harmonics
under several configurations and measures recovery against ground truth.

Phases exercised:
  1  spectral penalty + ellipsoid initialization      (surface.py)
  3  curvature (bending) + convexity priors            (surface.py)
  2  mask-quality view-weighting on corrupted masks    (view_quality.py)

Outputs (under visual_hull/test/<ts>-synthetic-shape-bench/):
    report.md, results.json, recovery.png, flower_stress.png

Run:
    python scripts/validate_synthetic_shapes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_hull import synthetic_shapes as ss
from visual_hull.camera import OpenLPTCameraSet
from visual_hull.hull import create_visual_hull
from visual_hull.io import load_tiff_mask
from visual_hull.refinement import find_surface_components, refine_surface_points
from visual_hull.test_runs import create_test_run
from visual_hull.improved.spherical_harmonics.surface import (
    SphericalHarmonicFitConfig,
    fit_spherical_harmonic_surface,
    surface_curvature_stats,
    _basis_terms,
    _design_matrix,
    _cartesian_to_spherical,
    _grid_vertices_faces,
)
from visual_hull.improved.view_quality import analyze_bubble_views, camera_weights

WORKING_DIR = Path(
    r"X:\Shijie Zhong\Bubble Shear Project\Processed\20260710\20Hz_r_b_1_lpt"
)
CAMERA_DIR = WORKING_DIR / "camFile_VSC"
NUM_CAMERAS = 4
COARSE = 0.5  # mm


# ── Reconstruction front-end (shared across SH variants) ──────────────────────
def reconstruct_surface_points(masks, cameras, center, extent):
    """Coarse hull -> largest surface component -> refined points (coarse/3)."""
    margin = 0.8 * extent + 2.0 * COARSE
    limits = np.array([
        center[0] - extent - margin, center[0] + extent + margin,
        center[1] - extent - margin, center[1] + extent + margin,
        center[2] - extent - margin, center[2] + extent + margin,
    ])
    voxel_size = np.full(3, COARSE)
    hull = create_visual_hull(masks, cameras, voxel_size, limits)
    if int(np.sum(hull.voxel_volume)) <= 0:
        return None
    components = find_surface_components(
        hull.voxel_volume, hull.grid_x, hull.grid_y, hull.grid_z)
    if not components:
        return None
    surface_points = max(components, key=lambda c: c.shape[0])
    refined = refine_surface_points(
        surface_points, coarse_voxel_size=voxel_size,
        masks=masks, cameras=cameras, mv=2, res_inc=3)
    if refined.shape[0] < 8:
        return None
    return refined


# ── Metrics ───────────────────────────────────────────────────────────────────
def _radial_inside_coeffs(points, center, coeffs, terms, min_r):
    _, theta, phi = _cartesian_to_spherical(points, center)
    r_surface = np.maximum(_design_matrix(theta, phi, terms) @ coeffs, min_r)
    dist = np.linalg.norm(np.asarray(points) - np.asarray(center), axis=1)
    return dist <= r_surface


def iou_3d(shape, sh_center, coeffs, terms, min_r, n=60):
    """Voxel-grid 3D IoU between GT radial shape and reconstructed SH surface."""
    pts_all = np.vstack([shape.surface_points(3000),
                         _grid_vertices_faces(sh_center, coeffs, terms,
                                              SphericalHarmonicFitConfig())[0]])
    lo = pts_all.min(axis=0) - 0.1
    hi = pts_all.max(axis=0) + 0.1
    gx, gy, gz = (np.linspace(lo[i], hi[i], n) for i in range(3))
    grid = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
    in_gt = ss.radial_inside(grid, shape.center, shape.radial_fn)
    in_re = _radial_inside_coeffs(grid, sh_center, coeffs, terms, min_r)
    union = int(np.count_nonzero(in_gt | in_re))
    return 0.0 if union == 0 else int(np.count_nonzero(in_gt & in_re)) / union


def sh_volume(sh_center, coeffs, terms, min_r, n=20000):
    dirs = ss.fibonacci_directions(n)
    theta, phi = ss.directions_to_angles(dirs)
    r = np.maximum(_design_matrix(theta, phi, terms) @ coeffs, min_r)
    return float((4.0 * np.pi / 3.0) * np.mean(r ** 3))


def sh_aspect_ratio(sh_center, coeffs, terms, min_r, n=8000):
    dirs = ss.fibonacci_directions(n)
    theta, phi = ss.directions_to_angles(dirs)
    r = np.maximum(_design_matrix(theta, phi, terms) @ coeffs, min_r)
    pts = dirs * r[:, None]
    eig = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(pts.T)), 0.0))
    eig = np.sort(eig)[::-1]
    return float(eig[0] / eig[-1]) if eig[-1] > 0 else float("nan")


def evaluate(shape, sh, terms, min_r):
    coeffs, c = sh.coefficients, sh.center
    recon_vol = sh_volume(c, coeffs, terms, min_r)
    recon_d = 2.0 * (3.0 * recon_vol / (4.0 * np.pi)) ** (1.0 / 3.0)
    stats = surface_curvature_stats(c, coeffs, terms)
    return {
        "volume_bias_pct": 100.0 * (recon_vol - shape.true_volume) / shape.true_volume,
        "diameter_bias_pct": 100.0 * (recon_d - shape.true_diameter) / shape.true_diameter,
        "iou_3d": iou_3d(shape, c, coeffs, terms, min_r),
        "aspect_recon": sh_aspect_ratio(c, coeffs, terms, min_r),
        "aspect_true": shape.true_aspect_ratio,
        "bending_energy": stats["bending_energy"],
        "concavity_fraction": stats["concavity_fraction"],
    }


# ── SH configuration variants (Phases 1 & 3) ──────────────────────────────────
def variants(degree=4):
    base = dict(max_degree=degree, regularization=1e-2, inscribed=True,
                theta_samples=32, phi_samples=64)
    return {
        "baseline (inscribed)": SphericalHarmonicFitConfig(**base),
        "+spectral": SphericalHarmonicFitConfig(**base, spectral_weight=1.0),
        "+curvature": SphericalHarmonicFitConfig(**base, curvature_weight=1e-3),
        "+convexity": SphericalHarmonicFitConfig(**base, convexity_weight=0.5),
        "+all priors": SphericalHarmonicFitConfig(
            **base, spectral_weight=1.0, curvature_weight=1e-3, convexity_weight=0.5),
    }


# ── Phase 2: view-weighting on corrupted masks ────────────────────────────────
def weighted_silhouette_iou(coeffs, center, terms, cameras, targets, weights,
                            ds_shape, scale):
    from visual_hull.improved.view_quality import _camera_fill
    verts, _ = _grid_vertices_faces(center, coeffs, terms,
                                    SphericalHarmonicFitConfig(theta_samples=24, phi_samples=48))
    ious = []
    for c in range(cameras.count):
        pred = _camera_fill(verts, cameras, c, ds_shape, scale)
        inter = int(np.count_nonzero(pred & targets[c]))
        union = int(np.count_nonzero(pred | targets[c]))
        ious.append(1.0 if union == 0 else inter / union)
    return float(np.average(ious, weights=weights))


def optimize_weighted(coeffs, center, terms, cameras, targets, weights,
                      ds_shape, scale, passes=5):
    best = np.asarray(coeffs, dtype=np.float64).copy()
    best_iou = weighted_silhouette_iou(best, center, terms, cameras, targets,
                                       weights, ds_shape, scale)
    step = 0.12 * abs(best[0]) if best[0] != 0 else 0.1
    for _ in range(passes):
        improved = False
        for i in range(best.shape[0]):
            for delta in (step, -step):
                cand = best.copy()
                cand[i] += delta
                iou = weighted_silhouette_iou(cand, center, terms, cameras, targets,
                                              weights, ds_shape, scale)
                if iou > best_iou + 1e-4:
                    best, best_iou, improved = cand, iou, True
        if not improved:
            step *= 0.5
            if step < 1e-4 * (abs(best[0]) + 1e-9):
                break
    return best, best_iou


def run_view_weighting(shape, cameras, hw, run):
    """Corrupt one camera's mask; compare equal vs quality-weighted fitting."""
    from visual_hull.improved.view_quality import _camera_fill

    clean = ss.render_masks_through_cameras(shape, cameras, hw)
    if clean is None:
        return None
    rng = np.random.default_rng(1)
    corrupt = [m.copy() for m in clean]
    corrupt[0] = ss.corrupt_mask(clean[0], hole_frac=0.35, rng=rng)  # cam0 damaged

    surface = reconstruct_surface_points(corrupt, cameras, shape.center,
                                         float(np.max(shape.true_axes)))
    if surface is None:
        return None
    cfg = SphericalHarmonicFitConfig(max_degree=4, regularization=1e-2, inscribed=True)
    sh = fit_spherical_harmonic_surface(surface, config=cfg)
    terms = _basis_terms(4)
    min_r = float(cfg.minimum_radius)

    analysis = analyze_bubble_views(surface, None, corrupt, cameras, scale=4)
    w_quality = camera_weights(analysis)
    w_equal = np.full(cameras.count, 1.0 / cameras.count)

    # Targets = re-projected hull silhouette of THIS bubble (overlap-free).
    scale = 4
    ds_shape = (hw[0] // scale, hw[1] // scale)
    targets = [_camera_fill(surface, cameras, c, ds_shape, scale) for c in range(cameras.count)]

    c_eq, _ = optimize_weighted(sh.coefficients, sh.center, terms, cameras,
                                targets, w_equal, ds_shape, scale)
    c_wq, _ = optimize_weighted(sh.coefficients, sh.center, terms, cameras,
                                targets, w_quality, ds_shape, scale)

    # Ellipsoid-init test: seed the (iterative) optimizer from the PCA ellipsoid
    # instead of the LS coefficients, using quality weights.
    from visual_hull.improved.spherical_harmonics.surface import (
        _ellipsoid_warm_start, _penalty_diagonal)
    pen = _penalty_diagonal(terms, cfg.regularization)
    try:
        ell0 = _ellipsoid_warm_start(surface, sh.center, terms, pen)
        c_ell, _ = optimize_weighted(ell0, sh.center, terms, cameras,
                                     targets, w_quality, ds_shape, scale)
    except Exception:
        c_ell = c_wq

    def metrics(coeffs):
        vol = sh_volume(sh.center, coeffs, terms, min_r)
        d = 2.0 * (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
        return {
            "diameter_bias_pct": 100.0 * (d - shape.true_diameter) / shape.true_diameter,
            "iou_3d": iou_3d(shape, sh.center, coeffs, terms, min_r),
        }

    return {
        "shape": shape.name,
        "corrupted_camera": 0,
        "view_weights_quality": w_quality.tolist(),
        "analysis": analysis,
        "equal_weight": metrics(c_eq),
        "quality_weight": metrics(c_wq),
        "quality_weight_ellipsoid_init": metrics(c_ell),
        "no_opt": metrics(sh.coefficients),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def build_shapes():
    ctr = np.array([-20.0, -8.0, 0.0])
    R = _rot_z(0.6) @ _rot_y(0.4)
    return [
        ss.sphere(ctr, 0.75),
        ss.sphere(ctr, 1.25),
        ss.ellipsoid(ctr, 1.6, 1.0, 0.9, R),        # AR ~1.8
        ss.ellipsoid(ctr, 2.0, 1.1, 0.8, R),        # AR ~2.5
        ss.convex_sh(ctr, 1.2, {(2, 0): 0.12, (2, 2): 0.08}),
        ss.convex_sh(ctr, 1.0, {(2, -1): 0.10, (2, 1): 0.10, (4, 0): 0.05}),
    ]


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def run_flower_stress(shape, cameras, hw):
    """Sparse high-degree fit: does spectral/curvature suppress flowers?"""
    masks = ss.render_masks_through_cameras(shape, cameras, hw)
    if masks is None:
        return None
    surface = reconstruct_surface_points(masks, cameras, shape.center,
                                         float(np.max(shape.true_axes)))
    if surface is None:
        return None
    # Subsample to stress an l=8 fit (few points per coefficient -> flowers).
    rng = np.random.default_rng(2)
    idx = rng.choice(surface.shape[0], size=min(120, surface.shape[0]), replace=False)
    sparse = surface[idx]
    terms8 = _basis_terms(8)
    min_r = 1e-3
    out = {}
    for name, cfg in {
        "deg8 no-penalty": SphericalHarmonicFitConfig(max_degree=8, regularization=1e-3, inscribed=False),
        "deg8 +spectral": SphericalHarmonicFitConfig(max_degree=8, regularization=1e-3, inscribed=False, spectral_weight=3.0),
        "deg8 +curvature": SphericalHarmonicFitConfig(max_degree=8, regularization=1e-3, inscribed=False, curvature_weight=5e-3),
    }.items():
        sh = fit_spherical_harmonic_surface(sparse, config=cfg)
        stats = surface_curvature_stats(sh.center, sh.coefficients, terms8)
        vol = sh_volume(sh.center, sh.coefficients, terms8, min_r)
        d = 2.0 * (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
        out[name] = {
            "bending_energy": stats["bending_energy"],
            "concavity_fraction": stats["concavity_fraction"],
            "diameter_bias_pct": 100.0 * (d - shape.true_diameter) / shape.true_diameter,
            "iou_3d": iou_3d(shape, sh.center, sh.coefficients, terms8, min_r),
        }
    return out


def main():
    run = create_test_run(PROJECT_ROOT, "synthetic-shape-bench")
    print(f"Output -> {run.root}")

    cams = OpenLPTCameraSet.from_camera_files(
        [CAMERA_DIR / f"vsc_cam{i}.txt" for i in range(NUM_CAMERAS)])
    real = load_tiff_mask(WORKING_DIR / "imgFile_bb" / "cam0" / "img000000.tif")
    hw = (real.shape[0], real.shape[1])
    print(f"Image size (HxW): {hw}")

    terms4 = _basis_terms(4)
    min_r = 1e-3
    shapes = build_shapes()
    variant_cfgs = variants(4)

    recovery = []  # per (shape, variant)
    for shape in shapes:
        masks = ss.render_masks_through_cameras(shape, cams, hw)
        if masks is None:
            print(f"  [{shape.name}] not visible — skipped")
            continue
        surface = reconstruct_surface_points(masks, cams, shape.center,
                                             float(np.max(shape.true_axes)))
        if surface is None:
            print(f"  [{shape.name}] empty hull — skipped")
            continue
        print(f"  [{shape.name}] surface pts={surface.shape[0]}  "
              f"D_true={shape.true_diameter:.3f}mm  AR_true={shape.true_aspect_ratio:.2f}")
        for vname, cfg in variant_cfgs.items():
            sh = fit_spherical_harmonic_surface(surface, config=cfg)
            m = evaluate(shape, sh, terms4, min_r)
            m.update({"shape": shape.name, "variant": vname,
                      "true_diameter": shape.true_diameter})
            recovery.append(m)
            print(f"      {vname:22s} Dbias={m['diameter_bias_pct']:+5.1f}%  "
                  f"IoU3D={m['iou_3d']:.3f}  bend={m['bending_energy']:6.1f}  "
                  f"concav={m['concavity_fraction']:.3f}")

    # Phase 2 — view weighting (use the AR~2.5 ellipsoid: directional info matters)
    print("\n-- Phase 2: mask-quality view weighting (cam0 corrupted) --")
    vw = run_view_weighting(shapes[3], cams, hw, run)
    if vw:
        print(f"  weights(quality)={[f'{w:.2f}' for w in vw['view_weights_quality']]}")
        print(f"  no-opt         Dbias={vw['no_opt']['diameter_bias_pct']:+5.1f}%  IoU3D={vw['no_opt']['iou_3d']:.3f}")
        print(f"  equal-wt       Dbias={vw['equal_weight']['diameter_bias_pct']:+5.1f}%  IoU3D={vw['equal_weight']['iou_3d']:.3f}")
        print(f"  quality-wt     Dbias={vw['quality_weight']['diameter_bias_pct']:+5.1f}%  IoU3D={vw['quality_weight']['iou_3d']:.3f}")
        print(f"  quality+ellip  Dbias={vw['quality_weight_ellipsoid_init']['diameter_bias_pct']:+5.1f}%  IoU3D={vw['quality_weight_ellipsoid_init']['iou_3d']:.3f}")

    # Flower stress (use a convex_sh shape)
    print("\n-- Flower stress (sparse deg-8 fit) --")
    flower = run_flower_stress(shapes[4], cams, hw)
    if flower:
        for k, v in flower.items():
            print(f"  {k:18s} bend={v['bending_energy']:8.1f}  IoU3D={v['iou_3d']:.3f}  Dbias={v['diameter_bias_pct']:+.1f}%")

    results = {"recovery": recovery, "view_weighting": vw, "flower_stress": flower}
    run.write_json("results.json", results)
    _plots(recovery, flower, run)
    _report(recovery, vw, flower, run)
    print(f"\nDone. Report: {run.path('report.md')}")


def _plots(recovery, flower, run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variant_names = list(dict.fromkeys(r["variant"] for r in recovery))
    shapes = list(dict.fromkeys(r["shape"] for r in recovery))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(variant_names))
    for i, shp in enumerate(shapes):
        d = [next(r for r in recovery if r["shape"] == shp and r["variant"] == v)["iou_3d"]
             for v in variant_names]
        ax1.plot(x, d, "o-", label=shp)
    ax1.set_xticks(x); ax1.set_xticklabels(variant_names, rotation=30, ha="right")
    ax1.set_ylabel("3D IoU vs ground truth"); ax1.set_title("Shape recovery by variant")
    ax1.grid(True, alpha=0.3); ax1.legend(fontsize=7)

    for i, shp in enumerate(shapes):
        d = [next(r for r in recovery if r["shape"] == shp and r["variant"] == v)["diameter_bias_pct"]
             for v in variant_names]
        ax2.plot(x, d, "o-", label=shp)
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.set_xticks(x); ax2.set_xticklabels(variant_names, rotation=30, ha="right")
    ax2.set_ylabel("diameter bias (%)"); ax2.set_title("Diameter bias by variant")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(run.path("recovery.png"), dpi=150); plt.close(fig)

    if flower:
        fig, ax = plt.subplots(figsize=(7, 5))
        names = list(flower.keys())
        ax.bar(range(len(names)), [flower[n]["bending_energy"] for n in names], color="#c0504d")
        ax.set_ylabel("bending energy (high-l roughness)")
        ax.set_title("Flower suppression (sparse deg-8 fit)")
        ax.set_yscale("log")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right")
        fig.tight_layout(); fig.savefig(run.path("flower_stress.png"), dpi=150); plt.close(fig)


def _report(recovery, vw, flower, run):
    variant_names = list(dict.fromkeys(r["variant"] for r in recovery))
    shapes = list(dict.fromkeys(r["shape"] for r in recovery))
    L = ["# Synthetic-shape benchmark — SH reconstruction improvements", "",
         "Known spheres / ellipsoids / convex-SH shapes rendered through the real",
         "4-camera PINPLATE model, reconstructed (hull -> components -> refine),",
         "then SH-fitted under each configuration. Metrics vs ground truth.", ""]

    L += ["## Shape recovery (degree 4)", "",
          "3D IoU (higher better) and diameter bias (nearer 0 better) per variant.", ""]
    L += ["| shape | metric | " + " | ".join(variant_names) + " |",
          "|" + "---|" * (len(variant_names) + 2)]
    for shp in shapes:
        row = {r["variant"]: r for r in recovery if r["shape"] == shp}
        L.append(f"| {shp} | IoU3D | " +
                 " | ".join(f"{row[v]['iou_3d']:.3f}" for v in variant_names) + " |")
        L.append(f"| {shp} | Dbias% | " +
                 " | ".join(f"{row[v]['diameter_bias_pct']:+.1f}" for v in variant_names) + " |")

    # Aggregate means.
    L += ["", "## Mean over shapes", "",
          "| variant | mean IoU3D | mean |Dbias|% | mean bending | mean concavity |",
          "|---|---|---|---|---|"]
    for v in variant_names:
        rs = [r for r in recovery if r["variant"] == v]
        L.append(f"| {v} | {np.mean([r['iou_3d'] for r in rs]):.3f} | "
                 f"{np.mean([abs(r['diameter_bias_pct']) for r in rs]):.1f} | "
                 f"{np.mean([r['bending_energy'] for r in rs]):.1f} | "
                 f"{np.mean([r['concavity_fraction'] for r in rs]):.3f} |")

    if vw:
        L += ["", "## Phase 2 — mask-quality view weighting", "",
              f"Ellipsoid with **camera 0 mask 35% corrupted**. Per-camera quality "
              f"weights: {[f'{w:.2f}' for w in vw['view_weights_quality']]} "
              f"(camera 0 down-weighted).", "",
              "| fitting | diameter bias % | 3D IoU |", "|---|---|---|",
              f"| no silhouette-opt | {vw['no_opt']['diameter_bias_pct']:+.1f} | {vw['no_opt']['iou_3d']:.3f} |",
              f"| equal-weight opt | {vw['equal_weight']['diameter_bias_pct']:+.1f} | {vw['equal_weight']['iou_3d']:.3f} |",
              f"| **quality-weight opt** | {vw['quality_weight']['diameter_bias_pct']:+.1f} | {vw['quality_weight']['iou_3d']:.3f} |",
              f"| quality-weight + ellipsoid-init | {vw['quality_weight_ellipsoid_init']['diameter_bias_pct']:+.1f} | {vw['quality_weight_ellipsoid_init']['iou_3d']:.3f} |"]

    if flower:
        L += ["", "## Flower stress (sparse degree-8 fit)", "",
              "Bending energy = high-l roughness; lower = fewer flower petals.", "",
              "| config | bending energy | 3D IoU | Dbias % |", "|---|---|---|---|"]
        for k, v in flower.items():
            L.append(f"| {k} | {v['bending_energy']:.1f} | {v['iou_3d']:.3f} | {v['diameter_bias_pct']:+.1f} |")

    L += ["", "## Notes", "",
          "- Diameter bias here isolates the **SH-fit** error; the ~+6% D / +19% V",
          "  visual-hull bias is separate and enters through the refined points.",
          "- See `recovery.png`, `flower_stress.png`."]
    run.write_text("report.md", "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
