"""Quantify the visual-hull volume bias using synthetic spheres.

Renders exact silhouettes of known-diameter spheres through the real
4-camera refractive (PINPLATE) model, runs the production reconstruction
pipeline, and compares the reconstructed equivalent diameter / volume
against ground truth.  Produces a bias-vs-size curve.

A 4-view visual hull is a strict *upper bound* on the true shape, so the
reconstructed volume is expected to be >= truth; this script measures by
how much, as a function of bubble size and coarse voxel resolution.

Outputs (under visual_hull/test/<ts>-synthetic-sphere-bias/):
    report.md, results.json, bias_curve.png

Run:
    python scripts/validate_synthetic_spheres.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scipy.spatial import ConvexHull
from skimage.draw import polygon as sk_polygon

from visual_hull.camera import OpenLPTCameraSet
from visual_hull.hull import create_visual_hull
from visual_hull.io import load_tiff_mask
from visual_hull.properties import get_bubble_props
from visual_hull.refinement import find_surface_components, refine_surface_points
from visual_hull.test_runs import create_test_run

# ── Configuration (matches the 20Hz_r_b_1_lpt working dir) ────────────────────
WORKING_DIR = Path(
    r"X:\Shijie Zhong\Bubble Shear Project\Processed\20260710\20Hz_r_b_1_lpt"
)
CAMERA_DIR = WORKING_DIR / "camFile_VSC"
CAMERA_TEMPLATE = "vsc_cam{camera}.txt"
NUM_CAMERAS = 4
LIMITS_FULL = np.array([-85.0, 45.0, -60.0, 50.0, -40.0, 70.0])  # for image size only

# Test locations (mm) — representative of where real bubbles appear.
LOCATIONS = {
    "L1_mid": np.array([-20.0, -10.0, -5.0]),
    "L2_center": np.array([-20.0, -5.0, 15.0]),
    "L3_edge": np.array([-40.0, 20.0, 0.0]),
}
DIAMETERS_MM = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
COARSE_SIZES = [0.5, 1.0]  # mm; fine = coarse / 3 (production default)
N_SURFACE_SAMPLES = 8000


def fibonacci_sphere(n: int) -> np.ndarray:
    """~Uniformly distributed unit-sphere points (n, 3)."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    golden = np.pi * (1.0 + 5.0 ** 0.5)
    theta = golden * i
    return np.column_stack(
        (np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi))
    )


def render_sphere_masks(
    center: np.ndarray, radius: float, cameras: OpenLPTCameraSet, hw: tuple[int, int]
) -> list[np.ndarray] | None:
    """Exact silhouette of a sphere in each camera via projected-limb hull fill.

    A sphere is convex and the refractive projection is smooth, so the 2D
    convex hull of its projected surface points is the silhouette boundary.
    """
    height, width = hw
    surface = fibonacci_sphere(N_SURFACE_SAMPLES) * float(radius) + center
    masks: list[np.ndarray] = []
    for cam in range(cameras.count):
        proj = cameras.project_points(cam, surface)
        uv = proj.pixels[proj.valid]
        if uv.shape[0] < 3:
            return None  # sphere not visible in this camera
        try:
            hull = ConvexHull(uv)
        except Exception:
            return None
        poly = uv[hull.vertices]
        rr, cc = sk_polygon(poly[:, 1], poly[:, 0], shape=(height, width))
        mask = np.zeros((height, width), dtype=bool)
        mask[rr, cc] = True
        if not mask.any():
            return None
        masks.append(mask)
    return masks


def reconstruct_sphere(
    center: np.ndarray,
    diameter: float,
    coarse: float,
    cameras: OpenLPTCameraSet,
    hw: tuple[int, int],
) -> dict:
    """Render + reconstruct one sphere; return recon vs true size metrics."""
    radius = 0.5 * diameter
    true_volume = (4.0 / 3.0) * np.pi * radius ** 3

    masks = render_sphere_masks(center, radius, cameras, hw)
    if masks is None:
        return {"ok": False, "reason": "not_visible"}

    # Tight limits around the sphere keep the grid small and fast.
    margin = max(2.0, 0.8 * diameter) + 2.0 * coarse
    limits = np.array(
        [
            center[0] - radius - margin, center[0] + radius + margin,
            center[1] - radius - margin, center[1] + radius + margin,
            center[2] - radius - margin, center[2] + radius + margin,
        ]
    )
    voxel_size = np.full(3, float(coarse))
    fine_voxel_size = voxel_size / 3.0

    hull = create_visual_hull(masks, cameras, voxel_size, limits)
    if int(np.sum(hull.voxel_volume)) <= 0:
        return {"ok": False, "reason": "empty_hull"}

    components = find_surface_components(
        hull.voxel_volume, hull.grid_x, hull.grid_y, hull.grid_z
    )
    if not components:
        return {"ok": False, "reason": "no_components"}

    # Largest component = the sphere.
    surface_points = max(components, key=lambda c: c.shape[0])
    refined = refine_surface_points(
        surface_points, coarse_voxel_size=voxel_size,
        masks=masks, cameras=cameras, mv=2, res_inc=3,
    )
    if refined.shape[0] < 4:
        return {"ok": False, "reason": "too_few_refined"}

    image_resolution = np.array([hw[1], hw[0]], dtype=np.float64)
    _, props = get_bubble_props(
        refined, voxel_size=fine_voxel_size, image_resolution=image_resolution,
        num_cameras=cameras.count, limits=limits, cameras=cameras,
        voxels_center=np.mean(surface_points, axis=0),
    )
    recon_radius = float(props[3])
    recon_volume = float(props[4])
    recon_diameter = 2.0 * recon_radius
    return {
        "ok": True,
        "true_diameter": float(diameter),
        "true_volume": float(true_volume),
        "recon_diameter": recon_diameter,
        "recon_volume": recon_volume,
        "diameter_bias_pct": 100.0 * (recon_diameter - diameter) / diameter,
        "volume_bias_pct": 100.0 * (recon_volume - true_volume) / true_volume,
        "n_surface_pts": int(refined.shape[0]),
    }


def main() -> None:
    run = create_test_run(PROJECT_ROOT, "synthetic-sphere-bias")
    print(f"Output → {run.root}")

    camera_paths = [CAMERA_DIR / CAMERA_TEMPLATE.format(camera=i) for i in range(NUM_CAMERAS)]
    cameras = OpenLPTCameraSet.from_camera_files(camera_paths)

    # Image size from a real mask.
    real = load_tiff_mask(WORKING_DIR / "imgFile_bb" / "cam0" / "img000000.tif")
    hw = (real.shape[0], real.shape[1])
    print(f"Image size (HxW): {hw}")

    results: list[dict] = []

    # Primary curve: location L1, both coarse sizes, all diameters.
    for coarse in COARSE_SIZES:
        for diameter in DIAMETERS_MM:
            r = reconstruct_sphere(LOCATIONS["L1_mid"], diameter, coarse, cameras, hw)
            r.update({"location": "L1_mid", "coarse_mm": coarse})
            results.append(r)
            tag = (f"D_bias={r['diameter_bias_pct']:+.1f}%  V_bias={r['volume_bias_pct']:+.1f}%"
                   if r["ok"] else f"FAILED ({r['reason']})")
            print(f"[coarse={coarse}mm  D={diameter:>4}mm]  {tag}")

    # Spatial check: all locations at D=1.5mm, coarse=0.5mm.
    print("\n-- spatial variation (D=1.5mm, coarse=0.5mm) --")
    for name, center in LOCATIONS.items():
        r = reconstruct_sphere(center, 1.5, 0.5, cameras, hw)
        r.update({"location": name, "coarse_mm": 0.5})
        results.append(r)
        tag = (f"D_bias={r['diameter_bias_pct']:+.1f}%  V_bias={r['volume_bias_pct']:+.1f}%"
               if r["ok"] else f"FAILED ({r['reason']})")
        print(f"[{name}]  {tag}")

    run.write_json("results.json", results)
    _plot(results, run.path("bias_curve.png"))
    _report(results, run)
    print(f"\nDone. Report: {run.path('report.md')}")


def _plot(results: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for coarse in COARSE_SIZES:
        pts = [r for r in results if r.get("ok") and r["location"] == "L1_mid" and r["coarse_mm"] == coarse]
        pts.sort(key=lambda r: r["true_diameter"])
        if not pts:
            continue
        d = [p["true_diameter"] for p in pts]
        ax1.plot(d, [p["diameter_bias_pct"] for p in pts], "o-", label=f"coarse={coarse}mm")
        ax2.plot(d, [p["volume_bias_pct"] for p in pts], "o-", label=f"coarse={coarse}mm")
    for ax, ylab, title in (
        (ax1, "diameter bias (%)", "Equivalent-diameter bias vs true size"),
        (ax2, "volume bias (%)", "Volume bias vs true size"),
    ):
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("true diameter (mm)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _report(results: list[dict], run) -> None:
    lines = [
        "# Synthetic-sphere visual-hull bias validation",
        "",
        "Exact sphere silhouettes rendered through the real 4-camera PINPLATE",
        "refractive model, reconstructed with the production pipeline",
        "(coarse hull -> surface components -> refine coarse/3 -> properties).",
        "Reconstructed **equivalent diameter** and **voxel volume** compared to truth.",
        "",
        "## Bias vs size (location L1_mid)",
        "",
        "| coarse (mm) | true D (mm) | recon D (mm) | D bias % | V bias % | surf pts |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        if r["location"] != "L1_mid":
            continue
        if not r["ok"]:
            lines.append(f"| {r['coarse_mm']} | {r.get('true_diameter','?')} | — | FAILED: {r['reason']} | | |")
            continue
        lines.append(
            f"| {r['coarse_mm']} | {r['true_diameter']:.2f} | {r['recon_diameter']:.3f} "
            f"| {r['diameter_bias_pct']:+.1f} | {r['volume_bias_pct']:+.1f} | {r['n_surface_pts']} |"
        )
    lines += ["", "## Spatial variation (D=1.5mm, coarse=0.5mm)", "",
              "| location | D bias % | V bias % |", "|---|---|---|"]
    for r in results:
        if r["coarse_mm"] == 0.5 and r["location"] != "L1_mid" and r.get("true_diameter") == 1.5:
            if r["ok"]:
                lines.append(f"| {r['location']} | {r['diameter_bias_pct']:+.1f} | {r['volume_bias_pct']:+.1f} |")
            else:
                lines.append(f"| {r['location']} | FAILED: {r['reason']} | |")

    ok = [r for r in results if r.get("ok") and r["location"] == "L1_mid" and r["coarse_mm"] == 0.5]
    if ok:
        mean_v = np.mean([r["volume_bias_pct"] for r in ok])
        mean_d = np.mean([r["diameter_bias_pct"] for r in ok])
        lines += ["", "## Takeaway", "",
                  f"- Mean volume bias @0.5mm coarse: **{mean_v:+.1f}%**",
                  f"- Mean diameter bias @0.5mm coarse: **{mean_d:+.1f}%**",
                  "- See `bias_curve.png`."]
    run.write_text("report.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
