"""
Reconstruct bubbles from TIFF mask images and OpenLPT VSC camera files.

Usage
-----
    # Dry-run: print config without running
    python scripts/reconstruct_tiff_data.py --dry-run

    # Reconstruct a single test frame
    python scripts/reconstruct_tiff_data.py --frames 0

    # Reconstruct frames 0-99 with spherical harmonics (degree 4)
    python scripts/reconstruct_tiff_data.py --frames 0 99 --sh-degree 4

    # Full run with 80% of CPU workers
    python scripts/reconstruct_tiff_data.py --frames 0 99 --sh-degree 4 --max-workers -1

Expected directory layout under ``--working-dir``::

    <working-dir>/
        imgFile_bb/
            cam0/img000000.tif ... img012604.tif
            cam1/img000000.tif ... img012604.tif
            cam2/img000000.tif ... img012604.tif
            cam3/img000000.tif ... img012604.tif
        camFile_VSC/
            vsc_cam0.txt ... vsc_cam3.txt
        Results/
            recon/          ← per-frame output written here
        config_recon/       ← run configuration saved here
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import savemat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_hull.camera import OpenLPTCameraSet
from visual_hull.io import load_tiff_masks
from visual_hull.reconstruction import run_full_reconstruction_from_data
from visual_hull.writers import write_reconstruction
from visual_hull.improved.spherical_harmonics.surface import (
    SphericalHarmonicFitConfig,
    fit_spherical_harmonic_surface,
)


# ── Defaults matching the config.txt in the user's working directory ──────────
DEFAULT_WORKING_DIR = Path(
    r"X:\Shijie Zhong\Bubble Shear Project\Processed\20260710\20Hz_r_b_1_lpt"
)
DEFAULT_NUM_CAMERAS = 4
DEFAULT_VOXEL_SIZE = [1.0, 1.0, 1.0]  # mm (coarse pass; fine pass at /3)
DEFAULT_LIMITS = [-85.0, 45.0, -60.0, 50.0, -40.0, 70.0]  # mm (config.txt "View Volume")


def _cpu_count() -> int:
    return os.cpu_count() or 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct bubbles from TIFF masks and VSC camera files."
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=DEFAULT_WORKING_DIR,
        help="Root working directory containing imgFile_bb/, camFile_VSC/, etc.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[0],
        help="Frame number(s) to reconstruct.  Two values = range [start, end] inclusive.",
    )
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=DEFAULT_NUM_CAMERAS,
        help="Number of cameras.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        default=DEFAULT_VOXEL_SIZE,
        metavar=("DX", "DY", "DZ"),
        help="Coarse voxel size in mm.  Default 1.0 mm.",
    )
    parser.add_argument(
        "--refine-to",
        type=float,
        default=None,
        metavar="MM",
        help=(
            "Target refinement voxel size in mm (single value, same for all axes). "
            "Enables multi-level refinement from coarse voxels down to this size. "
            "Default: use pipeline built-in refinement (coarse / 3). "
            "Practical limit: ~0.05 mm (camera pixel footprint). "
            "Example: --refine-to 0.05"
        ),
    )
    parser.add_argument(
        "--limits",
        type=float,
        nargs=6,
        default=DEFAULT_LIMITS,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Reconstruction volume limits in mm.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Per-frame output directory.  Defaults to <working-dir>/Results/recon.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Config output directory.  Defaults to <working-dir>/config_recon.",
    )
    parser.add_argument(
        "--format",
        choices=["mat", "h5", "hdf5"],
        default="mat",
        help="Per-frame export format.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=-1,
        help=(
            "Number of parallel worker processes. "
            "Use -1 for 80%% of CPU cores, 0 for all cores, "
            "or a positive integer for an exact count."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without reconstructing.",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default="imgFile_bb",
        help="Subdirectory containing per-camera mask folders (relative to working-dir).",
    )
    parser.add_argument(
        "--camera-dir",
        type=str,
        default="camFile_VSC",
        help="Subdirectory containing vsc_cam*.txt files (relative to working-dir).",
    )
    parser.add_argument(
        "--mask-template",
        type=str,
        default="img{frame:06d}.tif",
        help="Python format string for mask filenames. Receives 'frame'.",
    )
    parser.add_argument(
        "--camera-template",
        type=str,
        default="vsc_cam{camera}.txt",
        help="Python format string for camera filenames. Receives 'camera' (0-indexed).",
    )
    # ── Spherical Harmonics ─────────────────────────────────────────────────
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=0,
        help=(
            "Spherical harmonic max degree for surface fitting. "
            "0 = disabled (voxel output only).  "
            "4 = moderate detail, 8 = high detail."
        ),
    )
    parser.add_argument(
        "--sh-regularization",
        type=float,
        default=1e-2,
        help="Tikhonov regularization strength for SH fitting (higher = smoother).",
    )
    parser.add_argument(
        "--sh-no-adaptive",
        action="store_true",
        help="Use fixed SH degree for all bubbles (disable per-bubble adaptive degree).",
    )
    parser.add_argument(
        "--sh-theta-samples",
        type=int,
        default=40,
        help="Theta resolution for SH mesh generation.",
    )
    parser.add_argument(
        "--sh-phi-samples",
        type=int,
        default=80,
        help="Phi resolution for SH mesh generation.",
    )
    parser.add_argument(
        "--sh-inscribed",
        action="store_true",
        help=(
            "Constrain the SH surface to stay inside the visual hull: penalize "
            "outward overshoot (r_SH > r_voxel) during fitting so the surface "
            "hugs the inner side of the hull boundary instead of bulging past it."
        ),
    )
    parser.add_argument(
        "--sh-overshoot-weight",
        type=float,
        default=50.0,
        help="Weight of the outward-overshoot penalty for --sh-inscribed.",
    )
    parser.add_argument(
        "--sh-silhouette",
        action="store_true",
        help=(
            "After fitting, optimize each bubble's SH coefficients so its "
            "projected silhouette matches its own re-projected visual-hull "
            "silhouette in every camera (overlap-free target)."
        ),
    )
    parser.add_argument(
        "--sh-silhouette-scale",
        type=int,
        default=4,
        help="Downsample factor for the silhouette-matching optimization.",
    )
    parser.add_argument(
        "--sh-silhouette-passes",
        type=int,
        default=5,
        help="Coordinate-descent passes for silhouette matching.",
    )
    parser.add_argument(
        "--sh-min-points-per-coeff",
        type=float,
        default=3.0,
        help=(
            "Minimum surface points required per SH coefficient. Caps each "
            "bubble's degree so undersampled fits can't oscillate into "
            "'flower petals'. Higher = smoother/safer, 0 = disable cap. "
            "Default 3.0."
        ),
    )
    # ── Size / shape filtering ──────────────────────────────────────────────
    parser.add_argument(
        "--size-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help=(
            "Keep only bubbles whose equivalent diameter (mm) is within "
            "[MIN, MAX]. D_eq = 2*(3V/4pi)^(1/3). Bubbles outside the range "
            "are dropped entirely (no voxels, no SH). Default: no filtering."
        ),
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=None,
        help=(
            "Drop bubbles whose aspect ratio (major/minor axis) exceeds this. "
            "Removes elongated sliver phantoms. Real bubbles are ~1.5-3; "
            "phantoms >6. Default: no filtering."
        ),
    )
    parser.add_argument(
        "--clean-mask-border",
        action="store_true",
        help=(
            "Zero saturated border bands in the masks before reconstruction "
            "(removes segmentation border artifacts that create edge phantoms)."
        ),
    )
    return parser.parse_args()


def resolve_frame_list(args: argparse.Namespace) -> list[int]:
    if len(args.frames) == 1:
        return [args.frames[0]]
    if len(args.frames) == 2:
        start, end = args.frames
        if end < start:
            raise ValueError(f"Frame range end ({end}) < start ({start}).")
        return list(range(start, end + 1))
    return list(args.frames)


def resolve_workers(args: argparse.Namespace) -> int:
    raw = int(args.max_workers)
    cpu = _cpu_count()
    if raw == -1:
        return max(1, int(cpu * 0.8))
    if raw == 0:
        return cpu
    if raw < 1:
        return 1
    return raw


def save_config(config: dict, config_dir: Path) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = config_dir / f"config_recon_{ts}.json"
    with open(path, "w") as fh:
        json.dump(config, fh, indent=2, default=str)
    return path


def _parsimonious_sh_degree(
    vertices: np.ndarray,
    max_degree: int,
    regularization: float,
    improvement_threshold: float = 0.05,
    rmse_floor_mm: float = 0.01,
) -> int:
    """Select the minimal SH degree needed to represent a bubble.

    Fits increasing even degrees (2, 4, 6, …) and stops when adding
    more modes reduces RMSE by less than *improvement_threshold*
    (default 5 % relative improvement), OR when RMSE is already below
    *rmse_floor_mm* (default 0.01 mm — far below pixel resolution).

    Falls back to the voxel-count heuristic for very small bubbles
    (< 20 points).
    """
    from visual_hull.improved.spherical_harmonics.surface import (
        _basis_terms,
        _cartesian_to_spherical,
        _design_matrix,
        _fit_coefficients,
    )

    pts = np.asarray(vertices, dtype=np.float64)
    n_pts = pts.shape[0]

    if n_pts < 6:
        return 2
    if n_pts < 20:
        # Very sparse: use voxel-count heuristic
        return _adapt_sh_degree(n_pts, max_degree)

    center = np.mean(pts, axis=0)
    radius, theta, phi = _cartesian_to_spherical(pts, center)

    best_degree = 2
    best_rmse = float("inf")
    candidate_degrees = list(range(2, int(max_degree) + 1, 2))
    if not candidate_degrees:
        return 2

    for deg in candidate_degrees:
        terms = _basis_terms(deg)
        design = _design_matrix(theta, phi, terms)
        coeffs = _fit_coefficients(design, radius, float(regularization))
        fitted = np.maximum(design @ coeffs, 1e-6)
        rmse = float(np.sqrt(np.mean((fitted - radius) ** 2)))

        # Stop if we're already below the pixel-resolution floor
        if rmse <= float(rmse_floor_mm):
            best_degree = deg
            break

        if rmse < best_rmse * (1.0 - float(improvement_threshold)):
            best_degree = deg
            best_rmse = rmse
        else:
            # Improvement too small — stick with previous degree
            break

    return best_degree


def _adapt_sh_degree(n_voxels: int, max_degree: int) -> int:
    """Choose an appropriate SH degree based on the number of surface voxels.

    Small bubbles with few points can't support high-degree fits — the
    spherical harmonics oscillate wildly between sparse samples (the
    "flower petal" problem).  We clamp the degree so there are at least
    ~10 points per coefficient.
    """
    n = int(n_voxels)
    if max_degree <= 2:
        return max_degree
    # Each degree L has 2L+1 coefficients, total = (L+1)²
    # We want at least 10 points per coefficient
    for deg in range(2, max_degree + 1, 2):
        n_coeffs = (deg + 1) ** 2
        if n < n_coeffs * 10:
            return max(2, deg - 2)
    return max_degree


def _max_supported_sh_degree(
    n_points: int, max_degree: int, min_points_per_coeff: float
) -> int:
    """Largest even SH degree whose coefficients are supported by the data.

    A degree-``L`` fit has ``(L+1)²`` coefficients.  When the number of
    surface points is comparable to (or below) that, the least-squares
    system is underdetermined and the fitted radius interpolates the
    sparse samples while oscillating wildly between them — the "flower
    petal" artifact (near-zero RMSE, spiky mesh).

    Requiring at least ``min_points_per_coeff`` points per coefficient
    caps the degree so every retained mode is actually constrained by
    the data.  ``min_points_per_coeff <= 0`` disables the cap.
    """
    if min_points_per_coeff <= 0.0:
        return max(2, int(max_degree))
    best = 2
    for deg in range(2, int(max_degree) + 1, 2):
        if (deg + 1) ** 2 * float(min_points_per_coeff) <= n_points:
            best = deg
        else:
            break
    return best


def _fill_hull_silhouette(pixels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Filled convex-hull silhouette of projected points (bubbles are convex)."""
    from scipy.spatial import ConvexHull
    from skimage.draw import polygon as sk_polygon

    mask = np.zeros(shape, dtype=bool)
    if pixels.shape[0] < 3:
        return mask
    try:
        hull = ConvexHull(pixels)
    except Exception:
        return mask
    poly = pixels[hull.vertices]
    rows, cols = sk_polygon(poly[:, 1], poly[:, 0], shape=shape)
    mask[rows, cols] = True
    return mask


def _camera_fill(points_3d, cameras, camera_index, shape, scale):
    projection = cameras.project_points(camera_index, points_3d)
    pixels = projection.pixels[projection.valid] / float(scale)
    return _fill_hull_silhouette(pixels, shape)


def _silhouette_iou(pred: np.ndarray, target: np.ndarray) -> float:
    inter = int(np.count_nonzero(pred & target))
    union = int(np.count_nonzero(pred | target))
    return 1.0 if union == 0 else inter / union


def _optimize_coeffs_to_hull_silhouette(
    coefficients, center, terms, opt_config, cameras, bubble_voxels,
    image_shape, scale, passes,
):
    """Coordinate-descent the SH coefficients to match this bubble's own
    re-projected hull silhouette in every camera (overlap-free target)."""
    from visual_hull.improved.spherical_harmonics.surface import _grid_vertices_faces

    ds_shape = (int(image_shape[0]) // scale, int(image_shape[1]) // scale)
    n_cam = cameras.count
    targets = [_camera_fill(bubble_voxels, cameras, c, ds_shape, scale) for c in range(n_cam)]

    def mean_iou(coeffs):
        verts, _ = _grid_vertices_faces(center, coeffs, terms, opt_config)
        return float(np.mean([
            _silhouette_iou(_camera_fill(verts, cameras, c, ds_shape, scale), targets[c])
            for c in range(n_cam)
        ]))

    best = np.asarray(coefficients, dtype=np.float64).copy()
    best_iou = mean_iou(best)
    step = 0.12 * abs(best[0]) if best[0] != 0 else 0.1
    for _ in range(int(passes)):
        improved = False
        for i in range(best.shape[0]):
            for delta in (step, -step):
                cand = best.copy()
                cand[i] += delta
                iou = mean_iou(cand)
                if iou > best_iou + 1e-4:
                    best, best_iou = cand, iou
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-4 * (abs(best[0]) + 1e-9):
                break
    return best, best_iou


def _fit_sh_for_bubbles(
    voxels: np.ndarray,
    bubbles: np.ndarray,
    sh_config: SphericalHarmonicFitConfig,
    *,
    adaptive_degree: bool = True,
    min_points_per_coeff: float = 3.0,
    cameras=None,
    image_shape: tuple[int, int] | None = None,
    silhouette_optimize: bool = False,
    silhouette_scale: int = 4,
    silhouette_passes: int = 5,
) -> dict | None:
    """Fit spherical harmonics to each bubble's refined voxels.

    With ``adaptive_degree=True`` (the default), each bubble gets its own
    SH degree based on how many surface voxels it has.  This prevents
    high-degree oscillations ("flower petals") on sparse bubbles.

    ``min_points_per_coeff`` caps every bubble's degree so each SH
    coefficient is constrained by at least that many surface points —
    the primary guard against the flower-petal artifact (applied in both
    the adaptive and fixed-degree paths).  Set to 0 to disable.

    Returns None when there are no bubbles or SH is disabled.
    """
    if bubbles.shape[1] == 0:
        return None

    sh_centers: list[np.ndarray] = []
    sh_coefficients: list[np.ndarray] = []
    sh_basis_l: list[np.ndarray] = []
    sh_basis_m: list[np.ndarray] = []
    sh_vertices: list[np.ndarray] = []
    sh_faces: list[np.ndarray] = []
    sh_fit_rmse: list[float] = []
    sh_degrees: list[int] = []

    global_max_degree = int(sh_config.max_degree)
    # Pre-compute the full set of basis terms for the requested max degree
    from visual_hull.improved.spherical_harmonics.surface import _basis_terms
    full_terms = _basis_terms(global_max_degree)

    for b in range(bubbles.shape[1]):
        start = int(bubbles[0, b]) - 1  # MATLAB 1-indexed → 0-indexed
        end = int(bubbles[1, b])
        bubble_voxels = voxels[start:end, :]

        n_pts = bubble_voxels.shape[0]
        if n_pts < 6:
            continue  # too few points for SH fit

        # ── Degree cap: enough surface points per SH coefficient ─────────
        # Prevents underdetermined fits that oscillate into "flower petals".
        degree_cap = _max_supported_sh_degree(
            n_pts, global_max_degree, min_points_per_coeff
        )

        # ── Parsimonious degree for this bubble ──────────────────────────
        if adaptive_degree:
            bubble_degree = _parsimonious_sh_degree(
                bubble_voxels,
                max_degree=degree_cap,
                regularization=float(sh_config.regularization),
            )
        else:
            bubble_degree = min(global_max_degree, degree_cap)

        bubble_config = SphericalHarmonicFitConfig(
            max_degree=bubble_degree,
            regularization=float(sh_config.regularization),
            theta_samples=int(sh_config.theta_samples),
            phi_samples=int(sh_config.phi_samples),
            minimum_radius=float(sh_config.minimum_radius),
            inscribed=bool(sh_config.inscribed),
            overshoot_weight=float(sh_config.overshoot_weight),
            inscribed_iters=int(sh_config.inscribed_iters),
        )

        try:
            sh_surface = fit_spherical_harmonic_surface(
                bubble_voxels,
                config=bubble_config,
                masks=None,
                cameras=None,
            )
        except Exception:
            continue

        # ── Optional: match the re-projected per-bubble hull silhouette ─────
        if silhouette_optimize and cameras is not None and image_shape is not None:
            try:
                from dataclasses import replace as _dc_replace
                from visual_hull.improved.spherical_harmonics.surface import (
                    _basis_terms as _bt, _grid_vertices_faces as _gvf,
                    _cartesian_to_spherical as _c2s, _design_matrix as _dm,
                )
                terms_b = _bt(bubble_degree)
                opt_cfg = SphericalHarmonicFitConfig(
                    max_degree=bubble_degree, theta_samples=24, phi_samples=48)
                opt_coeffs, _ = _optimize_coeffs_to_hull_silhouette(
                    sh_surface.coefficients, sh_surface.center, terms_b, opt_cfg,
                    cameras, bubble_voxels, image_shape,
                    silhouette_scale, silhouette_passes)
                final_v, final_f = _gvf(sh_surface.center, opt_coeffs, terms_b, bubble_config)
                r_data, th, ph = _c2s(bubble_voxels, sh_surface.center)
                fitted = np.maximum(_dm(th, ph, terms_b) @ opt_coeffs, 1e-6)
                rmse = float(np.sqrt(np.mean((fitted - r_data) ** 2)))
                sh_surface = _dc_replace(
                    sh_surface, coefficients=opt_coeffs,
                    vertices=final_v, faces=final_f, fit_rmse=rmse)
            except Exception:
                pass

        # Pad coefficients to the global max_degree for uniform storage
        n_full = (global_max_degree + 1) ** 2
        n_bubble = sh_surface.coefficients.shape[0]
        coeffs_padded = np.zeros(n_full, dtype=np.float64)
        coeffs_padded[:n_bubble] = sh_surface.coefficients

        sh_centers.append(sh_surface.center)
        sh_coefficients.append(coeffs_padded)
        sh_basis_l.append(np.array([t[0] for t in full_terms], dtype=np.int32))
        sh_basis_m.append(np.array([t[1] for t in full_terms], dtype=np.int32))
        sh_vertices.append(sh_surface.vertices)
        sh_faces.append(sh_surface.faces + 1)  # 0-index → 1-index (MATLAB)
        sh_fit_rmse.append(sh_surface.fit_rmse)
        sh_degrees.append(bubble_degree)

    if not sh_centers:
        return None

    return {
        "sh_max_degree": global_max_degree,
        "sh_num_bubbles": len(sh_centers),
        "sh_centers": np.array(sh_centers, dtype=np.float64),
        "sh_coefficients": np.array(sh_coefficients, dtype=np.float64),
        "sh_basis_l": np.array(sh_basis_l, dtype=np.int32),
        "sh_basis_m": np.array(sh_basis_m, dtype=np.int32),
        "sh_vertices": np.array(
            [np.pad(v, ((0, max(v.shape[0] for v in sh_vertices) - v.shape[0]), (0, 0)))
             for v in sh_vertices],
            dtype=np.float64,
        ),
        "sh_faces": np.array(
            [np.pad(f, ((0, max(f.shape[0] for f in sh_faces) - f.shape[0]), (0, 0)))
             for f in sh_faces],
            dtype=np.int32,
        ),
        "sh_fit_rmse": np.array(sh_fit_rmse, dtype=np.float64),
        "sh_degree_used": np.array(sh_degrees, dtype=np.int32),
    }


def _multi_level_refine(
    surface_points: np.ndarray,
    coarse_voxel_size: np.ndarray,
    masks: list[np.ndarray],
    cameras: OpenLPTCameraSet,
    target_mm: float,
) -> np.ndarray:
    """Refine surface points through multiple levels down to *target_mm*.

    Each level subdivides by ~4-5×.  The coarse level uses a 2-voxel
    margin; finer levels use a 1-voxel margin.

    Returns refined points at approximately *target_mm* resolution.
    """
    from visual_hull.refinement import refine_surface_points

    current = np.asarray(surface_points, dtype=np.float64)
    current_size = np.asarray(coarse_voxel_size, dtype=np.float64)
    target = float(target_mm)

    level = 0
    while True:
        current_min = float(np.min(current_size))
        if current_min <= target * 1.01:  # close enough (1% tolerance)
            break

        # Choose res_inc to step down by ~4-5× per level
        res_inc = min(5, max(3, int(np.round(current_min / target))))
        if res_inc < 2:
            res_inc = 2

        mv = 2 if level == 0 else 1
        next_size = current_size / float(res_inc)

        current = refine_surface_points(
            current,
            coarse_voxel_size=current_size,
            masks=masks,
            cameras=cameras,
            mv=mv,
            res_inc=res_inc,
        )
        current_size = next_size
        level += 1

        if current.shape[0] == 0:
            break

    return current


def reconstruct_single_frame(
    frame: int,
    working_dir: Path,
    mask_dir_name: str,
    camera_dir_name: str,
    mask_template: str,
    camera_template: str,
    num_cameras: int,
    voxel_size: list[float],
    limits: list[float],
    output_dir: Path,
    export_format: str,
    sh_config: SphericalHarmonicFitConfig | None,
    sh_adaptive: bool = True,
    refine_to: float | None = None,
    size_range: tuple[float, float] | None = None,
    sh_min_points_per_coeff: float = 3.0,
    max_aspect_ratio: float | None = None,
    clean_mask_border: bool = False,
    silhouette_optimize: bool = False,
    silhouette_scale: int = 4,
    silhouette_passes: int = 5,
) -> dict:
    """Reconstruct a single frame and return a result summary dict.

    ``size_range`` — optional ``(min, max)`` equivalent-diameter bounds in
    mm.  Bubbles whose equivalent diameter falls outside the range are
    dropped entirely (excluded from voxel output and SH fitting).
    ``max_aspect_ratio`` — optional cap; bubbles more elongated than this are
    dropped (removes edge/border sliver phantoms).
    ``clean_mask_border`` — zero saturated border bands in the masks first.
    """
    from visual_hull.hull import create_visual_hull
    from visual_hull.io import stack_boolean_images
    from visual_hull.models import FullReconstructionResult
    from visual_hull.properties import get_bubble_props
    from visual_hull.refinement import find_surface_components

    mask_root = working_dir / mask_dir_name
    camera_root = working_dir / camera_dir_name

    # Load masks
    masks = load_tiff_masks(
        mask_root,
        frame,
        num_cameras,
        camera_base=0,
        name_template=mask_template,
        subdir_template="cam{camera}",
    )

    if clean_mask_border:
        from visual_hull.io import clean_mask_border as _clean_border
        masks = [_clean_border(m) for m in masks]

    # Load cameras
    camera_paths = [
        camera_root / camera_template.format(camera=idx)
        for idx in range(num_cameras)
    ]
    for cp in camera_paths:
        if not cp.is_file():
            raise FileNotFoundError(f"Camera file not found: {cp}")

    cameras = OpenLPTCameraSet.from_camera_files(camera_paths)

    # ── Coarse visual hull ─────────────────────────────────────────────────
    _voxel_size = np.asarray(voxel_size, dtype=np.float64)
    _limits = np.asarray(limits, dtype=np.float64)

    coarse_result = create_visual_hull(
        masks=masks,
        cameras=cameras,
        voxel_size=_voxel_size,
        limits=_limits,
    )
    real_images = stack_boolean_images(masks)

    # Determine fine voxel size for properties
    if refine_to is not None:
        fine_voxel_size = np.full(3, float(refine_to), dtype=np.float64)
    else:
        fine_voxel_size = _voxel_size / 3.0

    if int(np.sum(coarse_result.voxel_volume)) <= 0:
        result = FullReconstructionResult(
            voxel_size=_voxel_size,
            voxel_size_2=fine_voxel_size,
            limits=_limits,
            real_images=real_images,
            voxels=np.empty((0, 3), dtype=np.float64),
            bubbles=np.empty((2, 0), dtype=np.int64),
            properties=np.empty((0, 15), dtype=np.float64),
            completed=True,
            coarse_result=coarse_result,
        )
        suffix = ".mat" if export_format == "mat" else ".h5"
        output_path = output_dir / f"Bubble_Frame_{frame:06d}{suffix}"
        write_reconstruction(result, output_path, export_format=export_format)
        return {
            "frame": frame, "output": str(output_path),
            "voxel_count": 0, "bubble_count": 0,
            "completed": True, "sh_saved": False,
        }

    # ── Surface components ─────────────────────────────────────────────────
    surface_components = find_surface_components(
        coarse_result.voxel_volume,
        coarse_result.grid_x,
        coarse_result.grid_y,
        coarse_result.grid_z,
    )

    image_resolution = np.array([real_images.shape[1], real_images.shape[0]], dtype=np.float64)

    all_voxels: list[np.ndarray] = []
    bubbles: list[tuple[int, int]] = []
    properties: list[np.ndarray] = []
    count = 0
    filtered_out = 0

    for surface_points in surface_components:
        # ── Refinement (multi-level or default) ────────────────────────────
        if refine_to is not None:
            refined_points = _multi_level_refine(
                surface_points, _voxel_size, masks, cameras,
                target_mm=refine_to,
            )
        else:
            from visual_hull.refinement import refine_surface_points
            refined_points = refine_surface_points(
                surface_points,
                coarse_voxel_size=_voxel_size,
                masks=masks,
                cameras=cameras,
                mv=2,
                res_inc=3,
            )

        if refined_points.shape[0] < 4:
            continue

        voxel_list, props = get_bubble_props(
            refined_points,
            voxel_size=fine_voxel_size,
            image_resolution=image_resolution,
            num_cameras=num_cameras,
            limits=_limits,
            cameras=cameras,
            voxels_center=np.mean(surface_points, axis=0),
        )

        # ── Size-range filter (equivalent diameter, mm) ────────────────────
        # props[3] is the equal-volume-sphere radius; D_eq = 2 * radius.
        if size_range is not None:
            equiv_diameter = 2.0 * float(props[3])
            if equiv_diameter < size_range[0] or equiv_diameter > size_range[1]:
                filtered_out += 1
                continue

        # ── Aspect-ratio filter (drop elongated sliver phantoms) ───────────
        # props[5] is major_mag / minor_mag.
        if max_aspect_ratio is not None and float(props[5]) > max_aspect_ratio:
            filtered_out += 1
            continue

        all_voxels.append(voxel_list)
        bubbles.append((count + 1, count + voxel_list.shape[0]))
        properties.append(props)
        count += voxel_list.shape[0]

    final_voxels = np.vstack(all_voxels) if all_voxels else np.empty((0, 3), dtype=np.float64)
    bubble_array = np.array(bubbles, dtype=np.int64).T if bubbles else np.empty((2, 0), dtype=np.int64)
    props_array = np.vstack(properties) if properties else np.empty((0, 15), dtype=np.float64)

    result = FullReconstructionResult(
        voxel_size=_voxel_size,
        voxel_size_2=fine_voxel_size,
        limits=_limits,
        real_images=real_images,
        voxels=final_voxels,
        bubbles=bubble_array,
        properties=props_array,
        completed=True,
        coarse_result=coarse_result,
    )

    # Write baseline output
    suffix = ".mat" if export_format == "mat" else ".h5"
    output_path = output_dir / f"Bubble_Frame_{frame:06d}{suffix}"
    write_reconstruction(result, output_path, export_format=export_format)

    summary: dict = {
        "frame": frame,
        "output": str(output_path),
        "voxel_count": int(result.voxels.shape[0]),
        "bubble_count": int(result.bubbles.shape[1]) if result.bubbles.ndim == 2 else 0,
        "completed": bool(result.completed),
        "sh_saved": False,
        "filtered_out_of_size_range": int(filtered_out),
    }

    # ── Spherical Harmonics (optional) ──────────────────────────────────────
    if sh_config is not None and result.bubbles.ndim == 2 and result.bubbles.shape[1] > 0:
        sh_data = _fit_sh_for_bubbles(
            result.voxels,
            result.bubbles,
            sh_config,
            adaptive_degree=sh_adaptive,
            min_points_per_coeff=sh_min_points_per_coeff,
            cameras=cameras,
            image_shape=(real_images.shape[0], real_images.shape[1]),
            silhouette_optimize=silhouette_optimize,
            silhouette_scale=silhouette_scale,
            silhouette_passes=silhouette_passes,
        )
        if sh_data is not None:
            sh_path = output_dir / f"Bubble_Frame_{frame:06d}_sh.mat"
            savemat(str(sh_path), sh_data)
            summary["sh_output"] = str(sh_path)
            summary["sh_saved"] = True
            summary["sh_num_bubbles"] = sh_data["sh_num_bubbles"]
            summary["sh_fit_rmse_mean"] = float(np.mean(sh_data["sh_fit_rmse"]))

    return summary


def main() -> None:
    args = parse_args()
    working_dir = args.working_dir.resolve()
    workers = resolve_workers(args)

    frames = resolve_frame_list(args)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else working_dir / "Results" / "recon"
    )
    config_dir = (
        args.config_dir.resolve()
        if args.config_dir
        else working_dir / "config_recon"
    )

    # ── Size-range filter ─────────────────────────────────────────────────
    size_range: tuple[float, float] | None = None
    if args.size_range is not None:
        lo, hi = float(args.size_range[0]), float(args.size_range[1])
        if lo < 0 or hi <= lo:
            raise ValueError(
                f"--size-range must satisfy 0 <= MIN < MAX (got {lo}, {hi})."
            )
        size_range = (lo, hi)

    if args.max_aspect_ratio is not None and args.max_aspect_ratio <= 1.0:
        raise ValueError("--max-aspect-ratio must be > 1.0.")

    # ── SH config ───────────────────────────────────────────────────────────
    sh_config: SphericalHarmonicFitConfig | None = None
    sh_adaptive = not args.sh_no_adaptive
    if args.sh_degree > 0:
        sh_config = SphericalHarmonicFitConfig(
            max_degree=args.sh_degree,
            regularization=args.sh_regularization,
            theta_samples=args.sh_theta_samples,
            phi_samples=args.sh_phi_samples,
            inscribed=args.sh_inscribed,
            overshoot_weight=args.sh_overshoot_weight,
        )

    config = {
        "working_dir": str(working_dir),
        "mask_dir": str(working_dir / args.mask_dir),
        "camera_dir": str(working_dir / args.camera_dir),
        "output_dir": str(output_dir),
        "config_dir": str(config_dir),
        "frames": frames,
        "num_frames": len(frames),
        "num_cameras": args.num_cameras,
        "voxel_size_mm": args.voxel_size,
        "limits_mm": args.limits,
        "export_format": args.format,
        "max_workers_requested": args.max_workers,
        "max_workers_actual": workers,
        "cpu_count": _cpu_count(),
        "mask_template": args.mask_template,
        "camera_template": args.camera_template,
        "sh_enabled": sh_config is not None,
        "sh_max_degree": args.sh_degree if sh_config else 0,
        "sh_regularization": args.sh_regularization if sh_config else None,
        "sh_adaptive": sh_adaptive,
        "sh_inscribed": args.sh_inscribed,
        "sh_overshoot_weight": args.sh_overshoot_weight if args.sh_inscribed else None,
        "sh_min_points_per_coeff": args.sh_min_points_per_coeff,
        "refine_to_mm": args.refine_to,
        "size_range_mm": list(size_range) if size_range else None,
        "max_aspect_ratio": args.max_aspect_ratio,
        "clean_mask_border": args.clean_mask_border,
        "sh_silhouette": args.sh_silhouette,
        "sh_silhouette_scale": args.sh_silhouette_scale if args.sh_silhouette else None,
        "sh_silhouette_passes": args.sh_silhouette_passes if args.sh_silhouette else None,
    }

    if args.dry_run:
        print("=== DRY RUN — configuration ===")
        print(json.dumps(config, indent=2, default=str))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_config(config, config_dir)

    print(f"Working dir : {working_dir}")
    print(f"Frames      : {len(frames)} ({frames[0]} → {frames[-1]})")
    print(f"Output dir  : {output_dir}")
    print(f"Config      : {config_path}")
    print(f"Voxel size  : {args.voxel_size} mm")
    print(f"Limits      : {args.limits} mm")
    print(f"Cameras     : {args.num_cameras}")
    print(f"Workers     : {workers}  (of {_cpu_count()} CPUs)")
    print(f"SH degree   : {args.sh_degree}" + (" (disabled)" if args.sh_degree == 0 else ""))
    if args.sh_degree > 0:
        print(f"SH min pts/coeff : {args.sh_min_points_per_coeff}")
    print(f"Size range  : " + (f"{size_range[0]}–{size_range[1]} mm (equiv. diameter)" if size_range else "no filter"))
    print()

    kwargs = dict(
        working_dir=working_dir,
        mask_dir_name=args.mask_dir,
        camera_dir_name=args.camera_dir,
        mask_template=args.mask_template,
        camera_template=args.camera_template,
        num_cameras=args.num_cameras,
        voxel_size=args.voxel_size,
        limits=args.limits,
        output_dir=output_dir,
        export_format=args.format,
        sh_config=sh_config,
        sh_adaptive=sh_adaptive,
        refine_to=args.refine_to,
        size_range=size_range,
        sh_min_points_per_coeff=args.sh_min_points_per_coeff,
        max_aspect_ratio=args.max_aspect_ratio,
        clean_mask_border=args.clean_mask_border,
        silhouette_optimize=args.sh_silhouette,
        silhouette_scale=args.sh_silhouette_scale,
        silhouette_passes=args.sh_silhouette_passes,
    )

    if workers > 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=workers, prefer="processes")(
            delayed(reconstruct_single_frame)(frame, **kwargs) for frame in frames
        )
        results = sorted(results, key=lambda r: r["frame"])
    else:
        results = []
        for frame in frames:
            print(f"[{len(results)+1}/{len(frames)}] Frame {frame} ... ", end="", flush=True)
            try:
                r = reconstruct_single_frame(frame, **kwargs)
                results.append(r)
                extra = ""
                if r.get("sh_saved"):
                    extra = f", SH RMSE={r['sh_fit_rmse_mean']:.3f}"
                print(f"OK — {r['voxel_count']} voxels, {r['bubble_count']} bubbles{extra}")
            except Exception as exc:
                print(f"FAILED — {exc}")
                results.append({
                    "frame": frame,
                    "output": None,
                    "voxel_count": 0,
                    "bubble_count": 0,
                    "completed": False,
                    "error": str(exc),
                })

    # Summary
    completed = [r for r in results if r.get("completed")]
    failed = [r for r in results if not r.get("completed")]
    print(f"\n=== Summary ===")
    print(f"Completed : {len(completed)}")
    print(f"Failed    : {len(failed)}")
    if completed:
        total_voxels = sum(r["voxel_count"] for r in completed)
        total_bubbles = sum(r["bubble_count"] for r in completed)
        sh_count = sum(1 for r in completed if r.get("sh_saved"))
        total_filtered = sum(r.get("filtered_out_of_size_range", 0) for r in completed)
        print(f"Total voxels  : {total_voxels}")
        print(f"Total bubbles : {total_bubbles}")
        print(f"SH fitted     : {sh_count} frames")
        if size_range is not None:
            print(f"Filtered out  : {total_filtered} bubbles (outside {size_range[0]}–{size_range[1]} mm)")

    summary_path = output_dir / "reconstruction_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
