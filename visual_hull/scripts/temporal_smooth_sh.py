"""
Post-process spherical-harmonic bubble data with temporal tracking + smoothing.

Usage
-----
    # Smooth frames 0-9
    python scripts/temporal_smooth_sh.py --frames 0 9 \
        --recon-dir <path> --output-dir <path>

    # Visualize: original vs smoothed for a single frame
    python scripts/temporal_smooth_sh.py --frames 0 9 \
        --recon-dir <path> --output-dir <path> --visualize

How it works
------------
1. Load SH data for all frames
2. Hungarian matching across consecutive frames to build bubble trajectories
3. Apply temporal smoothing to SH coefficients along each trajectory
4. Enforce soft volume conservation
5. Save smoothed SH data + generate comparison visualizations
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_hull.improved.spherical_harmonics.surface import (
    _basis_terms,
    _grid_vertices_faces,
    SphericalHarmonicFitConfig,
)


DEFAULT_RECON_DIR = Path(
    r"X:\Shijie Zhong\Bubble Shear Project\Processed\20260710\20Hz_r_b_1_lpt\Results\recon_temporal"
)
DEFAULT_IMG_DIR = Path(
    r"X:\Shijie Zhong\Bubble Shear Project\Processed\20260710\20Hz_r_b_1_lpt\imgFile"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal smoothing of SH bubble reconstructions."
    )
    parser.add_argument(
        "--frames", type=int, nargs=2, default=[0, 9],
        metavar=("START", "END"),
        help="Frame range to process (inclusive).",
    )
    parser.add_argument(
        "--recon-dir", type=Path, default=DEFAULT_RECON_DIR,
        help="Directory containing Bubble_Frame_*_sh.mat and .mat files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for smoothed data. Defaults to <recon-dir>/../recon_temporal_smoothed/.",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate comparison PNGs (original vs smoothed).",
    )
    parser.add_argument(
        "--viz-dir", type=Path, default=None,
        help="Visualization output directory.",
    )
    parser.add_argument(
        "--match-dist-threshold", type=float, default=15.0,
        help="Max centroid distance (mm) for nearest-neighbour matching across frames.",
    )
    parser.add_argument(
        "--max-diameter-ratio", type=float, default=1.5,
        help="Max frame-to-frame diameter ratio for a valid match "
             "(reject if sizes differ by more than this factor).",
    )
    parser.add_argument(
        "--smooth-sigma", type=float, default=1.5,
        help="Gaussian smoothing sigma in frames. Larger = smoother.",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
    )
    parser.add_argument(
        "--format", choices=["png", "html"], default="png",
        help="Output format: png (static) or html (interactive plotly).",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_sh_frame(recon_dir: Path, frame: int) -> dict | None:
    path = recon_dir / f"Bubble_Frame_{frame:06d}_sh.mat"
    if not path.is_file():
        return None
    raw = loadmat(str(path))
    return {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")}


def load_voxel_frame(recon_dir: Path, frame: int) -> dict | None:
    path = recon_dir / f"Bubble_Frame_{frame:06d}.mat"
    if not path.is_file():
        return None
    raw = loadmat(str(path))
    return {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")}


def bubble_features(sh_data: dict) -> np.ndarray:
    """Extract (centroid_x, centroid_y, centroid_z, volume_approx) per bubble.

    Size comes from the mean radius of the actual SH mesh vertices, not the
    ``c_00`` coefficient — the c_00 formula is unreliable for inscribed /
    silhouette-optimized coefficients (it can be off by 10x).
    """
    centers = np.asarray(sh_data["sh_centers"])       # (B, 3)
    vertices = np.asarray(sh_data["sh_vertices"])      # (B, V, 3), zero-padded
    volumes = np.zeros(centers.shape[0], dtype=np.float64)
    for b in range(centers.shape[0]):
        v = vertices[b]
        v = v[np.any(v != 0.0, axis=1)]
        if v.shape[0] == 0:
            continue
        mean_radius = float(np.linalg.norm(v - v.mean(axis=0), axis=1).mean())
        volumes[b] = (4.0 / 3.0) * np.pi * mean_radius ** 3
    return np.column_stack((centers, volumes))


# ═══════════════════════════════════════════════════════════════════════════════
# Temporal matching (Hungarian algorithm)
# ═══════════════════════════════════════════════════════════════════════════════


def _diameter_from_volume(volume: float) -> float:
    return float((6.0 * max(volume, 1e-12) / np.pi) ** (1.0 / 3.0))


def match_bubbles(
    feats_t: np.ndarray,
    feats_tp1: np.ndarray,
    dist_threshold: float,
    max_diam_ratio: float = 1.5,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Match bubbles between frame t and t+1.

    Nearest-neighbour association gated by two physical constraints:
      * centroid distance <= ``dist_threshold`` (a bubble can't teleport)
      * diameter ratio <= ``max_diam_ratio`` (size can't jump between frames)
    The cost is pure centroid distance, so among candidates passing both
    gates the nearest one wins.  This prevents the identity-switches that a
    loose distance-only gate produced (matching across 20+ mm jumps to a
    differently-sized bubble).

    Returns (matched_pairs, unmatched_t, unmatched_tp1).
    """
    n_t = feats_t.shape[0]
    n_tp1 = feats_tp1.shape[0]

    if n_t == 0 or n_tp1 == 0:
        return [], set(range(n_t)), set(range(n_tp1))

    big = 1e9
    cost = np.full((max(n_t, n_tp1), max(n_t, n_tp1)), big)
    for i in range(n_t):
        d_i = _diameter_from_volume(feats_t[i, 3])
        for j in range(n_tp1):
            d_center = float(np.linalg.norm(feats_t[i, :3] - feats_tp1[j, :3]))
            if d_center > dist_threshold:
                continue  # nearest-neighbour distance gate
            d_j = _diameter_from_volume(feats_tp1[j, 3])
            ratio = max(d_i, d_j) / max(min(d_i, d_j), 1e-9)
            if ratio > max_diam_ratio:
                continue  # diameter-similarity gate
            cost[i, j] = d_center  # pure nearest neighbour

    row_ind, col_ind = linear_sum_assignment(cost)

    matched: list[tuple[int, int]] = []
    unmatched_t: set[int] = set(range(n_t))
    unmatched_tp1: set[int] = set(range(n_tp1))

    for r, c in zip(row_ind, col_ind):
        if r < n_t and c < n_tp1 and cost[r, c] < big:  # passed both gates
            matched.append((r, c))
            unmatched_t.discard(r)
            unmatched_tp1.discard(c)

    return matched, unmatched_t, unmatched_tp1


def build_trajectories(
    all_features: dict[int, np.ndarray],
    frames: list[int],
    dist_threshold: float,
    max_diam_ratio: float = 1.5,
) -> list[dict]:
    """Build bubble trajectories across all frames.

    Returns list of trajectories. Each trajectory is a dict:
      {frame: bubble_index, ...} mapping frame → local bubble index.
    """
    present = [f for f in frames if all_features.get(f) is not None]
    completed: list[dict] = []
    active: dict[int, dict] = {}  # bubble index at the current frame -> trajectory

    for k in range(len(present) - 1):
        f_t, f_tp1 = present[k], present[k + 1]
        matched, _, _ = match_bubbles(
            all_features[f_t], all_features[f_tp1], dist_threshold, max_diam_ratio,
        )

        new_active: dict[int, dict] = {}
        matched_from: set[int] = set()
        for a, c in matched:
            matched_from.add(a)
            traj = active.pop(a, None)
            if traj is None:
                traj = {f_t: a}  # seed a new trajectory at this match
            traj[f_tp1] = c
            new_active[c] = traj

        # Trajectories whose bubble did not match end here.
        completed.extend(active.values())
        active = new_active

    completed.extend(active.values())

    # Cover every bubble: bubbles never placed in a trajectory become singletons.
    covered: dict[int, set[int]] = {}
    for traj in completed:
        for f, b in traj.items():
            covered.setdefault(f, set()).add(b)
    for f in present:
        seen = covered.get(f, set())
        for b in range(all_features[f].shape[0]):
            if b not in seen:
                completed.append({f: b})

    return completed


# ═══════════════════════════════════════════════════════════════════════════════
# Temporal smoothing
# ═══════════════════════════════════════════════════════════════════════════════


def _gaussian_kernel(sigma: float, radius: int = 3) -> np.ndarray:
    """1D Gaussian kernel truncated at *radius* sigmas."""
    r = max(1, int(np.ceil(radius * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def smooth_coefficients(
    coeffs_sequence: list[np.ndarray],
    sigma: float,
) -> list[np.ndarray]:
    """Apply Gaussian-weighted temporal smoothing to a sequence of coefficient vectors."""
    kernel = _gaussian_kernel(sigma)
    half = len(kernel) // 2
    n = len(coeffs_sequence)
    k = coeffs_sequence[0].shape[0]
    padded = [coeffs_sequence[0]] * half + coeffs_sequence + [coeffs_sequence[-1]] * half
    smoothed: list[np.ndarray] = []
    for i in range(n):
        window = np.stack(padded[i:i + len(kernel)], axis=-1)  # (K, W)
        smoothed.append(window @ kernel)
    return smoothed


def apply_temporal_smoothing(
    trajectories: list[dict],
    all_sh: dict[int, dict],
    sigma: float,
) -> dict[int, dict]:
    """Create smoothed SH data for all frames.

    For each trajectory, smooth the coefficient sequence and reconstruct
    the per-frame SH data with smoothed coefficients.
    """
    # Build per-frame smoothed coefficient accumulators
    frame_data: dict[int, dict] = defaultdict(lambda: {
        "sh_centers": [],
        "sh_coefficients": [],
        "sh_basis_l": [],
        "sh_basis_m": [],
        "sh_vertices": [],
        "sh_faces": [],
        "sh_fit_rmse": [],
        "sh_degree_used": [],
        "sh_track_id": [],
        "sh_max_degree": 0,
    })

    for traj_id, traj in enumerate(trajectories):
        frames_in_traj = sorted(traj.keys())

        if len(traj) < 2:
            # Untracked (single-frame) bubble: pass through unsmoothed so it
            # still appears in the output instead of being dropped.
            f = frames_in_traj[0]
            b = traj[f]
            sh = all_sh.get(f)
            if sh is None:
                continue
            maxd = int(np.asarray(sh["sh_max_degree"]).flat[0])
            fd = frame_data[f]
            fd["sh_centers"].append(np.asarray(sh["sh_centers"][b]))
            fd["sh_coefficients"].append(np.asarray(sh["sh_coefficients"][b]))
            fd["sh_basis_l"].append(np.array([t[0] for t in _basis_terms(maxd)], dtype=np.int32))
            fd["sh_basis_m"].append(np.array([t[1] for t in _basis_terms(maxd)], dtype=np.int32))
            fd["sh_vertices"].append(np.asarray(sh["sh_vertices"][b]))
            fd["sh_faces"].append(np.asarray(sh["sh_faces"][b]))
            fd["sh_fit_rmse"].append(float(np.asarray(sh["sh_fit_rmse"]).flatten()[b]))
            fd["sh_degree_used"].append(int(np.asarray(sh["sh_degree_used"]).flatten()[b]))
            fd["sh_track_id"].append(traj_id)
            fd["sh_max_degree"] = max(fd["sh_max_degree"], maxd)
            continue

        max_degree = 0

        # Collect coefficient sequences
        coeff_seq: list[np.ndarray] = []
        centers_seq: list[np.ndarray] = []
        basis_l_seq: list[np.ndarray] = []
        basis_m_seq: list[np.ndarray] = []
        deg_seq: list[int] = []

        for f in frames_in_traj:
            sh = all_sh.get(f)
            if sh is None:
                continue
            b = traj[f]
            coeff_seq.append(np.asarray(sh["sh_coefficients"][b]))
            centers_seq.append(np.asarray(sh["sh_centers"][b]))
            basis_l_seq.append(np.asarray(sh["sh_basis_l"][b]))
            basis_m_seq.append(np.asarray(sh["sh_basis_m"][b]))
            deg_seq.append(int(np.asarray(sh["sh_degree_used"]).flatten()[b]))
            max_degree = max(max_degree, int(np.asarray(sh["sh_max_degree"]).flat[0]))

        if len(coeff_seq) < 2:
            continue

        # Smooth coefficients
        smoothed_coeffs = smooth_coefficients(coeff_seq, sigma)
        # Smooth centers similarly
        smoothed_centers = smooth_coefficients(centers_seq, sigma)

        # Enforce volume conservation: re-scale to mean volume
        c00_orig = np.array([c[0] for c in coeff_seq])
        c00_smooth = np.array([c[0] for c in smoothed_coeffs])
        mean_c00 = np.mean(np.abs(c00_orig))  # preserve mean volume
        for sc in smoothed_coeffs:
            if abs(sc[0]) > 1e-6:
                sc[:] = sc * (mean_c00 / abs(sc[0]))

        # Re-generate vertices for each frame
        for idx, f in enumerate(frames_in_traj):
            sc = smoothed_coeffs[idx]
            center = smoothed_centers[idx]
            deg = deg_seq[idx]

            config = SphericalHarmonicFitConfig(
                max_degree=deg,
                theta_samples=40,
                phi_samples=80,
            )
            # Truncate coefficients to match this bubble's degree
            n_coeffs_deg = (deg + 1) ** 2
            sc_deg = sc[:n_coeffs_deg]
            terms_deg = _basis_terms(deg)
            vertices, faces = _grid_vertices_faces(center, sc_deg, terms_deg, config)

            fd = frame_data[f]
            fd["sh_centers"].append(center)
            fd["sh_coefficients"].append(sc)
            fd["sh_basis_l"].append(np.array([t[0] for t in _basis_terms(max_degree)], dtype=np.int32))
            fd["sh_basis_m"].append(np.array([t[1] for t in _basis_terms(max_degree)], dtype=np.int32))
            fd["sh_vertices"].append(vertices)
            fd["sh_faces"].append(faces + 1)
            fd["sh_fit_rmse"].append(0.0)  # smoothed — no direct RMSE
            fd["sh_degree_used"].append(deg)
            fd["sh_track_id"].append(traj_id)
            fd["sh_max_degree"] = max(fd["sh_max_degree"], max_degree)

    # Pad and stack per frame
    result: dict[int, dict] = {}
    for f, fd in frame_data.items():
        if not fd["sh_centers"]:
            continue
        max_d = fd.pop("sh_max_degree")
        n_full = (max_d + 1) ** 2
        result[f] = {
            "sh_max_degree": np.array([[max_d]]),
            "sh_num_bubbles": np.array([[len(fd["sh_centers"])]]),
            "sh_centers": np.array(fd["sh_centers"], dtype=np.float64),
            "sh_coefficients": np.array([
                np.pad(c, (0, n_full - len(c))) for c in fd["sh_coefficients"]
            ], dtype=np.float64),
            "sh_basis_l": np.array(fd["sh_basis_l"], dtype=np.int32),
            "sh_basis_m": np.array(fd["sh_basis_m"], dtype=np.int32),
            "sh_vertices": np.array([
                np.pad(v, ((0, max(vv.shape[0] for vv in fd["sh_vertices"]) - v.shape[0]), (0, 0)))
                for v in fd["sh_vertices"]
            ], dtype=np.float64),
            "sh_faces": np.array([
                np.pad(fa, ((0, max(ff.shape[0] for ff in fd["sh_faces"]) - fa.shape[0]), (0, 0)))
                for fa in fd["sh_faces"]
            ], dtype=np.int32),
            "sh_fit_rmse": np.array([fd["sh_fit_rmse"]]),
            "sh_degree_used": np.array([fd["sh_degree_used"]], dtype=np.int32),
            "sh_track_id": np.array([fd["sh_track_id"]], dtype=np.int32),
        }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════


def _get_bubble_data(sh_data: dict, b_idx: int) -> tuple:
    vertices = np.asarray(sh_data["sh_vertices"][b_idx])
    faces = np.asarray(sh_data["sh_faces"][b_idx], dtype=np.int32) - 1
    center = np.asarray(sh_data["sh_centers"][b_idx]).flatten()
    return vertices, faces, center


def plot_comparison_3d(
    orig_sh: dict,
    smooth_sh: dict,
    voxel_data: dict | None,
    frame: int,
    output_path: Path,
    img_dir: Path | None,
) -> None:
    """Side-by-side 3D comparison: original SH vs smoothed SH, with voxel overlay."""
    n_orig = int(np.asarray(orig_sh["sh_num_bubbles"]).flat[0])
    n_smooth = int(np.asarray(smooth_sh["sh_num_bubbles"]).flat[0])
    n_bubbles = max(n_orig, n_smooth)

    ncols = min(n_bubbles, 3)
    nrows = int(np.ceil(n_bubbles / ncols)) * 2  # 2 rows: orig + smooth

    fig = plt.figure(figsize=(6 * ncols, 5.5 * nrows), dpi=100)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_bubbles, 1)))

    for b_idx in range(n_bubbles):
        # ── Original SH ──
        ax_orig = fig.add_subplot(nrows, ncols, b_idx + 1, projection="3d")
        if b_idx < n_orig:
            verts, faces, center = _get_bubble_data(orig_sh, b_idx)
            _plot_sh_surface(ax_orig, verts, faces, colors[b_idx], alpha=0.5, label="Original SH")
            ax_orig.set_title(f"B{b_idx+1} orig | c=({center[0]:.0f},{center[1]:.0f},{center[2]:.0f})", fontsize=7)

        # ── Smoothed SH ──
        ax_smooth = fig.add_subplot(nrows, ncols, ncols + b_idx + 1, projection="3d")
        if b_idx < n_smooth:
            verts, faces, center = _get_bubble_data(smooth_sh, b_idx)
            _plot_sh_surface(ax_smooth, verts, faces, colors[b_idx], alpha=0.8, label="Smoothed SH")
            ax_smooth.set_title(f"B{b_idx+1} smooth", fontsize=7)

        # Voxel overlay on both
        if voxel_data is not None:
            bubbles_arr = voxel_data.get("bubbles")
            voxels_arr = voxel_data.get("voxels")
            if bubbles_arr is not None and voxels_arr is not None and b_idx < bubbles_arr.shape[1]:
                start = int(bubbles_arr[0, b_idx]) - 1
                end = int(bubbles_arr[1, b_idx])
                bv = voxels_arr[start:end, :]
                for ax in (ax_orig, ax_smooth):
                    ax.scatter(bv[:, 0], bv[:, 1], bv[:, 2], c="black", s=0.3, alpha=0.2)

        for ax in (ax_orig, ax_smooth):
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            _equal_aspect_3d(ax, verts if b_idx < n_orig else None)

    fig.suptitle(f"Frame {frame} — Original vs Temporally Smoothed SH", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_sh_surface(ax, vertices, faces, color, alpha, label):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    faces_0 = np.clip(faces, 0, vertices.shape[0] - 1)
    mask = np.all(faces_0 >= 0, axis=1) & np.all(faces_0 < vertices.shape[0], axis=1)
    tris = [vertices[f] for f in faces_0[mask]]
    if tris:
        mesh = Poly3DCollection(tris, alpha=alpha, linewidths=0.05, edgecolors="none")
        mesh.set_facecolor(color)
        ax.add_collection3d(mesh)
    if label:
        ax.scatter([], [], [], color=color, label=label, s=10)


def _equal_aspect_3d(ax, pts):
    if pts is None:
        return
    ranges = np.ptp(pts, axis=0)
    mid = np.mean(pts, axis=0)
    r = max(ranges) * 0.6
    if r < 1:
        r = 5.0
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)


def plot_2d_projections(
    orig_sh: dict,
    smooth_sh: dict,
    voxel_data: dict | None,
    frame: int,
    working_dir: Path,
    output_path: Path,
) -> None:
    """Plot 2D camera projections of original vs smoothed SH surfaces.

    Shows all 4 camera views overlaid on the original images.
    """
    from visual_hull.camera import OpenLPTCameraSet
    from visual_hull.io import load_tiff_mask

    # Load cameras
    camera_dir = working_dir / "camFile_VSC"
    camera_paths = sorted(camera_dir.glob("vsc_cam*.txt"))
    if len(camera_paths) < 4:
        print("  WARNING: camera files not found, skipping 2D projections")
        return
    cameras = OpenLPTCameraSet.from_camera_files(camera_paths[:4])

    # Load images
    img_dir = working_dir / "imgFile" / "cam0"
    sample_img_path = sorted(img_dir.glob("*.tif*"))[0] if list(img_dir.glob("*.tif*")) else None
    if sample_img_path is None:
        img_dir = working_dir / "imgFile_bb" / "cam0"
        sample_img_path = sorted(img_dir.glob("*.tif*"))[0] if list(img_dir.glob("*.tif*")) else None

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=100)

    n_orig = int(np.asarray(orig_sh["sh_num_bubbles"]).flat[0])
    n_smooth = int(np.asarray(smooth_sh["sh_num_bubbles"]).flat[0])

    # Load images for this frame
    images = []
    for cam_idx in range(4):
        img_path = working_dir / "imgFile" / f"cam{cam_idx}" / f"cam{cam_idx}frame{frame:06d}.tif"
        if not img_path.is_file():
            img_path = working_dir / "imgFile_bb" / f"cam{cam_idx}" / f"img{frame:06d}.tif"
        if img_path.is_file():
            img = plt.imread(str(img_path))
            if img.ndim == 3 and img.shape[2] >= 3:
                img = img[:, :, 0]  # use first channel for grayscale
            images.append(img)
        else:
            images.append(np.zeros((100, 100)))

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_orig, n_smooth, 1)))

    # ── Row 1: Original SH projections ──
    for cam_idx in range(4):
        ax = axes[0, cam_idx]
        ax.imshow(images[cam_idx], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Cam {cam_idx} — Original SH", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        for b_idx in range(n_orig):
            verts = np.asarray(orig_sh["sh_vertices"][b_idx])
            _draw_projected_outline(ax, verts, cameras, cam_idx, colors[b_idx], "orig")

    # ── Row 2: Smoothed SH projections ──
    for cam_idx in range(4):
        ax = axes[1, cam_idx]
        ax.imshow(images[cam_idx], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Cam {cam_idx} — Smoothed SH", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        for b_idx in range(n_smooth):
            verts = np.asarray(smooth_sh["sh_vertices"][b_idx])
            _draw_projected_outline(ax, verts, cameras, cam_idx, colors[b_idx], "smooth")

        # Voxel overlay on smoothed row
        if voxel_data is not None and voxel_data.get("voxels") is not None:
            voxels_arr = np.asarray(voxel_data["voxels"])
            if voxels_arr.shape[0] > 0:
                proj = cameras.project_points(cam_idx, voxels_arr[:5000])
                valid = proj.valid
                if np.any(valid):
                    ax.scatter(proj.pixels[valid, 0], proj.pixels[valid, 1],
                               c="red", s=0.2, alpha=0.3)

    fig.suptitle(f"Frame {frame} — 2D Camera Projections", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_comparison_html(
    orig_sh: dict,
    smooth_sh: dict,
    voxel_data: dict | None,
    frame: int,
    output_path: Path,
) -> None:
    """Create interactive plotly HTML comparing original vs smoothed SH."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_orig = int(np.asarray(orig_sh["sh_num_bubbles"]).flat[0])
    n_smooth = int(np.asarray(smooth_sh["sh_num_bubbles"]).flat[0])
    n_bubbles = max(n_orig, n_smooth)

    # Two rows: top = original, bottom = smoothed
    ncols = min(n_bubbles, 3)
    nrows = n_bubbles * 2 // ncols + (1 if (n_bubbles * 2) % ncols else 0)

    specs = [[{"type": "scene"} for _ in range(ncols)] for _ in range(nrows)]
    titles = []
    for b in range(n_bubbles):
        titles.append(f"B{b+1} Original" if b < n_orig else "")
    for b in range(n_bubbles):
        titles.append(f"B{b+1} Smoothed" if b < n_smooth else "")

    fig = make_subplots(rows=nrows, cols=ncols, specs=specs, subplot_titles=titles)

    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b",
              "#e377c2","#7f7f7f","#bcbd22","#17becf","#aec7e8","#ffbb78",
              "#98df8a","#ff9896"]

    trace_idx = 0
    for b_idx in range(n_bubbles):
        color = colors[b_idx % len(colors)]

        # ── Original SH ──
        if b_idx < n_orig:
            row = (b_idx) // ncols + 1
            col = (b_idx) % ncols + 1
            verts = np.asarray(orig_sh["sh_vertices"][b_idx])
            faces = np.asarray(orig_sh["sh_faces"][b_idx], dtype=np.int32) - 1
            mask = np.all(faces >= 0, axis=1) & np.all(faces < verts.shape[0], axis=1)
            if np.any(mask) and verts.shape[0] > 0:
                fig.add_trace(go.Mesh3d(
                    x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                    i=faces[mask, 0], j=faces[mask, 1], k=faces[mask, 2],
                    color=color, opacity=0.55, name=f"B{b_idx+1} orig",
                    showlegend=True,
                ), row=row, col=col)
                trace_idx += 1

        # ── Smoothed SH ──
        if b_idx < n_smooth:
            row = (n_bubbles + b_idx) // ncols + 1
            col = (n_bubbles + b_idx) % ncols + 1
            verts = np.asarray(smooth_sh["sh_vertices"][b_idx])
            faces = np.asarray(smooth_sh["sh_faces"][b_idx], dtype=np.int32) - 1
            mask = np.all(faces >= 0, axis=1) & np.all(faces < verts.shape[0], axis=1)
            if np.any(mask) and verts.shape[0] > 0:
                fig.add_trace(go.Mesh3d(
                    x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                    i=faces[mask, 0], j=faces[mask, 1], k=faces[mask, 2],
                    color=color, opacity=0.8, name=f"B{b_idx+1} smooth",
                    showlegend=True,
                ), row=row, col=col)
                trace_idx += 1

    fig.update_layout(
        title=dict(text=f"Frame {frame} — Original (top) vs Smoothed (bottom)", font=dict(size=14)),
        height=400 * nrows // 2,
        width=500 * ncols,
    )
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)


def _draw_projected_outline(ax, vertices, cameras, cam_idx, color, label):
    """Project mesh vertices and draw the convex hull outline."""
    try:
        proj = cameras.project_points(cam_idx, vertices)
    except Exception:
        return
    if not np.any(proj.valid):
        return
    pts = proj.pixels[proj.valid]
    if pts.shape[0] < 3:
        return
    # Convex hull of projected points
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        for simplex in hull.simplices:
            ax.plot(pts[simplex, 0], pts[simplex, 1], color=color, linewidth=0.8, alpha=0.7)
    except Exception:
        ax.scatter(pts[::10, 0], pts[::10, 1], color=color, s=0.5, alpha=0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()
    frames = list(range(args.frames[0], args.frames[1] + 1))
    recon_dir = args.recon_dir
    output_dir = args.output_dir or (recon_dir.parent / "recon_temporal_smoothed")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SH data for {len(frames)} frames ...")
    all_sh: dict[int, dict] = {}
    all_features: dict[int, np.ndarray] = {}
    for f in frames:
        sh = load_sh_frame(recon_dir, f)
        if sh is not None and int(np.asarray(sh["sh_num_bubbles"]).flat[0]) > 0:
            all_sh[f] = sh
            all_features[f] = bubble_features(sh)

    if not all_sh:
        print("No SH data found — run reconstruction with --sh-degree first.")
        return

    print(f"  Loaded {len(all_sh)} frames with SH data")

    # ── Matching ────────────────────────────────────────────────────────────
    print(f"Matching bubbles across frames (dist<={args.match_dist_threshold} mm, "
          f"diam ratio<={args.max_diameter_ratio}) ...")
    trajectories = build_trajectories(
        all_features, frames, args.match_dist_threshold, args.max_diameter_ratio
    )
    traj_lengths = [len(t) for t in trajectories]
    print(f"  Found {len(trajectories)} trajectories")
    print(f"  Lengths: min={min(traj_lengths) if traj_lengths else 0}, "
          f"max={max(traj_lengths) if traj_lengths else 0}, "
          f"mean={np.mean(traj_lengths) if traj_lengths else 0:.1f} frames")

    # ── Smoothing ───────────────────────────────────────────────────────────
    print(f"Smoothing coefficients (sigma={args.smooth_sigma} frames) ...")
    smoothed = apply_temporal_smoothing(trajectories, all_sh, args.smooth_sigma)
    print(f"  Smoothed {len(smoothed)} frames")

    # ── Save ────────────────────────────────────────────────────────────────
    for f, sd in smoothed.items():
        path = output_dir / f"Bubble_Frame_{f:06d}_sh_smoothed.mat"
        savemat(str(path), {k: v for k, v in sd.items()})
    print(f"Saved smoothed SH data to {output_dir}")

    # ── Summary stats ───────────────────────────────────────────────────────
    print(f"\n=== Temporal Smoothing Summary ===")
    print(f"Frames processed     : {len(frames)}")
    print(f"Frames with SH data  : {len(all_sh)}")
    print(f"Trajectories found   : {len(trajectories)}")
    print(f"Traj mean length     : {np.mean(traj_lengths):.1f} frames")
    print(f"Smoothed frames saved: {len(smoothed)}")

    # ── Visualize ───────────────────────────────────────────────────────────
    if args.visualize:
        viz_dir = args.viz_dir or (output_dir.parent / "viz_temporal")
        viz_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating comparison visualizations ...")
        matplotlib.use("agg")

        for f in frames[:6]:  # first 6 frames
            if f not in all_sh or f not in smoothed:
                continue
            vox = load_voxel_frame(recon_dir, f)
            ext = "html" if args.format == "html" else "png"

            if args.format == "html":
                # Interactive HTML comparison
                out_path = viz_dir / f"compare_frame_{f:06d}.html"
                plot_comparison_html(all_sh[f], smoothed[f], vox, f, out_path)
                print(f"  Frame {f} HTML → {out_path}")
            else:
                # 3D comparison PNG
                out_path_3d = viz_dir / f"compare_frame_{f:06d}.png"
                plot_comparison_3d(all_sh[f], smoothed[f], vox, f, out_path_3d, None)
                print(f"  Frame {f} 3D → {out_path_3d}")
                # 2D projections
                working_dir = recon_dir.parents[1]
                out_path_2d = viz_dir / f"projection_frame_{f:06d}.png"
                try:
                    plot_2d_projections(all_sh[f], smoothed[f], vox, f, working_dir, out_path_2d)
                    print(f"  Frame {f} 2D → {out_path_2d}")
                except Exception as e:
                    print(f"  Frame {f} 2D SKIP: {e}")

        print(f"Visualizations saved to {viz_dir}")


if __name__ == "__main__":
    main()
