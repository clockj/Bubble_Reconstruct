"""
Visualize spherical-harmonic bubble reconstructions.

Usage
-----
    # Visualize frame 0 (saved as PNG)
    python scripts/visualize_sh_bubbles.py --frame 0

    # Visualize frames 0, 5, 10, 50, 99 interactively
    python scripts/visualize_sh_bubbles.py --frames 0 5 10 50 99 --interactive

    # Specify custom recon directory
    python scripts/visualize_sh_bubbles.py --frame 0 --recon-dir <path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_RECON_DIR = Path(
    r"X:\Shijie Zhong\Bubble Shear Project\Processed\20260710\20Hz_r_b_1_lpt\Results\recon"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize SH bubble reconstructions."
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[0, 5, 10, 50, 99],
        help="Frame numbers to visualize.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Single frame shortcut (overrides --frames).",
    )
    parser.add_argument(
        "--recon-dir",
        type=Path,
        default=DEFAULT_RECON_DIR,
        help="Directory containing Bubble_Frame_*_sh.mat files.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Show interactive 3D windows instead of saving PNGs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for PNG files.  Defaults to <recon-dir>/../viz_sh/.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI for saved figures.",
    )
    parser.add_argument(
        "--show-voxels",
        action="store_true",
        help="Overlay raw voxel points on SH surfaces.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title prefix for plots.",
    )
    return parser.parse_args()


def _squeeze(val: np.ndarray) -> np.ndarray | float | int:
    """Safely extract a MATLAB scalar/array into a Python value."""
    arr = np.asarray(val)
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return arr.flat[0]
    return arr.squeeze()


def load_sh_frame(recon_dir: Path, frame: int) -> dict | None:
    """Load SH data for a single frame.  Returns None if file missing."""
    path = recon_dir / f"Bubble_Frame_{frame:06d}_sh.mat"
    if not path.is_file():
        print(f"  SKIP: {path} not found")
        return None
    raw = loadmat(str(path))
    # Extract scalars that savemat wraps in 2D arrays
    result: dict = {}
    for key, val in raw.items():
        if key.startswith("__"):
            continue
        result[key] = np.asarray(val)
    return result


def load_voxel_frame(recon_dir: Path, frame: int) -> dict | None:
    """Load voxel data for a single frame."""
    path = recon_dir / f"Bubble_Frame_{frame:06d}.mat"
    if not path.is_file():
        return None
    raw = loadmat(str(path))
    return {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")}


def trim_padding(arr: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
    """Remove zero-padded rows from a 2D array."""
    if arr.ndim != 2:
        return arr
    valid_rows = np.any(np.abs(arr) > 1e-12, axis=1)
    return arr[valid_rows]


def plot_bubble_3d(
    ax: plt.Axes,
    vertices: np.ndarray,
    faces: np.ndarray,
    center: np.ndarray,
    color: str,
    alpha: float = 0.75,
    label: str | None = None,
) -> None:
    """Plot a single SH bubble surface on a 3D axis."""
    faces_0 = faces - 1  # MATLAB 1-indexed → 0-indexed
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tri_verts = []
    for face in faces_0:
        if np.all(face >= 0) and np.all(face < vertices.shape[0]):
            tri_verts.append(vertices[face])
    if not tri_verts:
        return
    mesh = Poly3DCollection(tri_verts, alpha=alpha, linewidths=0.1, edgecolors="none")
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)

    if label:
        # Use a proxy artist for the legend
        ax.scatter([], [], [], color=color, label=label, s=20)


def visualize_frame(
    sh_data: dict,
    voxel_data: dict | None,
    frame: int,
    output_path: Path | None,
    show_voxels: bool,
    title_prefix: str,
) -> plt.Figure:
    """Create a 3D figure for one frame's bubbles."""
    num_bubbles = int(np.asarray(sh_data["sh_num_bubbles"]).flat[0])
    ncols = min(num_bubbles, 3)
    nrows = int(np.ceil(num_bubbles / ncols))

    fig = plt.figure(figsize=(6 * ncols, 5.5 * nrows), dpi=100)
    fig.suptitle(
        f"{title_prefix}Frame {frame} — {num_bubbles} bubble(s)",
        fontsize=13,
        fontweight="bold",
    )

    colors = plt.cm.tab10(np.linspace(0, 1, max(num_bubbles, 1)))

    # Data arrays: (N, ...) — index first axis for per-bubble data
    centers = np.asarray(sh_data["sh_centers"])
    vertices_all = np.asarray(sh_data["sh_vertices"])
    faces_all = np.asarray(sh_data["sh_faces"], dtype=np.int32)
    rmse_all = np.asarray(sh_data["sh_fit_rmse"]).flatten()

    for b_idx in range(num_bubbles):
        ax = fig.add_subplot(nrows, ncols, b_idx + 1, projection="3d")

        vertices = vertices_all[b_idx]  # (V, 3)
        faces = faces_all[b_idx]        # (F, 3)
        center = centers[b_idx]         # (3,)
        rmse = float(rmse_all[b_idx])

        # SH surface
        color = colors[b_idx % len(colors)]
        plot_bubble_3d(
            ax, vertices, faces, center, color=color,
            label=f"SH surface (RMSE={rmse:.3f} mm)",
        )

        # Overlay voxels if requested
        if show_voxels and voxel_data is not None:
            bubbles_arr = voxel_data.get("bubbles")
            voxels_arr = voxel_data.get("voxels")
            if bubbles_arr is not None and voxels_arr is not None:
                start = int(bubbles_arr[0, b_idx]) - 1
                end = int(bubbles_arr[1, b_idx])
                bv = voxels_arr[start:end, :]
                ax.scatter(
                    bv[:, 0], bv[:, 1], bv[:, 2],
                    c="black", s=0.5, alpha=0.3, label="Voxels",
                )

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title(f"Bubble {b_idx+1}  |  center=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
        ax.legend(loc="upper right", fontsize=7)

        # Equal aspect ratio
        all_pts = vertices
        ranges = np.ptp(all_pts, axis=0)
        mid = np.mean(all_pts, axis=0)
        max_range = max(ranges) * 0.6
        for dim, m in enumerate(mid):
            getattr(ax, f"set_{['x','y','z'][dim]}lim")(m - max_range, m + max_range)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved → {output_path}")

    return fig


def main() -> None:
    args = parse_args()
    frames = [args.frame] if args.frame is not None else args.frames

    if args.interactive:
        matplotlib.use("qtagg")

    output_dir = args.output_dir
    if output_dir is None and not args.interactive:
        output_dir = args.recon_dir.parent / "viz_sh"

    title_prefix = args.title + " — " if args.title else ""

    if args.interactive:
        print(f"Opening interactive windows for frames: {frames}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving PNGs to {output_dir}")

    for frame in frames:
        print(f"\nFrame {frame}:")
        sh_data = load_sh_frame(args.recon_dir, frame)
        if sh_data is None:
            continue

        voxel_data = load_voxel_frame(args.recon_dir, frame) if args.show_voxels else None

        out_path = None
        if output_dir is not None:
            out_path = output_dir / f"sh_bubbles_frame_{frame:06d}.png"

        visualize_frame(
            sh_data,
            voxel_data,
            frame,
            out_path,
            show_voxels=args.show_voxels,
            title_prefix=title_prefix,
        )

    if not args.interactive:
        print(f"\nDone — {len(frames)} frames visualized in {output_dir}")
    else:
        print("\nClose figure windows to exit.")
        plt.show()


if __name__ == "__main__":
    main()
