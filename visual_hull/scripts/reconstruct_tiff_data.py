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
        default=1e-3,
        help="Tikhonov regularization strength for SH fitting.",
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


def _fit_sh_for_bubbles(
    voxels: np.ndarray,
    bubbles: np.ndarray,
    sh_config: SphericalHarmonicFitConfig,
) -> dict | None:
    """Fit spherical harmonics to each bubble's refined voxels.

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

    for b in range(bubbles.shape[1]):
        start = int(bubbles[0, b]) - 1  # MATLAB 1-indexed → 0-indexed
        end = int(bubbles[1, b])
        bubble_voxels = voxels[start:end, :]

        if bubble_voxels.shape[0] < 4:
            continue  # too few points for SH fit

        try:
            sh_surface = fit_spherical_harmonic_surface(
                bubble_voxels,
                config=sh_config,
                masks=None,
                cameras=None,
            )
        except Exception:
            continue

        sh_centers.append(sh_surface.center)
        sh_coefficients.append(sh_surface.coefficients)
        # store (l,m) pairs as two arrays for easy MATLAB reading
        terms = np.array(sh_surface.basis_terms, dtype=np.int32)
        sh_basis_l.append(terms[:, 0])
        sh_basis_m.append(terms[:, 1])
        sh_vertices.append(sh_surface.vertices)
        sh_faces.append(sh_surface.faces + 1)  # 0-index → 1-index (MATLAB)
        sh_fit_rmse.append(sh_surface.fit_rmse)

    if not sh_centers:
        return None

    return {
        "sh_max_degree": int(sh_config.max_degree),
        "sh_num_bubbles": len(sh_centers),
        "sh_centers": np.array(sh_centers, dtype=np.float64),
        "sh_coefficients": np.array(
            [np.pad(c, (0, max(len(c) for c in sh_coefficients) - len(c)))
             for c in sh_coefficients],
            dtype=np.float64,
        ),
        "sh_basis_l": np.array(
            [np.pad(a, (0, max(len(a) for a in sh_basis_l) - len(a)))
             for a in sh_basis_l],
            dtype=np.int32,
        ),
        "sh_basis_m": np.array(
            [np.pad(a, (0, max(len(a) for a in sh_basis_m) - len(a)))
             for a in sh_basis_m],
            dtype=np.int32,
        ),
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
    }


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
) -> dict:
    """Reconstruct a single frame and return a result summary dict."""
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

    # Load cameras
    camera_paths = [
        camera_root / camera_template.format(camera=idx)
        for idx in range(num_cameras)
    ]
    for cp in camera_paths:
        if not cp.is_file():
            raise FileNotFoundError(f"Camera file not found: {cp}")

    cameras = OpenLPTCameraSet.from_camera_files(camera_paths)

    # Run pipeline
    result = run_full_reconstruction_from_data(
        masks=masks,
        cameras=cameras,
        voxel_size=voxel_size,
        limits=limits,
        num_cameras=num_cameras,
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
    }

    # ── Spherical Harmonics (optional) ──────────────────────────────────────
    if sh_config is not None and result.bubbles.ndim == 2 and result.bubbles.shape[1] > 0:
        sh_data = _fit_sh_for_bubbles(result.voxels, result.bubbles, sh_config)
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

    # ── SH config ───────────────────────────────────────────────────────────
    sh_config: SphericalHarmonicFitConfig | None = None
    if args.sh_degree > 0:
        sh_config = SphericalHarmonicFitConfig(
            max_degree=args.sh_degree,
            regularization=args.sh_regularization,
            theta_samples=args.sh_theta_samples,
            phi_samples=args.sh_phi_samples,
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
        print(f"Total voxels  : {total_voxels}")
        print(f"Total bubbles : {total_bubbles}")
        print(f"SH fitted     : {sh_count} frames")

    summary_path = output_dir / "reconstruction_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
