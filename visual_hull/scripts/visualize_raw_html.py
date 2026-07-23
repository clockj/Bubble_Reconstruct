"""Interactive HTML visualizations of the RAW visual-hull voxel data.

Unlike visualize_smoothed_html.py (which shows the fitted SH surfaces of the
tracked/smoothed bubbles), this shows the raw refined surface voxels of every
bubble in every frame:

  1. <out>/raw_3d_all_frames.html
     3D voxel point cloud (all bubbles, colored per bubble), frame slider.

  2. <out>/raw_projection_cameras.html
     2x2 panel — raw voxels projected into each camera (colored per bubble)
     over the real binary-mask contours (gray), frame slider.

Figures are emitted as plain Plotly JSON with Plotly.js from its CDN (needs
internet when opened); no Python plotly package required.

Run:
    python scripts/visualize_raw_html.py \
        --recon-dir   <...>/Results/recon_test50 \
        --working-dir <...>/20Hz_r_b_1_lpt \
        --frames 0 49 --out <output dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"
for p in (str(SRC), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from visualize_smoothed_html import write_html, mask_contour_xy, _slider_and_buttons  # noqa: E402

MAX_POINTS_PROJ = 2500  # subsample voxels per frame for the projection view


def load_raw(recon_dir: Path, frame: int) -> dict | None:
    path = recon_dir / f"Bubble_Frame_{frame:06d}.mat"
    if not path.is_file():
        return None
    m = loadmat(str(path))
    vox = m.get("voxels")
    bub = m.get("bubbles")
    if vox is None or vox.ndim != 2 or vox.shape[0] == 0:
        return None
    return {"voxels": np.asarray(vox, dtype=np.float64),
            "bubbles": np.asarray(bub) if bub is not None else np.empty((2, 0))}


def voxel_bubble_index(bubbles: np.ndarray, n: int) -> np.ndarray:
    idx = np.full(n, 0, dtype=np.int64)
    for b in range(bubbles.shape[1]):
        s = int(bubbles[0, b]) - 1
        e = int(bubbles[1, b])
        idx[s:e] = b
    return idx


def _rlist(a: np.ndarray, dec: int) -> list:
    return np.round(np.asarray(a, dtype=float), dec).tolist()


# ════════════════════════════════════════════════════════════════════════════
# HTML 1 — 3D voxel field
# ════════════════════════════════════════════════════════════════════════════

def build_3d(frames: dict[int, dict], out_path: Path) -> None:
    order = sorted(frames)
    P = np.vstack([frames[f]["voxels"] for f in order])
    lo, hi = P.min(0), P.max(0)
    mid = 0.5 * (lo + hi)
    rng = 0.55 * float(np.max(hi - lo))
    axrange = [[float(mid[i] - rng), float(mid[i] + rng)] for i in range(3)]

    def trace(f: int) -> dict:
        v = frames[f]["voxels"]
        idx = voxel_bubble_index(frames[f]["bubbles"], v.shape[0])
        return dict(type="scatter3d", mode="markers",
                    x=_rlist(v[:, 0], 2), y=_rlist(v[:, 1], 2), z=_rlist(v[:, 2], 2),
                    marker=dict(size=1.6, color=idx.tolist(), colorscale="Turbo",
                                opacity=0.75), showlegend=False)

    sliders, updatemenus = _slider_and_buttons(order)
    fig = dict(
        data=[trace(order[0])],
        frames=[dict(name=str(f), data=[trace(f)]) for f in order],
        layout=dict(title="Raw visual-hull voxels — 3D field (colored per bubble)",
                    scene=dict(xaxis=dict(title="X (mm)", range=axrange[0]),
                               yaxis=dict(title="Y (mm)", range=axrange[1]),
                               zaxis=dict(title="Z (mm)", range=axrange[2]),
                               aspectmode="cube"),
                    sliders=sliders, updatemenus=updatemenus, height=760, width=920),
    )
    write_html(fig, out_path, "Raw voxels — 3D")
    print(f"  3D   → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# HTML 2 — camera projections
# ════════════════════════════════════════════════════════════════════════════

def build_projection(frames: dict[int, dict], working_dir: Path, out_path: Path,
                     num_cameras: int = 4) -> None:
    from visual_hull.camera import OpenLPTCameraSet
    from visual_hull.io import load_tiff_mask

    cameras = OpenLPTCameraSet.from_camera_files(
        [working_dir / "camFile_VSC" / f"vsc_cam{c}.txt" for c in range(num_cameras)])
    order = sorted(frames)
    H, W = load_tiff_mask(working_dir / "imgFile_bb" / "cam0" / f"img{order[0]:06d}.tif").shape

    axkey = ["", "2", "3", "4"]
    domains = [([0.0, 0.47], [0.55, 1.0]), ([0.53, 1.0], [0.55, 1.0]),
               ([0.0, 0.47], [0.0, 0.45]), ([0.53, 1.0], [0.0, 0.45])]
    rng = np.random.default_rng(0)

    def traces(f: int) -> list:
        v = frames[f]["voxels"]
        idx = voxel_bubble_index(frames[f]["bubbles"], v.shape[0])
        if v.shape[0] > MAX_POINTS_PROJ:
            sel = rng.choice(v.shape[0], MAX_POINTS_PROJ, replace=False)
            v = v[sel]
            idx = idx[sel]
        out = []
        for c in range(num_cameras):
            xa, ya = "x" + axkey[c], "y" + axkey[c]
            mask = load_tiff_mask(working_dir / "imgFile_bb" / f"cam{c}" / f"img{f:06d}.tif")
            mu, mv = mask_contour_xy(mask)
            out.append(dict(type="scatter", x=mu, y=mv, mode="lines", xaxis=xa, yaxis=ya,
                            line=dict(color="rgba(120,120,120,0.7)", width=1),
                            name="real mask", showlegend=(c == 0), legendgroup="mask"))
            pr = cameras.project_points(c, v)
            ok = pr.valid
            pts = pr.pixels[ok]
            out.append(dict(type="scattergl", x=_rlist(pts[:, 0], 1), y=_rlist(pts[:, 1], 1),
                            mode="markers", xaxis=xa, yaxis=ya,
                            marker=dict(size=2, color=idx[ok].tolist(), colorscale="Turbo",
                                        opacity=0.6),
                            name="raw voxels", showlegend=(c == 0), legendgroup="vox"))
        return out

    layout: dict = dict(
        title="Raw voxels projected into each camera (colored) vs real mask (gray)",
        height=1000, width=1050,
        annotations=[dict(text=f"cam{c}", x=sum(domains[c][0]) / 2, y=domains[c][1][1] + 0.02,
                          xref="paper", yref="paper", showarrow=False, font=dict(size=13))
                     for c in range(num_cameras)])
    for c in range(num_cameras):
        dx, dy = domains[c]
        layout["xaxis" + axkey[c]] = dict(domain=dx, range=[0, W], anchor="y" + axkey[c],
                                          showticklabels=False)
        layout["yaxis" + axkey[c]] = dict(domain=dy, range=[H, 0], anchor="x" + axkey[c],
                                          scaleanchor="x" + axkey[c], scaleratio=1,
                                          showticklabels=False)
    sliders, updatemenus = _slider_and_buttons(order)
    layout["sliders"] = sliders
    layout["updatemenus"] = updatemenus

    n_tr = 2 * num_cameras
    fig = dict(data=traces(order[0]),
               frames=[dict(name=str(f), data=traces(f), traces=list(range(n_tr))) for f in order],
               layout=layout)
    write_html(fig, out_path, "Raw voxels — camera projections")
    print(f"  proj → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon-dir", type=Path, required=True)
    ap.add_argument("--working-dir", type=Path, required=True)
    ap.add_argument("--frames", type=int, nargs=2, default=[0, 49])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-cameras", type=int, default=4)
    ap.add_argument("--skip-projection", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    frames = {}
    for f in range(args.frames[0], args.frames[1] + 1):
        d = load_raw(args.recon_dir, f)
        if d is not None:
            frames[f] = d
    if not frames:
        print("No raw voxel frames found.")
        return
    print(f"Loaded {len(frames)} raw frames ({min(frames)}-{max(frames)}).")

    build_3d(frames, args.out / "raw_3d_all_frames.html")
    if not args.skip_projection:
        build_projection(frames, args.working_dir, args.out / "raw_projection_cameras.html",
                         num_cameras=args.num_cameras)
    print("Done.")


if __name__ == "__main__":
    main()
