"""Overlay visualization: raw visual hull vs SH-recommended surface.

Reads a per-frame reconstruction directory (Bubble_Frame_*.mat voxels +
Bubble_Frame_*_sh.mat SH fits) and builds two self-contained HTML views with
a frame slider:

  raw_vs_sh_3d.html         raw hull voxels (gray) + SH surfaces (colored)
  raw_vs_sh_projection.html per camera: real mask (gray) + raw voxels (gray)
                            + SH silhouette outline (colored)

So you can see, frame by frame, how the SH-recommended surface sits inside the
raw carving and how its projection matches the real masks.

Run:
    python scripts/visualize_raw_vs_sh_html.py --recon-dir <Results> \
        --working-dir <X:...20Hz_r_b_1_lpt> --frames 0 4 --out <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

SCRIPTS = Path(__file__).resolve().parent
SRC = SCRIPTS.parent / "src"
for p in (str(SCRIPTS), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from visualize_smoothed_html import (  # noqa: E402
    write_html, mask_contour_xy, _slider_and_buttons, bubble_mesh, _lst, PALETTE)
from visualize_raw_html import load_raw, voxel_bubble_index  # noqa: E402

MAX_VOX = 4000  # subsample raw voxels per frame for display


def load_sh(recon_dir: Path, frame: int) -> dict | None:
    path = recon_dir / f"Bubble_Frame_{frame:06d}_sh.mat"
    if not path.is_file():
        return None
    m = loadmat(str(path))
    if "sh_num_bubbles" not in m:
        return None
    return {k: np.asarray(v) for k, v in m.items() if not k.startswith("__")}


def _load(recon_dir: Path, f0: int, f1: int):
    frames = {}
    for f in range(f0, f1 + 1):
        raw = load_raw(recon_dir, f)
        sh = load_sh(recon_dir, f)
        if raw is not None:
            frames[f] = {"raw": raw, "sh": sh}
    return frames


# ── 3D overlay ────────────────────────────────────────────────────────────────
def build_3d(frames, out_path):
    order = sorted(frames)
    rng = np.random.default_rng(0)
    max_b = max((int(frames[f]["sh"]["sh_num_bubbles"].ravel()[0])
                 if frames[f]["sh"] is not None else 0) for f in order)

    P = np.vstack([frames[f]["raw"]["voxels"] for f in order])
    lo, hi = P.min(0), P.max(0)
    mid = 0.5 * (lo + hi)
    rr = 0.55 * float(np.max(hi - lo))
    axrange = [[float(mid[i] - rr), float(mid[i] + rr)] for i in range(3)]

    def traces(f):
        v = frames[f]["raw"]["voxels"]
        if v.shape[0] > MAX_VOX:
            v = v[rng.choice(v.shape[0], MAX_VOX, replace=False)]
        out = [dict(type="scatter3d", mode="markers",
                    x=_lst(v[:, 0], 2), y=_lst(v[:, 1], 2), z=_lst(v[:, 2], 2),
                    marker=dict(size=1.4, color="rgba(140,140,140,0.35)"),
                    name="raw hull voxels", showlegend=True, legendgroup="raw")]
        sh = frames[f]["sh"]
        nb = int(sh["sh_num_bubbles"].ravel()[0]) if sh is not None else 0
        for b in range(max_b):
            if b < nb:
                vv, fa = bubble_mesh(sh, b)
                out.append(dict(type="mesh3d", x=_lst(vv[:, 0]), y=_lst(vv[:, 1]), z=_lst(vv[:, 2]),
                                i=fa[:, 0].tolist(), j=fa[:, 1].tolist(), k=fa[:, 2].tolist(),
                                color=PALETTE[b % len(PALETTE)], opacity=0.9, flatshading=True,
                                name=f"SH bubble {b + 1}", showlegend=False))
            else:
                out.append(dict(type="mesh3d", x=[], y=[], z=[], i=[], j=[], k=[], showlegend=False))
        return out

    sliders, updatemenus = _slider_and_buttons(order)
    n_tr = 1 + max_b
    fig = dict(data=traces(order[0]),
               frames=[dict(name=str(f), data=traces(f), traces=list(range(n_tr))) for f in order],
               layout=dict(title="Raw visual hull (gray) vs SH-recommended surface (colored)",
                           scene=dict(xaxis=dict(title="X (mm)", range=axrange[0]),
                                      yaxis=dict(title="Y (mm)", range=axrange[1]),
                                      zaxis=dict(title="Z (mm)", range=axrange[2]),
                                      aspectmode="cube"),
                           sliders=sliders, updatemenus=updatemenus, height=780, width=960))
    write_html(fig, out_path, "Raw hull vs SH — 3D")
    print(f"  3D   -> {out_path}")


# ── Projection overlay ────────────────────────────────────────────────────────
def _sh_outline(verts, cameras, c):
    from scipy.spatial import ConvexHull
    pr = cameras.project_points(c, verts)
    uv = pr.pixels[pr.valid]
    if uv.shape[0] < 3:
        return [], []
    try:
        h = ConvexHull(uv)
    except Exception:
        return [], []
    poly = uv[h.vertices]
    xs = poly[:, 0].tolist() + [poly[0, 0]]
    ys = poly[:, 1].tolist() + [poly[0, 1]]
    return xs, ys


def build_projection(frames, working_dir, out_path, num_cameras=4):
    from visual_hull.camera import OpenLPTCameraSet
    from visual_hull.io import load_tiff_mask

    cams = OpenLPTCameraSet.from_camera_files(
        [working_dir / "camFile_VSC" / f"vsc_cam{c}.txt" for c in range(num_cameras)])
    order = sorted(frames)
    H, W = load_tiff_mask(working_dir / "imgFile_bb" / "cam0" / f"img{order[0]:06d}.tif").shape
    max_b = max((int(frames[f]["sh"]["sh_num_bubbles"].ravel()[0])
                 if frames[f]["sh"] is not None else 0) for f in order)
    rng = np.random.default_rng(0)
    axkey = ["", "2", "3", "4"]
    domains = [([0.0, 0.47], [0.55, 1.0]), ([0.53, 1.0], [0.55, 1.0]),
               ([0.0, 0.47], [0.0, 0.45]), ([0.53, 1.0], [0.0, 0.45])]

    def traces(f):
        raw = frames[f]["raw"]["voxels"]
        if raw.shape[0] > 2500:
            raw = raw[rng.choice(raw.shape[0], 2500, replace=False)]
        sh = frames[f]["sh"]
        nb = int(sh["sh_num_bubbles"].ravel()[0]) if sh is not None else 0
        out = []
        for c in range(num_cameras):
            xa, ya = "x" + axkey[c], "y" + axkey[c]
            mask = load_tiff_mask(working_dir / "imgFile_bb" / f"cam{c}" / f"img{f:06d}.tif")
            mu, mv = mask_contour_xy(mask)
            out.append(dict(type="scatter", x=mu, y=mv, mode="lines", xaxis=xa, yaxis=ya,
                            line=dict(color="rgba(120,120,120,0.7)", width=1),
                            name="real mask", showlegend=(c == 0), legendgroup="mask"))
            pr = cams.project_points(c, raw)
            pts = pr.pixels[pr.valid]
            out.append(dict(type="scattergl", x=np.round(pts[:, 0], 1).tolist(),
                            y=np.round(pts[:, 1], 1).tolist(), mode="markers", xaxis=xa, yaxis=ya,
                            marker=dict(size=2, color="rgba(150,150,150,0.4)"),
                            name="raw hull", showlegend=(c == 0), legendgroup="raw"))
            for b in range(max_b):
                if b < nb:
                    vv, _ = bubble_mesh(sh, b)
                    xs, ys = _sh_outline(vv, cams, c)
                else:
                    xs, ys = [], []
                out.append(dict(type="scatter", x=xs, y=ys, mode="lines", xaxis=xa, yaxis=ya,
                                line=dict(color=PALETTE[b % len(PALETTE)], width=2),
                                name=f"SH {b + 1}", showlegend=False))
        return out

    layout = dict(title="Raw hull (gray) & SH-recommended silhouette (colored) vs real mask",
                  height=1000, width=1050,
                  annotations=[dict(text=f"cam{c}", x=sum(domains[c][0]) / 2, y=domains[c][1][1] + 0.02,
                                    xref="paper", yref="paper", showarrow=False, font=dict(size=13))
                               for c in range(num_cameras)])
    for c in range(num_cameras):
        dx, dy = domains[c]
        layout["xaxis" + axkey[c]] = dict(domain=dx, range=[0, W], anchor="y" + axkey[c], showticklabels=False)
        layout["yaxis" + axkey[c]] = dict(domain=dy, range=[H, 0], anchor="x" + axkey[c],
                                          scaleanchor="x" + axkey[c], scaleratio=1, showticklabels=False)
    sliders, updatemenus = _slider_and_buttons(order)
    layout["sliders"] = sliders
    layout["updatemenus"] = updatemenus
    n_tr = num_cameras * (2 + max_b)
    fig = dict(data=traces(order[0]),
               frames=[dict(name=str(f), data=traces(f), traces=list(range(n_tr))) for f in order],
               layout=layout)
    write_html(fig, out_path, "Raw hull vs SH — projections")
    print(f"  proj -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon-dir", type=Path, required=True)
    ap.add_argument("--working-dir", type=Path, required=True)
    ap.add_argument("--frames", type=int, nargs=2, default=[0, 4])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-cameras", type=int, default=4)
    ap.add_argument("--skip-projection", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    frames = _load(args.recon_dir, args.frames[0], args.frames[1])
    if not frames:
        print("No frames found.")
        return
    n_sh = sum(1 for f in frames if frames[f]["sh"] is not None)
    print(f"Loaded {len(frames)} frames ({min(frames)}-{max(frames)}), {n_sh} with SH.")
    build_3d(frames, args.out / "raw_vs_sh_3d.html")
    if not args.skip_projection:
        build_projection(frames, args.working_dir, args.out / "raw_vs_sh_projection.html",
                         num_cameras=args.num_cameras)
    print("Done.")


if __name__ == "__main__":
    main()
