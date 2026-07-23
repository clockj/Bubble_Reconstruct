"""Interactive HTML visualizations of the temporally-smoothed SH bubbles.

Produces two self-contained HTML files, each with a frame slider spanning
all available smoothed frames:

  1. <out>/smoothed_3d_all_frames.html
     3D bubble field (all smoothed SH surfaces in world coordinates).

  2. <out>/smoothed_projection_cameras.html
     2x2 panel — the smoothed SH silhouette outlines projected into each
     camera (red), overlaid on the real binary-mask contours (gray), in
     pixel coordinates.  Lets you check alignment against the data.

The figures are emitted as plain Plotly JSON; Plotly.js is loaded from its
CDN (needs internet when the HTML is opened).  No Python plotly package is
required, so the conda environment is left untouched.

Run:
    python scripts/visualize_smoothed_html.py \
        --smoothed-dir <...>/Results/recon_test50_smoothed \
        --working-dir  <...>/20Hz_r_b_1_lpt \
        --out <output dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.spatial import ConvexHull

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from visual_hull.improved.spherical_harmonics.surface import (  # noqa: E402
    _basis_terms, _grid_vertices_faces, SphericalHarmonicFitConfig)

# Display mesh resolution (independent of the stored fit mesh; keeps files light).
MESH_THETA = 24
MESH_PHI = 48

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
           "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78"]


def track_color(track_id: int) -> str:
    """Stable, well-separated color per track id (golden-ratio hue spacing)."""
    import colorsys
    h = (int(track_id) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.62, 0.92)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _lst(a: np.ndarray, decimals: int = 3) -> list:
    return np.round(np.asarray(a, dtype=float), decimals).tolist()


def write_html(fig: dict, path: Path, title: str) -> None:
    fig_js = json.dumps(fig, allow_nan=True)  # NaN is a valid JS literal
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<script src="{CDN}"></script>
<style>body{{margin:0;font-family:sans-serif}}#g{{width:100%;}}</style>
</head><body><div id="g"></div>
<script>
var fig = {fig_js};
Plotly.newPlot('g', fig.data, fig.layout, {{responsive:true}})
      .then(function(){{ if (fig.frames) Plotly.addFrames('g', fig.frames); }});
</script></body></html>"""
    path.write_text(html, encoding="utf-8")


def frame_of(p: str) -> int:
    return int(os.path.basename(p).split("_")[2])


def load_smoothed(smoothed_dir: Path) -> dict[int, dict]:
    files = sorted(glob.glob(str(smoothed_dir / "*_sh_smoothed.mat")))
    out: dict[int, dict] = {}
    for f in files:
        m = loadmat(f)
        out[frame_of(f)] = {k: np.asarray(v) for k, v in m.items() if not k.startswith("__")}
    return out


def bubble_mesh(sh: dict, b: int, ntheta: int = MESH_THETA,
                nphi: int = MESH_PHI) -> tuple[np.ndarray, np.ndarray]:
    """(vertices, faces0) for bubble b, regenerated from its SH coefficients
    at a light display resolution (avoids the stored 3042-vertex meshes)."""
    deg = int(np.asarray(sh["sh_degree_used"]).ravel()[b])
    center = np.asarray(sh["sh_centers"][b], dtype=np.float64).ravel()
    coeffs = np.asarray(sh["sh_coefficients"][b], dtype=np.float64)[:(deg + 1) ** 2]
    terms = _basis_terms(deg)
    cfg = SphericalHarmonicFitConfig(max_degree=deg, theta_samples=ntheta, phi_samples=nphi)
    verts, faces = _grid_vertices_faces(center, coeffs, terms, cfg)
    return verts, np.asarray(faces, dtype=np.int64)


def _slider_and_buttons(order: list[int], redraw: bool = True) -> tuple[list, list]:
    steps = [dict(method="animate", label=str(f),
                  args=[[str(f)], dict(mode="immediate",
                                       frame=dict(duration=0, redraw=redraw),
                                       transition=dict(duration=0))])
             for f in order]
    sliders = [dict(active=0, currentvalue=dict(prefix="Frame "), pad=dict(t=50), steps=steps)]
    updatemenus = [dict(type="buttons", showactive=False, x=0.05, y=0,
                        xanchor="right", yanchor="top", buttons=[
                            dict(label="Play", method="animate",
                                 args=[None, dict(frame=dict(duration=250, redraw=redraw),
                                                  fromcurrent=True)]),
                            dict(label="Pause", method="animate",
                                 args=[[None], dict(mode="immediate",
                                                    frame=dict(duration=0, redraw=False))])])]
    return sliders, updatemenus


# ════════════════════════════════════════════════════════════════════════════
# HTML 1 — 3D field
# ════════════════════════════════════════════════════════════════════════════

def build_3d(frames: dict[int, dict], out_path: Path) -> None:
    order = sorted(frames)
    max_b = max(int(frames[f]["sh_num_bubbles"].ravel()[0]) for f in order)

    allv = []
    for f in order:
        nb = int(frames[f]["sh_num_bubbles"].ravel()[0])
        for b in range(nb):
            v, _ = bubble_mesh(frames[f], b)
            allv.append(v)
    P = np.vstack(allv)
    lo, hi = P.min(0), P.max(0)
    mid = 0.5 * (lo + hi)
    rng = 0.55 * float(np.max(hi - lo))
    axrange = [[float(mid[i] - rng), float(mid[i] + rng)] for i in range(3)]

    def traces(f: int) -> list:
        sh = frames[f]
        nb = int(sh["sh_num_bubbles"].ravel()[0])
        tids = sh.get("sh_track_id")
        tids = tids.ravel() if tids is not None else None
        out = []
        for b in range(max_b):
            if b < nb:
                v, fa = bubble_mesh(sh, b)
                if tids is not None:
                    tid = int(tids[b])
                    color, name = track_color(tid), f"track {tid}"
                else:
                    color, name = PALETTE[b % len(PALETTE)], f"bubble {b + 1}"
                out.append(dict(type="mesh3d", x=_lst(v[:, 0]), y=_lst(v[:, 1]), z=_lst(v[:, 2]),
                                i=fa[:, 0].tolist(), j=fa[:, 1].tolist(), k=fa[:, 2].tolist(),
                                color=color, opacity=0.85, flatshading=True,
                                name=name, hovertext=name, showlegend=False))
            else:
                out.append(dict(type="mesh3d", x=[], y=[], z=[], i=[], j=[], k=[], showlegend=False))
        return out

    sliders, updatemenus = _slider_and_buttons(order)
    fig = dict(
        data=traces(order[0]),
        frames=[dict(name=str(f), data=traces(f)) for f in order],
        layout=dict(
            title="Smoothed SH bubbles — 3D field (drag slider / press Play)",
            scene=dict(xaxis=dict(title="X (mm)", range=axrange[0]),
                       yaxis=dict(title="Y (mm)", range=axrange[1]),
                       zaxis=dict(title="Z (mm)", range=axrange[2]),
                       aspectmode="cube"),
            sliders=sliders, updatemenus=updatemenus, height=760, width=920),
    )
    write_html(fig, out_path, "Smoothed SH — 3D")
    print(f"  3D   → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# HTML 2 — camera projections
# ════════════════════════════════════════════════════════════════════════════

def mask_contour_xy(mask: np.ndarray, step: int = 4) -> tuple[list, list]:
    from skimage.measure import find_contours
    small = mask[::step, ::step].astype(float)
    us: list[float] = []
    vs: list[float] = []
    for c in find_contours(small, 0.5):
        us.extend((c[:, 1] * step).tolist() + [float("nan")])
        vs.extend((c[:, 0] * step).tolist() + [float("nan")])
    return us, vs


def build_projection(frames: dict[int, dict], working_dir: Path, out_path: Path,
                     num_cameras: int = 4) -> None:
    from visual_hull.camera import OpenLPTCameraSet
    from visual_hull.io import load_tiff_mask

    cam_paths = [working_dir / "camFile_VSC" / f"vsc_cam{c}.txt" for c in range(num_cameras)]
    cameras = OpenLPTCameraSet.from_camera_files(cam_paths)
    order = sorted(frames)
    m0 = load_tiff_mask(working_dir / "imgFile_bb" / "cam0" / f"img{order[0]:06d}.tif")
    H, W = m0.shape

    # 2x2 axis anchors and domains.
    axkey = ["", "2", "3", "4"]
    domains = [([0.0, 0.47], [0.55, 1.0]), ([0.53, 1.0], [0.55, 1.0]),
               ([0.0, 0.47], [0.0, 0.45]), ([0.53, 1.0], [0.0, 0.45])]

    max_b = max(int(frames[f]["sh_num_bubbles"].ravel()[0]) for f in order)

    def traces(f: int) -> list:
        sh = frames[f]
        nb = int(sh["sh_num_bubbles"].ravel()[0])
        tids = sh.get("sh_track_id")
        tids = tids.ravel() if tids is not None else None
        out = []
        for c in range(num_cameras):
            xa = "x" + axkey[c]
            ya = "y" + axkey[c]
            mask = load_tiff_mask(working_dir / "imgFile_bb" / f"cam{c}" / f"img{f:06d}.tif")
            mu, mv = mask_contour_xy(mask)
            out.append(dict(type="scatter", x=mu, y=mv, mode="lines", xaxis=xa, yaxis=ya,
                            line=dict(color="rgba(120,120,120,0.6)", width=1),
                            name="real mask", showlegend=False))
            # one outline trace per bubble slot, colored by track id
            for b in range(max_b):
                ox: list[float] = []
                oy: list[float] = []
                color = "#d62728"
                if b < nb:
                    verts, _ = bubble_mesh(sh, b)
                    proj = cameras.project_points(c, verts)
                    pts = proj.pixels[proj.valid]
                    if pts.shape[0] >= 3:
                        try:
                            hull = ConvexHull(pts)
                            poly = pts[hull.vertices]
                            ox = poly[:, 0].tolist() + [poly[0, 0]]
                            oy = poly[:, 1].tolist() + [poly[0, 1]]
                        except Exception:
                            pass
                    if tids is not None:
                        color = track_color(int(tids[b]))
                out.append(dict(type="scatter", x=ox, y=oy, mode="lines", xaxis=xa, yaxis=ya,
                                line=dict(color=color, width=2), showlegend=False,
                                hoverinfo="skip" if not ox else "text",
                                name=(f"track {int(tids[b])}" if (tids is not None and b < nb) else "")))
        return out

    layout: dict = dict(
        title="Smoothed SH projected into each camera (red) vs real mask (gray)",
        height=1000, width=1050,
        annotations=[dict(text=f"cam{c}", x=sum(domains[c][0]) / 2, y=domains[c][1][1] + 0.02,
                          xref="paper", yref="paper", showarrow=False,
                          font=dict(size=13)) for c in range(num_cameras)],
    )
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

    n_tr = num_cameras * (1 + max_b)
    fig = dict(
        data=traces(order[0]),
        frames=[dict(name=str(f), data=traces(f), traces=list(range(n_tr))) for f in order],
        layout=layout,
    )
    write_html(fig, out_path, "Smoothed SH — camera projections")
    print(f"  proj → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoothed-dir", type=Path, required=True)
    ap.add_argument("--working-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-cameras", type=int, default=4)
    ap.add_argument("--skip-projection", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    frames = load_smoothed(args.smoothed_dir)
    if not frames:
        print("No *_sh_smoothed.mat found.")
        return
    print(f"Loaded {len(frames)} smoothed frames ({min(frames)}-{max(frames)}).")

    build_3d(frames, args.out / "smoothed_3d_all_frames.html")
    if not args.skip_projection:
        build_projection(frames, args.working_dir,
                         args.out / "smoothed_projection_cameras.html",
                         num_cameras=args.num_cameras)
    print("Done.")


if __name__ == "__main__":
    main()
