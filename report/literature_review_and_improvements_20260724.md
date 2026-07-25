# Literature Review: Physics-Informed Bubble Surface Reconstruction — Findings & Improvement Roadmap

**Date:** 2026-07-24
**Project:** 3D bubble surface reconstruction from 4 refractive multi-view cameras
**Scope:** Five papers (`Ref/`) reviewed for methods we can adopt to improve the
current visual-hull + virtual-camera-rounding pipeline. Companion to
`surface_representation_report_20260724.md`.

---

## 0. Headline

The single most important finding: **Paper 1 (Masuk, Salibindla & Ni, 2019) is
from the Rui Ni lab at JHU — the same group that produced this project's data.**
Our pipeline is a re-derivation of their virtual-camera (VC) method, so the
literature tells us *exactly* how the state of the art refines it. The two most
valuable, low-risk upgrades are already proven there and in its SJTU successor
(Paper 3): a **real-camera reprojection gate** and **conic (Bézier) outline
smoothing**. Two further ideas (curvature-regularized minimal surface; volume
conservation / PDE assimilation) are principled but higher-effort.

---

## 1. The five papers at a glance

| # | Paper | Core idea | Relevance |
|---|---|---|---|
| 1 | Masuk, Salibindla, **Ni** 2019, *IJMF* 120:103088 | **Virtual-camera** method: spread the *minimum-surface-curvature* (surface-tension) prior to virtual cameras in the gap directions; smooth their 2-D silhouette corners; **re-project to real cameras and accept only if real silhouettes preserved** | **Foundational — same lab as our data.** VH over-est. 20%→10% |
| 2 | Gong et al. 2022 (BubDepth), *IJMF* 152:104100 | Single-view **shape-from-shading** CNN: grayscale → surface-normal → per-bubble depth map | Alternative info source (uses intensity, not just silhouette) |
| 3 | Huang et al. 2025, *IJMF* 184:105106 | Improved VC: smooth outline corners with **rational-quadratic Bézier (conic) curves** (exact for sphere/ellipsoid); 5% reproject gate; 4 real + 132 virtual views | Direct upgrade to our rounding; ~30° 4-cam setup like ours |
| 4 | He, Kang, Liu 2020, arXiv:2001.07884 | **Curvature-regularized minimal surface** from point cloud: `E=∫|d|ˢdσ + η∫|κ|ˢdσ`, level-set, stable operator-splitting / ALM solvers | Rigorous form + solver for our physics prior ① |
| 5 | Dreisbach et al. 2025 (PINNs4Drops), arXiv:2411.15949 | **Video-conditioned PINN**: NN infers 3-D phase+velocity+pressure; physics loss = continuity (∇·u=0) + interface advection + Navier–Stokes; enforces temporal coherence | The ambitious PDE-assimilation route (shape **and** flow) |

---

## 2. What we learn from each

### Paper 1 — Masuk/Ni 2019 (the origin of our method)
- The VC method **is** what we built: translate "surface tension → minimum
  curvature/area" into smoothing on **virtual cameras placed in the least-covered
  (gap) directions** (minimise the max angular gap; ~17 virtual cams suffice).
- **Critical technique we are missing — the reprojection gate:** each smoothing
  step is re-projected onto the 4 real cameras and **accepted only if the real
  silhouette area is preserved (within 10%)**; otherwise reverted. This is what
  stops over-trimming, and is the *correct* form of the "anisotropic" idea our
  earlier experiment failed at (we restored geometry by normal direction; the
  right test is real-silhouette consistency).
- **Uncertainty quantification:** estimate per-frame reconstruction uncertainty
  from **aspect ratio + orientation** → select/deweight frames for statistics.
- **Overlap-artifact rejection:** flag bubbles where **volume ≫ track mean** and
  **reprojected area ≪ detected mask area**.
- **Honest floor:** cannot refine a feature (e.g. a breakup neck) that *no* real
  camera sees; over-corrects when cameras are too few. The reproject gate bounds
  the over-correction.

### Paper 3 — Huang/SJTU 2025 (the best refinement of the VC method)
- Replaces neighbour-averaging smoothing with **rational-quadratic Bézier
  curves** on the 2-D outline. One weight `ω₁` morphs the corner
  hyperbola→ellipse→circle; the curve is **tangent to the corner edges** (C¹).
  Claim: **exactly reconstructs spheres and ellipsoids.** This is the correct
  primitive for our regime (Bo≈0.15, near-ellipsoidal bubbles) and would remove
  the "pillow/cushion" residual our morphological ρ-opening leaves.
- Same **reproject gate** (5% area threshold); volume carved outside the smoothed
  outline; virtual cams added until they cover the sphere (132 views).
- Surface-area error **< 5%** on synthetic non-affine shapes.
- Their scalar-field tomography (aside) uses **L1 sparsity + Total-Variation +
  interpolated missing-view projections** — the "morph real projections to
  synthesise missing views" trick is a data-driven alternative to assuming
  smooth silhouettes.

### Paper 4 — He/Kang/Liu 2020 (rigorous curvature regularization)
- The exact mathematical form of our physics idea ①: minimise
  **`E = ∫_Γ |d(x)|ˢ dσ + η ∫_Γ |κ|ˢ dσ`** (distance fidelity + mean-curvature
  regularization) on a level-set surface.
- **L1 curvature** preserves sharp corners/necks; **L2** (bending energy
  `∫κ²dσ ≥ 4π`, Willmore) gives smoother surfaces. → choose L2 for quiescent
  near-spherical bubbles, L1 near breakup.
- Provides **stable solvers** (operator-splitting semi-implicit; augmented
  Lagrangian) for the high-order curvature PDE — far more robust than our crude
  explicit mesh evolve, and naturally handles noisy/sparse point clouds.

### Paper 2 — Gong/SJTU 2022 (shape-from-shading, orthogonal idea)
- Uses the **grayscale** the silhouette hull discards: brighter pixel ⇒ surface
  more face-on to the camera (`g = M(cos∠(n, camera))`), inverted by a CNN to a
  per-bubble depth map; trained on synthetic (Joukowski-transform) bubbles.
- Only reconstructs the **front side**; needs multi-view for full shape.
- Take-away: intensity carries 3-D information — the one idea here that **adds
  data** rather than a prior. Hard in our *backlit refractive* setup (masks are
  near-binary), but worth noting if we ever want to break the 4-view floor.

### Paper 5 — PINNs4Drops 2025 (full PDE assimilation)
- A **video-conditioned PINN** represents continuous 3-D **phase φ, velocity u,
  pressure p**; trained on a composite loss `L = L_Data + Σ w·(L_Conti + L_Adv +
  L_NSE)` — i.e. **∇·u=0 (continuity)**, interface advection (VoF/phase-field/
  level-set), and Navier–Stokes momentum, plus data terms from images.
- Conditioning on CNN image-sequence features makes the stiff two-phase
  optimisation tractable and **enforces temporal coherence** — reconstructing
  shape *and* the surrounding flow from limited views.
- Practical caveats they report: high water/air density-viscosity ratio → stiff;
  localized surface-tension force needs careful loss weighting + adaptive
  sampling near the interface; needs DNS for validation.
- Take-away: this is the route if the **flow field** is also a deliverable;
  otherwise heavy. The transferable sub-idea (even without a full PINN) is
  **enforcing continuity/temporal coherence**.

---

## 3. Improvement roadmap (prioritised)

### Tier 1 — adopt now (low risk, high value; proven in Papers 1 & 3)
1. **Real-camera reprojection gate.** After any rounding/smoothing step,
   re-project the surface into the 4 real cameras (OpenLPT) and accept only if
   each real silhouette IoU/area is preserved (≥95%). Fixes our over-trim
   (D −1%, reproj-IoU 0.82) and is the correct, working form of "anisotropic".
2. **Conic (rational-quadratic Bézier) outline smoothing** in place of
   morphological ρ-opening — provably exact for spheres/ellipsoids, removes the
   pillow residual, one parameter `ω₁`.
3. **Per-frame uncertainty + artifact rejection** (aspect ratio + orientation
   proxy; volume-vs-track-mean and reprojected-vs-detected-area overlap flags) —
   plugs into our tracking / phantom-bubble filtering.

→ Implement as `virtual_camera/silhouette_refine.py` (needs real masks +
OpenLPT, which we have) and benchmark vs the current isotropic rounding on the
reference shapes.

### Tier 2 — principled upgrades (medium effort)
4. **Curvature-regularized minimal-surface fit** (Paper 4): replace the explicit
   mesh evolve with `∫|d|ˢ + η∫|κ|ˢ`, using L2 curvature (quiescent) or L1 (near
   breakup) and a semi-implicit/ALM solver. More robust to sparse/noisy hull
   voxels; unifies smoothing + fidelity in one variational principle.
5. **Volume conservation along a track** (from ∇·u=0; **not done by any of these
   papers** → novel here): constrain a bubble's volume to be constant frame-to-
   frame (barring detected coalescence/breakup), removing volume jitter and
   flagging topology changes.

### Tier 3 — research-grade (high effort/risk)
6. **Video-conditioned PINN / 4D-Var** (Paper 5): joint shape + flow from the
   image sequence with continuity + advection + momentum residuals. Justified
   only if the surrounding flow field is itself a target.
7. **Shading/grayscale cue** (Paper 2): add intensity information beyond the
   binary mask — the only avenue that adds *data* past the 4-view floor; hard
   under refractive backlighting.

---

## 4. Mapping to the current code

| Improvement | Where it lands | Reuses |
|---|---|---|
| Reproject gate (#1) | new `virtual_camera/silhouette_refine.py` | `camera.OpenLPTCameraSet`, `silhouette_metrics` |
| Bézier smoothing (#2) | same module (2-D per virtual view) | `views.virtual_directions/viewing_directions` |
| Uncertainty/artifact (#3) | `scripts/temporal_smooth_sh.py`, filtering in `reconstruct_tiff_data.py` | `properties` (aspect/orientation) |
| Curvature min-surface (#4) | `mesh_surface/` (swap the evolve) or new level-set module | `mesh_ops.mean_curvature`, `sdf` |
| Volume conservation (#5) | `scripts/temporal_smooth_sh.py` | fixed-topology meshes |

---

## 5. Honest limitations (confirmed by the literature)
- **4 views is a hard floor.** All silhouette methods (Papers 1, 3, and ours)
  cannot recover a feature no camera sees; roundness between views is an imposed
  surface-tension prior, not a measurement. The reproject gate bounds — but does
  not eliminate — over-smoothing.
- **Physics constrains shape and *relative* volume, not absolute calibration.**
  The +6% hull bias's absolute part still needs calibration/self-calibration
  (Paper 1 stresses volumetric self-calibration; even their 6-camera volume still
  over-estimates).
- **Regime matters.** Conic/CMC priors are near-exact for our small
  (Bo≈0.15) near-ellipsoidal bubbles, but degrade for strongly non-affine,
  near-breakup shapes — exactly where Paper 1 reports residual ~20% error.

---

## 6. References
1. A.U.M. Masuk, A. Salibindla, R. Ni. *A robust virtual-camera 3D shape reconstruction of deforming bubbles/droplets with additional physical constraints.* Int. J. Multiphase Flow 120 (2019) 103088.
2. C. Gong et al. *BubDepth: A neural network approach to 3D reconstruction of bubble geometry from single-view images.* Int. J. Multiphase Flow 152 (2022) 104100.
3. G. Huang, B. Liu, Y. Song, J. Yin, D. Wang. *3D measurement of interfacial mass transfer … Multi-view SI-VLIF …* Int. J. Multiphase Flow 184 (2025) 105106.
4. Y. He, S.H. Kang, H. Liu. *Curvature Regularized Surface Reconstruction from Point Cloud.* arXiv:2001.07884 (2020).
5. M. Dreisbach, E. Kiyani, J. Kriegseis, G.E. Karniadakis, A. Stroh. *PINNs4Drops: Video-conditioned physics-informed neural networks for two-phase flow reconstruction.* arXiv:2411.15949 (2025).
