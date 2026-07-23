from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np
from ...camera import OpenLPTCameraSet
from ...silhouette_metrics import project_meshes_to_camera_masks, summarize_mask_overlap
try:
    from scipy.special import sph_harm as _complex_spherical_harmonic

    def _evaluate_complex_harmonic(degree: int, order: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return _complex_spherical_harmonic(order, degree, phi, theta)

except ImportError:
    from scipy.special import sph_harm_y as _complex_spherical_harmonic

    def _evaluate_complex_harmonic(degree: int, order: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return _complex_spherical_harmonic(degree, order, theta, phi)

from ...surface_utils import surface_mesh_from_voxels


@dataclass(slots=True)
class SphericalHarmonicFitConfig:
    max_degree: int = 4
    regularization: float = 1e-3
    theta_samples: int = 40
    phi_samples: int = 80
    minimum_radius: float = 1e-3
    silhouette_enabled: bool = False
    silhouette_weight: float = 0.0
    silhouette_max_passes: int = 0
    silhouette_step_scale: float = 0.05
    silhouette_top_k: int = 12
    coefficient_drift_weight: float = 0.1
    inscribed: bool = False
    overshoot_weight: float = 50.0
    inscribed_iters: int = 20
    # ── Phase 1 / 3 additions ─────────────────────────────────────────────
    spectral_weight: float = 0.0      # (l-2)^2 penalty on l>=3 (anti-flower)
    curvature_weight: float = 0.0     # [l(l+1)]^2 bending-energy penalty (P2/P3)
    convexity_weight: float = 0.0     # blend toward convex hull, fills dents (P4)
    init_mode: str = "lstsq"          # "lstsq" | "ellipsoid" warm start

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "max_degree": int(self.max_degree),
            "regularization": float(self.regularization),
            "theta_samples": int(self.theta_samples),
            "phi_samples": int(self.phi_samples),
            "minimum_radius": float(self.minimum_radius),
            "silhouette_enabled": bool(self.silhouette_enabled),
            "silhouette_weight": float(self.silhouette_weight),
            "silhouette_max_passes": int(self.silhouette_max_passes),
            "silhouette_step_scale": float(self.silhouette_step_scale),
            "silhouette_top_k": int(self.silhouette_top_k),
            "coefficient_drift_weight": float(self.coefficient_drift_weight),
            "inscribed": bool(self.inscribed),
            "overshoot_weight": float(self.overshoot_weight),
            "inscribed_iters": int(self.inscribed_iters),
            "spectral_weight": float(self.spectral_weight),
            "curvature_weight": float(self.curvature_weight),
            "convexity_weight": float(self.convexity_weight),
            "init_mode": str(self.init_mode),
        }


@dataclass(slots=True)
class SphericalHarmonicSurface:
    center: np.ndarray
    coefficients: np.ndarray
    basis_terms: list[tuple[int, int]]
    vertices: np.ndarray
    faces: np.ndarray
    fit_rmse: float
    silhouette_iou: float | None = None
    objective_value: float | None = None
    evaluation_count: int = 0


def _cartesian_to_spherical(points: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shifted = np.asarray(points, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    radius = np.linalg.norm(shifted, axis=1)
    safe_radius = np.maximum(radius, 1e-12)
    theta = np.arccos(np.clip(shifted[:, 2] / safe_radius, -1.0, 1.0))
    phi = np.mod(np.arctan2(shifted[:, 1], shifted[:, 0]), 2.0 * pi)
    return radius, theta, phi


def _basis_terms(max_degree: int) -> list[tuple[int, int]]:
    terms: list[tuple[int, int]] = []
    for degree in range(max(int(max_degree), 0) + 1):
        for order in range(-degree, degree + 1):
            terms.append((degree, order))
    return terms


def _real_spherical_harmonic(degree: int, order: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    if order < 0:
        harmonic = _evaluate_complex_harmonic(degree, -order, theta, phi)
        return np.sqrt(2.0) * ((-1) ** (-order)) * np.imag(harmonic)
    if order == 0:
        return np.real(_evaluate_complex_harmonic(degree, 0, theta, phi))
    harmonic = _evaluate_complex_harmonic(degree, order, theta, phi)
    return np.sqrt(2.0) * ((-1) ** order) * np.real(harmonic)


def _design_matrix(theta: np.ndarray, phi: np.ndarray, terms: list[tuple[int, int]]) -> np.ndarray:
    return np.column_stack([_real_spherical_harmonic(degree, order, theta, phi) for degree, order in terms]).astype(
        np.float64,
        copy=False,
    )


def _penalty_diagonal(
    terms: list[tuple[int, int]],
    regularization: float,
    spectral_weight: float = 0.0,
    curvature_weight: float = 0.0,
) -> np.ndarray:
    """Per-coefficient L2 penalty (lambda_k) folded into the linear fit.

    Combines three linear priors on the SH coefficients:
      - base Tikhonov ``regularization`` on every term (R1 baseline),
      - spectral ``spectral_weight * max(l-2, 0)^2`` on l>=3 (anti-flower, R1),
      - bending energy ``curvature_weight * [l(l+1)]^2`` (P2/P3 — for a radial
        SH surface, mean-curvature roughness lives in the high-l band, so
        curvature uniformity / max-curvature reduce to a high-l penalty).

    ``l=0,1,2`` (sphere / translation / ellipsoid) carry no spectral or
    curvature penalty, so genuine low-order shape is never suppressed.
    """
    base = max(float(regularization), 0.0)
    sw = max(float(spectral_weight), 0.0)
    cw = max(float(curvature_weight), 0.0)
    diag = np.empty(len(terms), dtype=np.float64)
    for k, (l, _m) in enumerate(terms):
        diag[k] = (
            base
            + sw * float(max(l - 2, 0)) ** 2
            + cw * float(l * (l + 1)) ** 2
        )
    return diag


def _fit_coefficients(
    design: np.ndarray,
    radius: np.ndarray,
    regularization: float,
    penalty: np.ndarray | None = None,
) -> np.ndarray:
    if penalty is None:
        lam = max(float(regularization), 0.0)
        if lam <= 0.0:
            coefficients, *_ = np.linalg.lstsq(design, radius, rcond=None)
            return coefficients.astype(np.float64, copy=False)
        penalty = np.full(design.shape[1], lam, dtype=np.float64)

    sqrt_pen = np.sqrt(np.maximum(penalty, 0.0))
    augmented_design = np.vstack((design, np.diag(sqrt_pen)))
    augmented_radius = np.concatenate((radius, np.zeros(design.shape[1], dtype=np.float64)))
    coefficients, *_ = np.linalg.lstsq(augmented_design, augmented_radius, rcond=None)
    return coefficients.astype(np.float64, copy=False)


def _outer_envelope_indices(
    theta: np.ndarray, phi: np.ndarray, radius: np.ndarray, nbins: int = 14
) -> np.ndarray:
    """Indices of the outermost surface point in each angular bin.

    The hull voxels form a shell of finite thickness; fitting the SH to the
    *mean* of that shell pulls the surface inward. Selecting the max-radius
    point per (theta, phi) bin gives the hull's outer boundary, so the fit
    tracks the silhouette rather than the shell interior.
    """
    ti = np.clip((theta / pi * nbins).astype(np.int64), 0, nbins - 1)
    pj = np.clip((phi / (2.0 * pi) * nbins).astype(np.int64), 0, nbins - 1)
    keys = ti * nbins + pj
    chosen: list[int] = []
    for key in np.unique(keys):
        members = np.where(keys == key)[0]
        chosen.append(int(members[np.argmax(radius[members])]))
    return np.asarray(sorted(chosen), dtype=np.int64)


def _fit_coefficients_inscribed(
    design: np.ndarray,
    radius: np.ndarray,
    regularization: float,
    overshoot_weight: float,
    iters: int,
    penalty: np.ndarray | None = None,
    warm_start: np.ndarray | None = None,
) -> np.ndarray:
    """Fit SH coefficients so the surface stays *inside* the hull samples.

    Standard least squares fits the mean of the surface points, so the SH
    radius bulges *outward* past the samples in some directions (the source
    of both the flower petals and extra over-estimate beyond the hull).

    Here we add a one-sided penalty ``w * sum(max(0, Yc - r)^2)`` that acts
    only where the fitted radius exceeds the voxel radius, solved by
    iteratively re-weighted least squares: each pass re-solves the normal
    equations with the currently-overshooting samples up-weighted, pulling
    the surface down until it hugs the inner side of the hull boundary.
    """
    n_coeff = design.shape[1]
    if penalty is None:
        lam = max(float(regularization), 0.0)
        penalty = np.full(n_coeff, lam, dtype=np.float64)
    gram = design.T @ design + np.diag(np.maximum(penalty, 0.0))
    rhs = design.T @ radius
    if warm_start is not None and warm_start.shape[0] == n_coeff:
        coefficients = np.asarray(warm_start, dtype=np.float64).copy()
    else:
        coefficients = _fit_coefficients(design, radius, regularization, penalty)

    w = max(float(overshoot_weight), 0.0)
    for _ in range(max(int(iters), 0)):
        overshoot = (design @ coefficients) > radius
        if not np.any(overshoot):
            break
        design_over = design[overshoot]
        radius_over = radius[overshoot]
        lhs = gram + w * (design_over.T @ design_over)
        b = rhs + w * (design_over.T @ radius_over)
        coefficients = np.linalg.solve(lhs, b)
    return coefficients.astype(np.float64, copy=False)


def _coefficient_indices_for_refinement(coefficients: np.ndarray, top_k: int) -> np.ndarray:
    if coefficients.size == 0:
        return np.empty((0,), dtype=np.int64)
    if top_k <= 0 or top_k >= coefficients.size:
        return np.arange(coefficients.size, dtype=np.int64)
    ranking = np.argsort(-np.abs(coefficients))
    return np.sort(ranking[:top_k].astype(np.int64, copy=False))


def _silhouette_objective(
    coefficients: np.ndarray,
    *,
    center: np.ndarray,
    design: np.ndarray,
    radius: np.ndarray,
    initial_coefficients: np.ndarray,
    terms: list[tuple[int, int]],
    masks: list[np.ndarray],
    cameras: OpenLPTCameraSet,
    config: SphericalHarmonicFitConfig,
) -> tuple[float, float, float]:
    fitted_radius = np.maximum(design @ coefficients, float(config.minimum_radius))
    mesh_rmse = float(np.sqrt(np.mean((fitted_radius - radius) ** 2)))
    drift_penalty = float(np.mean((coefficients - initial_coefficients) ** 2))
    vertices, faces = _grid_vertices_faces(center, coefficients, terms, config)
    predicted_masks = project_meshes_to_camera_masks([(vertices, faces)], masks, cameras)
    overlap = summarize_mask_overlap(predicted_masks, masks)
    silhouette_iou = float(overlap["overall"]["iou"])
    loss = (
        mesh_rmse
        + float(config.silhouette_weight) * (1.0 - silhouette_iou)
        + float(config.coefficient_drift_weight) * drift_penalty
    )
    return loss, silhouette_iou, mesh_rmse


def _refine_coefficients_with_silhouette(
    coefficients: np.ndarray,
    *,
    center: np.ndarray,
    design: np.ndarray,
    radius: np.ndarray,
    terms: list[tuple[int, int]],
    masks: list[np.ndarray],
    cameras: OpenLPTCameraSet,
    config: SphericalHarmonicFitConfig,
) -> tuple[np.ndarray, float | None, float | None, int]:
    if not bool(config.silhouette_enabled):
        return coefficients, None, None, 0
    if float(config.silhouette_weight) <= 0.0 or int(config.silhouette_max_passes) <= 0:
        return coefficients, None, None, 0

    refined = coefficients.astype(np.float64, copy=True)
    initial = coefficients.astype(np.float64, copy=True)
    active_indices = _coefficient_indices_for_refinement(refined, int(config.silhouette_top_k))
    if active_indices.size == 0:
        return refined, None, None, 0

    radius_scale = max(float(np.mean(radius)), float(config.minimum_radius))
    step_size = max(radius_scale * float(config.silhouette_step_scale), float(config.minimum_radius))
    evaluation_count = 0

    best_loss, best_iou, best_rmse = _silhouette_objective(
        refined,
        center=center,
        design=design,
        radius=radius,
        initial_coefficients=initial,
        terms=terms,
        masks=masks,
        cameras=cameras,
        config=config,
    )
    evaluation_count += 1

    for _ in range(int(config.silhouette_max_passes)):
        improved = False
        for coefficient_index in active_indices:
            for delta in (step_size, -step_size):
                candidate = refined.copy()
                candidate[coefficient_index] += delta
                loss, silhouette_iou, fit_rmse = _silhouette_objective(
                    candidate,
                    center=center,
                    design=design,
                    radius=radius,
                    initial_coefficients=initial,
                    terms=terms,
                    masks=masks,
                    cameras=cameras,
                    config=config,
                )
                evaluation_count += 1
                if loss + 1e-12 < best_loss:
                    refined = candidate
                    best_loss = loss
                    best_iou = silhouette_iou
                    best_rmse = fit_rmse
                    improved = True
        if not improved:
            step_size *= 0.5
            if step_size <= float(config.minimum_radius):
                break

    return refined, best_iou, best_loss, evaluation_count


def _grid_vertices_faces(
    center: np.ndarray,
    coefficients: np.ndarray,
    terms: list[tuple[int, int]],
    config: SphericalHarmonicFitConfig,
) -> tuple[np.ndarray, np.ndarray]:
    theta_count = max(int(config.theta_samples), 4)
    phi_count = max(int(config.phi_samples), 8)
    ring_thetas = np.linspace(0.0, pi, theta_count, dtype=np.float64)
    ring_phis = np.linspace(0.0, 2.0 * pi, phi_count, endpoint=False, dtype=np.float64)

    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []

    north_basis = _design_matrix(np.array([ring_thetas[0]]), np.array([0.0]), terms)
    north_radius = max(float(np.squeeze(north_basis @ coefficients)), float(config.minimum_radius))
    vertices.append(center + np.array([0.0, 0.0, north_radius], dtype=np.float64))

    ring_start_indices: list[int] = []
    for theta in ring_thetas[1:-1]:
        start_index = len(vertices)
        ring_start_indices.append(start_index)
        theta_array = np.full(phi_count, theta, dtype=np.float64)
        basis = _design_matrix(theta_array, ring_phis, terms)
        radii = np.maximum(basis @ coefficients, float(config.minimum_radius))
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        x = center[0] + radii * sin_theta * np.cos(ring_phis)
        y = center[1] + radii * sin_theta * np.sin(ring_phis)
        z = center[2] + radii * cos_theta
        vertices.extend(np.column_stack((x, y, z)))

    south_basis = _design_matrix(np.array([ring_thetas[-1]]), np.array([0.0]), terms)
    south_radius = max(float(np.squeeze(south_basis @ coefficients)), float(config.minimum_radius))
    south_index = len(vertices)
    vertices.append(center + np.array([0.0, 0.0, -south_radius], dtype=np.float64))

    if ring_start_indices:
        first_ring_start = ring_start_indices[0]
        for phi_index in range(phi_count):
            next_phi = (phi_index + 1) % phi_count
            faces.append((0, first_ring_start + next_phi, first_ring_start + phi_index))

        for ring_index in range(len(ring_start_indices) - 1):
            top_start = ring_start_indices[ring_index]
            bottom_start = ring_start_indices[ring_index + 1]
            for phi_index in range(phi_count):
                next_phi = (phi_index + 1) % phi_count
                top_left = top_start + phi_index
                top_right = top_start + next_phi
                bottom_left = bottom_start + phi_index
                bottom_right = bottom_start + next_phi
                faces.append((top_left, bottom_right, bottom_left))
                faces.append((top_left, top_right, bottom_right))

        last_ring_start = ring_start_indices[-1]
        for phi_index in range(phi_count):
            next_phi = (phi_index + 1) % phi_count
            faces.append((south_index, last_ring_start + phi_index, last_ring_start + next_phi))
    else:
        for phi_index in range(1, phi_count - 1):
            faces.append((0, phi_index, phi_index + 1))

    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _ellipsoid_warm_start(
    points: np.ndarray, center: np.ndarray, terms: list[tuple[int, int]],
    penalty: np.ndarray,
) -> np.ndarray:
    """Degree<=2 SH coefficients of the PCA ellipsoid through the points.

    Gives the fit a physically sensible starting shape (sphere + ellipsoid
    deformation) instead of a flat sphere, which matters for the inscribed
    IRLS and the convexity refit.
    """
    shifted = np.asarray(points, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    cov = np.cov(shifted.T)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-9)
    # Scale semi-axes so the mean radius matches the data's mean radius.
    axes = np.sqrt(evals)
    axes *= float(np.mean(np.linalg.norm(shifted, axis=1))) / float(np.mean(axes))
    # Radius of that ellipsoid along each data direction.
    dist = np.maximum(np.linalg.norm(shifted, axis=1), 1e-12)
    dirs = shifted / dist[:, None]
    local = dirs @ evecs  # into ellipsoid-aligned frame
    denom = np.sqrt(np.sum((local / axes[None, :]) ** 2, axis=1))
    r_ell = 1.0 / np.maximum(denom, 1e-12)
    _, theta, phi = _cartesian_to_spherical(points, center)
    design = _design_matrix(theta, phi, terms)
    # Only fit l<=2 terms; zero the rest.
    mask_low = np.array([l <= 2 for (l, _m) in terms])
    coeffs = np.zeros(len(terms), dtype=np.float64)
    if np.any(mask_low):
        coeffs[mask_low] = _fit_coefficients(
            design[:, mask_low], r_ell, 0.0, penalty[mask_low]
        )
    return coeffs


def _hull_radius_along(hull, center: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Ray/convex-hull intersection distance from *center* along each dir.

    hull.equations give rows [a | b] with ``a·x + b <= 0`` inside.  For the
    ray ``center + t d`` the surface is the smallest positive t that hits a
    bounding face.
    """
    eq = hull.equations  # (F, 4): [nx, ny, nz, offset]
    a = eq[:, :3]
    b = eq[:, 3]
    ac = a @ center + b            # (F,)  a·center + b  (<0 inside)
    ad = dirs @ a.T                # (N, F) a·d
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -ac[None, :] / ad      # candidate distances to each face
    t[ad <= 1e-12] = np.inf        # only faces the ray exits through
    t[t <= 0] = np.inf
    return np.min(t, axis=1)


def _apply_convexity(
    coefficients: np.ndarray, center: np.ndarray, terms: list[tuple[int, int]],
    design: np.ndarray, radius: np.ndarray, theta: np.ndarray, phi: np.ndarray,
    penalty: np.ndarray, config: SphericalHarmonicFitConfig,
) -> np.ndarray:
    """Blend the fitted radius toward the surface's own convex hull (P4).

    A convex hull fills concave dents, so moving the radius toward the hull
    radius (only where the hull is *outside* the current surface) removes
    saddle regions.  Weight 0 -> unchanged, 1 -> fully convex hull.
    """
    from scipy.spatial import ConvexHull

    w = float(config.convexity_weight)
    if w <= 0.0:
        return coefficients
    verts, _ = _grid_vertices_faces(center, coefficients, terms, config)
    if verts.shape[0] < 5:
        return coefficients
    try:
        hull = ConvexHull(verts)
    except Exception:
        return coefficients
    dist = np.maximum(np.linalg.norm(
        np.column_stack((np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(theta))), axis=1), 1e-12)
    dirs = np.column_stack((np.sin(theta) * np.cos(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(theta))) / dist[:, None]
    r_hull = _hull_radius_along(hull, np.asarray(center, dtype=np.float64), dirs)
    r_fit = design @ coefficients
    fill = np.maximum(r_hull - r_fit, 0.0)
    fill[~np.isfinite(fill)] = 0.0
    r_target = r_fit + w * fill
    return _fit_coefficients(design, r_target, config.regularization, penalty)


def surface_curvature_stats(
    center: np.ndarray, coefficients: np.ndarray, terms: list[tuple[int, int]],
    config: SphericalHarmonicFitConfig | None = None,
) -> dict[str, float]:
    """Report-only shape descriptors: high-l bending energy + concavity.

    - ``bending_energy``: sum([l(l+1)]^2 c^2) / sum(c^2), a scale-free measure
      of mean-curvature roughness (0 for a sphere, grows with wrinkling).
    - ``concavity_fraction``: fraction of surface vertices lying strictly
      inside their own convex hull (0 for a convex shape).
    """
    settings = config or SphericalHarmonicFitConfig()
    coeffs = np.asarray(coefficients, dtype=np.float64)
    l_arr = np.array([l for (l, _m) in terms], dtype=np.float64)
    energy = float(np.sum(coeffs ** 2))
    bending = float(np.sum((l_arr * (l_arr + 1.0)) ** 2 * coeffs ** 2) / energy) if energy > 0 else 0.0

    from scipy.spatial import ConvexHull
    verts, _ = _grid_vertices_faces(np.asarray(center, dtype=np.float64), coeffs, terms, settings)
    concavity = 0.0
    if verts.shape[0] >= 5:
        try:
            hull = ConvexHull(verts)
            eq = hull.equations
            # signed distance to each face; inside-hull => all <=0.
            sd = verts @ eq[:, :3].T + eq[:, 3]
            depth = -np.max(sd, axis=1)  # >0 means strictly interior
            tol = 1e-3 * float(np.mean(np.linalg.norm(verts - center, axis=1)))
            concavity = float(np.mean(depth > tol))
        except Exception:
            concavity = 0.0
    return {"bending_energy": bending, "concavity_fraction": concavity}


def fit_spherical_harmonic_surface(
    vertices: np.ndarray,
    *,
    config: SphericalHarmonicFitConfig | None = None,
    masks: list[np.ndarray] | None = None,
    cameras: OpenLPTCameraSet | None = None,
) -> SphericalHarmonicSurface:
    settings = config or SphericalHarmonicFitConfig()
    source_vertices = np.asarray(vertices, dtype=np.float64)
    if source_vertices.ndim != 2 or source_vertices.shape[1] != 3 or source_vertices.shape[0] < 4:
        raise ValueError("At least four 3D vertices are required to fit a spherical harmonic surface.")

    center = np.mean(source_vertices, axis=0)
    radius, theta, phi = _cartesian_to_spherical(source_vertices, center)
    terms = _basis_terms(settings.max_degree)
    design = _design_matrix(theta, phi, terms)
    penalty = _penalty_diagonal(
        terms, settings.regularization,
        settings.spectral_weight, settings.curvature_weight,
    )
    if settings.inscribed:
        # Fit the hull's outer boundary (max radius per angular bin), then cap
        # outward overshoot, so the surface tracks the silhouette without
        # bulging past the hull.  Note: the inscribed IRLS only pulls radii
        # *down*, so it must start from the least-squares solution (correct
        # scale) — an under-scaled warm start would never grow back.
        envelope = _outer_envelope_indices(theta, phi, radius)
        if envelope.size >= 2 * len(terms):
            fit_design, fit_radius = design[envelope], radius[envelope]
        else:  # too few bins for a stable envelope — use all points
            fit_design, fit_radius = design, radius
        coefficients = _fit_coefficients_inscribed(
            fit_design, fit_radius, settings.regularization,
            settings.overshoot_weight, settings.inscribed_iters,
            penalty=penalty,
        )
    else:
        coefficients = _fit_coefficients(design, radius, settings.regularization, penalty)
    if settings.convexity_weight > 0.0:
        coefficients = _apply_convexity(
            coefficients, center, terms, design, radius, theta, phi, penalty, settings,
        )
    silhouette_iou: float | None = None
    objective_value: float | None = None
    evaluation_count = 0
    if masks is not None and cameras is not None:
        coefficients, silhouette_iou, objective_value, evaluation_count = _refine_coefficients_with_silhouette(
            coefficients,
            center=center,
            design=design,
            radius=radius,
            terms=terms,
            masks=[np.asarray(mask, dtype=bool) for mask in masks],
            cameras=cameras,
            config=settings,
        )
    fitted_radius = np.maximum(design @ coefficients, float(settings.minimum_radius))
    fit_rmse = float(np.sqrt(np.mean((fitted_radius - radius) ** 2)))
    fitted_vertices, faces = _grid_vertices_faces(center, coefficients, terms, settings)

    return SphericalHarmonicSurface(
        center=center.astype(np.float64, copy=False),
        coefficients=coefficients,
        basis_terms=terms,
        vertices=fitted_vertices,
        faces=faces,
        fit_rmse=fit_rmse,
        silhouette_iou=silhouette_iou,
        objective_value=objective_value,
        evaluation_count=evaluation_count,
    )


def fit_spherical_harmonic_surface_from_voxels(
    voxels: np.ndarray,
    voxel_size: np.ndarray,
    *,
    config: SphericalHarmonicFitConfig | None = None,
    masks: list[np.ndarray] | None = None,
    cameras: OpenLPTCameraSet | None = None,
) -> SphericalHarmonicSurface | None:
    mesh = surface_mesh_from_voxels(voxels, voxel_size)
    if mesh is None:
        return None
    vertices, _ = mesh
    return fit_spherical_harmonic_surface(vertices, config=config, masks=masks, cameras=cameras)


__all__ = [
    "SphericalHarmonicFitConfig",
    "SphericalHarmonicSurface",
    "fit_spherical_harmonic_surface",
    "fit_spherical_harmonic_surface_from_voxels",
    "surface_curvature_stats",
]