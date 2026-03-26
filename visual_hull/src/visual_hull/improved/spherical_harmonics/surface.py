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

    def to_dict(self) -> dict[str, float | int]:
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


def _fit_coefficients(design: np.ndarray, radius: np.ndarray, regularization: float) -> np.ndarray:
    lam = max(float(regularization), 0.0)
    if lam <= 0.0:
        coefficients, *_ = np.linalg.lstsq(design, radius, rcond=None)
        return coefficients.astype(np.float64, copy=False)

    augmented_design = np.vstack((design, np.sqrt(lam) * np.eye(design.shape[1], dtype=np.float64)))
    augmented_radius = np.concatenate((radius, np.zeros(design.shape[1], dtype=np.float64)))
    coefficients, *_ = np.linalg.lstsq(augmented_design, augmented_radius, rcond=None)
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
    coefficients = _fit_coefficients(design, radius, settings.regularization)
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
]