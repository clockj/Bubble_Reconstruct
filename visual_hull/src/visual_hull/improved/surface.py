from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..surface_utils import surface_mesh_from_voxels


@dataclass(slots=True)
class MeshSmoothingConfig:
    iterations: int = 12
    lambda_factor: float = 0.35
    mu_factor: float = -0.36

    def to_dict(self) -> dict[str, float | int]:
        return {
            "iterations": int(self.iterations),
            "lambda_factor": float(self.lambda_factor),
            "mu_factor": float(self.mu_factor),
        }


@dataclass(slots=True)
class LaplacianSmoothingConfig:
    iterations: int = 10
    lambda_factor: float = 0.18
    preserve_volume: bool = True

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "iterations": int(self.iterations),
            "lambda_factor": float(self.lambda_factor),
            "preserve_volume": bool(self.preserve_volume),
        }


def build_vertex_adjacency(faces: np.ndarray, vertex_count: int) -> list[np.ndarray]:
    neighbors: list[set[int]] = [set() for _ in range(int(vertex_count))]
    for face in np.asarray(faces, dtype=np.int64):
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return [np.array(sorted(group), dtype=np.int64) for group in neighbors]


def laplacian_step(vertices: np.ndarray, adjacency: list[np.ndarray], factor: float) -> np.ndarray:
    updated = np.asarray(vertices, dtype=np.float64).copy()
    for index, neighbor_ids in enumerate(adjacency):
        if neighbor_ids.size == 0:
            continue
        neighbor_mean = np.mean(vertices[neighbor_ids], axis=0)
        updated[index] = vertices[index] + float(factor) * (neighbor_mean - vertices[index])
    return updated


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    tris = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    signed_volume = np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])) / 6.0
    return float(abs(np.sum(signed_volume)))


def laplacian_smooth_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    config: LaplacianSmoothingConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    settings = config or LaplacianSmoothingConfig()
    source_vertices = np.asarray(vertices, dtype=np.float64)
    source_faces = np.asarray(faces, dtype=np.int64)
    smoothed = source_vertices.copy()
    adjacency = build_vertex_adjacency(source_faces, smoothed.shape[0])
    target_volume = mesh_volume(smoothed, source_faces)

    for _ in range(max(int(settings.iterations), 0)):
        smoothed = laplacian_step(smoothed, adjacency, settings.lambda_factor)
        if settings.preserve_volume:
            current_volume = mesh_volume(smoothed, source_faces)
            if current_volume > 1e-12 and target_volume > 1e-12:
                centroid = np.mean(smoothed, axis=0)
                scale = (target_volume / current_volume) ** (1.0 / 3.0)
                smoothed = centroid + scale * (smoothed - centroid)

    return smoothed, source_faces


def taubin_smooth_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    config: MeshSmoothingConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    settings = config or MeshSmoothingConfig()
    smoothed = np.asarray(vertices, dtype=np.float64).copy()
    adjacency = build_vertex_adjacency(faces, smoothed.shape[0])

    for _ in range(max(int(settings.iterations), 0)):
        smoothed = laplacian_step(smoothed, adjacency, settings.lambda_factor)
        smoothed = laplacian_step(smoothed, adjacency, settings.mu_factor)

    return smoothed, np.asarray(faces, dtype=np.int64)


__all__ = [
    "MeshSmoothingConfig",
    "LaplacianSmoothingConfig",
    "laplacian_smooth_mesh",
    "mesh_volume",
    "surface_mesh_from_voxels",
    "taubin_smooth_mesh",
]