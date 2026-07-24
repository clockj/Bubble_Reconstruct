"""Discrete differential operators on a triangle mesh (numpy only).

Provides the pieces the energies need: vertex adjacency, the uniform (umbrella)
Laplacian, area-weighted vertex normals, and a discrete mean-curvature estimate.
All are pure functions of (vertices, faces) so they compose cleanly in the
optimizer.
"""

from __future__ import annotations

import numpy as np


def build_adjacency(faces: np.ndarray, n_vertices: int) -> list[np.ndarray]:
    """Neighbor vertex indices per vertex (list of int arrays)."""
    nbrs: list[set] = [set() for _ in range(n_vertices)]
    for a, b, c in faces:
        nbrs[a].update((b, c)); nbrs[b].update((a, c)); nbrs[c].update((a, b))
    return [np.fromiter(s, dtype=np.int64) for s in nbrs]


def uniform_laplacian(vertices: np.ndarray, adjacency: list[np.ndarray]) -> np.ndarray:
    """Umbrella operator: mean(neighbors) - vertex, per vertex (N,3).

    ``L v`` points toward the local centroid; ‖L v‖ is a roughness/curvature
    proxy and -L v is the mean-curvature-flow (smoothing) direction.
    """
    out = np.zeros_like(vertices)
    for i, nb in enumerate(adjacency):
        if nb.size:
            out[i] = vertices[nb].mean(axis=0) - vertices[i]
    return out


def face_normals_areas(vertices: np.ndarray, faces: np.ndarray):
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    nrm = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)
    return nrm, area


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted outward vertex normals (N,3), unit length."""
    fn, fa = face_normals_areas(vertices, faces)
    vn = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn * fa[:, None])
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def mean_curvature(vertices: np.ndarray, faces: np.ndarray,
                   adjacency: list[np.ndarray]) -> np.ndarray:
    """Signed mean curvature per vertex: +convex (bulging out), -concave.

    Uses the umbrella vector projected onto the outward normal:
    concave dents have the umbrella pointing outward (same side as the normal),
    convex bumps have it pointing inward.
    """
    lap = uniform_laplacian(vertices, adjacency)
    nrm = vertex_normals(vertices, faces)
    # -<L v, n>: convex bump -> L v points inward (-n) -> positive.
    return -np.sum(lap * nrm, axis=1)


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(np.abs(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0))


__all__ = [
    "build_adjacency", "uniform_laplacian", "face_normals_areas",
    "vertex_normals", "mean_curvature", "mesh_volume",
]
