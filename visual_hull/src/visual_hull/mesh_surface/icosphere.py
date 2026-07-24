"""Subdivided icosphere — a fixed-topology genus-0 mesh.

A subdivision-level-L icosphere has 20*4^L faces and 10*4^L + 2 vertices, with
uniform triangles.  Because the topology depends only on L, every bubble and
every frame that uses the same L shares vertex indices — which is what makes
per-vertex temporal correspondence (and temporal filtering) trivial.
"""

from __future__ import annotations

import numpy as np

_PHI = (1.0 + 5.0 ** 0.5) / 2.0


def _base_icosahedron() -> tuple[np.ndarray, np.ndarray]:
    v = np.array([
        [-1,  _PHI, 0], [1,  _PHI, 0], [-1, -_PHI, 0], [1, -_PHI, 0],
        [0, -1,  _PHI], [0, 1,  _PHI], [0, -1, -_PHI], [0, 1, -_PHI],
        [_PHI, 0, -1], [_PHI, 0, 1], [-_PHI, 0, -1], [-_PHI, 0, 1],
    ], dtype=np.float64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    return v, f


def icosphere(subdivisions: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices (N,3) on the unit sphere, faces (M,3) int)."""
    vertices, faces = _base_icosahedron()
    vertices = [tuple(row) for row in vertices]
    index_of = {v: i for i, v in enumerate(vertices)}
    midpoint_cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        cached = midpoint_cache.get(key)
        if cached is not None:
            return cached
        pa = np.array(vertices[a]); pb = np.array(vertices[b])
        m = (pa + pb) / 2.0
        m /= np.linalg.norm(m)
        mt = tuple(m)
        idx = index_of.get(mt)
        if idx is None:
            idx = len(vertices)
            vertices.append(mt)
            index_of[mt] = idx
        midpoint_cache[key] = idx
        return idx

    for _ in range(max(int(subdivisions), 0)):
        new_faces = []
        for a, b, c in faces:
            ab = midpoint(a, b); bc = midpoint(b, c); ca = midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = np.asarray(new_faces, dtype=np.int64)

    return np.asarray(vertices, dtype=np.float64), faces


def edges_from_faces(faces: np.ndarray) -> np.ndarray:
    """Unique undirected edges (E,2), sorted per row."""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


__all__ = ["icosphere", "edges_from_faces"]
