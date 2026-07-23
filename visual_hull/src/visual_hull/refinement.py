from __future__ import annotations

import numpy as np
from scipy import ndimage

from .camera import OpenLPTCameraSet
from .hull import vote_visual_hull_for_points
from .surface_utils import points_from_mask


def _shell_points(component_solid: np.ndarray, structure_6, grid_x, grid_y, grid_z):
    """Surface shell (1-voxel skin) of a solid component, as 3D points."""
    eroded = ndimage.binary_erosion(component_solid, structure=structure_6, border_value=0)
    shell = component_solid & ~eroded
    if not np.any(shell):  # single-voxel-thick component — use the solid itself
        shell = component_solid
    return points_from_mask(grid_x, grid_y, grid_z, shell)


def _watershed_split(component_solid: np.ndarray, min_distance: int):
    """Split a solid component at necks via a distance-transform watershed.

    Touching bubbles form one connected blob but each has its own distance-
    transform maximum; watershed on the inverted distance transform cuts them
    apart at the narrow neck between them.  Returns a labeled array (0 = bg).
    """
    distance = ndimage.distance_transform_edt(component_solid)
    if distance.max() <= 0:
        return component_solid.astype(np.int32)
    # Seeds = local maxima of the distance transform (one per bubble core).
    footprint = np.ones((3, 3, 3), dtype=bool)
    local_max = distance == ndimage.maximum_filter(distance, footprint=footprint)
    local_max &= distance > max(float(min_distance), 1.0)
    markers, n = ndimage.label(local_max, structure=footprint)
    if n <= 1:
        return component_solid.astype(np.int32)
    try:
        from skimage.segmentation import watershed
    except Exception:
        return component_solid.astype(np.int32)
    return watershed(-distance, markers, mask=component_solid)


def find_surface_components(
    voxel_volume: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    *,
    mode: str = "shell",
    min_component_voxels: int = 0,
    watershed_min_distance: int = 2,
) -> list[np.ndarray]:
    """Split the combined visual hull into per-bubble surface-point sets.

    ``mode``:
      - ``"shell"``   (legacy): label the eroded 1-voxel shell of the whole
        hull.  Cheap, but a thin/pinched shell can fragment a solid bubble and
        adjacent shells can merge — the source of "dumbbell"/sliver artifacts.
      - ``"filled"``: label the *solid* volume, so a bubble stays one component
        regardless of shell thickness (fixes fragmentation).
      - ``"watershed"``: label the solid volume, then split each blob at its
        necks via a distance-transform watershed (also separates *touching*
        bubbles).

    ``min_component_voxels`` drops components smaller than this (removes sliver
    fragments before they get a bogus SH fit).  Counted on the solid component
    in filled/watershed mode, on the shell in legacy mode.
    """
    structure_6 = ndimage.generate_binary_structure(3, 1)
    structure_26 = np.ones((3, 3, 3), dtype=bool)

    if mode == "shell":
        eroded = ndimage.binary_erosion(voxel_volume, structure=structure_6, border_value=0)
        surface = voxel_volume & ~eroded
        labeled, num_features = ndimage.label(surface, structure=structure_26)
        components: list[np.ndarray] = []
        for label_id in range(1, num_features + 1):
            mask = labeled == label_id
            if not np.any(mask):
                continue
            if min_component_voxels and int(np.count_nonzero(mask)) < min_component_voxels:
                continue
            components.append(points_from_mask(grid_x, grid_y, grid_z, mask))
        return components

    # filled / watershed: work on the solid volume for stable labeling.
    labeled, num_features = ndimage.label(voxel_volume, structure=structure_26)
    components = []
    for label_id in range(1, num_features + 1):
        blob = labeled == label_id
        if not np.any(blob):
            continue
        sub_labels = (
            _watershed_split(blob, watershed_min_distance)
            if mode == "watershed" else blob.astype(np.int32)
        )
        for sub_id in np.unique(sub_labels):
            if sub_id == 0:
                continue
            comp_solid = sub_labels == sub_id
            if min_component_voxels and int(np.count_nonzero(comp_solid)) < min_component_voxels:
                continue
            components.append(_shell_points(comp_solid, structure_6, grid_x, grid_y, grid_z))
    return components


def refine_surface_points(
    surface_points: np.ndarray,
    *,
    coarse_voxel_size: np.ndarray,
    masks: list[np.ndarray],
    cameras: OpenLPTCameraSet,
    mv: int = 2,
    res_inc: int = 3,
) -> np.ndarray:
    points = np.asarray(surface_points, dtype=np.float64)
    coarse_size = np.asarray(coarse_voxel_size, dtype=np.float64)
    fine_size = coarse_size / float(res_inc)

    offset_x = np.arange(-mv * coarse_size[0], mv * coarse_size[0] + fine_size[0] * 0.5, fine_size[0], dtype=np.float64)
    offset_y = np.arange(-mv * coarse_size[1], mv * coarse_size[1] + fine_size[1] * 0.5, fine_size[1], dtype=np.float64)
    offset_z = np.arange(-mv * coarse_size[2], mv * coarse_size[2] + fine_size[2] * 0.5, fine_size[2], dtype=np.float64)
    mesh = np.meshgrid(offset_x, offset_y, offset_z, indexing="xy")
    offsets = np.column_stack([axis.reshape(-1) for axis in mesh])

    candidate_points = (points[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    candidate_points = np.unique(np.round(candidate_points, decimals=10), axis=0)

    voted = vote_visual_hull_for_points(masks, candidate_points, cameras)
    kept = voted[voted[:, 3] >= float(cameras.count), :3]
    return kept.astype(np.float64, copy=False)
