"""Virtual-camera silhouette rounding of the limited-view visual hull.

Rounds the faceted 4-view hull into a smooth bubble by carving with many
synthetic orthographic silhouettes, each morphologically rounded at scale
``rho``.  See DESIGN.md.  Public API:

    from visual_hull.virtual_camera import (
        round_bubble_surface, RoundedBubble,
        round_hull_occupancy, VirtualCarveConfig,
    )
"""

from __future__ import annotations

from .carve import VirtualCarveConfig, round_hull_occupancy
from .round_surface import RoundedBubble, round_bubble_surface
from .anisotropic import round_bubble_surface_anisotropic
from .silhouette_refine import refine_bubble_silhouette, RefineConfig
from .views import virtual_directions, orthonormal_basis, viewing_directions

__all__ = [
    "round_bubble_surface", "RoundedBubble",
    "round_bubble_surface_anisotropic",
    "refine_bubble_silhouette", "RefineConfig",
    "round_hull_occupancy", "VirtualCarveConfig",
    "virtual_directions", "orthonormal_basis", "viewing_directions",
]
