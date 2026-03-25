
# 3D Bubble Surface Reconstruction from Multi-View Cameras

## Overview

This document describes a pipeline for accurate 3D bubble surface reconstruction from multi-view cameras with refractive interfaces (e.g., water tank with glass/acrylic walls).

---

## Pipeline Summary

```
Multi-view Images
       ↓
[1. Binary Segmentation] ──→ Combined masks (all bubbles)
       ↓
[2. Visual Hull] ──→ 3D voxel grid (hierarchical)
       ↓
[3. 3D Region Separation] ──→ Individual bubble voxels (watershed)
       ↓
[4. Surface Initialization] ──→ Spherical harmonics per bubble (regularized fit)
       ↓
[5. Joint Surface Optimization] ──→ Refined surfaces (differentiable rendering)
       ↓
[6. Validation & Output] ──→ Accurate 3D surfaces with volume, area, curvature
```

---

## Step 1: Binary Segmentation

### Input
- Multi-view images from $N_{cam}$ cameras
- Background images (optional, for background subtraction)

### Output
- Combined binary masks $M_c$ for each camera $c$ (all bubbles together)

### Method

**Option A: Simple Thresholding**
```python
def segment_bubbles_threshold(image, background, threshold=25):
    diff = cv2.absdiff(image, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask
```

**Option B: Adaptive Thresholding (for uneven illumination)**
```python
def segment_bubbles_adaptive(image, background, block_size=35, C=10):
    diff = cv2.absdiff(image, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold handles uneven lighting
    mask = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 
        block_size, C
    )
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask
```

### Notes
- Bubbles can overlap in 2D images
- Separation handled later in 3D
- **No instance segmentation required**
- Bubble edges often have specular highlights and dark rings due to refraction—consider edge-aware methods if simple thresholding fails

### Quality Check
```python
def validate_segmentation(mask, expected_fill_ratio=(0.01, 0.3)):
    """Check if segmentation is reasonable"""
    fill_ratio = np.sum(mask > 0) / mask.size
    if not expected_fill_ratio[0] <= fill_ratio <= expected_fill_ratio[1]:
        warnings.warn(f"Unusual mask fill ratio: {fill_ratio:.3f}")
    return fill_ratio
```

---

## Step 2: Visual Hull Reconstruction

### Input
- Combined binary masks $M_c$
- Camera parameters with refractive model

### Output
- 3D binary voxel grid $V$

### Method: Hierarchical Visual Hull

Direct voxelization is memory-intensive. Use a two-pass hierarchical approach:

```python
def hierarchical_visual_hull(masks, cameras, interface_params,
                             bounds, 
                             coarse_res=4.0,  # mm
                             fine_res=0.5):   # mm
    """
    Two-pass visual hull for efficiency
    
    Memory comparison for 200mm × 200mm × 200mm volume:
    - Direct at 0.5mm: 64 million voxels
    - Coarse at 4mm: 125,000 voxels
    - Fine refinement: ~1-2 million voxels (only near surfaces)
    """
    # Pass 1: Coarse voxelization
    coarse_hull = visual_hull(masks, cameras, interface_params, bounds, coarse_res)
    
    # Pass 2: Refine only near surface
    coarse_labels = label(coarse_hull)
    
    refined_regions = []
    for region in regionprops(coarse_labels):
        # Expand bounding box slightly
        bbox = expand_bbox(region.bbox, margin=2*coarse_res)
        
        # Fine voxelization in local region only
        local_hull = visual_hull(masks, cameras, interface_params, bbox, fine_res)
        refined_regions.append(local_hull)
    
    return refined_regions


def expand_bbox(bbox, margin):
    """Expand 3D bounding box by margin"""
    min_coords = np.array(bbox[:3]) - margin
    max_coords = np.array(bbox[3:]) + margin
    return tuple(min_coords) + tuple(max_coords)
```

### Refractive Projection Model

For a 3D point $\vec{X}$, find $L_1$ by solving:

$$f(L_1, \vec{X}) = L_{tot}(\vec{X})$$

where:

$$f(L_1, \vec{X}) = \sum_{i=1}^{N} L_i = L_1 + \sum_{i=2}^{N} \frac{L_1}{\sqrt{H_1^2 + \alpha_i L_1^2}} \cdot \frac{n_1}{n_i} \cdot H_i$$

with $\alpha_i = 1 - \frac{n_1^2}{n_i^2}$ and $H_1 = (\vec{X} - \vec{P}_1) \cdot \vec{n}$.

### Visual Hull Core Algorithm

```python
def visual_hull(masks, cameras, interface_params, bounds, resolution):
    """
    Compute visual hull via back-projection intersection
    
    Args:
        masks: list of binary masks
        cameras: list of camera parameter dicts
        interface_params: refractive interface parameters
        bounds: ((x_min, y_min, z_min), (x_max, y_max, z_max))
        resolution: voxel size in mm
    
    Returns:
        Binary voxel grid
    """
    # Create voxel grid
    x = np.arange(bounds[0][0], bounds[1][0], resolution)
    y = np.arange(bounds[0][1], bounds[1][1], resolution)
    z = np.arange(bounds[0][2], bounds[1][2], resolution)
    
    # Initialize all voxels as occupied
    hull = np.ones((len(x), len(y), len(z)), dtype=bool)
    
    # Intersect with each camera's viewing cone
    for cam_idx, (mask, cam) in enumerate(zip(masks, cameras)):
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                for k, zk in enumerate(z):
                    if not hull[i, j, k]:
                        continue  # Already carved out
                    
                    # Project point through refractive interface
                    point_3d = np.array([xi, yj, zk])
                    pixel = refractive_project(point_3d, cam, interface_params)
                    
                    # Check if inside mask
                    u, v = int(round(pixel[0])), int(round(pixel[1]))
                    if 0 <= u < mask.shape[1] and 0 <= v < mask.shape[0]:
                        if mask[v, u] == 0:
                            hull[i, j, k] = False
                    else:
                        hull[i, j, k] = False  # Outside image
    
    return hull
```

### Memory Estimates

| Volume Size | Resolution | Voxels | Memory (bool) |
|-------------|------------|--------|---------------|
| 200³ mm³ | 2.0 mm | 1M | 1 MB |
| 200³ mm³ | 1.0 mm | 8M | 8 MB |
| 200³ mm³ | 0.5 mm | 64M | 64 MB |
| 200³ mm³ | 0.25 mm | 512M | 512 MB |

---

## Step 3: 3D Region Separation

### Input
- 3D binary voxel grid $V$

### Output
- List of bubble regions, each with:
  - Voxel coordinates
  - Centroid
  - Bounding box
  - Approximate volume

### Method: Watershed Separation

Simple connected components (`label`) will merge touching bubbles. Use watershed for robust separation:

```python
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

def separate_bubbles(binary_volume, min_distance=5, min_size=100):
    """
    Separate touching bubbles using distance transform + watershed
    
    Args:
        binary_volume: 3D binary array from visual hull
        min_distance: minimum distance between bubble centers (voxels)
        min_size: minimum bubble size in voxels
    
    Returns:
        labels: labeled volume with separate regions
        regions: list of regionprops objects
    """
    # Distance transform - distance to nearest background
    distance = ndimage.distance_transform_edt(binary_volume)
    
    # Find local maxima (bubble centers)
    coords = peak_local_max(
        distance, 
        min_distance=min_distance,
        labels=binary_volume,
        exclude_border=False
    )
    
    # Create markers for watershed
    markers = np.zeros_like(binary_volume, dtype=np.int32)
    for i, coord in enumerate(coords):
        markers[tuple(coord)] = i + 1
    
    # Watershed segmentation
    # Use negative distance so watershed fills from peaks
    labels = watershed(-distance, markers, mask=binary_volume)
    
    # Remove small regions
    regions = regionprops(labels)
    for region in regions:
        if region.area < min_size:
            labels[labels == region.label] = 0
    
    # Relabel to ensure consecutive labels
    labels, num_labels = ndimage.label(labels > 0)
    regions = regionprops(labels)
    
    return labels, regions


def validate_separation(labels, expected_range=(1, 100)):
    """Sanity check on number of detected bubbles"""
    n_regions = labels.max()
    if not expected_range[0] <= n_regions <= expected_range[1]:
        warnings.warn(f"Unexpected number of bubbles: {n_regions}")
    return n_regions
```

### Alternative: Simple Connected Components

For well-separated bubbles, simple labeling may suffice:

```python
def separate_bubbles_simple(binary_volume, min_size=100):
    """Simple connected components (use when bubbles don't touch)"""
    labels = label(binary_volume)
    regions = regionprops(labels)
    
    # Filter by size
    for region in regions:
        if region.area < min_size:
            labels[labels == region.label] = 0
    
    return labels, regionprops(labels)
```

### Choosing Between Methods

| Condition | Recommended Method |
|-----------|-------------------|
| Bubbles well-separated | Simple connected components |
| Bubbles may touch | Watershed separation |
| Dense bubble clouds | Watershed + careful tuning |

---

## Step 4: Surface Initialization

### Input
- Bubble voxels from regionprops

### Output
- Initial spherical harmonics surface per bubble

### Surface Representation

Radius as function of spherical angles:

$$r(\theta, \phi) = \sum_{l=0}^{L_{max}} \sum_{m=-l}^{l} c_{lm} Y_l^m(\theta, \phi)$$

### Parameters per Bubble

| Parameter | Description | Size |
|-----------|-------------|------|
| $\vec{x}_c$ | Centroid position | 3 |
| $c_{lm}$ | SH coefficients | $(L_{max}+1)^2$ |

### Implementation

```python
import numpy as np
from scipy.special import sph_harm
from scipy.linalg import lstsq

class SphericalHarmonicsSurface:
    def __init__(self, L_max=8):
        self.L_max = L_max
        self.n_coeffs = (L_max + 1) ** 2
        self.coeffs = np.zeros(self.n_coeffs)
        self.center = np.zeros(3)
    
    def _sh_index(self, l, m):
        """Convert (l,m) to flat index"""
        return l * l + l + m
    
    def _compute_basis(self, theta, phi):
        """
        Compute real spherical harmonics basis matrix
        
        Args:
            theta: polar angles (N,)
            phi: azimuthal angles (N,)
        
        Returns:
            Y: basis matrix (N, n_coeffs)
        """
        n_points = len(theta)
        Y = np.zeros((n_points, self.n_coeffs))
        
        for l in range(self.L_max + 1):
            for m in range(-l, l + 1):
                idx = self._sh_index(l, m)
                
                # Real spherical harmonics
                if m < 0:
                    # Y_l^{-|m|} = sqrt(2) * Im(Y_l^|m|)
                    Y[:, idx] = np.sqrt(2) * (-1)**m * \
                                np.imag(sph_harm(abs(m), l, phi, theta))
                elif m > 0:
                    # Y_l^m = sqrt(2) * Re(Y_l^m)
                    Y[:, idx] = np.sqrt(2) * (-1)**m * \
                                np.real(sph_harm(m, l, phi, theta))
                else:
                    # Y_l^0 is real
                    Y[:, idx] = np.real(sph_harm(0, l, phi, theta))
        
        return Y
    
    def fit_from_voxels(self, voxel_coords, regularization=1e-4):
        """
        Fit SH surface to voxel coordinates with Tikhonov regularization
        
        Args:
            voxel_coords: (N, 3) array of voxel positions
            regularization: regularization strength (higher = smoother)
        
        Returns:
            residual: RMS fitting residual
        """
        # Compute centroid
        self.center = np.mean(voxel_coords, axis=0)
        
        # Convert to spherical coordinates
        rel_coords = voxel_coords - self.center
        r = np.linalg.norm(rel_coords, axis=1)
        
        # Handle points at origin
        r_safe = np.maximum(r, 1e-10)
        theta = np.arccos(np.clip(rel_coords[:, 2] / r_safe, -1, 1))
        phi = np.arctan2(rel_coords[:, 1], rel_coords[:, 0])
        
        # Build design matrix
        Y = self._compute_basis(theta, phi)
        
        # Tikhonov regularization - penalize higher frequencies more
        weights = np.array([
            l**2 for l in range(self.L_max + 1) 
            for m in range(-l, l + 1)
        ])
        Gamma = np.diag(regularization * weights)
        
        # Solve regularized least squares: (Y^T Y + Gamma) c = Y^T r
        A = Y.T @ Y + Gamma
        b = Y.T @ r
        self.coeffs = np.linalg.solve(A, b)
        
        # Compute fit quality
        r_fit = Y @ self.coeffs
        residual = np.sqrt(np.mean((r - r_fit)**2))
        
        return residual
    
    def radius(self, theta, phi):
        """Evaluate radius at given angles"""
        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)
        Y = self._compute_basis(theta, phi)
        return Y @ self.coeffs
    
    def surface_points(self, n_theta=50, n_phi=100):
        """Generate surface point cloud"""
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
        
        r = self.radius(theta_grid.ravel(), phi_grid.ravel())
        r = r.reshape(theta_grid.shape)
        
        # Convert to Cartesian
        x = r * np.sin(theta_grid) * np.cos(phi_grid) + self.center[0]
        y = r * np.sin(theta_grid) * np.sin(phi_grid) + self.center[1]
        z = r * np.cos(theta_grid) + self.center[2]
        
        return x, y, z
    
    def volume(self, n_theta=50, n_phi=100):
        """Compute enclosed volume via numerical integration"""
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        dtheta = np.pi / n_theta
        dphi = 2 * np.pi / n_phi
        
        vol = 0.0
        for t in theta:
            for p in phi:
                r = self.radius(t, p)
                # Volume element in spherical coords: (1/3) r^3 sin(θ) dθ dφ
                vol += (1/3) * r**3 * np.sin(t) * dtheta * dphi
        
        return vol
    
    def surface_area(self, n_theta=50, n_phi=100):
        """Compute surface area via numerical integration"""
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        dtheta = np.pi / n_theta
        dphi = 2 * np.pi / n_phi
        
        area = 0.0
        eps = 1e-6
        
        for t in theta[1:-1]:  # Avoid poles
            for p in phi:
                r = self.radius(t, p)
                
                # Numerical derivatives
                dr_dtheta = (self.radius(t + eps, p) - self.radius(t - eps, p)) / (2 * eps)
                dr_dphi = (self.radius(t, p + eps) - self.radius(t, p - eps)) / (2 * eps)
                
                # Surface area element
                # dA = sqrt(r^4 sin^2(θ) + r^2 sin^2(θ) (dr/dθ)^2 + r^2 (dr/dφ)^2) dθ dφ
                dA = np.sqrt(
                    r**4 * np.sin(t)**2 + 
                    r**2 * np.sin(t)**2 * dr_dtheta**2 + 
                    r**2 * dr_dphi**2
                ) * dtheta * dphi
                
                area += dA
        
        return area
```

### Adaptive L_max Selection

```python
def choose_L_max(voxel_coords, max_L=20, residual_threshold=0.05):
    """
    Automatically choose L_max based on fitting residual
    
    Args:
        voxel_coords: (N, 3) array of voxel positions
        max_L: maximum L to try
        residual_threshold: target relative residual
    
    Returns:
        Optimal L_max value
    """
    # Compute mean radius for normalization
    center = np.mean(voxel_coords, axis=0)
    rel_coords = voxel_coords - center
    r = np.linalg.norm(rel_coords, axis=1)
    mean_r = np.mean(r)
    
    for L in range(2, max_L + 1, 2):
        surface = SphericalHarmonicsSurface(L_max=L)
        residual = surface.fit_from_voxels(voxel_coords)
        
        relative_residual = residual / mean_r
        if relative_residual < residual_threshold:
            print(f"Selected L_max={L} with relative residual {relative_residual:.4f}")
            return L
    
    print(f"Using max L_max={max_L}, residual may be high")
    return max_L
```

### Star-Shaped Validation

```python
def check_star_shaped(voxel_coords, center=None, threshold=0.15):
    """
    Check if bubble is approximately star-shaped from given center
    
    A surface is star-shaped if every ray from the center intersects
    the surface exactly once.
    
    Args:
        voxel_coords: (N, 3) array
        center: center point (default: centroid)
        threshold: maximum allowed coefficient of variation per angular bin
    
    Returns:
        is_star_shaped: bool
        violation_fraction: fraction of angular bins with multiple radii
    """
    if center is None:
        center = np.mean(voxel_coords, axis=0)
    
    rel_coords = voxel_coords - center
    r = np.linalg.norm(rel_coords, axis=1)
    r_safe = np.maximum(r, 1e-10)
    
    theta = np.arccos(np.clip(rel_coords[:, 2] / r_safe, -1, 1))
    phi = np.arctan2(rel_coords[:, 1], rel_coords[:, 0])
    
    # Bin by angle
    n_bins = 15
    theta_bins = np.digitize(theta, np.linspace(0, np.pi, n_bins + 1))
    phi_bins = np.digitize(phi, np.linspace(-np.pi, np.pi, n_bins + 1))
    
    violations = 0
    total_bins = 0
    
    for tb in range(1, n_bins + 1):
        for pb in range(1, n_bins + 1):
            mask = (theta_bins == tb) & (phi_bins == pb)
            if np.sum(mask) > 2:
                total_bins += 1
                r_std = np.std(r[mask])
                r_mean = np.mean(r[mask])
                if r_mean > 0 and r_std / r_mean > threshold:
                    violations += 1
    
    violation_fraction = violations / max(total_bins, 1)
    is_star_shaped = violation_fraction < 0.1
    
    return is_star_shaped, violation_fraction
```

### Recommended Parameters

| Bubble Type | $L_{max}$ | Number of Coefficients | Regularization |
|-------------|-----------|------------------------|----------------|
| Nearly spherical | 4-6 | 25-49 | 1e-3 |
| Elongated smooth | 8-12 | 81-169 | 1e-4 |
| Complex shape | 15-20 | 256-441 | 1e-5 |

---

## Step 5: Joint Surface Optimization

### Overview

Jointly optimize all bubble surfaces to match observed silhouettes across all cameras.

### Objective Function

$$\min_{\{S_k\}} \sum_{c=1}^{N_{cam}} \mathcal{L}_{sil}^{(c)} + \lambda_1 \sum_k \mathcal{L}_{smooth}^{(k)} + \lambda_2 \mathcal{L}_{overlap} + \lambda_3 \sum_k \mathcal{L}_{volume}^{(k)}$$

### 5.1 Silhouette Loss

$$\mathcal{L}_{sil}^{(c)} = \mathcal{L}\left( \bigcup_k \text{Project}_c(S_k), M_c \right)$$

Key: Compare **union of all projected surfaces** against **combined mask**.

#### Differentiable Silhouette Rendering

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableSilhouetteRenderer(nn.Module):
    def __init__(self, image_size, sigma=2.0):
        """
        Soft silhouette renderer using Gaussian splatting
        
        Args:
            image_size: (H, W) tuple
            sigma: Gaussian splat standard deviation in pixels
        """
        super().__init__()
        self.H, self.W = image_size
        self.sigma = sigma
        
        # Pre-compute pixel coordinates
        y, x = torch.meshgrid(
            torch.arange(self.H), 
            torch.arange(self.W), 
            indexing='ij'
        )
        self.register_buffer('pixel_x', x.float())
        self.register_buffer('pixel_y', y.float())
    
    def forward(self, projected_points):
        """
        Render soft silhouette from projected 2D points
        
        Args:
            projected_points: (N, 2) tensor of (u, v) coordinates
            
        Returns:
            silhouette: (H, W) soft silhouette image in [0, 1]
        """
        if len(projected_points) == 0:
            return torch.zeros(self.H, self.W, device=self.pixel_x.device)
        
        u = projected_points[:, 0]  # (N,)
        v = projected_points[:, 1]  # (N,)
        
        # Compute distance from each pixel to each point
        # Shape: (H, W, N)
        dx = self.pixel_x.unsqueeze(-1) - u.unsqueeze(0).unsqueeze(0)
        dy = self.pixel_y.unsqueeze(-1) - v.unsqueeze(0).unsqueeze(0)
        
        dist_sq = dx**2 + dy**2
        
        # Gaussian splat
        gaussians = torch.exp(-dist_sq / (2 * self.sigma**2))
        
        # Max over all points (soft union)
        silhouette, _ = torch.max(gaussians, dim=-1)
        
        return silhouette


class MultiSurfaceSilhouetteRenderer(nn.Module):
    """Render union of multiple bubble surfaces"""
    
    def __init__(self, image_size, sigma=2.0):
        super().__init__()
        self.renderer = DifferentiableSilhouetteRenderer(image_size, sigma)
    
    def forward(self, surfaces_projected_points):
        """
        Args:
            surfaces_projected_points: list of (N_k, 2) tensors
            
        Returns:
            combined_silhouette: (H, W) tensor
        """
        if len(surfaces_projected_points) == 0:
            return torch.zeros(
                self.renderer.H, 
                self.renderer.W, 
                device=self.renderer.pixel_x.device
            )
        
        silhouettes = []
        for points in surfaces_projected_points:
            sil = self.renderer(points)
            silhouettes.append(sil)
        
        # Soft union via max
        combined = torch.stack(silhouettes, dim=0)
        return torch.max(combined, dim=0)[0]
```

#### Loss Function Options

```python
def silhouette_loss(rendered, target, method='bce'):
    """
    Compute silhouette matching loss
    
    Args:
        rendered: (H, W) predicted silhouette in [0, 1]
        target: (H, W) ground truth binary mask
        method: 'bce', 'mse', 'focal', or 'iou'
    """
    rendered = rendered.clamp(1e-6, 1 - 1e-6)
    
    if method == 'bce':
        # Binary cross-entropy
        return F.binary_cross_entropy(rendered, target)
    
    elif method == 'mse':
        # Mean squared error
        return F.mse_loss(rendered, target)
    
    elif method == 'focal':
        # Focal loss - handles class imbalance
        gamma = 2.0
        bce = F.binary_cross_entropy(rendered, target, reduction='none')
        p_t = rendered * target + (1 - rendered) * (1 - target)
        focal_weight = (1 - p_t) ** gamma
        return (focal_weight * bce).mean()
    
    elif method == 'iou':
        # Soft IoU loss
        intersection = (rendered * target).sum()
        union = rendered.sum() + target.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        return 1 - iou
    
    else:
        raise ValueError(f"Unknown loss method: {method}")
```

### 5.2 Smoothness Loss

Penalize high-frequency spherical harmonic components:

$$\mathcal{L}_{smooth}^{(k)} = \sum_{l=0}^{L_{max}} \sum_{m=-l}^{l} w_l \cdot c_{lm}^2$$

where $w_l = l^2(l+1)^2$ (biharmonic) or $w_l = l(l+1)$ (Laplacian).

```python
def smoothness_loss(coeffs, L_max, weighting='biharmonic'):
    """
    Smoothness regularization on SH coefficients
    
    Args:
        coeffs: (n_coeffs,) tensor of SH coefficients
        L_max: maximum spherical harmonic degree
        weighting: 'laplacian' or 'biharmonic'
    """
    loss = 0.0
    idx = 0
    
    for l in range(L_max + 1):
        if weighting == 'laplacian':
            weight = l * (l + 1)
        elif weighting == 'biharmonic':
            weight = (l * (l + 1)) ** 2
        else:
            weight = l ** 2
        
        for m in range(-l, l + 1):
            loss = loss + weight * coeffs[idx] ** 2
            idx += 1
    
    return loss
```

### 5.3 Overlap Penalty

Prevent bubbles from intersecting:

$$\mathcal{L}_{overlap} = \sum_{i < j} \left[ \max(0, r_i + r_j - \|\vec{x}_c^{(i)} - \vec{x}_c^{(j)}\| + \epsilon) \right]^2$$

where $r_i = |c_{00}^{(i)}| \cdot \sqrt{4\pi}$ is the mean radius of bubble $i$.

```python
def overlap_loss(centers, coeffs_list, margin=0.5):
    """
    Penalize overlapping bubbles
    
    Args:
        centers: list of (3,) tensors - bubble centers
        coeffs_list: list of coefficient tensors
        margin: minimum separation margin in mm
    """
    n_bubbles = len(centers)
    if n_bubbles < 2:
        return torch.tensor(0.0)
    
    loss = 0.0
    n_pairs = 0
    
    for i in range(n_bubbles):
        for j in range(i + 1, n_bubbles):
            # Mean radii from c_00 coefficient
            # c_00 = r_mean / sqrt(4π) for a sphere
            r_i = torch.abs(coeffs_list[i][0]) * np.sqrt(4 * np.pi)
            r_j = torch.abs(coeffs_list[j][0]) * np.sqrt(4 * np.pi)
            
            # Distance between centers
            d = torch.norm(centers[i] - centers[j])
            
            # Overlap penalty with margin
            overlap = torch.relu(r_i + r_j - d + margin)
            loss = loss + overlap ** 2
            n_pairs += 1
    
    return loss / n_pairs
```

### 5.4 Volume Conservation (Optional)

Soft constraint to preserve approximate volume from visual hull:

$$\mathcal{L}_{volume}^{(k)} = \left( \frac{V_k - V_{k,0}}{V_{k,0}} \right)^2$$

```python
def volume_loss(coeffs, target_volume, L_max):
    """
    Soft constraint on bubble volume
    
    For nearly spherical bubbles:
    V ≈ (4π/3) * (c_00 * sqrt(4π))^3
    """
    c_00 = coeffs[0]
    mean_radius = c_00 * np.sqrt(4 * np.pi)
    approx_volume = (4/3) * np.pi * mean_radius ** 3
    
    relative_error = (approx_volume - target_volume) / target_volume
    return relative_error ** 2
```

### 5.5 Complete Optimization Implementation

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

class BubbleSurfaceOptimizer:
    def __init__(self, initial_surfaces, cameras, masks, 
                 interface_params, config=None):
        """
        Joint optimization of all bubble surfaces
        
        Args:
            initial_surfaces: list of SphericalHarmonicsSurface objects
            cameras: list of camera parameter dicts
            masks: list of (H, W) binary mask arrays
            interface_params: refractive interface parameters
            config: dict with optimization parameters
        """
        self.config = config or {}
        self.n_bubbles = len(initial_surfaces)
        self.n_cameras = len(cameras)
        self.cameras = cameras
        self.interface_params = interface_params
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert masks to tensors
        self.masks = [
            torch.from_numpy(m.astype(np.float32)).to(self.device) 
            for m in masks
        ]
        
        # Initialize optimizable parameters
        self.centers = nn.ParameterList([
            nn.Parameter(torch.from_numpy(s.center.copy()).float().to(self.device))
            for s in initial_surfaces
        ])
        
        self.sh_coeffs = nn.ParameterList([
            nn.Parameter(torch.from_numpy(s.coeffs.copy()).float().to(self.device))
            for s in initial_surfaces
        ])
        
        # Store L_max and initial volumes for each surface
        self.L_max = [s.L_max for s in initial_surfaces]
        self.initial_volumes = [s.volume() for s in initial_surfaces]
        
        # Setup renderers
        H, W = masks[0].shape
        self.renderers = [
            MultiSurfaceSilhouetteRenderer(
                (H, W), 
                sigma=self.config.get('sigma', 2.0)
            ).to(self.device)
            for _ in range(self.n_cameras)
        ]
    
    def sample_surface_points(self, bubble_idx, n_theta=30, n_phi=60):
        """
        Sample points on bubble surface for rendering
        """
        center = self.centers[bubble_idx]
        coeffs = self.sh_coeffs[bubble_idx]
        L_max = self.L_max[bubble_idx]
        
        # Create angular grid
        theta = torch.linspace(0, np.pi, n_theta, device=self.device)
        phi = torch.linspace(0, 2*np.pi, n_phi, device=self.device)
        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')
        theta_flat = theta_grid.reshape(-1)
        phi_flat = phi_grid.reshape(-1)
        
        # Compute SH basis and radius
        Y = self._compute_sh_basis_torch(theta_flat, phi_flat, L_max)
        r = Y @ coeffs
        
        # Ensure positive radius
        r = torch.relu(r) + 1e-6
        
        # Convert to Cartesian
        x = r * torch.sin(theta_flat) * torch.cos(phi_flat) + center[0]
        y = r * torch.sin(theta_flat) * torch.sin(phi_flat) + center[1]
        z = r * torch.cos(theta_flat) + center[2]
        
        return torch.stack([x, y, z], dim=-1)
    
    def _compute_sh_basis_torch(self, theta, phi, L_max):
        """Compute real spherical harmonics basis in PyTorch"""
        # Implementation similar to numpy version but with torch tensors
        # [Full implementation would go here]
        pass
    
    def compute_silhouette_loss(self):
        """Compute silhouette matching loss across all cameras"""
        total_loss = 0.0
        
        for cam_idx in range(self.n_cameras):
            # Project all bubble surfaces
            projected_points = []
            
            for bubble_idx in range(self.n_bubbles):
                points_3d = self.sample_surface_points(bubble_idx)
                points_2d = self._refractive_project_torch(
                    points_3d, 
                    self.cameras[cam_idx]
                )
                projected_points.append(points_2d)
            
            # Render combined silhouette
            rendered = self.renderers[cam_idx](projected_points)
            
            # Compare with mask
            target = self.masks[cam_idx]
            
            loss_type = self.config.get('silhouette_loss', 'bce')
            loss = silhouette_loss(rendered, target, method=loss_type)
            total_loss = total_loss + loss
        
        return total_loss / self.n_cameras
    
    def compute_smoothness_loss(self):
        """Smoothness regularization on all bubbles"""
        total_loss = 0.0
        
        for bubble_idx in range(self.n_bubbles):
            coeffs = self.sh_coeffs[bubble_idx]
            L_max = self.L_max[bubble_idx]
            loss = smoothness_loss(coeffs, L_max, weighting='biharmonic')
            total_loss = total_loss + loss
        
        return total_loss / self.n_bubbles
    
    def compute_overlap_loss(self):
        """Overlap penalty between bubbles"""
        return overlap_loss(
            list(self.centers), 
            list(self.sh_coeffs),
            margin=self.config.get('overlap_margin', 0.5)
        )
    
    def compute_volume_loss(self):
        """Volume conservation loss"""
        total_loss = 0.0
        
        for bubble_idx in range(self.n_bubbles):
            coeffs = self.sh_coeffs[bubble_idx]
            target_vol = self.initial_volumes[bubble_idx]
            loss = volume_loss(coeffs, target_vol, self.L_max[bubble_idx])
            total_loss = total_loss + loss
        
        return total_loss / self.n_bubbles
    
    def compute_total_loss(self):
        """Combined objective function"""
        L_sil = self.compute_silhouette_loss()
        L_smooth = self.compute_smoothness_loss()
        L_overlap = self.compute_overlap_loss()
        L_volume = self.compute_volume_loss()
        
        lambda_1 = self.config.get('lambda_smooth', 0.01)
        lambda_2 = self.config.get('lambda_overlap', 0.1)
        lambda_3 = self.config.get('lambda_volume', 0.01)
        
        total = L_sil + lambda_1 * L_smooth + lambda_2 * L_overlap + lambda_3 * L_volume
        
        return total, {
            'silhouette': L_sil.item(),
            'smoothness': L_smooth.item(),
            'overlap': L_overlap.item(),
            'volume': L_volume.item(),
            'total': total.item()
        }
    
    def optimize(self, n_iterations=500, lr=0.001, verbose=True):
        """
        Run optimization
        
        Args:
            n_iterations: number of optimization steps
            lr: learning rate
            verbose: print progress
        
        Returns:
            history: dict of loss histories
            final_surfaces: list of optimized SphericalHarmonicsSurface objects
        """
        # Setup optimizer
        params = list(self.centers.parameters()) + list(self.sh_coeffs.parameters())
        optimizer = Adam(params, lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
        
        history = {
            'total': [], 'silhouette': [], 'smoothness': [], 
            'overlap': [], 'volume': []
        }
        
        best_loss = float('inf')
        best_state = None
        
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            loss, loss_dict = self.compute_total_loss()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            
            optimizer.step()
            scheduler.step(loss)
            
            # Record history
            for key in history:
                history[key].append(loss_dict[key])
            
            # Save best
            if loss_dict['total'] < best_loss:
                best_loss = loss_dict['total']
                best_state = {
                    'centers': [c.data.clone() for c in self.centers],
                    'coeffs': [c.data.clone() for c in self.sh_coeffs]
                }
            
            # Logging
            if verbose and iteration % 50 == 0:
                print(f"Iter {iteration:4d}: "
                      f"Total={loss_dict['total']:.4f}, "
                      f"Sil={loss_dict['silhouette']:.4f}, "
                      f"Smooth={loss_dict['smoothness']:.4f}, "
                      f"Overlap={loss_dict['overlap']:.4f}")
        
        # Restore best state
        if best_state is not None:
            for i, (c, coef) in enumerate(zip(best_state['centers'], best_state['coeffs'])):
                self.centers[i].data = c
                self.sh_coeffs[i].data = coef
        
        # Convert to numpy surfaces
        final_surfaces = self._extract_surfaces()
        
        return history, final_surfaces
    
    def _extract_surfaces(self):
        """Convert optimized parameters back to SphericalHarmonicsSurface objects"""
        surfaces = []
        
        for i in range(self.n_bubbles):
            surface = SphericalHarmonicsSurface(L_max=self.L_max[i])
            surface.center = self.centers[i].detach().cpu().numpy()
            surface.coeffs = self.sh_coeffs[i].detach().cpu().numpy()
            surfaces.append(surface)
        
        return surfaces
```

### 5.6 Recommended Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lambda_smooth` | 0.01 | Smoothness weight |
| `lambda_overlap` | 0.1 | Overlap penalty weight |
| `lambda_volume` | 0.01 | Volume conservation weight |
| `sigma` | 2.0 | Gaussian splat size (pixels) |
| `overlap_margin` | 0.5 | Minimum bubble separation (mm) |
| Learning rate | 0.001 | Adam optimizer |
| Iterations | 500-1000 | Until convergence |
| Gradient clipping | 1.0 | Max gradient norm |

---

## Step 6: Gradient Computation for Refractive Projection

### Implicit Function Theorem

From $g(L_1, \vec{X}) = f(L_1, \vec{X}) - L_{tot}(\vec{X}) = 0$:

$$\frac{\partial L_1}{\partial \vec{X}} = \frac{1}{A}\left(\hat{d} + B \cdot \vec{n}\right)$$

where:

$$A = \frac{\partial f}{\partial L_1}\bigg|_{\vec{X}} = 1 + \sum_{i=2}^{N} \frac{n_1 H_i H_1^2}{n_i D_i^{3/2}}$$

$$B = -\frac{\partial f}{\partial H_1}\bigg|_{L_1} = \sum_{i=2}^{N} \frac{n_1 H_i L_1 H_1}{n_i D_i^{3/2}}$$

$$D_i = H_1^2 + \alpha_i L_1^2$$

$$\hat{d} = \frac{\vec{X}_{\parallel} - \vec{P}_{0,\parallel}}{L_{tot}}$$

### Pixel Coordinate Gradient

$$\frac{\partial \vec{u}}{\partial L_1} = \frac{1}{H_0 L_{tot}} \begin{pmatrix} f_x & 0 \\ 0 & f_y \end{pmatrix} (\vec{X}_{\parallel} - \vec{P}_{0,\parallel})_{\perp}$$

### Full Chain Rule

$$\frac{\partial \vec{u}}{\partial \vec{X}} = \frac{\partial \vec{u}}{\partial L_1} \cdot \frac{\partial L_1}{\partial \vec{X}} + \frac{\partial \vec{u}}{\partial \vec{X}}\bigg|_{L_1}$$

### PyTorch Custom Autograd Implementation

```python
class RefractiveProjection(torch.autograd.Function):
    """
    Custom autograd function for refractive projection with analytical gradients
    """
    
    @staticmethod
    def forward(ctx, points_3d, camera_params, interface_params):
        """
        Forward pass: project 3D points to 2D through refractive interfaces
        
        Args:
            points_3d: (N, 3) tensor
            camera_params: dict with intrinsics and extrinsics
            interface_params: dict with interface geometry and refractive indices
        
        Returns:
            pixels: (N, 2) tensor of pixel coordinates
        """
        device = points_3d.device
        points_np = points_3d.detach().cpu().numpy()
        
        # Call existing refractive projection code
        pixels, L1_values, aux_data = refractive_project_with_aux(
            points_np, camera_params, interface_params
        )
        
        # Save for backward pass
        ctx.save_for_backward(points_3d)
        ctx.camera_params = camera_params
        ctx.interface_params = interface_params
        ctx.L1_values = L1_values
        ctx.aux_data = aux_data
        
        return torch.from_numpy(pixels).float().to(device)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients using analytical formulas
        """
        points_3d, = ctx.saved_tensors
        device = grad_output.device
        
        points_np = points_3d.detach().cpu().numpy()
        L1 = ctx.L1_values
        aux = ctx.aux_data
        
        # Compute analytical Jacobians du/dX (N, 2, 3)
        jacobians = compute_refractive_jacobians(
            points_np, L1, aux,
            ctx.camera_params, 
            ctx.interface_params
        )
        
        jacobians = torch.from_numpy(jacobians).float().to(device)
        
        # Chain rule: grad_points[i] = sum_j grad_output[i,j] * jacobians[i,j,:]
        grad_points = torch.einsum('nj,njk->nk', grad_output, jacobians)
        
        return grad_points, None, None


def compute_refractive_jacobians(points, L1, aux, camera_params, interface_params):
    """
    Compute analytical Jacobians for refractive projection
    
    Returns:
        jacobians: (N, 2, 3) array where jacobians[i] = d(u,v)/d(x,y,z)
    """
    N = len(points)
    jacobians = np.zeros((N, 2, 3))
    
    # Extract interface parameters
    n = interface_params['refractive_indices']  # [n1, n2, ...]
    H = aux['H_values']  # Layer thicknesses for each point
    
    for i in range(N):
        # Compute A and B terms
        A = 1.0
        B = 0.0
        
        for layer in range(1, len(n)):
            alpha = 1 - (n[0] / n[layer]) ** 2
            D = H[i, 0]**2 + alpha * L1[i]**2
            
            A += (n[0] * H[i, layer] * H[i, 0]**2) / (n[layer] * D**1.5)
            B += (n[0] * H[i, layer] * L1[i] * H[i, 0]) / (n[layer] * D**1.5)
        
        # Compute dL1/dX
        L_tot = aux['L_tot'][i]
        d_hat = aux['d_hat'][i]  # Unit direction (3,)
        normal = aux['normal']   # Interface normal (3,)
        
        dL1_dX = (d_hat + B * normal) / A
        
        # Compute du/dL1
        fx, fy = camera_params['fx'], camera_params['fy']
        H0 = aux['H0'][i]
        X_parallel = aux['X_parallel'][i]
        
        du_dL1 = np.array([fx, fy]) * X_parallel[:2] / (H0 * L_tot)
        
        # Compute du/dX at fixed L1
        du_dX_fixed = compute_direct_projection_jacobian(
            points[i], camera_params, interface_params, L1[i]
        )
        
        # Chain rule
        jacobians[i] = np.outer(du_dL1, dL1_dX) + du_dX_fixed
    
    return jacobians
```

### Gradient Validation

```python
def validate_gradients(projector, points_3d, camera_params, interface_params, eps=1e-5):
    """
    Validate analytical gradients against numerical gradients
    """
    points_3d.requires_grad_(True)
    
    # Analytical gradient
    pixels = projector(points_3d, camera_params, interface_params)
    loss = pixels.sum()
    loss.backward()
    analytical_grad = points_3d.grad.clone()
    
    # Numerical gradient
    numerical_grad = torch.zeros_like(points_3d)
    
    for i in range(points_3d.shape[0]):
        for j in range(3):
            points_plus = points_3d.detach().clone()
            points_minus = points_3d.detach().clone()
            
            points_plus[i, j] += eps
            points_minus[i, j] -= eps
            
            pixels_plus = projector(points_plus, camera_params, interface_params)
            pixels_minus = projector(points_minus, camera_params, interface_params)
            
            numerical_grad[i, j] = (pixels_plus.sum() - pixels_minus.sum()) / (2 * eps)
    
    # Compare
    error = torch.abs(analytical_grad - numerical_grad)
    relative_error = error / (torch.abs(numerical_grad) + 1e-8)
    
    print(f"Max absolute error: {error.max().item():.2e}")
    print(f"Max relative error: {relative_error.max().item():.2e}")
    
    return error.max().item() < 1e-4
```

---

## Step 7: Output - Surface Properties

For each reconstructed bubble surface $S_k$:

### Geometric Properties

| Property | Formula | Description |
|----------|---------|-------------|
| **Volume** | $V = \frac{1}{3} \oint_S \vec{r} \cdot \hat{n} \, dA$ | Enclosed volume |
| **Surface Area** | $A = \oint_S dA$ | Total surface area |
| **Centroid** | $\vec{x}_c$ | Optimized center position |
| **Equivalent Diameter** | $D_{eq} = \left(\frac{6V}{\pi}\right)^{1/3}$ | Diameter of equal-volume sphere |
| **Sphericity** | $\Psi = \frac{\pi^{1/3}(6V)^{2/3}}{A}$ | 1 for perfect sphere |

### Shape Descriptors

| Property | Formula | Description |
|----------|---------|-------------|
| **Aspect Ratio** | From principal axes of inertia tensor | Elongation measure |
| **Mean Curvature** | $H = \frac{1}{2}(\kappa_1 + \kappa_2)$ | Average local curvature |
| **Gaussian Curvature** | $K = \kappa_1 \kappa_2$ | Intrinsic curvature |

### Implementation

```python
class BubbleProperties:
    """Compute geometric properties from SH surface"""
    
    def __init__(self, surface, n_theta=50, n_phi=100):
        self.surface = surface
        self.n_theta = n_theta
        self.n_phi = n_phi
    
    def compute_all(self):
        """Compute all properties"""
        return {
            'volume': self.volume(),
            'surface_area': self.surface_area(),
            'centroid': self.surface.center.copy(),
            'equivalent_diameter': self.equivalent_diameter(),
            'sphericity': self.sphericity(),
            'aspect_ratio': self.aspect_ratio(),
            'mean_curvature': self.mean_curvature_stats(),
        }
    
    def volume(self):
        """Compute volume using divergence theorem"""
        return self.surface.volume(self.n_theta, self.n_phi)
    
    def surface_area(self):
        """Compute surface area"""
        return self.surface.surface_area(self.n_theta, self.n_phi)
    
    def equivalent_diameter(self):
        """Diameter of sphere with equal volume"""
        V = self.volume()
        return (6 * V / np.pi) ** (1/3)
    
    def sphericity(self):
        """Ratio of sphere surface area to actual surface area at equal volume"""
        V = self.volume()
        A = self.surface_area()
        return (np.pi ** (1/3)) * ((6 * V) ** (2/3)) / A
    
    def aspect_ratio(self):
        """Compute aspect ratio from inertia tensor"""
        # Sample surface points
        x, y, z = self.surface.surface_points(self.n_theta, self.n_phi)
        points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        
        # Center points
        centered = points - self.surface.center
        
        # Compute covariance matrix
        cov = np.cov(centered.T)
        
        # Eigenvalues give principal axis lengths squared
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
        
        # Aspect ratio: largest / smallest
        return np.sqrt(eigenvalues[0] / eigenvalues[-1])
    
    def mean_curvature_stats(self):
        """Compute mean curvature statistics over surface"""
        theta = np.linspace(0.1, np.pi - 0.1, self.n_theta)  # Avoid poles
        phi = np.linspace(0, 2*np.pi, self.n_phi)
        
        curvatures = []
        eps = 1e-5
        
        for t in theta:
            for p in phi:
                H = self._compute_mean_curvature_at(t, p, eps)
                curvatures.append(H)
        
        curvatures = np.array(curvatures)
        
        return {
            'mean': np.mean(curvatures),
            'std': np.std(curvatures),
            'min': np.min(curvatures),
            'max': np.max(curvatures)
        }
    
    def _compute_mean_curvature_at(self, theta, phi, eps):
        """Compute mean curvature at a point using finite differences"""
        # Get radius and derivatives
        r = self.surface.radius(theta, phi)
        
        r_t = (self.surface.radius(theta + eps, phi) - 
               self.surface.radius(theta - eps, phi)) / (2 * eps)
        r_p = (self.surface.radius(theta, phi + eps) - 
               self.surface.radius(theta, phi - eps)) / (2 * eps)
        
        r_tt = (self.surface.radius(theta + eps, phi) - 
                2 * r + self.surface.radius(theta - eps, phi)) / eps**2
        r_pp = (self.surface.radius(theta, phi + eps) - 
                2 * r + self.surface.radius(theta, phi - eps)) / eps**2
        r_tp = (self.surface.radius(theta + eps, phi + eps) - 
                self.surface.radius(theta + eps, phi - eps) -
                self.surface.radius(theta - eps, phi + eps) + 
                self.surface.radius(theta - eps, phi - eps)) / (4 * eps**2)
        
        # First fundamental form coefficients
        sin_t = np.sin(theta)
        E = r**2 + r_t**2
        F = r_t * r_p
        G = r**2 * sin_t**2 + r_p**2
        
        # Second fundamental form coefficients (simplified)
        denom = np.sqrt(E * G - F**2)
        
        # Mean curvature (approximate)
        H = (E * r_pp - 2 * F * r_tp + G * r_tt) / (2 * denom * r)
        
        return H


def export_to_mesh(surface, filename, n_theta=50, n_phi=100):
    """Export surface to STL/PLY mesh file"""
    from stl import mesh as stl_mesh
    
    x, y, z = surface.surface_points(n_theta, n_phi)
    
    # Create triangular mesh
    vertices = []
    faces = []
    
    for i in range(n_theta - 1):
        for j in range(n_phi - 1):
            # Get 4 corners of quad
            v00 = len(vertices)
            vertices.append([x[i, j], y[i, j], z[i, j]])
            v01 = len(vertices)
            vertices.append([x[i, j+1], y[i, j+1], z[i, j+1]])
            v10 = len(vertices)
            vertices.append([x[i+1, j], y[i+1, j], z[i+1, j]])
            v11 = len(vertices)
            vertices.append([x[i+1, j+1], y[i+1, j+1], z[i+1, j+1]])
            
            # Two triangles per quad
            faces.append([v00, v01, v11])
            faces.append([v00, v11, v10])
    
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Create and save mesh
    bubble_mesh = stl_mesh.Mesh(np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            bubble_mesh.vectors[i][j] = vertices[face[j]]
    
    bubble_mesh.save(filename)
    print(f"Saved mesh to {filename}")
```

---

## Implementation Checklist

### Prerequisites
- [ ] Multi-view camera calibration (intrinsics + extrinsics)
- [ ] Refractive interface parameters (plate positions, thicknesses, refractive indices)
- [ ] Existing refractive projection code
- [ ] Background images for segmentation

### Step 1: Segmentation
- [ ] Implement background subtraction
- [ ] Add morphological cleanup
- [ ] Validation check for mask quality

### Step 2: Visual Hull
- [ ] Implement hierarchical visual hull
- [ ] Integrate refractive projection
- [ ] Memory-efficient voxelization

### Step 3: Region Separation
- [ ] Implement watershed separation
- [ ] Fallback to simple connected components
- [ ] Size filtering

### Step 4: Surface Initialization
- [ ] SphericalHarmonicsSurface class
- [ ] Regularized least squares fitting
- [ ] Adaptive L_max selection
- [ ] Star-shaped validation

### Step 5: Optimization
- [ ] Differentiable silhouette renderer
- [ ] Refractive projection with analytical gradients
- [ ] Loss functions (silhouette, smoothness, overlap, volume)
- [ ] Joint optimization loop
- [ ] Learning rate scheduling

### Step 6: Gradients
- [ ] Implement analytical Jacobians
- [ ] Custom PyTorch autograd function
- [ ] Gradient validation tests

### Step 7: Output
- [ ] Volume computation
- [ ] Surface area computation
- [ ] Curvature computation
- [ ] Mesh export (STL/PLY)

### Validation
- [ ] Synthetic test with known ground truth
- [ ] Compare optimized vs. visual hull volumes
- [ ] Check reprojection error
- [ ] Sensitivity analysis (calibration noise, segmentation errors)

---

## Testing and Validation

### Synthetic Data Test

```python
def run_synthetic_test():
    """Full pipeline test with known ground truth"""
    
    # 1. Create ground truth bubbles
    gt_surfaces = [
        create_ellipsoid_sh(center=[0, 0, 50], semi_axes=[5, 5, 7], L_max=6),
        create_ellipsoid_sh(center=[20, 10, 60], semi_axes=[4, 4, 4], L_max=4),
        create_ellipsoid_sh(center=[-15, 5, 45], semi_axes=[3, 4, 5], L_max=8),
    ]
    
    # 2. Render ground truth silhouettes
    cameras = load_camera_calibration()
    interface_params = load_interface_params()
    
    gt_masks = []
    for cam in cameras:
        mask = render_surfaces_to_mask(gt_surfaces, cam, interface_params)
        gt_masks.append(mask)
    
    # 3. Run full pipeline
    # Step 1: Segmentation (masks already binary)
    masks = gt_masks
    
    # Step 2: Visual hull
    bounds = ((-50, -50, 0), (50, 50, 100))
    hull = hierarchical_visual_hull(masks, cameras, interface_params, bounds)
    
    # Step 3: Separation
    labels, regions = separate_bubbles(hull)
    print(f"Detected {len(regions)} bubbles (ground truth: {len(gt_surfaces)})")
    
    # Step 4: Initialize surfaces
    initial_surfaces = []
    for region in regions:
        voxel_coords = get_voxel_coordinates(region, bounds)
        L_max = choose_L_max(voxel_coords)
        surface = SphericalHarmonicsSurface(L_max=L_max)
        surface.fit_from_voxels(voxel_coords)
        initial_surfaces.append(surface)
    
    # Step 5: Optimize
    config = {
        'lambda_smooth': 0.01,
        'lambda_overlap': 0.1,
        'lambda_volume': 0.01,
        'sigma': 2.0
    }
    optimizer = BubbleSurfaceOptimizer(
        initial_surfaces, cameras, masks, interface_params, config
    )
    history, final_surfaces = optimizer.optimize(n_iterations=500)
    
    # 4. Compare with ground truth
    print("\n=== Validation Results ===")
    for i, (gt, recon) in enumerate(zip(gt_surfaces, final_surfaces)):
        gt_vol = gt.volume()
        recon_vol = recon.volume()
        vol_error = abs(gt_vol - recon_vol) / gt_vol * 100
        
        centroid_error = np.linalg.norm(gt.center - recon.center)
        
        print(f"Bubble {i+1}:")
        print(f"  Volume error: {vol_error:.2f}%")
        print(f"  Centroid error: {centroid_error:.3f} mm")
    
    return final_surfaces, history


def create_ellipsoid_sh(center, semi_axes, L_max=8):
    """Create SH surface representing an ellipsoid"""
    surface = SphericalHarmonicsSurface(L_max=L_max)
    surface.center = np.array(center)
    
    # Sample ellipsoid surface
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 100)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    
    a, b, c = semi_axes
    x = a * np.sin(theta_grid) * np.cos(phi_grid)
    y = b * np.sin(theta_grid) * np.sin(phi_grid)
    z = c * np.cos(theta_grid)
    
    voxels = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1) + center
    surface.fit_from_voxels(voxels)
    
    return surface
```

### Sensitivity Analysis

```python
def sensitivity_analysis():
    """Test robustness to various error sources"""
    
    results = {}
    
    # 1. Camera calibration noise
    for noise_level in [0.0, 0.1, 0.5, 1.0]:  # pixels
        cameras_noisy = add_calibration_noise(cameras, noise_level)
        surfaces = run_pipeline(masks, cameras_noisy, interface_params)
        results[f'calib_noise_{noise_level}'] = evaluate_accuracy(surfaces, gt_surfaces)
    
    # 2. Segmentation errors
    for erosion in [0, 1, 2, 3]:  # pixels
        masks_eroded = [cv2.erode(m, np.ones((erosion*2+1,)*2)) for m in masks]
        surfaces = run_pipeline(masks_eroded, cameras, interface_params)
        results[f'erosion_{erosion}'] = evaluate_accuracy(surfaces, gt_surfaces)
    
    # 3. Refractive index uncertainty
    for n_error in [0.0, 0.01, 0.02, 0.05]:
        interface_noisy = add_refractive_index_noise(interface_params, n_error)
        surfaces = run_pipeline(masks, cameras, interface_noisy)
        results[f'n_error_{n_error}'] = evaluate_accuracy(surfaces, gt_surfaces)
    
    return results
```

---

## Advantages of This Approach

| Feature | Benefit |
|---------|---------|
| No 2D instance segmentation | Robust to overlapping bubbles in images |
| 3D separation via watershed | Handles touching bubbles |
| Combined mask loss | No multi-view association problem |
| Spherical harmonics | Smooth, compact, differentiable representation |
| Analytical gradients | Fast, accurate optimization |
| Hierarchical visual hull | Memory efficient |
| Multiple loss terms | Balanced optimization |
| Builds on existing code | Minimal changes to current pipeline |

---

## Limitations and Future Work

### Current Limitations

1. **Star-shaped assumption**: SH representation requires bubbles to be star-shaped from centroid
2. **Topology changes**: Cannot handle bubble splitting/merging during optimization
3. **Touching bubbles**: May merge in regionprops if strongly overlapping in 3D
4. **Computational cost**: Optimization can be slow for many bubbles

### Potential Extensions

1. **Neural SDF**: Replace SH with neural signed distance function for non-star-shaped bubbles
2. **Temporal tracking**: Link bubbles across frames for trajectory analysis
3. **Physics constraints**: Add surface tension, pressure equilibrium regularization
4. **Adaptive $L_{max}$**: Dynamically adjust SH order during optimization
5. **Coarse-to-fine optimization**: Start with low $L_{max}$, progressively increase
6. **GPU acceleration**: Parallelize silhouette rendering and projection
7. **Uncertainty quantification**: Estimate reconstruction confidence

---

## Configuration Template

```yaml
# config.yaml - Pipeline configuration

segmentation:
  method: "background_subtraction"  # or "adaptive_threshold"
  threshold: 25
  morph_kernel_size: 5

visual_hull:
  coarse_resolution: 4.0  # mm
  fine_resolution: 0.5    # mm
  bounds:
    min: [-50, -50, 0]
    max: [50, 50, 100]

separation:
  method: "watershed"  # or "connected_components"
  min_distance: 5      # voxels
  min_size: 100        # voxels

initialization:
  L_max_default: 8
  L_max_max: 20
  residual_threshold: 0.05
  regularization: 1e-4

optimization:
  n_iterations: 500
  learning_rate: 0.001
  lambda_smooth: 0.01
  lambda_overlap: 0.1
  lambda_volume: 0.01
  sigma: 2.0
  overlap_margin: 0.5
  silhouette_loss: "bce"  # or "focal", "iou"
  gradient_clip: 1.0

output:
  export_mesh: true
  mesh_format: "stl"
  compute_curvature: true
```

---

## References

1. **Spherical Harmonics**: Green, R. "Spherical Harmonic Lighting: The Gritty Details" (2003)
2. **Visual Hull**: Laurentini, A. "The Visual Hull Concept for Silhouette-Based Image Understanding" (1994)
3. **Differentiable Rendering**: Liu, S. et al. "Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning" (2019)
4. **Multi-view Reconstruction**: Seitz, S. et al. "A Comparison and Evaluation of Multi-View Stereo Reconstruction Algorithms" (2006)
5. **Watershed Segmentation**: Meyer, F. "Topographic distance and watershed lines" (1994)
6. **Refractive Projection**: Agrawal, A. et al. "A Theory of Multi-Layer Flat Refractive Geometry" (2012)
