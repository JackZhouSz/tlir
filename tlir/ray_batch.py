"""
Ray batch utilities for ray-based training.

This module provides classes and functions for storing and sampling rays
from the training set, enabling more efficient training by sampling random
rays instead of random images.
"""

import numpy as np
import mitsuba as mi
import drjit as dr
from typing import List, Tuple, Optional, Dict, Any


class RayBatch:
    """
    Container for ray data that supports efficient indexing and slicing.
    
    Stores all rays from the training set along with their properties:
    - Ray origins and directions
    - Ground truth colors
    - Object masks (for masked noise augmentation)
    - Metadata (sensor index, pixel coordinates)
    
    Supports NumPy-style indexing for random sampling during training.
    """
    
    def __init__(self,
                 origins: np.ndarray,
                 directions: np.ndarray,
                 colors: np.ndarray,
                 masks: Optional[np.ndarray] = None,
                 sensor_indices: Optional[np.ndarray] = None,
                 pixel_coords: Optional[np.ndarray] = None):
        """
        Initialize ray batch.
        
        Args:
            origins: Ray origins, shape (N, 3)
            directions: Ray directions (normalized), shape (N, 3)
            colors: Ground truth colors, shape (N, 3)
            masks: Object masks (optional), shape (N, 1) or (N,)
            sensor_indices: Index of sensor/camera for each ray (optional), shape (N,)
            pixel_coords: Pixel coordinates (x, y) for each ray (optional), shape (N, 2)
        """
        self.origins = origins
        self.directions = directions
        self.colors = colors
        self.masks = masks
        self.sensor_indices = sensor_indices
        self.pixel_coords = pixel_coords
        
        # Validate shapes
        assert origins.shape == directions.shape, "Origins and directions must have same shape"
        assert origins.shape[0] == colors.shape[0], "Number of rays must match colors"
        assert origins.shape[1] == 3, "Origins must be (N, 3)"
        assert directions.shape[1] == 3, "Directions must be (N, 3)"
        
        if masks is not None:
            if masks.ndim == 1:
                masks = masks[:, np.newaxis]
            assert masks.shape[0] == origins.shape[0], "Number of masks must match rays"
            self.masks = masks
        
        self.num_rays = origins.shape[0]
    
    def __len__(self) -> int:
        """Return number of rays in the batch."""
        return self.num_rays
    
    def __getitem__(self, idx) -> 'RayBatch':
        """
        Index or slice the ray batch.
        
        Supports:
        - Integer indexing: batch[0] returns a RayBatch with 1 ray
        - Slice indexing: batch[10:20] returns a RayBatch with 10 rays
        - Array indexing: batch[np.array([0, 5, 10])] returns selected rays
        - Boolean indexing: batch[mask] returns rays where mask is True
        
        Args:
            idx: Index, slice, or array of indices
            
        Returns:
            New RayBatch containing the selected rays
        """
        # Handle different indexing types
        if isinstance(idx, int):
            # Single index - convert to array for consistent handling
            idx = np.array([idx])
        elif isinstance(idx, slice):
            # Slice - convert to array
            idx = np.arange(*idx.indices(self.num_rays))
        elif isinstance(idx, (list, tuple)):
            idx = np.array(idx)
        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            # Boolean indexing
            idx = np.where(idx)[0]
        
        # Index all arrays
        origins = self.origins[idx]
        directions = self.directions[idx]
        colors = self.colors[idx]
        
        masks = self.masks[idx] if self.masks is not None else None
        sensor_indices = self.sensor_indices[idx] if self.sensor_indices is not None else None
        pixel_coords = self.pixel_coords[idx] if self.pixel_coords is not None else None
        
        return RayBatch(origins, directions, colors, masks, sensor_indices, pixel_coords)
    
    def sample(self, num_rays: int, rng: Optional[np.random.Generator] = None) -> 'RayBatch':
        """
        Randomly sample rays from the batch.
        
        Args:
            num_rays: Number of rays to sample
            rng: NumPy random generator (optional)
            
        Returns:
            New RayBatch with sampled rays
        """
        if rng is None:
            rng = np.random.default_rng()
        
        indices = rng.choice(self.num_rays, size=num_rays, replace=False)
        return self[indices]
    
    def to_mitsuba(self) -> Tuple[mi.Ray3f, mi.Color3f]:
        """
        Convert ray batch to Mitsuba types for rendering.

        Returns:
            Tuple of (rays, target_colors) as Mitsuba types
            - rays: Mitsuba Ray3f object
            - target_colors: Mitsuba Color3f to match integrator output type
        """
        # Convert to Mitsuba tensors
        origins_mi = mi.TensorXf(self.origins)
        directions_mi = mi.TensorXf(self.directions)

        # Create Mitsuba rays
        rays = mi.Ray3f(
            o=mi.Point3f(origins_mi[:, 0], origins_mi[:, 1], origins_mi[:, 2]),
            d=mi.Vector3f(directions_mi[:, 0], directions_mi[:, 1], directions_mi[:, 2])
        )

        # Convert colors to Color3f to match integrator output type
        # self.colors is (num_rays, 3)
        colors_mi = mi.Color3f(
            mi.Float(self.colors[:, 0]),  # R channel
            mi.Float(self.colors[:, 1]),  # G channel
            mi.Float(self.colors[:, 2])   # B channel
        )

        return rays, colors_mi
    
    def shuffle(self, rng: Optional[np.random.Generator] = None) -> 'RayBatch':
        """
        Return a shuffled copy of the ray batch.
        
        Args:
            rng: NumPy random generator (optional)
            
        Returns:
            New shuffled RayBatch
        """
        if rng is None:
            rng = np.random.default_rng()
        
        indices = rng.permutation(self.num_rays)
        return self[indices]
    
    def split(self, batch_size: int) -> List['RayBatch']:
        """
        Split ray batch into smaller batches.
        
        Args:
            batch_size: Size of each sub-batch
            
        Returns:
            List of RayBatch objects
        """
        num_batches = (self.num_rays + batch_size - 1) // batch_size
        batches = []
        
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, self.num_rays)
            batches.append(self[start:end])
        
        return batches


def extract_rays_from_sensors(
    sensors: List[mi.Sensor],
    ref_images: List[mi.TensorXf],
    ref_masks: Optional[List[mi.TensorXf]] = None
) -> RayBatch:
    """
    Extract all rays from a list of sensors and their corresponding images.
    
    This function creates a ray for each pixel in each image, storing the
    ray's origin, direction, and ground truth color.
    
    Args:
        sensors: List of Mitsuba sensors (cameras)
        ref_images: List of reference images (ground truth colors)
        ref_masks: Optional list of object masks
        
    Returns:
        RayBatch containing all rays from all images
    """
    all_origins = []
    all_directions = []
    all_colors = []
    all_masks = []
    all_sensor_indices = []
    all_pixel_coords = []
    
    for sensor_idx, (sensor, ref_image) in enumerate(zip(sensors, ref_images)):
        # Get film dimensions
        film = sensor.film()
        res = film.size()
        width, height = res[0], res[1]
        num_pixels = width * height

        # Generate pixel positions using DrJit (same as render_camera)
        idx = dr.arange(mi.UInt32, num_pixels)
        x = idx % width
        y = idx // width

        # Convert to normalized coordinates [0, 1]
        pos_x = (x + 0.5) / width
        pos_y = (y + 0.5) / height
        pos_sample = mi.Point2f(pos_x, pos_y)

        # Sample rays from sensor (vectorized)
        wavelength_sample = 0.5
        time_sample = 0.0
        aperture_sample = mi.Point2f(0.5, 0.5)
        rays, _ = sensor.sample_ray(time_sample, wavelength_sample, pos_sample, aperture_sample)

        # Extract origins and directions to NumPy arrays
        ray_origins = np.stack([np.array(rays.o.x), np.array(rays.o.y), np.array(rays.o.z)], axis=1).astype(np.float32)
        ray_directions = np.stack([np.array(rays.d.x), np.array(rays.d.y), np.array(rays.d.z)], axis=1).astype(np.float32)

        # Store pixel coordinates for metadata
        pixel_x = np.array(x, dtype=np.float32)
        pixel_y = np.array(y, dtype=np.float32)
        
        # Extract colors from reference image
        # Convert Mitsuba tensor to numpy
        ref_image_np = np.array(ref_image)  # Shape: (height, width, channels)

        # Extract only RGB channels (drop alpha if present)
        num_channels = ref_image_np.shape[-1]
        if num_channels == 4:
            # RGBA - extract only RGB
            ref_image_np = ref_image_np[..., :3]
        elif num_channels != 3:
            raise ValueError(f"Expected 3 or 4 channels in reference image, got {num_channels}")

        # Flatten to get colors for each pixel
        colors = ref_image_np.reshape(-1, 3)  # Shape: (N, 3)
        
        # Extract masks if provided
        if ref_masks is not None and ref_masks[sensor_idx] is not None:
            mask_np = np.array(ref_masks[sensor_idx])
            mask = mask_np.reshape(-1, mask_np.shape[-1] if mask_np.ndim > 2 else 1)
            # Take first channel if multi-channel
            if mask.shape[-1] > 1:
                mask = mask[:, 0:1]
            all_masks.append(mask)
        
        # Store rays
        all_origins.append(ray_origins)
        all_directions.append(ray_directions)
        all_colors.append(colors)
        all_sensor_indices.append(np.full(num_pixels, sensor_idx, dtype=np.int32))
        all_pixel_coords.append(np.stack([pixel_x, pixel_y], axis=1))
    
    # Concatenate all rays
    origins = np.concatenate(all_origins, axis=0)
    directions = np.concatenate(all_directions, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    sensor_indices = np.concatenate(all_sensor_indices, axis=0)
    pixel_coords = np.concatenate(all_pixel_coords, axis=0)
    
    masks = np.concatenate(all_masks, axis=0) if len(all_masks) > 0 else None
    
    print(f"Extracted {len(origins)} rays from {len(sensors)} images")
    print(f"  Ray origins shape: {origins.shape}")
    print(f"  Ray directions shape: {directions.shape}")
    print(f"  Colors shape: {colors.shape}")
    if masks is not None:
        print(f"  Masks shape: {masks.shape}")
    
    return RayBatch(origins, directions, colors, masks, sensor_indices, pixel_coords)


def render_ray_batch(
    scene: mi.Scene,
    ray_batch: RayBatch,
    params: Dict[str, Any],
    spp: int = 1,
    **kwargs
) -> mi.TensorXf:
    """
    Render a batch of rays through the scene.

    Args:
        scene: Mitsuba scene
        ray_batch: Batch of rays to render
        params: Scene parameters
        spp: Samples per pixel
        **kwargs: Additional arguments passed to integrator

    Returns:
        Rendered colors for each ray
    """
    # Get integrator
    integrator = scene.integrator()

    # Convert rays to Mitsuba format
    rays, _ = ray_batch.to_mitsuba()

    # Create sampler
    sampler = mi.load_dict({'type': 'independent', 'sample_count': spp})
    sampler.seed(0, ray_batch.num_rays)

    # Render rays using TLIRIntegrator convenience method
    L, valid, aovs, state_out = integrator.render_forward(
        rays=rays,
        sampler=sampler,
        scene=scene,
        **kwargs
    )

    return L
