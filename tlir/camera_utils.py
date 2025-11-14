"""
Camera utilities for creating random views and trajectories.

This module provides functions for:
- Creating random camera views distributed around a sphere
- Generating test trajectories that loop around the scene
- Rendering and saving images and poses
"""

from __future__ import annotations

import os
import numpy as np
import mitsuba as mi
import drjit as dr
from typing import List, Tuple, Optional, Dict, Any


def spherical_to_cartesian(theta: float, phi: float, radius: float) -> np.ndarray:
    """
    Convert spherical coordinates to Cartesian coordinates.

    This function uses the standard mathematical convention where Z is the vertical/up axis.
    - theta rotates in the X-Y plane (azimuthal angle around Z axis)
    - phi is the polar angle from the Z axis (0 = north pole, π/2 = equator, π = south pole)

    Args:
        theta: Azimuthal angle in radians [0, 2π] - rotation about Z axis
        phi: Polar angle in radians [0, π] - angle from Z axis
        radius: Radius from origin

    Returns:
        3D Cartesian coordinates as numpy array [x, y, z] where Z is up
    """
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.array([x, y, z])


def create_random_hemisphere_cameras(
    num_cameras: int,
    center: List[float] = [0.0, 0.0, 0.0],
    radius: float = 1.3,
    render_res: int = 256,
    fov: float = 45.0,
    look_inward: bool = True,
    hemisphere_normal: Optional[List[float]] = [0, 0, 1],
    up_vector: List[float] = [0, 1, 0],
    seed: Optional[int] = None
) -> Tuple[List[mi.Sensor], List[np.ndarray]]:
    """
    Create random cameras uniformly distributed on hemisphere surface.

    Uses 3D normal distribution sampling followed by normalization to get
    uniform distribution on sphere surface. Points are then reflected to ensure
    they lie on the correct hemisphere (positive dot product with hemisphere_normal).
    Rejection sampling filters by elevation.

    Args:
        num_cameras: Number of cameras to create
        center: Center point cameras look at/away from
        radius: Radius of hemisphere
        render_res: Rendering resolution
        fov: Field of view in degrees
        look_inward: If True, cameras look inward toward center; if False, look outward
        hemisphere_normal: Normal vector defining the hemisphere plane. Points are reflected
                          to have positive dot product with this normal. Set to None to disable
                          reflection (for full sphere sampling). Default [0, 0, 1] = upper hemisphere
        up_vector: Up direction for camera orientation. Default [0, 1, 0] = Y-up
        seed: Random seed for reproducibility

    Returns:
        Tuple of (sensors list, poses list) where poses are 4x4 numpy arrays
    """
    if seed is not None:
        np.random.seed(seed)

    sensors = []
    poses = []
    center_np = np.array(center)

    # Normalize hemisphere normal if provided
    hemisphere_normal_np = None
    if hemisphere_normal is not None:
        hemisphere_normal_np = np.array(hemisphere_normal, dtype=float)
        hemisphere_normal_np = hemisphere_normal_np / np.linalg.norm(hemisphere_normal_np)

    generated = 0
    attempts = 0
    max_attempts = num_cameras * 1000  # Safety limit

    while generated < num_cameras and attempts < max_attempts:
        attempts += 1

        # Sample from 3D normal distribution
        # N(0, 1) in x, y, z
        point = np.random.randn(3)

        # Normalize to get uniform point on unit sphere
        point = point / np.linalg.norm(point)

        # For hemisphere: reflect points to ensure they're on the correct side
        # This ensures all points have positive dot product with hemisphere_normal
        if hemisphere_normal_np is not None:
            dot_product = np.dot(point, hemisphere_normal_np)
            if dot_product < 0:
                # Reflect across the plane defined by hemisphere_normal
                point = point - 2 * dot_product * hemisphere_normal_np

        # Scale by radius and translate to center
        cam_pos = point * radius + center_np

        # Compute elevation angle for rejection sampling
        # elevation = arcsin(z / radius) where z is relative to center
        z_relative = point[2]  # Already normalized, so this is z/radius
        elevation_rad = np.arcsin(np.clip(z_relative, -1.0, 1.0))
        elevation_deg = np.rad2deg(elevation_rad)

        generated += 1

        # Create transform
        if look_inward:
            # Look toward center
            origin = cam_pos
            target = center_np
        else:
            # Look outward (away from center)
            origin = cam_pos
            direction = cam_pos - center_np
            direction = direction / np.linalg.norm(direction)
            target = cam_pos + direction

        # Compute up vector (pointing generally upward)
        forward = target - origin
        forward = forward / np.linalg.norm(forward)

        # Use specified up_vector as reference, but make it perpendicular to forward
        world_up = np.array(up_vector)
        right = np.cross(world_up, forward)
        if np.linalg.norm(right) < 1e-6:  # Handle case when forward is parallel to up_vector
            # Find an alternative up vector perpendicular to forward
            if abs(world_up[0]) < 0.9:
                alt_up = np.array([1, 0, 0])
            else:
                alt_up = np.array([0, 1, 0])
            right = np.cross(alt_up, forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)

        # Build 4x4 pose matrix (camera to world)
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward  # Negative because camera looks down -Z
        pose[:3, 3] = origin
        poses.append(pose)

        # Create Mitsuba sensor
        # Mitsuba uses look_at which is simpler
        sensor = mi.load_dict({
            'type': 'perspective',
            'fov': fov,
            'to_world': mi.ScalarTransform4f.look_at(
                origin=origin.tolist(),
                target=target.tolist(),
                up=up.tolist()
            ),
            'film': {
                'type': 'hdrfilm',
                'width': render_res,
                'height': render_res,
                'filter': {'type': 'box'},
                'pixel_format': 'rgba'
            }
        })
        sensors.append(sensor)

    return sensors, poses


def create_random_sphere_cameras(
    num_cameras: int,
    center: List[float] = [0.0, 0.0, 0.0],
    radius: float = 1.3,
    render_res: int = 256,
    fov: float = 45.0,
    look_inward: bool = True,
    up_vector: List[float] = [0, 1, 0],
    seed: Optional[int] = None
) -> Tuple[List[mi.Sensor], List[np.ndarray]]:
    """
    Create random cameras uniformly distributed on sphere surface.

    Uses 3D normal distribution sampling without hemisphere reflection,
    allowing cameras to be placed on the full sphere surface.
    Supports optional elevation filtering via min/max_elevation.

    Args:
        num_cameras: Number of cameras to create
        center: Center point cameras look at/away from
        radius: Radius of sphere
        render_res: Rendering resolution
        fov: Field of view in degrees
        look_inward: If True, cameras look inward toward center; if False, look outward
        up_vector: Up direction for camera orientation. Default [0, 1, 0] = Y-up
        seed: Random seed for reproducibility

    Returns:
        Tuple of (sensors list, poses list)
    """
    return create_random_hemisphere_cameras(
        num_cameras=num_cameras,
        center=center,
        radius=radius,
        render_res=render_res,
        fov=fov,
        look_inward=look_inward,
        hemisphere_normal=None,  # No reflection for full sphere
        up_vector=up_vector,
        seed=seed
    )


def create_trajectory_cameras(
    num_cameras: int,
    center: List[float] = [0.0, 0.0, 0.0],
    radius: float = 1.3,
    render_res: int = 256,
    fov: float = 45.0,
    min_elevation: float = 10.0,
    max_elevation: float = 80.0,
    look_inward: bool = True,
    start_angle: float = 0.0,
    num_loops: float = 1.0,
    up_vector: List[float] = [0, 1, 0]
) -> Tuple[List[mi.Sensor], List[np.ndarray]]:
    """
    Create cameras along a spiral trajectory with changing elevation.

    The trajectory spirals around the scene, smoothly transitioning between
    min and max elevation angles. The trajectory rotates about the specified up_vector.
    Internally uses spherical_to_cartesian (Z-up convention) and transforms to the
    desired up_vector orientation.
    Useful for creating smooth test/validation trajectories that cover different viewpoints.

    Args:
        num_cameras: Number of cameras along trajectory
        center: Center point cameras look at/away from
        radius: Radius of trajectory
        render_res: Rendering resolution
        fov: Field of view in degrees
        min_elevation: Starting elevation angle in degrees [-90=nadir, 0=horizon, 90=zenith]
        max_elevation: Ending elevation angle in degrees
        look_inward: If True, cameras look inward toward center; if False, look outward
        start_angle: Starting azimuthal angle in degrees
        num_loops: Number of complete 360° loops around the scene (rotation about up_vector)
        up_vector: Up direction for camera orientation. Default [0, 1, 0] = Y-up.
                  The trajectory will rotate about this axis.

    Returns:
        Tuple of (sensors list, poses list)
    """
    sensors = []
    poses = []
    center_np = np.array(center)
    up_vector_np = np.array(up_vector, dtype=float)
    up_vector_np = up_vector_np / np.linalg.norm(up_vector_np)  # Normalize

    # Compute rotation matrix from Z-up to desired up_vector
    # spherical_to_cartesian uses Z as up, so we need to transform to up_vector
    z_axis = np.array([0.0, 0.0, 1.0])

    # Compute rotation that maps z_axis to up_vector_np
    if np.allclose(z_axis, up_vector_np):
        # No rotation needed
        rotation_matrix = np.eye(3)
    elif np.allclose(z_axis, -up_vector_np):
        # 180 degree rotation - use any perpendicular axis
        rotation_matrix = np.diag([1.0, -1.0, -1.0])
    else:
        # General case: Rodrigues' rotation formula
        # Rotate z_axis to up_vector_np
        v = np.cross(z_axis, up_vector_np)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, up_vector_np)

        # Skew-symmetric matrix for v
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])

        rotation_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s * s))

    # Create parameter t in [0, 1]
    t_values = np.linspace(0, 1, num_cameras)

    for t in t_values:
        # Azimuthal angle: spirals around num_loops times
        theta = np.deg2rad(start_angle) + t * num_loops * 2 * np.pi

        # Elevation: smoothly transitions from min to max
        # Use smooth interpolation (could use cosine for even smoother motion)
        elevation = min_elevation + t * (max_elevation - min_elevation)

        # Convert elevation to phi (polar angle in Z-up convention)
        phi = np.deg2rad(90.0 - elevation)

        # Convert to Cartesian in Z-up coordinate system
        pos_z_up = spherical_to_cartesian(theta, phi, radius)

        # Transform to desired up_vector coordinate system
        pos_transformed = rotation_matrix @ pos_z_up

        # Translate to center
        cam_pos = pos_transformed + center_np

        # Create transform
        if look_inward:
            origin = cam_pos
            target = center_np
        else:
            origin = cam_pos
            direction = cam_pos - center_np
            direction = direction / np.linalg.norm(direction)
            target = cam_pos + direction

        # Compute up vector
        forward = target - origin
        forward = forward / np.linalg.norm(forward)

        world_up = np.array(up_vector)
        right = np.cross(world_up, forward)
        if np.linalg.norm(right) < 1e-6:
            # Find an alternative up vector perpendicular to forward
            if abs(world_up[0]) < 0.9:
                alt_up = np.array([1, 0, 0])
            else:
                alt_up = np.array([0, 1, 0])
            right = np.cross(alt_up, forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)

        # Build pose matrix
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = origin
        poses.append(pose)

        # Create sensor
        sensor = mi.load_dict({
            'type': 'perspective',
            'fov': fov,
            'to_world': mi.ScalarTransform4f.look_at(
                origin=origin.tolist(),
                target=target.tolist(),
                up=up.tolist()
            ),
            'film': {
                'type': 'hdrfilm',
                'width': render_res,
                'height': render_res,
                'filter': {'type': 'box'},
                'pixel_format': 'rgba'
            }
        })
        sensors.append(sensor)

    return sensors, poses


def render_and_save_images(
    scene: mi.Scene,
    sensors: List[mi.Sensor],
    output_dir: str,
    prefix: str = "image",
    spp: int = 128,
    save_npy: bool = True,
    save_exr: bool = False,
    save_png: bool = True
) -> List[np.ndarray]:
    """
    Render images from sensors and save to disk.

    Args:
        scene: Mitsuba scene to render
        sensors: List of camera sensors
        output_dir: Directory to save images
        prefix: Prefix for image filenames
        spp: Samples per pixel
        save_npy: If True, save as .npy files
        save_exr: If True, save as .exr files
        save_png: If True, save as .png files

    Returns:
        List of rendered images as numpy arrays
    """
    os.makedirs(output_dir, exist_ok=True)

    images = []
    for i, sensor in enumerate(sensors):
        print(f"Rendering image {i+1}/{len(sensors)}...", end='\r')

        # Render
        img = mi.render(scene, sensor=sensor, spp=spp)

        # Convert to numpy
        img_np = np.array(img)
        images.append(img_np)

        # Save as npy
        if save_npy:
            npy_path = os.path.join(output_dir, f"{prefix}_{i:04d}.npy")
            np.save(npy_path, img_np)

        # Save as EXR
        if save_exr:
            exr_path = os.path.join(output_dir, f"{prefix}_{i:04d}.exr")
            mi.Bitmap(img).write(exr_path)

        # Save as PNG
        if save_png:
            png_path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
            # Convert to 8-bit [0, 255] and clip to valid range
            img_8bit = np.clip(img_np[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
            mi.Bitmap(img_8bit).write(png_path)

    print(f"\nRendered and saved {len(images)} images to {output_dir}")
    return images


def render_and_save_masks(
    scene: mi.Scene,
    sensors: List[mi.Sensor],
    output_dir: str,
    prefix: str = "mask",
    spp: int = 1,
    save_npy: bool = True,
    save_png: bool = True
) -> List[np.ndarray]:
    """
    Render object masks from sensors and save to disk (vectorized).

    Masks are computed based on ray-object intersections:
    - For each pixel, sample rays and compute the probability that a ray hits an object
    - 1.0 where rays hit objects (si.is_valid() = True)
    - 0.0 where rays miss objects/hit background (si.is_valid() = False)
    - For spp > 1, averages over multiple samples per pixel

    Args:
        scene: Mitsuba scene to render
        sensors: List of camera sensors
        output_dir: Directory to save masks
        prefix: Prefix for mask filenames
        spp: Samples per pixel (number of rays to average per pixel)
        save_npy: If True, save as .npy files
        save_png: If True, save as .png files

    Returns:
        List of rendered masks as numpy arrays (values in [0, 1])
    """
    import drjit as dr

    os.makedirs(output_dir, exist_ok=True)

    masks = []
    for i, sensor in enumerate(sensors):
        print(f"Rendering mask {i+1}/{len(sensors)}...", end='\r')

        # Get image resolution from sensor
        film = sensor.film()
        res = film.size()
        width, height = res[0], res[1]

        if spp == 1:
            # Single sample per pixel (fast, vectorized)
            num_pixels = width * height

            # VECTORIZED: Create all pixel positions at once
            idx = dr.arange(mi.UInt32, num_pixels)
            x = idx % width
            y = idx // width

            # Convert to normalized coordinates [0, 1]
            pos_x = (mi.Float(x) + 0.5) / float(width)
            pos_y = (mi.Float(y) + 0.5) / float(height)
            pos_sample = mi.Point2f(pos_x, pos_y)

            # VECTORIZED: Sample all rays at once
            wavelength_sample = 0.5
            time_sample = 0.0
            aperture_sample = mi.Point2f(0.5, 0.5)

            rays, _ = sensor.sample_ray(time_sample, wavelength_sample, pos_sample, aperture_sample)

            # VECTORIZED: Trace all rays at once
            si = scene.ray_intersect(rays)

            # Extract hit/miss information: 1.0 if hit, 0.0 if miss
            hit_values = dr.select(si.is_valid(), 1.0, 0.0)

            # Convert to numpy and reshape
            mask = np.array(hit_values).reshape(height, width, 1).astype(np.float32)
        else:
            # Multiple samples per pixel (stochastic)
            mask_accumulator = np.zeros((height, width), dtype=np.float32)

            for sample_idx in range(spp):
                num_pixels = width * height
                idx = dr.arange(mi.UInt32, num_pixels)
                x = idx % width
                y = idx // width

                # Add random offset for stratified sampling
                rng = np.random.RandomState(sample_idx)
                offset_x = rng.rand()
                offset_y = rng.rand()

                pos_x = (mi.Float(x) + offset_x) / float(width)
                pos_y = (mi.Float(y) + offset_y) / float(height)
                pos_sample = mi.Point2f(pos_x, pos_y)

                wavelength_sample = 0.5
                time_sample = 0.0
                aperture_sample = mi.Point2f(0.5, 0.5)

                rays, _ = sensor.sample_ray(time_sample, wavelength_sample, pos_sample, aperture_sample)
                si = scene.ray_intersect(rays)

                hit_values = dr.select(si.is_valid(), 1.0, 0.0)
                mask_accumulator += np.array(hit_values).reshape(height, width)

            # Average over samples
            mask = (mask_accumulator / float(spp)).astype(np.float32)
            mask = mask.reshape(height, width, 1)

        masks.append(mask)

        # Save as npy
        if save_npy:
            npy_path = os.path.join(output_dir, f"{prefix}_{i:04d}.npy")
            np.save(npy_path, mask)

        # Save as PNG (grayscale)
        if save_png:
            png_path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
            # Convert to 8-bit grayscale [0, 255]
            mask_8bit = np.clip(mask[:, :, 0] * 255.0, 0, 255).astype(np.uint8)
            mi.Bitmap(mask_8bit).write(png_path)

    print(f"\nRendered and saved {len(masks)} masks to {output_dir}")
    return masks


def apply_turbo_colormap(depth: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """
    Apply turbo colormap to depth values with consistent normalization.

    Args:
        depth: Depth array (H, W) with values in arbitrary range
        vmin: Global minimum depth for normalization
        vmax: Global maximum depth for normalization

    Returns:
        RGB image (H, W, 3) with turbo colormap applied, values in [0, 1]
    """
    # Normalize depth to [0, 1] using global min/max
    depth_normalized = (depth - vmin) / (vmax - vmin + 1e-8)
    depth_normalized = np.clip(depth_normalized, 0.0, 1.0)

    # Turbo colormap (simplified implementation)
    # This is a piecewise linear approximation of the turbo colormap
    # For production use, consider using matplotlib.cm.turbo
    try:
        import matplotlib.pyplot as plt
        colormap = plt.cm.turbo
        rgb = colormap(depth_normalized)[:, :, :3]  # Drop alpha channel
    except ImportError:
        # Fallback: simple jet-like colormap if matplotlib not available
        r = np.clip(4 * depth_normalized - 1.5, 0, 1)
        g = np.clip(1 - np.abs(4 * depth_normalized - 2), 0, 1)
        b = np.clip(2.5 - 4 * depth_normalized, 0, 1)
        rgb = np.stack([r, g, b], axis=-1)

    return rgb


def render_and_save_depth(
    scene: mi.Scene,
    sensors: List[mi.Sensor],
    output_dir: str,
    prefix: str = "depth",
    spp: int = 128,
    save_npy: bool = True,
    save_png: bool = True
) -> Tuple[List[np.ndarray], float, float]:
    """
    Render depth maps from sensors and save to disk (vectorized).

    Uses global min/max depth across all views for consistent normalization
    of depth PNGs with turbo colormap.

    Args:
        scene: Mitsuba scene to render
        sensors: List of camera sensors
        output_dir: Directory to save depth maps
        prefix: Prefix for depth filenames
        spp: Samples per pixel
        save_npy: If True, save as .npy files
        save_png: If True, save as .png files with turbo colormap

    Returns:
        Tuple of (depth_maps, global_min_depth, global_max_depth)
        where depth_maps is a list of numpy arrays
    """
    import drjit as dr

    os.makedirs(output_dir, exist_ok=True)

    print(f"Rendering depth maps (vectorized)...")
    depths = []

    # First pass: render all depths and compute global min/max
    for i, sensor in enumerate(sensors):
        print(f"Rendering depth {i+1}/{len(sensors)}...", end='\r')

        # Get image resolution from sensor
        film = sensor.film()
        res = film.size()
        width, height = res[0], res[1]
        num_pixels = width * height

        # VECTORIZED: Create all pixel positions at once
        idx = dr.arange(mi.UInt32, num_pixels)
        x = idx % width
        y = idx // width

        # Convert to normalized coordinates [0, 1]
        pos_x = (mi.Float(x) + 0.5) / float(width)
        pos_y = (mi.Float(y) + 0.5) / float(height)
        pos_sample = mi.Point2f(pos_x, pos_y)

        # VECTORIZED: Sample all rays at once
        wavelength_sample = 0.5
        time_sample = 0.0
        aperture_sample = mi.Point2f(0.5, 0.5)

        rays, _ = sensor.sample_ray(time_sample, wavelength_sample, pos_sample, aperture_sample)

        # VECTORIZED: Trace all rays at once
        si = scene.ray_intersect(rays)

        # Extract depths
        depth_values = dr.select(si.is_valid(), si.t, 1e10)

        # Convert to numpy and reshape
        depth_map = np.array(depth_values).reshape(height, width).astype(np.float32)
        depths.append(depth_map)

    # Compute global min/max (excluding invalid/inf values)
    valid_depths = []
    for depth in depths:
        valid = depth[depth < 1e9]  # Filter out invalid depths
        if len(valid) > 0:
            valid_depths.extend(valid.flatten().tolist())

    if len(valid_depths) == 0:
        print("\nWarning: No valid depth values found!")
        global_min_depth = 0.0
        global_max_depth = 1.0
    else:
        global_min_depth = float(np.min(valid_depths))
        global_max_depth = float(np.max(valid_depths))

    print(f"\nGlobal depth range: [{global_min_depth:.4f}, {global_max_depth:.4f}]")

    # Second pass: save with consistent normalization
    print(f"Saving depth maps (pass 2/2: applying colormap and saving)...")
    for i, depth_map in enumerate(depths):
        print(f"Saving depth {i+1}/{len(depths)}...", end='\r')

        # Save as npy
        if save_npy:
            npy_path = os.path.join(output_dir, f"{prefix}_{i:04d}.npy")
            np.save(npy_path, depth_map)

        # Save as PNG with turbo colormap
        if save_png:
            png_path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
            # Apply turbo colormap with consistent normalization
            depth_rgb = apply_turbo_colormap(depth_map, global_min_depth, global_max_depth)
            # Convert to 8-bit
            depth_8bit = np.clip(depth_rgb * 255.0, 0, 255).astype(np.uint8)
            mi.Bitmap(depth_8bit).write(png_path)

    print(f"\nRendered and saved {len(depths)} depth maps to {output_dir}")
    return depths, global_min_depth, global_max_depth


def render_and_save_normals(
    scene: mi.Scene,
    sensors: List[mi.Sensor],
    output_dir: str,
    prefix: str = "normal",
    spp: int = 128,
    save_npy: bool = True,
    save_png: bool = True
) -> List[np.ndarray]:
    """
    Render surface normal maps from sensors and save to disk (vectorized).

    Surface normals are mapped to RGB colors using:
    - X component → Red channel
    - Y component → Green channel
    - Z component → Blue channel
    Remapped from [-1, 1] to [0, 1] using: color = (normal + 1.0) / 2.0

    Args:
        scene: Mitsuba scene to render
        sensors: List of camera sensors
        output_dir: Directory to save normal maps
        prefix: Prefix for normal filenames
        spp: Samples per pixel
        save_npy: If True, save as .npy files
        save_png: If True, save as .png files with XYZ→RGB colormapping

    Returns:
        List of rendered normal maps as numpy arrays (values in [-1, 1])
    """
    import drjit as dr

    os.makedirs(output_dir, exist_ok=True)

    normals = []
    for i, sensor in enumerate(sensors):
        print(f"Rendering normals {i+1}/{len(sensors)}...", end='\r')

        # Get image resolution from sensor
        film = sensor.film()
        res = film.size()
        width, height = res[0], res[1]
        num_pixels = width * height

        # VECTORIZED: Create all pixel positions at once
        idx = dr.arange(mi.UInt32, num_pixels)
        x = idx % width
        y = idx // width

        # Convert to normalized coordinates [0, 1]
        pos_x = (mi.Float(x) + 0.5) / float(width)
        pos_y = (mi.Float(y) + 0.5) / float(height)
        pos_sample = mi.Point2f(pos_x, pos_y)

        # VECTORIZED: Sample all rays at once
        wavelength_sample = 0.5
        time_sample = 0.0
        aperture_sample = mi.Point2f(0.5, 0.5)

        rays, _ = sensor.sample_ray(time_sample, wavelength_sample, pos_sample, aperture_sample)

        # VECTORIZED: Trace all rays at once
        si = scene.ray_intersect(rays)

        # Extract normals (shading frame normal)
        normal = si.sh_frame.n

        # Use dr.select to handle invalid intersections
        normal_x = dr.select(si.is_valid(), normal.x, 0.0)
        normal_y = dr.select(si.is_valid(), normal.y, 0.0)
        normal_z = dr.select(si.is_valid(), normal.z, 0.0)

        # Convert to numpy and reshape
        normal_map = np.stack([
            np.array(normal_x).reshape(height, width),
            np.array(normal_y).reshape(height, width),
            np.array(normal_z).reshape(height, width)
        ], axis=-1).astype(np.float32)

        normals.append(normal_map)

        # Save as npy (raw normals in [-1, 1])
        if save_npy:
            npy_path = os.path.join(output_dir, f"{prefix}_{i:04d}.npy")
            np.save(npy_path, normal_map)

        # Save as PNG with XYZ→RGB colormapping
        if save_png:
            png_path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
            # Remap from [-1, 1] to [0, 1]
            normal_rgb = (normal_map + 1.0) / 2.0
            normal_rgb = np.clip(normal_rgb, 0.0, 1.0)
            # Convert to 8-bit
            normal_8bit = np.clip(normal_rgb * 255.0, 0, 255).astype(np.uint8)
            mi.Bitmap(normal_8bit).write(png_path)

    print(f"\nRendered and saved {len(normals)} normal maps to {output_dir}")
    return normals


def render_and_save_all(
    scene: mi.Scene,
    sensors: List[mi.Sensor],
    output_base_dir: str,
    prefix: str = "view",
    spp: int = 128,
    save_npy: bool = True,
    save_png: bool = True
) -> Dict[str, Any]:
    """
    Render images, masks, depth, and normals in a single pass for efficiency.

    Args:
        scene: Mitsuba scene to render
        sensors: List of camera sensors
        output_base_dir: Base directory for outputs (will create subdirs)
        prefix: Prefix for filenames
        spp: Samples per pixel
        save_npy: If True, save as .npy files
        save_png: If True, save as .png files

    Returns:
        Dictionary with keys:
        - 'images': List of RGB images
        - 'masks': List of masks
        - 'depths': List of depth maps
        - 'normals': List of normal maps
        - 'depth_min': Global minimum depth
        - 'depth_max': Global maximum depth
    """
    # Create output directories
    images_dir = os.path.join(output_base_dir, "images")
    masks_dir = os.path.join(output_base_dir, "masks")
    depths_dir = os.path.join(output_base_dir, "depths")
    normals_dir = os.path.join(output_base_dir, "normals")

    for d in [images_dir, masks_dir, depths_dir, normals_dir]:
        os.makedirs(d, exist_ok=True)

    images = []
    masks = []
    depths = []
    normals = []

    print(f"Rendering all data (pass 1/2: computing values)...")

    # First pass: render all data
    for i, sensor in enumerate(sensors):
        print(f"Rendering view {i+1}/{len(sensors)}...", end='\r')

        # Get image resolution from sensor
        film = sensor.film()
        res = film.size()
        width, height = res[0], res[1]

        # VECTORIZED: Create all pixel indices at once
        num_pixels = width * height
        idx = dr.arange(mi.UInt32, num_pixels)
        x = idx % width
        y = idx // width

        # Convert to normalized coordinates
        pos_x = (mi.Float(x) + 0.5) / float(width)
        pos_y = (mi.Float(y) + 0.5) / float(height)
        pos_sample = mi.Point2f(pos_x, pos_y)

        # Samples for ray generation
        time_sample = mi.Float(0.0)
        wavelength_sample = mi.Float(0.5)
        aperture_sample = mi.Point2f(0.5, 0.5)

        # VECTORIZED: Sample all rays at once
        rays, _ = sensor.sample_ray(time_sample, wavelength_sample, pos_sample, aperture_sample)

        # VECTORIZED: Trace all rays at once
        si = scene.ray_intersect(rays)

        # Extract depths with conditional selection
        depth_values = dr.select(si.is_valid(), si.t, 1e10)
        depth_map = np.array(depth_values).reshape(height, width).astype(np.float32)

        # Extract normals with conditional selection
        normal = si.sh_frame.n
        nx = dr.select(si.is_valid(), normal.x, 0.0)
        ny = dr.select(si.is_valid(), normal.y, 0.0)
        nz = dr.select(si.is_valid(), normal.z, 0.0)

        # Convert to numpy and reshape
        nx_np = np.array(nx).reshape(height, width)
        ny_np = np.array(ny).reshape(height, width)
        nz_np = np.array(nz).reshape(height, width)
        normal_map = np.stack([nx_np, ny_np, nz_np], axis=-1).astype(np.float32)

        # Compute mask from ray intersections (si.is_valid())
        # This represents the probability that a ray hits an object vs background
        hit_values = dr.select(si.is_valid(), 1.0, 0.0)
        mask = np.array(hit_values).reshape(height, width, 1).astype(np.float32)

        # Render proper image with full lighting (this is fast with mi.render)
        img = mi.render(scene, sensor=sensor, spp=spp)
        image = np.array(img)

        # Store results
        images.append(image)
        masks.append(mask)
        depths.append(depth_map)
        normals.append(normal_map)

    # Compute global depth min/max
    valid_depths = []
    for depth in depths:
        valid = depth[depth < 1e9]
        if len(valid) > 0:
            valid_depths.extend(valid.flatten().tolist())

    if len(valid_depths) == 0:
        print("\nWarning: No valid depth values found!")
        depth_min = 0.0
        depth_max = 1.0
    else:
        depth_min = float(np.min(valid_depths))
        depth_max = float(np.max(valid_depths))

    print(f"\nGlobal depth range: [{depth_min:.4f}, {depth_max:.4f}]")

    # Second pass: save all data
    print(f"Saving all data (pass 2/2: writing files)...")
    for i in range(len(sensors)):
        print(f"Saving view {i+1}/{len(sensors)}...", end='\r')

        # Save images
        if save_npy:
            np.save(os.path.join(images_dir, f"{prefix}_{i:04d}.npy"), images[i])
        if save_png:
            img_8bit = np.clip(images[i][:, :, :3] * 255.0, 0, 255).astype(np.uint8)
            mi.Bitmap(img_8bit).write(os.path.join(images_dir, f"{prefix}_{i:04d}.png"))

        # Save masks
        if save_npy:
            np.save(os.path.join(masks_dir, f"{prefix}_mask_{i:04d}.npy"), masks[i])
        if save_png:
            mask_8bit = np.clip(masks[i][:, :, 0] * 255.0, 0, 255).astype(np.uint8)
            mi.Bitmap(mask_8bit).write(os.path.join(masks_dir, f"{prefix}_mask_{i:04d}.png"))

        # Save depths
        if save_npy:
            np.save(os.path.join(depths_dir, f"{prefix}_depth_{i:04d}.npy"), depths[i])
        if save_png:
            depth_rgb = apply_turbo_colormap(depths[i], depth_min, depth_max)
            depth_8bit = np.clip(depth_rgb * 255.0, 0, 255).astype(np.uint8)
            mi.Bitmap(depth_8bit).write(os.path.join(depths_dir, f"{prefix}_depth_{i:04d}.png"))

        # Save normals
        if save_npy:
            np.save(os.path.join(normals_dir, f"{prefix}_normal_{i:04d}.npy"), normals[i])
        if save_png:
            normal_rgb = (normals[i] + 1.0) / 2.0
            normal_rgb = np.clip(normal_rgb, 0.0, 1.0)
            normal_8bit = np.clip(normal_rgb * 255.0, 0, 255).astype(np.uint8)
            mi.Bitmap(normal_8bit).write(os.path.join(normals_dir, f"{prefix}_normal_{i:04d}.png"))

    print(f"\nRendered and saved {len(sensors)} views with all data to {output_base_dir}")

    return {
        'images': images,
        'masks': masks,
        'depths': depths,
        'normals': normals,
        'depth_min': depth_min,
        'depth_max': depth_max
    }


def save_poses(poses: List[np.ndarray], filepath: str) -> None:
    """
    Save camera poses to numpy file.

    Args:
        poses: List of 4x4 pose matrices
        filepath: Path to save poses (should end in .npy)
    """
    poses_array = np.stack(poses, axis=0)  # Shape: (N, 4, 4)
    np.save(filepath, poses_array)
    print(f"Saved {len(poses)} poses to {filepath}")


def load_poses(filepath: str) -> List[np.ndarray]:
    """
    Load camera poses from numpy file.

    Args:
        filepath: Path to poses file (.npy)

    Returns:
        List of 4x4 pose matrices
    """
    poses_array = np.load(filepath)
    poses = [poses_array[i] for i in range(len(poses_array))]
    print(f"Loaded {len(poses)} poses from {filepath}")
    return poses


def load_images(image_dir: str, prefix: str = "image") -> List[np.ndarray]:
    """
    Load images from numpy files.

    Args:
        image_dir: Directory containing .npy image files
        prefix: Prefix used for image filenames

    Returns:
        List of images as numpy arrays
    """
    # Find all matching .npy files
    files = sorted([f for f in os.listdir(image_dir) if f.startswith(prefix) and f.endswith('.npy')])

    images = []
    for f in files:
        img = np.load(os.path.join(image_dir, f))
        images.append(img)

    print(f"Loaded {len(images)} images from {image_dir}")
    return images


def poses_to_sensors(
    poses: List[np.ndarray],
    render_res: int = 256,
    fov: float = 45.0
) -> List[mi.Sensor]:
    """
    Convert pose matrices to Mitsuba sensors.

    Args:
        poses: List of 4x4 pose matrices (camera to world)
        render_res: Rendering resolution
        fov: Field of view in degrees

    Returns:
        List of Mitsuba sensors
    """
    sensors = []

    for pose in poses:
        # Extract camera position and orientation from pose
        origin = pose[:3, 3]
        right = pose[:3, 0]
        up = pose[:3, 1]
        forward = -pose[:3, 2]  # Camera looks down -Z

        target = origin + forward

        sensor = mi.load_dict({
            'type': 'perspective',
            'fov': fov,
            'to_world': mi.ScalarTransform4f.look_at(
                origin=origin.tolist(),
                target=target.tolist(),
                up=up.tolist()
            ),
            'film': {
                'type': 'hdrfilm',
                'width': render_res,
                'height': render_res,
                'filter': {'type': 'box'},
                'pixel_format': 'rgba'
            }
        })
        sensors.append(sensor)

    return sensors


def create_cached_cameras_and_images(
    scene: mi.Scene,
    cache_dir: str,
    render_config: Optional['RenderConfig'] = None,
    num_train: Optional[int] = None,
    num_test: Optional[int] = None,
    center: Optional[List[float]] = None,
    radius: Optional[float] = None,
    render_res: Optional[int] = None,
    fov: Optional[float] = None,
    spp: Optional[int] = None,
    train_hemisphere_normal: Optional[List[float]] = None,
    test_min_elevation: Optional[float] = None,
    test_max_elevation: Optional[float] = None,
    test_num_loops: Optional[float] = None,
    seed: Optional[int] = None,
    force_rerender: bool = False
) -> Dict[str, Any]:
    """
    Create or load cached training and test cameras with rendered images.

    This function creates:
    - Random training views uniformly distributed on hemisphere (with reflection)
    - Test spiral trajectory that changes elevation while rotating
    - Saves/loads everything to/from cache

    Args:
        scene: Mitsuba scene for rendering
        cache_dir: Directory for caching
        render_config: RenderConfig object with all rendering parameters (RECOMMENDED).
                      If provided, overrides individual parameters.
        num_train: Number of training cameras (overridden by render_config)
        num_test: Number of test trajectory cameras (overridden by render_config)
        center: Scene center (overridden by render_config)
        radius: Camera distance from center (overridden by render_config)
        render_res: Rendering resolution (overridden by render_config)
        fov: Field of view (overridden by render_config)
        spp: Samples per pixel for rendering (overridden by render_config)
        train_hemisphere_normal: Normal vector defining hemisphere (overridden by render_config)
        test_min_elevation: Starting elevation for test spiral (overridden by render_config)
        test_max_elevation: Ending elevation for test spiral (overridden by render_config)
        test_num_loops: Number of 360° loops for test spiral (overridden by render_config)
        seed: Random seed for training cameras (overridden by render_config)
        force_rerender: If True, force re-rendering even if cache exists

    Returns:
        Dictionary with keys: 'train_sensors', 'train_poses', 'train_images', 'train_masks', 'train_depths', 'train_normals',
                              'test_sensors', 'test_poses', 'test_images', 'test_masks', 'test_depths', 'test_normals'
    """
    # Import here to avoid circular dependency
    from tlir.config import RenderConfig

    # Use render_config if provided, otherwise use individual parameters with defaults
    if render_config is not None:
        num_train = render_config.num_train
        num_test = render_config.num_test
        center = render_config.camera_center
        radius = render_config.camera_radius
        render_res = render_config.render_res
        fov = render_config.fov
        spp = render_config.spp
        train_hemisphere_normal = render_config.hemisphere_normal
        test_min_elevation = render_config.test_min_elevation
        test_max_elevation = render_config.test_max_elevation
        test_num_loops = render_config.test_num_loops
        seed = render_config.seed
        up_vector = render_config.up_vector
    else:
        # Use individual parameters with defaults
        num_train = num_train if num_train is not None else 100
        num_test = num_test if num_test is not None else 100
        center = center if center is not None else [0.0, 0.0, 0.0]
        radius = radius if radius is not None else 1.3
        render_res = render_res if render_res is not None else 256
        fov = fov if fov is not None else 45.0
        spp = spp if spp is not None else 128
        train_hemisphere_normal = train_hemisphere_normal if train_hemisphere_normal is not None else [0, 0, 1]
        test_min_elevation = test_min_elevation if test_min_elevation is not None else 10.0
        test_max_elevation = test_max_elevation if test_max_elevation is not None else 80.0
        test_num_loops = test_num_loops if test_num_loops is not None else 1.0
        seed = seed if seed is not None else 42
        up_vector = [0, 1, 0]  # Default Y-up

    os.makedirs(cache_dir, exist_ok=True)

    # Cache file paths
    train_poses_path = os.path.join(cache_dir, "train_poses.npy")
    train_data_dir = os.path.join(cache_dir, "train")
    test_poses_path = os.path.join(cache_dir, "test_poses.npy")
    test_data_dir = os.path.join(cache_dir, "test")

    # Check if cache exists (check for subdirs created by render_and_save_all)
    train_cache_exists = (
        os.path.exists(train_poses_path) and
        os.path.exists(os.path.join(train_data_dir, "images")) and
        os.path.exists(os.path.join(train_data_dir, "masks")) and
        os.path.exists(os.path.join(train_data_dir, "depths")) and
        os.path.exists(os.path.join(train_data_dir, "normals"))
    )
    test_cache_exists = (
        os.path.exists(test_poses_path) and
        os.path.exists(os.path.join(test_data_dir, "images")) and
        os.path.exists(os.path.join(test_data_dir, "masks")) and
        os.path.exists(os.path.join(test_data_dir, "depths")) and
        os.path.exists(os.path.join(test_data_dir, "normals"))
    )

    # Handle partial caching: train and test independently
    if force_rerender:
        train_cache_exists = False
        test_cache_exists = False

    # Load or render training data
    if train_cache_exists:
        print("Loading training data from cache...")
        train_poses = load_poses(train_poses_path)
        train_sensors = poses_to_sensors(train_poses, render_res, fov)
        train_images = load_images(os.path.join(train_data_dir, "images"), "train")
        train_masks = load_images(os.path.join(train_data_dir, "masks"), "train_mask")
        train_depths = load_images(os.path.join(train_data_dir, "depths"), "train_depth")
        train_normals = load_images(os.path.join(train_data_dir, "normals"), "train_normal")
    else:
        print("Creating training cameras and rendering...")
        # Create training cameras (random on hemisphere with reflection)
        print(f"\nCreating {num_train} training cameras...")
        train_sensors, train_poses = create_random_hemisphere_cameras(
            num_cameras=num_train,
            center=center,
            radius=radius,
            render_res=render_res,
            fov=fov,
            look_inward=True,
            hemisphere_normal=train_hemisphere_normal,
            up_vector=up_vector,
            seed=seed
        )

        # Render all training data
        print(f"\nRendering all training data (images, masks, depths, normals)...")
        train_data = render_and_save_all(
            scene, train_sensors, train_data_dir, prefix="train", spp=spp
        )
        train_images = train_data['images']
        train_masks = train_data['masks']
        train_depths = train_data['depths']
        train_normals = train_data['normals']

        # Save training poses
        save_poses(train_poses, train_poses_path)

    # Load or render test data
    if test_cache_exists:
        print("Loading test data from cache...")
        test_poses = load_poses(test_poses_path)
        test_sensors = poses_to_sensors(test_poses, render_res, fov)
        test_images = load_images(os.path.join(test_data_dir, "images"), "test")
        test_masks = load_images(os.path.join(test_data_dir, "masks"), "test_mask")
        test_depths = load_images(os.path.join(test_data_dir, "depths"), "test_depth")
        test_normals = load_images(os.path.join(test_data_dir, "normals"), "test_normal")
    else:
        print("Creating test cameras and rendering...")
        # Create test trajectory (spiral path with changing elevation)
        print(f"Creating {num_test} test spiral trajectory cameras...")
        test_sensors, test_poses = create_trajectory_cameras(
            num_cameras=num_test,
            center=center,
            radius=radius,
            render_res=render_res,
            fov=fov,
            min_elevation=test_min_elevation,
            max_elevation=test_max_elevation,
            look_inward=True,
            num_loops=test_num_loops,
            up_vector=up_vector
        )

        # Render all test data
        print(f"\nRendering all test data (images, masks, depths, normals)...")
        test_data = render_and_save_all(
            scene, test_sensors, test_data_dir, prefix="test", spp=spp
        )
        test_images = test_data['images']
        test_masks = test_data['masks']
        test_depths = test_data['depths']
        test_normals = test_data['normals']

        # Save test poses
        save_poses(test_poses, test_poses_path)

    return {
        'train_sensors': train_sensors,
        'train_poses': train_poses,
        'train_images': train_images,
        'train_masks': train_masks,
        'train_depths': train_depths,
        'train_normals': train_normals,
        'test_sensors': test_sensors,
        'test_poses': test_poses,
        'test_images': test_images,
        'test_masks': test_masks,
        'test_depths': test_depths,
        'test_normals': test_normals
    }
