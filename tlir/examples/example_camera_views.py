"""
Example script demonstrating camera view utilities.

Shows how to:
- Create random training views on upper hemisphere
- Create test trajectory that loops around
- Render and cache images and poses
- Use cached data for training
"""

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

import numpy as np
from tlir import config as cf
from tlir import training
from tlir import camera_utils

# Configuration
CENTER = [0.0, 0.0, 0.0]
RADIUS = 1.3
RENDER_RES = 256
FOV = 45.0
SPP = 128

print("=" * 70)
print("Example 1: Random cameras with uniform hemisphere distribution")
print("=" * 70)

# Create random cameras uniformly distributed on hemisphere surface
# Uses 3D normal distribution sampling + reflection to ensure all cameras
# are on the correct hemisphere (positive dot product with normal)
# Can filter by elevation using min/max_elevation
train_sensors, train_poses = camera_utils.create_random_hemisphere_cameras(
    num_cameras=8,
    center=CENTER,
    radius=RADIUS,
    render_res=RENDER_RES,
    fov=FOV,
    look_inward=True,
    min_elevation=10.0,   # Filter: avoid horizon
    max_elevation=80.0,   # Filter: avoid zenith
    hemisphere_normal=[0, 0, 1],  # Upper hemisphere (default)
    seed=42
)

print(f"Created {len(train_sensors)} training cameras")
print(f"Distribution: Uniform on hemisphere surface (3D normal + reflection)")
print(f"Method: Sample N(0,1) in 3D, normalize, reflect if needed")
print(f"Hemisphere normal: [0, 0, 1] (upper hemisphere)")
print(f"Elevation range: 10° to 80° (rejection sampling)")
print(f"\nFirst camera pose (4x4 matrix):")
print(train_poses[0])

print("\n" + "=" * 70)
print("Example 2: Spiral trajectory cameras (heldout test views)")
print("=" * 70)

# Create spiral test trajectory with changing elevation
test_sensors, test_poses = camera_utils.create_trajectory_cameras(
    num_cameras=36,
    center=CENTER,
    radius=RADIUS,
    render_res=RENDER_RES,
    fov=FOV,
    min_elevation=10.0,   # Starts at 10 degrees
    max_elevation=80.0,   # Ends at 80 degrees
    look_inward=True,
    start_angle=0.0,
    num_loops=1.0         # One complete 360° rotation
)

print(f"Created {len(test_sensors)} test spiral trajectory cameras")
print(f"Spiral: elevation changes from 10° to 80° during rotation")
print(f"Number of complete rotations: 1.0 (360 degrees)")
print(f"This creates a smooth spiral path covering multiple elevations")

print("\n" + "=" * 70)
print("Example 3: Create and cache training/test data")
print("=" * 70)

# Create config for scene
config = cf.create_config(
    "camera_example",
    scene_name="lego",
    render_res=RENDER_RES
)

# Create reference scene
scene_ref = training.create_scene_reference(config)

# Create and cache everything (or load from cache if it exists)
data = camera_utils.create_cached_cameras_and_images(
    scene=scene_ref,
    num_train=16,
    num_test=36,
    cache_dir="./cache/camera_views",
    center=CENTER,
    radius=RADIUS,
    render_res=RENDER_RES,
    fov=FOV,
    spp=SPP,
    test_min_elevation=10.0,    # Spiral start elevation
    test_max_elevation=80.0,    # Spiral end elevation
    test_num_loops=1.0,         # One full rotation
    seed=42,
    force_rerender=False  # Set to True to force re-rendering
)

print(f"\nLoaded data:")
print(f"  Training: {len(data['train_sensors'])} cameras, {len(data['train_images'])} images")
print(f"  Test: {len(data['test_sensors'])} cameras, {len(data['test_images'])} images")
print(f"  Training image shape: {data['train_images'][0].shape}")
print(f"  Test image shape: {data['test_images'][0].shape}")

print("\n" + "=" * 70)
print("Example 4: Using cached data for training")
print("=" * 70)

# Create trainable scene
scene = training.create_scene('rf_prb')

# Create training config
train_config = training.TrainingConfig(
    num_stages=4,
    num_iterations_per_stage=15,
    learning_rate=0.2,
    grid_init_res=16,
    spp=1,  # Low spp for fast training
    loss_type='l2'
)

print(f"Training configuration:")
print(f"  Stages: {train_config.num_stages}")
print(f"  Iterations per stage: {train_config.num_iterations_per_stage}")
print(f"  Training views: {len(data['train_sensors'])}")
print(f"  Heldout test views: {len(data['test_sensors'])}")

# Uncomment to run training:
# print("\nStarting training...")
# results = training.train_radiance_field(
#     scene=scene,
#     sensors=data['train_sensors'],
#     ref_images=data['train_images'],
#     config=train_config
# )
# print(f"Training complete! Final loss: {results['losses'][-1]:.6f}")

# # Evaluate on test views
# print("\nEvaluating on test views...")
# params = results['final_params']
# test_loss = 0.0
# for i, (sensor, ref_img) in enumerate(zip(data['test_sensors'], data['test_images'])):
#     rendered = mi.render(scene, params, sensor=sensor, spp=16)
#     loss = training.compute_loss(rendered, ref_img, train_config.loss_type)
#     test_loss += loss.array[0]
# test_loss /= len(data['test_sensors'])
# print(f"Test loss: {test_loss:.6f}")

print("\n" + "=" * 70)
print("Example 5: Hemisphere vs Sphere sampling")
print("=" * 70)

# Create cameras on upper hemisphere (with reflection)
hemisphere_sensors, hemisphere_poses = camera_utils.create_random_hemisphere_cameras(
    num_cameras=10,
    center=CENTER,
    radius=RADIUS,
    render_res=RENDER_RES,
    fov=FOV,
    look_inward=True,
    hemisphere_normal=[0, 0, 1],  # Upper hemisphere
    min_elevation=-90.0,  # No filtering, just hemisphere constraint
    max_elevation=90.0,
    seed=42
)

# Create cameras on full sphere (no reflection)
sphere_sensors, sphere_poses = camera_utils.create_random_sphere_cameras(
    num_cameras=10,
    center=CENTER,
    radius=RADIUS,
    render_res=RENDER_RES,
    fov=FOV,
    look_inward=True,
    min_elevation=-90.0,
    max_elevation=90.0,
    seed=42  # Same seed for comparison
)

# Analyze camera positions
print(f"\nHemisphere cameras (with reflection):")
hemisphere_z_coords = [pose[2, 3] - CENTER[2] for pose in hemisphere_poses]
print(f"  Z-coordinates relative to center: {[f'{z:.3f}' for z in hemisphere_z_coords]}")
print(f"  All positive (upper hemisphere): {all(z >= 0 for z in hemisphere_z_coords)}")

print(f"\nSphere cameras (no reflection):")
sphere_z_coords = [pose[2, 3] - CENTER[2] for pose in sphere_poses]
print(f"  Z-coordinates relative to center: {[f'{z:.3f}' for z in sphere_z_coords]}")
print(f"  Some negative (full sphere): {any(z < 0 for z in sphere_z_coords)}")

print("\n" + "=" * 70)
print("Example 6: Full sphere and outward-looking cameras")
print("=" * 70)

# Create cameras looking outward from full sphere
outward_sensors, outward_poses = camera_utils.create_random_sphere_cameras(
    num_cameras=10,
    center=CENTER,
    radius=RADIUS,
    render_res=RENDER_RES,
    fov=FOV,
    look_inward=False,  # Look outward
    seed=123
)

print(f"Created {len(outward_sensors)} cameras on full sphere")
print(f"Cameras look outward (away from center)")

print("\n" + "=" * 70)
print("Summary of camera creation functions")
print("=" * 70)
print("\n1. create_random_hemisphere_cameras():")
print("   - **Uniformly distributed** on hemisphere surface")
print("   - Method: Sample N(0,1) in 3D, normalize, reflect if needed")
print("   - Reflection ensures positive dot product with hemisphere_normal")
print("   - Not biased toward poles like naive sampling")
print("   - Rejection sampling for elevation filtering")
print("   - hemisphere_normal parameter defines the hemisphere plane")
print("   - Default hemisphere_normal=[0,0,1] = upper hemisphere")
print("   - Good for training views with proper coverage")
print("\n2. create_random_sphere_cameras():")
print("   - **Uniformly distributed** on full sphere surface")
print("   - Same as hemisphere but WITHOUT reflection step")
print("   - hemisphere_normal is set to None (no reflection)")
print("   - Supports elevation filtering")
print("   - Use when you need full sphere coverage")
print("\n3. create_trajectory_cameras():")
print("   - **Spiral path with changing elevation**")
print("   - Smoothly transitions from min_elevation to max_elevation")
print("   - Rotates num_loops times (360° per loop)")
print("   - Good for test/validation with diverse viewpoints")
print("   - Creates smooth videos when rendered")
print("\n4. Hemisphere vs Sphere:")
print("   - Hemisphere: reflection ensures all cameras on one side")
print("   - Sphere: no reflection, cameras on both sides")
print("   - Same seed produces different distributions")
print("\n5. create_cached_cameras_and_images():")
print("   - All-in-one function")
print("   - Creates training (uniform hemisphere) + test (spiral) cameras")
print("   - Renders and caches everything to disk")
print("   - Fast loading on subsequent runs")
print("\nKey parameters:")
print("  - look_inward: cameras look toward/away from center")
print("  - elevation angles: [-90=nadir, 0=horizon, 90=zenith]")
print("  - min/max_elevation: filter range or spiral range")
print("  - hemisphere_normal: defines hemisphere plane (None = full sphere)")
print("  - num_loops: for spiral trajectory (e.g., 1.0 = 360°)")
print("  - seed: for reproducible random camera placement")
print("\nImportant:")
print("  ✓ Hemisphere uses 3D normal + REFLECTION for uniform sampling")
print("  ✓ Sphere uses 3D normal WITHOUT reflection")
print("  ✓ Method: Sample from N(0,1)^3, normalize, reflect if hemisphere")
print("  ✓ Test uses SPIRAL trajectory (covers multiple elevations)")
print("  ✓ No bias toward poles - proper coverage")
print("  ✓ Rejection sampling for clean elevation filtering")
