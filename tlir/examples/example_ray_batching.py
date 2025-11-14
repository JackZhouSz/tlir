"""
Example script demonstrating ray-based training for radiance fields.

Ray-based training samples random rays from all training images instead of
rendering complete images. This approach:
- Improves training efficiency
- Reduces memory usage
- Often leads to better convergence
- Is the standard approach used in NeRF-style methods

Comparison with image-based training:
- Image-based: Iterates over images, renders full images, computes loss per image
- Ray-based: Samples random rays from all images, renders only those rays, computes loss per batch
"""

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from tlir import config as cf
from tlir import training

# Example 1: Basic ray-based training
print("=" * 70)
print("Example 1: Ray-based Training (Basic)")
print("=" * 70)

config_ray = cf.create_config(
    "test_ray_batching",
    num_stages=4,
    num_iterations_per_stage=50,  # More iterations, smaller batches
    grid_init_res=16,
    use_ray_batching=True,        # Enable ray batching
    rays_per_batch=4096,          # Number of rays per iteration
    spp=1,
    loss_type='l2'
)

print(f"Ray-based training configuration:")
print(f"  - use_ray_batching: {config_ray.use_ray_batching}")
print(f"  - rays_per_batch: {config_ray.rays_per_batch}")
print(f"  - num_stages: {config_ray.num_stages}")
print(f"  - iterations_per_stage: {config_ray.num_iterations_per_stage}")
print()

# Example 2: Comparison of image vs ray-based training
print("=" * 70)
print("Example 2: Image-based vs Ray-based Training")
print("=" * 70)

# Image-based configuration
config_image = cf.create_config(
    "test_image_based",
    num_stages=4,
    num_iterations_per_stage=15,  # Fewer iterations, full images
    grid_init_res=16,
    use_ray_batching=False,       # Traditional image-based
    spp=1,
    loss_type='l2'
)

print("Image-based training:")
print(f"  - Mode: Iterate over {4} full images per iteration")
print(f"  - Iterations: {config_image.num_iterations_per_stage} per stage")
print(f"  - Total gradient steps: {config_image.num_stages * config_image.num_iterations_per_stage}")
print()

print("Ray-based training:")
print(f"  - Mode: Sample {config_ray.rays_per_batch} random rays per iteration")
print(f"  - Iterations: {config_ray.num_iterations_per_stage} per stage")
print(f"  - Total gradient steps: {config_ray.num_stages * config_ray.num_iterations_per_stage}")
print(f"  - Total rays sampled: {config_ray.rays_per_batch * config_ray.num_stages * config_ray.num_iterations_per_stage:,}")
print()

# Example 3: Complete training workflow with ray batching
print("=" * 70)
print("Example 3: Complete Workflow with Ray Batching")
print("=" * 70)

# Create scene and sensors
scene = training.create_scene('rf_prb')
sensors = training.create_sensors(num_sensors=8, render_res=128)

# Create reference scene and render ground truth images
scene_ref = training.create_scene_reference(config_ray)
ref_images = [mi.render(scene_ref, sensor=sensors[i], spp=128) for i in range(len(sensors))]

print(f"Created {len(ref_images)} reference images")
print(f"Image shape: {ref_images[0].shape}")
print(f"Total pixels: {ref_images[0].shape[0] * ref_images[0].shape[1] * len(ref_images):,}")
print()

# Create training config
train_config = training.TrainingConfig(
    num_stages=config_ray.num_stages,
    num_iterations_per_stage=config_ray.num_iterations_per_stage,
    learning_rate=config_ray.learning_rate,
    grid_init_res=config_ray.grid_init_res,
    sh_degree=config_ray.sh_degree,
    use_relu=config_ray.use_relu,
    max_initial_density=config_ray.max_initial_density,
    spp=config_ray.spp,
    loss_type=config_ray.loss_type,
    enable_upsampling=config_ray.enable_upsampling,
    use_ray_batching=config_ray.use_ray_batching,
    rays_per_batch=config_ray.rays_per_batch
)

print("Training configuration:")
print(f"  - Ray batching: {train_config.use_ray_batching}")
print(f"  - Rays per batch: {train_config.rays_per_batch}")
print(f"  - Learning rate: {train_config.learning_rate}")
print()

# Uncomment to run training with ray batching:
# print("Starting ray-based training...")
# results = training.train_radiance_field_ray_batching(
#     scene=scene,
#     sensors=sensors,
#     ref_images=ref_images,
#     config=train_config
# )
# print(f"Training complete! Final loss: {results['losses'][-1]:.6f}")

# Example 4: Ray batching with advanced features
print("=" * 70)
print("Example 4: Ray Batching with Advanced Features")
print("=" * 70)

total_iterations = 4 * 50  # stages * iterations

config_advanced = cf.create_config(
    "test_ray_advanced",
    num_stages=4,
    num_iterations_per_stage=50,
    grid_init_res=16,
    use_ray_batching=True,
    rays_per_batch=4096,
    enable_upsampling=False,                                     # Disable for SPN
    gt_noise_std=0.02,                                          # GT noise augmentation
    gt_noise_use_mask=True,                                     # Apply noise to background only
    stochastic_preconditioning_starting_alpha=0.01,             # SPN
    stochastic_preconditioning_iterations=total_iterations // 2
)

print("Advanced ray-based training:")
print(f"  ✓ Ray batching: {config_advanced.rays_per_batch} rays/batch")
print(f"  ✓ GT noise: std={config_advanced.gt_noise_std}, masked={config_advanced.gt_noise_use_mask}")
print(f"  ✓ Stochastic preconditioning: alpha={config_advanced.stochastic_preconditioning_starting_alpha}")
print(f"  ✓ Multi-resolution: disabled (using SPN)")
print()

print("=" * 70)
print("Example complete!")
print("=" * 70)
print()
print("Key Benefits of Ray Batching:")
print("  1. More efficient gradient updates (random rays vs sequential images)")
print("  2. Better memory utilization (small batches vs full images)")
print("  3. Improved convergence (more diverse gradients per epoch)")
print("  4. Standard in modern NeRF methods")
print()
print("When to use:")
print("  - Many training views (>8 images): Ray batching recommended")
print("  - Few training views (<8 images): Image-based may be sufficient")
print("  - Memory constrained: Use smaller rays_per_batch")
print("  - Fast iteration: Increase rays_per_batch")
print()
print("Recommended batch sizes:")
print("  - Small scenes, low res: 1024-2048 rays")
print("  - Medium scenes: 4096-8192 rays (default)")
print("  - Large scenes, high res: 16384+ rays")
