"""
Example script demonstrating how to use Gaussian noise augmentation for radiance field training.

This script shows how to configure and use the gt_noise_std parameter to add random
Gaussian noise to ground truth images during training, which can help improve
generalization to heldout views.

Two modes are supported:
1. Uniform noise: Noise applied to all pixels (gt_noise_use_mask=False)
2. Masked noise: Noise applied only to background regions (gt_noise_use_mask=True, default)
"""

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from tlir import config as cf
from tlir import training

# Example 1: Quick test with no noise (baseline)
print("=" * 60)
print("Example 1: Training without noise (baseline)")
print("=" * 60)

config_no_noise = cf.create_config(
    "test_no_noise",
    num_stages=2,
    num_iterations_per_stage=5,
    grid_init_res=16,
    gt_noise_std=0.0  # No noise
)

print(f"Ground truth noise std: {config_no_noise.gt_noise_std}")

# Example 2: Training with small noise
print("\n" + "=" * 60)
print("Example 2: Training with small Gaussian noise")
print("=" * 60)

config_small_noise = cf.create_config(
    "test_small_noise",
    num_stages=2,
    num_iterations_per_stage=5,
    grid_init_res=16,
    gt_noise_std=0.01  # Small noise, 1% std deviation
)

print(f"Ground truth noise std: {config_small_noise.gt_noise_std}")

# Example 3: Training with moderate noise
print("\n" + "=" * 60)
print("Example 3: Training with moderate Gaussian noise")
print("=" * 60)

config_moderate_noise = cf.create_config(
    "test_moderate_noise",
    num_stages=2,
    num_iterations_per_stage=5,
    grid_init_res=16,
    gt_noise_std=0.05  # Moderate noise, 5% std deviation
)

print(f"Ground truth noise std: {config_moderate_noise.gt_noise_std}")

# Example 4: Complete training workflow with noise
print("\n" + "=" * 60)
print("Example 4: Complete training workflow")
print("=" * 60)

# Create scene and sensors
scene = training.create_scene('rf_prb')
sensors = training.create_sensors(num_sensors=4, render_res=128)

# Create reference scene and render ground truth images
scene_ref = training.create_scene_reference(config_moderate_noise)
ref_images = [mi.render(scene_ref, sensor=sensors[i], spp=128) for i in range(len(sensors))]

print(f"Created {len(ref_images)} reference images")
print(f"Image shape: {ref_images[0].shape}")

# Create training config from experiment config
train_config = training.TrainingConfig(
    num_stages=config_moderate_noise.num_stages,
    num_iterations_per_stage=config_moderate_noise.num_iterations_per_stage,
    learning_rate=config_moderate_noise.learning_rate,
    grid_init_res=config_moderate_noise.grid_init_res,
    sh_degree=config_moderate_noise.sh_degree,
    use_relu=config_moderate_noise.use_relu,
    max_initial_density=config_moderate_noise.max_initial_density,
    spp=config_moderate_noise.spp,
    loss_type=config_moderate_noise.loss_type,
    enable_upsampling=config_moderate_noise.enable_upsampling,
    gt_noise_std=config_moderate_noise.gt_noise_std  # Pass the noise parameter
)

print(f"\nTraining config:")
print(f"  - Noise std: {train_config.gt_noise_std}")
print(f"  - Loss type: {train_config.loss_type}")
print(f"  - Num stages: {train_config.num_stages}")
print(f"  - Iterations per stage: {train_config.num_iterations_per_stage}")

# Uncomment below to actually run training:
# print("\nStarting training...")
# results = training.train_radiance_field(
#     scene=scene,
#     sensors=sensors,
#     ref_images=ref_images,
#     config=train_config
# )
# print(f"Training complete! Final loss: {results['losses'][-1]:.6f}")

print("\n" + "=" * 60)
print("Example 5: Masked vs Uniform Noise")
print("=" * 60)

# Masked noise (default): applies noise only to background regions
config_masked = cf.create_config(
    "test_masked_noise",
    num_stages=2,
    num_iterations_per_stage=5,
    grid_init_res=16,
    gt_noise_std=0.03,
    gt_noise_use_mask=True  # Apply noise only to background (default)
)

print(f"Masked noise configuration:")
print(f"  - gt_noise_std: {config_masked.gt_noise_std}")
print(f"  - gt_noise_use_mask: {config_masked.gt_noise_use_mask}")
print(f"  - Effect: Noise applied only to background (mask=0), not objects (mask=1)")

# Uniform noise: applies noise to all pixels
config_uniform = cf.create_config(
    "test_uniform_noise",
    num_stages=2,
    num_iterations_per_stage=5,
    grid_init_res=16,
    gt_noise_std=0.03,
    gt_noise_use_mask=False  # Apply noise uniformly to all pixels
)

print(f"\nUniform noise configuration:")
print(f"  - gt_noise_std: {config_uniform.gt_noise_std}")
print(f"  - gt_noise_use_mask: {config_uniform.gt_noise_use_mask}")
print(f"  - Effect: Noise applied uniformly to all pixels")

print("\nMasked noise workflow:")
print("  1. Render object masks from reference scene at training start")
print("  2. Mask is 1 where objects are, 0 where background is")
print("  3. Noise multiplied by (1 - mask): noise * (1 - mask)")
print("  4. Objects get no noise, background gets full noise")
print("  5. Semi-transparent regions get partial noise")

# Uncomment to run training with masked noise:
# print("\nStarting training with masked noise...")
# results_masked = training.train_radiance_field(
#     scene=scene,
#     sensors=sensors,
#     ref_images=ref_images,
#     config=train_config_masked,
#     ref_scene=scene_ref  # Required for rendering masks
# )

print("\n" + "=" * 60)
print("Example complete!")
print("=" * 60)
print("\nRecommended noise levels:")
print("  - No noise (baseline):  gt_noise_std=0.0")
print("  - Small noise:          gt_noise_std=0.01 to 0.02")
print("  - Moderate noise:       gt_noise_std=0.03 to 0.05")
print("  - Large noise:          gt_noise_std=0.05 to 0.10")
print("\nNoise modes:")
print("  - Masked (default):     gt_noise_use_mask=True  (noise on background only)")
print("  - Uniform:              gt_noise_use_mask=False (noise on all pixels)")
print("\nNote: Noise is added independently for each iteration, sensor, and stage.")
print("This provides diverse augmentation throughout training.")
print("When using masked noise, pass ref_scene to train_radiance_field().")
