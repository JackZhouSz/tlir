"""
Example script demonstrating stochastic preconditioning for radiance field training.

Stochastic preconditioning adds normally distributed noise to volume query points
with a decaying schedule. This can serve as an alternative to multi-resolution
training for improving generalization.

Based on the RawNeRF implementation in nerfstudio.
"""

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from tlir import config as cf
from tlir import training

# Example 1: Training without stochastic preconditioning (baseline)
print("=" * 70)
print("Example 1: Training without stochastic preconditioning (baseline)")
print("=" * 70)

config_no_spn = cf.create_config(
    "test_no_spn",
    num_stages=4,
    num_iterations_per_stage=15,
    grid_init_res=16,
    stochastic_preconditioning_starting_alpha=0.0,  # Disabled
    stochastic_preconditioning_iterations=-1
)

print(f"Stochastic preconditioning: DISABLED")
print(f"  Starting alpha: {config_no_spn.stochastic_preconditioning_starting_alpha}")
print(f"  Iterations: {config_no_spn.stochastic_preconditioning_iterations}")

# Example 2: Training with moderate stochastic preconditioning
print("\n" + "=" * 70)
print("Example 2: Training with moderate stochastic preconditioning")
print("=" * 70)

# Total iterations = num_stages * num_iterations_per_stage
total_iterations = 4 * 15  # 60 iterations

config_moderate_spn = cf.create_config(
    "test_moderate_spn",
    num_stages=4,
    num_iterations_per_stage=15,
    grid_init_res=16,
    enable_upsampling=False,  # Disable multi-resolution when using SPN
    stochastic_preconditioning_starting_alpha=0.01,  # 1% of grid cell
    stochastic_preconditioning_iterations=total_iterations // 2  # Decay over first half
)

print(f"Stochastic preconditioning: ENABLED")
print(f"  Starting alpha: {config_moderate_spn.stochastic_preconditioning_starting_alpha}")
print(f"  Decay iterations: {config_moderate_spn.stochastic_preconditioning_iterations}")
print(f"  Total training iterations: {total_iterations}")
print(f"  Note: Multi-resolution training disabled (use SPN as alternative)")

# Calculate decay
if config_moderate_spn.stochastic_preconditioning_iterations > 0:
    gamma = (1e-16 / config_moderate_spn.stochastic_preconditioning_starting_alpha) ** (
        1.0 / config_moderate_spn.stochastic_preconditioning_iterations
    )
    print(f"  Decay gamma: {gamma:.8f}")
    print(f"  Final alpha (after decay): ~1e-16")

# Example 3: Training with strong stochastic preconditioning
print("\n" + "=" * 70)
print("Example 3: Training with strong stochastic preconditioning")
print("=" * 70)

config_strong_spn = cf.create_config(
    "test_strong_spn",
    num_stages=4,
    num_iterations_per_stage=15,
    grid_init_res=16,
    enable_upsampling=False,
    stochastic_preconditioning_starting_alpha=0.05,  # 5% of grid cell
    stochastic_preconditioning_iterations=total_iterations  # Decay over full training
)

print(f"Stochastic preconditioning: ENABLED (STRONG)")
print(f"  Starting alpha: {config_strong_spn.stochastic_preconditioning_starting_alpha}")
print(f"  Decay iterations: {config_strong_spn.stochastic_preconditioning_iterations}")

# Example 4: Complete training workflow with stochastic preconditioning
print("\n" + "=" * 70)
print("Example 4: Complete training workflow")
print("=" * 70)

# Create scene and sensors
scene = training.create_scene('rf_prb')
sensors = training.create_sensors(num_sensors=4, render_res=128)

# Create reference scene and render ground truth images
scene_ref = training.create_scene_reference(config_moderate_spn)
ref_images = [mi.render(scene_ref, sensor=sensors[i], spp=128) for i in range(len(sensors))]

print(f"Created {len(ref_images)} reference images")
print(f"Image shape: {ref_images[0].shape}")

# Create training config from experiment config
train_config = training.TrainingConfig(
    num_stages=config_moderate_spn.num_stages,
    num_iterations_per_stage=config_moderate_spn.num_iterations_per_stage,
    learning_rate=config_moderate_spn.learning_rate,
    grid_init_res=config_moderate_spn.grid_init_res,
    sh_degree=config_moderate_spn.sh_degree,
    use_relu=config_moderate_spn.use_relu,
    max_initial_density=config_moderate_spn.max_initial_density,
    spp=config_moderate_spn.spp,
    loss_type=config_moderate_spn.loss_type,
    enable_upsampling=config_moderate_spn.enable_upsampling,
    gt_noise_std=config_moderate_spn.gt_noise_std,
    stochastic_preconditioning_starting_alpha=config_moderate_spn.stochastic_preconditioning_starting_alpha,
    stochastic_preconditioning_iterations=config_moderate_spn.stochastic_preconditioning_iterations
)

print(f"\nTraining config:")
print(f"  - Stochastic preconditioning starting alpha: {train_config.stochastic_preconditioning_starting_alpha}")
print(f"  - Stochastic preconditioning iterations: {train_config.stochastic_preconditioning_iterations}")
print(f"  - Multi-resolution training: {train_config.enable_upsampling}")
print(f"  - Loss type: {train_config.loss_type}")
print(f"  - Num stages: {train_config.num_stages}")
print(f"  - Iterations per stage: {train_config.num_iterations_per_stage}")

# Uncomment below to actually run training:
# print("\nStarting training with stochastic preconditioning...")
# results = training.train_radiance_field(
#     scene=scene,
#     sensors=sensors,
#     ref_images=ref_images,
#     config=train_config
# )
# print(f"Training complete! Final loss: {results['losses'][-1]:.6f}")

print("\n" + "=" * 70)
print("Example complete!")
print("=" * 70)
print("\nRecommended configurations:")
print("\n1. BASELINE (no stochastic preconditioning):")
print("   - stochastic_preconditioning_starting_alpha = 0.0")
print("   - stochastic_preconditioning_iterations = -1")
print("   - enable_upsampling = True")
print("\n2. MODERATE stochastic preconditioning (alternative to multi-res):")
print("   - stochastic_preconditioning_starting_alpha = 0.01 to 0.02")
print("   - stochastic_preconditioning_iterations = total_iterations // 2")
print("   - enable_upsampling = False")
print("\n3. STRONG stochastic preconditioning:")
print("   - stochastic_preconditioning_starting_alpha = 0.03 to 0.05")
print("   - stochastic_preconditioning_iterations = total_iterations")
print("   - enable_upsampling = False")
print("\nKey points:")
print("  - Noise is added to volume query points (not ray origins)")
print("  - Alpha decays exponentially from starting value to ~1e-16")
print("  - Use as alternative to multi-resolution training (disable upsampling)")
print("  - Noise scale is relative to grid cell size (alpha / grid_res)")
print("  - Different noise for each query point (stochastic)")
