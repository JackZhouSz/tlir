"""
Training utilities for radiance field reconstruction.

This module contains reusable functions for training loops, optimization,
and parameter management for NeRF-like reconstruction pipelines.
"""

from __future__ import annotations

from tlir.config import ExperimentConfig
import drjit as dr
import mitsuba as mi
from typing import List, Dict, Any, Optional, Callable
import numpy as np


class TrainingConfig:
    """Configuration class for training parameters."""

    def __init__(self,
                 num_stages: int = 4,
                 num_iterations_per_stage: int = 15,
                 learning_rate: float = 0.2,
                 grid_init_res: int = 16,
                 sh_degree: int = 2,
                 use_relu: bool = True,
                 max_initial_density: float = 5.0,
                 spp: int = 1,
                 loss_type: str = 'l1',
                 enable_upsampling: bool = True,
                 use_ray_batching: bool = False,
                 rays_per_batch: int = 4096,
                 num_test_images: int = 1,
                 stochastic_preconditioning_starting_alpha: float = 0.0,
                 stochastic_preconditioning_iterations: int = -1,
                 opacity_loss_weight: float = 0.0,
                 empty_space_loss_weight: float = 0.0):
        """
        Initialize training configuration.

        Args:
            num_stages: Number of training stages
            num_iterations_per_stage: Number of iterations per stage
            learning_rate: Learning rate for optimizer
            grid_init_res: Initial grid resolution
            sh_degree: Spherical harmonics degree
            use_relu: Whether to use ReLU activation
            max_initial_density: Maximum initial density value
            spp: Samples per pixel for rendering
            loss_type: Type of loss function ('l1' or 'l2')
            enable_upsampling: Whether to enable grid upsampling
            use_ray_batching: If True, sample random rays instead of random images
            rays_per_batch: Number of rays per training batch (only used if use_ray_batching=True)
            num_test_images: Number of test images to render at end of each stage for PSNR evaluation
            stochastic_preconditioning_starting_alpha: Starting scale of noise for query points
            stochastic_preconditioning_iterations: Number of iterations for preconditioning decay
            opacity_loss_weight: Weight for opacity loss (throughput=1 where object_mask=1)
            empty_space_loss_weight: Weight for empty space loss (throughput=0 where object_mask=0)
        """
        self.num_stages = num_stages
        self.num_iterations_per_stage = num_iterations_per_stage
        self.learning_rate = learning_rate
        self.grid_init_res = grid_init_res
        self.sh_degree = sh_degree
        self.use_relu = use_relu
        self.max_initial_density = max_initial_density
        self.spp = spp
        self.loss_type = loss_type
        self.enable_upsampling = enable_upsampling
        self.use_ray_batching = use_ray_batching
        self.rays_per_batch = rays_per_batch
        self.num_test_images = num_test_images
        self.stochastic_preconditioning_starting_alpha = stochastic_preconditioning_starting_alpha
        self.stochastic_preconditioning_iterations = stochastic_preconditioning_iterations
        self.opacity_loss_weight = opacity_loss_weight
        self.empty_space_loss_weight = empty_space_loss_weight


def create_optimizer(params: Dict[str, Any], learning_rate: float) -> mi.ad.Adam:
    """
    Create an Adam optimizer for the given parameters.
    
    Args:
        params: Dictionary of parameters to optimize
        learning_rate: Learning rate for the optimizer
        
    Returns:
        Adam optimizer instance
    """
    return mi.ad.Adam(lr=learning_rate, params=params)


def upsample_parameters(opt: mi.ad.Adam, factor: int = 2) -> None:
    """
    Upsample the 3D texture parameters by the given factor.

    Args:
        opt: Optimizer containing the parameters
        factor: Upsampling factor (default: 2)
    """
    if 'sigmat' in opt:
        # RadianceFieldPRB or RadianceFieldPRBRT (density only)
        new_res = factor * opt['sigmat'].shape[0]
        new_shape = [new_res, new_res, new_res]
        opt['sigmat'] = dr.upsample(opt['sigmat'], new_shape)
        opt['sh_coeffs'] = dr.upsample(opt['sh_coeffs'], new_shape)

        # Also upsample majorant grid if it exists (ratio tracking)
        if 'majorant_grid' in opt:
            opt['majorant_grid'] = dr.upsample(opt['majorant_grid'], new_shape)


def sample_rays_from_image(
    sensor: mi.Sensor,
    ref_image: mi.TensorXf,
    ref_mask: Optional[mi.TensorXf] = None
) -> tuple:
    """
    Extract all rays from a single image/sensor.

    Args:
        sensor: Camera sensor
        ref_image: Reference image
        ref_mask: Optional reference mask

    Returns:
        Tuple of (rays_mi, target_colors, masks_mi, num_rays)
    """
    # Get image dimensions
    film = sensor.film()
    res = film.size()
    width, height = res[0], res[1]
    num_rays = width * height

    # Extract RGB channels only (drop alpha if present) and flatten
    # ref_image shape: (height, width, channels) where channels is 3 or 4
    # We need shape: (3, num_rays) to match integrator output

    # Convert to numpy for easier manipulation
    ref_image_np = np.array(ref_image)
    num_channels = ref_image_np.shape[2]

    if num_channels == 4:
        # RGBA - extract only RGB channels
        ref_image_rgb = ref_image_np[..., :3]
    elif num_channels == 3:
        # Already RGB
        ref_image_rgb = ref_image_np
    else:
        raise ValueError(f"Expected 3 or 4 channels, got {num_channels}")

    # Reshape to (num_rays, 3) - keep as channels-last for now
    colors_flat = ref_image_rgb.reshape(num_rays, 3)  # Shape: (num_rays, 3)

    # Convert to Mitsuba Color3f (Spectrum) to match integrator output type
    # Create Color3f array from the flattened colors
    target_colors = mi.Color3f(
        mi.Float(colors_flat[:, 0]),  # R channel
        mi.Float(colors_flat[:, 1]),  # G channel
        mi.Float(colors_flat[:, 2])   # B channel
    )

    # Flatten mask if provided
    masks_mi = None
    if ref_mask is not None:
        # Mask should be single channel, flatten to (num_rays,)
        mask_shape = ref_mask.shape
        mask_channels = mask_shape[2] if len(mask_shape) > 2 else 1

        if mask_channels > 1:
            # If multi-channel, take first channel
            ref_mask_single = ref_mask[..., 0:1]
        else:
            ref_mask_single = ref_mask

        masks_mi = dr.ravel(ref_mask_single)  # Shape: (num_rays,)

    # Sample rays from sensor for all pixels
    # Create pixel positions
    idx = dr.arange(mi.UInt32, num_rays)
    x = idx % width
    y = idx // width

    # Convert to normalized coordinates [0, 1]
    pos_x = (x + 0.5) / width
    pos_y = (y + 0.5) / height
    pos_sample = mi.Point2f(pos_x, pos_y)

    # Sample rays
    wavelength_sample = 0.5
    time_sample = 0.0
    aperture_sample = mi.Point2f(0.5, 0.5)

    rays, _ = sensor.sample_ray(time_sample, wavelength_sample, pos_sample, aperture_sample)

    return rays, target_colors, masks_mi, num_rays


def train_stage(scene: mi.Scene,
                sensors: List[mi.Sensor],
                ref_images: List[mi.TensorXf],
                config: TrainingConfig,
                stage_idx: int,
                progress_callback: Optional[Callable] = None,
                ref_masks: Optional[List[mi.TensorXf]] = None,
                ray_batch: Optional['RayBatch'] = None) -> tuple[List[float], List[mi.TensorXf], List[float]]:
    """
    Train for one stage of the optimization.

    Supports both image-based and ray-based training modes:
    - Image-based: Samples all rays from one random image per iteration
    - Ray-based: Samples random rays from all images per iteration

    Args:
        scene: Mitsuba scene
        sensors: List of camera sensors
        ref_images: List of reference images
        config: Training configuration
        stage_idx: Current stage index
        progress_callback: Optional callback for progress updates
        ref_masks: Optional list of object masks (for image-based training)
        ray_batch: Optional RayBatch (for ray-based training)

    Returns:
        Tuple of (losses, test_images, test_psnrs)
    """
    import numpy as np
    from tlir.metrics import compute_psnr

    losses = []
    test_images = []
    test_psnrs = []

    # Get integrator and configure
    integrator = scene.integrator()
    integrator.loss_type = config.loss_type

    if 'sigmat' in integrator.optimizer:
        print(f"Stage {stage_idx+1:02d}, feature voxel grids resolution -> {integrator.optimizer['sigmat'].shape[0]}")

    # Setup RNG
    rng = np.random.default_rng(seed=stage_idx * 1000)

    # Setup for ray batching if enabled
    shuffled_rays = None
    if config.use_ray_batching:
        shuffled_rays = ray_batch.shuffle(rng)

    for it in range(config.num_iterations_per_stage):
        # ========== RAY SAMPLING (ONLY DIFFERENCE BETWEEN MODES) ==========
        if config.use_ray_batching:
            # Ray-based: Sample random rays from all images
            sampled_rays = shuffled_rays.sample(config.rays_per_batch, rng)
            rays_mi, target_colors = sampled_rays.to_mitsuba()
            masks_mi = mi.TensorXf(sampled_rays.masks) if sampled_rays.masks is not None else None
            num_rays = sampled_rays.num_rays
        else:
            # Image-based: Sample all rays from one random image
            sensor_idx = rng.integers(0, len(sensors))
            ref_image = ref_images[sensor_idx]
            ref_mask = ref_masks[sensor_idx] if ref_masks is not None else None
            rays_mi, target_colors, masks_mi, num_rays = sample_rays_from_image(
                sensors[sensor_idx], ref_image, ref_mask
            )

        # ========== COMMON RENDERING AND TRAINING (SAME FOR BOTH MODES) ==========
        # Get current spn_alpha from integrator
        spn_alpha = integrator.get_spn_alpha()

        # Set masks for AOV loss (if available)
        if masks_mi is not None:
            integrator.set_aov_loss_config(masks=dr.ravel(masks_mi))

        # Render rays using integrator
        sampler = mi.load_dict({'type': 'independent'})
        sampler.seed(it, num_rays)

        # Render with automatic gradient computation using integrator loss methods
        # Loss functions are defined as methods on the integrator class
        # Gradients are accumulated automatically - DO NOT call dr.backward(loss)
        rendered_colors, loss, aovs = integrator.render_rays_with_gradient(
            rays=rays_mi,
            target_colors=target_colors,
            sampler=sampler,
            scene=scene,
            spp=config.spp,
            spn_alpha=spn_alpha
        )

        # Store loss value
        losses.append(loss.array[0])

        # Optimizer step and post-step updates
        integrator.step_optimizer()
        integrator.post_step_update(config)

        # Progress reporting
        if progress_callback:
            progress_callback(stage_idx, it, losses[-1])
        else:
            if config.use_ray_batching:
                print(f"  --> iteration {it+1:02d}: error={losses[-1]:.6f}, spn_alpha={spn_alpha:.6f}, rays={config.rays_per_batch}", end='\r')
            else:
                print(f"  --> iteration {it+1:02d}: error={losses[-1]:.6f}, spn_alpha={spn_alpha:.6f}, rays={num_rays}", end='\r')

    # Render test images and compute PSNR at end of stage
    if config.num_test_images > 0:
        print()  # New line after training iterations
        num_test_to_render = min(config.num_test_images, len(sensors))

        # Select random test images
        test_indices = rng.choice(len(sensors), size=num_test_to_render, replace=False)

        for i, test_idx in enumerate(test_indices):
            test_sensor = sensors[test_idx]
            test_ref = ref_images[test_idx]

            # Render test image with current scene state
            rendered_img = mi.render(scene, sensor=test_sensor, spp=config.spp)

            # Compute PSNR
            psnr = compute_psnr(rendered_img, test_ref)

            test_images.append(rendered_img)
            test_psnrs.append(psnr)

            print(f"  Test image {i+1}/{num_test_to_render} (sensor {test_idx}): PSNR = {psnr:.2f} dB")

    return losses, test_images, test_psnrs


def setup_training(
    scene: mi.Scene,
    config: TrainingConfig
) -> None:
    """
    Common setup for training: initialize integrator optimizer and SPN.

    Args:
        scene: Mitsuba scene
        config: Training configuration
    """
    integrator = scene.integrator()
    integrator.initialize_optimizer(learning_rate=config.learning_rate)
    integrator.setup_stochastic_preconditioning(
        starting_alpha=config.stochastic_preconditioning_starting_alpha,
        num_iterations=config.stochastic_preconditioning_iterations
    )


def setup_stochastic_preconditioning(config: TrainingConfig) -> Dict[str, Any]:
    """
    Initialize stochastic preconditioning state.

    Args:
        config: Training configuration

    Returns:
        Dictionary with SPN state (alpha, gamma, step)
    """
    spn_state = {
        'alpha': config.stochastic_preconditioning_starting_alpha,
        'step': 0
    }

    # Calculate gamma for exponential decay
    if config.stochastic_preconditioning_iterations > 0:
        spn_state['gamma'] = (1e-16 / config.stochastic_preconditioning_starting_alpha) ** (
            1.0 / config.stochastic_preconditioning_iterations
        )
    else:
        spn_state['gamma'] = 0.0
        spn_state['alpha'] = 0.0

    return spn_state


def convert_masks_to_mitsuba(ref_masks: Optional[List[np.ndarray]]) -> Optional[List[mi.TensorXf]]:
    """
    Convert numpy masks to Mitsuba tensors.

    Args:
        ref_masks: List of numpy mask arrays or None

    Returns:
        List of Mitsuba tensors or None
    """
    if ref_masks is None:
        return None

    masks_mi = []
    for mask in ref_masks:
        # Ensure mask has correct shape (H, W, 1) or (H, W, 3)
        if mask.ndim == 2:
            mask = mask[:, :, np.newaxis]
        # Convert to Mitsuba and potentially broadcast to 3 channels
        mask_mi = mi.TensorXf(mask)
        masks_mi.append(mask_mi)

    return masks_mi


def train_radiance_field(scene: mi.Scene,
                        sensors: List[mi.Sensor],
                        ref_images: List[mi.TensorXf],
                        config: TrainingConfig,
                        ref_masks: Optional[List[np.ndarray]] = None,
                        progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Complete training loop for radiance field reconstruction.

    Automatically selects between image-based and ray-based training based on config.use_ray_batching.
    - Image-based (default): Iterates over images, renders full images
    - Ray-based: Samples random rays from all images for each iteration

    Args:
        scene: Mitsuba scene (trainable)
        sensors: List of camera sensors
        ref_images: List of reference images
        config: Training configuration
        ref_masks: Optional list of object masks (loaded from dataset, not rendered during training)
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary containing training results including losses, test images, and test PSNRs
    """
    # Setup training
    setup_training(scene, config)

    # Setup for ray-based or image-based training
    ray_batch = None
    ref_masks_mi = None

    if config.use_ray_batching:
        # Ray-based training: extract all rays upfront
        from ray_batch import RayBatch, extract_rays_from_sensors

        print("=" * 70)
        print("Ray-based training mode")
        print("=" * 70)

        print(f"\nExtracting rays from {len(sensors)} training images...")
        ray_batch = extract_rays_from_sensors(sensors, ref_images, ref_masks)
        print(f"✓ Extracted {len(ray_batch)} total rays")
        print(f"✓ Ray batch size per iteration: {config.rays_per_batch}")
        print(f"✓ Total iterations per stage: {config.num_iterations_per_stage}")
        print()
    else:
        # Image-based training: convert masks to Mitsuba tensors if provided
        ref_masks_mi = convert_masks_to_mitsuba(ref_masks)

    all_losses = []
    all_test_images = []
    all_test_psnrs = []

    integrator = scene.integrator()

    for stage in range(config.num_stages):
        stage_losses, test_images, test_psnrs = train_stage(
            scene, sensors, ref_images, config, stage,
            progress_callback, ref_masks_mi, ray_batch
        )

        all_losses.extend(stage_losses)
        all_test_images.append(test_images)
        all_test_psnrs.append(test_psnrs)

        # Upsample parameters if enabled and not the last stage
        if config.enable_upsampling and stage < config.num_stages - 1:
            integrator.upsample_parameters()

    print('')
    print('Done')

    return {
        'losses': all_losses,
        'test_images': all_test_images,
        'test_psnrs': all_test_psnrs
    }


def train_prb_volpath(scene: mi.Scene,
                        sensors: List[mi.Sensor],
                        ref_images: List[mi.TensorXf],
                        config: TrainingConfig,
                        progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Complete training loop for radiance field reconstruction.

    Args:
        scene: Mitsuba scene
        sensors: List of camera sensors
        ref_images: List of reference images
        config: Training configuration
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary containing training results
    """
    # Get scene parameters and create optimizer
    params = mi.traverse(scene)

    # Determine which parameters to optimize based on integrator type
    key_sigmat = 'object.interior_medium.sigma_t.data'
    key_albedo = 'object.interior_medium.albedo.data'
    opt_params = {key_sigmat: params[key_sigmat]}

    if 'sigmat' in params:
        # RadianceFieldPRB or RadianceFieldPRBRT (density only)
        opt_params['sigmat'] = params['sigmat']

    opt = create_optimizer(opt_params, config.learning_rate)
    params.update(opt)

    # Initialize stochastic preconditioning state
    spn_state = {
        'alpha': config.stochastic_preconditioning_starting_alpha,
        'step': 0
    }

    # Calculate gamma for exponential decay
    if config.stochastic_preconditioning_iterations > 0:
        spn_state['gamma'] = (1e-16 / config.stochastic_preconditioning_starting_alpha) ** (
            1.0 / config.stochastic_preconditioning_iterations
        )
    else:
        spn_state['gamma'] = 0.0
        spn_state['alpha'] = 0.0

    all_losses = []
    all_test_images = []
    all_test_psnrs = []

    for stage in range(config.num_stages):
        stage_losses, test_images, test_psnrs = train_stage(
            scene, sensors, ref_images, params, opt, config, stage, spn_state, progress_callback
        )

        all_losses.extend(stage_losses)
        all_test_images.append(test_images)
        all_test_psnrs.append(test_psnrs)

        # Upsample parameters if enabled and not the last stage
        if config.enable_upsampling and stage < config.num_stages - 1:
            upsample_parameters(opt)
            params.update(opt)

    print('')
    print('Done')

    return {
        'losses': all_losses,
        'test_images': all_test_images,
        'test_psnrs': all_test_psnrs,
        'final_params': params,
        'optimizer': opt
    }


def copy_parameters(source_params: Dict[str, Any], 
                   target_params: Dict[str, Any],
                   max_density: Optional[float] = None) -> None:
    """
    Copy parameters from source to target, optionally clamping density values.
    
    Args:
        source_params: Source parameter dictionary
        target_params: Target parameter dictionary
        max_density: Optional maximum density value for clamping
    """
    if max_density is not None:
        target_params['sigmat'] = dr.minimum(source_params['sigmat'], max_density)
    else:
        target_params['sigmat'] = source_params['sigmat']
    
    target_params['sh_coeffs'] = source_params['sh_coeffs']
    target_params.update()


def create_sensors(num_sensors: int, 
                  render_res: int = 256,
                  fov: float = 45.0,
                  center: List[float] = [0.0, 0.0, 0.0],
                  radius: float = 1.3) -> List[mi.Sensor]:
    """
    Create multiple camera sensors arranged in a circle.
    
    Args:
        num_sensors: Number of sensors to create
        render_res: Rendering resolution
        fov: Field of view in degrees
        center: Center point for camera arrangement
        radius: Radius of camera circle
        
    Returns:
        List of camera sensors
    """
    sensors = []
    
    for i in range(num_sensors):
        angle = 360.0 / num_sensors * i
        sensors.append(mi.load_dict({
            'type': 'perspective', 
            'fov': fov, 
            'to_world': mi.ScalarTransform4f().translate(center) \
                                        .rotate([0, 1, 0], angle)   \
                                        .look_at(target=[0, 0, 0], 
                                                 origin=[0, 0, radius], 
                                                 up=[0, 1, 0]),
            'film': {
                'type': 'hdrfilm', 
                'width': render_res, 
                'height': render_res, 
                'filter': {'type': 'box'}, 
                'pixel_format': 'rgba'
            }
        }))
    
    return sensors


def create_scene(integrator_type: str = 'rf_prb') -> mi.Scene:
    """
    Create a simple scene with the specified integrator.
    
    Args:
        integrator_type: Type of integrator to use
        
    Returns:
        Mitsuba scene
    """
    if integrator_type == 'prb_volpath':
        scene_dict = {
            'type': 'scene',
            'integrator': {'type': 'prbvolpath', 'hide_emitters': True},
            'object': {
                'type': 'cube',
                'bsdf': {'type': 'null'},
                'interior': {
                    'type': 'heterogeneous',
                    'sigma_t': {
                        'type': 'gridvolume',
                        'grid': mi.VolumeGrid(dr.full(mi.TensorXf, 0.002, (16, 16, 16, 1))),
                        # 'to_world': mi.ScalarTransform4f().translate(0.5).scale(0.35).rotate([1, 0, 0], -90).scale(2).translate(-0.5)
                    },
                    # 'albedo': {
                    #     'type': 'gridvolume',
                    #     'grid': mi.VolumeGrid(dr.full(mi.TensorXf, 0.002, (16, 16, 16, 3))),
                    #     # 'to_world': mi.ScalarTransform4f().translate(0.5).scale(0.35).rotate([1, 0, 0], -90).scale(2).translate(-0.5)
                    # },
                    # 'scale': 40
                },
                'to_world': mi.ScalarTransform4f().translate(0.5).scale(0.35),
            },
            'emitter': {'type': 'constant'}
        }
        return mi.load_dict(scene_dict)
    elif integrator_type in ['rf_prb', 'rf_prb_rt', 'rf_prb_drt', 'rf_eikonal']:
        return mi.load_dict({
            'type': 'scene', 
            'integrator': {
                'type': integrator_type
            }, 
            'emitter': {
                'type': 'constant'
            }
        })
    else:
        raise ValueError(f"Unknown scene for integrator type: {integrator_type}")


def create_scene_reference(config: ExperimentConfig) -> mi.Scene:
    """
    Create a reference scene for generating ground truth images.

    Uses the scene registry to automatically load scenes from the scenes/ directory.
    Scenes are discovered by looking for scene.xml files in subdirectories.

    Args:
        config: Experiment configuration containing scene_name

    Returns:
        Loaded Mitsuba scene

    Raises:
        ValueError: If scene name is not found in registry
    """
    from tlir.scene_registry import get_scene_registry

    # Get the global scene registry
    registry = get_scene_registry()

    # Check if scene exists
    if config.scene_name not in registry:
        available = ", ".join(registry.list_scenes())
        raise ValueError(
            f"Scene '{config.scene_name}' not found. "
            f"Available scenes: {available}"
        )

    # Load scene from registry
    scene_path = registry[config.scene_name]
    scene = mi.load_file(scene_path)

    # Apply config-specific modifications if needed
    # For example, fog scene might need hide_emitters parameter
    if hasattr(config, 'hide_emitters') and config.scene_name == "fog":
        # Modify integrator parameters if needed
        # Note: This is scene-specific and could be extended to a plugin system
        params = mi.traverse(scene)
        if 'integrator.hide_emitters' in params:
            params['integrator.hide_emitters'] = config.hide_emitters
            params.update()

    return scene