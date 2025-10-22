"""
Training utilities for radiance field reconstruction.

This module contains reusable functions for training loops, optimization,
and parameter management for NeRF-like reconstruction pipelines.
"""

from config import ExperimentConfig
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
                 enable_upsampling: bool = True):
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


def compute_loss(predicted: mi.TensorXf, target: mi.TensorXf, loss_type: str = 'l1') -> mi.Float:
    """
    Compute loss between predicted and target images.
    
    Args:
        predicted: Predicted image tensor
        target: Target image tensor
        loss_type: Type of loss ('l1' or 'l2')
        
    Returns:
        Loss value
    """
    if loss_type == 'l1':
        return dr.mean(dr.abs(predicted - target), axis=None)
    elif loss_type == 'l2':
        return dr.mean(dr.square(predicted - target), axis=None)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


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


def train_stage(scene: mi.Scene, 
                sensors: List[mi.Sensor],
                ref_images: List[mi.TensorXf],
                params: Dict[str, Any],
                opt: mi.ad.Adam,
                config: TrainingConfig,
                stage_idx: int,
                progress_callback: Optional[Callable] = None) -> tuple[List[float], List[mi.TensorXf]]:
    """
    Train for one stage of the optimization.
    
    Args:
        scene: Mitsuba scene
        sensors: List of camera sensors
        ref_images: List of reference images
        params: Scene parameters
        opt: Optimizer
        config: Training configuration
        stage_idx: Current stage index
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (losses, final_images)
    """
    losses = []
    final_images = []
    
    if 'sigmat' in opt:
        print(f"Stage {stage_idx+1:02d}, feature voxel grids resolution -> {opt['sigmat'].shape[0]}")
    
    for it in range(config.num_iterations_per_stage):
        total_loss = 0.0
        images = []
        
        for sensor_idx in range(len(sensors)):
            img = mi.render(scene, params, sensor=sensors[sensor_idx], 
                          spp=config.spp, seed=it)
            loss = compute_loss(img, ref_images[sensor_idx], config.loss_type)
            # loss += dr.mean(dr.abs(dr.grad(params['sigmat']))) * 100.0

            dr.backward(loss)
            total_loss += loss
            
            # Store images at the end of every stage
            if it == config.num_iterations_per_stage - 1:
                dr.eval(img)
                images.append(img)

        losses.append(total_loss.array[0])
        opt.step()
        
        # Apply constraints if not using ReLU
        if not config.use_relu:
            if 'sigmat' in opt:
                opt['sigmat'] = dr.maximum(opt['sigmat'], 0.0)
        
        params.update(opt)
        
        if progress_callback:
            progress_callback(stage_idx, it, total_loss)
        else:
            print(f"  --> iteration {it+1:02d}: error={total_loss}", end='\r')
    
    final_images = images
    return losses, final_images


def train_radiance_field(scene: mi.Scene,
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
    params = mi.traverse(scene.integrator())

    # Determine which parameters to optimize based on integrator type
    opt_params = {'sh_coeffs': params['sh_coeffs']}

    if 'sigmat' in params:
        # RadianceFieldPRB or RadianceFieldPRBRT (density only)
        opt_params['sigmat'] = params['sigmat']

    opt = create_optimizer(opt_params, config.learning_rate)
    params.update(opt)
    
    all_losses = []
    intermediate_images = []
    
    for stage in range(config.num_stages):
        stage_losses, stage_images = train_stage(
            scene, sensors, ref_images, params, opt, config, stage, progress_callback
        )
        
        all_losses.extend(stage_losses)
        intermediate_images.append(stage_images)
        
        # Upsample parameters if enabled and not the last stage
        if config.enable_upsampling and stage < config.num_stages - 1:
            upsample_parameters(opt)
            params.update(opt)
    
    print('')
    print('Done')
    
    return {
        'losses': all_losses,
        'intermediate_images': intermediate_images,
        'final_params': params,
        'optimizer': opt
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
    
    all_losses = []
    intermediate_images = []
    
    for stage in range(config.num_stages):
        stage_losses, stage_images = train_stage(
            scene, sensors, ref_images, params, opt, config, stage, progress_callback
        )
        
        all_losses.extend(stage_losses)
        intermediate_images.append(stage_images)
        
        # Upsample parameters if enabled and not the last stage
        if config.enable_upsampling and stage < config.num_stages - 1:
            upsample_parameters(opt)
            params.update(opt)
    
    print('')
    print('Done')
    
    return {
        'losses': all_losses,
        'intermediate_images': intermediate_images,
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
                  center: List[float] = [0.5, 0.5, 0.5],
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
    
    Args:
        config: Training configuration
    """
    if config.scene_name == "lego":
        return mi.load_file('./scenes/lego/scene.xml')
    elif config.scene_name == "fog":
        scene_dict = {
            'type': 'scene',
            'integrator': {'type': 'prbvolpath', 'hide_emitters': config.hide_emitters},
            'object': {
                'type': 'cube',
                'bsdf': {'type': 'null'},
                'interior': {
                    'type': 'heterogeneous',
                    'sigma_t': {
                        'type': 'gridvolume',
                        'filename': 'scenes/volume.vol',
                        'to_world': mi.ScalarTransform4f().translate(0.5).scale(0.35).rotate([1, 0, 0], -90).scale(2).translate(-0.5)
                    },
                    'scale': 40
                },
                'to_world': mi.ScalarTransform4f().translate(0.5).scale(0.35),
            },
            'emitter': {'type': 'constant'}
        }

        return mi.load_dict(scene_dict)