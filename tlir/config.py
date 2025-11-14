"""
Configuration and parameter management utilities.

This module contains classes and functions for managing configuration
parameters, scene setup, and experiment tracking.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
import mitsuba as mi
from datetime import datetime


@dataclass
class RenderConfig:
    """Configuration for rendering and camera setup."""

    # Camera parameters
    camera_center: List[float] = None
    camera_radius: float = 1.3
    up_vector: List[float] = None  # Up direction for all cameras (default: [0, 1, 0] = Y-up)
    hemisphere_normal: List[float] = None  # Normal for hemisphere sampling (defaults to up_vector)

    # Image counts
    num_train: int = 100
    num_test: int = 100

    # Rendering quality
    render_res: int = 256
    fov: float = 45.0
    spp: int = 128  # Samples per pixel

    # Test camera trajectory
    test_min_elevation: float = 30.0
    test_max_elevation: float = 30.0
    test_num_loops: float = 1.0

    # Random seed
    seed: int = 42

    def __post_init__(self):
        """Initialize default values after object creation."""
        if self.camera_center is None:
            self.camera_center = [0.0, 0.0, 0.0]

        if self.up_vector is None:
            self.up_vector = [0.0, 1.0, 0.0]  # Default Y-up

        if self.hemisphere_normal is None:
            self.hemisphere_normal = self.up_vector.copy()


@dataclass
class ExperimentConfig:
    """Configuration class for radiance field experiments."""

    # Experiment metadata
    experiment_name: str = "radiance_field_experiment"
    timestamp: str = ""
    output_dir: str = "./outputs"
    scene_name : str = "lego"

    # Rendering configuration (subconfig)
    render_config: RenderConfig = None

    # Legacy rendering parameters (for backward compatibility)
    render_res: int = 256
    sensor_count: int = 7
    fov: float = 45.0
    camera_center: List[float] = None
    camera_radius: float = 1.3
    hide_emitters: bool = True
    
    # Training parameters
    num_stages: int = 4
    num_iterations_per_stage: int = 15
    learning_rate: float = 0.2
    grid_init_res: int = 16
    sh_degree: int = 2
    use_relu: bool = True
    max_initial_density: float = 5.0
    spp: int = 1
    loss_type: str = 'l2'
    enable_upsampling: bool = True
    gt_noise_std: float = 0.0  # Standard deviation of Gaussian noise added to ground truth images (0.0 = no noise)
    gt_noise_use_mask: bool = True  # If True, apply noise only to background (1 - mask); if False, apply uniformly

    # Ray-based training parameters
    use_ray_batching: bool = False  # If True, sample random rays instead of random images
    rays_per_batch: int = 4096  # Number of rays per training batch (only used if use_ray_batching=True)

    # Stochastic preconditioning parameters (alternative to multi-resolution training)
    stochastic_preconditioning_starting_alpha: float = 0.0  # Starting scale of noise added to query points (0.0 = disabled)
    stochastic_preconditioning_iterations: int = -1  # Number of iterations for preconditioning decay (-1 = disabled)
    
    # Integrator parameters
    integrator_type: str = 'rf_prb'
    initial_density: float = 0.01
    initial_sh: float = 0.1

    # Ratio tracking specific parameters (for rf_prb_rt)
    initial_majorant: float = 10.0
    stopgrad_density: bool = False
    min_step_size: float = 1e-4
    min_throughput: float = 1e-6
    max_num_steps: int = 10000

    # Eikonal integrator specific parameters (for rf_eikonal)
    initial_ior: float = 1.0
    min_ior: float = 1.0
    max_ior: float = 2.0
    step_size_scale: float = 0.5
    curvature_damping: float = 1.0
    eikonal_initial_majorant: float = 10.0
    eikonal_stopgrad_density: bool = False
    eikonal_min_throughput: float = 1e-6
    
    # Scene parameters
    scene_file: str = './scenes/lego/scene.xml'
    bbox_min: List[float] = None
    bbox_max: List[float] = None
    
    # Logging and visualization
    save_images: bool = True
    save_logs: bool = True
    create_gif: bool = False
    log_interval: int = 1
    
    def __post_init__(self):
        """Initialize default values after object creation."""
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.camera_center is None:
            self.camera_center = [0.0, 0.0, 0.0]

        if self.bbox_min is None:
            self.bbox_min = [0.0, 0.0, 0.0]

        if self.bbox_max is None:
            self.bbox_max = [1.0, 1.0, 1.0]

        # Initialize render_config if not provided (using legacy parameters)
        if self.render_config is None:
            self.render_config = RenderConfig(
                camera_center=self.camera_center,
                camera_radius=self.camera_radius,
                render_res=self.render_res,
                fov=self.fov
            )

        # Create output directory
        self.output_dir = os.path.join(self.output_dir, f"{self.experiment_name}_{self.timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, "config.json")
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            ExperimentConfig instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(**data)
    
    def get_integrator_params(self) -> Dict[str, Any]:
        """Get integrator-specific parameters."""
        base_params = {
            'bbox': mi.ScalarBoundingBox3f(self.bbox_min, self.bbox_max),
            'use_relu': self.use_relu,
            'grid_res': self.grid_init_res,
            'sh_degree': self.sh_degree,
            'initial_sh': self.initial_sh
        }

        if self.integrator_type == 'rf_prb':
            base_params.update({
                'initial_density': self.initial_density
            })
        elif self.integrator_type == 'rf_prb_rt':
            base_params.update({
                'initial_density': self.initial_density,
                'initial_majorant': self.initial_majorant,
                'stopgrad_density': self.stopgrad_density,
                'min_step_size': self.min_step_size,
                'min_throughput': self.min_throughput,
                'max_num_steps': self.max_num_steps
            })
        elif self.integrator_type == 'rf_eikonal':
            base_params.update({
                'initial_ior': self.initial_ior,
                'initial_density': self.initial_density,
                'min_ior': self.min_ior,
                'max_ior': self.max_ior,
                'step_size_scale': self.step_size_scale,
                'max_steps': self.max_num_steps,
                'curvature_damping': self.curvature_damping,
                'initial_majorant': self.eikonal_initial_majorant,
                'stopgrad_density': self.eikonal_stopgrad_density,
                'min_throughput': self.eikonal_min_throughput
            })

        return base_params
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get training-specific parameters."""
        return {
            'num_stages': self.num_stages,
            'num_iterations_per_stage': self.num_iterations_per_stage,
            'learning_rate': self.learning_rate,
            'grid_init_res': self.grid_init_res,
            'sh_degree': self.sh_degree,
            'use_relu': self.use_relu,
            'max_initial_density': self.max_initial_density,
            'spp': self.spp,
            'loss_type': self.loss_type,
            'enable_upsampling': self.enable_upsampling,
            'gt_noise_std': self.gt_noise_std,
            'stochastic_preconditioning_starting_alpha': self.stochastic_preconditioning_starting_alpha,
            'stochastic_preconditioning_iterations': self.stochastic_preconditioning_iterations
        }


def create_config(experiment_name: str = "experiment", **kwargs) -> ExperimentConfig:
    """
    Create a default configuration for radiance field experiments.
    
    Args:
        experiment_name: Name for the experiment
        
    Returns:
        ExperimentConfig instance with default values
    """
    return ExperimentConfig(experiment_name=experiment_name, **kwargs)

def create_default_config(experiment_name: str = "default_experiment") -> ExperimentConfig:
    """
    Create a default configuration for radiance field experiments.
    
    Args:
        experiment_name: Name for the experiment
        
    Returns:
        ExperimentConfig instance with default values
    """
    return ExperimentConfig(experiment_name=experiment_name)

def create_quick_config(experiment_name: str = "quick_test",
                       num_stages: int = 4,
                       num_iterations_per_stage: int = 5,
                       grid_init_res: int = 16) -> ExperimentConfig:
    """
    Create a configuration for quick testing.
    
    Args:
        experiment_name: Name for the experiment
        num_stages: Number of training stages
        num_iterations_per_stage: Number of iterations per stage
        grid_init_res: Initial grid resolution
        
    Returns:
        ExperimentConfig instance optimized for quick testing
    """
    return ExperimentConfig(
        experiment_name=experiment_name,
        num_stages=num_stages,
        num_iterations_per_stage=num_iterations_per_stage,
        grid_init_res=grid_init_res,
        render_res=128,  # Lower resolution for speed
        sensor_count=3,  # Fewer sensors for speed
        spp=1
    )


def create_high_quality_config(experiment_name: str = "high_quality",
                              num_stages: int = 5,
                              num_iterations_per_stage: int = 30,
                              grid_init_res: int = 32) -> ExperimentConfig:
    """
    Create a configuration for high-quality results.
    
    Args:
        experiment_name: Name for the experiment
        num_stages: Number of training stages
        num_iterations_per_stage: Number of iterations per stage
        grid_init_res: Initial grid resolution
        
    Returns:
        ExperimentConfig instance optimized for high quality
    """
    return ExperimentConfig(
        experiment_name=experiment_name,
        num_stages=num_stages,
        num_iterations_per_stage=num_iterations_per_stage,
        grid_init_res=grid_init_res,
        render_res=512,  # Higher resolution
        sensor_count=12,  # More sensors for better coverage
        spp=4,  # More samples per pixel
        learning_rate=0.1  # Lower learning rate for stability
    )


def create_ratio_tracking_config(experiment_name: str = "ratio_tracking",
                                initial_majorant: float = 10.0,
                                stopgrad_density: bool = False) -> ExperimentConfig:
    """
    Create a configuration specifically for ratio tracking experiments.

    Args:
        experiment_name: Name for the experiment
        initial_majorant: Initial majorant value
        stopgrad_density: Whether to stop gradients on density

    Returns:
        ExperimentConfig instance for ratio tracking
    """
    return ExperimentConfig(
        experiment_name=experiment_name,
        integrator_type='rf_prb_rt',
        initial_majorant=initial_majorant,
        stopgrad_density=stopgrad_density,
        min_step_size=1e-4,
        min_throughput=1e-6,
        max_num_steps=10000
    )


def create_eikonal_config(experiment_name: str = "eikonal",
                         initial_ior: float = 1.0,
                         initial_density: float = 0.01,
                         min_ior: float = 1.0,
                         max_ior: float = 2.0,
                         step_size_scale: float = 0.5,
                         curvature_damping: float = 1.0,
                         initial_majorant: float = 10.0,
                         stopgrad_density: bool = False) -> ExperimentConfig:
    """
    Create a configuration specifically for eikonal raymarching experiments.

    Args:
        experiment_name: Name for the experiment
        initial_ior: Initial index of refraction value
        initial_density: Initial density value for scattering
        min_ior: Minimum IOR (typically 1.0 for vacuum)
        max_ior: Maximum IOR (typical materials: 1.3-2.5)
        step_size_scale: Step size as fraction of grid resolution
        curvature_damping: Damping factor for ray curvature
        initial_majorant: Initial majorant for null-collision sampling
        stopgrad_density: Whether to stop gradients on density

    Returns:
        ExperimentConfig instance for eikonal raymarching
    """
    return ExperimentConfig(
        experiment_name=experiment_name,
        integrator_type='rf_eikonal',
        initial_ior=initial_ior,
        initial_density=initial_density,
        min_ior=min_ior,
        max_ior=max_ior,
        step_size_scale=step_size_scale,
        curvature_damping=curvature_damping,
        max_num_steps=1000,
        eikonal_initial_majorant=initial_majorant,
        eikonal_stopgrad_density=stopgrad_density,
        eikonal_min_throughput=1e-6
    )


class ConfigManager:
    """Manager class for handling multiple experiment configurations."""
    
    def __init__(self, base_dir: str = "./experiments"):
        """
        Initialize the configuration manager.
        
        Args:
            base_dir: Base directory for storing experiment configurations
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def save_experiment(self, config: ExperimentConfig) -> str:
        """
        Save an experiment configuration.
        
        Args:
            config: Experiment configuration to save
            
        Returns:
            Path to saved configuration
        """
        config_dir = os.path.join(self.base_dir, f"{config.experiment_name}_{config.timestamp}")
        os.makedirs(config_dir, exist_ok=True)
        
        config_path = os.path.join(config_dir, "config.json")
        config.save(config_path)
        
        return config_path
    
    def list_experiments(self) -> List[str]:
        """
        List all available experiment configurations.
        
        Returns:
            List of experiment directory names
        """
        if not os.path.exists(self.base_dir):
            return []
        
        experiments = []
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path):
                config_path = os.path.join(item_path, "config.json")
                if os.path.exists(config_path):
                    experiments.append(item)
        
        return sorted(experiments)
    
    def load_experiment(self, experiment_name: str) -> ExperimentConfig:
        """
        Load an experiment configuration by name.
        
        Args:
            experiment_name: Name of the experiment to load
            
        Returns:
            ExperimentConfig instance
        """
        config_path = os.path.join(self.base_dir, experiment_name, "config.json")
        return ExperimentConfig.load(config_path)
    
    def compare_configs(self, configs: List[ExperimentConfig]) -> None:
        """
        Compare multiple configurations and print differences.
        
        Args:
            configs: List of configurations to compare
        """
        if len(configs) < 2:
            print("Need at least 2 configurations to compare.")
            return
        
        print("Configuration Comparison:")
        print("=" * 50)
        
        # Get all unique keys
        all_keys = set()
        for config in configs:
            all_keys.update(config.to_dict().keys())
        
        for key in sorted(all_keys):
            values = [config.to_dict().get(key, "N/A") for config in configs]
            if len(set(str(v) for v in values)) > 1:  # If values differ
                print(f"{key}:")
                for i, (config, value) in enumerate(zip(configs, values)):
                    print(f"  {config.experiment_name}: {value}")
                print()


def setup_environment(config: ExperimentConfig) -> None:
    """
    Set up the environment based on configuration.
    
    Args:
        config: Experiment configuration
    """
    # Set CUDA device if specified
    if hasattr(config, 'cuda_device') and config.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_device)
    
    # Set Mitsuba variant
    mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    
    if config.save_images:
        os.makedirs(os.path.join(config.output_dir, "images"), exist_ok=True)
    
    if config.save_logs:
        os.makedirs(os.path.join(config.output_dir, "logs"), exist_ok=True)


def validate_config(config: ExperimentConfig) -> List[str]:
    """
    Validate a configuration and return any issues found.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []

    valid_scene_names = ['lego', 'fog']
    if config.scene_name not in valid_scene_names:
        issues.append(f"scene_name must be one of {valid_scene_names}")
    
    # Check basic parameter ranges
    if config.render_res <= 0:
        issues.append("render_res must be positive")
    
    if config.sensor_count <= 0:
        issues.append("sensor_count must be positive")
    
    if config.learning_rate <= 0:
        issues.append("learning_rate must be positive")
    
    if config.grid_init_res <= 0:
        issues.append("grid_init_res must be positive")
    
    if config.sh_degree < 0:
        issues.append("sh_degree must be non-negative")
    
    if config.num_stages <= 0:
        issues.append("num_stages must be positive")
    
    if config.num_iterations_per_stage <= 0:
        issues.append("num_iterations_per_stage must be positive")
    
    # Check integrator type
    valid_integrators = ['prb_volpath', 'rf_prb', 'rf_prb_rt', 'rf_prb_drt', 'rf_eikonal']
    if config.integrator_type not in valid_integrators:
        issues.append(f"integrator_type must be one of {valid_integrators}")
    
    # Check loss type
    valid_losses = ['l1', 'l2']
    if config.loss_type not in valid_losses:
        issues.append(f"loss_type must be one of {valid_losses}")
    
    # Check scene file exists
    if not os.path.exists(config.scene_file):
        issues.append(f"scene_file does not exist: {config.scene_file}")
    
    return issues
