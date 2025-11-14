# TLIR: Trainable Light and Image Rendering

A framework for differentiable volumetric rendering and radiance field reconstruction using Mitsuba 3.

## Project Structure

```
tlir/
├── tlir/                    # Main package directory
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration management (ExperimentConfig, RenderConfig)
│   ├── training.py         # Training loops and optimization
│   ├── camera_utils.py     # Camera generation and rendering utilities
│   ├── scene_registry.py   # Scene discovery and loading
│   ├── visualization.py    # Visualization utilities
│   ├── ray_batch.py        # Ray batching for training
│   ├── integrators/        # Custom Mitsuba integrators
│   ├── tests/              # Unit tests
│   │   ├── test_scene_registry.py
│   │   ├── test_vectorized_rendering.py
│   │   ├── test_render_config.py
│   │   └── test_refactored_training.py
│   └── examples/           # Example scripts
│       ├── example_camera_views.py
│       ├── example_noise_training.py
│       ├── example_ray_batching.py
│       └── example_stochastic_preconditioning.py
├── notebooks/              # Jupyter notebooks
│   ├── mitsuba_test.ipynb
│   └── mitsuba_test2.ipynb
├── scenes/                 # Scene definitions
│   ├── lego/
│   │   └── scene.xml
│   ├── fog/
│   │   ├── scene.xml
│   │   └── README.md
│   └── README.md
├── setup.py               # Package installation
└── README.md             # This file
```

## Installation

1. **Install Conda Environment** (if not already done):
   ```bash
   conda create -n tlir python=3.10
   conda activate tlir
   conda install -c conda-forge mitsuba drjit
   ```

2. **Install TLIR Package**:
   ```bash
   pip install -e .
   ```

This installs the `tlir` package in development mode, allowing you to import modules from anywhere.

## Usage

### Basic Import Pattern

```python
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')  # Set variant BEFORE importing tlir modules

from tlir import config as cf
from tlir import training
from tlir import camera_utils
from tlir import scene_registry
```

### Creating and Using Configurations

```python
# Basic configuration
config = cf.create_config("my_experiment", scene_name="fog")

# With custom render configuration
from tlir.config import RenderConfig

render_cfg = RenderConfig(
    camera_center=[0.0, 0.0, 0.0],
    camera_radius=2.0,
    up_vector=[0, 1, 0],  # Y-up
    num_train=200,
    num_test=150,
    render_res=512
)

config = cf.create_config(
    "high_quality_experiment",
    scene_name="fog",
    render_config=render_cfg
)
```

### Rendering with Cameras

```python
# Load scene
scene = training.create_scene_reference(config)

# Create and cache cameras/images using render_config
data = camera_utils.create_cached_cameras_and_images(
    scene=scene,
    cache_dir="./data/fog",
    render_config=config.render_config
)
```

## Key Features

### 1. RenderConfig
Centralized rendering configuration that includes:
- Camera setup (center, radius, up vector)
- Image counts (num_train, num_test)
- Rendering quality (resolution, FOV, SPP)
- Camera distribution parameters

### 2. Scene Registry
Automatic scene discovery from `scenes/` directory:
```python
from tlir.scene_registry import get_scene_registry

registry = get_scene_registry()
print(registry.list_scenes())  # ['lego', 'fog']
```

### 3. Vectorized Rendering
High-performance rendering functions that process all pixels simultaneously using DrJit.

### 4. Ray Batching
Efficient training mode that samples random rays instead of full images.

## Running Tests

```bash
# Run all tests
python -m pytest tlir/tests/

# Run specific test
python tlir/tests/test_scene_registry.py
```

## Running Examples

```bash
# Camera views example
python tlir/examples/example_camera_views.py

# Ray batching example
python tlir/examples/example_ray_batching.py

# Noise training example
python tlir/examples/example_noise_training.py
```

## Running Notebooks

```bash
jupyter notebook notebooks/mitsuba_test.ipynb
```

## Important Notes

1. **Always set Mitsuba variant first**: Call `mi.set_variant('cuda_ad_rgb')` before importing tlir modules
2. **Development mode**: The package is installed in editable mode (`pip install -e .`), so changes to source files are immediately reflected
3. **Up vector**: The `up_vector` parameter in RenderConfig ensures consistent camera orientation across all views

## Adding New Scenes

1. Create a directory under `scenes/`:
   ```bash
   mkdir scenes/my_scene
   ```

2. Create `scene.xml` in that directory:
   ```xml
   <?xml version="1.0" encoding="utf-8"?>
   <scene version="3.0.0">
       <integrator type="prbvolpath"/>
       <!-- Your scene content -->
   </scene>
   ```

3. The scene will be automatically discovered:
   ```python
   config = cf.create_config("test", scene_name="my_scene")
   ```

## License

[Add your license information here]
