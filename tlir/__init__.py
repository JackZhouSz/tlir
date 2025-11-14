"""
TLIR: Trainable Light and Image Rendering

A framework for differentiable volumetric rendering and radiance field reconstruction.
"""

__version__ = "0.1.0"

# Core modules are available via explicit imports:
# from tlir import config
# from tlir import training
# from tlir import camera_utils
# from tlir import visualization
# from tlir import ray_batch
# from tlir import scene_registry

__all__ = [
    'config',
    'training',
    'camera_utils',
    'visualization',
    'ray_batch',
    'scene_registry',
]
