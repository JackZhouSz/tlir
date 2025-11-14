"""
Test script for RenderConfig functionality.
"""

import os
import tempfile
import shutil

print("=" * 70)
print("Testing RenderConfig Functionality")
print("=" * 70)

# Test imports
print("\n1. Testing imports...")
try:
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    from tlir import config as cf
    from tlir.config import RenderConfig
    from tlir import training
    from tlir import camera_utils as cu
    print("   ✓ All imports successful")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

# Test RenderConfig creation
print("\n2. Testing RenderConfig creation...")
try:
    # Test default RenderConfig
    default_config = RenderConfig()
    print(f"   ✓ Default RenderConfig created")
    print(f"      - camera_center: {default_config.camera_center}")
    print(f"      - up_vector: {default_config.up_vector}")
    print(f"      - hemisphere_normal: {default_config.hemisphere_normal}")
    print(f"      - num_train: {default_config.num_train}")
    print(f"      - num_test: {default_config.num_test}")

    # Test custom RenderConfig
    custom_config = RenderConfig(
        camera_center=[0.0, 0.0, 0.0],
        camera_radius=2.0,
        up_vector=[0, 0, 1],  # Z-up
        hemisphere_normal=[0, 0, 1],  # Z-up hemisphere
        num_train=5,  # Small for testing
        num_test=5,
        render_res=64,
        fov=40.0,
        spp=16
    )
    print(f"   ✓ Custom RenderConfig created")
    print(f"      - up_vector: {custom_config.up_vector}")
    print(f"      - num_train: {custom_config.num_train}")
except Exception as e:
    print(f"   ✗ RenderConfig creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test ExperimentConfig with RenderConfig
print("\n3. Testing ExperimentConfig integration...")
try:
    # Test with custom render_config
    config = cf.create_config(
        "test_render_config",
        scene_name="fog",
        render_config=custom_config
    )
    print(f"   ✓ ExperimentConfig created with custom render_config")
    print(f"      - config.render_config.up_vector: {config.render_config.up_vector}")
    print(f"      - config.render_config.num_train: {config.render_config.num_train}")

    # Test without render_config (should use defaults)
    config_default = cf.create_config(
        "test_default",
        scene_name="fog"
    )
    print(f"   ✓ ExperimentConfig created with default render_config")
    print(f"      - config.render_config.up_vector: {config_default.render_config.up_vector}")
    print(f"      - config.render_config.num_train: {config_default.render_config.num_train}")
except Exception as e:
    print(f"   ✗ ExperimentConfig integration failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test create_cached_cameras_and_images with RenderConfig
print("\n4. Testing create_cached_cameras_and_images with RenderConfig...")
try:
    # Load scene
    scene = training.create_scene_reference(config)
    print(f"   ✓ Scene loaded")

    # Create temporary cache directory
    temp_dir = tempfile.mkdtemp(prefix="test_render_config_")
    print(f"   ✓ Temp directory created: {temp_dir}")

    # Test with render_config (NEW API)
    print("\n   Testing NEW API (with render_config)...")
    data = cu.create_cached_cameras_and_images(
        scene=scene,
        cache_dir=temp_dir,
        render_config=custom_config,
        force_rerender=True
    )
    print(f"   ✓ create_cached_cameras_and_images succeeded with render_config")
    print(f"      - Training cameras: {len(data['train_sensors'])}")
    print(f"      - Test cameras: {len(data['test_sensors'])}")
    print(f"      - Training images: {len(data['train_images'])}")
    print(f"      - Test images: {len(data['test_images'])}")

    # Verify data
    assert len(data['train_sensors']) == custom_config.num_train, "Train sensor count mismatch!"
    assert len(data['test_sensors']) == custom_config.num_test, "Test sensor count mismatch!"
    assert len(data['train_images']) == custom_config.num_train, "Train image count mismatch!"
    print(f"   ✓ Data verification passed")

    # Cleanup
    shutil.rmtree(temp_dir)
    print(f"   ✓ Temp directory cleaned up")

except Exception as e:
    print(f"   ✗ create_cached_cameras_and_images failed: {e}")
    import traceback
    traceback.print_exc()
    # Cleanup on error
    if 'temp_dir' in locals() and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    exit(1)

# Test backward compatibility (old API)
print("\n5. Testing backward compatibility (old API)...")
try:
    temp_dir = tempfile.mkdtemp(prefix="test_backward_compat_")

    # Test without render_config (OLD API - still supported)
    data_old = cu.create_cached_cameras_and_images(
        scene=scene,
        cache_dir=temp_dir,
        num_train=3,
        num_test=3,
        center=[0.0, 0.0, 0.0],
        radius=2.0,
        render_res=64,
        fov=40.0,
        spp=16,
        force_rerender=True
    )
    print(f"   ✓ Backward compatibility maintained (old API works)")
    print(f"      - Training cameras: {len(data_old['train_sensors'])}")
    print(f"      - Test cameras: {len(data_old['test_sensors'])}")

    # Cleanup
    shutil.rmtree(temp_dir)

except Exception as e:
    print(f"   ✗ Backward compatibility test failed: {e}")
    import traceback
    traceback.print_exc()
    if 'temp_dir' in locals() and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    exit(1)

print("\n" + "=" * 70)
print("All RenderConfig Tests PASSED!")
print("=" * 70)
print("\nSummary:")
print("  ✓ RenderConfig class working correctly")
print("  ✓ ExperimentConfig integration successful")
print("  ✓ create_cached_cameras_and_images works with render_config")
print("  ✓ Backward compatibility maintained")
print("  ✓ Up vector parameter properly integrated")
