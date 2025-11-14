"""
Test script to verify vectorized rendering functions work correctly.
"""

import os
import tempfile
import numpy as np

print("=" * 70)
print("Testing Vectorized Rendering Functions")
print("=" * 70)

# Test imports
print("\n1. Testing imports...")
try:
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    import drjit as dr
    from tlir import config as cf
    from tlir import training
    from tlir import camera_utils as cu
    print("   ✓ All imports successful")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

# Test scene loading
print("\n2. Loading fog scene...")
try:
    config = cf.create_config("test_vectorized", scene_name="fog")
    scene = training.create_scene_reference(config)
    print("   ✓ Fog scene loaded successfully")
except Exception as e:
    print(f"   ✗ Scene loading failed: {e}")
    exit(1)

# Create test camera
print("\n3. Creating test camera...")
try:
    sensors, poses = cu.create_random_sphere_cameras(
        num_cameras=3,
        center=[0.0, 0.0, 0.0],
        radius=2.0,
        render_res=64,
        fov=40.0
    )
    print(f"   ✓ Created {len(sensors)} test cameras (64x64 resolution)")
except Exception as e:
    print(f"   ✗ Camera creation failed: {e}")
    exit(1)

# Create temporary directory for outputs
temp_dir = tempfile.mkdtemp(prefix="test_vectorized_")
print(f"\n4. Testing vectorized rendering functions...")
print(f"   Output directory: {temp_dir}")

# Test render_and_save_depth
print("\n   Testing render_and_save_depth...")
try:
    depths = cu.render_and_save_depth(
        scene=scene,
        sensors=sensors,
        output_dir=os.path.join(temp_dir, "depths"),
        prefix="test",
        save_npy=True,
        save_png=True
    )
    print(f"   ✓ render_and_save_depth: {len(depths)} depth maps rendered")
    print(f"      - Shape: {depths[0].shape}")
    print(f"      - Min depth: {np.min([d[d < 1e9].min() if (d < 1e9).any() else 0 for d in depths]):.4f}")
    print(f"      - Max depth: {np.max([d[d < 1e9].max() if (d < 1e9).any() else 1 for d in depths]):.4f}")
except Exception as e:
    print(f"   ✗ render_and_save_depth failed: {e}")
    import traceback
    traceback.print_exc()

# Test render_and_save_normals
print("\n   Testing render_and_save_normals...")
try:
    normals = cu.render_and_save_normals(
        scene=scene,
        sensors=sensors,
        output_dir=os.path.join(temp_dir, "normals"),
        prefix="test",
        save_npy=True,
        save_png=True
    )
    print(f"   ✓ render_and_save_normals: {len(normals)} normal maps rendered")
    print(f"      - Shape: {normals[0].shape}")
    print(f"      - Range: [{np.min([n.min() for n in normals]):.2f}, {np.max([n.max() for n in normals]):.2f}]")
except Exception as e:
    print(f"   ✗ render_and_save_normals failed: {e}")
    import traceback
    traceback.print_exc()

# Test render_and_save_all
print("\n   Testing render_and_save_all...")
try:
    data = cu.render_and_save_all(
        scene=scene,
        sensors=sensors,
        output_base_dir=os.path.join(temp_dir, "all"),
        prefix="test",
        spp=16,  # Lower spp for faster testing
        save_npy=True,
        save_png=True
    )
    print(f"   ✓ render_and_save_all: {len(data['images'])} views rendered")
    print(f"      - Images shape: {data['images'][0].shape}")
    print(f"      - Depths shape: {data['depths'][0].shape}")
    print(f"      - Normals shape: {data['normals'][0].shape}")
    print(f"      - Masks shape: {data['masks'][0].shape}")
    print(f"      - Depth range: [{data['depth_min']:.4f}, {data['depth_max']:.4f}]")
except Exception as e:
    print(f"   ✗ render_and_save_all failed: {e}")
    import traceback
    traceback.print_exc()

# Verify files were created
print("\n5. Verifying output files...")
expected_files = [
    "depths/test_0000.npy",
    "depths/test_0000.png",
    "normals/test_0000.npy",
    "normals/test_0000.png",
    "all/images/test_0000.npy",
    "all/images/test_0000.png",
    "all/depths/test_depth_0000.npy",
    "all/depths/test_depth_0000.png",
    "all/normals/test_normal_0000.npy",
    "all/normals/test_normal_0000.png",
    "all/masks/test_mask_0000.npy",
    "all/masks/test_mask_0000.png",
]

all_exist = True
for rel_path in expected_files:
    full_path = os.path.join(temp_dir, rel_path)
    if os.path.exists(full_path):
        size = os.path.getsize(full_path)
        print(f"   ✓ {rel_path} ({size} bytes)")
    else:
        print(f"   ✗ {rel_path} MISSING")
        all_exist = False

# Cleanup
print(f"\n6. Cleaning up temporary directory: {temp_dir}")
import shutil
shutil.rmtree(temp_dir)
print("   ✓ Cleanup complete")

print("\n" + "=" * 70)
if all_exist:
    print("All Vectorized Rendering Tests PASSED!")
else:
    print("Some tests FAILED - see output above")
print("=" * 70)
