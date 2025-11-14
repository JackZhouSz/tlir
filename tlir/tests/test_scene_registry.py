"""
Test script for scene registry functionality.
"""

from tlir import scene_registry

print("=" * 70)
print("Testing Scene Registry")
print("=" * 70)

# Test 1: Get global registry
print("\n1. Getting global scene registry...")
registry = scene_registry.get_scene_registry()
print(f"   Registry: {registry}")

# Test 2: List available scenes
print("\n2. Available scenes:")
scenes = registry.list_scenes()
for scene in scenes:
    path = registry[scene]
    print(f"   - {scene}: {path}")

# Test 3: Check specific scenes
print("\n3. Checking specific scenes:")
for scene_name in ["lego", "fog"]:
    if scene_name in registry:
        print(f"   ✓ {scene_name}: {registry[scene_name]}")
    else:
        print(f"   ✗ {scene_name}: NOT FOUND")

# Test 4: Test loading with config
print("\n4. Testing scene loading with config:")
try:
    from tlir import config as cf
    from tlir import training

    # Test lego scene
    print("\n   Testing lego scene...")
    config_lego = cf.create_config("test_lego", scene_name="lego")
    scene_lego = training.create_scene_reference(config_lego)
    print(f"   ✓ Lego scene loaded successfully")

    # Test fog scene
    print("\n   Testing fog scene...")
    config_fog = cf.create_config("test_fog", scene_name="fog")
    scene_fog = training.create_scene_reference(config_fog)
    print(f"   ✓ Fog scene loaded successfully")

    # Test invalid scene
    print("\n   Testing invalid scene...")
    try:
        config_invalid = cf.create_config("test_invalid", scene_name="nonexistent")
        scene_invalid = training.create_scene_reference(config_invalid)
        print(f"   ✗ Should have raised error!")
    except ValueError as e:
        print(f"   ✓ Correctly raised error: {e}")

except ImportError as e:
    print(f"   Skipping (Mitsuba not available): {e}")

print("\n" + "=" * 70)
print("Scene Registry Test Complete!")
print("=" * 70)
