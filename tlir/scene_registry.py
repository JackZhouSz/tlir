"""
Scene Registry: Automatic mapping from scene names to scene files.

This module provides a registry system for scenes, allowing automatic
discovery and loading of scene files without hardcoded if/else chains.
"""

import os
from typing import Dict, Optional


class SceneRegistry:
    """
    Registry for mapping scene names to their file paths.

    Scenes are automatically discovered by looking for scene.xml files
    in subdirectories of the scenes/ folder.
    """

    def __init__(self, scenes_dir: str = "./scenes"):
        """
        Initialize the scene registry.

        Args:
            scenes_dir: Base directory containing scene subdirectories
        """
        self.scenes_dir = scenes_dir
        self._registry: Dict[str, str] = {}
        self._discover_scenes()

    def _discover_scenes(self):
        """
        Automatically discover scenes by scanning the scenes directory.

        Looks for scene.xml files in subdirectories and registers them
        using the subdirectory name as the scene name.
        """
        if not os.path.exists(self.scenes_dir):
            return

        # Scan each subdirectory
        for item in os.listdir(self.scenes_dir):
            item_path = os.path.join(self.scenes_dir, item)

            # Check if it's a directory
            if os.path.isdir(item_path):
                scene_file = os.path.join(item_path, "scene.xml")

                # Check if scene.xml exists
                if os.path.exists(scene_file):
                    # Register using directory name as scene name
                    self._registry[item] = scene_file

    def register(self, name: str, path: str):
        """
        Manually register a scene.

        Args:
            name: Scene name (e.g., "lego", "fog")
            path: Path to scene file
        """
        self._registry[name] = path

    def get_scene_path(self, name: str) -> Optional[str]:
        """
        Get the file path for a scene by name.

        Args:
            name: Scene name

        Returns:
            Path to scene file, or None if not found
        """
        return self._registry.get(name)

    def list_scenes(self) -> list:
        """
        Get list of all registered scene names.

        Returns:
            List of scene names
        """
        return sorted(list(self._registry.keys()))

    def exists(self, name: str) -> bool:
        """
        Check if a scene is registered.

        Args:
            name: Scene name

        Returns:
            True if scene exists, False otherwise
        """
        return name in self._registry

    def __contains__(self, name: str) -> bool:
        """Allow 'scene_name in registry' syntax."""
        return self.exists(name)

    def __getitem__(self, name: str) -> str:
        """
        Get scene path using dictionary-like syntax.

        Args:
            name: Scene name

        Returns:
            Path to scene file

        Raises:
            KeyError: If scene not found
        """
        if name not in self._registry:
            available = ", ".join(self.list_scenes())
            raise KeyError(
                f"Scene '{name}' not found. Available scenes: {available}"
            )
        return self._registry[name]

    def __repr__(self):
        scenes = self.list_scenes()
        return f"SceneRegistry({len(scenes)} scenes: {scenes})"


# Global scene registry instance
_global_registry = None


def get_scene_registry(scenes_dir: str = "./scenes") -> SceneRegistry:
    """
    Get the global scene registry instance.

    Creates a singleton registry that is reused across calls.

    Args:
        scenes_dir: Base directory for scenes (only used on first call)

    Returns:
        SceneRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = SceneRegistry(scenes_dir)
    return _global_registry


def get_scene_path(name: str) -> Optional[str]:
    """
    Convenience function to get scene path by name.

    Args:
        name: Scene name

    Returns:
        Path to scene file, or None if not found
    """
    registry = get_scene_registry()
    return registry.get_scene_path(name)


def list_available_scenes() -> list:
    """
    Convenience function to list all available scenes.

    Returns:
        List of scene names
    """
    registry = get_scene_registry()
    return registry.list_scenes()
