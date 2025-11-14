"""
Visualization and logging utilities for radiance field reconstruction.

This module contains functions for plotting, logging, and visualizing
training progress and results.
"""

import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from typing import List, Optional, Dict, Any
import os
from datetime import datetime


def plot_list(images: List[mi.TensorXf], title: Optional[str] = None, 
              figsize: tuple = (18, 3), save_path: Optional[str] = None) -> None:
    """
    Plot a list of images in one row.
    
    Args:
        images: List of Mitsuba image tensors
        title: Optional title for the plot
        figsize: Figure size tuple
        save_path: Optional path to save the plot
    """
    fig, axs = plt.subplots(1, len(images), figsize=figsize)
    
    if len(images) == 1:
        axs = [axs]  # Make it iterable for single image
    
    for i, img in enumerate(images):
        axs[i].imshow(mi.util.convert_to_bitmap(img))
        axs[i].axis('off')
    
    if title is not None:
        plt.suptitle(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_comparison(images1: List[mi.TensorXf], images2: List[mi.TensorXf], 
                   title1: str = "Method 1", title2: str = "Method 2",
                   view_idx: int = 1, figsize: tuple = (10, 4),
                   save_path: Optional[str] = None) -> None:
    """
    Plot side-by-side comparison of two sets of images.
    
    Args:
        images1: First set of images
        images2: Second set of images
        title1: Title for first set
        title2: Title for second set
        view_idx: Index of view to compare
        figsize: Figure size tuple
        save_path: Optional path to save the plot
    """
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    axs[0].imshow(mi.util.convert_to_bitmap(images1[view_idx]))
    axs[0].set_title(title1)
    axs[0].axis('off')
    
    axs[1].imshow(mi.util.convert_to_bitmap(images2[view_idx]))
    axs[1].set_title(title2)
    axs[1].axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_loss_curves(losses_dict: Dict[str, List[float]], 
                    title: str = "Training Loss Comparison",
                    figsize: tuple = (12, 6), 
                    save_path: Optional[str] = None) -> None:
    """
    Plot multiple loss curves for comparison.
    
    Args:
        losses_dict: Dictionary mapping method names to loss lists
        title: Plot title
        figsize: Figure size tuple
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=figsize)
    
    for method_name, losses in losses_dict.items():
        plt.plot(losses, label=method_name, alpha=0.7, linewidth=2)
    
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_training_progress(losses: List[float], 
                          intermediate_images: List[List[mi.TensorXf]],
                          ref_images: List[mi.TensorXf],
                          stage_names: Optional[List[str]] = None,
                          save_dir: Optional[str] = None) -> None:
    """
    Plot comprehensive training progress including loss curve and stage results.
    
    Args:
        losses: List of loss values
        intermediate_images: List of image lists for each stage
        ref_images: Reference images
        stage_names: Optional names for stages
        save_dir: Optional directory to save plots
    """
    if stage_names is None:
        stage_names = [f"Stage {i+1}" for i in range(len(intermediate_images))]
    
    # Plot loss curve
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    # Plot final result vs reference
    plt.subplot(1, 2, 2)
    if intermediate_images:
        final_images = intermediate_images[-1]
        view_idx = min(1, len(final_images) - 1)
        plt.imshow(mi.util.convert_to_bitmap(final_images[view_idx]))
        plt.title('Final Result')
        plt.axis('off')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_progress.png'), 
                   dpi=150, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    
    # Plot stage-by-stage results
    for i, (stage_images, stage_name) in enumerate(zip(intermediate_images, stage_names)):
        plot_list(stage_images, f'{stage_name} Results', 
                 save_path=os.path.join(save_dir, f'{stage_name.lower().replace(" ", "_")}.png') 
                 if save_dir else None)


def log_training_stats(losses: List[float], 
                      config: Any,
                      method_name: str = "Training",
                      save_path: Optional[str] = None) -> None:
    """
    Log training statistics to console and optionally to file.
    
    Args:
        losses: List of loss values
        config: Training configuration object
        method_name: Name of the training method
        save_path: Optional path to save log file
    """
    print(f"\n=== {method_name} Statistics ===")
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Total improvement: {losses[0] - losses[-1]:.6f}")
    print(f"Number of iterations: {len(losses)}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Grid resolution: {config.grid_init_res} -> {config.grid_init_res * (2 ** (config.num_stages - 1))}")
    print(f"Loss type: {config.loss_type}")
    print(f"ReLU enabled: {config.use_relu}")
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(f"=== {method_name} Statistics ===\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Initial loss: {losses[0]:.6f}\n")
            f.write(f"Final loss: {losses[-1]:.6f}\n")
            f.write(f"Total improvement: {losses[0] - losses[-1]:.6f}\n")
            f.write(f"Number of iterations: {len(losses)}\n")
            f.write(f"Learning rate: {config.learning_rate}\n")
            f.write(f"Grid resolution: {config.grid_init_res} -> {config.grid_init_res * (2 ** (config.num_stages - 1))}\n")
            f.write(f"Loss type: {config.loss_type}\n")
            f.write(f"ReLU enabled: {config.use_relu}\n")


def create_training_summary(results_dict: Dict[str, Dict[str, Any]], 
                          save_path: Optional[str] = None) -> None:
    """
    Create a comprehensive training summary comparing multiple methods.
    
    Args:
        results_dict: Dictionary mapping method names to result dictionaries
        save_path: Optional path to save summary
    """
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for method_name, results in results_dict.items():
        losses = results['losses']
        config = results.get('config', None)
        
        print(f"\n{method_name}:")
        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Improvement: {losses[0] - losses[-1]:.6f}")
        if config:
            print(f"  Learning rate: {config.learning_rate}")
            print(f"  Iterations: {len(losses)}")
    
    # Find best method
    best_method = min(results_dict.keys(), 
                     key=lambda k: results_dict[k]['losses'][-1])
    print(f"\nBest method: {best_method} (loss: {results_dict[best_method]['losses'][-1]:.6f})")
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write("TRAINING SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for method_name, results in results_dict.items():
                losses = results['losses']
                config = results.get('config', None)
                
                f.write(f"{method_name}:\n")
                f.write(f"  Final loss: {losses[-1]:.6f}\n")
                f.write(f"  Improvement: {losses[0] - losses[-1]:.6f}\n")
                if config:
                    f.write(f"  Learning rate: {config.learning_rate}\n")
                    f.write(f"  Iterations: {len(losses)}\n")
                f.write("\n")
            
            best_method = min(results_dict.keys(), 
                             key=lambda k: results_dict[k]['losses'][-1])
            f.write(f"Best method: {best_method} (loss: {results_dict[best_method]['losses'][-1]:.6f})\n")


def save_images(images: List[mi.TensorXf], 
               save_dir: str, 
               prefix: str = "image",
               format: str = "png") -> None:
    """
    Save a list of images to disk.
    
    Args:
        images: List of Mitsuba image tensors
        save_dir: Directory to save images
        prefix: Prefix for filenames
        format: Image format ('png', 'jpg', etc.)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for i, img in enumerate(images):
        filename = f"{prefix}_{i:03d}.{format}"
        filepath = os.path.join(save_dir, filename)
        
        # Convert to bitmap and save
        bitmap = mi.util.convert_to_bitmap(img)
        plt.imsave(filepath, bitmap)


def create_gif_from_images(image_lists: List[List[mi.TensorXf]], 
                          save_path: str,
                          duration: float = 0.5) -> None:
    """
    Create a GIF from multiple lists of images (e.g., training progress).
    
    Args:
        image_lists: List of image lists (e.g., one per training stage)
        save_path: Path to save the GIF
        duration: Duration per frame in seconds
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL not available. Cannot create GIF.")
        return
    
    frames = []
    
    for image_list in image_lists:
        for img in image_list:
            bitmap = mi.util.convert_to_bitmap(img)
            # Convert to PIL Image
            pil_img = Image.fromarray((bitmap * 255).astype(np.uint8))
            frames.append(pil_img)
    
    if frames:
        frames[0].save(save_path, save_all=True, append_images=frames[1:], 
                      duration=int(duration * 1000), loop=0)
        print(f"GIF saved to {save_path}")


class TrainingLogger:
    """Simple training logger for tracking progress."""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize the logger.
        
        Args:
            log_file: Optional file to write logs to
        """
        self.log_file = log_file
        self.start_time = datetime.now()
        
        if self.log_file:
            with open(self.log_file, 'w') as f:
                f.write(f"Training started at {self.start_time}\n")
                f.write("="*50 + "\n")
    
    def log(self, message: str) -> None:
        """Log a message to console and optionally to file."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_message + "\n")
    
    def log_progress(self, stage: int, iteration: int, loss: float) -> None:
        """Log training progress."""
        self.log(f"Stage {stage+1:02d}, Iteration {iteration+1:02d}: Loss = {loss}")
    
    def log_stage_complete(self, stage: int, final_loss: float) -> None:
        """Log stage completion."""
        self.log(f"Stage {stage+1:02d} completed. Final loss: {final_loss}")
    
    def log_training_complete(self, total_losses: List[float]) -> None:
        """Log training completion."""
        duration = datetime.now() - self.start_time
        self.log(f"Training completed in {duration}")
        self.log(f"Initial loss: {total_losses[0]:.6f}")
        self.log(f"Final loss: {total_losses[-1]:.6f}")
        self.log(f"Total improvement: {total_losses[0] - total_losses[-1]}")
