"""
Utility functions for losses used in TLIR training.

This module contains loss functions for both AOV-based losses and
sample point dependent losses.
"""

import drjit as dr
import mitsuba as mi


# ========== Point-Dependent Loss Functions ==========

def create_normal_gt_alignment_loss(target_normals, weight=1.0):
    """
    Create a loss function that encourages normals at sample points to align
    with ground truth normals.

    This loss computes 1 - dot(normal_at_point, target_normal) for each sample,
    encouraging the normals to point in the same direction as the ground truth.

    Args:
        target_normals: Ground truth normal vectors for each ray
        weight: Weight for this loss term

    Returns:
        Loss function that takes (normal_at_p, ray_idx) and returns weighted loss
    """
    def loss_fn(normal_at_p, ray_idx):
        """
        Compute alignment loss between sampled normal and ground truth normal.

        Args:
            normal_at_p: Normal vector at the sampled point (normalized)
            ray_idx: Index of the ray this sample belongs to

        Returns:
            Scalar loss value
        """
        # Get ground truth normal for this ray
        gt_normal = dr.gather(mi.Vector3f, target_normals, ray_idx)

        # Dot product between predicted and ground truth normals
        # Normalized dot product is in [-1, 1], where 1 means perfect alignment
        dot_product = dr.dot(normal_at_p, gt_normal)

        # Convert to loss: 0 when aligned, 2 when opposite
        # Using 1 - dot gives: 0 when aligned, 1 when perpendicular, 2 when opposite
        alignment_loss = 1.0 - dot_product

        return weight * alignment_loss

    return loss_fn


def create_normal_ray_alignment_loss(rays, weight=1.0, prefer_same_direction=False):
    """
    Create a loss function that encourages normals at sample points to be aligned
    (or anti-aligned) with ray directions.

    This is useful for encouraging normals to face the camera or face away from it.

    Args:
        rays: Ray batch being rendered
        weight: Weight for this loss term
        prefer_same_direction: If True, encourage normals to align with ray direction (face away from camera).
                              If False (default), encourage normals to anti-align (face toward camera).

    Returns:
        Loss function that takes (normal_at_p, ray_idx) and returns weighted loss
    """
    # Extract ray directions
    ray_directions = rays.d

    def loss_fn(normal_at_p, ray_idx):
        """
        Compute alignment loss between sampled normal and ray direction.

        Args:
            normal_at_p: Normal vector at the sampled point (normalized)
            ray_idx: Index of the ray this sample belongs to

        Returns:
            Scalar loss value
        """
        # Get ray direction for this sample
        ray_dir = dr.gather(mi.Vector3f, ray_directions, ray_idx)

        # Dot product between normal and ray direction
        dot_product = dr.dot(normal_at_p, ray_dir)

        if prefer_same_direction:
            # Encourage normals to point in same direction as ray (away from camera)
            # Loss is 0 when dot=1 (same direction), 2 when dot=-1 (opposite)
            alignment_loss = 1.0 - dot_product
        else:
            # Encourage normals to point opposite to ray direction (toward camera)
            # Loss is 0 when dot=-1 (opposite), 2 when dot=1 (same)
            alignment_loss = 1.0 + dot_product

        return weight * alignment_loss

    return loss_fn


def create_normal_smoothness_loss(weight=1.0):
    """
    Create a loss function that encourages normals at adjacent sample points
    to be similar (smoothness regularization).

    NOTE: This requires tracking adjacent samples, which is not currently
    supported in the integrator. This is a placeholder for future implementation.

    Args:
        weight: Weight for this loss term

    Returns:
        Loss function that takes (normal_at_p, prev_normal) and returns weighted loss
    """
    def loss_fn(normal_at_p, prev_normal):
        """
        Compute smoothness loss between consecutive normals along a ray.

        Args:
            normal_at_p: Normal vector at current sample point
            prev_normal: Normal vector at previous sample point

        Returns:
            Scalar loss value
        """
        # Encourage consecutive normals to be similar
        dot_product = dr.dot(normal_at_p, prev_normal)
        smoothness_loss = 1.0 - dot_product

        return weight * smoothness_loss

    return loss_fn


# ========== AOV Loss Functions ==========

def create_opacity_loss(masks, weight=1.0):
    """
    Create an AOV loss that encourages throughput=1 where object mask=1.

    Args:
        masks: Binary mask indicating where objects should be opaque
        weight: Weight for this loss term

    Returns:
        Loss function that takes aovs_dict and returns weighted loss
    """
    mask = mi.Float(masks)

    def loss_fn(aovs):
        """Encourage throughput=1 where object mask=1."""
        throughput = aovs.get('throughput', None)
        if throughput is None:
            return mi.Float(0.0)

        opacity_error = mask * dr.abs(throughput - 1.0)
        opacity_loss = dr.mean(opacity_error, axis=None)
        return weight * opacity_loss

    return loss_fn


def create_empty_space_loss(masks, weight=1.0):
    """
    Create an AOV loss that encourages throughput=0 where object mask=0.

    Args:
        masks: Binary mask indicating where objects exist
        weight: Weight for this loss term

    Returns:
        Loss function that takes aovs_dict and returns weighted loss
    """
    mask = mi.Float(masks)

    def loss_fn(aovs):
        """Encourage throughput=0 where object mask=0."""
        throughput = aovs.get('throughput', None)
        if throughput is None:
            return mi.Float(0.0)

        empty_space_error = (1.0 - mask) * dr.abs(throughput - 0.0)
        empty_space_loss = dr.mean(empty_space_error, axis=None)
        return weight * empty_space_loss

    return loss_fn
