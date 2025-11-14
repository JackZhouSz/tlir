"""
Image quality metrics for evaluating reconstruction results.

This module provides common metrics like PSNR, MSE, and MAE for comparing
predicted and target images.
"""

from __future__ import annotations

import drjit as dr
import mitsuba as mi
import numpy as np
from typing import Union


def compute_mse(predicted: mi.TensorXf, target: mi.TensorXf) -> float:
    """
    Compute Mean Squared Error (MSE) between predicted and target images.

    Args:
        predicted: Predicted image tensor
        target: Target image tensor

    Returns:
        MSE value
    """
    mse = dr.mean(dr.square(predicted - target), axis=None)
    return float(mse.array[0])


def compute_mae(predicted: mi.TensorXf, target: mi.TensorXf) -> float:
    """
    Compute Mean Absolute Error (MAE) between predicted and target images.

    Args:
        predicted: Predicted image tensor
        target: Target image tensor

    Returns:
        MAE value
    """
    mae = dr.mean(dr.abs(predicted - target), axis=None)
    return float(mae.array[0])


def compute_psnr(predicted: mi.TensorXf, target: mi.TensorXf) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between predicted and target images.

    PSNR is commonly used to measure the quality of reconstruction in image processing.
    Higher values indicate better quality, with typical values ranging from 20-50 dB.

    Args:
        predicted: Predicted image tensor (values in [0, 1])
        target: Target image tensor (values in [0, 1])

    Returns:
        PSNR value in dB
    """
    mse = compute_mse(predicted, target)

    if mse < 1e-10:
        return 100.0  # Perfect match, return high PSNR

    # PSNR = -10 * log10(MSE) for normalized images with max value 1.0
    # This is equivalent to 20 * log10(MAX_PIXEL_VALUE / sqrt(MSE))
    # where MAX_PIXEL_VALUE = 1.0 for normalized images
    psnr = -10.0 * np.log10(mse)
    return psnr


def compute_metrics(predicted: mi.TensorXf, target: mi.TensorXf) -> dict:
    """
    Compute multiple metrics between predicted and target images.

    Args:
        predicted: Predicted image tensor
        target: Target image tensor

    Returns:
        Dictionary containing MSE, MAE, and PSNR values
    """
    return {
        'mse': compute_mse(predicted, target),
        'mae': compute_mae(predicted, target),
        'psnr': compute_psnr(predicted, target)
    }
