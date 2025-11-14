"""
Base integrator class for TLIR project.

This module provides the TLIRIntegrator base class that wraps Mitsuba's RBIntegrator
with convenient methods for rendering arbitrary ray sets (not just whole images).
"""

import numpy as np
import drjit as dr
import mitsuba as mi
from typing import Tuple, Optional


class TLIRIntegrator(mi.python.ad.integrators.common.RBIntegrator):
    """
    Base integrator class for TLIR project.

    Extends Mitsuba's RBIntegrator with convenient methods for rendering and
    differentiating arbitrary sets of rays. This enables efficient ray-based
    training without requiring full image rendering.

    All TLIR integrators should inherit from this class instead of directly
    from RBIntegrator.
    """

    def render_rays(self,
                      rays: mi.Ray3f,
                      sampler: mi.Sampler,
                      scene: mi.Scene,
                      **kwargs) -> Tuple[mi.Spectrum, mi.Bool, list, mi.Spectrum]:
        """
        Render a batch of rays in forward (primal) mode.

        This method wraps the integrator's `sample()` method for forward-mode
        rendering of arbitrary ray sets. It uses a cloned sampler to preserve
        the original sampler state for the backward pass.

        The rendering is wrapped in dr.suspend_grad() to prevent automatic
        differentiation from tracking through the integrator's complex loops.

        Args:
            rays: Batch of rays to render (mi.Ray3f)
            sampler: Sampler for random number generation
            scene: Mitsuba scene
            **kwargs: Additional arguments passed to sample() (e.g., spn_alpha)

        Returns:
            Tuple of (L, valid, aovs, state_out):
                - L: Radiance values (mi.Spectrum)
                - valid: Ray validity mask (mi.Bool)
                - aovs: Arbitrary output values (list)
                - state_out: Integrator state for backward pass

        Example:
            >>> rays, target_colors = ray_batch.to_mitsuba()
            >>> L, valid, aovs, state = integrator.render_rays(
            ...     rays, sampler, scene, spn_alpha=0.1
            ... )
            >>> loss = compute_loss(L, target_colors)
        """
        with dr.suspend_grad():
            num_rays = dr.width(rays.o)

            # Forward pass using cloned sampler (preserves original sampler state)
            L, valid, aovs, state_out = self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler,
                ray=rays,
                δL=None,
                state_in=None,
                active=dr.ones(mi.Bool, num_rays),
                **kwargs
            )

        return L, valid, aovs, state_out

    def render_backward_rays(self,
                       rays: mi.Ray3f,
                       sampler: mi.Sampler,
                       scene: mi.Scene,
                       δL: mi.Spectrum,
                       state_in: mi.Spectrum,
                       δaovs=None,
                       **kwargs) -> None:
        """
        Propagate gradients through a batch of rays in backward mode.

        This method wraps the integrator's `sample()` method for backward-mode
        gradient propagation. It must be called after `render_rays()` with
        the same rays and sampler (in the same state), along with the gradient
        w.r.t. radiance (δL), optional AOV gradients (δaovs), and the state
        from the forward pass.

        The method accumulates gradients on scene parameters that have gradient
        tracking enabled (via `dr.enable_grad()`).

        The rendering is wrapped in dr.suspend_grad() to prevent automatic
        differentiation. The integrator uses dr.backward_from() internally
        to manually propagate gradients.

        Args:
            rays: Same batch of rays used in forward pass
            sampler: Same sampler used in forward pass (original, not cloned)
            scene: Mitsuba scene
            δL: Gradient w.r.t. radiance from loss (mi.Spectrum)
            state_in: Integrator state from forward pass
            δaovs: Optional list of gradients w.r.t. AOVs (list of mi.Float)
            **kwargs: Additional arguments passed to sample() (e.g., spn_alpha)

        Returns:
            None (gradients are accumulated on parameters)

        Example:
            >>> # Forward pass
            >>> L, valid, aovs, state = integrator.render_rays(rays, sampler, scene)
            >>>
            >>> # Compute loss and gradients
            >>> loss = compute_loss(L, target_colors)
            >>> δL = compute_loss_gradient(L, target_colors)
            >>> δaovs = [compute_aov_gradient(aov, target) for aov, target in zip(aovs, targets)]
            >>>
            >>> # Backward pass
            >>> integrator.render_backward_rays(rays, sampler, scene, δL, state, δaovs=δaovs)
            >>> opt.step()  # Update parameters
        """
        with dr.suspend_grad():
            num_rays = dr.width(rays.o)

            # Backward pass using original sampler and state from forward pass
            # Pass δaovs via kwargs if provided (for integrators that support it)
            backward_kwargs = dict(kwargs)
            if δaovs is not None:
                backward_kwargs['δaovs'] = δaovs

            L_2, valid_2, aovs_2, state_out_2 = self.sample(
                mode=dr.ADMode.Backward,
                scene=scene,
                sampler=sampler,
                ray=rays,
                δL=δL,
                state_in=state_in,
                active=dr.ones(mi.Bool, num_rays),
                **backward_kwargs
            )

            # Clean up intermediate variables
            del L_2, valid_2, aovs_2, state_out_2

    def render_rays_with_gradient(self,
                            rays: mi.Ray3f,
                            target_colors: mi.Spectrum,
                            sampler: mi.Sampler,
                            scene: mi.Scene,
                            loss_fn,
                            aov_loss_fn=None,
                            target_aovs=None,
                            spp: int = 1,
                            **kwargs) -> Tuple[mi.Spectrum, mi.Float, list]:
        """
        Render rays and compute gradients in a single call.

        This is a convenience method that combines forward pass, loss computation,
        gradient computation, and backward pass into a single call. It follows
        the two-pass rendering pattern automatically.

        This method supports ARBITRARY loss functions by using DrJit's automatic
        differentiation to compute the gradient of the loss w.r.t. rendered colors
        and AOVs (Arbitrary Output Values), following Mitsuba's RBIntegrator.render_backward()
        pattern.

        Multi-sample rendering (spp > 1):
        1. Uncorrelated forward pass: Average over spp samples to get stable L for gradient computation
        2. Compute loss and gradients (δL, δaovs) from averaged L
        3. Correlated forward + backward passes: For each spp sample, accumulate gradients

        IMPORTANT: The entire rendering process is wrapped in dr.suspend_grad()
        to prevent automatic differentiation from trying to track through the
        integrator's complex loops. A local dr.resume_grad() block is used to
        compute the loss gradient. The integrator uses dr.backward_from()
        internally to manually propagate gradients. DO NOT call dr.backward(loss)
        on the returned loss value.

        Args:
            rays: Batch of rays to render
            target_colors: Ground truth colors for loss computation
            sampler: Sampler for random number generation
            scene: Mitsuba scene
            loss_fn: REQUIRED loss function for colors. Should take (rendered, target)
                    and return a scalar loss. No default provided.
            aov_loss_fn: Optional loss function for AOVs. Can take (rendered_aovs, target_aovs)
                        or just (rendered_aovs) for regularization. Returns scalar loss.
                        If None, no AOV loss is computed.
            target_aovs: Optional list of target AOV values. Can be None if using
                        aov_loss_fn for regularization only.
            spp: Samples per pixel (number of times to render each ray). Default: 1
            **kwargs: Additional arguments passed to sample() (e.g., spn_alpha)

        Returns:
            Tuple of (rendered_colors, loss, aovs):
                - rendered_colors: Rendered radiance values (detached)
                - loss: Computed total loss value (detached)
                - aovs: List of AOV values (detached)

        Side Effects:
            Accumulates gradients on scene parameters

        Example:
            >>> # Using L2 loss for colors
            >>> def l2_loss(rendered, target):
            ...     return dr.mean(dr.square(rendered - target))
            >>> rays, target_colors = ray_batch.to_mitsuba()
            >>> rendered, loss, aovs = integrator.render_rays_with_gradient(
            ...     rays, target_colors, sampler, scene, loss_fn=l2_loss
            ... )
            >>> opt.step()  # Update parameters using accumulated gradients

            >>> # Using loss for both colors and AOVs (supervised auxiliary task)
            >>> def depth_loss(rendered_aovs, target_aovs):
            ...     # AOVs[0] is depth, supervise with target
            ...     return dr.mean(dr.square(rendered_aovs[0] - target_aovs[0]))
            >>> rendered, loss, aovs = integrator.render_rays_with_gradient(
            ...     rays, target_colors, sampler, scene,
            ...     loss_fn=l2_loss,
            ...     aov_loss_fn=depth_loss,
            ...     target_aovs=[target_depth]
            ... )
            >>> opt.step()

            >>> # Using AOV regularizer (no target AOVs needed)
            >>> def sparsity_regularizer(rendered_aovs):
            ...     # Regularize AOV[0] to be sparse
            ...     return 0.01 * dr.mean(dr.abs(rendered_aovs[0]))
            >>> rendered, loss, aovs = integrator.render_rays_with_gradient(
            ...     rays, target_colors, sampler, scene,
            ...     loss_fn=l2_loss,
            ...     aov_loss_fn=sparsity_regularizer
            ... )
            >>> opt.step()
        """
        # Wrap everything in suspend_grad to prevent automatic differentiation
        # through the integrator's complex loops
        with dr.suspend_grad():
            # ========== STEP 1: Uncorrelated forward pass (averaged over spp) ==========
            # Following Mitsuba's RBIntegrator.render_backward() pattern
            # Average over multiple samples to get stable L for gradient computation
            L_accum = None
            aovs_accum = None

            for sample_idx in range(spp):
                L_sample, valid, aovs_sample, state_out = self.render_rays(
                    rays, sampler, scene, **kwargs
                )

                # Accumulate results
                if L_accum is None:
                    L_accum = L_sample
                else:
                    L_accum += L_sample

                # Accumulate AOVs
                if aovs_sample and len(aovs_sample) > 0:
                    if aovs_accum is None:
                        aovs_accum = list(aovs_sample)
                    else:
                        for i, aov in enumerate(aovs_sample):
                            aovs_accum[i] += aov

            # Average over samples
            L = L_accum / float(spp)
            aovs = [aov / float(spp) for aov in aovs_accum] if aovs_accum else []

            # ========== STEP 2: Compute gradients from averaged L and AOVs ==========
            # Use dr.resume_grad() to enable gradient tracking locally for loss computation
            with dr.resume_grad():
                # Enable gradient tracking on rendered colors (adjoint radiance)
                dr.enable_grad(L)

                # Enable gradient tracking on AOVs if we have an AOV loss function
                δaovs = None
                if aov_loss_fn is not None and len(aovs) > 0:
                    # Enable gradient tracking on each AOV
                    for aov in aovs:
                        dr.enable_grad(aov)

                # Compute color loss (required)
                loss = loss_fn(L, target_colors)

                # Add AOV loss if loss function provided
                if aov_loss_fn is not None and len(aovs) > 0:
                    aov_loss = aov_loss_fn(aovs)
                    loss = loss + aov_loss

                # Backward through loss to get gradients w.r.t. rendered colors and AOVs
                dr.backward(loss)

                # Extract gradient (adjoint radiance δL)
                δL = dr.grad(L)

                # Extract AOV gradients (δaovs) if AOV loss function was used
                if aov_loss_fn is not None and len(aovs) > 0:
                    δaovs = [dr.grad(aov) for aov in aovs]

            # ========== STEP 3: Correlated forward + backward passes (one per spp) ==========
            # For each sample: do correlated forward pass + backward pass
            # Gradients accumulate in parameters
            for sample_idx in range(spp):
                # Forward pass for PRB (correlated with backward pass via sampler state)
                L_sample, valid, aovs_sample, state_out = self.render_rays(
                    rays, sampler.clone(), scene, **kwargs
                )

                δL_scaled = δL / float(spp)
                δaovs_scaled = [δaov / float(spp) for δaov in δaovs] if δaovs is not None else None
                self.render_backward_rays(rays, sampler, scene, δL_scaled, state_out,
                                            δaovs=δaovs_scaled, **kwargs)

        return L, loss, aovs

    def render_camera(self,
                      sensor: mi.Sensor,
                      sampler: mi.Sampler,
                      scene: mi.Scene,
                      spp: int = 1,
                      **kwargs) -> Tuple[mi.TensorXf, list]:
        """
        Render an image from a camera/sensor.

        This is a convenience method that generates rays for all pixels in the
        sensor's film and renders them using render_rays(). It replaces the need
        to use mi.render() which doesn't work well with integrators that return AOVs.

        Args:
            sensor: Mitsuba sensor/camera to render from
            sampler: Sampler for random number generation
            scene: Mitsuba scene
            spp: Samples per pixel (default: 1)
            **kwargs: Additional arguments passed to render_rays() (e.g., spn_alpha)

        Returns:
            Tuple of (rgb_image, aovs):
                - rgb_image: RGB image as TensorXf with shape (height, width, 3)
                - aovs: List of AOV images as TensorXf with shape (height, width, channels)

        Example:
            >>> sensor = data['test_sensors'][0]
            >>> sampler = mi.load_dict({'type': 'independent', 'sample_count': 16})
            >>> rgb, aovs = integrator.render_camera(sensor, sampler, scene, spp=16)
            >>> depth = aovs[0]  # First AOV is depth
        """
        with dr.suspend_grad():
            # Get film resolution
            film = sensor.film()
            film_size = film.size()
            width, height = film_size[0], film_size[1]
            num_pixels = width * height

            # Accumulate multiple samples per pixel
            L_accum = None
            aovs_accum = None

            for sample_idx in range(spp):
                # VECTORIZED: Create all pixel indices at once
                idx = dr.arange(mi.UInt32, num_pixels)
                x = idx % width
                y = idx // width

                # Convert to normalized coordinates [0, 1]
                # Add random offset for stratified sampling (when spp > 1)
                if spp > 1:
                    # Stratified sampling with random jitter
                    rng = np.random.RandomState(sample_idx)
                    offset_x = rng.rand()
                    offset_y = rng.rand()
                else:
                    # Single sample: use pixel center
                    offset_x = 0.5
                    offset_y = 0.5

                pos_x = (mi.Float(x) + offset_x) / float(width)
                pos_y = (mi.Float(y) + offset_y) / float(height)
                pos_sample = mi.Point2f(pos_x, pos_y)

                # VECTORIZED: Sample all rays at once
                time_sample = mi.Float(0.0)
                wavelength_sample = mi.Float(0.5)
                aperture_sample = mi.Point2f(0.5, 0.5)

                rays, _ = sensor.sample_ray(time_sample, wavelength_sample, pos_sample, aperture_sample)

                # Render all rays
                L, valid, aovs, state = self.render_rays(
                    rays, sampler, scene, **kwargs
                )

                # Accumulate results
                if L_accum is None:
                    L_accum = L
                else:
                    L_accum += L

                # Accumulate AOVs
                if aovs and len(aovs) > 0:
                    if aovs_accum is None:
                        aovs_accum = list(aovs)
                    else:
                        for i, aov in enumerate(aovs):
                            aovs_accum[i] += aov

            # Average over samples
            L = L_accum / float(spp)
            aovs = [aov / float(spp) for aov in aovs_accum] if aovs_accum else []

            # Reshape to image dimensions (avoid unnecessary conversions)
            # L is mi.Spectrum, reshape directly to (height, width, 3)
            rgb_image = dr.reshape(mi.TensorXf, L, shape=[height, width, 3])

            # Reshape AOVs
            aov_images = []
            for aov in aovs:
                # Check if scalar or vector AOV by looking at the total size
                total_size = dr.width(aov)
                if total_size == num_pixels:
                    # Scalar AOV (e.g., depth)
                    aov_images.append(dr.reshape(mi.TensorXf, aov, shape=[height, width]))
                else:
                    # Vector AOV (e.g., normals) - assume channels divide evenly
                    channels = total_size // num_pixels
                    aov_images.append(dr.reshape(mi.TensorXf, aov, shape=[height, width, channels]))

        return rgb_image, aov_images
