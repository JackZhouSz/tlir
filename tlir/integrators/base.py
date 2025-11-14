"""
Base integrator class for TLIR project.

Provides TLIRIntegrator that extends Mitsuba's RBIntegrator with methods for
rendering and differentiating arbitrary ray sets.
"""

import numpy as np
import drjit as dr
import mitsuba as mi
from typing import Tuple, Optional


class TLIRIntegrator(mi.python.ad.integrators.common.RBIntegrator):
    """
    Base integrator for TLIR, extending RBIntegrator with ray-based rendering
    and differentiation. All TLIR integrators should inherit from this class.
    """

    def __init__(self, props):
        """Initialize the TLIR integrator.

        Args:
            props: Mitsuba properties
        """
        super().__init__(props)
        # Configurable loss type for color loss
        self.loss_type = 'l2'  # 'l1' or 'l2'

        # Optimizer and parameters (initialized via initialize_optimizer)
        self.optimizer = None
        self.params = None

        # Stochastic preconditioning state
        self.spn_alpha = 0.0
        self.spn_gamma = 0.0
        self.spn_step = 0

        # AOV loss configuration (opacity + empty space)
        self.target_masks = None
        self.opacity_weight = 0.1
        self.empty_space_weight = 0.1

    # ========== Optimizer Management Methods ==========

    def get_trainable_parameters(self):
        """Get dictionary of parameters to optimize.

        Override in subclasses to specify which parameters should be optimized.
        Default implementation returns sigmat and sh_coeffs if available.

        Returns:
            Dict of parameter names to parameter tensors
        """
        if self.params is None:
            self.params = mi.traverse(self)

        opt_params = {}
        if 'sh_coeffs' in self.params:
            opt_params['sh_coeffs'] = self.params['sh_coeffs']
        if 'sigmat' in self.params:
            opt_params['sigmat'] = self.params['sigmat']
        if 'majorant_grid' in self.params:
            opt_params['majorant_grid'] = self.params['majorant_grid']

        return opt_params

    def initialize_optimizer(self, learning_rate=0.2):
        """Initialize optimizer with trainable parameters.

        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        self.params = mi.traverse(self)
        trainable_params = self.get_trainable_parameters()
        self.optimizer = mi.ad.Adam(lr=learning_rate, params=trainable_params)
        self.params.update(self.optimizer)

    def step_optimizer(self):
        """Step the optimizer and update parameters."""
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized. Call initialize_optimizer() first.")
        self.optimizer.step()
        self.params.update(self.optimizer)

    def upsample_parameters(self, factor=2):
        """Upsample grid parameters by given factor.

        Args:
            factor: Upsampling factor (default: 2)
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized. Call initialize_optimizer() first.")

        if 'sigmat' in self.optimizer:
            new_res = factor * self.optimizer['sigmat'].shape[0]
            new_shape = [new_res, new_res, new_res]
            self.optimizer['sigmat'] = dr.upsample(self.optimizer['sigmat'], new_shape)
            self.optimizer['sh_coeffs'] = dr.upsample(self.optimizer['sh_coeffs'], new_shape)

            # Upsample majorant grid if it exists
            if 'majorant_grid' in self.optimizer:
                self.optimizer['majorant_grid'] = dr.upsample(self.optimizer['majorant_grid'], new_shape)

        self.params.update(self.optimizer)

    def apply_density_constraints(self, use_relu=True):
        """Apply constraints to density parameters.

        Args:
            use_relu: If True, do nothing (ReLU handles it). If False, clamp to >= 0.
        """
        if not use_relu and self.optimizer is not None and 'sigmat' in self.optimizer:
            self.optimizer['sigmat'] = dr.maximum(self.optimizer['sigmat'], 0.0)
            self.params.update(self.optimizer)

    def setup_stochastic_preconditioning(self, starting_alpha=0.0, num_iterations=-1):
        """Initialize stochastic preconditioning parameters.

        Args:
            starting_alpha: Starting noise scale
            num_iterations: Number of iterations for exponential decay
        """
        self.spn_alpha = starting_alpha
        self.spn_step = 0

        # Calculate gamma for exponential decay
        if num_iterations > 0 and starting_alpha > 0:
            self.spn_gamma = (1e-16 / starting_alpha) ** (1.0 / num_iterations)
        else:
            self.spn_gamma = 0.0
            self.spn_alpha = 0.0

    def update_spn_alpha(self, num_iterations=-1):
        """Update stochastic preconditioning alpha (exponential decay).

        Args:
            num_iterations: Total number of iterations for decay
        """
        if self.spn_step < num_iterations:
            self.spn_alpha *= self.spn_gamma
            self.spn_step += 1
        else:
            self.spn_alpha = 0.0

    def get_spn_alpha(self):
        """Get current stochastic preconditioning alpha.

        Returns:
            Current alpha value
        """
        return self.spn_alpha

    def post_step_update(self, config):
        """Update integrator state after optimizer step.

        This is called after each optimizer step to update integrator-specific state
        (e.g., SPN alpha decay, density constraints). Override in subclasses for
        custom behavior.

        Args:
            config: Training configuration object with attributes like:
                   - stochastic_preconditioning_iterations
                   - use_relu
        """
        # Update stochastic preconditioning alpha
        if hasattr(config, 'stochastic_preconditioning_iterations'):
            self.update_spn_alpha(num_iterations=config.stochastic_preconditioning_iterations)

        # Apply density constraints
        if hasattr(config, 'use_relu'):
            self.apply_density_constraints(use_relu=config.use_relu)

    def set_aov_loss_config(self, masks=None, opacity_weight=None, empty_space_weight=None):
        """Configure AOV loss parameters.

        Args:
            masks: Target masks (1 where object exists, 0 for background)
            opacity_weight: Weight for opacity loss (None to keep current)
            empty_space_weight: Weight for empty space loss (None to keep current)
        """
        if masks is not None:
            self.target_masks = masks
        if opacity_weight is not None:
            self.opacity_weight = opacity_weight
        if empty_space_weight is not None:
            self.empty_space_weight = empty_space_weight

    # ========== Loss Function Methods ==========

    def compute_color_loss(self, rendered, target):
        """Compute loss between rendered and target colors.

        Uses self.loss_type ('l1' or 'l2'). Override in subclasses for custom losses.

        Args:
            rendered: Rendered radiance values
            target: Ground truth colors

        Returns:
            Scalar loss value
        """
        if self.loss_type == 'l1':
            return dr.mean(dr.abs(rendered - target), axis=None)
        elif self.loss_type == 'l2':
            return dr.mean(dr.square(rendered - target), axis=None)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def compute_aov_loss(self, aovs):
        """Compute loss on AOVs (arbitrary output values).

        Default implementation computes opacity + empty space losses if masks are set.
        Override in subclasses for custom AOV losses.

        Args:
            aovs: Dict with keys 'throughput', 'depth', 'normal'

        Returns:
            Scalar loss value or None
        """
        if self.target_masks is None or 'throughput' not in aovs:
            return None

        total_loss = mi.Float(0.0)
        throughput = aovs['throughput']

        # Opacity loss: throughput should be 1 where objects exist
        if self.opacity_weight > 0.0:
            opacity_error = dr.square(throughput - 1.0) * self.target_masks
            opacity_loss = dr.mean(opacity_error, axis=None)
            total_loss += self.opacity_weight * opacity_loss

        # Empty space loss: throughput should be 1 where background exists
        if self.empty_space_weight > 0.0:
            empty_space_error = dr.square(throughput) * (1.0 - self.target_masks)
            empty_space_loss = dr.mean(empty_space_error, axis=None)
            total_loss += self.empty_space_weight * empty_space_loss

        return total_loss if (self.opacity_weight > 0.0 or self.empty_space_weight > 0.0) else None

    def compute_point_loss(self, point_data):
        """Compute point-dependent losses during backward pass.

        Default implementation returns None (no point loss). Override in subclasses
        to add point-dependent losses applied at each sample location.

        Args:
            point_data: Dict with point-specific data (normal_at_p, position, etc.)

        Returns:
            Scalar loss value or None
        """
        return None

    @dr.syntax
    def sample_aovs(self, mode, scene, sampler, ray, δaovs, state_in, active, **kwargs):
        """Render AOVs (throughput, depth, normals) with radiative backpropagation.

        Default implementation returns empty dict. Override in subclasses that support AOVs.

        Args:
            mode: dr.ADMode.Primal or dr.ADMode.Backward
            scene: Mitsuba scene
            sampler: Sampler for random numbers
            ray: Rays to render
            δaovs: Dict of AOV gradients (None in forward pass)
            state_in: State from forward pass (None in forward pass)
            active: Active ray mask
            **kwargs: Additional arguments

        Returns:
            (aovs, valid, state_out) tuple
        """
        return {}, mi.Bool(False), None

    # ========== Common Helper Functions for AOV Rendering ==========

    def compute_density_gradient(self, p, epsilon=1e-4):
        """Compute numerical gradient of density using finite differences.

        Args:
            p: Query point
            epsilon: Step size

        Returns:
            Density gradient vector (∇σ)
        """
        grad_x = (self.sigmat.eval(dr.clip(p, 0.0, 1.0) + mi.Vector3f(epsilon, 0, 0))[0] -
                 self.sigmat.eval(dr.clip(p, 0.0, 1.0) - mi.Vector3f(epsilon, 0, 0))[0]) / (2 * epsilon)
        grad_y = (self.sigmat.eval(dr.clip(p, 0.0, 1.0) + mi.Vector3f(0, epsilon, 0))[0] -
                 self.sigmat.eval(dr.clip(p, 0.0, 1.0) - mi.Vector3f(0, epsilon, 0))[0]) / (2 * epsilon)
        grad_z = (self.sigmat.eval(dr.clip(p, 0.0, 1.0) + mi.Vector3f(0, 0, epsilon))[0] -
                 self.sigmat.eval(dr.clip(p, 0.0, 1.0) - mi.Vector3f(0, 0, epsilon))[0]) / (2 * epsilon)
        return mi.Vector3f(grad_x, grad_y, grad_z)

    def compute_normal_from_density_gradient(self, density_grad):
        """Compute surface normal from density gradient (-∇σ / ||∇σ||).

        Args:
            density_grad: Density gradient

        Returns:
            Normalized normal vector
        """
        grad_length = dr.norm(density_grad)
        return dr.select(grad_length > 1e-6,
                        -density_grad / grad_length,
                        mi.Vector3f(0.0))

    def compute_depth_gradient_term(self, depth_contrib):
        """Compute ∂(output_depth)/∂(current_sample).

        Args:
            depth_contrib: Depth contribution at current sample

        Returns:
            Gradient term to multiply by δdepth
        """
        return depth_contrib

    def compute_normal_gradient_term(self, normal_contrib, δnormal, normal_out, normal_accum_length):
        """Compute ∂(output_normal)/∂(current_sample) accounting for normalization.

        Uses ∂normalize(v)/∂v = (I - n⊗n) / ||v|| to transform gradient through normalization.

        Args:
            normal_contrib: Normal contribution at current sample
            δnormal: Gradient w.r.t. normalized output normal
            normal_out: Normalized output normal from forward pass
            normal_accum_length: Length of accumulated normal

        Returns:
            Gradient term vector
        """
        # Transform δnormal through normalization gradient
        # ∂normalize(v)/∂v = (I - n⊗n) / ||v||
        projection = dr.dot(normal_out, δnormal)
        normal_grad_term = (δnormal - normal_out * projection) / dr.maximum(normal_accum_length, 1e-8)
        return normal_grad_term

    # ========== Common Utility Helper Functions ==========

    def _setup_ray_bbox_intersection(self, ray, active, primal, has_gradients=True):
        """Setup ray-bbox intersection with early termination logic.

        Args:
            ray: Input ray
            active: Active lanes mask
            primal: Whether in primal or backward mode
            has_gradients: Whether gradients are non-zero

        Returns:
            (ray, mint, maxt, active) tuple
        """
        ray = mi.Ray3f(ray)
        hit, mint, maxt = self.bbox.ray_intersect(ray)
        active = mi.Bool(active)
        active &= hit  # Ignore rays that miss the bbox

        # Early termination in backward mode if no gradients
        if not primal and not has_gradients:
            active = mi.Bool(False)

        return ray, mint, maxt, active

    def _apply_spn_noise(self, p, spn_alpha, sampler, active):
        """Apply stochastic preconditioning noise to query point.

        Args:
            p: Query point
            spn_alpha: Noise scale parameter
            sampler: Random sampler
            active: Active lanes mask

        Returns:
            Perturbed or original query point
        """
        if spn_alpha > 0.0:
            noise = mi.Vector3f(
                sampler.next_1d(active) * 2.0 - 1.0,
                sampler.next_1d(active) * 2.0 - 1.0,
                sampler.next_1d(active) * 2.0 - 1.0
            )
            noise_scale = spn_alpha / self.grid_res
            return p + noise * noise_scale
        return p

    def _eval_density(self, p, clip_bounds=True, use_majorant=False, majorant=None):
        """Evaluate density at query point with optional clipping and clamping.

        Args:
            p: Query point
            clip_bounds: Clip point to [0, 1]^3
            use_majorant: Clamp density to majorant
            majorant: Majorant value

        Returns:
            Density value
        """
        pos = dr.clip(p, 0.0, 1.0) if clip_bounds else p
        sigmat = self.sigmat.eval(pos)[0]

        if self.use_relu:
            if use_majorant and majorant is not None:
                sigmat = dr.clip(sigmat, 0.0, majorant)
            else:
                sigmat = dr.maximum(sigmat, 0.0)
        elif use_majorant and majorant is not None:
            sigmat = dr.minimum(sigmat, majorant)

        return sigmat

    def _compute_aov_contributions(self, t, weight, p_query, compute_normal=True):
        """Compute AOV contributions (depth and normal) at current sample.

        Args:
            t: Distance along ray
            weight: Weighting factor
            p_query: Query point for normal computation
            compute_normal: Whether to compute normal

        Returns:
            (depth_contrib, normal_contrib) tuple
        """
        # Depth contribution
        depth_contrib = t * weight

        # Normal contribution
        if compute_normal:
            density_grad = self.compute_density_gradient(p_query)
            normal_at_p = self.compute_normal_from_density_gradient(density_grad)
            normal_contrib = normal_at_p * weight
        else:
            normal_contrib = mi.Vector3f(0.0)

        return depth_contrib, normal_contrib

    def _normalize_normal_output(self, normal_accum):
        """Normalize accumulated normal vector.

        Args:
            normal_accum: Accumulated normal vector

        Returns:
            (normal_out, normal_accum_length) tuple
        """
        normal_accum_length = dr.norm(normal_accum)
        normal_out = dr.select(normal_accum_length > 1e-6,
                              normal_accum / normal_accum_length,
                              mi.Vector3f(0.0))
        return normal_out, normal_accum_length

    # ========== Gradient Propagation Helper Functions ==========

    def propagate_depth_gradient(self, δdepth, depth_contrib):
        """Propagate depth gradient through backward pass.

        Args:
            δdepth: Gradient w.r.t. output depth
            depth_contrib: Depth contribution at current sample
        """
        if δdepth is not None:
            depth_grad_term = self.compute_depth_gradient_term(depth_contrib)
            dr.backward_from(δdepth * depth_grad_term)

    def propagate_normal_gradient(self, δnormal, normal_contrib, state_in):
        """Propagate normal gradient through backward pass.

        Args:
            δnormal: Gradient w.r.t. output normal
            normal_contrib: Normal contribution at current sample
            state_in: State dict containing 'normal_out' and 'normal_accum_length'
        """
        if δnormal is not None:
            # Extract state for normalization
            normal_out = state_in.get('normal_out', mi.Vector3f(0.0)) if isinstance(state_in, dict) else mi.Vector3f(0.0)
            normal_accum_length = state_in.get('normal_accum_length', mi.Float(1.0)) if isinstance(state_in, dict) else mi.Float(1.0)

            # Compute gradient term and propagate
            normal_grad_term = self.compute_normal_gradient_term(
                normal_contrib, δnormal, normal_out, normal_accum_length
            )
            dr.backward_from(δnormal * normal_grad_term)

    def propagate_point_losses(self, point_data):
        """Propagate gradients from point-dependent loss functions.

        Args:
            point_data: Dict with point-specific data for loss functions
        """
        loss_value = self.compute_point_loss(point_data)
        if loss_value is not None:
            dr.backward_from(loss_value)

    def render_rays_with_gradient(self,
                            rays: mi.Ray3f,
                            target_colors: mi.Spectrum,
                            sampler: mi.Sampler,
                            scene: mi.Scene,
                            spp: int = 1,
                            **kwargs) -> Tuple[mi.Spectrum, mi.Float, dict]:
        """Render rays and compute gradients in a single call.

        Combines forward pass, loss computation, and backward pass. Uses dr.suspend_grad()
        with manual gradient propagation via dr.backward_from(). Do not call dr.backward()
        on the returned loss value.

        Loss functions are defined as methods on the integrator class:
        - compute_color_loss(rendered, target): Main color loss

        Args:
            rays: Batch of rays to render
            target_colors: Ground truth colors
            sampler: Random number sampler
            scene: Mitsuba scene
            spp: Samples per ray
            **kwargs: Additional arguments (e.g., spn_alpha)

        Returns:
            (rendered_colors, loss, aovs) tuple
        """
        # Wrap everything in suspend_grad to prevent automatic differentiation
        # through the integrator's complex loops
        with dr.suspend_grad():
            # Set sample count for multi-sampling
            sampler.set_sample_count(spp)

            # ========== Uncorrelated forward pass (spp handled internally) ==========
            L, _, _, _ = self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler,
                ray=rays,
                δL=None,
                state_in=None,
                active=mi.Bool(True),
                **kwargs
            )

            # ========== Render AOVs using sample_aovs() ==========
            aovs, _, _ = self.sample_aovs(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler,
                ray=rays,
                δaovs=None,
                state_in=None,
                active=mi.Bool(True),
                **kwargs
            )

            # ========== Compute gradients from rendered values ==========
            # Use dr.resume_grad() to enable gradient tracking locally for loss computation
            with dr.resume_grad():
                # Enable gradient tracking on rendered colors (adjoint radiance)
                dr.enable_grad(L)

                # Enable gradient tracking on all AOVs
                if aovs:
                    for aov in aovs.values():
                        if aov is not None:
                            dr.enable_grad(aov)

                # Compute color loss (required)
                loss = self.compute_color_loss(L, target_colors)

                # Backward through loss to get gradients w.r.t. rendered colors and AOVs
                dr.backward(loss)

                # Add AOV loss if available
                if aovs:
                    aov_loss = self.compute_aov_loss(aovs)

                    if aov_loss is not None:
                        dr.backward(aov_loss)

                # Extract gradient (adjoint radiance δL)
                δL = dr.grad(L)
                # Set to None if all gradients are zero
                if not dr.any(δL != 0, axis=None):
                    δL = None

                # Extract gradients for each AOV using dict comprehension
                δaovs = None
                if aovs:
                    δaovs = {}
                    for k, v in aovs.items():
                        if v is not None:
                            grad = dr.grad(v)
                            # Set to None if all gradients are zero
                            if not dr.any(grad != 0, axis=None):
                                grad = None
                            δaovs[k] = grad
                        else:
                            δaovs[k] = None

            # ========== Backward pass ==========
            _, _, _, state_out = self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler.clone(),
                ray=rays,
                δL=None,
                state_in=None,
                active=mi.Bool(True),
                **kwargs
            )

            δL_scaled = δL / float(spp)

            # Backward pass using original sampler and state from forward pass
            sample_backward_outputs = self.sample(
                mode=dr.ADMode.Backward,
                scene=scene,
                sampler=sampler,
                ray=rays,
                δL=δL_scaled,
                state_in=state_out,
                active=mi.Bool(True),
                **kwargs
            )

            # Clean up intermediate variables
            del sample_backward_outputs

            # Backward pass for AOV gradients
            if aovs and δaovs is not None:
                aovs, _, aov_state = self.sample_aovs(
                    mode=dr.ADMode.Primal,
                    scene=scene,
                    sampler=sampler.clone(),
                    ray=rays,
                    δaovs=None,
                    state_in=None,
                    active=mi.Bool(True),
                    **kwargs
                )

                # Scale gradients for spp
                δaovs_scaled = {k: (v / float(spp) if v is not None else None)
                               for k, v in δaovs.items()} if δaovs is not None else None

                sample_aovs_backward_outputs = self.sample_aovs(
                    mode=dr.ADMode.Backward,
                    scene=scene,
                    sampler=sampler,
                    ray=rays,
                    δaovs=δaovs_scaled,
                    state_in=aov_state,
                    active=mi.Bool(True),
                    **kwargs
                )

                # Clean up intermediate variables
                del sample_aovs_backward_outputs

        return L, loss, aovs

    def render_camera(self, sensor: mi.Sensor, sampler: mi.Sampler, scene: mi.Scene, spp: int = 1, **kwargs):
        """Render all rays from a camera sensor.

        Args:
            sensor: Camera sensor to render from
            sampler: Random number sampler
            scene: Mitsuba scene
            spp: Samples per pixel
            **kwargs: Additional arguments (e.g., spn_alpha)

        Returns:
            (rendered_image, aovs) tuple where:
            - rendered_image: mi.TensorXf of shape (height, width, 3)
            - aovs: List of AOV tensors [depth] if available
        """
        # Get film dimensions
        film = sensor.film()
        res = film.size()
        width, height = res[0], res[1]
        num_rays = width * height

        # Generate pixel positions
        idx = dr.arange(mi.UInt32, num_rays)
        x = idx % width
        y = idx // width

        # Convert to normalized coordinates [0, 1]
        pos_x = (x + 0.5) / width
        pos_y = (y + 0.5) / height
        pos_sample = mi.Point2f(pos_x, pos_y)

        # Sample rays from sensor
        wavelength_sample = 0.5
        time_sample = 0.0
        aperture_sample = mi.Point2f(0.5, 0.5)
        rays, _ = sensor.sample_ray(time_sample, wavelength_sample, pos_sample, aperture_sample)

        # Set sample count for multi-sampling
        sampler.set_sample_count(spp)
        sampler.seed(0, num_rays)

        # Render colors using sample()
        L, _, _, _ = self.sample(
            mode=dr.ADMode.Primal,
            scene=scene,
            sampler=sampler,
            ray=rays,
            δL=None,
            state_in=None,
            active=mi.Bool(True),
            **kwargs
        )

        # Render AOVs using sample_aovs()
        aovs_dict, _, _ = self.sample_aovs(
            mode=dr.ADMode.Primal,
            scene=scene,
            sampler=sampler.clone(),
            ray=rays,
            δaovs=None,
            state_in=None,
            active=mi.Bool(True),
            **kwargs
        )

        # Reshape colors to image format (height, width, 3)
        # Interleave RGB channels: [r0, g0, b0, r1, g1, b1, ...]
        indices_r = dr.arange(mi.UInt32, num_rays) * 3 + 0
        indices_g = dr.arange(mi.UInt32, num_rays) * 3 + 1
        indices_b = dr.arange(mi.UInt32, num_rays) * 3 + 2

        rgb_flat = dr.zeros(mi.Float, num_rays * 3)
        dr.scatter(rgb_flat, L.x, indices_r)
        dr.scatter(rgb_flat, L.y, indices_g)
        dr.scatter(rgb_flat, L.z, indices_b)

        rendered_image = mi.TensorXf(rgb_flat, (height, width, 3))

        # Extract depth AOV if available
        aovs_list = []
        if 'depth' in aovs_dict and aovs_dict['depth'] is not None:
            depth = aovs_dict['depth']
            # Depth is already a flat array (num_rays,), reshape to (height, width)
            depth_tensor = mi.TensorXf(depth, (height, width))
            aovs_list.append(depth_tensor)

        return rendered_image, aovs_list
