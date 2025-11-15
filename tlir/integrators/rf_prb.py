import drjit as dr
import mitsuba as mi
from .base import TLIRIntegrator


class RadianceFieldPRB(TLIRIntegrator):
    """
    A differentiable integrator for emissive volumes using regular ray marching.
    
    This integrator implements a NeRF-like approach for reconstructing 3D scenes
    using density and spherical harmonics coefficient grids.
    """
    
    def __init__(self, props=mi.Properties(), bbox=None, use_relu=True, 
                 grid_res=16, sh_degree=2, initial_density=0.01, initial_sh=0.1):
        """
        Initialize the RadianceFieldPRB integrator.
        
        Args:
            props: Mitsuba properties
            bbox: Bounding box for the volume (default: [0,0,0] to [1,1,1])
            use_relu: Whether to apply ReLU to density values
            grid_res: Initial grid resolution
            sh_degree: Spherical harmonics degree
            initial_density: Initial density value
            initial_sh: Initial SH coefficient value
        """
        super().__init__(props)
        self.bbox = bbox if bbox is not None else mi.ScalarBoundingBox3f([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        self.use_relu = use_relu
        self.grid_res = grid_res
        self.sh_degree = sh_degree
        
        # Initialize the 3D texture for the density and SH coefficients
        res = self.grid_res
        self.sigmat = mi.Texture3f(dr.full(mi.TensorXf, initial_density, shape=(res, res, res, 1)))
        self.sh_coeffs = mi.Texture3f(dr.full(mi.TensorXf, initial_sh, shape=(res, res, res, 3 * (sh_degree + 1) ** 2)))

    def eval_emission(self, pos, direction): 
        """Evaluate directionally varying emission using spherical harmonics."""
        spec = mi.Spectrum(0)
        sh_dir_coef = dr.sh_eval(direction, self.sh_degree)
        sh_coeffs = self.sh_coeffs.eval(pos)
        for i, sh in enumerate(sh_dir_coef):
            spec += sh * mi.Spectrum(sh_coeffs[3 * i:3 * (i + 1)])
        return dr.clip(spec, 0.0, 1.0)

    @dr.syntax
    def sample(self, mode, scene, sampler,
               ray, δL, state_in, active, **kwargs):
        """
        Main ray marching implementation.

        Returns the radiance along a single input ray using regular ray marching.

        AOVs:
            [0]: Expected ray depth (distance along ray)
        """
        primal = mode == dr.ADMode.Primal

        # Extract stochastic preconditioning alpha from kwargs
        spn_alpha = kwargs.get('spn_alpha', 0.0)

        # Extract point-dependent loss functions
        point_loss_fns = kwargs.get('point_loss_fns', None)

        ray = mi.Ray3f(ray)
        hit, mint, maxt = self.bbox.ray_intersect(ray)

        active = mi.Bool(active)
        active &= hit  # ignore rays that miss the bbox
        if not primal:  # if the gradient is zero, stop early
            active &= dr.any(δL != 0)

        step_size = mi.Float(1.0 / self.grid_res)
        t = mi.Float(mint) + sampler.next_1d(active) * step_size
        L = mi.Spectrum(0.0 if primal else state_in)
        δL = mi.Spectrum(δL if δL is not None else 0)
        β = mi.Spectrum(1.0) # throughput

        # Depth accumulator (expected distance along ray)
        depth = mi.Float(0.0)

        while active:
            p = ray(t)

            # Apply stochastic preconditioning: add Gaussian noise to query point
            if spn_alpha > 0.0:
                noise = mi.Vector3f(
                    sampler.next_1d(active) * 2.0 - 1.0,
                    sampler.next_1d(active) * 2.0 - 1.0,
                    sampler.next_1d(active) * 2.0 - 1.0
                )
                # Scale noise by alpha and grid resolution (noise in world space)
                noise_scale = spn_alpha / self.grid_res
                p = p + noise * noise_scale

            with dr.resume_grad(when=not primal):
                sigmat = self.sigmat.eval(p)[0]
                if self.use_relu:
                    sigmat = dr.maximum(sigmat, 0.0)
                tr = dr.exp(-sigmat * step_size)
                # Evaluate the directionally varying emission (weighted by transmittance)
                Le = β * (1.0 - tr) * self.eval_emission(p, ray.d)

                # Accumulate expected depth
                # Use mean throughput for depth weighting
                β_mean = dr.mean(β)
                depth_contrib = t * (1.0 - tr) * β_mean

            β *= tr
            L = L + Le if primal else L - dr.detach(Le)

            with dr.resume_grad(when=not primal):
                if not primal:
                    # Propagate gradients using helper functions
                    self.propagate_radiance_gradient(δL, L, tr, Le)

                    # TODO: Point-dependent losses can be added here if needed
                    # Example:
                    # if point_loss_fns is not None:
                    #     point_data = {...}
                    #     self.propagate_point_losses(point_loss_fns, point_data)

            t += step_size
            active &= (t < maxt) & dr.any(β != 0.0)

        return L if primal else δL, mi.Bool(True), [], L

    def compute_opacity_gradient_term(self, opacity, tr, Le):
        """
        Compute ∂(output_opacity)/∂(current_sample) for radiative backprop.

        For opacity with unit emission, this is:
        ∂opacity/∂σ = opacity * tr / detach(tr) + Le

        Args:
            opacity: Accumulated opacity so far
            tr: Transmittance at current sample (exp(-σ*Δt))
            Le: Emission at current sample (default 1.0 for opacity)

        Returns:
            Gradient term to multiply by δopacity
        """
        return opacity * tr / dr.detach(tr) + Le

    def propagate_opacity_gradient(self, δopacity, opacity, tr, Le):
        """
        Propagate opacity gradient through backward pass.

        Args:
            δopacity: Gradient w.r.t. output opacity (or None)
            opacity: Accumulated opacity so far
            tr: Transmittance at current sample
            Le: Emission at current sample (default 1.0)

        Returns:
            None (accumulates gradients on parameters)
        """
        if δopacity is not None:
            opacity_grad_term = self.compute_opacity_gradient_term(opacity, tr, Le)
            dr.backward_from(δopacity * opacity_grad_term)

    def compute_radiance_gradient_term(self, L, tr, Le):
        """
        Compute ∂(output_radiance)/∂(current_sample) for radiative backprop.

        This is the same formula as throughput, but with arbitrary emission Le:
        ∂L/∂σ = L * tr / detach(tr) + Le

        Args:
            L: Accumulated radiance so far
            tr: Transmittance at current sample (exp(-σ*Δt))
            Le: Emission at current sample

        Returns:
            Gradient term to multiply by δL
        """
        return L * tr / dr.detach(tr) + Le

    def propagate_radiance_gradient(self, δL, L, tr, Le):
        """
        Propagate radiance gradient through backward pass.

        Args:
            δL: Gradient w.r.t. output radiance (or None)
            L: Accumulated radiance so far
            tr: Transmittance at current sample
            Le: Emission at current sample

        Returns:
            None (accumulates gradients on parameters)
        """
        if δL is not None:
            radiance_grad_term = self.compute_radiance_gradient_term(L, tr, Le)
            dr.backward_from(δL * radiance_grad_term)

    @dr.syntax
    def sample_aovs(self, mode, scene, sampler, ray, δaovs, state_in, active, **kwargs):
        """
        Render AOVs (throughput, depth, normals) with radiative backpropagation.

        Throughput uses unit emission (Le=1) everywhere, accumulated via:
        L_throughput = Σ β(t) * (1 - e^(-σΔt)) * 1.0

        Backward pass propagates gradients using radiative backprop with Le=1.
        """
        primal = mode == dr.ADMode.Primal

        # Extract stochastic preconditioning alpha
        spn_alpha = kwargs.get('spn_alpha', 0.0)

        # Extract gradients from dict (if in backward mode)
        δopacity = None
        δdepth = None
        δnormal = None
        if δaovs is not None and isinstance(δaovs, dict):
            if δaovs.get('opacity') is not None:
                δopacity = mi.Float(δaovs['opacity'])
            if δaovs.get('depth') is not None:
                δdepth = mi.Float(δaovs['depth'])
            if δaovs.get('normal') is not None:
                δnormal = mi.Vector3f(δaovs['normal'])

        # Check if any gradients are non-zero (for early termination)
        has_grad = True
        if not primal:
            has_grad = mi.Bool(False)
            if δopacity is not None:
                has_grad |= dr.any(δopacity != 0)
            if δdepth is not None:
                has_grad |= dr.any(δdepth != 0)
            if δnormal is not None:
                has_grad |= dr.any(δnormal != 0)

        # Setup ray-bbox intersection with early termination
        ray, mint, maxt, active = self._setup_ray_bbox_intersection(ray, active, primal, has_grad)

        step_size = mi.Float(1.0 / self.grid_res)
        t = mi.Float(mint) + sampler.next_1d(active) * step_size

        # Initialize opacity and state
        if primal:
            opacity = mi.Float(0.0)
        else:
            # Unpack state from forward pass
            opacity = state_in.get('opacity', mi.Float(0.0)) if isinstance(state_in, dict) else mi.Float(0.0)

        β = mi.Float(1.0)  # throughput along ray

        # AOV accumulators
        depth = mi.Float(0.0)
        normal_accum = mi.Vector3f(0.0)

        while active:
            p = ray(t)
            p_query = self._apply_spn_noise(p, spn_alpha, sampler, active)

            with dr.resume_grad(when=not primal):
                sigmat = self._eval_density(p_query)
                tr = dr.exp(-sigmat * step_size)

                # Unit emission contribution (Le = 1)
                Le_aov = β * (1.0 - tr) * 1.0

                # AOV contributions (depth and normal)
                weight = (1.0 - tr) * β
                depth_contrib, normal_contrib = self._compute_aov_contributions(t, weight, p_query)

            # Accumulate (forward: add, backward: subtract)
            if primal:
                depth = depth + depth_contrib
                normal_accum = normal_accum + normal_contrib
            else:
                depth = depth - depth_contrib
                normal_accum = normal_accum - normal_contrib

            β *= tr
            opacity = opacity + Le_aov if primal else opacity - dr.detach(Le_aov)

            # BACKWARD PASS: Propagate gradients using helper functions
            with dr.resume_grad(when=not primal):
                if not primal:
                    # Propagate gradients using convenience helpers
                    self.propagate_opacity_gradient(δopacity, opacity, tr, Le_aov)
                    self.propagate_depth_gradient(δdepth, depth_contrib)
                    self.propagate_normal_gradient(δnormal, normal_contrib, state_in)

                    # TODO: Point-dependent losses can be added here if needed

            t += step_size
            active &= (t < maxt) & (β != 0.0)

        # Normalize normal
        normal_out, normal_accum_length = self._normalize_normal_output(normal_accum)

        # Return dict of AOVs (in primal) or gradients (in backward)
        if primal:
            aovs_out = {
                'opacity': opacity,
                'depth': depth,
                'normal': normal_out
            }
            # State for backward pass
            state_out = {
                'opacity': opacity,
                'normal_out': normal_out,
                'normal_accum_length': normal_accum_length
            }
        else:
            aovs_out = {
                'opacity': δopacity if δopacity is not None else None,
                'depth': δdepth if δdepth is not None else None,
                'normal': δnormal if δnormal is not None else None
            }
            state_out = {'opacity': opacity}

        return aovs_out, mi.Bool(True), state_out

    def traverse(self, cb):
        """Return differentiable parameters for optimization."""
        cb.put("sigmat", self.sigmat.tensor(), mi.ParamFlags.Differentiable)
        cb.put('sh_coeffs', self.sh_coeffs.tensor(), mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        """Update 3D textures when parameters change."""
        self.sigmat.update_inplace()
        self.sh_coeffs.update_inplace()
        self.grid_res = self.sigmat.shape[0]
    
mi.register_integrator("rf_prb", lambda props: RadianceFieldPRB(props))