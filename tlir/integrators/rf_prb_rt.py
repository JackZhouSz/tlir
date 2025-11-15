import drjit as dr
import mitsuba as mi
from .base import TLIRIntegrator


class RadianceFieldPRBRT(TLIRIntegrator):
    """
    A differentiable integrator for emissive volumes using ratio tracking.
    
    This integrator uses ratio tracking for more efficient sampling of volume
    interactions, particularly useful for high-density volumes.
    """
    
    def __init__(self, props=mi.Properties(), bbox=None, use_relu=True, 
                 grid_res=16, sh_degree=2, initial_density=0.01, initial_sh=0.1,
                 initial_majorant=10.0, stopgrad_density=False, min_step_size=1e-4,
                 min_throughput=1e-6, max_num_steps=10000):
        """
        Initialize the RadianceFieldPRBRT integrator.
        
        Args:
            props: Mitsuba properties
            bbox: Bounding box for the volume (default: [0,0,0] to [1,1,1])
            use_relu: Whether to apply ReLU to density values
            grid_res: Initial grid resolution
            sh_degree: Spherical harmonics degree
            initial_density: Initial density value
            initial_sh: Initial SH coefficient value
            initial_majorant: Initial majorant value for ratio tracking
            stopgrad_density: Whether to stop gradients on density
            min_step_size: Minimum step size for ray marching
            min_throughput: Minimum throughput threshold
            max_num_steps: Maximum number of ray marching steps
        """
        super().__init__(props)
        self.bbox = bbox if bbox is not None else mi.ScalarBoundingBox3f([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        self.use_relu = use_relu
        self.grid_res = grid_res
        self.sh_degree = sh_degree
        self.stopgrad_density = stopgrad_density
        self.min_step_size = min_step_size
        self.min_throughput = min_throughput
        self.max_num_steps = max_num_steps
        
        # Initialize the 3D texture for the density and SH coefficients
        res = self.grid_res
        self.sigmat = mi.Texture3f(dr.full(mi.TensorXf, initial_density, shape=(res, res, res, 1)))
        self.sh_coeffs = mi.Texture3f(dr.full(mi.TensorXf, initial_sh, shape=(res, res, res, 3 * (sh_degree + 1) ** 2)))
        # Grid-based majorant for ratio tracking
        self.majorant_grid = mi.Texture3f(dr.full(mi.TensorXf, initial_majorant, shape=(res, res, res, 1)))

    def eval_emission(self, pos, direction): 
        """Evaluate directionally varying emission using spherical harmonics."""
        spec = mi.Spectrum(0)
        sh_dir_coef = dr.sh_eval(direction, self.sh_degree)
        sh_coeffs = self.sh_coeffs.eval(dr.clip(pos, 0.0, 1.0))
        for i, sh in enumerate(sh_dir_coef):
            spec += sh * mi.Spectrum(sh_coeffs[3 * i:3 * (i + 1)])
        return dr.clip(spec, 0.0, 1.0)

    @dr.syntax
    def sample(self, mode, scene, sampler,
               ray, δL, state_in, active, **kwargs):
        """
        Main ratio tracking implementation.

        Returns the radiance along a single input ray using ratio tracking.
        """
        primal = mode == dr.ADMode.Primal

        # Extract stochastic preconditioning alpha from kwargs
        spn_alpha = kwargs.get('spn_alpha', 0.0)

        # Setup ray-bbox intersection with early termination
        has_grad = dr.any(δL != 0) if not primal else True
        ray, mint, maxt, active = self._setup_ray_bbox_intersection(ray, active, primal, has_grad)

        L = mi.Spectrum(0.0 if primal else state_in)
        δL = mi.Spectrum(δL if δL is not None else 0)
        Tr = mi.Float(1.0)  # throughput

        # Ratio tracking: sample distances using the majorant
        t = mi.Float(mint)
        num_steps = mi.Int32(0)

        # Accumulated transmittance gradient
        trans_grad_buffer = mi.Float(0.0)

        while active:
            # Sample next interaction distance using majorant
            u = dr.clip(sampler.next_1d(active), 0.0, 1.0 - 1e-6)

            # Get majorant at current position
            p = ray(t)
            p_query = self._apply_spn_noise(p, spn_alpha, sampler, active)
            majorant = self.majorant_grid.eval(dr.clip(p_query, 0.0, 1.0))[0]

            # Compute step size
            dt = dr.maximum(-dr.log(1.0 - u) / majorant, self.min_step_size)
            t += dt
            num_steps += 1

            # Check if we've exited the volume
            active &= (t < maxt)
            active &= (num_steps < self.max_num_steps)

            # Update ray position and evaluate density
            p = ray(t)
            p_query = self._apply_spn_noise(p, spn_alpha, sampler, active)

            with dr.resume_grad(when=not primal):
                # Get majorant and actual extinction coefficient
                majorant = self.majorant_grid.eval(dr.clip(p_query, 0.0, 1.0))[0]
                sigmat = self._eval_density(p_query, use_majorant=True, majorant=majorant)

                if self.stopgrad_density:
                    sigmat = dr.detach(sigmat)
                
                # Ratio tracking: probability of interaction = σ / σ_majorant
                interaction_prob = sigmat / majorant
                
                # Only emit when should_interact is true
                should_interact = sampler.next_1d(active) < interaction_prob
                interaction_mask = dr.select(should_interact, 1.0, 0.0)

                if should_interact:
                    Le = self.eval_emission(p_query, ray.d)
                else:
                    Le = mi.Spectrum(0.0)

                    # Transmittance gradient
                    trans_grad_buffer += sigmat

            # L = L + Le if primal else L - Le
            L = Le

            with dr.resume_grad(when=not primal):
                if not primal:
                    # Propagate radiance gradient using helper
                    self.propagate_radiance_gradient_rt(δL, sigmat, trans_grad_buffer, interaction_mask, Le)

                    # TODO: Point-dependent losses can be added here if needed

            if should_interact:
                trans_grad_buffer = mi.Float(0.0)

            # Update transmittance
            Tr *= (1 - interaction_prob)
            Tr = dr.detach(Tr)

            # Stop if we've hit a particle
            active &= ~should_interact
            
            # Stop if throughput becomes too small
            active &= dr.any(mi.Spectrum(Tr) > self.min_throughput)

        return L if primal else δL, mi.Bool(True), [], L

    def compute_radiance_gradient_term_rt(self, sigmat, trans_grad_buffer, interaction_mask, Le):
        """
        Compute ∂(output_radiance)/∂(current_sample) for ratio tracking.

        General formula that works for both opacity (Le=1.0) and radiance (arbitrary Le).

        Args:
            sigmat: Extinction coefficient at current sample
            trans_grad_buffer: Accumulated transmittance gradient
            interaction_mask: 1.0 if interaction occurred, 0.0 otherwise
            Le: Emission at current sample

        Returns:
            Gradient term to multiply by δL
        """
        return (sigmat * dr.detach(Le / sigmat) + Le -
                trans_grad_buffer * dr.detach(Le) * interaction_mask)

    def propagate_opacity_gradient_rt(self, δopacity, sigmat, trans_grad_buffer, interaction_mask, Le):
        """
        Propagate opacity gradient through backward pass (ratio tracking).

        Args:
            δopacity: Gradient w.r.t. output opacity (or None)
            sigmat: Extinction coefficient at current sample
            trans_grad_buffer: Accumulated transmittance gradient
            interaction_mask: 1.0 if interaction occurred, 0.0 otherwise
            Le: Emission at current sample (typically 1.0 for opacity)

        Returns:
            None (accumulates gradients on parameters)
        """
        if δopacity is not None:
            opacity_grad_term = self.compute_radiance_gradient_term_rt(
                sigmat, trans_grad_buffer, interaction_mask, Le
            )
            dr.backward_from(δopacity * opacity_grad_term)

    def propagate_radiance_gradient_rt(self, δL, sigmat, trans_grad_buffer, interaction_mask, Le):
        """
        Propagate radiance gradient through backward pass (ratio tracking).

        Handles both stopgrad_density cases:
        - If stopgrad_density: simple dr.backward_from(δL * Le)
        - Otherwise: full gradient with transmittance term

        Args:
            δL: Gradient w.r.t. output radiance (or None)
            sigmat: Extinction coefficient at current sample
            trans_grad_buffer: Accumulated transmittance gradient
            interaction_mask: 1.0 if interaction occurred, 0.0 otherwise
            Le: Emission at current sample

        Returns:
            None (accumulates gradients on parameters)
        """
        if δL is not None:
            if self.stopgrad_density:
                dr.backward_from(δL * Le)
            else:
                radiance_grad_term = self.compute_radiance_gradient_term_rt(
                    sigmat, trans_grad_buffer, interaction_mask, Le
                )
                dr.backward_from(δL * radiance_grad_term)

    @dr.syntax
    def sample_aovs(self, mode, scene, sampler, ray, δaovs, state_in, active, **kwargs):
        """Render AOVs using ratio tracking with radiative backpropagation."""
        primal = mode == dr.ADMode.Primal
        spn_alpha = kwargs.get('spn_alpha', 0.0)

        # Extract gradients from dict
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

        ray = mi.Ray3f(ray)
        hit, mint, maxt = self.bbox.ray_intersect(ray)

        active = mi.Bool(active)
        active &= hit
        if not primal:
            has_grad = mi.Bool(False)
            if δopacity is not None:
                has_grad |= dr.any(δopacity != 0)
            if δdepth is not None:
                has_grad |= dr.any(δdepth != 0)
            if δnormal is not None:
                has_grad |= dr.any(δnormal != 0)
            active &= has_grad

        # Initialize state
        if primal:
            opacity = mi.Float(0.0)
        else:
            opacity = state_in.get('opacity', mi.Float(0.0)) if isinstance(state_in, dict) else mi.Float(0.0)

        # AOV accumulators
        depth = mi.Float(0.0)
        normal_accum = mi.Vector3f(0.0)

        # Ratio tracking variables
        t = mi.Float(mint)
        num_steps = mi.Int32(0)
        trans_grad_buffer = mi.Float(0.0)

        while active:
            u = dr.clip(sampler.next_1d(active), 0.0, 1.0 - 1e-6)
            p = ray(t)
            p_query = self._apply_spn_noise(p, spn_alpha, sampler, active)

            majorant = self.majorant_grid.eval(dr.clip(p_query, 0.0, 1.0))[0]
            dt = dr.maximum(-dr.log(1.0 - u) / majorant, self.min_step_size)
            t += dt
            num_steps += 1

            active &= (t < maxt)
            active &= (num_steps < self.max_num_steps)

            p = ray(t)
            p_query = self._apply_spn_noise(p, spn_alpha, sampler, active)

            with dr.resume_grad(when=not primal):
                majorant = self.majorant_grid.eval(dr.clip(p_query, 0.0, 1.0))[0]
                sigmat = self._eval_density(p_query, use_majorant=True, majorant=majorant)

                if self.stopgrad_density:
                    sigmat = dr.detach(sigmat)

                interaction_prob = sigmat / majorant
                should_interact = sampler.next_1d(active) < interaction_prob
                interaction_mask = dr.select(should_interact, 1.0, 0.0)

                # Unit emission
                Le_aov = dr.select(should_interact, 1.0, 0.0)

                # Depth contribution
                depth_contrib = dr.select(should_interact, t * 1.0, 0.0)

                # Normal contribution
                density_grad = self.compute_density_gradient(p_query)
                normal_at_p = self.compute_normal_from_density_gradient(density_grad)
                normal_contrib = dr.select(should_interact, normal_at_p * 1.0, mi.Vector3f(0.0))

                if not should_interact:
                    trans_grad_buffer += sigmat

            # Accumulate
            if primal:
                depth = depth + depth_contrib
                normal_accum = normal_accum + normal_contrib
            else:
                depth = depth - depth_contrib
                normal_accum = normal_accum - normal_contrib

            opacity = Le_aov if primal else opacity - Le_aov

            # BACKWARD PASS
            with dr.resume_grad(when=not primal):
                if not primal and not self.stopgrad_density:
                    # Propagate gradients using convenience helpers
                    self.propagate_opacity_gradient_rt(δopacity, sigmat, trans_grad_buffer, interaction_mask, Le_aov)
                    self.propagate_depth_gradient(δdepth, depth_contrib)
                    self.propagate_normal_gradient(δnormal, normal_contrib, state_in)

                    # TODO: Point-dependent losses can be added here if needed

            if should_interact:
                trans_grad_buffer = mi.Float(0.0)

            active &= ~should_interact
            active &= (interaction_prob < 1.0 - self.min_throughput)

        # Normalize normal
        normal_out, normal_accum_length = self._normalize_normal_output(normal_accum)

        # Return dicts
        if primal:
            aovs_out = {
                'opacity': opacity,
                'depth': depth,
                'normal': normal_out
            }
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
        cb.put('majorant_grid', self.majorant_grid.tensor(), mi.ParamFlags.NonDifferentiable)

    def parameters_changed(self, keys):
        """Update 3D textures when parameters change."""
        self.sigmat.update_inplace()
        self.sh_coeffs.update_inplace()
        self.majorant_grid.update_inplace()
        self.grid_res = self.sigmat.shape[0]
        self.min_step_size = dr.max(self.bbox.extents()) / self.sigmat.shape[0] * 1e-2

mi.register_integrator("rf_prb_rt", lambda props: RadianceFieldPRBRT(props))