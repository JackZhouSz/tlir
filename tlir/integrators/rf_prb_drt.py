import drjit as dr
import mitsuba as mi
from .base import TLIRIntegrator


class RadianceFieldPRBDRT(TLIRIntegrator):
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
        mi.Log(mi.LogLevel.Info, "Using RadianceFieldPRBDRT integrator.")
        primal = mode == dr.ADMode.Primal

        # Extract stochastic preconditioning alpha from kwargs
        spn_alpha = kwargs.get('spn_alpha', 0.0)

        ray = mi.Ray3f(ray)
        hit, mint, maxt = self.bbox.ray_intersect(ray)

        active = mi.Bool(active)
        active &= hit  # ignore rays that miss the bbox
        if not primal:  # if the gradient is zero, stop early
            active &= dr.any(δL != 0)

        L = mi.Spectrum(0.0 if primal else state_in)
        δL = mi.Spectrum(δL if δL is not None else 0)
        Tr = mi.Float(1.0)  # throughput
        
        # Ratio tracking: sample distances using the majorant
        t = mi.Float(mint)
        num_steps = mi.Int32(0)

        # Accumulated transmittance gradient
        trans_grad_buffer = mi.Float(0.0)

        # Reservoir sampling for DRT
        w_acc = mi.Float(0.0)
        reservoir_t = mi.Float(0.0)
        reservoir_dt = mi.Float(0.0)
        
        while active:
            # Sample next interaction distance using majorant
            # -log(1 - ξ) / σ_majorant where ξ is uniform random
            u = dr.clip(sampler.next_1d(active), 0.0, 1.0 - 1e-6)

            # Get current majorant value at current position
            p = ray(t)

            # Apply stochastic preconditioning to first query
            p_query = p
            if spn_alpha > 0.0:
                noise = mi.Vector3f(
                    sampler.next_1d(active) * 2.0 - 1.0,
                    sampler.next_1d(active) * 2.0 - 1.0,
                    sampler.next_1d(active) * 2.0 - 1.0
                )
                noise_scale = spn_alpha / self.grid_res
                p_query = p + noise * noise_scale

            majorant = self.majorant_grid.eval(dr.clip(p_query, 0.0, 1.0))[0]

            # Get steps size
            dt = dr.maximum(-dr.log(1.0 - u) / majorant, self.min_step_size)
            t += dt
            num_steps += 1

            # Check if we've exited the volume
            active &= (t < maxt)
            active &= (num_steps < self.max_num_steps)

            # Update ray position
            p = ray(t)

            # Apply stochastic preconditioning to second query
            p_query = p
            if spn_alpha > 0.0:
                noise = mi.Vector3f(
                    sampler.next_1d(active) * 2.0 - 1.0,
                    sampler.next_1d(active) * 2.0 - 1.0,
                    sampler.next_1d(active) * 2.0 - 1.0
                )
                noise_scale = spn_alpha / self.grid_res
                p_query = p + noise * noise_scale

            # Reservoir sampling for DRT
            w_step = Tr * dt
            w_acc += w_step

            should_update_reservoir = (sampler.next_1d(active) * w_acc) < w_step

            if should_update_reservoir:
                reservoir_t = t
                reservoir_dt = dt

            with dr.resume_grad(when=not primal):
                # Get actual extinction coefficient at this point
                majorant = self.majorant_grid.eval(dr.clip(p_query, 0.0, 1.0))[0]

                if self.use_relu:
                    sigmat = dr.clip(self.sigmat.eval(dr.clip(p_query, 0.0, 1.0))[0], 0.0, majorant)
                else:
                    sigmat = dr.minimum(self.sigmat.eval(dr.clip(p_query, 0.0, 1.0))[0], majorant)

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

            L = Le

            with dr.resume_grad(when=not primal):
                if not primal:
                    if self.stopgrad_density:
                        dr.backward_from(δL * Le)
                    else:
                        dr.backward_from(
                            δL * 
                            (
                                sigmat * dr.detach(Le * sigmat / (sigmat * sigmat + 1.0))
                                + Le
                                # - trans_grad_buffer * dr.detach(Le) * interaction_mask
                            )
                        )


            # Update transmittance
            Tr *= (1 - interaction_prob)
            Tr = dr.detach(Tr)

            # Stop if we've hit a particle
            active &= ~should_interact
            
            # Stop if throughput becomes too small
            active &= dr.any(mi.Spectrum(Tr) > self.min_throughput)

        # Compute transmittance gradient
        trans_grad_buffer = mi.Float(0.0)
        interval = t - mint
        n_samples = 1

        # Transmittance gradient
        for _ in range(n_samples):
            new_t = sampler.next_1d(mi.Bool(True)) * interval + mint
            new_p = ray(new_t)

            # Apply stochastic preconditioning to transmittance gradient query
            new_p_query = new_p
            if spn_alpha > 0.0:
                noise = mi.Vector3f(
                    sampler.next_1d(mi.Bool(True)) * 2.0 - 1.0,
                    sampler.next_1d(mi.Bool(True)) * 2.0 - 1.0,
                    sampler.next_1d(mi.Bool(True)) * 2.0 - 1.0
                )
                noise_scale = spn_alpha / self.grid_res
                new_p_query = new_p + noise * noise_scale

            with dr.resume_grad():
                majorant = self.majorant_grid.eval(dr.clip(new_p_query, 0.0, 1.0))[0]

                if self.use_relu:
                    sigmat = dr.clip(self.sigmat.eval(dr.clip(new_p_query, 0.0, 1.0))[0], 0.0, majorant)
                else:
                    sigmat = dr.minimum(self.sigmat.eval(dr.clip(new_p_query, 0.0, 1.0))[0], majorant)

                trans_grad_buffer += sigmat

        inv_pdf = interval / n_samples

        with dr.resume_grad():
            trans_grad_buffer = dr.select(active, trans_grad_buffer, 0.)
            dr.backward_from(-trans_grad_buffer * dr.detach(inv_pdf * L))

        # DRT gradient update
        with dr.resume_grad(when=not primal):
            if not self.stopgrad_density and not primal:
                t, dt = reservoir_t, reservoir_dt
                u = dr.clip(sampler.next_1d(mi.Bool(True)), 1e-6, 1.0 - 1e-6)
                t += u * dt

                p = ray(t)

                # Apply stochastic preconditioning to DRT gradient update query
                p_query = p
                if spn_alpha > 0.0:
                    noise = mi.Vector3f(
                        sampler.next_1d(mi.Bool(True)) * 2.0 - 1.0,
                        sampler.next_1d(mi.Bool(True)) * 2.0 - 1.0,
                        sampler.next_1d(mi.Bool(True)) * 2.0 - 1.0
                    )
                    noise_scale = spn_alpha / self.grid_res
                    p_query = p + noise * noise_scale

                Le = self.eval_emission(p_query, ray.d)

                majorant = self.majorant_grid.eval(dr.clip(p_query, 0.0, 1.0))[0]

                if self.use_relu:
                    sigmat = dr.clip(self.sigmat.eval(dr.clip(p_query, 0.0, 1.0))[0], 0.0, majorant)
                else:
                    sigmat = dr.minimum(self.sigmat.eval(dr.clip(p_query, 0.0, 1.0))[0], majorant)

                dr.backward_from(
                    δL * (
                        sigmat * dr.detach(w_acc * Le / (sigmat * sigmat + 1.0))
                    )
                )

        return L if primal else δL, mi.Bool(True), [], L

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

mi.register_integrator("rf_prb_drt", lambda props: RadianceFieldPRBDRT(props))