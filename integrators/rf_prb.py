import drjit as dr
import mitsuba as mi


class RadianceFieldPRB(mi.python.ad.integrators.common.RBIntegrator):
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
        """
        primal = mode == dr.ADMode.Primal
        
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
                
        while active:
            p = ray(t)
            with dr.resume_grad(when=not primal):
                sigmat = self.sigmat.eval(p)[0]
                if self.use_relu:
                    sigmat = dr.maximum(sigmat, 0.0)
                tr = dr.exp(-sigmat * step_size)
                # Evaluate the directionally varying emission (weighted by transmittance)
                Le = β * (1.0 - tr) * self.eval_emission(p, ray.d) 

            β *= tr
            L = L + Le if primal else L - Le

            with dr.resume_grad(when=not primal):
                if not primal:
                    dr.backward_from(δL * (L * tr / dr.detach(tr) + Le))

            t += step_size
            active &= (t < maxt) & dr.any(β != 0.0)

        return L if primal else δL, mi.Bool(True), [], L

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