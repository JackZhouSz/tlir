# TLIR Development Guidelines

## Code Documentation

### Documentation Style
- Keep docstrings concise and focused on essential information
- Avoid verbose explanations - prefer clarity over exhaustiveness
- Only document arguments that are actually used
- Remove outdated documentation when refactoring

### Docstring Structure
- One-line summary (what the function does)
- Brief description (if needed)
- Args: Only list arguments that exist in the signature
- Returns: Brief description of return values
- Skip lengthy examples unless they clarify complex usage

### Example
```python
def compute_loss(rendered, target):
    """
    Compute L2 loss between rendered and target values.

    Args:
        rendered: Rendered output
        target: Ground truth values

    Returns:
        Scalar loss value
    """
    return dr.mean(dr.square(rendered - target))
```

## Code Style
- Avoid numbering steps in comments (use bullets if needed)
- Remove redundant helper methods - inline when appropriate
- Keep parameter names consistent across related functions

## Loss Function Architecture

Loss functions are defined as methods on integrator classes, not passed as arguments.

### Loss Function Methods

Each integrator class should define these methods:

1. **`compute_color_loss(self, rendered, target)`** - Main color loss (required)
   - Default: L2 loss
   - Return: Scalar loss value

2. **`compute_aov_loss(self, aovs)`** - AOV losses (optional)
   - Default: Returns None
   - Return: Scalar loss value or None
   - AOVs dict contains: 'throughput', 'depth', 'normal'

3. **`compute_point_loss(self, point_data)`** - Point-dependent losses (optional)
   - Default: Returns None
   - Return: Scalar loss value or None
   - Applied at each sample point during backward pass

### Default AOV Losses

The default `compute_aov_loss()` implementation provides opacity and empty space losses:
- **Opacity loss**: Throughput should be 1 where objects exist (uses masks)
- **Empty space loss**: Throughput should be 0 in background (uses masks)
- Default weights: `opacity_weight=0.1`, `empty_space_weight=0.1`

Configure via:
```python
integrator.set_aov_loss_config(
    masks=masks,  # Binary mask (1=object, 0=background)
    opacity_weight=0.1,
    empty_space_weight=0.1
)
```

### Custom Loss Example

```python
class MyIntegrator(TLIRIntegrator):
    def compute_color_loss(self, rendered, target):
        # Custom color loss
        return dr.mean(dr.abs(rendered - target))

    def compute_aov_loss(self, aovs):
        # Call parent for default opacity/empty space losses
        base_loss = super().compute_aov_loss(aovs)

        # Add custom depth loss
        depth_loss = None
        if 'depth' in aovs:
            depth_loss = 0.1 * dr.mean(dr.square(aovs['depth']))

        # Combine losses
        if base_loss is not None and depth_loss is not None:
            return base_loss + depth_loss
        return base_loss if base_loss is not None else depth_loss

    def compute_point_loss(self, point_data):
        # Optional: Regularize normals
        if 'normal_at_p' in point_data:
            normal = point_data['normal_at_p']
            return dr.mean(dr.square(dr.norm(normal) - 1.0))
        return None
```

### Training Loop

```python
# Setup: integrator owns optimizer and parameters
integrator.initialize_optimizer(learning_rate=0.2)
integrator.setup_stochastic_preconditioning(starting_alpha=0.1, num_iterations=100)

# Training loop: integrator handles everything
for it in range(num_iterations):
    spn_alpha = integrator.get_spn_alpha()

    rendered, loss, aovs = integrator.render_rays_with_gradient(
        rays, target_colors, sampler, scene, spp=1, spn_alpha=spn_alpha
    )

    integrator.step_optimizer()
    integrator.post_step_update(config)  # Update SPN, apply constraints, etc.
```

## Optimizer Management

Integrators own their optimizer and parameters. This simplifies the training API.

### Core Optimization Methods

- **`initialize_optimizer(learning_rate=0.2)`** - Initialize Adam optimizer
- **`step_optimizer()`** - Step optimizer and update parameters
- **`post_step_update(config)`** - Update integrator state after step (SPN, constraints, etc.)
- **`upsample_parameters(factor=2)`** - Upsample grid resolution
- **`setup_stochastic_preconditioning(starting_alpha, num_iterations)`** - Setup SPN
- **`get_spn_alpha()`** - Get current SPN alpha value

### Customizing Trainable Parameters

Override `get_trainable_parameters()` to customize which parameters are optimized:

```python
class MyIntegrator(TLIRIntegrator):
    def get_trainable_parameters(self):
        if self.params is None:
            self.params = mi.traverse(self)

        # Only optimize sigmat, not sh_coeffs
        return {'sigmat': self.params['sigmat']}
```

## Rendering Methods

### render_camera

Render all rays from a camera sensor in one call:

```python
# Render full image with depth AOVs
rgb_image, aovs = integrator.render_camera(
    sensor=test_sensor,
    sampler=sampler,
    scene=scene,
    spp=16
)

# rgb_image: mi.TensorXf of shape (height, width, 3)
# aovs: List of AOV tensors [depth] if available
```

This is useful for evaluation and visualization.
