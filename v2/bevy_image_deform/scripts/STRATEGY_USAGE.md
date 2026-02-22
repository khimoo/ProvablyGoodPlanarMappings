# Strategy Pattern Usage Guide

## Overview

The refactored `deform_algo.py` now implements the Strategy Pattern for grid guarantee strategies, allowing runtime switching between two approaches:

- **Strategy 1**: Pessimistic, pre-guarantee approach
- **Strategy 2**: Optimistic, post-verification approach

## Usage Examples

### Using Strategy 2 (Default - Optimistic)

```python
from deform_algo import BetterFitwithGaussian, Strategy2

# Domain bounds: (x_min, x_max, y_min, y_max)
domain_bounds = (0.0, 100.0, 0.0, 100.0)

# Create with default Strategy 2 (interactive resolution 200x200)
mapper = BetterFitwithGaussian(
    domain_bounds=domain_bounds,
    s_param=1.0,
    K_solver=2.0,
    K_max=5.0
)

# Or explicitly pass Strategy 2
strategy = Strategy2(domain_bounds, interactive_resolution=200)
mapper = BetterFitwithGaussian(
    domain_bounds=domain_bounds,
    guarantee_strategy=strategy,
    s_param=1.0,
    K_solver=2.0,
    K_max=5.0
)
```

### Using Strategy 1 (Pessimistic)

```python
from deform_algo import BetterFitwithGaussian, Strategy1

domain_bounds = (0.0, 100.0, 0.0, 100.0)

# Create with Strategy 1 (assumes max coefficient norm of 100)
strategy = Strategy1(domain_bounds, max_expected_c_norm=100.0)
mapper = BetterFitwithGaussian(
    domain_bounds=domain_bounds,
    guarantee_strategy=strategy,
    s_param=1.0,
    K_solver=2.0,
    K_max=5.0
)
```

### Runtime Strategy Switching

```python
# Start with Strategy 2 for interactive performance
mapper = BetterFitwithGaussian(domain_bounds)

# Initialize with current strategy
src_handles = np.array([[10.0, 10.0], [50.0, 50.0], [90.0, 90.0]])
mapper.initialize_mapping(src_handles)

# ... perform interactive dragging ...

# Switch to Strategy 1 for guaranteed safety
new_strategy = Strategy1(domain_bounds, max_expected_c_norm=150.0)
mapper.strategy = new_strategy

# Re-initialize with new strategy
mapper.initialize_mapping(src_handles)
```

## Strategy Comparison

| Aspect | Strategy 1 | Strategy 2 |
|--------|-----------|-----------|
| **Initial Grid** | Very fine (strict) | Coarse (fast) |
| **Drag Performance** | Slower | Faster |
| **Post-Drag Verification** | Not needed | May refine grid |
| **Use Case** | Guaranteed safety | Interactive responsiveness |
| **Parameter** | `max_expected_c_norm` | `interactive_resolution` |

## Key Methods

### GuaranteeStrategy Interface

```python
# Get initial grid spacing for the current strategy
h = strategy.get_initial_h(basis, K_solver, K_max)

# Get strict h after drag (may trigger grid refinement)
strict_h = strategy.get_strict_h_after_drag(basis, K_solver, K_max, coefficients)

# Generate grid points at spacing h
grid = strategy.generate_grid(h)
```

### BetterFitwithGaussian Workflow

```python
# 1. Initialize mapping (called once or when handles change)
mapper.initialize_mapping(src_handles)

# 2. On mouse down (optional pre-computation)
mapper.start_drag(src_handles)

# 3. On mouse move (called every frame during drag)
mapper.update_drag(target_handles, num_iterations=2)

# 4. On mouse up (verification and optional grid refinement)
grid_was_refined = mapper.end_drag(target_handles)
```

## Implementation Details

### Strategy 1 Behavior

- Computes the strictest possible `h` upfront using worst-case coefficient norm
- Grid remains constant throughout interaction
- `get_strict_h_after_drag()` returns `inf` (no refinement needed)
- Suitable when you want mathematical guarantees from the start

### Strategy 2 Behavior

- Uses a coarse interactive grid for fast feedback
- After drag ends, computes actual coefficient norm from the solution
- If actual `h` is stricter than current, refines grid and re-optimizes
- Suitable for interactive applications where responsiveness matters

## Extending with New Strategies

To add a new strategy (e.g., adaptive quadtree-based grid):

```python
class Strategy3(GuaranteeStrategy):
    """Adaptive quadtree-based grid refinement"""
    
    def get_initial_h(self, basis, K_solver, K_max):
        # Return initial coarse h
        pass
    
    def get_strict_h_after_drag(self, basis, K_solver, K_max, c):
        # Compute adaptive h based on local distortion
        pass
```

Then use it the same way:

```python
strategy = Strategy3(domain_bounds)
mapper = BetterFitwithGaussian(domain_bounds, guarantee_strategy=strategy)
```
