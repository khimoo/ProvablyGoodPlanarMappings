# Architecture: Provably Good Planar Mappings

## Overview

This implementation follows the paper "Provably Good Planar Mappings" (Poranne & Lipman, 2014). The key insight is that **the algorithm is meshless** - it uses smooth basis functions to define a continuous deformation mapping.

## Core Concept

The deformation is defined as:

```
f(x) = Σ c_i * φ_i(x)
```

Where:
- `φ_i(x)` are Gaussian RBF basis functions: `exp(-||x - center_i||² / (2s²))`
- `c_i` are 2D coefficient vectors computed by the optimization algorithm
- The last 3 basis functions are affine terms: `[1, x, y]`

## Architecture Layers

### 1. Python Backend (`deform_algo.py`)

**Responsibilities:**
- Compute deformation coefficients with provable distortion bounds
- Maintain collocation grid for distortion constraints
- Implement Local-Global solver (Algorithm 1 from paper)
- Verify distortion bounds using Strategy 2

**Key Classes:**
- `GaussianRBF`: Basis function implementation
- `Strategy2`: Grid management and distortion verification
- `BetterFitwithGaussian`: Main solver with drag optimization

**Performance Optimizations:**
- Precomputes invariant matrices (B_mat, H_term) during initialization
- Uses active set method (only constrains points with high distortion)
- Caches active indices between frames for smooth dragging

### 2. Python Bridge (`bevy_bridge.py`)

**Responsibilities:**
- Interface between Rust and Python
- Create visualization mesh (dense grid)
- Evaluate mapping on mesh vertices
- Manage control points

**Key Methods:**
- `load_image_and_generate_mesh()`: Initialize domain and create 50x50 mesh
- `add_control_point()`: Add control point during setup
- `finalize_setup()`: Build solver with basis functions
- `start_drag_operation()`: Precompute matrices (onMouseDown)
- `update_control_point()`: Update position (onMouseMove)
- `solve_frame()`: Solve and return deformed mesh vertices
- `end_drag_operation()`: Verify bounds (onMouseUp)

### 3. Rust Frontend (`main.rs`)

**Responsibilities:**
- Render deformed mesh with texture
- Handle user input (mouse, keyboard)
- Manage application state (Setup/Deform modes)
- Communicate with Python via async channels

**State Machine:**
```
Setup Mode:
  - Click to add control points
  - Press Enter to finalize

Deform Mode:
  - Drag control points to deform
  - Press R to reset
```

## Data Flow

### Setup Phase

```
User clicks → Rust captures position → Python stores control point
                                    ↓
User presses Enter → Rust sends FinalizeSetup → Python builds solver
                                               ↓
                                    Creates basis functions at control points
                                    Generates collocation grid
                                    Precomputes invariant matrices
```

### Deformation Phase

```
Mouse Down → Rust sends StartDrag → Python precomputes matrices
                                  ↓
Mouse Move → Rust sends UpdatePoint → Python updates position
                                    → Python solves (2 iterations)
                                    → Python evaluates f(mesh_vertices)
                                    → Rust receives deformed vertices
                                    → Rust updates mesh and renders
                                  ↓
Mouse Up → Rust sends EndDrag → Python verifies distortion (Strategy 2)
                              → May refine grid if needed
                              → Final solve with refined grid
```

## Why Hybrid Approach?

**Theoretical Purity:**
- Send coefficients to Rust
- Evaluate Gaussian RBF in Rust for each pixel
- Mathematically elegant

**Practical Reality:**
- Evaluating Gaussian RBF for every pixel is expensive
- Python/NumPy is highly optimized for batch operations
- Sending ~2500 vertices (50x50 mesh) is negligible

**Our Solution:**
- Python computes coefficients (with provable guarantees)
- Python evaluates mapping on visualization mesh
- Rust renders the mesh (fast GPU operations)

This gives us:
✓ Theoretical guarantees from the paper
✓ Interactive performance (60+ FPS)
✓ Smooth, high-quality deformations

## Key Differences from Mesh-Based Methods

| Aspect | Mesh-Based | Our Meshless Approach |
|--------|------------|----------------------|
| Representation | Triangle mesh | Continuous function |
| Smoothness | C⁰ (piecewise linear) | C^∞ (Gaussian basis) |
| Distortion Control | Per-triangle | Per-collocation-point |
| Refinement | Subdivide triangles | Refine collocation grid |
| Visualization | Mesh is the model | Mesh is just for display |

## Distortion Guarantees

The algorithm ensures:

1. **Local Injectivity**: No fold-overs (det(J) > 0)
2. **Bounded Isometric Distortion**: K_solver ≤ 2.0 on collocation points
3. **Global Distortion Bound**: K_max ≤ 5.0 everywhere (via Strategy 2)

Where distortion is measured by singular values:
```
K = max(σ_max, 1/σ_min)
```

## Performance Characteristics

- **Setup**: O(N³) where N = number of control points (typically < 20)
- **Drag Start**: O(M²N) where M = collocation grid size (~40,000 points)
- **Drag Update**: O(A²N) where A = active set size (typically < 1000)
- **Mesh Evaluation**: O(V*N) where V = mesh vertices (2500)

Interactive performance is achieved by:
1. Precomputing invariant matrices
2. Using active set (sparse constraints)
3. Caching active indices between frames
4. Running only 2 iterations during drag

## Future Optimizations

Potential improvements:
1. GPU evaluation of basis functions (compute shader)
2. Adaptive mesh resolution based on deformation
3. Parallel evaluation of collocation points
4. Incremental SVD for active set updates
