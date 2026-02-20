"""
Bevy Bridge for Provably Good Planar Mappings

This bridge connects the Rust frontend (Bevy) with the Python backend (deform_algo.py).

Architecture:
- The algorithm is meshless (uses Gaussian RBF basis functions)
- Python computes deformation coefficients with provable distortion bounds
- Python sends mapping parameters (coefficients, centers, s) to Rust
- Rust evaluates f(x) for each pixel to render the deformed image

Workflow:
1. Rust extracts contour from PNG image
2. Rust sends contour to Python (domain boundary)
3. Python computes mapping with distortion guarantees
4. Python sends mapping parameters to Rust
5. Rust evaluates f(x) for each pixel and renders
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from deform_algo import BetterFitwithGaussian


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Ray casting algorithm to check if a point is inside a polygon.
    
    Args:
        point: (2,) array [x, y]
        polygon: (N, 2) array of polygon vertices
        
    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def filter_points_inside_contour(points: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Filter points to keep only those inside the contour.
    
    Args:
        points: (M, 2) array of points to filter
        contour: (N, 2) array of contour vertices
        
    Returns:
        (K, 2) array of points inside contour (K <= M)
    """
    mask = np.array([point_in_polygon(p, contour) for p in points])
    return points[mask]


class BevyBridge:
    """
    Bridge between Bevy (Rust) and the deformation algorithm (Python).
    
    Workflow:
    1. initialize_domain: Set image dimensions
    2. set_contour: Set domain boundary from image contour
    3. add_control_point: Add control points during setup phase
    4. finalize_setup: Build solver with all control points
    5. start_drag_operation: Called on mouse down (precomputation)
    6. update_control_point: Called on mouse move (update position)
    7. solve_frame: Solve and return mapping parameters
    8. end_drag_operation: Called on mouse up (Strategy 2 verification)
    """
    
    def __init__(self):
        self.solver: Optional[BetterFitwithGaussian] = None
        self.image_width: float = 0.0
        self.image_height: float = 0.0
        self.contour: Optional[np.ndarray] = None  # (N, 2) contour vertices
        self.control_points: List[Tuple[int, float, float]] = []  # (control_idx, x, y)
        self.is_setup_finalized: bool = False
        
    def initialize_domain(self, image_width: float, image_height: float, epsilon: float) -> None:
        """
        Initialize the deformation domain.
        
        Args:
            image_width: Width of the image
            image_height: Height of the image
            epsilon: Gaussian RBF parameter (s in the paper)
        """
        self.image_width = image_width
        self.image_height = image_height
        
        # Initialize solver with bounding box
        domain_bounds = (0.0, image_width, 0.0, image_height)
        self.solver = BetterFitwithGaussian(
            domain_bounds=domain_bounds,
            s_param=epsilon,
            K_solver=2.0,
            K_max=5.0
        )
        
        print(f"Initialized domain: {image_width}x{image_height}, epsilon={epsilon}")
    
    def set_contour(self, contour_points: List[Tuple[float, float]]) -> None:
        """
        Set the domain boundary from image contour.
        
        Args:
            contour_points: List of (x, y) coordinates defining the contour
        """
        self.contour = np.array(contour_points, dtype=np.float64)
        print(f"Set contour with {len(contour_points)} points")
    
    def add_control_point(self, control_index: int, x: float, y: float) -> None:
        """
        Add a control point during setup phase.
        
        Args:
            control_index: Index for tracking (not used internally, just for Rust side)
            x, y: Position in image coordinates
        """
        if self.is_setup_finalized:
            raise RuntimeError("Cannot add control points after finalize_setup")
        
        self.control_points.append((control_index, x, y))
        print(f"Added control point {control_index} at ({x:.1f}, {y:.1f})")
    
    def finalize_setup(self) -> None:
        """
        Finalize setup phase and initialize the solver.
        This builds the basis functions centered at control points.
        If contour is set, collocation points are filtered to be inside the contour.
        """
        if self.is_setup_finalized:
            return
        
        if len(self.control_points) == 0:
            raise RuntimeError("No control points added")
        
        if self.solver is None:
            raise RuntimeError("Solver not initialized")
        
        # Extract source handle positions
        src_handles = np.array(
            [[x, y] for (_, x, y) in self.control_points],
            dtype=np.float64
        )
        
        # Initialize mapping with identity (no deformation)
        self.solver.initialize_mapping(src_handles)
        
        # If contour is set, filter collocation points to be inside contour
        if self.contour is not None:
            original_count = len(self.solver.collocation_points)
            self.solver.collocation_points = filter_points_inside_contour(
                self.solver.collocation_points,
                self.contour
            )
            filtered_count = len(self.solver.collocation_points)
            print(f"Filtered collocation points: {original_count} -> {filtered_count}")
            
            # Recompute hessian term with filtered points
            self.solver._update_hessian_term()
        
        self.is_setup_finalized = True
        print(f"Finalized setup with {len(self.control_points)} control points")
    
    def start_drag_operation(self) -> None:
        """
        Called when user presses mouse button (onMouseDown).
        Performs heavy precomputation for fast dragging.
        """
        if not self.is_setup_finalized:
            raise RuntimeError("Setup not finalized")
        
        if self.solver is None:
            return
        
        # Extract current source positions
        src_handles = np.array(
            [[x, y] for (_, x, y) in self.control_points],
            dtype=np.float64
        )
        
        self.solver.start_drag(src_handles)
        print("Started drag operation")
    
    def update_control_point(self, control_index: int, x: float, y: float) -> None:
        """
        Update a control point position during drag (onMouseMove).
        
        Args:
            control_index: Index in the control_points list
            x, y: New position in image coordinates
        """
        if not self.is_setup_finalized:
            raise RuntimeError("Setup not finalized")
        
        if control_index < 0 or control_index >= len(self.control_points):
            raise ValueError(f"Invalid control_index: {control_index}")
        
        # Update the stored position
        idx, _, _ = self.control_points[control_index]
        self.control_points[control_index] = (idx, x, y)
    
    def solve_frame(self, inverse_grid_resolution: int = 64) -> Dict:
        """
        Solve for new deformation and return mapping parameters including inverse grid.
        
        Args:
            inverse_grid_resolution: Resolution of inverse mapping grid (default 64x64)
        
        Returns:
            Dictionary with:
            - coefficients: List[List[float]] - (2, N+3) coefficient matrix
            - centers: List[List[float]] - (N, 2) RBF centers
            - s_param: float - Gaussian width parameter
            - n_rbf: int - Number of RBF basis functions
            - image_width: float
            - image_height: float
            - inverse_grid: List[List[List[float]]] - (H, W, 2) inverse mapping grid
            - grid_width: int - Width of inverse grid
            - grid_height: int - Height of inverse grid
        """
        if not self.is_setup_finalized:
            raise RuntimeError("Setup not finalized")
        
        if self.solver is None:
            raise RuntimeError("Solver not initialized")
        
        # Extract target handle positions
        target_handles = np.array(
            [[x, y] for (_, x, y) in self.control_points],
            dtype=np.float64
        )
        
        # Run optimization (fast during drag, uses cached matrices)
        self.solver.update_drag(target_handles, num_iterations=2)
        
        # Compute inverse mapping grid
        inverse_grid = self._compute_inverse_grid(inverse_grid_resolution)
        
        # Return mapping parameters
        return {
            'coefficients': self.solver.coefficients.tolist(),  # (2, N+3)
            'centers': self.solver.basis.centers.tolist(),      # (N, 2)
            's_param': float(self.solver.basis.s),
            'n_rbf': len(self.control_points),
            'image_width': float(self.image_width),
            'image_height': float(self.image_height),
            'inverse_grid': inverse_grid.tolist(),  # (H, W, 2)
            'grid_width': inverse_grid.shape[1],
            'grid_height': inverse_grid.shape[0],
        }
    
    def _compute_inverse_grid(self, resolution: int) -> np.ndarray:
        """
        Compute inverse mapping f^{-1}(y) at a regular grid using Newton-Raphson.
        
        For each output pixel y, find x such that f(x) = y.
        
        Args:
            resolution: Grid resolution (will create resolution x resolution grid)
            
        Returns:
            (H, W, 2) array where [i, j] contains the source coordinates for output pixel (j, i)
        """
        if self.solver is None:
            raise RuntimeError("Solver not initialized")
        
        # Create output grid (where we want to sample from)
        y_coords = np.linspace(0, self.image_height, resolution)
        x_coords = np.linspace(0, self.image_width, resolution)
        Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Target positions (output space)
        target_positions = np.stack([X.ravel(), Y.ravel()], axis=1)  # (N, 2)
        
        # Initial guess: identity (x = y)
        source_positions = target_positions.copy()
        
        # Newton-Raphson iterations
        max_iterations = 10
        tolerance = 1e-3
        
        for iteration in range(max_iterations):
            # Evaluate forward mapping: f(x_current)
            mapped = self.solver.evaluate_map(source_positions)  # (N, 2)
            
            # Residual: f(x) - y
            residual = mapped - target_positions  # (N, 2)
            
            # Check convergence
            max_error = np.max(np.abs(residual))
            if max_error < tolerance:
                break
            
            # Evaluate Jacobian: J_f(x_current)
            jacobians = self.solver.evaluate_jacobian(source_positions)  # (N, 2, 2)
            
            # Solve: J * delta_x = -residual for each point
            # delta_x = -J^{-1} * residual
            try:
                # Compute inverse of each 2x2 Jacobian
                det = jacobians[:, 0, 0] * jacobians[:, 1, 1] - jacobians[:, 0, 1] * jacobians[:, 1, 0]
                det = np.where(np.abs(det) > 1e-8, det, 1e-8)  # Avoid division by zero
                
                inv_jac = np.zeros_like(jacobians)
                inv_jac[:, 0, 0] = jacobians[:, 1, 1] / det
                inv_jac[:, 0, 1] = -jacobians[:, 0, 1] / det
                inv_jac[:, 1, 0] = -jacobians[:, 1, 0] / det
                inv_jac[:, 1, 1] = jacobians[:, 0, 0] / det
                
                # delta_x = -J^{-1} @ residual
                delta = -np.einsum('nij,nj->ni', inv_jac, residual)
                
                # Update with damping for stability
                damping = 0.8
                source_positions += damping * delta
                
            except np.linalg.LinAlgError:
                print(f"Warning: Singular Jacobian at iteration {iteration}")
                break
        
        # Reshape to grid
        inverse_grid = source_positions.reshape(resolution, resolution, 2)
        
        return inverse_grid
    
    def end_drag_operation(self) -> bool:
        """
        Called when user releases mouse button (onMouseUp).
        Performs Strategy 2 verification and refinement if needed.
        
        Returns:
            True if grid was refined (requires re-solving), False otherwise
        """
        if not self.is_setup_finalized:
            raise RuntimeError("Setup not finalized")
        
        if self.solver is None:
            return False
        
        # Extract target handle positions
        target_handles = np.array(
            [[x, y] for (_, x, y) in self.control_points],
            dtype=np.float64
        )
        
        # Run Strategy 2 verification
        was_refined = self.solver.end_drag(target_handles)
        
        if was_refined:
            print("Grid was refined by Strategy 2")
            
            # Re-filter collocation points if contour is set
            if self.contour is not None:
                self.solver.collocation_points = filter_points_inside_contour(
                    self.solver.collocation_points,
                    self.contour
                )
                self.solver._update_hessian_term()
        
        return was_refined
    
    def reset_mesh(self) -> None:
        """
        Reset to identity mapping (no deformation).
        """
        self.control_points.clear()
        self.is_setup_finalized = False
        self.solver = None
        self.contour = None
        print("Reset mesh")

