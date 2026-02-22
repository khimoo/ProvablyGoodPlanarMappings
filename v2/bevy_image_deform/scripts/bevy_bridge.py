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
    1. initialize_domain: Set image dimensions with strategy selection
    2. set_contour: Set domain boundary from image contour
    3. add_control_point: Add control points during setup phase
    4. finalize_setup: Build solver with all control points
    5. start_drag_operation: Called on mouse down (precomputation)
    6. update_control_point: Called on mouse move (update position)
    7. solve_frame: Solve and return mapping parameters
    8. end_drag_operation: Called on mouse up (verification)
    """

    def __init__(self, strategy_type: str = "strategy2",
                 strategy_params: Optional[Dict] = None):
        """
        Args:
            strategy_type: "strategy1" or "strategy2"
            strategy_params: Strategy 固有のパラメータ
                - strategy1:
                    - collocation_resolution: int
                    - K_on_collocation: float (コロケーション点上の K)
                - strategy2:
                    - interactive_resolution: int
                    - K_solver: float (コロケーション点上の K)
                    - K_max: float (目標 K_max)
        """
        self.strategy_type = strategy_type
        self.strategy_params = strategy_params or {}
        self.solver: Optional[BetterFitwithGaussian] = None
        self.image_width: float = 0.0
        self.image_height: float = 0.0
        self.contour: Optional[np.ndarray] = None  # (N, 2) contour vertices
        self.control_points: List[Tuple[int, float, float]] = []  # (control_idx, x, y)
        self.is_setup_finalized: bool = False
        self.guaranteed_K_max: Optional[float] = None  # Strategy 1 の出力

    def initialize_domain(self, image_width: float, image_height: float, epsilon: float) -> None:
        """
        Initialize the deformation domain with specified strategy.

        Args:
            image_width: Width of the image
            image_height: Height of the image
            epsilon: Gaussian RBF parameter (s in the paper)
        """
        self.image_width = image_width
        self.image_height = image_height
        
        # epsilon を strategy_params に保存（_create_strategy1 で使用）
        self.strategy_params['epsilon'] = epsilon

        domain_bounds = (0.0, image_width, 0.0, image_height)

        if self.strategy_type == "strategy1":
            strategy, K_solver, K_max = self._create_strategy1(domain_bounds)
        else:  # strategy2
            strategy, K_solver, K_max = self._create_strategy2(domain_bounds)

        self.solver = BetterFitwithGaussian(
            domain_bounds=domain_bounds,
            guarantee_strategy=strategy,
            s_param=epsilon,
            K_solver=K_solver,
            K_max=K_max
        )

        print(f"Initialized domain: {image_width}x{image_height}, epsilon={epsilon}")
        print(f"Strategy: {self.strategy_type}")
        if self.strategy_type == "strategy1":
            print(f"  K_on_collocation: {K_solver}")
            print(f"  (K_max will be computed after finalize_setup)")
        else:
            print(f"  K_solver: {K_solver}, K_max: {K_max}")

    def _create_strategy1(self, domain_bounds: Tuple[float, float, float, float]):
        """
        Create Strategy 1: Given Z and K → bound K_max
        
        K を自動計算する場合:
        - 'K_on_collocation' が指定されていない場合、論文の式(11)から自動計算
        - 'collocation_resolution' から h を計算
        - ω(h) = 2 * h / s² (Gaussians の場合)
        - K_auto = α / ω(h) (α < 1, 安全マージン)
        
        Returns:
            (strategy, K_solver, K_max)
            K_max は Strategy 2 との互換性のため渡すが、Strategy 1 では使用しない
        """
        from deform_algo import Strategy1
        
        collocation_resolution = self.strategy_params.get(
            'collocation_resolution', 500
        )
        epsilon = self.strategy_params.get('epsilon', 40.0)
        
        print(f"DEBUG: strategy_params = {self.strategy_params}")
        print(f"DEBUG: 'K_on_collocation' in strategy_params = {'K_on_collocation' in self.strategy_params}")
        
        # K_on_collocation が明示的に指定されているか確認
        if 'K_on_collocation' in self.strategy_params:
            K_on_collocation = self.strategy_params['K_on_collocation']
            print(f"Using explicit K_on_collocation: {K_on_collocation}")
        else:
            # K を自動計算
            x_min, x_max, y_min, y_max = domain_bounds
            max_span = max(x_max - x_min, y_max - y_min)
            
            # h = max_span / collocation_resolution
            h = max_span / collocation_resolution
            
            # ω(h) = 2 * h / s² (Gaussians の場合、初期状態で |||c||| = 1)
            omega_h = 2.0 * h / (epsilon ** 2)
            
            # 条件: 1/K > ω(h) ⟹ K < 1/ω(h)
            # 安全マージン β = 0.5 を適用: K_auto = β / ω(h)
            # （β を小さくすることで、式(11)の第2項 1/(1/K - ω(h)) が爆発するのを防ぐ）
            safety_margin = 0.5
            K_on_collocation = safety_margin / omega_h
            
            print(f"Strategy 1: Auto-calculated K_on_collocation")
            print(f"  max_span: {max_span:.2f}")
            print(f"  collocation_resolution: {collocation_resolution}")
            print(f"  h: {h:.4f}")
            print(f"  epsilon (s): {epsilon:.2f}")
            print(f"  ω(h): {omega_h:.6f}")
            print(f"  K_on_collocation: {K_on_collocation:.4f}")
        
        strategy = Strategy1(
            domain_bounds=domain_bounds,
            collocation_resolution=collocation_resolution,
            K_on_collocation=K_on_collocation
        )
        
        # Strategy 1 では K_max は計算結果なので、ダミー値を渡す
        # （Strategy 2 との互換性のため）
        K_max_dummy = 10.0  # 実際の値は finalize_setup() で計算
        
        return strategy, K_on_collocation, K_max_dummy

    def _create_strategy2(self, domain_bounds: Tuple[float, float, float, float]):
        """
        Create Strategy 2: Given K and K_max → calculate h
        
        Returns:
            (strategy, K_solver, K_max)
            K_max は必須（h を計算するため）
        """
        from deform_algo import Strategy2
        
        interactive_resolution = self.strategy_params.get(
            'interactive_resolution', 200
        )
        K_solver = self.strategy_params.get('K_solver', 3.5)
        K_max = self.strategy_params.get('K_max', 5.0)
        
        strategy = Strategy2(
            domain_bounds=domain_bounds,
            interactive_resolution=interactive_resolution
        )
        
        return strategy, K_solver, K_max

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
        
        Strategy 1 の場合、ここで K_max を計算して記録する。
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

        # Strategy 1 の場合、K_max を計算
        if self.strategy_type == "strategy1":
            h = self.solver.current_h
            strategy = self.solver.strategy
            self.guaranteed_K_max = strategy.compute_guaranteed_K_max(
                self.solver.basis, h
            )
            print(f"Strategy 1: Guaranteed K_max = {self.guaranteed_K_max:.4f}")

        collocation_count = len(self.solver.collocation_points)
        print(f"Generated {collocation_count} collocation points")

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

    def solve_frame(self, inverse_grid_resolution: int = 64) -> None:
        """
        Solve for new deformation during drag operation.
        This method is called on every mouse move to update the mapping coefficients.

        Args:
            inverse_grid_resolution: Unused (kept for compatibility)
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

    def get_basis_parameters(self) -> Dict:
        """
        Get current basis function parameters.

        Returns:
            Dictionary with:
            - coefficients: List[List[float]] - (2, N+3) coefficient matrix
            - centers: List[List[float]] - (N, 2) RBF centers
            - s_param: float - Gaussian width parameter
            - n_rbf: int - Number of RBF basis functions
        """
        if self.solver is None:
            raise RuntimeError("Solver not initialized")

        return {
            'coefficients': self.solver.coefficients.tolist(),  # (2, N+3)
            'centers': self.solver.basis.centers.tolist(),      # (N, 2)
            's_param': float(self.solver.basis.s),
            'n_rbf': len(self.control_points),
        }

    def end_drag_operation(self) -> None:
        """
        Called when user releases mouse button (onMouseUp).
        Performs Strategy 2 verification and refinement if needed.
        """
        if not self.is_setup_finalized:
            raise RuntimeError("Setup not finalized")

        if self.solver is None:
            return

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

    def reset_mesh(self) -> None:
        """
        Reset to identity mapping (no deformation).
        """
        self.control_points.clear()
        self.is_setup_finalized = False
        self.solver = None
        self.contour = None
        self.guaranteed_K_max = None
        print("Reset mesh")

