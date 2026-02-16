#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cvxpy as cp
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Union
import warnings
from scipy.spatial import Delaunay
from PIL import Image
import os

# --- Configuration & Strategy ---

@dataclass
class DistortionStrategyConfig(ABC):
    """Base configuration for distortion control strategy (Section 4 of Paper)."""
    @abstractmethod
    def compute_h(self, basis: 'BasisFunction') -> float:
        """
        Calculate required grid spacing h based on strategy. 
        Returns -1.0 if h is determined by grid_resolution.
        """
        pass

    @abstractmethod
    def resolve_constraints(self, basis: 'BasisFunction', h: float) -> Tuple[float, float]:
        """
        Calculate solver constraints (K_upper, Sigma_lower) to enforce on grid.
        Args:
           basis: The basis function (provides omega(h)).
           h: The actual grid spacing used.
        Returns:
           (K_upper, Sigma_lower)
        """
        pass

@dataclass
class FixedBoundCalcGrid(DistortionStrategyConfig):
    """
    Strategy 2 (Guarantee Mode):
    Given target K on grid and required global K_max, calculate required grid spacing h.
    Satisfies: omega(h) <= min( K_max - K, 1/K - 1/K_max )
    """
    K: float = 2.0       # Target bound for distortion on grid
    K_max: float = 10.0  # Guaranteed upper bound for distortion everywhere

    def compute_h(self, basis: 'BasisFunction') -> float:
        return basis.compute_h_strict(self.K, self.K_max)

    def resolve_constraints(self, basis: 'BasisFunction', h: float) -> Tuple[float, float]:
        # For Strategy 2, K is fixed and given.
        return (self.K, 1.0 / self.K)

@dataclass
class FixedGridCalcBound(DistortionStrategyConfig):
    """
    Strategy 1 (Fixed Grid Mode):
    Given grid (h) and target K on grid, calculate resulting K_max (informative).
    """
    grid_resolution: Tuple[int, int]
    K: float = 2.0

    def compute_h(self, basis: 'BasisFunction') -> float:
        return -1.0

    def resolve_constraints(self, basis: 'BasisFunction', h: float) -> Tuple[float, float]:
        # Calculate theoretical K_max for information
        omega = basis.compute_omega(h)
        
        # Valid range for K_max given K and omega?
        # K_max >= K + omega
        # K_max >= 1 / (1/K - omega)
        
        K_max_est = self.K + omega
        if (1.0/self.K) > omega:
             K_max_est = max(K_max_est, 1.0 / (1.0/self.K - omega))
        else:
             K_max_est = float('inf')

        print(f"[Strategy 1] With h={h:.4f}, K={self.K} on grid => Theoretical Global K_max <= {K_max_est:.4f}")
        return (self.K, 1.0 / self.K)

@dataclass
class CalculateKFromBound(DistortionStrategyConfig):
    """
    Strategy 3 (Calculation of K Mode):
    Given grid (h) and target global K_max, calculate stricter K required on grid.
    Satisfies: K <= min( K_max - omega(h), 1 / (1/K_max + omega(h)) )
    """
    grid_resolution: Tuple[int, int]
    K_max: float = 10.0

    def compute_h(self, basis: 'BasisFunction') -> float:
        return -1.0

    def resolve_constraints(self, basis: 'BasisFunction', h: float) -> Tuple[float, float]:
        omega = basis.compute_omega(h)
        
        # 1. Condition derived from K_max >= K + omega
        # => K <= K_max - omega
        cond1 = self.K_max - omega
        
        # 2. Condition derived from K_max >= 1 / (1/K - omega)  [Injectivity]
        # => 1/K - omega >= 1/K_max
        # => 1/K >= 1/K_max + omega
        # => K <= 1 / (1/K_max + omega)
        cond2 = 1.0 / ((1.0 / self.K_max) + omega)
        
        K_target = min(cond1, cond2)
        
        if K_target < 1.0:
            warnings.warn(f"[Strategy 3] Impossible to guarantee K_max={self.K_max} with h={h:.4f} (omega={omega:.4f}). Grid too coarse.")
            K_target = 1.001 # Minimal valid K
            
        print(f"[Strategy 3] To guarantee K_max <= {self.K_max}, enforcing K <= {K_target:.4f} on grid (h={h:.4f})")
        return (K_target, 1.0 / K_target)


@dataclass
class SolverConfig:
    """Main configuration for the Solver."""
    domain_bounds: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    source_handles: np.ndarray  # (N, 2) Initial positions of control points
 
    # Strategy pattern instance
    # デフォルト値を設定して引数順序エラー (non-default argument follows default argument) を回避
    # リファクタリング計画に従い、Strategy 2 (Guarantee Mode) をデフォルトとする。
    strategy: 'DistortionStrategyConfig' = field(default_factory=lambda: FixedBoundCalcGrid())

    epsilon: float = 100.0     # Width parameter for Gaussian RBF

    lambda_biharmonic: float = 1e-4  # Regularization weight

    fps_k: int = 50  # Number of permanently active points Z'' (Farthest Point Sampling)
                      # Paper Section 5: "keep a small subset of equally spread collocation
                      # points always active" to stabilize against fast handle movement.

# --- Basis Function Interface ---

class BasisFunction(ABC):
    @abstractmethod
    def set_centers(self, centers: np.ndarray) -> None:
        """Set the RBF centers (normally the source handle positions)."""
        pass

    @abstractmethod
    def evaluate(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute Basis Matrix Phi.
        Args:
            coords: (M, 2)
        Returns:
            Phi: (M, N_basis)
        """
        pass

    @abstractmethod
    def jacobian(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute Gradient of Basis Matrix.
        Args:
            coords: (M, 2)
        Returns:
            GradPhi: (M, N_basis, 2)
        """
        pass

    @abstractmethod
    def compute_omega(self, h: float) -> float:
        """Calculate modulus of continuity omega(h)."""
        pass

    @abstractmethod
    def compute_h_strict(self, K: float, K_max: float) -> float:
        """Calculate grid spacing h to guarantee bounds strictly."""
        pass

    @abstractmethod
    def get_identity_coefficients(self, src_handles: np.ndarray) -> np.ndarray:
        """Return coefficients c that result in an identity mapping."""
        pass

    @abstractmethod
    def get_basis_count(self) -> int:
        pass

# --- Gaussian RBF Implementation ---

class GaussianRBF(BasisFunction):
    def __init__(self, epsilon: float = 100.0):
        self.epsilon = float(epsilon)
        # Relationship between epsilon and paper's s: phi(r) = exp(-r^2 / 2s^2)
        # Implementation uses: exp(-r^2 / epsilon^2)
        # Thus: 2s^2 = epsilon^2  => s = epsilon / sqrt(2)
        self.s = self.epsilon / np.sqrt(2.0)
        self.centers: Optional[np.ndarray] = None

    def set_centers(self, centers: np.ndarray) -> None:
        self.centers = np.asarray(centers, dtype=float)

    def _rbf(self, r):
        # phi(r) = exp( - (r/eps)^2 )
        return np.exp(-(r / self.epsilon) ** 2)

    def evaluate(self, coords: np.ndarray) -> np.ndarray:
        if self.centers is None:
            raise RuntimeError("GaussianRBF centers not set.")

        x = np.asarray(coords, dtype=float)    # (M, 2)
        c = self.centers                       # (N, 2)
        M = x.shape[0]
        N = c.shape[0]

        # Pairwise distance squared: |x-c|^2 = |x|^2 + |c|^2 - 2<x,c>
        # Using broadcasting can be memory intense for very large M*N,
        # but for typical usage (M~1000-5000, N~10-100) it is fine.
        x2 = np.sum(x**2, axis=1).reshape((M, 1))
        c2 = np.sum(c**2, axis=1).reshape((1, N))
        dist2 = x2 + c2 - 2 * (x @ c.T)
        dist2 = np.maximum(dist2, 0.0)
        r = np.sqrt(dist2)

        vals = self._rbf(r)  # (M, N)

        # Affine terms: [1, x, y]
        ones = np.ones((M, 1))

        # Phi = [RBFs, 1, x, y] -> (M, N+3)
        return np.hstack([vals, ones, x])

    def jacobian(self, coords: np.ndarray) -> np.ndarray:
        if self.centers is None:
            raise RuntimeError("GaussianRBF centers not set.")

        x = np.asarray(coords, dtype=float) # (M, 2)
        c = self.centers                    # (N, 2)
        M = x.shape[0]
        N = c.shape[0]

        # We need gradients with respect to x (input coordinates)
        # Gradient of RBF part:
        # phi(r) = exp( - ||x-c||^2 / eps^2 )
        # d/dx phi = phi * ( -1/eps^2 ) * d/dx ( ||x-c||^2 )
        #          = phi * ( -1/eps^2 ) * 2(x-c)
        #          = -2/eps^2 * (x-c) * phi

        # diff: (M, N, 2)  (x_i - c_j)
        diff = x[:, None, :] - c[None, :, :]
        r2 = np.sum(diff**2, axis=2) # (M, N)
        val = np.exp(-r2 / (self.epsilon**2)) # (M, N)

        # grad_rbf: (M, N, 2)
        grad_rbf = -2.0 / (self.epsilon**2) * diff * val[:, :, None]

        # Gradient of Affine part: [1, x, y]
        # d/dx(1) = [0, 0]
        # d/dx(x) = [1, 0]
        # d/dx(y) = [0, 1]

        grad_affine = np.zeros((M, 3, 2))
        grad_affine[:, 1, 0] = 1.0
        grad_affine[:, 2, 1] = 1.0

        # GradPhi: (M, N+3, 2)
        return np.concatenate([grad_rbf, grad_affine], axis=1)

    def compute_omega(self, h: float) -> float:
        """
        Paper Table 1: omega(t) approaches t/s^2 for Gaussian kernel.
        """
        # omega(h) = h / s^2
        return h / (self.s ** 2)

    def compute_h_strict(self, K: float, K_max: float) -> float:
        """
        Calculate grid spacing h to guarantee bounds strictly.
        Condition: omega(h) <= min( K_max - K, 1/K - 1/K_max )
        """
        if K_max <= K:
            warnings.warn("K_max <= K in configuration. Forcing minimal h.")
            return 0.1 * self.s

        # 1. Upper Bound condition: omega(h) <= K_max - K
        cond1 = K_max - K

        # 2. Lower Bound (Injectivity) condition: omega(h) <= 1/K - 1/K_max
        cond2 = (1.0 / K) - (1.0 / K_max)
        
        # Stricter constraint wins
        max_omega = min(cond1, cond2)
        
        if max_omega <= 0:
             return 0.1 * self.s

        # max_omega = h / s^2  =>  h = max_omega * s^2
        h = max_omega * (self.s ** 2)

        # Safety factor just in case
        return h * 0.95

    def get_identity_coefficients(self, src_handles: np.ndarray) -> np.ndarray:
        N = src_handles.shape[0]
        # c shape: (2, N+3)
        # RBF weights = 0
        # Affine part: u = x (index N+1), v = y (index N+2)
        c = np.zeros((2, N + 3), dtype=float)
        c[0, N + 1] = 1.0
        c[1, N + 2] = 1.0
        return c

    def get_basis_count(self) -> int:
        if self.centers is None:
            return 0
        return self.centers.shape[0] + 3

# --- Main Solver Class ---

class ProvablyGoodPlanarMapping(ABC):
    def __init__(self, config: SolverConfig):
        self.config = config

        # Internal State
        self.basis: Optional[BasisFunction] = None
        self.collocation_grid: Optional[np.ndarray] = None # Z
        self.Phi: Optional[np.ndarray] = None
        self.GradPhi: Optional[np.ndarray] = None
        self.h_grid: float = 0.0                     # Actual grid spacing h

        self.c: Optional[np.ndarray] = None          # Coefficients (2, N+3)
        self.di: Optional[np.ndarray] = None         # Frame directions for checking (M, 2)
        self.activated_indices: List[int] = []       # Active set indices

        self.H_reg: Optional[np.ndarray] = None      # Regularization matrix (Biharmonic)

        # Initialize
        self._initialize_solver()

    def _initialize_solver(self):
        """Constructs grid, precomputes matrices, and sets initial identity state."""
        print("[Solver] Initializing...")

        # 1. Setup Basis (Abstract or via Config in subclass)
        self._setup_basis()

        # 2. Setup Grid based on Strategy
        self._setup_grid()

        # 3. Resolve Constraints (Strategy 3 needs h_grid which is now set)
        self.K_upper, self.Sigma_lower = self.config.strategy.resolve_constraints(
            self.basis, self.h_grid
        )
        print(f"[Solver] Constraints set: K <= {self.K_upper:.4f}, Sigma >= {self.Sigma_lower:.4f}")

        # 4. Precompute Basis Matrices (Phi, GradPhi) on Grid
        self._precompute_basis_on_grid()

        # 5. Initialize Coefficients (Identity)
        self.c = self.basis.get_identity_coefficients(self.config.source_handles)

        # 6. Initialize Frames (di)
        # Default direction (1,0) for all grid points
        M = self.collocation_grid.shape[0]
        self.di = np.tile(np.array([1.0, 0.0]), (M, 1))

        # 7. K_high, K_low thresholds (Paper Section 5)
        # K_high = 0.1 + 0.9*K: points above this are added to active set
        # K_low  = 0.5 + 0.5*K: points below this are removed from active set
        self.K_high = 0.1 + 0.9 * self.K_upper
        self.K_low  = 0.5 + 0.5 * self.K_upper

        # 8. Z'': Farthest Point Sampling for stabilization (Algorithm 1)
        # These points are always active and never removed.
        self.permanent_indices = self._farthest_point_sampling(self.config.fps_k)

        # 9. Initialize active set with Z''
        self.activated_indices = list(self.permanent_indices)

        print(f"[Solver] Initialized. Grid size: {M} points. h={self.h_grid:.4f}")
        print(f"[Solver] K_high={self.K_high:.4f}, K_low={self.K_low:.4f}")
        print(f"[Solver] Z'' (permanent active points): {len(self.permanent_indices)}")

    @abstractmethod
    def _setup_basis(self):
        """Initialize self.basis"""
        pass

    def _farthest_point_sampling(self, k: int) -> List[int]:
        """
        FPS on collocation grid to select Z'' (permanently active points).
        Paper Algorithm 1: "Initialize set Z'' with farthest point samples."
        Paper Section 5: "keep a small subset of equally spread collocation
        points always active" for stabilization.
        """
        points = self.collocation_grid
        m = points.shape[0]
        if k >= m:
            return list(range(m))
        if k <= 0:
            return []

        indices = np.empty(k, dtype=int)
        # Start from the point with minimum x-coordinate (deterministic)
        indices[0] = int(np.argmin(points[:, 0]))

        chosen = points[indices[0]]
        min_d2 = np.sum((points - chosen) ** 2, axis=1)

        for i in range(1, k):
            next_idx = int(np.argmax(min_d2))
            indices[i] = next_idx
            d2 = np.sum((points - points[next_idx]) ** 2, axis=1)
            min_d2 = np.minimum(min_d2, d2)

        return indices.tolist()

    def _find_local_maxima(self, K_vals: np.ndarray) -> np.ndarray:
        """
        Find local maxima of distortion on the collocation grid (4-neighborhood).
        Paper Algorithm 1: "Find the set Z_max of local maxima of D(z)."

        Uses >= for comparisons to handle plateaus where distortion is
        uniformly high. In such cases, all plateau points are treated as
        local maxima, ensuring they can be added to the active set if
        they exceed K_high. This is conservative but safe.

        Args:
            K_vals: (M,) distortion values at each grid point
        Returns:
            Array of indices that are local maxima
        """
        nx, ny = self.grid_shape
        # collocation_grid is built from meshgrid(x, y) with y as outer loop
        # so the grid layout is (ny, nx) when reshaped
        K_grid = K_vals.reshape(ny, nx)

        # A point is a local maximum if it is >= all its existing neighbors.
        # Using >= instead of > to handle plateaus (uniform distortion regions).
        # On a plateau, every point qualifies, which is conservative but prevents
        # the scenario where no local maxima are found despite high distortion.
        local_max_mask = np.ones((ny, nx), dtype=bool)

        local_max_mask[1:, :]  &= K_grid[1:, :]  >= K_grid[:-1, :]   # vs above
        local_max_mask[:-1, :] &= K_grid[:-1, :] >= K_grid[1:, :]    # vs below
        local_max_mask[:, 1:]  &= K_grid[:, 1:]  >= K_grid[:, :-1]   # vs left
        local_max_mask[:, :-1] &= K_grid[:, :-1] >= K_grid[:, 1:]    # vs right

        # Boundary points: only compare with existing neighbors (already handled
        # by the slicing above - boundary edges have fewer comparisons, which is
        # conservative and correct)

        return np.where(local_max_mask.ravel())[0]

    def _setup_grid(self):
        strategy = self.config.strategy
        bounds = self.config.domain_bounds
        min_x, min_y, max_x, max_y = bounds
        width = max_x - min_x
        height = max_y - min_y

        nx, ny = 20, 20 # Default fallback
        h: float = -1.0

        if isinstance(strategy, FixedBoundCalcGrid):
            h = strategy.compute_h(self.basis)
            print(f"[Solver] Strategy: FixedBoundCalcGrid. Calculated h={h:.4f} for K={strategy.K}, K_max={strategy.K_max}")

            # Simple safeguard against infinite grid
            if h < 1e-4: h = 1e-4

            nx = int(np.ceil(width / h)) + 1
            ny = int(np.ceil(height / h)) + 1
            
            # Recalculate actual h to match domain
            # Or assume h is strict upper bound. 
            # We use the computed h for grid generation which implies actual spacing.
            
            # We removed the clamping logic here to respect the theoretical guarantees.
            # If the grid is too large, it might be slow, but it will be correct.

        elif isinstance(strategy, (FixedGridCalcBound, CalculateKFromBound)):
            nx, ny = strategy.grid_resolution
            print(f"[Solver] Strategy: Fixed Resolution {nx}x{ny}")
            # Calculate h from resolution
            h_x = width / (nx - 1) if nx > 1 else width
            h_y = height / (ny - 1) if ny > 1 else height
            h = max(h_x, h_y)

        # Fallback calculation if h was not set by strategy (e.g. unknown strategy with default 20x20)
        if h < 0:
            h_x = width / (nx - 1) if nx > 1 else width
            h_y = height / (ny - 1) if ny > 1 else height
            h = max(h_x, h_y)
            print(f"[Solver] Strategy: Unknown/Fallback. Using defaults {nx}x{ny}, h={h:.4f}")

        x = np.linspace(min_x, max_x, nx)
        y = np.linspace(min_y, max_y, ny)
        xv, yv = np.meshgrid(x, y)

        # Z: (M, 2)
        self.collocation_grid = np.column_stack([xv.ravel(), yv.ravel()])
        self.grid_shape = (nx, ny)
        self.h_grid = h

    def _precompute_basis_on_grid(self):
        """Precomputes Phi, GradPhi and Regularization Matrix H_reg."""
        if self.collocation_grid is None:
            raise RuntimeError("Grid not setup.")

        print("[Solver] Precomputing basis functions on grid...")
        self.Phi = self.basis.evaluate(self.collocation_grid)
        self.GradPhi = self.basis.jacobian(self.collocation_grid)

        # Compute Biharmonic Regularization Matrix
        # Algorithm:
        # 1. Build discrete Laplacian L on the grid (using Delaunay or grid connectivity)
        # 2. M = L^T L
        # 3. H_reg = Phi^T M Phi

        try:
            # Using Delaunay for generic grid connectivity
            tri = Delaunay(self.collocation_grid)
            simplices = tri.simplices

            m = self.collocation_grid.shape[0]
            # Adjacency
            # Note: This loop can be slow for very large grids in Python.
            # But for solver grids (e.g. 50x50=2500) it is fast enough.

            # Helper for faster adjacency filling
            edges = set()
            for s in simplices:
                s = np.sort(s)
                edges.add((s[0], s[1]))
                edges.add((s[1], s[2]))
                edges.add((s[0], s[2]))

            rows = []
            cols = []
            values = []

            # Build L = D - A directly?
            # We use dense matrix for M because m is small enough usually.
            # If m is large, we should use sparse matrices.
            # For robustness with cvxpy, keeping it dense if N is small is better,
            # BUT H_reg is (N_basis x N_basis), which is small.
            # L is (M x M).

            # Let's use sparse for L construction
            from scipy import sparse

            row_idx = []
            col_idx = []
            data = []

            for (i, j) in edges:
                # A[i, j] = 1
                row_idx.extend([i, j])
                col_idx.extend([j, i])
                data.extend([1.0, 1.0])

            A = sparse.coo_matrix((data, (row_idx, col_idx)), shape=(m, m))
            deg = np.array(A.sum(axis=1)).flatten()
            D = sparse.diags(deg)
            L = D - A

            # M = L.T @ L
            M_mat = L.T @ L

            # H_reg = Phi.T @ M @ Phi
            # Phi is dense (M, N_basis). M_mat is sparse(M, M).
            # Result H_reg is dense (N_basis, N_basis).

            M_dot_Phi = M_mat @ self.Phi # (M, N_basis)
            H = self.Phi.T @ M_dot_Phi   # (N_basis, N_basis)

            self.H_reg = 0.5 * (H + H.T) # Ensure symmetry

        except Exception as e:
            warnings.warn(f"Failed to compute regularization matrix: {e}. Regularization disabled.")
            self.H_reg = None

    def compute_mapping(self, target_handles: np.ndarray):
        """
        Main Loop (Algorithm 1 from Paper):
        
        Paper Algorithm 1 order:
          Initialization: Evaluate D(z), find Z_max, update Z' (add/remove)
          Optimization:   Solve SOCP to find c
          Postprocessing: Update d_i using Eq. 27
        
        Loop: Active Set update → Optimize → Frame update
        """
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            # 1. Update frames based on current c (Eq. 27)
            #    On first call after _initialize_solver, c is identity and di is (1,0).
            #    This is consistent with the paper's initialization.
            self._update_frames()

            # 2. Evaluate distortion and update active set (Algorithm 1: Initialization)
            #    - Evaluate D(z) for all z in Z
            #    - Find local maxima Z_max of D(z)
            #    - Insert z in Z_max with D(z) > K_high into Z'
            #    - Remove z in Z' with D(z) < K_low from Z' (except Z'')
            new_violations = self._update_active_set()

            # 3. Solve SOCP with updated active set (Algorithm 1: Optimization)
            self._optimize_step(target_handles)

            # 4. Convergence check:
            #    On iteration 0, always continue (handle positions changed).
            #    After that, if no new violations were added, we have converged.
            if new_violations == 0 and iteration > 0:
                break

            iteration += 1

        if iteration == max_iterations:
            warnings.warn(f"[Solver] Max iterations ({max_iterations}) reached. Solution might not be fully valid.")

        # Final frame update for next call's warm start (Algorithm 1: Postprocessing)
        self._update_frames()

    def transform(self, points: np.ndarray) -> np.ndarray:
        """
        Apply current mapping to arbitrary points (e.g. render mesh).
        points: (K, 2)
        returns: (K, 2)
        """
        if self.c is None:
            return points # Identity if not initialized

        Phi_points = self.basis.evaluate(points) # (K, N+3)
        # c is (2, N+3)
        # result = (Phi * c^T) -> (K, 2)
        return Phi_points @ self.c.T

    # --- Internal Methods ---

    def _update_frames(self):
        """Update global frames (di) based on CURRENT coefficients c."""
        # Calculate gradients
        G = self.GradPhi # (M, N+3, 2)
        grad_u = np.einsum('j,mjk->mk', self.c[0], G)
        grad_v = np.einsum('j,mjk->mk', self.c[1], G)

        ux, uy = grad_u[:, 0], grad_u[:, 1]
        vx, vy = grad_v[:, 0], grad_v[:, 1]

        # f_z = 0.5 * (ux + vy + i(vx - uy))
        re_fz = 0.5 * (ux + vy)
        im_fz = 0.5 * (vx - uy)

        fz_vecs = np.stack([re_fz, im_fz], axis=1) # (M, 2)
        norm_fz = np.linalg.norm(fz_vecs, axis=1, keepdims=True)

        # Avoid zero division
        # If |fz| ~ 0, we simply keep the previous di or default (1,0)
        # We only update where norm is sufficient.
        mask = (norm_fz > 1e-12).flatten()
        
        # In a strict implementation, we might not update di for points NOT in the active set,
        # but updating all helps for future activations.
        # Crucially, we MUST update di for Active Set points to rotate the cut plane.
        
        self.di[mask] = fz_vecs[mask] / norm_fz[mask]

    def _compute_distortion_on_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute distortion K at each grid point.
        Returns:
           K_vals: (M,)
           extras: dict
        """
        # G: (M, N+3, 2)
        G = self.GradPhi

        # c: (2, N+3)
        # u = c[0] @ Phi^T  => grad_u = c[0] @ GradPhi

        # grad_u: (M, 2)
        grad_u = np.einsum('j,mjk->mk', self.c[0], G)
        grad_v = np.einsum('j,mjk->mk', self.c[1], G)

        # f_z  = 0.5 * ( (ux + vy) + i(vx - uy) )
        # f_zb = 0.5 * ( (ux - vy) + i(vx + uy) )

        ux, uy = grad_u[:, 0], grad_u[:, 1]
        vx, vy = grad_v[:, 0], grad_v[:, 1]

        re_fz  = 0.5 * (ux + vy)
        im_fz  = 0.5 * (vx - uy)
        re_fzb = 0.5 * (ux - vy)
        im_fzb = 0.5 * (vx + uy)

        abs_fz  = np.sqrt(re_fz**2 + im_fz**2)
        abs_fzb = np.sqrt(re_fzb**2 + im_fzb**2)

        eps = 1e-12
        mu = abs_fzb / np.maximum(abs_fz, eps)

        # K = (1+|mu|) / (1-|mu|)
        # Clamp mu to 1-eps to avoid division by zero
        mu_clamped = np.minimum(mu, 1.0 - 1e-6)
        K_vals = (1.0 + mu_clamped) / (1.0 - mu_clamped)

        return K_vals, {"abs_fz": abs_fz, "abs_fzb": abs_fzb, "mu": mu}

    def _update_active_set(self) -> int:
        """
        Update Active Set Z' based on distortion evaluation (Algorithm 1).

        Paper Algorithm 1:
          1. Evaluate D(z) for z in Z
          2. Find Z_max = local maxima of D(z)
          3. foreach z in Z_max with D(z) > K_high: insert z into Z'
          4. foreach z in Z' with D(z) < K_low: remove z from Z' (but never remove Z'')

        Paper Section 5:
          K_high = 0.1 + 0.9*K  (add threshold, slightly below K)
          K_low  = 0.5 + 0.5*K  (remove threshold)

        Returns: number of new points added to the active set.
        """
        # 1. Evaluate distortion D(z) at all collocation points
        K_vals, _ = self._compute_distortion_on_grid()

        # 2. Find local maxima of D(z) on the grid
        local_maxima = self._find_local_maxima(K_vals)

        # 3. Remove points with D(z) < K_low from Z' (but protect Z'')
        permanent_set = set(self.permanent_indices)
        remaining = []
        for idx in self.activated_indices:
            if idx in permanent_set:
                remaining.append(idx)  # Z'' is never removed
            elif K_vals[idx] >= self.K_low:
                remaining.append(idx)  # Distortion still significant, keep
            # else: distortion sufficiently below bound, remove from active set
        self.activated_indices = remaining

        # 4. Add local maxima with D(z) > K_high to Z'
        current_set = set(self.activated_indices)
        new_added_count = 0

        for idx in local_maxima:
            idx = int(idx)
            if K_vals[idx] > self.K_high and idx not in current_set:
                current_set.add(idx)
                new_added_count += 1

        self.activated_indices = sorted(list(current_set))

        return new_added_count

    def _compute_di_local(self, indices: List[int]) -> np.ndarray:
        """
        Retrieve frames d_i for specified indices.
        Now simply returns the stored self.di which are updated only when added to active set.
        """
        if not indices:
            return np.zeros((0, 2))

        idx_arr = np.array(indices)
        return self.di[idx_arr]

    def _optimize_step_DISABLED(self, target_handles: np.ndarray):
        # Renamed old method to avoid confusion, though logic is mostly same
        pass

    def _optimize_step(self, target_handles: np.ndarray):
        # CVXPY Setup
        N_basis = self.basis.get_basis_count() # N + 3 usually
        c_var = cp.Variable((2, N_basis))

        constraints = []

        # 1. Positional Constraints (Soft)
        # Minimize sum of distances to target_handles
        # or constrain them?
        # Usually soft constraints (Minimize ||Phi c - target||)

        # Basis at source handles (centers)
        # Note: Provide direct way to get Phi for handles without re-evaluating?
        # It's evaluating at centers.
        # self.basis.evaluate(self.config.source_handles)
        # This is (N, N+3).
        Phi_src = self.basis.evaluate(self.config.source_handles)

        # Objective: Position fitting
        # sum |c * phi - target|^2
        diff = c_var @ Phi_src.T - target_handles.T # (2, N)
        # L2 norm per point:
        # We want simple L2 minimization. cp.sum_squares(diff)
        position_loss = cp.sum_squares(diff)

        # 2. Distortion Constraints (Active Set)
        if self.activated_indices:
            idx_list = self.activated_indices
            G_sub = self.GradPhi[idx_list] # (K, N+3, 2)

            # Retrieve fixed local frames di (from start of frame / last valid configuration).
            # We use the cached self.di which is updated only in _postprocess.
            # This ensures we are always linearizing around a valid (unfolded) state, 
            # forcing the solver to "unfold" if it tries to violate injectivity.

            di_local = self._compute_di_local(idx_list) # (K, 2)
            # Constraints construction based on Poranne & Lipman 2014
            # We enforce 3 guarantees on the collocation points:
            # 1. Upper Bound on Distortion (Max Singular Value): Sigma_1 <= K
            # 2. Lower Bound on Distortion (Min Singular Value): Sigma_2 >= 1/K (Injectivity)
            # 3. Orientation Preservation det(J) > 0 (Implicitly satisfied by Sigma_2 > 0)

            for k, idx in enumerate(idx_list):
                G_k = G_sub[k] # (N+3, 2)
                d_k = di_local[k] # (2,)

                grad_u_k = c_var[0] @ G_k # (2,)
                grad_v_k = c_var[1] @ G_k # (2,)

                # fz components
                fz_re = 0.5 * (grad_u_k[0] + grad_v_k[1])
                fz_im = 0.5 * (grad_v_k[0] - grad_u_k[1])

                fzb_re = 0.5 * (grad_u_k[0] - grad_v_k[1])
                fzb_im = 0.5 * (grad_v_k[0] + grad_u_k[1])

                fz_vec = cp.hstack([fz_re, fz_im])
                fzb_vec = cp.hstack([fzb_re, fzb_im])

                # Guarantee 1: Upper Bound (Sigma_1 <= K)
                # |fz| + |fzb| <= K
                constraints.append(cp.norm(fz_vec, 2) + cp.norm(fzb_vec, 2) <= self.K_upper)

                # Guarantee 2: Injectivity / Lower Bound (Sigma_2 >= 1/K)
                # Linearized constraint: Re(<fz, d>) - |fzb| >= 1/K
                # This ensures |fz| - |fzb| >= 1/K, hence Sigma_2 >= 1/K.
                # d_k is the normalized direction of fz from previous iteration.
                dot_prod = fz_re * float(d_k[0]) + fz_im * float(d_k[1])
                constraints.append(dot_prod - cp.norm(fzb_vec, 2) >= self.Sigma_lower)

                # Guarantee 3: Orientation (Jacobian Determinant > 0)
                # det(J) = |fz|^2 - |fzb|^2 = (|fz|-|fzb|)(|fz|+|fzb|) = Sigma_2 * Sigma_1
                # Since we enforce Sigma_2 >= 1/K > 0 and Sigma_1 <= K, det(J) > 0 is guaranteed.

        # 3. Regularization Term
        reg_term = 0
        if self.H_reg is not None and self.config.lambda_biharmonic > 0:
            # H_reg is (N_basis, N_basis)
            # Typically applied only to RBF part (first N terms)?
            # V1 logic applied to RBF only.
            N = self.config.source_handles.shape[0]
            H_rbf = self.H_reg[:N, :N]

            c_rbf_u = c_var[0, :N]
            c_rbf_v = c_var[1, :N]

            quad = cp.quad_form(c_rbf_u, H_rbf) + cp.quad_form(c_rbf_v, H_rbf)
            reg_term = self.config.lambda_biharmonic * quad

        objective = cp.Minimize(position_loss + reg_term)

        problem = cp.Problem(objective, constraints)

        try:
            # ECOS is standard, but sometimes CLARABEL or SCS is available/better.
            problem.solve(solver=cp.ECOS, verbose=False)
        except Exception as e:
            warnings.warn(f"Solver failed: {e}")
            return

        if problem.status not in ["infeasible", "unbounded"]:
            self.c = c_var.value
        else:
            warnings.warn(f"Optimization failed: {problem.status}")

    def _postprocess(self):
        """Update global frames (di) based on new coefficients."""
        # For visualization or next frame warm start visualisation
        # We need to update ALL d_i to match the new configuration
        # so that when the user starts the NEXT drag operation, 
        # the initial linearization frames are correct.
        
        G = self.GradPhi # (M, N+3, 2)
        grad_u = np.einsum('j,mjk->mk', self.c[0], G)
        grad_v = np.einsum('j,mjk->mk', self.c[1], G)

        # This method is effectively deprecated since we use _update_frames() in the loop.
        # But keeping it empty or calling _update_frames() is fine.
        self._update_frames()

class BetterFitwithGaussianRBF(ProvablyGoodPlanarMapping):
    """
    Concrete implementation of Provably Good Planar Mappings using Gaussian RBF.
    """
    def _setup_basis(self):
        # Create the basis instance
        self.basis = GaussianRBF(epsilon=self.config.epsilon)

        # Setup centers
        self.basis.set_centers(self.config.source_handles)

# --- Bevy Interface ---

class BevyBridge:
    def __init__(self):
        self.solver: Optional[ProvablyGoodPlanarMapping] = None

        # Data stored during setup
        self.mesh_vertices: Optional[np.ndarray] = None # (N_verts, 2)
        self.mesh_indices: Optional[List[int]] = None
        
        self.epsilon: float = 100.0  # Default epsilon, updated by load_image
        self.image_width: int = 0
        self.image_height: int = 0

        self.control_point_indices: List[int] = []
        self.control_points_initial: List[Tuple[float, float]] = []
        self.control_points_current: List[Tuple[float, float]] = []

        self.is_setup_finalized = False

    def load_image_and_generate_mesh(self, image_path: str, epsilon: float) -> Tuple[List[float], List[int], List[float], int, int]:
        """
        Load image, calculate guaranteed h, generate mesh.
        Returns: (flat_vertices, flat_indices, flat_uvs, width, height)
        """
        print(f"[Py] Loading image: {image_path} with epsilon={epsilon}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        self.epsilon = float(epsilon)
        
        # 1. Load Image
        img = Image.open(image_path)
        w, h_img = img.size
        self.image_width = w
        self.image_height = h_img
        
        # Access pixel data (RGBA)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        pixels = img.load()

        # 2. Determine mesh sampling stride
        # Mesh density is for visual quality, independent of the solver's
        # collocation grid spacing h. Target ~40 points along the longest axis
        # for good visual quality while keeping triangle count manageable.
        max_dim = max(w, h_img)
        stride = max(max_dim // 40, 3)  # At least 3px to avoid degenerate triangles
            
        print(f"[Py] Using sampling stride: {stride} px (image: {w}x{h_img})")

        # 3. Sample Points
        points = []
        uvs = []
        
        # Grid sampling
        for y in range(0, h_img, stride):
            for x in range(0, w, stride):
                r, g, b, a = pixels[x, y]
                if a > 10: # Threshold
                    # Bevy World Coordinates: Center Origin, Y-Up
                    # Image Coordinates: TopLeft Origin, Y-Down
                    # We store in SOLVER COORDINATES (Internal).
                    # Let's map 1:1 to Image Coordinates for simpler debugging first?
                    # NO, the Plan says "Internal data unified to Solver coordinates".
                    # And Solver coordinates usually match the domain.
                    # Let's use Image Coordinates internally (0,0 is Top-Left) 
                    # because RBF doesn't care about origin, and it's easier to map back to UV.
                    # The Rust side expects Bevy World Coordinates for display?
                    # Actually, the previous implementation did the conversion in Rust.
                    # "Internal data stored in Solver mode" -> Let's keep Image Coordinates (0..W, 0..H)
                    # and let Rust transform to World for Bevy display.
                    
                    points.append([float(x), float(y)])
                    uvs.append([x / w, 1.0 - (y / h_img)]) # UV (0..1), V is 1 at Top if we want? No, usually V=1 is Top in Bevy?
                    # Bevy: UV (0,0) is bottom-left? 
                    # Standard: (0,0) Top-Left often in textures. 
                    # Bevy standard: (0,0) Top-Left or Bottom-Left? 
                    # In main.rs: uvs.push([x as f32 / width_f, 1.0 - (y as f32 / height_f)]);
                    # This implies V=0 at Top (because y=0 -> V=1). Wait.
                    # y=0(Top) -> 1.0 - 0 = 1.0. So V=1 is Top.
                    # y=H(Bottom) -> 1.0 - 1 = 0.0. So V=0 is Bottom.
                    # This matches OpenGL/Bevy usually (0,0) is Bottom-Left.

        points_np = np.array(points, dtype=np.float64)
        print(f"[Py] Sampled {len(points)} points.") 

        if len(points) < 3:
             print("[Py] Not enough points to triangulate.")
             return ([], [], [], w, h_img)

        # 4. Delaunay Triangulation
        tri = Delaunay(points_np)
        
        # 5. Filter Triangles (Centroid check)
        valid_indices = []
        for simplex in tri.simplices: # simplex is (3,) indices
            pts = points_np[simplex]
            centroid = np.mean(pts, axis=0) # (2,)
            cx, cy = int(centroid[0]), int(centroid[1])
            
            # Check bounds
            if 0 <= cx < w and 0 <= cy < h_img:
                # Check alpha
                try:
                    r, g, b, a = pixels[cx, cy]
                    if a > 10:
                        valid_indices.extend(simplex)
                except IndexError:
                    pass
        
        # Prepare Return Data
        flat_verts = points_np.flatten().tolist()
        flat_uvs = np.array(uvs).flatten().tolist()
        
        # Store locally
        self.mesh_vertices = points_np
        self.mesh_indices = valid_indices
        self.control_point_indices = []
        self.control_points_initial = []
        self.control_points_current = []
        self.solver = None
        self.is_setup_finalized = False
        
        return (flat_verts, valid_indices, flat_uvs, float(w), float(h_img))

    def initialize_mesh_from_data(self, vertices: List[float], indices: List[int]):
        """
        Receive mesh data from Bevy.
        vertices: Flat list [x0, y0, x1, y1, ...]
        indices: Flat list of triangle indices
        """
        print(f"[Py] initializing mesh: {len(vertices)//2} verts, {len(indices)//3} tris")
        # Ensure flat list to numpy array
        if isinstance(vertices, list):
            verts_array = np.array(vertices, dtype=np.float32)
        else:
             # If it comes as something else (e.g. from pyo3 it might be a list)
             verts_array = np.array(vertices, dtype=np.float32)

        self.mesh_vertices = verts_array.reshape(-1, 2)
        self.mesh_indices = indices

        # Reset control points when new mesh is loaded
        # self.reset_mesh()
        # Actually reset_mesh clears everything including solver state
        self.control_point_indices = []
        self.control_points_initial = []
        self.control_points_current = []
        self.solver = None
        self.is_setup_finalized = False

    def add_control_point(self, index: int, x: float, y: float):
        """Add a control point during setup."""
        if self.is_setup_finalized:
            print("[Py] Warning: Cannot add points after finalization.")
            return

        # Rustからの座標入力を無視し、頂点インデックスに対応する正確な内部座標を使用する
        # これにより、初期状態でのソース位置とターゲット位置の不一致（歪みの原因）を防ぐ
        final_x, final_y = x, y
        if self.mesh_vertices is not None and 0 <= index < len(self.mesh_vertices):
            pt = self.mesh_vertices[index]
            final_x = float(pt[0])
            final_y = float(pt[1])
            
        print(f"[Py] Adding control point: mesh_idx={index} at ({final_x:.1f}, {final_y:.1f})")
        self.control_point_indices.append(index)
        self.control_points_initial.append((final_x, final_y))
        self.control_points_current.append((final_x, final_y))

    def update_control_point(self, index_in_list: int, x: float, y: float):
        """Update a control point position (Deform mode)."""
        if index_in_list >= 0 and index_in_list < len(self.control_points_current):
            self.control_points_current[index_in_list] = (x, y)
        else:
            print(f"[Py] Warning: update index {index_in_list} out of range")

    def finalize_setup(self):
        """Build the solver."""
        if not self.control_points_initial:
            print("[Py] No control points. Cannot finalize.")
            return

        print("[Py] Finalizing setup. Building solver...")

        if self.mesh_vertices is None:
            print("[Py] Error: No mesh vertices.")
            return

        bounds_min = np.min(self.mesh_vertices, axis=0)
        bounds_max = np.max(self.mesh_vertices, axis=0)

        # Add some margin
        margin = 50.0
        domain = [
            float(bounds_min[0] - margin), float(bounds_min[1] - margin),
            float(bounds_max[0] + margin), float(bounds_max[1] + margin)
        ]

        sources = np.array(self.control_points_initial)

        domain_width  = domain[2] - domain[0]
        domain_height = domain[3] - domain[1]

        # --- Paper Section 6 workflow ---
        # Paper: "In all experiments we used a 200^2 grid during interaction"
        # Use Strategy 1 (FixedGridCalcBound): fix grid resolution, set K on grid,
        # compute theoretical K_max for information only.
        #
        # Grid resolution: scale with domain aspect ratio.
        # ECOS handles ~100-200 active constraints well; the full grid can be
        # larger since only active-set points become SOCP constraints.
        nx = 30
        ny = max(int(nx * domain_height / max(domain_width, 1e-6)), 10)
        K_target = 4.0

        # Compute actual h from grid resolution
        h_x = domain_width  / (nx - 1) if nx > 1 else domain_width
        h_y = domain_height / (ny - 1) if ny > 1 else domain_height
        h = max(h_x, h_y)

        # Auto-derive minimum epsilon from h and K (Paper Eq. 11 requirement)
        # For Gaussian RBF: omega(h) = h / s^2 = 2h / epsilon^2
        # Injectivity guarantee requires: omega(h) < 1/K
        # => 2h / epsilon^2 < 1/K
        # => epsilon > sqrt(2 * h * K)
        epsilon_min = np.sqrt(2.0 * h * K_target)
        effective_epsilon = max(self.epsilon, epsilon_min)

        # Verify omega(h) is valid
        s = effective_epsilon / np.sqrt(2.0)
        omega_h = h / (s * s)
        print(f"[Py] Grid: {nx}x{ny} = {nx*ny} points, h={h:.2f}")
        print(f"[Py] epsilon: user={self.epsilon:.1f}, min={epsilon_min:.1f}, effective={effective_epsilon:.1f}")
        print(f"[Py] omega(h)={omega_h:.6f}, 1/K={1.0/K_target:.6f}, valid={omega_h < 1.0/K_target}")

        strategy = FixedGridCalcBound(grid_resolution=(nx, ny), K=K_target)

        config = SolverConfig(
            domain_bounds=tuple(domain),
            source_handles=sources,
            epsilon=effective_epsilon,
            lambda_biharmonic=1e-6,
            strategy=strategy,
        )

        self.solver = BetterFitwithGaussianRBF(config)
        
        print("[Py] Identity mapping established (Implicitly).")
        
        self.is_setup_finalized = True

    def reset_mesh(self):
        """Reset control points to initial state."""
        # Clears everything to start fresh setup
        print("[Py] Resetting mesh state.")
        self.control_point_indices = []
        self.control_points_initial = []
        self.control_points_current = []
        self.solver = None
        self.is_setup_finalized = False

    def solve_frame(self) -> List[float]:
        """
        Run one step of optimization and return transformed mesh vertices.
        """
        # If not finalized, return original mesh
        if not self.is_setup_finalized or self.solver is None:
            if self.mesh_vertices is not None:
                return self.mesh_vertices.flatten().tolist()
            return []

        targets = np.array(self.control_points_current)
        initial = np.array(self.control_points_initial)
        
        # Check if control points have actually moved from initial positions
        # 修正: 移動がない場合はソルバーを一切実行せず、元のメッシュをそのまま返すことで
        # 数値誤差による微小な歪みを完全に防ぐ。
        if np.allclose(targets, initial, atol=1e-5):
             return self.mesh_vertices.flatten().tolist()

        # If moved, compute mapping
        self.solver.compute_mapping(targets)

        # Transform all mesh vertices
        # mesh_vertices: (N, 2)
        deformed_verts = self.solver.transform(self.mesh_vertices)

        return deformed_verts.flatten().tolist()

