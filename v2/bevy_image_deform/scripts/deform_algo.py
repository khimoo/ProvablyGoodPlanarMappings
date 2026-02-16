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

        self.activated_indices = []

        print(f"[Solver] Initialized. Grid size: {M} points. h={self.h_grid:.4f}")

    @abstractmethod
    def _setup_basis(self):
        """Initialize self.basis"""
        pass

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
            
            # Clamp to reasonable size to prevent OOM
            if nx * ny > 250000: # 500x500
                warnings.warn(f"Calculated grid size {nx}x{ny} is too large. Clamping resolution.")
                ratio = np.sqrt(250000 / (nx*ny))
                nx = int(nx * ratio)
                ny = int(ny * ratio)
                # If clamped, effective h increases
                h = max(width/nx, height/ny)

        elif isinstance(strategy, (FixedGridCalcBound, CalculateKFromBound)):
            nx, ny = strategy.grid_resolution
            print(f"[Solver] Strategy: Fixed Resolution {nx}x{ny}")
            # Calculate h from resolution
            h_x = width / (nx - 1) if nx > 1 else width
            h_y = height / (ny - 1) if ny > 1 else height
            h = max(h_x, h_y)

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
        Iteratively add constraints and re-solve until no violations occur.
        """
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            # 1. Check for violations and update active set
            # Returns number of NEW violations found
            new_violations = self._update_active_set()
            
            # If no new violations and we have solved at least once (or active set is empty initially),
            # we check if we are done.
            # Actually, the paper says: "We repeat this process until no more points are added to the active set."
            # But we must ensure the current solution satisfies the constraints.
            
            # If it's the first iteration, we simply solve.
            # If subsequent iteration found NO new violations, we are done.
            if iteration > 0 and new_violations == 0:
                # print(f"[Solver] Converged at iteration {iteration}")
                break

            # 2. Optimize with current active set
            # Note: The frames d_i for the active set are fixed at the moment they were added.
            # _update_active_set handles setting d_i for new points.
            self._optimize_step(target_handles)
            
            iteration += 1
        
        if iteration == max_iterations:
             warnings.warn(f"[Solver] Max iterations ({max_iterations}) reached. Solution might not be fully valid.")

        # 3. Post-process (Optional, for next frame warm start visualisation)
        self._postprocess()

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
        Update Active Set based on violations of the 3 guarantees.
        Returns: number of new violations added.
        """
        # We need to check violations of:
        # 1. Sigma_1 <= K (Upper Bound)
        # 2. Sigma_2 >= 1/K (Lower Bound / Injectivity)
        # 3. Orientation (Implicit)
        
        # We check them numerically on the grid.
        
        # Get gradient components for ALL grid points based on current c
        G = self.GradPhi
        grad_u = np.einsum('j,mjk->mk', self.c[0], G)
        grad_v = np.einsum('j,mjk->mk', self.c[1], G)

        ux, uy = grad_u[:, 0], grad_u[:, 1]
        vx, vy = grad_v[:, 0], grad_v[:, 1]

        # fz, fzb
        re_fz  = 0.5 * (ux + vy)
        im_fz  = 0.5 * (vx - uy)
        re_fzb = 0.5 * (ux - vy)
        im_fzb = 0.5 * (vx + uy)
        
        abs_fz = np.sqrt(re_fz**2 + im_fz**2)
        abs_fzb = np.sqrt(re_fzb**2 + im_fzb**2)
        
        # Current directions di (from previous iteration/current state)
        # Important: For checking violation 2, what d should we use?
        # The paper says: "If a point x_i violates the constraints... we add it to the active set...
        # and set d_i = J_S f(x_i) / |J_S f(x_i)|".
        # So we check if there exists a direction d such that constraints are satisfied?
        # No, we check if the current map satisfies the bounds.
        # Condition 2 is simply sigma_min >= 1/K => |fz| - |fzb| >= 1/K.
        # The linearization with d_i is only for the SOLVER constraint.
        # The CHECK is on the singular values directly.
        
        # Check Condition 1: Sigma_1 <= K
        # Sigma_1 = |fz| + |fzb|
        sigma_1 = abs_fz + abs_fzb
        violation_1 = sigma_1 > self.K_upper + 1e-4 # Tolerance
        
        # Check Condition 2: Sigma_2 >= 1/K
        sigma_2 = abs_fz - abs_fzb
        violation_2 = sigma_2 < self.Sigma_lower - 1e-4
        
        # Combined violations
        violators = np.where(violation_1 | violation_2)[0]
        
        new_added_count = 0
        if len(violators) > 0:
            current_set = set(self.activated_indices)
            
            for v in violators:
                idx = int(v)
                if idx not in current_set:
                    current_set.add(idx)
                    new_added_count += 1
                    
                    # --- CRITICAL: Set frame d_i for NEW active point ---
                    # According to paper: "When a point x_i is added to the active set... we set d_i = J_S f(x_i) / ..."
                    # We compute d_i based on the CURRENT map state (before optimization).
                    # This fixes the linearization frame.
                    
                    # Extract fz for this point
                    fz_re_val = re_fz[idx]
                    fz_im_val = im_fz[idx]
                    norm = np.sqrt(fz_re_val**2 + fz_im_val**2)
                    
                    if norm > 1e-12:
                        self.di[idx] = np.array([fz_re_val/norm, fz_im_val/norm])
                    else:
                        # Fallback for singular points
                        self.di[idx] = np.array([1.0, 0.0])

            self.activated_indices = sorted(list(current_set))
            
        return new_added_count
        
        # Note: We do not remove points in this strict version (monotonically increasing active set)
        # unless we implement a specific drop strategy.
        # For performance, could reset active set if handles change significantly.

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

            # Recompute local di for the active set based on *current* (previous step) coefficients
            # Recompute local di for the active set based on *current* (previous step) coefficients
            # This linearizes the constraint around current state.
            # FIX: Use FIXED di stored in self.di, not recomputed from current c every time.
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
        # For visualization or next frame starting point.
        # In the iterative algorithm, d_i for active constraints should remain fixed 
        # until valid solution is found or constraint is deactivated.
        # However, for the NEXT user input frame, we can reset d_i to current deformation?
        # Yes, usually between time steps we update the linearization.
        
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

        mask = (norm_fz > 1e-12).flatten()

        # Reset to default
        self.di[...] = np.array([1.0, 0.0])
        self.di[mask] = fz_vecs[mask] / norm_fz[mask]

class BetterFitwithGaussianRBF(ProvablyGoodPlanarMapping):
    """
    Concrete implementation of Provably Good Planar Mappings using Gaussian RBF.
    """
    def _setup_basis(self):
        # Create the basis instance
        self.basis = GaussianRBF(epsilon=self.config.epsilon)

        # Setup centers
        # This part was common, but since each basis might have different requirements for centers
        # (e.g. B-Spline uses grid control points, not scattered centers), it's good here.
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

        # 2. Calculate optimal grid spacing h
        # Use Strategy 2 (Guarantee Mode) to find h
        # We need a temporary basis to compute s from epsilon
        temp_basis = GaussianRBF(epsilon=self.epsilon)
        strategy = FixedBoundCalcGrid(K=2.0, K_max=10.0)
        
        h_theoretical = strategy.compute_h(temp_basis)
        print(f"[Py] Theoretical h={h_theoretical:.4f} for epsilon={self.epsilon}")
        
        # Use a sampling stride slightly smaller than h to be safe, 
        # or exactly h if we trust the grid alignment.
        # Ensure we don't have too many points (limit for performance).
        # Also ensure stride is not too large (minimum density guarantee).
        # Let's say at least ~30x30 grid points for the image?
        min_density_stride = max(w, h_img) // 30
        stride = max(int(h_theoretical), 5)
        
        if stride > min_density_stride:
            print(f"[Py] Clamping stride from {stride} to {min_density_stride} to ensure density.")
            stride = min_density_stride
            
        print(f"[Py] Using sampling stride: {stride} px")

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
        
        return (flat_verts, valid_indices, flat_uvs, w, h_img)

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

        print(f"[Py] Adding control point: mesh_idx={index} at ({x:.1f}, {y:.1f})")
        self.control_point_indices.append(index)
        self.control_points_initial.append((x, y))
        self.control_points_current.append((x, y))

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

        # Strategy Config configuration
        # For better stability, we can try different settings
        strategy = FixedBoundCalcGrid(K=2.0, K_max=10.0)
        # strategy = CalculateKFromBound(grid_resolution=(20, 20), K_max=20.0)
        # strategy = FixedGridCalcBound(grid_resolution=(25, 25), K=10)
        

        config = SolverConfig(
            domain_bounds=tuple(domain),
            source_handles=sources,
            epsilon=self.epsilon,
            lambda_biharmonic=1e-5,
            strategy=strategy
        )

        self.solver = BetterFitwithGaussianRBF(config)
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

        # Compute deform
        # Note: If targets haven't moved, compute_mapping might still do iterative updates if needed
        # But here it's likely stateless per frame unless we used prev solution as warm start
        # The current implementation of compute_mapping uses current 'targets'
        self.solver.compute_mapping(targets)

        # Transform all mesh vertices
        # mesh_vertices: (N, 2)
        deformed_verts = self.solver.transform(self.mesh_vertices)

        return deformed_verts.flatten().tolist()

