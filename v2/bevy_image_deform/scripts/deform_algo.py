#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cvxpy as cp
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Union
import warnings
from scipy.spatial import Delaunay

# --- Configuration & Strategy ---

@dataclass
class DistortionStrategyConfig:
    """Base configuration for distortion control strategy."""
    pass

@dataclass
class FixedBoundCalcGrid(DistortionStrategyConfig):
    """
    Strategy 2 (Guarantee Mode):
    Calculate required grid spacing 'h' from K and K_max to guarantee injectivity.
    """
    K: float = 2.0       # Target bound for distortion on grid
    K_max: float = 10.0  # Guaranteed upper bound for distortion everywhere

@dataclass
class FixedGridCalcBound(DistortionStrategyConfig):
    """
    Strategy 1 (Fixed Grid Mode):
    Use a fixed grid resolution. Theoretical K_max is calculated (informative).
    """
    grid_resolution: Tuple[int, int]
    K: float = 2.0

@dataclass
class HeuristicFast(DistortionStrategyConfig):
    """
    Strategy 3 (Fast Mode):
    Looser constraints, heuristic update, potentially skipping rigorous checks.
    """
    grid_resolution: Tuple[int, int]
    K: float = 3.0

@dataclass
class SolverConfig:
    """Main configuration for the Solver."""
    domain_bounds: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    source_handles: np.ndarray  # (N, 2) Initial positions of control points
    
    basis_type: str = "Gaussian"
    epsilon: float = 100.0     # Width parameter for Gaussian RBF
    
    lambda_biharmonic: float = 1e-4  # Regularization weight
    
    strategy: DistortionStrategyConfig = field(default_factory=FixedBoundCalcGrid)

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
    def compute_h_from_distortion(self, K: float, K_max: float) -> float:
        """Calculate grid spacing h to satisfy K_max >= K + omega(h)."""
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

    def compute_h_from_distortion(self, K: float, K_max: float) -> float:
        # Paper (Table 1 & Sec 4): Modulus of continuity omega(t) approx t * (Lip(Phi')?)
        # Detailed logic: K_max >= K + omega(h).
        # For Gaussian: omega(t) ~= t / s^2 (Linear approximation valid for small t)
        # s = epsilon / sqrt(2)
        # K_max = K + h / s^2  ==>  h = (K_max - K) * s^2
        
        if K_max <= K:
            warnings.warn("K_max <= K in configuration. Forcing minimal h.")
            return 1e-3 * self.epsilon
            
        s2 = self.s ** 2
        # Use a safety factor or derived Lipschitz constant. 
        # Here we use the simplified relation derived in the context of the paper's examples.
        h = (K_max - K) * s2
        
        if h <= 0:
            return 1e-3 * self.epsilon
        return h

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

class ProvablyGoodMapping:
    def __init__(self, config: SolverConfig):
        self.config = config
        
        # 1. Setup Basis
        if self.config.basis_type == "Gaussian":
            self.basis = GaussianRBF(epsilon=self.config.epsilon)
        else:
            raise NotImplementedError(f"Basis type {self.config.basis_type} not implemented.")
            
        self.basis.set_centers(self.config.source_handles)
        
        # Internal State
        self.collocation_grid: Optional[np.ndarray] = None # Z
        self.Phi: Optional[np.ndarray] = None
        self.GradPhi: Optional[np.ndarray] = None
        
        self.c: Optional[np.ndarray] = None          # Coefficients (2, N+3)
        self.di: Optional[np.ndarray] = None         # Frame directions for checking (M, 2)
        self.activated_indices: List[int] = []       # Active set indices
        
        self.H_reg: Optional[np.ndarray] = None      # Regularization matrix (Biharmonic)
        
        # Thresholds derived from Strategy
        if hasattr(self.config.strategy, 'K'):
            self.K_target = self.config.strategy.K
        else:
            self.K_target = 2.0
            
        self.K_high = 0.5 + 0.9 * self.K_target
        self.K_low = 0.5 + 0.5 * self.K_target
        
        # Initialize
        self._initialize_solver()

    def _initialize_solver(self):
        """Constructs grid, precomputes matrices, and sets initial identity state."""
        print("[Solver] Initializing...")
        
        # 2. Setup Grid based on Strategy
        self._setup_grid()
        
        # 3. Precompute Basis Matrices (Phi, GradPhi) on Grid
        self._precompute_basis_on_grid()
        
        # 4. Initialize Coefficients (Identity)
        self.c = self.basis.get_identity_coefficients(self.config.source_handles)
        
        # 5. Initialize Frames (di)
        # Default direction (1,0) for all grid points
        M = self.collocation_grid.shape[0]
        self.di = np.tile(np.array([1.0, 0.0]), (M, 1))
        
        self.activated_indices = []
        
        print(f"[Solver] Initialized. Grid size: {M} points.")

    def _setup_grid(self):
        strategy = self.config.strategy
        bounds = self.config.domain_bounds
        min_x, min_y, max_x, max_y = bounds
        width = max_x - min_x
        height = max_y - min_y
        
        nx, ny = 20, 20 # Default fallback
        
        if isinstance(strategy, FixedBoundCalcGrid):
            h = self.basis.compute_h_from_distortion(strategy.K, strategy.K_max)
            print(f"[Solver] Strategy: FixedBoundCalcGrid. Calculated h={h:.4f} for K={strategy.K}, K_max={strategy.K_max}")
            
            # Simple safeguard against infinite grid
            if h < 1e-4: h = 1e-4
            
            nx = int(np.ceil(width / h)) + 1
            ny = int(np.ceil(height / h)) + 1
            
            # Clamp to reasonable size to prevent OOM during interactive editing if h is tiny
            if nx * ny > 250000: # 500x500
                warnings.warn(f"Calculated grid size {nx}x{ny} is too large. Clamping resolution.")
                ratio = np.sqrt(250000 / (nx*ny))
                nx = int(nx * ratio)
                ny = int(ny * ratio)
                
        elif isinstance(strategy, FixedGridCalcBound) or isinstance(strategy, HeuristicFast):
            nx, ny = strategy.grid_resolution
            print(f"[Solver] Strategy: Fixed Resolution {nx}x{ny}")

        x = np.linspace(min_x, max_x, nx)
        y = np.linspace(min_y, max_y, ny)
        xv, yv = np.meshgrid(x, y)
        
        # Z: (M, 2)
        self.collocation_grid = np.column_stack([xv.ravel(), yv.ravel()])
        self.grid_shape = (nx, ny)

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
        Main Loop:
        1. Check distortion on grid (update active set)
        2. Solve optimization problem
        3. Post-process (update constraints frames)
        """
        # Update target positions in config? Or just use local variable?
        # Ideally, we optimize to match target_handles.
        
        # 1. Update Active Set
        self._update_active_set()
        
        # 2. Optimize
        self._optimize_step(target_handles)
        
        # 3. Post-process
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

    def _update_active_set(self):
        K_vals, _ = self._compute_distortion_on_grid()
        
        # Find points violating K_high
        violators = np.where(K_vals > self.K_high)[0]
        
        # Add to set
        current_set = set(self.activated_indices)
        for v in violators:
            current_set.add(int(v))
            
        # Optional: Remove points satisfying K_low (Hysteresis)
        # To be safe, we might just keep them or remove carefully.
        # Filter:
        kept_indices = []
        for idx in current_set:
            if K_vals[idx] >= self.K_low:
                kept_indices.append(idx)
        
        self.activated_indices = sorted(kept_indices)
        
        # If strategy is Heuristic, maybe limit the size?
        # For now, stick to logic.

    def _compute_di_local(self, indices: List[int]) -> np.ndarray:
        """Compute frames d_i for specified indices using current coefficients."""
        if not indices:
            return np.zeros((0, 2))
            
        idx_arr = np.array(indices)
        G = self.GradPhi[idx_arr] # (K, N+3, 2)
        
        grad_u = np.einsum('j,mjk->mk', self.c[0], G)
        grad_v = np.einsum('j,mjk->mk', self.c[1], G)
        
        ux, uy = grad_u[:, 0], grad_u[:, 1]
        vx, vy = grad_v[:, 0], grad_v[:, 1]
        
        # f_z vector (real, imag)
        re_fz  = 0.5 * (ux + vy)
        im_fz  = 0.5 * (vx - uy)
        
        fz_vecs = np.stack([re_fz, im_fz], axis=1) # (K, 2)
        norm_fz = np.linalg.norm(fz_vecs, axis=1, keepdims=True)
        
        # Normalized direction
        # Handle zero norm
        mask = (norm_fz > 1e-12).flatten()
        
        di_subset = np.zeros_like(fz_vecs)
        # Copy existing di for near-zero gradients? Or default (1,0)?
        # For this temporary calculation, we use (1,0) fallback if singular
        di_subset[...] = np.array([1.0, 0.0])
        
        if np.any(mask):
            di_subset[mask] = fz_vecs[mask] / norm_fz[mask]
            
        return di_subset

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
            # This linearizes the constraint around current state.
            di_local = self._compute_di_local(idx_list) # (K, 2)
            
            # Constraints construction
            # For each point in active set:
            # |f_z| + |f_zb| <= K_max (or K_high to be safe)
            # Re(f_z * di_bar) - |f_zb| >= 1/K (approx of lower bound |f_z| - |f_zb| >= 1/K?)
            # Wait, let's check the paper or v1 code.
            
            # V1 code:
            # fz = 0.5 * (grad_u.x + grad_v.y, grad_v.x - grad_u.y)
            # fzb = 0.5 * (grad_u.x - grad_v.y, grad_v.x + grad_u.y)
            # 1. |fz| + |fzb| <= K_high
            # 2. <fz, di> - |fzb| >= 1/K  (Linearized lower bound for |fz|-|fzb|)
            
            # Efficient implementation in CVXPY?
            # CVXPY overhead is high for loops. We need vectorization if possible.
            # But G_sub is specific per point.
            
            # It's hard to fully vectorize with G_sub structure in CVXPY without loops 
            # unless we flatten carefully.
            # Loop for now. If slow, optimization needed.
            
            for k, idx in enumerate(idx_list):
                G_k = G_sub[k] # (N+3, 2)
                d_k = di_local[k] # (2,)
                
                # grad_u_vec = c_var[0] @ G_k
                # grad_v_vec = c_var[1] @ G_k
                
                # But c_var is (2, N). 
                # grad_u at point k is scalar? No.
                # G_k is matrix (N_basis, 2).
                # c_var[0] is (N_basis,).
                # product is (2,) vector? -> [du/dx, du/dy]
                
                grad_u_k = c_var[0] @ G_k # (2,)
                grad_v_k = c_var[1] @ G_k # (2,)
                
                # fz components
                fz_re = 0.5 * (grad_u_k[0] + grad_v_k[1])
                fz_im = 0.5 * (grad_v_k[0] - grad_u_k[1])
                
                fzb_re = 0.5 * (grad_u_k[0] - grad_v_k[1])
                fzb_im = 0.5 * (grad_v_k[0] + grad_u_k[1])
                
                fz_vec = cp.hstack([fz_re, fz_im])
                fzb_vec = cp.hstack([fzb_re, fzb_im])
                
                # Constraint 1: Upper Bound
                constraints.append(cp.norm(fz_vec, 2) + cp.norm(fzb_vec, 2) <= self.K_high)
                
                # Constraint 2: Lower Bound (Injectivity)
                # <fz, d> = fz_re * d_re + fz_im * d_im
                dot_prod = fz_re * float(d_k[0]) + fz_im * float(d_k[1])
                constraints.append(dot_prod - cp.norm(fzb_vec, 2) >= 1.0 / self.K_target)

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
        # This updates self.di for ALL grid points to be ready for next iteration 
        # (or for checking global validation if needed).
        # Actually, Algorithm 1 updates d_i only for update_active_set or use in next step?
        # Refactoring doc says: "Frame (d_i) management: redundant parts..."
        # We only really need d_i for the active set during optimization.
        # But `postprocess` in V1 updated all di.
        
        # Let's update all d_i on the grid.
        # This allows visualizing the field or using it for next seeding.
        
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

# --- Bevy Interface ---

class BevyBridge:
    def __init__(self):
        self.solver: Optional[ProvablyGoodMapping] = None
        
        # Data stored during setup
        self.mesh_vertices: Optional[np.ndarray] = None # (N_verts, 2)
        self.mesh_indices: Optional[List[int]] = None
        
        self.control_point_indices: List[int] = []
        self.control_points_initial: List[Tuple[float, float]] = []
        self.control_points_current: List[Tuple[float, float]] = []
        
        self.is_setup_finalized = False

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
        # strategy = FixedBoundCalcGrid(K=2.0, K_max=10.0) 
        strategy = HeuristicFast(grid_resolution=(20, 20), K=20.0)
        
        config = SolverConfig(
            domain_bounds=tuple(domain),
            source_handles=sources,
            basis_type="Gaussian",
            epsilon=150.0, 
            lambda_biharmonic=1e-5,
            strategy=strategy
        )
        
        self.solver = ProvablyGoodMapping(config)
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

