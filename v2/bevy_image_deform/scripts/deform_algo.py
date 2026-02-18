#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cvxpy as cp
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import warnings
from scipy.spatial import Delaunay
from PIL import Image
import os
from typing import Tuple, Optional, List, cast

# --- Basis Function Interface ---

class BasisFunction(ABC):
    @abstractmethod
    def evaluate(self, coords: np.ndarray) -> np.ndarray:
        """Compute Basis Matrix Phi (Eq. 3)."""
        pass

    @abstractmethod
    def jacobian(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute Gradient of Basis Matrix (Eq. 786).
        Needed for distortion constraints (Eq. 23, 26, 28) and ARAP energy.
        """
        pass

    @abstractmethod
    def hessian(self, coords: np.ndarray) -> np.ndarray:  # <--- 追加
        """
        Compute Hessian of Basis Matrix.
        Needed for Biharmonic regularization energy (Eq. 31, cite: 1002).
        Returns: (M, N_basis, 2, 2)
        """
        pass

    @abstractmethod
    def compute_basis_gradient_modulus(self, h: float) -> float: # <--- 名前をより正確に変更（推奨）
        """
        Calculate modulus of continuity omega_nabla_F(h) for the basis itself.
        Corresponds to Table 1 in the paper.
        """
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
    def __init__(self, s: float = 1.0):
        """
        Initialize Gaussian RBF Basis.

        Args:
            s (float): The width parameter 's' from the paper (Table 1).
                       Basis function: phi(r) = exp( -r^2 / (2s^2) )
        """
        self.s = float(s)
        self.centers: Optional[np.ndarray] = None

    def set_centers(self, centers: np.ndarray) -> None:
        self.centers = np.asarray(centers, dtype=float)

    def _get_diff_vectors(self, coords: np.ndarray):
        """Helper to compute difference vectors (x - c)."""
        if self.centers is None:
            raise RuntimeError("GaussianRBF centers not set.")

        x = np.asarray(coords, dtype=float)  # (M, 2)
        c = self.centers                     # (N, 2)

        # diff: (M, N, 2) -> (x_i - c_j)
        # Broadcasting: (M, 1, 2) - (1, N, 2)
        diff = x[:, None, :] - c[None, :, :]
        return diff

    def evaluate(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute Phi matrix.
        Includes Gaussian terms and Affine terms [1, x, y].
        """
        diff = self._get_diff_vectors(coords)

        # r^2 = ||x - c||^2
        r2 = np.sum(diff**2, axis=2) # (M, N)

        # Gaussian: exp( -r^2 / 2s^2 )
        vals = np.exp(-r2 / (2.0 * self.s**2))

        # Affine parts: [1, x, y]
        M = coords.shape[0]
        ones = np.ones((M, 1))

        # Result: [Gaussians, 1, x, y] -> Size: (M, N + 3)
        return np.hstack([vals, ones, coords])

    def jacobian(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute Gradient (Jacobian of basis functions).
        Returns: (M, N_basis, 2)
        """
        diff = self._get_diff_vectors(coords) # (M, N, 2)
        r2 = np.sum(diff**2, axis=2)          # (M, N)

        # phi(r)
        phi = np.exp(-r2 / (2.0 * self.s**2)) # (M, N)

        # Gradient of Gaussian:
        # d/dx phi = phi * (-1/2s^2) * d/dx( ||x-c||^2 )
        #          = phi * (-1/2s^2) * 2(x-c)
        #          = -1/s^2 * (x-c) * phi
        # Shape: (M, N, 2) broadcasted
        grad_rbf = -(1.0 / self.s**2) * diff * phi[:, :, None]

        # Gradient of Affine part [1, x, y]
        # d/dx(1) = [0, 0]
        # d/dx(x) = [1, 0]
        # d/dx(y) = [0, 1]
        M = coords.shape[0]
        grad_affine = np.zeros((M, 3, 2))
        grad_affine[:, 1, 0] = 1.0 # Gradient of x w.r.t x
        grad_affine[:, 2, 1] = 1.0 # Gradient of y w.r.t y

        return np.concatenate([grad_rbf, grad_affine], axis=1)

    def hessian(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute Hessian of Basis Matrix.
        Required for Biharmonic Energy (Eq. 31).
        Returns: (M, N_basis, 2, 2)
        """
        diff = self._get_diff_vectors(coords) # (M, N, 2)
        r2 = np.sum(diff**2, axis=2)          # (M, N)
        phi = np.exp(-r2 / (2.0 * self.s**2)) # (M, N)

        M, N = phi.shape

        # Hessian of Gaussian:
        # H(phi) = (phi / s^2) * [ ( (x-c)(x-c)^T ) / s^2  -  I ]
        # See derivation in Appendix A for similar logic

        # 1. Outer product (x-c)(x-c)^T
        # Shape: (M, N, 2, 2)
        outer_prod = diff[:, :, :, None] * diff[:, :, None, :]

        # 2. Identity matrix I
        I = np.eye(2)[None, None, :, :] # Broadcastable to (1, 1, 2, 2)

        # 3. Combine
        # term_bracket = (outer_prod / s^2) - I
        term_bracket = (outer_prod / (self.s**2)) - I

        # 4. Multiply by scaling factor (phi / s^2)
        # phi is (M, N), need (M, N, 1, 1)
        scale = phi[:, :, None, None] / (self.s**2)

        hess_rbf = scale * term_bracket # (M, N, 2, 2)

        # Hessian of Affine part [1, x, y] is all zeros
        hess_affine = np.zeros((M, 3, 2, 2))

        return np.concatenate([hess_rbf, hess_affine], axis=1)

    def compute_basis_gradient_modulus(self, h: float) -> float:
        """
        Calculate modulus of continuity omega(h).
        For Gaussians, Table 1 states: omega(t) = (1/s^2) * t
        (The paper lists t/s^2 under 'fi' column and mentions L-Lipschitz in Appendix A)
        """
        # アフィン部分の勾配は定数（Hessian=0）なので、
        # 勾配の変動（Modulus）はガウス基底の最大変動に支配されます。
        return h / (self.s**2)

    def get_identity_coefficients(self, src_handles: np.ndarray) -> np.ndarray:
        """
        Return coefficients c that result in an identity mapping.
        RBF weights = 0
        Affine part: u = x, v = y
        """
        N = src_handles.shape[0]
        # c shape: (2, N_basis) = (2, N + 3)
        c = np.zeros((2, N + 3), dtype=float)

        # Affine terms are at indices N, N+1, N+2 corresponding to [1, x, y]
        # u(x,y) = 0*1 + 1*x + 0*y
        c[0, N + 1] = 1.0

        # v(x,y) = 0*1 + 0*x + 1*y
        c[1, N + 2] = 1.0

        return c

    def get_basis_count(self) -> int:
        if self.centers is None:
            return 3 # Only affine
        return self.centers.shape[0] + 3

# --- Strategy 2 Implementation (Grid & Fill Distance Manager) ---

class Strategy2:
    """
    Manages the collocation points and calculates the required fill distance 'h'
    based on the current mapping distortion to guarantee bounds (Strategy 2).
    """
    def __init__(self, domain_bounds: Tuple[float, float, float, float]):
        """
        Args:
            domain_bounds: (x_min, x_max, y_min, y_max) defining the bounding box of the image/domain.
        """
        self.bounds = domain_bounds

    def compute_required_h(self,
                           basis: BasisFunction,
                           K_current: float,
                           K_max: float,
                           c: np.ndarray) -> float:
        """
        Calculate required grid spacing (fill distance) h based on current distortion.
        Eq. 14 (Isometric) or Eq. 15 (Conformal) in the paper.

        Args:
            basis: The BasisFunction instance (to get continuity modulus).
            K_current: The maximum distortion evaluated at current collocation points.
            K_max: The global maximum allowed distortion constraint.
            c: The current mapping coefficients matrix (2, N_basis).
        """
        if K_max <= K_current:
            warnings.warn("K_current has exceeded K_max. Returning minimal safety spacing.")
            # return a very small default h if constraints are violated
            return 0.01

        # Calculate isometric distortion tolerance margin
        # min(K_max - K, 1/K - 1/K_max)
        margin = min(K_max - K_current, (1.0 / K_current) - (1.0 / K_max))
        if margin <= 0:
            return 0.01

        # Calculate max coefficient norm |||c|||
        # The paper uses various norms, typically max row sum of absolute values
        max_coeff_norm = np.max(np.sum(np.abs(c), axis=1))

        # We need: 2 * max_coeff_norm * omega_basis(h) <= margin
        # Assuming omega_basis(h) is linear w.r.t h (which is true for Gaussians: h / s^2)
        # omega_basis(1.0) gives us the constant factor C.
        C = basis.compute_basis_gradient_modulus(1.0)

        if C == 0 or max_coeff_norm == 0:
            return float('inf') # No practical limit

        # Solve for h
        h_limit = margin / (2.0 * max_coeff_norm * C)

        return h_limit * 0.95  # 5% safety margin to ensure strict inequality

    def generate_grid(self, h: float) -> np.ndarray:
        """
        Generates a regular grid of collocation points covering the domain with spacing h.
        (In a full implementation, points outside the non-convex domain mask would be filtered out here).
        """
        x_min, x_max, y_min, y_max = self.bounds

        # Create 1D arrays for x and y
        x_coords = np.arange(x_min, x_max + h, h)
        y_coords = np.arange(y_min, y_max + h, h)

        # Create 2D meshgrid
        X, Y = np.meshgrid(x_coords, y_coords)

        # Flatten to (M, 2) array
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T
        return grid_points


# --- Main Mapping Class ---

class ProvablyGoodPlanarMapping(ABC):
    """
    メッシュレス画像変形フレームワークの抽象基底クラス。
    共通の数学的評価（評価、ヤコビアン、歪み計算、Strategy2のグリッド更新）を提供する。
    """
    def __init__(self, basis: 'BasisFunction', strategy: 'Strategy2', K_max: float = 5.0):
        self.basis = basis
        self.strategy = strategy
        self.K_max = K_max
        self.coefficients: Optional[np.ndarray] = None
        self.collocation_points: Optional[np.ndarray] = None

    def evaluate_map(self, coords: np.ndarray) -> np.ndarray:
        """ f(x) = c * Phi(x) を評価して変換後の座標を返す """
        if self.coefficients is None:
            raise RuntimeError("マッピング係数が初期化されていません。")
        phi = self.basis.evaluate(coords)
        return phi @ self.coefficients.T

    def evaluate_jacobian(self, coords: np.ndarray) -> np.ndarray:
        """ ヤコビアン J_f(x) を計算する """
        if self.coefficients is None:
            raise RuntimeError("マッピング係数が初期化されていません。")
        grad_phi = self.basis.jacobian(coords)
        J = np.einsum('dj, mjk -> mdk', self.coefficients, grad_phi)
        return J

    def compute_max_distortion(self, coords: np.ndarray) -> float:
        """ 与えられた座標群における最大アイソメトリック歪み K を計算する """
        J = self.evaluate_jacobian(coords)
        _, S, _ = np.linalg.svd(J)

        sigma_1 = S[:, 0]
        sigma_2 = np.maximum(S[:, 1], 1e-8) # ゼロ除算防止

        point_distortions = np.maximum(sigma_1, 1.0 / sigma_2)
        return float(np.max(point_distortions))

    def update_collocation_grid(self):
        """ Strategy 2: 現在の歪みに基づいてコロケーションポイントのグリッドを更新する """
        if self.collocation_points is None or self.coefficients is None:
            return

        K_current = self.compute_max_distortion(self.collocation_points)
        required_h = self.strategy.compute_required_h(
            basis=self.basis,
            K_current=K_current,
            K_max=self.K_max,
            c=self.coefficients
        )
        self.collocation_points = self.strategy.generate_grid(required_h)
        print(f"[Grid Update] New h = {required_h:.4f}, Active Points = {len(self.collocation_points)}")

    # ==========================================
    # サブクラスで実装すべき抽象メソッド
    # ==========================================
    @abstractmethod
    def initialize_mapping(self, src_handles: np.ndarray) -> None:
        """ 初期状態（恒等写像など）をセットアップする """
        pass

    @abstractmethod
    def optimize_mapping(self, target_handles: np.ndarray, num_iterations: int) -> None:
        """ 指定されたハンドル位置に向けて、係数 self.coefficients を最適化する """
        pass


class BetterFitwithGaussian(ProvablyGoodPlanarMapping):
    """
    ガウス基底関数を用いて実際の変形処理（最適化）を担当する具象クラス。
    """
    def __init__(self, domain_bounds: Tuple[float, float, float, float], s_param: float = 1.0, K_max: float = 5.0):
        # このクラス自身がGaussianRBFとStrategy2をインスタンス化し、親クラスに渡す
        basis = GaussianRBF(s=s_param)
        strategy = Strategy2(domain_bounds)
        super().__init__(basis=basis, strategy=strategy, K_max=K_max)

    def initialize_mapping(self, src_handles: np.ndarray) -> None:
        """
        ソースハンドル（制御点）の位置をRBFのセンターとして設定し、
        恒等写像（動いていない状態）の係数で初期化する。
        """
        # GaussianRBF特有のセットアップ
        self.basis.set_centers(src_handles)
        self.coefficients = self.basis.get_identity_coefficients(src_handles)

        # 初期の粗いグリッドを生成 (ドメインサイズに応じて適宜調整)
        initial_h = (self.strategy.bounds[1] - self.strategy.bounds[0]) / 10.0
        self.collocation_points = self.strategy.generate_grid(initial_h)
        print("Initialized BetterFitwithGaussian mapping.")

    def optimize_mapping(self, target_handles: np.ndarray, num_iterations: int = 5) -> None:
        """
        実際の最適化ループ。
        変形エネルギー（ARAPやBiharmonic）を最小化しつつ、K_maxの制約を満たすように
        係数 c を更新していく。
        """
        for i in range(num_iterations):
            print(f"--- Iteration {i+1}/{num_iterations} ---")

            # 1. 現在の状態に合わせて監視点（コロケーションポイント）の密度を自動調整
            self.update_collocation_grid()

            # 2. ここに実際の最適化ソルバ（SciPyのminimizeや独自のニュートン法など）を記述
            # 目的関数: E_data(c) + E_reg(c)
            # 制約条件: self.compute_max_distortion(self.collocation_points) <= self.K_max
            #
            # (※ 実装時は self.coefficients を更新します)

            # [ダミーコード] 今回は更新されたフリをする
            # self.coefficients = updated_c
            pass

        print("Optimization completed.")
