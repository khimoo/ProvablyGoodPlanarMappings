import numpy as np
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import cvxpy as cp
from dataclasses import dataclass


@dataclass
class ProvablyGoodConfig:
    """論文「Provably Good Planar Mappings」に明記されているパラメータ群"""

    # [Section 6: Results] エネルギーの重み λ
    # 【重要】論文は単位正方形（1×1）を前提としているが、ピクセル座標（数百px）では
    # E_pos（単位: px）と E_bh（無次元）の次元が不一致になり、正則化が消失する。
    # ドメインサイズ L に合わせて lambda_bh を L 倍にスケーリングする必要がある。
    # 例: 500px ドメインなら lambda_bh = 0.1 * 500 = 50.0 〜 500.0 程度
    lambda_bh: float = 1      # E_pos + λ_bh * E_bh (ピクセルスケール対応)
    lambda_arap: float = 0.0     # E_pos + λ_arap * E_arap (ピクセルスケール対応)

    # [Section 6: Results] インタラクティブ時のグリッド解像度
    interactive_resolution: int = 200  # 200^2 grid

    # [Section 5] ARAP計算用のサンプル点数 (論文では n_s として定義)
    n_arap_samples: int = 100

    # --- 実装上の追加パラメータ（数値計算の安定化用） ---
    # 行列の特異性（Ill-conditioned）による係数爆発を防ぐためのTikhonov正則化の重み
    # 不要な場合はクラス初期化時に 0.0 を渡すことで無効化可能
    lambda_coeff_reg: float = 1e-4


class BasisFunction(ABC):
    @abstractmethod
    def evaluate(self, coords: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def jacobian(self, coords: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def hessian(self, coords: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def compute_basis_gradient_modulus(self, h: float) -> float:
        pass

    @abstractmethod
    def get_identity_coefficients(self, src_handles: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_basis_count(self) -> int:
        pass


class GaussianRBF(BasisFunction):
    def __init__(self, s: float = 1.0):
        self.s = float(s)
        self.centers: Optional[np.ndarray] = None

    def set_centers(self, centers: np.ndarray) -> None:
        self.centers = np.asarray(centers, dtype=float)

    def _get_diff_vectors(self, coords: np.ndarray):
        if self.centers is None:
            raise RuntimeError("GaussianRBF centers not set.")
        x = np.asarray(coords, dtype=float)
        c = self.centers
        return x[:, None, :] - c[None, :, :]

    def evaluate(self, coords: np.ndarray) -> np.ndarray:
        diff = self._get_diff_vectors(coords)
        r2 = np.sum(diff**2, axis=2)
        vals = np.exp(-r2 / (2.0 * self.s**2))
        M = coords.shape[0]
        ones = np.ones((M, 1))
        return np.hstack([vals, ones, coords])

    def jacobian(self, coords: np.ndarray) -> np.ndarray:
        diff = self._get_diff_vectors(coords)
        r2 = np.sum(diff**2, axis=2)
        phi = np.exp(-r2 / (2.0 * self.s**2))

        grad_rbf = -(1.0 / self.s**2) * diff * phi[:, :, None]

        M = coords.shape[0]
        grad_affine = np.zeros((M, 3, 2))
        grad_affine[:, 1, 0] = 1.0
        grad_affine[:, 2, 1] = 1.0

        return np.concatenate([grad_rbf, grad_affine], axis=1)

    def hessian(self, coords: np.ndarray) -> np.ndarray:
        diff = self._get_diff_vectors(coords)
        r2 = np.sum(diff**2, axis=2)
        phi = np.exp(-r2 / (2.0 * self.s**2))
        M, N = phi.shape

        outer_prod = diff[:, :, :, None] * diff[:, :, None, :]
        I = np.eye(2)[None, None, :, :]
        term_bracket = (outer_prod / (self.s**2)) - I
        scale = phi[:, :, None, None] / (self.s**2)

        hess_rbf = scale * term_bracket
        hess_affine = np.zeros((M, 3, 2, 2))

        return np.concatenate([hess_rbf, hess_affine], axis=1)

    def compute_basis_gradient_modulus(self, h: float) -> float:
        # Table 1: ガウス基底の勾配の連続性係数
        return h / (self.s**2)

    def get_identity_coefficients(self, src_handles: np.ndarray) -> np.ndarray:
        N = src_handles.shape[0]
        c = np.zeros((2, N + 3), dtype=float)
        c[0, N + 1] = 1.0 # u = x
        c[1, N + 2] = 1.0 # v = y
        return c

    def get_basis_count(self) -> int:
        if self.centers is None:
            return 3
        return self.centers.shape[0] + 3

class GuaranteeStrategy(ABC):
    """
    論文における歪み保証のグリッド解像度(h)を決定する戦略の基底クラス。
    Strategy 1（悲観的・事前保証）と Strategy 2（楽観的・事後検証）の
    共通インターフェースを定義する。
    """
    def __init__(self, domain_bounds: Tuple[float, float, float, float]):
        """
        Args:
            domain_bounds: (x_min, x_max, y_min, y_max) 領域のバウンディングボックス
        """
        self.bounds = domain_bounds

    def generate_grid(self, h: float) -> np.ndarray:
        """
        [共通処理] 指定された h に基づいてグリッドを生成する。
        """
        x_min, x_max, y_min, y_max = self.bounds

        if h <= 0:
            raise ValueError("h must be strictly positive.")

        x_coords = np.arange(x_min, x_max + 1e-8, h)
        y_coords = np.arange(y_min, y_max + 1e-8, h)

        X, Y = np.meshgrid(x_coords, y_coords)
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T

        return grid_points

    @staticmethod
    def _compute_margin(K_solver: float, K_max: float) -> float:
        """
        Eq. 14: アイソメトリック歪みのマージン計算
        min( K_max - K,  1/K - 1/K_max )
        """
        if K_max <= K_solver:
            raise ValueError(f"K_max ({K_max}) must be strictly greater than K_solver ({K_solver}).")
        return min(K_max - K_solver, (1.0 / K_solver) - (1.0 / K_max))

    @staticmethod
    def _extract_rbf_coefficients(c: np.ndarray) -> np.ndarray:
        """
        係数行列 c からアフィン項を除いた RBF 係数部分を抽出する。
        アフィン項（最後の3列: [1, x, y]）の勾配は定数であり、
        ヘッシアンがゼロのため連続性係数に寄与しない。
        """
        if c.shape[1] >= 3:
            return c[:, :-3]
        return c

    @abstractmethod
    def get_initial_h(self, basis: 'BasisFunction', K_solver: float, K_max: float) -> float:
        """
        初期化時（またはドラッグ開始時）に使用するグリッド間隔 h を返す。
        """
        pass

    @abstractmethod
    def get_strict_h_after_drag(self, basis: 'BasisFunction', K_solver: float, K_max: float,
                                c: np.ndarray) -> float:
        """
        ドラッグ終了後に、数学的保証を満たすための厳密な h を返す。
        """
        pass


class Strategy1(GuaranteeStrategy):
    """
    Strategy 1: Given Z and K → bound K_max

    コロケーションポイント密度 h と K を指定し、
    結果として得られる K_max を計算する。
    K_max は出力であり、入力ではない。

    論文の式(11)または(13)から K_max を計算する。
    """
    def __init__(self, domain_bounds: Tuple[float, float, float, float],
                 collocation_resolution: int = 500,
                 K_on_collocation: float = 3.5):
        """
        Args:
            domain_bounds: (x_min, x_max, y_min, y_max) 領域のバウンディングボックス
            collocation_resolution: グリッド解像度
            K_on_collocation: コロケーション点上の歪み上限 K

        K_max は計算結果として得られる（入力ではない）
        """
        super().__init__(domain_bounds)
        self.collocation_resolution = collocation_resolution
        self.K_on_collocation = K_on_collocation

    def get_initial_h(self, basis: 'BasisFunction', K_solver: float, K_max: float) -> float:
        """
        Strategy 1: h は事前に決定されている

        注: K_solver と K_max は渡されるが、Strategy 1 では使用しない
        （Strategy 2 との互換性のため）
        """
        x_min, x_max, y_min, y_max = self.bounds
        max_span = max(x_max - x_min, y_max - y_min)
        return float(max_span / self.collocation_resolution)

    def get_strict_h_after_drag(self, basis: 'BasisFunction', K_solver: float, K_max: float,
                                c: np.ndarray) -> float:
        """Strategy 1: グリッド細分化なし"""
        return float('inf')

    def compute_guaranteed_K_max(self, basis: 'BasisFunction', h: float) -> float:
        """
        Strategy 1 の出力: 実際の K_max を計算

        式(11): D_iso(x) ≤ max{ K + ω(h), 1/(1/K - ω(h)) }

        ω(h) = 2 * |||c||| * ω_∇F(h)
        初期状態では |||c||| = 1（恒等写像）

        Args:
            basis: 基底関数のインスタンス
            h: コロケーションポイント密度

        Returns:
            K_max: 領域全体で保証される最大歪み
        """
        K = self.K_on_collocation

        # ω_∇F(h) を計算
        # compute_basis_gradient_modulus(t) は ω_∇F(t) を返す
        # Gaussians の場合: ω_∇F(t) = t / s²
        omega_grad_F = basis.compute_basis_gradient_modulus(h)

        # ω(h) = 2 * |||c||| * ω_∇F(h)
        # 初期状態では |||c||| = 1（恒等写像）
        omega_h = 2.0 * 1.0 * omega_grad_F

        print(f"DEBUG compute_guaranteed_K_max:")
        print(f"  K (K_on_collocation): {K:.4f}")
        print(f"  h: {h:.4f}")
        print(f"  ω_∇F(h): {omega_grad_F:.6f}")
        print(f"  ω(h) = 2 * ω_∇F(h): {omega_h:.6f}")

        # 式(11): D_iso(x) ≤ max{ K + ω(h), 1/(1/K - ω(h)) }
        term1 = K + omega_h
        term2 = 1.0 / (1.0 / K - omega_h)
        K_max_bound = max(term1, term2)

        print(f"  K + ω(h): {term1:.4f}")
        print(f"  1/(1/K - ω(h)): {term2:.4f}")
        print(f"  K_max_bound: {K_max_bound:.4f}")

        return float(K_max_bound)


class Strategy2(GuaranteeStrategy):
    """
    Strategy 2: 楽観的・事後検証アプローチ

    インタラクティブ操作中は粗い固定グリッドを使って高速に解き、
    ドラッグ終了後に「実際の係数ノルム」に基づいて厳密な h を計算し、
    必要に応じてグリッドを細かくして再計算する。
    """
    def __init__(self, domain_bounds: Tuple[float, float, float, float], interactive_resolution: int = 200):
        """
        Args:
            domain_bounds: (x_min, x_max, y_min, y_max) 領域のバウンディングボックス
            interactive_resolution: ドラッグ中に使用するグリッド解像度（デフォルト 200x200）
        """
        super().__init__(domain_bounds)
        self.interactive_resolution = interactive_resolution

    def get_initial_h(self, basis: 'BasisFunction', K_solver: float, K_max: float) -> float:
        """
        ドラッグ中は高速化のため、指定解像度で適当に区切った粗い h を返す。
        """
        x_min, x_max, y_min, y_max = self.bounds
        width = x_max - x_min
        height = y_max - y_min

        max_span = max(width, height)
        return float(max_span / self.interactive_resolution)

    def get_strict_h_after_drag(self, basis: 'BasisFunction', K_solver: float, K_max: float,
                                c: np.ndarray) -> float:
        """
        実際の変形結果 (c) に基づいて、ギリギリ安全な h を逆算する。
        """
        margin = self._compute_margin(K_solver, K_max)

        c_rbf = self._extract_rbf_coefficients(c)
        max_coeff_norm = np.max(np.sum(np.abs(c_rbf), axis=1))

        # 恒等写像（RBF係数がすべてゼロ）の場合、歪みは一切発生しないため h は無限大
        if max_coeff_norm < 1e-12:
            return float('inf')

        C = basis.compute_basis_gradient_modulus(1.0)

        if C <= 0:
            return float('inf')

        h_strict = margin / (2.0 * max_coeff_norm * C)
        return float(h_strict)


class ProvablyGoodPlanarMapping(ABC):
    """
    論文「Provably Good Planar Mappings」のコアアルゴリズム（Algorithm 1）
    を提供する抽象基底クラス。

    GuaranteeStrategy を通じて Strategy 1（事前保証）と Strategy 2（事後検証）を
    実行時に切り替え可能にする。
    """
    def __init__(self, basis: 'BasisFunction', guarantee_strategy: 'GuaranteeStrategy',
                 K_solver: float = 2.0, K_max: float = 5.0, use_arap: bool = True,
                 config: Optional[ProvablyGoodConfig] = None):
        self.basis = basis
        self.strategy = guarantee_strategy
        self.K_solver = K_solver
        self.K_max = K_max
        self.use_arap = use_arap
        self.config = config if config is not None else ProvablyGoodConfig()

        self.coefficients: Optional[np.ndarray] = None
        self.collocation_points: Optional[np.ndarray] = None
        self.current_h: float = 0.0

        # --- 事前計算のキャッシュ用 ---
        self._B_mat: Optional[np.ndarray] = None
        self._H_term: Optional[np.ndarray] = None

        # [Section 5] Active Set Thresholds
        self.K_high = 0.1 + (0.9 * K_solver)
        self.K_low = 0.5 + (0.5 * K_solver)

        self._active_indices: np.ndarray = np.array([], dtype=int)

        # [Section 5.3] Frame vectors for convexification (Eq. 27)
        self._frames: Optional[np.ndarray] = None  # (M, 2) unit vectors d_i

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
        # cは(2, N)、grad_phiは(M, N, 2)。J[m, d, k] = sum_j c[d, j] * grad_phi[m, j, k]
        J = np.einsum('dj, mjk -> mdk', self.coefficients, grad_phi)
        return J

    def compute_J_S_and_J_A(self, J: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        [論文 Eq. 19] J_S と J_A を計算する

        J_S f(x) = (∇u(x) + I∇v(x)) / 2  (similarity part)
        J_A f(x) = (∇u(x) - I∇v(x)) / 2  (anti-similarity part)

        ここで I は π/2 反時計回り回転行列: [[0, -1], [1, 0]]

        Args:
            J: (M, 2, 2) Jacobian matrices

        Returns:
            J_S: (M, 2) similarity vectors
            J_A: (M, 2) anti-similarity vectors
        """
        # J[:, 0, :] = ∇u = [∂u/∂x, ∂u/∂y]
        # J[:, 1, :] = ∇v = [∂v/∂x, ∂v/∂y]
        grad_u = J[:, 0, :]  # (M, 2)
        grad_v = J[:, 1, :]  # (M, 2)

        # I * ∇v = [[0, -1], [1, 0]] * [∂v/∂x, ∂v/∂y] = [-∂v/∂y, ∂v/∂x]
        I_grad_v = np.stack([-grad_v[:, 1], grad_v[:, 0]], axis=1)  # (M, 2)

        J_S = (grad_u + I_grad_v) / 2.0
        J_A = (grad_u - I_grad_v) / 2.0

        return J_S, J_A

    def update_frames(self, coords: np.ndarray) -> None:
        """
        [論文 Eq. 27] Frame vectors を更新する

        d_i = J_S f(x_i) / ||J_S f(x_i)||

        これは Eq. 26 の半平面制約を最適化するために使用される。
        """
        J = self.evaluate_jacobian(coords)
        J_S, _ = self.compute_J_S_and_J_A(J)

        norms = np.linalg.norm(J_S, axis=1, keepdims=True)
        # ゼロ除算を避ける
        norms = np.where(norms > 1e-8, norms, 1.0)

        self._frames = J_S / norms
    @abstractmethod
    def initialize_mapping(self, src_handles: np.ndarray) -> None:
        pass

    @abstractmethod
    def start_drag(self, src_handles: np.ndarray) -> None:
        """ onMouseDown: ドラッグ開始時に1回だけ呼ばれる（事前計算） """
        pass

    @abstractmethod
    def update_drag(self, target_handles: np.ndarray, num_iterations: int = 2) -> None:
        """ onMouseMove: ドラッグ中に毎フレーム呼ばれる（高速な最適化） """
        pass

    @abstractmethod
    def end_drag(self, target_handles: np.ndarray) -> bool:
        """ onMouseUp: ドラッグ終了時に呼ばれる（Strategy 2 の厳密な検証） """
        pass

class BetterFitwithGaussian(ProvablyGoodPlanarMapping):
    def __init__(self, domain_bounds: Tuple[float, float, float, float],
                 guarantee_strategy: Optional['GuaranteeStrategy'] = None,
                 s_param: float = 1.0, K_solver: float = 2.0, K_max: float = 5.0,
                 use_arap: bool = False, config: Optional[ProvablyGoodConfig] = None):
        """
        Args:
            domain_bounds: (x_min, x_max, y_min, y_max) 領域のバウンディングボックス
            guarantee_strategy: GuaranteeStrategy のインスタンス（デフォルトは Strategy2）
            s_param: ガウス基底関数のスケールパラメータ
            K_solver: 選点上での歪み上限
            K_max: 領域全体での歪み上限
            use_arap: ARAP エネルギーを使用するか（デフォルト: False）
            config: 論文準拠のパラメータ設定
        """
        basis = GaussianRBF(s=s_param)
        if guarantee_strategy is None:
            guarantee_strategy = Strategy2(domain_bounds)
        super().__init__(basis=basis, guarantee_strategy=guarantee_strategy,
                        K_solver=K_solver, K_max=K_max, use_arap=use_arap, config=config)
        self.basis: GaussianRBF = basis

    def initialize_mapping(self, src_handles: np.ndarray) -> None:
        """
        アプリケーション起動時・またはハンドル構成が変わった時に1回呼ばれる。
        ここで最も重い不変行列の計算をすべて終わらせる。
        """
        self.basis.set_centers(src_handles)
        self.coefficients = self.basis.get_identity_coefficients(src_handles)

        # Strategy に初期 h を要求する
        self.current_h = self.strategy.get_initial_h(self.basis, self.K_solver, self.K_max)
        self.collocation_points = self.strategy.generate_grid(self.current_h)

        # === 完全に不変な行列の事前計算 ===
        # 1. 位置拘束行列 (B_mat) は src_handles (不変) にのみ依存する
        self._B_mat = self.basis.evaluate(src_handles)

        # 2. 初期グリッドに基づくヘッシアン行列 (H_term)
        self._update_hessian_term()

        # [修正] ここが論文の "if first step then" に該当する
        self._active_indices = np.array([], dtype=int)

        # [論文 Section 5.3] Frame を恒等写像の状態で初期化: d_i = (1, 0)
        M = len(self.collocation_points)
        self._frames = np.zeros((M, 2))
        self._frames[:, 0] = 1.0  # d_i = (1, 0) for all i

    def _update_hessian_term(self) -> None:
        """ グリッドが更新されたときのみ呼ばれる内部メソッド """
        if self.collocation_points is None:
            raise RuntimeError("Collocation points not initialized.")
        collocation_points = self.collocation_points
        N_basis = self.basis.get_basis_count()
        hess = self.basis.hessian(collocation_points)
        H_mat = np.transpose(hess, (0, 2, 3, 1)).reshape(-1, N_basis)

        # [修正点2] 積分(Eq.31)の近似のため、面積要素 (h^2) を掛けてスケーリングする
        # これにより、hを細かくしてもエネルギーのスケールが不変になる
        area_element = self.current_h ** 2
        self._H_term = (H_mat.T @ H_mat) * area_element

        # Frame も再初期化
        M = len(collocation_points)
        self._frames = np.zeros((M, 2))
        self._frames[:, 0] = 1.0

    def start_drag(self, src_handles: np.ndarray) -> None:
        """
        [UI連携: onMouseDown]
        前回からのアクティブセット（_active_indices）をそのまま維持し、
        変形の連続性と滑らかさを保つ。リセットは行わない。
        """
        pass

    def update_drag(self, target_handles: np.ndarray, num_iterations: int = 2) -> None:
        """
        [UI連携: onMouseMove]
        マウスが動くたびに呼ばれる。SOCP 制約付き最適化で論文通りに解く。
        論文に存在しない独自のフェールセーフ（L2正則化、係数制限、Active Set制限）は完全に削除。
        """
        if self.collocation_points is None:
            raise RuntimeError("Collocation points not initialized.")
        if self._B_mat is None:
            raise RuntimeError("B_mat not initialized.")
        if self._H_term is None:
            raise RuntimeError("H_term not initialized.")
        if self._frames is None:
            raise RuntimeError("Frames not initialized.")

        grid = self.collocation_points
        B_mat = self._B_mat
        H_term = self._H_term
        N_basis = self.basis.get_basis_count()

        for iteration in range(num_iterations):
            # === [Active Set 抽出] ===
            J_all = self.evaluate_jacobian(grid)

            # 特異値と行列式（ヤコビアン）を計算
            U, S, Vh = np.linalg.svd(J_all)

            # 【追加】行列式 det(J) = (∂u/∂x * ∂v/∂y) - (∂u/∂y * ∂v/∂x) を手動で高速計算
            dets = J_all[:, 0, 0] * J_all[:, 1, 1] - J_all[:, 0, 1] * J_all[:, 1, 0]

            # 論文定義の歪み D_iso(x) = max{Σ(x), 1/σ(x)}
            # 【重要】SVDの特異値は常に正になるため、裏返り(det <= 0)を検知できない。
            # 行列式が負の場合は、物理的に破綻（裏返り）しているため歪みを無限大とする。
            with np.errstate(divide='ignore'):
                distortions = np.where(
                    (dets <= 1e-8) | (S[:, 1] <= 1e-8),
                    np.inf,
                    np.maximum(S[:, 0], 1.0 / S[:, 1])
                )

            # [Section 5] Active Set の更新
            is_above_high = distortions > self.K_high
            is_below_low = distortions < self.K_low

            active_mask = np.zeros(len(grid), dtype=bool)
            active_mask[self._active_indices] = True

            # 論文通り、K_highを超えた点を追加し、K_lowを下回った点を削除
            active_mask |= is_above_high
            active_mask &= ~is_below_low

            self._active_indices = np.where(active_mask)[0]
            active_indices = self._active_indices

            # === [SOCP による Global Step] ===
            c_var = cp.Variable((2, N_basis))
            if self.coefficients is not None and iteration > 0:
                c_var.value = self.coefficients

            # [Eq. 30: Positional Constraints Energy] [cite: 361, 364]
            n_handles = len(target_handles)
            r_vars = cp.Variable(n_handles)
            pos_constraints = []
            for l in range(n_handles):
                residual_u = B_mat[l, :] @ c_var[0, :] - target_handles[l, 0]
                residual_v = B_mat[l, :] @ c_var[1, :] - target_handles[l, 1]
                residual_vec = cp.vstack([residual_u, residual_v])
                pos_constraints.append(cp.norm(residual_vec) <= r_vars[l])

            E_pos = cp.sum(r_vars)

            # [Eq. 31: Biharmonic Energy] [cite: 368]
            E_bh = 0
            for d in range(2):
                E_bh += cp.quad_form(c_var[d, :], H_term)

            # 係数 c_var に対する L2 正則化（Tikhonov regularization）
            E_coeff_reg = cp.sum_squares(c_var)

            # 論文に基づく正則化エネルギーに、微小な係数正則化を足し合わせる
            E_reg = self.config.lambda_bh * E_bh + self.config.lambda_coeff_reg * E_coeff_reg

            # [Eq. 33: ARAP Energy] [cite: 379]
            if self.use_arap:
                n_samples = min(self.config.n_arap_samples, len(grid))
                arap_sample_indices = np.linspace(0, len(grid) - 1, n_samples, dtype=int)
                grad_phi_arap = self.basis.jacobian(grid[arap_sample_indices])
                frames_arap = self._frames[arap_sample_indices]

                E_arap = 0
                for i, idx in enumerate(arap_sample_indices):
                    grad_u_x = grad_phi_arap[i, :, 0] @ c_var[0, :]
                    grad_u_y = grad_phi_arap[i, :, 1] @ c_var[0, :]
                    grad_v_x = grad_phi_arap[i, :, 0] @ c_var[1, :]
                    grad_v_y = grad_phi_arap[i, :, 1] @ c_var[1, :]

                    grad_u = cp.vstack([grad_u_x, grad_u_y])
                    grad_v = cp.vstack([grad_v_x, grad_v_y])
                    I_grad_v = cp.vstack([-grad_v[1], grad_v[0]])

                    J_S_i = (grad_u + I_grad_v) / 2.0
                    J_A_i = (grad_u - I_grad_v) / 2.0

                    d_s = frames_arap[i]
                    E_arap += cp.sum_squares(J_A_i) + cp.sum_squares(J_S_i - d_s)

                E_reg += self.config.lambda_arap * E_arap

            # [Eq. 23 & 26: Distortion Constraints on Active Set] [cite: 312, 330]
            distortion_constraints = []
            if len(active_indices) > 0:
                grad_phi_active = self.basis.jacobian(grid[active_indices])
                frames_active = self._frames[active_indices]

                for i, idx in enumerate(active_indices):
                    grad_u_x = grad_phi_active[i, :, 0] @ c_var[0, :]
                    grad_u_y = grad_phi_active[i, :, 1] @ c_var[0, :]
                    grad_v_x = grad_phi_active[i, :, 0] @ c_var[1, :]
                    grad_v_y = grad_phi_active[i, :, 1] @ c_var[1, :]

                    grad_u = cp.vstack([grad_u_x, grad_u_y])
                    grad_v = cp.vstack([grad_v_x, grad_v_y])
                    I_grad_v = cp.vstack([-grad_v[1], grad_v[0]])

                    J_S_i = (grad_u + I_grad_v) / 2.0
                    J_A_i = (grad_u - I_grad_v) / 2.0

                    t_i = cp.Variable()
                    s_i = cp.Variable()

                    distortion_constraints.append(cp.norm(J_S_i) <= t_i)
                    distortion_constraints.append(cp.norm(J_A_i) <= s_i)
                    distortion_constraints.append(t_i + s_i <= self.K_solver)

                    d_i = frames_active[i]
                    distortion_constraints.append(
                        J_S_i[0] * d_i[0] + J_S_i[1] * d_i[1] - s_i >= 1.0 / self.K_solver
                    )

            # [論文 Eq. 251] Objective
            objective = cp.Minimize(E_pos + E_reg)
            problem = cp.Problem(objective, pos_constraints + distortion_constraints)

            try:
                problem.solve(solver=cp.CLARABEL, verbose=False, max_iter=1000)

                if problem.status == "infeasible":
                    print(f"[SOCP] Problem infeasible! Attempting frame re-extraction...")

                    # 論文 Section 7 Discussion: 制約なし問題を解いてフレームを再抽出
                    problem_relaxed = cp.Problem(objective, pos_constraints)
                    problem_relaxed.solve(solver=cp.CLARABEL, verbose=False)

                    if problem_relaxed.status in ["optimal", "optimal_inaccurate"] and c_var.value is not None:
                        # 【修正】フレーム(d_i)だけを更新し、係数(self.coefficients)は更新しない！
                        # これにより、ジャンプせずに「安全な変形限界」で止まる
                        temp_coeffs = c_var.value
                        old_coeffs = self.coefficients

                        # 一時的に係数を適用して Jacobian を評価し、フレームを更新
                        self.coefficients = temp_coeffs
                        self.update_frames(grid)
                        self.coefficients = old_coeffs  # 元の安全な値に戻す（ジャンプを防ぐ）

                        print(f"[SOCP] Extracted new frames. Retrying in next iteration.")
                        continue  # 次のイテレーションへ
                    else:
                        print(f"[SOCP] Even relaxed problem failed. Freezing mesh.")
                        break  # リカバリ不可能なため、ループを抜けて現在の安全な形状を維持

                # 最適化が最適解以外（エラーなど）で終了した場合も維持
                if problem.status not in ["optimal", "optimal_inaccurate"]:
                    print(f"[SOCP] WARNING: status={problem.status}, active_set={len(active_indices)}")
                    break  # 安全な形状を維持

                # 【重要】正常に解け、歪み制約をクリアした保証のある係数のみを適用する
                if c_var.value is not None:
                    self.coefficients = c_var.value

            except Exception as e:
                print(f"[SOCP] ERROR: {e}")
                break  # エラー発生時も安全な形状を維持

            # 正常終了時、次ステップのためにフレームを更新
            self.update_frames(grid)

    def end_drag(self, target_handles: np.ndarray) -> bool:
        """
        [UI連携: onMouseUp]
        マウス操作が完了した段階で Strategy に基づいた検証を行う。
        Strategy 1 は再検証不要、Strategy 2 は必要に応じてグリッドを細かくする。
        """
        if self.coefficients is None:
            raise RuntimeError("Coefficients not initialized.")

        # Strategy に厳密な h の計算を依頼する
        strict_h = self.strategy.get_strict_h_after_drag(
            basis=self.basis,
            K_solver=self.K_solver,
            K_max=self.K_max,
            c=self.coefficients
        )

        # 現在のグリッドが理論値よりも粗い（安全ではない）場合のみ、再計算が発生する
        if strict_h < self.current_h:
            print(f"[{self.strategy.__class__.__name__}] Refining grid... (Old h: {self.current_h:.4f} -> New h: {strict_h:.4f})")

            # グリッドの再生成
            self.current_h = strict_h
            self.collocation_points = self.strategy.generate_grid(strict_h)

            # グリッドが変わったため、ここで初めて H_term を「再」計算する
            self._update_hessian_term()

            # [重要] グリッド点が新しくなりインデックスが変わったため、古い履歴を破棄する
            self._active_indices = np.array([], dtype=int)

            # 新しい細かいグリッド上で最適化を数ステップ回し、
            # この中で新しいグリッドに対する _active_indices を自然に再構築させる
            self.update_drag(target_handles, num_iterations=5)

            return True

        return False
