import numpy as np
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List


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
    論文「Provably Good Planar Mappings」のコアアルゴリズム（Algorithm 1, Local-Global Solver）
    を提供する抽象基底クラス。

    GuaranteeStrategy を通じて Strategy 1（事前保証）と Strategy 2（事後検証）を
    実行時に切り替え可能にする。
    """
    def __init__(self, basis: 'BasisFunction', guarantee_strategy: 'GuaranteeStrategy', K_solver: float = 2.0, K_max: float = 5.0):
        self.basis = basis
        self.strategy = guarantee_strategy
        self.K_solver = K_solver
        self.K_max = K_max
        self.coefficients: Optional[np.ndarray] = None
        self.collocation_points: Optional[np.ndarray] = None
        self.current_h: float = 0.0

        # --- 事前計算のキャッシュ用 ---
        self._B_mat: Optional[np.ndarray] = None
        self._H_term: Optional[np.ndarray] = None
        self._W_P: float = 1000
        self._W_R: float = 1.0

        # [修正点1] 論文 Section 5.3 に基づく Active Set のヒステリシス閾値
        self.K_high = 0.1 + (K_solver * 0.9)
        self.K_low = 0.5 + (K_solver * 0.5)

        # [修正点1] フレーム間で Active Set を維持するためのキャッシュ
        self._active_indices: np.ndarray = np.array([], dtype=int)

    def evaluate_map(self, coords: np.ndarray) -> np.ndarray:
        """ f(x) = c * Phi(x) を評価して変換後の座標を返す """
        if self.coefficients is None:
            raise RuntimeError("マッピング係数が初期化されていません。")
        phi = self.basis.evaluate(coords)
        return phi @ self.coefficients.T

    def evaluate_jacobian(self, coords: np.ndarray) -> np.ndarray:
        """ ヤコビアン J_f(x) を計算する (Eq. 4) """
        if self.coefficients is None:
            raise RuntimeError("マッピング係数が初期化されていません。")
        grad_phi = self.basis.jacobian(coords)
        # cは(2, N)、grad_phiは(M, N, 2)。J[m, d, k] = sum_j c[d, j] * grad_phi[m, j, k]
        J = np.einsum('dj, mjk -> mdk', self.coefficients, grad_phi)
        return J

    def project_jacobians(self, J: np.ndarray, K: float) -> np.ndarray:
        """
        [論文 Algorithm 1: Projection on T_K]
        ヤコビアン行列の集合を指定されたアイソメトリック歪み K の範囲に射影する。
        折り畳み（fold-over）を防ぐため、行列式が負のものは反転して正の領域へ押し上げる。
        """
        # SVD: J = U * S * Vh  (numpyでは Vh は V^T)
        U, S, Vh = np.linalg.svd(J)

        # det(U * V^T) をチェックし、回転を保証する (Section 5.1 "positive determinant" constraint)
        dets = np.linalg.det(U @ Vh)
        neg_dets = dets < 0

        # 行列式が負（折り畳み発生）の場合、最小特異値の符号を反転させる
        S[neg_dets, 1] *= -1
        U[neg_dets, :, 1] *= -1

        # Eq. 21 に基づき、特異値を安全な領域 [1/K, K] にクランプする (フロベニウスノルム上での最適射影)
        S_proj = np.clip(S, 1.0 / K, K)

        # 射影されたヤコビアン X を再構築: X = U * diag(S_proj) * Vh
        X = (U * S_proj[:, np.newaxis, :]) @ Vh
        return X
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
    def __init__(self, domain_bounds: Tuple[float, float, float, float], guarantee_strategy: Optional['GuaranteeStrategy'] = None, s_param: float = 1.0, K_solver: float = 2.0, K_max: float = 5.0):
        """
        Args:
            domain_bounds: (x_min, x_max, y_min, y_max) 領域のバウンディングボックス
            guarantee_strategy: GuaranteeStrategy のインスタンス（デフォルトは Strategy2）
            s_param: ガウス基底関数のスケールパラメータ
            K_solver: 選点上での歪み上限
            K_max: 領域全体での歪み上限
        """
        basis = GaussianRBF(s=s_param)
        if guarantee_strategy is None:
            guarantee_strategy = Strategy2(domain_bounds)
        super().__init__(basis=basis, guarantee_strategy=guarantee_strategy, K_solver=K_solver, K_max=K_max)
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
        マウスが動くたびに呼ばれる。キャッシュされた行列を使って高速に解く。
        """
        if self.collocation_points is None:
            raise RuntimeError("Collocation points not initialized.")
        if self._B_mat is None:
            raise RuntimeError("B_mat not initialized.")
        if self._H_term is None:
            raise RuntimeError("H_term not initialized.")

        grid = self.collocation_points
        B_mat = self._B_mat
        H_term = self._H_term
        N_basis = self.basis.get_basis_count()
        area_element = self.current_h ** 2

        # ループ回数はドラッグ中なので少なめ（1〜2回）で十分
        for _ in range(num_iterations):
            # === [Active Set 抽出] ===
            J_all = self.evaluate_jacobian(grid)
            U, S, Vh = np.linalg.svd(J_all)
            dets = np.linalg.det(U @ Vh)
            neg_dets = dets < 0
            S[neg_dets, 1] *= -1

            sig2_safe = np.where(S[:, 1] > 1e-8, S[:, 1], 1e-8)
            sig2_safe[S[:, 1] <= 0] = 1e-8
            distortions = np.maximum(S[:, 0], 1.0 / sig2_safe)

            # === [修正点1: 論文通りのヒステリシス Active Set 更新] ===
            is_foldover = S[:, 1] <= 0
            is_above_high = distortions > self.K_high
            is_below_low = distortions < self.K_low

            # 現在のインデックスを boolean マスクに変換
            active_mask = np.zeros(len(grid), dtype=bool)
            active_mask[self._active_indices] = True

            # K_high を超えたもの、または折りたたまれたものを追加
            active_mask |= is_above_high | is_foldover

            # K_low を下回った「かつ」折りたたまれていないものを削除
            active_mask &= ~(is_below_low & ~is_foldover)

            self._active_indices = np.where(active_mask)[0]
            active_indices = self._active_indices

            # === [Local Step] ===
            if len(active_indices) > 0:
                J_active = J_all[active_indices]
                X_active = self.project_jacobians(J_active, self.K_solver)

                grad_phi_active = self.basis.jacobian(grid[active_indices])
                A_mat = np.transpose(grad_phi_active, (0, 2, 1)).reshape(-1, N_basis)
                # [修正点2] Distortion Energy (Eq. 5) も積分ベースであるため h^2 を掛ける
                A_term = (A_mat.T @ A_mat) * area_element
            else:
                A_mat = np.zeros((0, N_basis))
                X_active = np.zeros((0, 2, 2))
                A_term = np.zeros((N_basis, N_basis))

            # === [Global Step] ===
            # M行列は事前計算済みの H_term と B_mat を使用して高速に構築
            M = A_term + self._W_P * (B_mat.T @ B_mat) + self._W_R * H_term
            new_c = np.zeros_like(self.coefficients)

            for d in range(2):
                if len(active_indices) > 0:
                    X_d = X_active[:, d, :].reshape(-1)
                    # 右辺も A_mat の転置がかかるため面積要素を掛ける
                    rhs_D = (A_mat.T @ X_d) * area_element
                else:
                    rhs_D = np.zeros(N_basis)

                rhs_P = self._W_P * (B_mat.T @ target_handles[:, d])
                new_c[d, :] = np.linalg.solve(M, rhs_D + rhs_P)

            self.coefficients = new_c

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
