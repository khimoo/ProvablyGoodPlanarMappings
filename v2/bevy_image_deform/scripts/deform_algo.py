import numpy as np
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import cvxpy as cp


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

class Strategy2:
    """
    論文「Provably Good Planar Mappings」における Strategy 2 の完全な実装。

    インタラクティブ操作中は経験的に細かい固定グリッドを使用し、
    操作完了後（係数確定後）に、実際の係数ノルムから理論的に安全な h を逆算し、
    歪みの上界(Bounded Distortion)を数学的に保証する。
    """
    def __init__(self, domain_bounds: Tuple[float, float, float, float]):
        """
        Args:
            domain_bounds: (x_min, x_max, y_min, y_max) 領域のバウンディングボックス
        """
        self.bounds = domain_bounds

    def get_interactive_h(self, resolution: int = 200) -> float:
        """
        インタラクティブなマウス操作中に使用する、固定の h を返す。
        論文の「6 Results」に記載されている 200x200 グリッド等の初期設定用。
        """
        x_min, x_max, y_min, y_max = self.bounds
        width = x_max - x_min
        height = y_max - y_min

        # 長い方の辺を指定解像度で分割する間隔を返す
        max_span = max(width, height)
        return float(max_span / resolution)

    def generate_grid(self, h: float) -> np.ndarray:
        """
        指定された間隔 h に基づいて、選点（Collocation points）のグリッドを生成する。
        """
        x_min, x_max, y_min, y_max = self.bounds

        if h <= 0:
            raise ValueError("h must be strictly positive.")

        # arange の終端に +1e-8 を足すことで、バウンディングボックスの右端/上端もカバーする
        x_coords = np.arange(x_min, x_max + 1e-8, h)
        y_coords = np.arange(y_min, y_max + 1e-8, h)

        X, Y = np.meshgrid(x_coords, y_coords)
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T

        return grid_points

    def compute_strict_h(self,
                         basis: 'BasisFunction',
                         K_solver: float,
                         K_max: float,
                         c: np.ndarray) -> float:
        """
        マウス操作完了後（変形係数 c 確定後）に呼ばれ、
        論文の Strategy 2 (Eq. 14) に基づく厳密な h を計算する。

        Args:
            basis: 基底関数のインスタンス
            K_solver: 選点上でソルバーが実際に抑え込んだ歪みの最大値 (例: 2.0)
            K_max: 領域全体で数学的に保証したい最大の歪み許容値 (例: 5.0)
            c: 確定したマッピング係数行列。shape=(2, N_basis)

        Returns:
            float: 領域全体で歪みが K_max を超えないことを保証できる最大のグリッド間隔 h_strict
        """
        if K_max <= K_solver:
            raise ValueError(f"K_max ({K_max}) must be strictly greater than K_solver ({K_solver}).")

        # Eq. 14: アイソメトリック歪みのマージン計算
        # min( K_max - K,  1/K - 1/K_max )
        margin = min(K_max - K_solver, (1.0 / K_solver) - (1.0 / K_max))

        if margin <= 0:
            warnings.warn("Margin is zero or negative. Returning fallback h.")
            return 1e-5

        # 実際の係数ノルム |||c||| を計算 (Eq. 8 のマトリックス最大行和ノルム)
        # アフィン項（配列の最後の3列: [1, x, y]）の勾配は定数であり、
        # 変動（ヘッシアン）がゼロのため連続性係数に寄与しない。したがってノルム計算から除外する。
        if c.shape[1] >= 3:
            c_rbf = c[:, :-3]
        else:
            c_rbf = c

        max_coeff_norm = np.max(np.sum(np.abs(c_rbf), axis=1))

        # 恒等写像（RBF係数がすべてゼロ）の場合、歪みは一切発生しないため、理論上 h は無限大でよい
        if max_coeff_norm < 1e-12:
            return float('inf')

        # 基底関数の勾配の連続性係数 (h=1のときの係数C)
        # ガウス基底の場合、C = 1 / s^2 になる
        C = basis.compute_basis_gradient_modulus(1.0)

        if C <= 0:
            return float('inf')

        # Eq. 9 (ω = 2 * |||c||| * ω_∇F) と Eq. 14 を組み合わせた不等式:
        # h_strict * 2 * |||c||| * C <= margin を h_strict について解く
        h_strict = margin / (2.0 * max_coeff_norm * C)

        return float(h_strict)


class ProvablyGoodPlanarMapping(ABC):
    """
    論文「Provably Good Planar Mappings」のコアアルゴリズム（Algorithm 1）
    を提供する抽象基底クラス。
    """
    def __init__(self, basis: 'BasisFunction', strategy: 'Strategy2', K_solver: float = 2.0, K_max: float = 5.0, 
                 distortion_type: str = 'isometric'):
        self.basis = basis
        self.strategy = strategy
        self.K_solver = K_solver
        self.K_max = K_max
        self.distortion_type = distortion_type  # 'isometric' or 'conformal'
        self.coefficients: Optional[np.ndarray] = None
        self.collocation_points: Optional[np.ndarray] = None
        self.current_h: float = 0.0

        # --- 事前計算のキャッシュ用 ---
        self._B_mat: Optional[np.ndarray] = None
        self._H_term: Optional[np.ndarray] = None
        self._W_P: float = 1000.0
        self._W_R: float = 1.0
        self._lambda_reg: float = 0.01  # Regularization weight

        # 論文 Section 5.3 に基づく Active Set のヒステリシス閾値
        self.K_high = 0.1 + (K_solver * 0.9)
        self.K_low = 0.5 + (K_solver * 0.5)

        # Active Set とフレームのキャッシュ
        self._active_indices: np.ndarray = np.array([], dtype=int)
        self._frames: Optional[np.ndarray] = None  # 論文 Eq. 27 のフレームベクトル d_i
        
        # Conformal distortion用のδパラメータ (Eq. 12)
        self._delta_conf: float = 0.1

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
    
    def compute_J_S_and_J_A(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        論文 Eq. 19-20: Jacobianを similarity と anti-similarity 成分に分解
        J_S f(x) = (∇u(x) + I∇v(x)) / 2
        J_A f(x) = (∇u(x) - I∇v(x)) / 2
        
        Returns:
            J_S: shape (M, 2) - similarity part
            J_A: shape (M, 2) - anti-similarity part
        """
        if self.coefficients is None:
            raise RuntimeError("マッピング係数が初期化されていません。")
        
        grad_phi = self.basis.jacobian(coords)  # (M, N, 2)
        
        # ∇u と ∇v を計算
        grad_u = np.einsum('j, mjk -> mk', self.coefficients[0, :], grad_phi)  # (M, 2)
        grad_v = np.einsum('j, mjk -> mk', self.coefficients[1, :], grad_phi)  # (M, 2)
        
        # I は π/2 反時計回りの回転行列: [[0, -1], [1, 0]]
        I_grad_v = np.stack([-grad_v[:, 1], grad_v[:, 0]], axis=1)  # (M, 2)
        
        J_S = (grad_u + I_grad_v) / 2.0
        J_A = (grad_u - I_grad_v) / 2.0
        
        return J_S, J_A
    
    def compute_singular_values_from_J_S_J_A(self, J_S: np.ndarray, J_A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        論文 Eq. 20: J_S と J_A から特異値を計算
        Σ(x) = ||J_S f(x)|| + ||J_A f(x)||
        σ(x) = | ||J_S f(x)|| - ||J_A f(x)|| |
        """
        norm_J_S = np.linalg.norm(J_S, axis=1)
        norm_J_A = np.linalg.norm(J_A, axis=1)
        
        Sigma = norm_J_S + norm_J_A
        sigma = np.abs(norm_J_S - norm_J_A)
        
        return Sigma, sigma
    
    def compute_distortion(self, coords: np.ndarray) -> np.ndarray:
        """
        指定された座標での歪みを計算
        """
        J_S, J_A = self.compute_J_S_and_J_A(coords)
        Sigma, sigma = self.compute_singular_values_from_J_S_J_A(J_S, J_A)
        
        if self.distortion_type == 'isometric':
            # D_iso(x) = max{Σ(x), 1/σ(x)}
            sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
            distortion = np.maximum(Sigma, 1.0 / sigma_safe)
        elif self.distortion_type == 'conformal':
            # D_conf(x) = Σ(x) / σ(x)
            sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
            distortion = Sigma / sigma_safe
        else:
            raise ValueError(f"Unknown distortion type: {self.distortion_type}")
        
        return distortion
    @abstractmethod
    def initialize_mapping(self, src_handles: np.ndarray) -> None:
        pass

    @abstractmethod
    def update_mapping(self, target_handles: np.ndarray) -> None:
        """ 
        論文 Algorithm 1: 単一の最適化ステップを実行
        SOCP を使って係数 c を更新する
        """
        pass

    @abstractmethod
    def end_drag(self, target_handles: np.ndarray) -> bool:
        """ onMouseUp: ドラッグ終了時に呼ばれる（Strategy 2 の厳密な検証） """
        pass

class BetterFitwithGaussian(ProvablyGoodPlanarMapping):
    def __init__(self, domain_bounds: Tuple[float, float, float, float], s_param: float = 1.0, 
                 K_solver: float = 2.0, K_max: float = 5.0, distortion_type: str = 'isometric',
                 use_arap: bool = False):
        basis = GaussianRBF(s=s_param)
        strategy = Strategy2(domain_bounds)
        super().__init__(basis=basis, strategy=strategy, K_solver=K_solver, K_max=K_max, 
                        distortion_type=distortion_type)
        self.basis: GaussianRBF = basis
        self.use_arap = use_arap
        self._arap_points: Optional[np.ndarray] = None  # 論文 Eq. 32 の r_s
        self._arap_frames: Optional[np.ndarray] = None  # ARAP用のフレーム

    def initialize_mapping(self, src_handles: np.ndarray) -> None:
        """
        アプリケーション起動時・またはハンドル構成が変わった時に1回呼ばれる。
        論文 Algorithm 1 の "if first step then" に対応
        """
        self.basis.set_centers(src_handles)
        self.coefficients = self.basis.get_identity_coefficients(src_handles)

        # インタラクティブ用の初期グリッド
        # 輪郭フィルタリングを考慮して、より細かいグリッドを使用
        # 論文では 200x200 だが、フィルタリング後も十分な密度を保つため 300x300 に
        self.current_h = self.strategy.get_interactive_h(resolution=300)
        self.collocation_points = self.strategy.generate_grid(self.current_h)

        # 事前計算: 位置拘束行列 (B_mat) は src_handles (不変) にのみ依存する
        self._B_mat = self.basis.evaluate(src_handles)

        # ヘッシアン項の事前計算
        self._update_hessian_term()

        # 論文 Algorithm 1: "if first step then" - フレームを (1,0) で初期化
        M = len(self.collocation_points)
        self._frames = np.zeros((M, 2))
        self._frames[:, 0] = 1.0  # d_i = (1, 0)
        
        # Active set を空で初期化
        self._active_indices = np.array([], dtype=int)
        
        # ARAP エネルギー用のサンプル点 (論文 Eq. 32)
        if self.use_arap:
            # 等間隔にサンプル点を配置
            n_samples = min(100, len(self.collocation_points))
            indices = np.linspace(0, len(self.collocation_points) - 1, n_samples, dtype=int)
            self._arap_points = self.collocation_points[indices]
            self._arap_frames = np.zeros((n_samples, 2))
            self._arap_frames[:, 0] = 1.0
        
        print(f"Initialized with {len(self.collocation_points)} collocation points (h={self.current_h:.4f})")

    def _update_hessian_term(self) -> None:
        """ グリッドが更新されたときのみ呼ばれる内部メソッド """
        if self.collocation_points is None:
            raise RuntimeError("Collocation points not initialized.")
        collocation_points = self.collocation_points
        N_basis = self.basis.get_basis_count()
        hess = self.basis.hessian(collocation_points)
        H_mat = np.transpose(hess, (0, 2, 3, 1)).reshape(-1, N_basis)

        # 論文 Eq. 31: 数値積分のための離散化
        # 面積要素を考慮（グリッド間隔の2乗）
        area_element = self.current_h ** 2
        self._H_term = (H_mat.T @ H_mat) * area_element

    def _find_local_maxima(self, distortions: np.ndarray) -> np.ndarray:
        """
        論文 Section 5.3: グリッド上の歪みの局所最大値を見つける
        """
        if self.collocation_points is None:
            return np.array([], dtype=int)
        
        # グリッドの形状を推定（正方形グリッドを仮定）
        n_points = len(distortions)
        grid_size = int(np.sqrt(n_points))
        
        if grid_size * grid_size != n_points:
            # 正方形でない場合は全ての点を候補とする
            return np.arange(n_points)
        
        # 2Dグリッドに reshape
        dist_grid = distortions.reshape(grid_size, grid_size)
        
        # 局所最大値を見つける（8近傍）
        local_max_mask = np.zeros_like(dist_grid, dtype=bool)
        
        for i in range(grid_size):
            for j in range(grid_size):
                val = dist_grid[i, j]
                is_max = True
                
                # 8近傍をチェック
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid_size and 0 <= nj < grid_size:
                            if dist_grid[ni, nj] > val:
                                is_max = False
                                break
                    if not is_max:
                        break
                
                local_max_mask[i, j] = is_max
        
        return np.where(local_max_mask.ravel())[0]
    
    def _update_active_set(self) -> None:
        """
        論文 Section 5.3: Active set の更新（改良版）
        - 歪みの局所最大値で K_high を超えるものを追加
        - K_low を下回るものを削除
        - 折り畳み（fold-over）が発生している点は常に active
        - 折り畳みに近い点（σ が小さい）も予防的に追加
        """
        if self.collocation_points is None:
            return
        
        grid = self.collocation_points
        distortions = self.compute_distortion(grid)
        
        # 折り畳みチェック（σ ≤ 0）と予防的チェック（σ が小さい）
        J_S, J_A = self.compute_J_S_and_J_A(grid)
        _, sigma = self.compute_singular_values_from_J_S_J_A(J_S, J_A)
        is_foldover = sigma <= 1e-8
        is_near_foldover = (sigma > 1e-8) & (sigma < 0.1)  # 予防的に追加
        
        # 局所最大値を見つける
        local_max_indices = self._find_local_maxima(distortions)
        
        # 現在の active set を boolean mask に変換
        active_mask = np.zeros(len(grid), dtype=bool)
        active_mask[self._active_indices] = True
        
        # 追加: 局所最大値で K_high を超えるもの、折り畳み、または折り畳みに近いもの
        is_above_high = distortions > self.K_high
        should_add = np.zeros(len(grid), dtype=bool)
        should_add[local_max_indices] = is_above_high[local_max_indices]
        should_add |= is_foldover
        should_add |= is_near_foldover  # 予防的に追加
        
        active_mask |= should_add
        
        # 削除: K_low を下回り、かつ折り畳みでなく、σ が十分大きいもの
        is_below_low = distortions < self.K_low
        is_safe = sigma > 0.5  # σ が十分大きい
        should_remove = is_below_low & ~is_foldover & is_safe
        active_mask &= ~should_remove
        
        self._active_indices = np.where(active_mask)[0]
        
        # 論文 Section 5.3: 安定性のため、等間隔に配置された点を常に active に保つ
        # ここでは簡単のため、グリッドの角と中心を常に active にする
        if len(grid) > 0:
            grid_size = int(np.sqrt(len(grid)))
            if grid_size * grid_size == len(grid):
                # 4隅と中心
                corners = [0, grid_size - 1, grid_size * (grid_size - 1), grid_size * grid_size - 1]
                center = grid_size * (grid_size // 2) + (grid_size // 2)
                stable_points = corners + [center]
                self._active_indices = np.unique(np.concatenate([self._active_indices, stable_points]))
        
        # Active set が大きすぎる場合は制限（パフォーマンスのため）
        max_active = 5000  # 最大500点
        if len(self._active_indices) > max_active:
            # 歪みが最も大きい点を優先
            distortions_active = distortions[self._active_indices]
            top_indices = np.argsort(distortions_active)[-max_active:]
            self._active_indices = self._active_indices[top_indices]
            warnings.warn(f"Active set limited to {max_active} points (was {len(self._active_indices)})")

    def update_mapping(self, target_handles: np.ndarray) -> None:
        """
        論文 Algorithm 1: SOCP を使った単一の最適化ステップ
        
        Optimization (論文 Section 5):
        - Active set の更新
        - SOCP 問題の構築と解決
        - フレームの更新
        """
        if self.collocation_points is None or self._B_mat is None or self._H_term is None:
            raise RuntimeError("Mapping not initialized.")
        
        # === Active Set の更新 ===
        self._update_active_set()
        
        grid = self.collocation_points
        N_basis = self.basis.get_basis_count()
        n_handles = target_handles.shape[0]
        active_indices = self._active_indices
        n_active = len(active_indices)
        
        # === SOCP 問題の構築 ===
        # 変数: c (2 x N_basis の係数行列を flatten)
        c_var = cp.Variable((2, N_basis))
        
        constraints = []
        objective_terms = []
        
        # --- 位置拘束エネルギー (Eq. 29-30) ---
        # E_pos = Σ_l ||f(p_l) - q_l|| = Σ_l ||Σ_i c_i f_i(p_l) - q_l||
        B_mat = self._B_mat  # shape: (n_handles, N_basis)
        
        # 各ハンドルに対して補助変数 r_l を導入（非負）
        r_vars = []
        for l in range(n_handles):
            r_l = cp.Variable(nonneg=True)
            r_vars.append(r_l)
            
            # ||c @ B_mat[l, :] - q_l|| <= r_l (Eq. 30)
            # c @ B_mat[l, :] は (2,) ベクトル
            residual = c_var @ B_mat[l, :] - target_handles[l, :]
            constraints.append(cp.norm(residual, 2) <= r_l)
            
            objective_terms.append(self._W_P * r_l)
        
        # --- 正則化エネルギー: Biharmonic (Eq. 31) ---
        # E_bh = ||H_mat @ c.T||_F^2 (Frobenius norm)
        # H_term = H_mat.T @ H_mat が事前計算済み
        # E_bh = trace(c @ H_term @ c.T) = sum_{d} c[d,:] @ H_term @ c[d,:].T
        for d in range(2):
            bh_term = cp.quad_form(c_var[d, :], self._H_term)
            objective_terms.append(self._lambda_reg * self._W_R * bh_term)
        
        # --- 正則化エネルギー: ARAP (Eq. 33) ---
        # E_arap = Σ_s (||J_A f(r_s)||^2 + ||J_S f(r_s) - d_s||^2)
        if self.use_arap and self._arap_points is not None:
            grad_phi_arap = self.basis.jacobian(self._arap_points)  # (n_arap, N_basis, 2)
            
            for s in range(len(self._arap_points)):
                grad_phi_s = grad_phi_arap[s, :, :]  # (N_basis, 2)
                
                # ∇u と ∇v
                grad_u = c_var[0, :] @ grad_phi_s
                grad_v = c_var[1, :] @ grad_phi_s
                
                # I∇v: I = [[0, -1], [1, 0]]
                I_grad_v = cp.hstack([-grad_v[1], grad_v[0]])
                
                # J_S と J_A
                J_S = (grad_u + I_grad_v) / 2.0
                J_A = (grad_u - I_grad_v) / 2.0
                
                # フレーム d_s
                d_s = self._arap_frames[s, :]
                
                # ||J_A||^2 + ||J_S - d_s||^2
                arap_term = cp.sum_squares(J_A) + cp.sum_squares(J_S - d_s)
                objective_terms.append(self._lambda_reg * arap_term)
        
        # --- 歪み制約 (Active set のみ) ---
        if n_active > 0:
            # Active な collocation points での基底関数の勾配を計算
            grad_phi_active = self.basis.jacobian(grid[active_indices])  # (n_active, N_basis, 2)
            
            for idx, active_idx in enumerate(active_indices):
                grad_phi_i = grad_phi_active[idx, :, :]  # (N_basis, 2)
                
                # ∇u と ∇v を計算
                grad_u = c_var[0, :] @ grad_phi_i  # (2,) ベクトル
                grad_v = c_var[1, :] @ grad_phi_i  # (2,) ベクトル
                
                # I∇v を計算: I = [[0, -1], [1, 0]]
                I_grad_v = cp.hstack([-grad_v[1], grad_v[0]])
                
                # J_S と J_A (Eq. 19)
                J_S = (grad_u + I_grad_v) / 2.0
                J_A = (grad_u - I_grad_v) / 2.0
                
                # フレームベクトル d_i (前ステップから)
                d_i = self._frames[active_idx, :]
                
                if self.distortion_type == 'isometric':
                    # 論文 Eq. 21-26: Isometric distortion constraints
                    # 補助変数 t_i, s_i (非負)
                    t_i = cp.Variable(nonneg=True)
                    s_i = cp.Variable(nonneg=True)
                    
                    # Eq. 23a-c
                    constraints.append(cp.norm(J_S, 2) <= t_i)
                    constraints.append(cp.norm(J_A, 2) <= s_i)
                    constraints.append(t_i + s_i <= self.K_solver)
                    
                    # Eq. 26 (convexified version of Eq. 22)
                    # J_S @ d_i は内積なのでスカラー
                    constraints.append(J_S @ d_i - s_i >= 1.0 / self.K_solver)
                    
                elif self.distortion_type == 'conformal':
                    # 論文 Eq. 28: Conformal distortion constraints
                    K = self.K_solver
                    delta = self._delta_conf
                    
                    # J_S @ d_i は内積（スカラー）
                    J_S_dot_d = J_S @ d_i
                    
                    # Eq. 28a
                    constraints.append(
                        cp.norm(J_A, 2) <= ((K - 1) / (K + 1)) * J_S_dot_d
                    )
                    
                    # Eq. 28b
                    constraints.append(
                        cp.norm(J_A, 2) <= J_S_dot_d - delta
                    )
        
        # === 目的関数 ===
        objective = cp.Minimize(cp.sum(objective_terms))
        
        # === SOCP を解く ===
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                warnings.warn(f"SOCP solver status: {problem.status}. Objective: {problem.value}")
                # 解が見つからない場合は係数を更新しない
                return
            
            # 係数を更新
            self.coefficients = c_var.value
            
            # デバッグ: 制約の充足状況を確認
            if n_active > 0:
                violations = []
                for c in constraints:
                    if hasattr(c, 'violation'):
                        viol = c.violation()
                        if viol is not None and viol > 1e-6:
                            violations.append(viol)
                if violations:
                    warnings.warn(f"Constraint violations detected: max={max(violations):.6f}")
            
        except Exception as e:
            warnings.warn(f"SOCP solver failed: {e}")
            return
        
        # === Postprocessing: フレームの更新 (Eq. 27) ===
        if n_active > 0:
            J_S_active, _ = self.compute_J_S_and_J_A(grid[active_indices])
            norm_J_S = np.linalg.norm(J_S_active, axis=1, keepdims=True)
            norm_J_S = np.where(norm_J_S > 1e-8, norm_J_S, 1.0)  # ゼロ除算回避
            self._frames[active_indices] = J_S_active / norm_J_S
        
        # ARAP フレームの更新
        if self.use_arap and self._arap_points is not None:
            J_S_arap, _ = self.compute_J_S_and_J_A(self._arap_points)
            norm_J_S_arap = np.linalg.norm(J_S_arap, axis=1, keepdims=True)
            norm_J_S_arap = np.where(norm_J_S_arap > 1e-8, norm_J_S_arap, 1.0)
            self._arap_frames = J_S_arap / norm_J_S_arap

    def end_drag(self, target_handles: np.ndarray) -> bool:
        """
        論文 Section 4: Strategy 2 を用いた検証
        マウス操作が完了した段階で、理論的に保証された h を計算し、
        必要に応じてグリッドを細分化する。
        """
        if self.coefficients is None:
            raise RuntimeError("Coefficients not initialized.")

        strict_h = self.strategy.compute_strict_h(
            basis=self.basis,
            K_solver=self.K_solver,
            K_max=self.K_max,
            c=self.coefficients
        )

        # 現在のグリッドが理論値よりも粗い（安全ではない）場合のみ、再計算が発生する
        if strict_h < self.current_h:
            print(f"[Strategy 2] Refining grid... (Old h: {self.current_h:.4f} -> New h: {strict_h:.4f})")

            # グリッドの再生成
            self.current_h = strict_h
            self.collocation_points = self.strategy.generate_grid(strict_h)

            # グリッドが変わったため、H_term を再計算
            self._update_hessian_term()

            # フレームを再初期化
            M = len(self.collocation_points)
            self._frames = np.zeros((M, 2))
            self._frames[:, 0] = 1.0
            
            # Active set をリセット
            self._active_indices = np.array([], dtype=int)
            
            # ARAP サンプル点も更新
            if self.use_arap:
                n_samples = min(100, len(self.collocation_points))
                indices = np.linspace(0, len(self.collocation_points) - 1, n_samples, dtype=int)
                self._arap_points = self.collocation_points[indices]
                self._arap_frames = np.zeros((n_samples, 2))
                self._arap_frames[:, 0] = 1.0

            # 新しいグリッド上で最適化を実行
            # 論文では単一ステップだが、実用上は収束まで繰り返す
            for _ in range(5):
                self.update_mapping(target_handles)

            return True

        return False
