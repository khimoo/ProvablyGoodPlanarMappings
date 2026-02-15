#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cvxpy as cp
import numpy as np
from abc import ABCMeta, abstractmethod
import warnings
from scipy.spatial import Delaunay

class RBF2DandAffine:
    def __init__(self, epsilon=1.0):
        self.epsilon = float(epsilon)
        self.src = None
        self.Phi = None
        self.GradPhi = None

    def set_centers(self, centers):
        self.src = np.asarray(centers, dtype=float)

    def _rbf(self, r):
        return np.exp(-(r / self.epsilon) ** 2)

    def basis_functions(self, x):
        """x -> [phi_1(x),...,phi_n(x), 1, x, y]  （n+3次元）"""
        if self.src is None:
            raise RuntimeError("self.src is not set.")
        x = np.asarray(x, dtype=float)
        r = np.sqrt(np.sum((self.src - x) ** 2, axis=1))
        vals = self._rbf(r)                            # (n,)
        return np.concatenate([vals, [1.0, x[0], x[1]]])  # (n+3,)

    def basis_functions_with_grad(self, x):
        """x -> d/dx [phi_1,...,phi_n, 1, x, y]  （(n+3,2)）"""
        if self.src is None:
            raise RuntimeError("self.src is not set.")
        x = np.asarray(x, dtype=float)
        diff = x[None, :] - self.src           # (n,2)
        r2 = np.sum(diff * diff, axis=1)       # (n,)
        val = np.exp(-r2 / (self.epsilon**2))  # (n,)
        grad_rbf = -2.0 * diff / (self.epsilon**2) * val[:, None]  # (n,2)
        # affine の勾配は [1,0], [0,1]、定数は [0,0]
        grad_affine = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # (3,2)
        return np.vstack([grad_rbf, grad_affine])  # (n+3,2)

    def precompute(self, vertices):
        """vertices 全点で Phi, GradPhi を作る。Phi:(m,n+3), GradPhi:(m,n+3,2)"""
        if self.src is None:
            raise RuntimeError("self.src must be set before precompute().")
        Z = np.asarray(vertices, dtype=float)
        m = Z.shape[0]
        n = self.src.shape[0]
        Phi = np.zeros((m, n + 3), dtype=float)
        GradPhi = np.zeros((m, n + 3, 2), dtype=float)
        for j, x in enumerate(Z):
            Phi[j] = self.basis_functions(x)
            GradPhi[j] = self.basis_functions_with_grad(x)
        self.Phi = Phi
        self.GradPhi = GradPhi


class ProvablyGoodPlanarMapping(metaclass=ABCMeta):
    def __init__(self, vertices, src, K):
        # 軽量化：データ保存と初期状態のセットのみ
        self.vertices = np.asarray(vertices, dtype=float)
        self.src = np.asarray(src, dtype=float)
        self.dst = np.asarray(src, dtype=float) # 最初のステップは恒等写像!!!!

        m = self.vertices.shape[0]
        # n = self.src.shape[0]  # 必要なら使うが今は保持のみ

        # coefficients (2 x (n+3)) を None にしておく（初期化は別メソッドで）
        self.c = None

        # frame directions di の領域だけ確保（値は initialize で入れる）
        self.di = np.zeros((m, 2), dtype=float)

        # precompute キャッシュ（precompute() 呼び出しで埋める）
        self.Phi = None
        self.GradPhi = None

        # sets & thresholds（activated は initialize で作る）
        self.activated = []
        self.farthest_points = None
        self.K = K
        # K_high, K_low の初期値は常套式を残す（必要なら外部から上書き可）
        self.K_high = 0.5 + 0.9 * self.K
        self.K_low  = 0.5 + 0.5 * self.K

    # ---- ここを “x だけ” に合わせる ----
    @abstractmethod
    def basis_functions(self, x):
        pass

    @abstractmethod
    def basis_functions_with_grad(self, x):
        pass

    @abstractmethod
    def precompute(self):
        pass


    def farthest_point_sampling(self, k, initial_index=None, rng=None):
        """
        FPS (Farthest Point Sampling) on self.vertices.

        This fills self.farthest_points (論文中の Z'') with the selected indices.

        Parameters
        ----------
        k : int
            Number of samples to pick (k <= len(vertices)).
        initial_index : int or None
            If None, choose a deterministic start (e.g., min x). Otherwise use provided index.
        rng : np.random.Generator or None
            Random generator for reproducibility.

        Returns
        -------
        indices : ndarray, shape (k,)
            Indices of selected points in self.vertices (and stored in self.farthest_points).
        """
        points = self.vertices
        m = points.shape[0]
        if k <= 0:
            self.farthest_points = np.array([], dtype=int)
            return self.farthest_points
        if k >= m:
            self.farthest_points = np.arange(m, dtype=int)
            return self.farthest_points

        if rng is None:
            rng = np.random.default_rng()

        if initial_index is None:
            # pick a deterministic point: min x-coordinate
            initial_index = int(np.argmin(points[:, 0]))

        indices = np.empty(k, dtype=int)
        indices[0] = initial_index

        # initialize distances to first chosen point
        chosen = points[initial_index]
        diff = points - chosen
        min_d2 = np.sum(diff * diff, axis=1)

        for i in range(1, k):
            # pick farthest point
            next_idx = int(np.argmax(min_d2))
            indices[i] = next_idx
            new_chosen = points[next_idx]
            d2 = np.sum((points - new_chosen) ** 2, axis=1)
            min_d2 = np.minimum(min_d2, d2)

        self.farthest_points = indices
        return self.farthest_points

    def compute_isometric_distortion(self, c, indices=None, eps_sigma=1e-10):
        """
        Compute conformal/isometric distortion K at selected collocation points.

        Parameters
        ----------
        c : ndarray, shape (2, n)
            Coefficients for u and v components (rows: [u, v], cols over n bases).
        indices : array-like or None
            Indices of collocation points to evaluate. If None, use all.
        eps_sigma : float
            Small positive value for numerical stability in denominators.

        Returns
        -------
        K : ndarray, shape (len(indices),)
            Distortion K >= 1 at the selected points.
        extras : dict
            {
            "Js": |f_z|,           # similarity norm
            "Ja": |f_bar|,         # anti-similarity norm
            "mu": |f_bar|/|f_z|,   # Beltrami magnitude
            "J" : (len, 2, 2)      # Jacobians at points
            }
        """
        if self.GradPhi is None:
            raise RuntimeError("GradPhi is not precomputed. Call precompute() first.")

        m, n, _ = self.GradPhi.shape
        if c.shape != (2, n):
            raise ValueError(f"c must have shape (2, {n}), got {c.shape}")

        if indices is None:
            indices = np.arange(m, dtype=int)
        else:
            indices = np.asarray(indices, dtype=int)

        # gather gradients at selected points: (k, n, 2)
        G = self.GradPhi[indices, :, :]          # ∂phi/∂x = G[...,0], ∂phi/∂y = G[...,1]
        # u_x, u_y
        ux = np.einsum('kn,kn->k', G[:, :, 0], np.broadcast_to(c[0, :], (len(indices), n)))
        uy = np.einsum('kn,kn->k', G[:, :, 1], np.broadcast_to(c[0, :], (len(indices), n)))
        # v_x, v_y
        vx = np.einsum('kn,kn->k', G[:, :, 0], np.broadcast_to(c[1, :], (len(indices), n)))
        vy = np.einsum('kn,kn->k', G[:, :, 1], np.broadcast_to(c[1, :], (len(indices), n)))

        # Jacobian entries
        a = ux  # ∂u/∂x
        b = vx  # ∂v/∂x
        c_ = uy # ∂u/∂y
        d = vy  # ∂v/∂y

        # complex derivatives
        fz_real  = 0.5*(a + d)
        fz_imag  = 0.5*(b - c_)
        fzb_real = 0.5*(a - d)
        fzb_imag = 0.5*(c_ + b)

        abs_fz  = np.sqrt(fz_real**2  + fz_imag**2)
        abs_fzb = np.sqrt(fzb_real**2 + fzb_imag**2)

        # warn if eps_sigma is actually used because |f_z| < eps_sigma somewhere
        unstable_mask = abs_fz < eps_sigma
        if np.any(unstable_mask):
            cnt = int(np.sum(unstable_mask))
            total = len(abs_fz)
            warnings.warn(
                f"[compute_isometric_distortion] Numerical stabilizer eps_sigma={eps_sigma} "
                f"was used at {cnt}/{total} points because |f_z| < eps_sigma. "
                "This may indicate near-singular Jacobian at those points; consider inspecting "
                "coefficients, Phi/GradPhi or increasing eps_sigma.",
                RuntimeWarning, stacklevel=2
            )

        # Beltrami magnitude and distortion
        mu = abs_fzb / np.maximum(abs_fz, eps_sigma)
        # K = (1+|μ|)/(1-|μ|)
        # warn if mu is extremely close to 1 (leading to huge K)
        close_to_one_mask = mu >= 1.0 - 1e-12
        if np.any(close_to_one_mask):
            cnt = int(np.sum(close_to_one_mask))
            total = len(mu)
            warnings.warn(
                f"[compute_isometric_distortion] Beltrami magnitude |mu| is numerically >= 1 at "
                f"{cnt}/{total} points (mu values near 1), resulting in extremely large distortion K. "
                "This may indicate severe local inversion or numerical issues.",
                RuntimeWarning, stacklevel=2
            )

        K = (1.0 + mu) / np.maximum(1.0 - mu, eps_sigma)

        # pack Jacobians per point for debugging/inspection
        J = np.stack([np.stack([a, b], axis=-1),
                    np.stack([c_, d], axis=-1)], axis=-2)  # shape (k,2,2)

        extras = {"Js": abs_fz, "Ja": abs_fzb, "mu": mu, "J": J}
        return K, extras

    @abstractmethod
    def update_active_set(self):
        pass

    @abstractmethod
    def optimize_step(self):
        pass

    @abstractmethod
    def postprocess(self):
        pass


class BetterFitwithGaussianRBF(ProvablyGoodPlanarMapping, RBF2DandAffine):
    def __init__(self, vertices, src, K, epsilon=1, fps_k=50, lambda_biharmonic=1e-4):
        RBF2DandAffine.__init__(self, epsilon=epsilon)
        self.set_centers(src)

        # 親クラスの軽量 init（状態のみセット）
        ProvablyGoodPlanarMapping.__init__(self, vertices, src, K)

        # 固有パラメータ
        self.epsilon = epsilon
        self.fps_k = fps_k

        # Biharmonic 正則化の重み（係数空間での二次形式を作る）
        self.lambda_biharmonic = float(lambda_biharmonic)
        self.H_reg = None              # (n+3, n+3) の二次形式行列を後で作る
        self.mesh_simplices = None     # Delaunay 三角形情報を保存（可視化に使ってもよい）

        # デフォルトで di は (1,0) にしておく（ただし initialize で明示的に上書きされる）
        m = self.vertices.shape[0]
        for j in range(m):
            self.di[j] = np.array([1.0, 0.0], dtype=float)

        # self.Phi / self.GradPhi / activated / farthest_points は initialize() で構築する
        # self.c は compute_initial_coefficients_from_least_squares() や _set_identity_mapping() で作る

    def initialize_first_step(self):
        """
        Algorithm 1 の 'if first step then ...' に対応する初期化。
        （フラグ無し版：first step でのみ呼ぶ想定）
        """
        fps_k = self.fps_k

        # --- 1) Phi / GradPhi の準備（未計算なら一度だけ計算） ---
        if self.Phi is None or self.GradPhi is None or self.H_reg is None:
            print("[INFO] Precomputing Phi, GradPhi and Biharmonic matrix H_reg ...")
            self.precompute()  # ここで H_reg も構築される

        # --- 2) di の初期化 (all -> (1,0)) ---
        m = self.vertices.shape[0]
        # 再実行しても安全なように再割当（idempotent）
        self.di = np.tile(np.array([1.0, 0.0], dtype=float), (m, 1))

        # --- 3) active set / farthest sampling ---
        self.activated = []
        # farthest_point_sampling が self.farthest_points を設定する前提
        self.farthest_point_sampling(fps_k)

        self._set_identity_mapping()

        print("[INFO] initialize_first_step done.")

    # ---- ABC 実装（RBF2DandAffine を委譲）----
    def basis_functions(self, x):
        return RBF2DandAffine.basis_functions(self, x)

    def basis_functions_with_grad(self, x):
        return RBF2DandAffine.basis_functions_with_grad(self, x)

    def precompute(self):
        """
        既存の Phi/GradPhi を作る処理に加え、メッシュから単純なグラフラプラシアン
        を作り、H_reg = Phi^T (L^T L) Phi を構成する。
        """
        # まず基本の Phi/GradPhi を作る（RBF2DandAffine.precompute に委譲）
        RBF2DandAffine.precompute(self, self.vertices)  # sets self.Phi, self.GradPhi

        m = self.vertices.shape[0]
        # Delaunay で三角形を作る（simple, robust enough）
        try:
            tris = Delaunay(self.vertices)
            simplices = tris.simplices
        except Exception as e:
            # Delaunay が失敗したら空にしておく（極端な状況）
            simplices = np.empty((0, 3), dtype=int)

        self.mesh_simplices = simplices

        # --- adjacency (m x m) を作る（各三角形の辺を 1 とする） ---
        A = np.zeros((m, m), dtype=float)
        for tri in simplices:
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            A[i, j] = A[j, i] = 1.0
            A[j, k] = A[k, j] = 1.0
            A[k, i] = A[i, k] = 1.0

        # degree と単純ラプラシアン L = D - A
        deg = np.sum(A, axis=1)
        L = np.diag(deg) - A  # shape (m, m)

        # 安定化のため、L の行和が 0 になることを確認（理論上そうなる）
        # Biharmonic の二次形式は L^T L を使う
        M = L.T @ L  # PSD 行列 (m x m)

        # Phi (m, n+3) を使って H = Phi^T M Phi を作る
        Phi = self.Phi  # (m, n)
        H = Phi.T @ (M @ Phi)  # (n, n)

        # 数値対称化
        H = 0.5 * (H + H.T)


        # # 制御点が動いていない初期状態で変形が発生しなくなります。
        # n_rbf = self.src.shape[0]
        # # H は (n_rbf+3, n_rbf+3)。最後の3行・3列（定数, x, y）を0クリア
        # H[n_rbf:, :] = 0.0
        # H[:, n_rbf:] = 0.0

        self.H_reg = H
        print(f"[INFO] precompute: built Phi({self.Phi.shape}), GradPhi({self.GradPhi.shape}), H_reg({self.H_reg.shape})")

    # 恒等写像（アフィン部だけ有効）
    def _set_identity_mapping(self):
        n = self.src.shape[0]
        self.c = np.zeros((2, n + 3), dtype=float)
        self.c[0, n + 1] = 1.0  # u = x
        self.c[1, n + 2] = 1.0  # v = y
        print("[INFO] Identity mapping (affine) set.")

    def update_active_set(self):
        """
        Distortionを評価し、active set Z'を更新する。
        新方針: K_high を超えた点をすべて追加。
        """
        # 全点の歪み評価
        K_values, _ = self.compute_isometric_distortion(self.c)
        new_activated = [i for i, Kval in enumerate(K_values) if Kval > self.K_high]

        # 追加
        for idx in new_activated:
            if idx not in self.activated:
                self.activated.append(idx)

        # K_low を下回る点を削除
        self.activated = [idx for idx in self.activated if K_values[idx] >= self.K_low]

        print(f"[INFO] Active set updated: size={len(self.activated)}")


    # --- helpers: coefficients -> di (副作用なし) ---
    def _compute_di_from_coeffs(self, c_coeffs, indices=None):
        # (既存の実装をそのまま利用)
        if indices is None:
            idx_set = set(self.activated)
            if getattr(self, 'farthest_points', None) is not None:
                idx_set.update(self.farthest_points.tolist())
            indices = sorted(list(idx_set))

        di_tmp = {}
        for idx in indices:
            G = self.GradPhi[idx]  # shape (n+3, 2)
            # grad_u and grad_v computed with given c_coeffs
            grad_u = c_coeffs[0, :] @ G
            grad_v = c_coeffs[1, :] @ G
            fz_vec = 0.5 * np.array([ grad_u[0] + grad_v[1], grad_v[0] - grad_u[1] ])
            norm_fz = np.linalg.norm(fz_vec)
            if norm_fz > 1e-12:
                di_tmp[idx] = fz_vec / norm_fz
            else:
                di_tmp[idx] = self.di[idx].copy() if hasattr(self, 'di') else np.array([1.0, 0.0])
        return di_tmp, indices

    # --- optimize_step の修正版（Biharmonic 正則化を追加） ---
    def optimize_step(self):
        if self.Phi is None or self.GradPhi is None:
            raise RuntimeError("Call precompute() before optimize_step().")

        # --- candidate indices for distortion constraints (activated ∪ farthest) ---
        cand = set(self.activated)
        if self.farthest_points is not None:
            cand.update(self.farthest_points.tolist())
        candidate_indices = sorted(list(cand))

        # --- compute temporary di from current self.c (no side effect) ---
        di_local, candidate_indices = self._compute_di_from_coeffs(self.c, indices=candidate_indices)

        n = self.src.shape[0] + 3  # n+3 total basis count
        c_var = cp.Variable((2, n))

        # 1) positional objective (sum of norms)
        r_vars = []
        constraints = []
        for i in range(len(self.src)):
            phi_i = self.basis_functions(self.src[i])
            target = self.dst[i]
            diff = c_var @ phi_i - target
            r = cp.Variable(nonneg=True)
            constraints += [cp.norm(diff, 2) <= r]
            r_vars.append(r)

        # 2) distortion constraints on candidate_indices, using di_local (not self.di)
        for idx in candidate_indices:
            G = self.GradPhi[idx]  # numpy array (n+3, 2)

            grad_u = c_var[0, :] @ G   # cvxpy expression length-2
            grad_v = c_var[1, :] @ G

            fz  = 0.5 * cp.hstack([ grad_u[0] + grad_v[1],  grad_v[0] - grad_u[1] ])
            fzb = 0.5 * cp.hstack([ grad_u[0] - grad_v[1],  grad_v[0] + grad_u[1] ])

            constraints += [cp.norm(fz, 2) + cp.norm(fzb, 2) <= self.K_high]

            di_vec = di_local[idx]  # numpy array length-2
            constraints += [fz[0]*float(di_vec[0]) + fz[1]*float(di_vec[1]) - cp.norm(fzb, 2) >= 1.0 / self.K]

        # Build objective with optional Biharmonic term
        objective_expr = cp.sum(r_vars)

        if getattr(self, 'H_reg', None) is not None and self.lambda_biharmonic > 0.0:
            # cvxpy quad_form expects a matrix (H is symmetric PSD)
            H = self.H_reg
            # convert to numpy array if it's not
            H_np = np.asarray(H, dtype=float)

            # アフィン成分（最後の3要素: 定数, x, y）に対する正則化を除外する
            n_rbf = self.src.shape[0]

            # RBF係数部分に対応する H の部分行列 (n_rbf x n_rbf) を抽出
            H_rbf = H_np[:n_rbf, :n_rbf]

            # 最適化変数から RBF 係数部分のみスライス
            c_rbf_u = c_var[0, :n_rbf]
            c_rbf_v = c_var[1, :n_rbf]

            # Regularization term for u and v components (only RBF part)
            reg_term = self.lambda_biharmonic * (cp.quad_form(c_rbf_u, H_rbf) +
                                                 cp.quad_form(c_rbf_v, H_rbf))
            objective_expr = objective_expr + reg_term

        objective = cp.Minimize(objective_expr)
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.ECOS, verbose=False)
        except Exception as e:
            raise RuntimeError(f"ECOS solver failed or is not available: {e}")

        if prob.status not in ["infeasible", "unbounded"]:
            self.c = c_var.value
            print(f"[INFO] Optimization done: status={prob.status}, objective={prob.value}")
        else:
            print(f"[WARN] Problem status: {prob.status}")

    # --- postprocess の修正版（indices を受けてその範囲だけ更新） ---
    def postprocess(self):
        """
        Update self.di for all vertices using eq.(27):
        d_i <- J_S f(x_i) / ||J_S f(x_i)|| for every vertex.
        """
        if self.Phi is None or self.GradPhi is None:
            raise RuntimeError("Phi and GradPhi must be precomputed before postprocess().")

        n_points = len(self.vertices)
        for idx in range(n_points):
            G = self.GradPhi[idx]
            # grad_u = [u_x, u_y], grad_v = [v_x, v_y]
            grad_u = self.c[0, :] @ G
            grad_v = self.c[1, :] @ G

            # J_S f = 0.5 * (∇u + I ∇v)
            # I ∇v = [ -v_y, v_x ]
            j_s_vec = 0.5 * np.array([grad_u[0] - grad_v[1],
                                    grad_u[1] + grad_v[0]])

            norm_js = np.linalg.norm(j_s_vec)
            if norm_js > 1e-12:
                self.di[idx] = j_s_vec / norm_js
            else:
                # fallback: keep previous di or identity
                self.di[idx] = np.array([1.0, 0.0])

class BevyBridge:
    def __init__(self, epsilon=50, K=10.0, fps_k=5):
        self.width = 512.0
        self.height = 512.0
        self.rows = 15
        self.cols = 15

        self.vertices = None # 初期頂点 (N, 2)
        self.mapper = None   # ソルバーインスタンス
        self.is_solver_ready = False # ソルバーが準備完了したか

        # 制御点の管理
        self.control_indices = [] # [idx1, idx2, ...]
        self.control_src = []     # [[x,y], ...] (初期位置)
        self.control_dst = []     # [[x,y], ...] (現在のターゲット位置)

        self.target_K = K
        self.target_fps_k = fps_k
        self.target_epsilon = epsilon

    def initialize_mesh_from_data(self, vertices, indices):
        """Rustから頂点データ(flat list)とインデックスを受け取って初期化"""
        flat_verts = np.array(vertices, dtype=np.float32)
        if flat_verts.ndim == 1:
            self.vertices = flat_verts.reshape(-1, 2)
        else:
            self.vertices = flat_verts

        self.indices = np.array(indices, dtype=np.int32)
        print(f"Python: Mesh initialized from data with {len(self.vertices)} vertices.")
        return True

    def initialize_mesh_from_data(self, vertices, indices):
        """Rustから頂点データ(flat list)とインデックスを受け取って初期化"""
        import numpy as np # 関数内importまたはグローバルimport確認
        # verticesはPyListとして受け取るが、numpy配列に変換
        flat_verts = np.array(vertices, dtype=np.float32)

        # フラットなリストを受け取った場合はreshape
        if flat_verts.ndim == 1:
             self.vertices = flat_verts.reshape(-1, 2)
        # List[List] なら (N, 2) になるはず
        else:
             self.vertices = flat_verts

        self.indices = np.array(indices, dtype=np.int32)
        print(f"Python: Mesh initialized from data with {len(self.vertices)} vertices.")
        return True

    def initialize_mesh(self, w, h, rows, cols):
        self.width = w
        self.height = h
        self.rows = rows
        self.cols = cols

        # グリッド生成
        xs = np.linspace(0, w, cols)
        ys = np.linspace(0, h, rows)
        verts = []
        for r in range(rows):
            v_y = (r / (rows - 1)) * h
            for c in range(cols):
                v_x = (c / (cols - 1)) * w
                verts.append([v_x, v_y])
        self.vertices = np.array(verts, dtype=np.float32)
        print(f"Python: Mesh initialized with {len(self.vertices)} vertices.")
        return True

    def _rebuild_mapper(self):
        """制御点を用いてソルバーを構築・初期化する"""
        if not self.control_indices:
            self.mapper = None
            self.is_solver_ready = False
            return

        src_points = np.array(self.control_src, dtype=np.float32)

        # epsilonの自動計算ロジック（例: バウンディングボックスの対角線の15%）
        if self.target_epsilon is None:
            min_xy = np.min(self.vertices, axis=0)
            max_xy = np.max(self.vertices, axis=0)
            diag = np.linalg.norm(max_xy - min_xy)
            # 制御点が極端に近い場合などのガードも入れるとなお良し
            epsilon_val = diag * 0.15
            if epsilon_val < 1e-6: epsilon_val = 1.0 # 安全策
        else:
            epsilon_val = float(self.target_epsilon)

        # マッパー生成 (BetterFitwithGaussianRBF)
        self.mapper = BetterFitwithGaussianRBF(
            vertices=self.vertices.tolist(),
            src=src_points,
            K=self.target_K,            # 変数を使用
            epsilon=epsilon_val,        # 変数を使用
            fps_k=self.target_fps_k     # 変数を使用
        )

        print(f"Python: Building solver with {len(src_points)} control points...")
        self.mapper.initialize_first_step()

        # 現在のターゲット位置をセット
        self.mapper.dst = np.array(self.control_dst, dtype=np.float32)

        self.is_solver_ready = True
        print("Python: Solver is ready.")

    def add_control_point(self, index, x, y):
        """Setupモード: 制御点リストに追加のみ行う（ソルバーは再構築しない）"""
        try:
            if index in self.control_indices:
                return True # 重複は無視

            if index >= len(self.vertices):
                return False

            orig_pos = self.vertices[index]
            self.control_indices.append(index)
            # 変形量ゼロから始めるため、srcもdstも「元の頂点位置」にする
            self.control_src.append(orig_pos.tolist())
            self.control_dst.append(orig_pos.tolist())

            print(f"Python: Added setup point {index} (pinned at orig pos). Total: {len(self.control_indices)}")
            return True

        except Exception as e:
            print(f"Python Error in add_control_point: {e}")
            return False

    def finalize_setup(self):
        """Setup完了: ここで初めて重いソルバー構築を行う"""
        try:
            if len(self.control_indices) == 0:
                print("Python: No control points to finalize.")
                return False

            self._rebuild_mapper()
            return True
        except Exception as e:
            print(f"Python Error in finalize_setup: {e}")
            return False

    def update_control_point(self, list_index, x, y):
        """Deformモード: 制御点の移動"""
        try:
            if not self.is_solver_ready or self.mapper is None:
                return False

            if 0 <= list_index < len(self.control_dst):
                self.control_dst[list_index] = [x, y]
                self.mapper.dst = np.array(self.control_dst, dtype=np.float32)
                return True
            return False
        except Exception as e:
            print(f"Python Error in update: {e}")
            return False

    def reset_mesh(self):
        """リセット"""
        self.control_indices = []
        self.control_src = []
        self.control_dst = []
        self.mapper = None
        self.is_solver_ready = False
        print("Python: Mesh reset.")
        return True

    def solve_frame(self):
        """現在の状態から変形後の頂点を計算して返す"""
        try:
            # ソルバーが準備できていない場合は初期形状を返す（ハルシネーション防止）
            if not self.is_solver_ready or self.mapper is None:
                if self.vertices is None:
                    return []
                return self.vertices.flatten().tolist()

            # 計算実行
            self.mapper.update_active_set()
            self.mapper.optimize_step()

            # 結果取得
            new_verts = self.mapper.Phi @ self.mapper.c.T
            return new_verts.flatten().tolist()

        except Exception as e:
            print(f"Python Solve Error: {e}")
            if self.vertices is None:
                return []
            return self.vertices.flatten().tolist()
