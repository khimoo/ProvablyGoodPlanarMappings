#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
import warnings
import multiprocessing as mp
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
    def __init__(self, vertices, src, K, epsilon=80.0, fps_k=20, lambda_biharmonic=1e-3):
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
            # Regularization term for u and v components
            reg_term = self.lambda_biharmonic * (cp.quad_form(c_var[0, :], H_np) +
                                                 cp.quad_form(c_var[1, :], H_np))
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




# --- 回転用ユーティリティ関数 ---
def rotate_control_point(src_ctrl, ctrl_index, angle_deg, center=None):
    if center is None:
        center = src_ctrl.mean(axis=0)
    theta = np.deg2rad(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    vec = src_ctrl[ctrl_index] - center
    new_pos = (R @ vec) + center
    dst = src_ctrl.copy()
    dst[ctrl_index] = new_pos
    return dst

def apply_mapper_step(mapper, dst_ctrl):
    mapper.dst = dst_ctrl
    mapper.update_active_set()
    mapper.optimize_step()
    mapper.postprocess()
    return mapper.Phi @ mapper.c.T  # (m,2)


# --- 初回にメッシュの接続関係を構築 ---
def build_mesh_connectivity(points):
    tri = Delaunay(points)
    return tri.simplices  # 各要素は3頂点のインデックス

def load_mesh_connectivity(csv_path):
    pass

# --- 子プロセス用プロット関数（メッシュ対応版） ---
# 変更: activated_indices 引数を追加（各ステップの active set を渡す）
def plot_step_window(sample_points, src_ctrl, dstctrl, transformed_points,
                     title, mapper_farthest, mesh_simplices, activated_indices,
                     show_index_labels=True):
    ox, oy = sample_points[:, 0], sample_points[:, 1]
    tx, ty = transformed_points[:, 0], transformed_points[:, 1]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1)

    # --- メッシュをワイヤーフレームで表示 ---
    for tri in mesh_simplices:
        # original
        ax.plot(ox[tri], oy[tri], linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
        # transformed
        ax.plot(tx[tri], ty[tri], linestyle='-', linewidth=1.0, alpha=0.8, zorder=1)

    # --- 点群や制御点を表示 ---
    ax.scatter(ox, oy, s=18, alpha=0.5, label="original points", zorder=1)
    ax.scatter(tx, ty, marker='x', s=28, linewidths=1.5, label="transformed points", zorder=2)
    ax.scatter(src_ctrl[:, 0], src_ctrl[:, 1], marker='s', s=70, label="control points (src)", zorder=4)
    ax.scatter(dstctrl[:, 0], dstctrl[:, 1], marker='o', s=70, label="control points (dst)", zorder=4)

    # FPS点のハイライト（元座標で描画）
    if mapper_farthest is not None and len(mapper_farthest) > 0:
        fp = np.asarray(mapper_farthest)
        ax.scatter(sample_points[fp, 0], sample_points[fp, 1],
                   marker='*', s=140, label="Z'' (FPS)", zorder=5)

    # ----- activated points の強調表示 -----
    if activated_indices is not None and len(activated_indices) > 0:
        ai = np.asarray(activated_indices, dtype=int)
        # 大きめの輪郭付きマーカーで目立たせる
        ax.scatter(sample_points[ai, 0], sample_points[ai, 1],
                   facecolors='none', edgecolors='red', s=220, linewidths=1.8,
                   label="activated (Z')", zorder=6)
        # 中心点も小さな塗りつぶしで
        ax.scatter(sample_points[ai, 0], sample_points[ai, 1],
                   c='red', s=20, alpha=0.9, zorder=7)

        # ラベル（インデックス）を付ける（必要なければ show_index_labels=False で消せます）
        if show_index_labels:
            for j, idx in enumerate(ai):
                x, y = sample_points[idx]
                ax.text(x + 0.04, y + 0.04, str(idx), fontsize=8, color='darkred', zorder=8)

        # オプション: activated を含む三角形を太めの線でハイライト（見やすくする）
        # ここでは mesh_simplices のうち、いずれかの頂点が activated に含まれる三角形を選ぶ
        activated_set = set(ai.tolist())
        for tri in mesh_simplices:
            if activated_set.intersection(tri):
                ax.plot(ox[tri].tolist() + [ox[tri][0]], oy[tri].tolist() + [oy[tri][0]],
                        linewidth=2.0, linestyle='-', color='orange', alpha=0.6, zorder=2)

    # 制御点の変位を矢印で表示
    ox_ctrl, oy_ctrl = src_ctrl[:, 0], src_ctrl[:, 1]
    tx_ctrl, ty_ctrl = dstctrl[:, 0], dstctrl[:, 1]
    dx_ctrl, dy_ctrl = tx_ctrl - ox_ctrl, ty_ctrl - oy_ctrl
    ax.quiver(ox_ctrl, oy_ctrl, dx_ctrl, dy_ctrl,
              angles='xy', scale_units='xy', scale=1, width=0.001, alpha=0.95,
              label="control displacement", zorder=5)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', frameon=True)

    plt.tight_layout()
    plt.show()


import os
def load_sample_points(csv_path):
    # x,yヘッダありのCSV
    return np.genfromtxt(csv_path, delimiter=",", skip_header=1, dtype=float)
# ---------------- main ----------------
if __name__ == "__main__":
    # Windows 安全用（既に設定済なら例外キャッチして無視）
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # ----- データ準備 -----
    xs = np.linspace(1.0, 20.0, 10)
    ys = np.linspace(8.5, 10.5, 3)
    XX, YY = np.meshgrid(xs, ys)
    sample_points = np.vstack([XX.ravel(), YY.ravel()]).T.astype(float)


    src_ctrl = np.array([[1.0, 10.0], [13.0, 9.0], [20.0, 9.0], [20.0, 10.0]], dtype=float)

    K_threshold = 2
    first_step_flag = True
    fps_k = 100

    # --- ここで BetterFitwithGaussianRBF を初期化 ---
    # (あなたの環境でこのクラスが定義されている前提)
    mapper = BetterFitwithGaussianRBF(
        vertices=sample_points,
        src=src_ctrl,
        K=K_threshold,
        epsilon=70.0,
        fps_k=fps_k
    )
    mapper.initialize_first_step()

    print(f"[INFO] Phi shape: {mapper.Phi.shape}")
    print(f"[INFO] GradPhi shape: {mapper.GradPhi.shape}")
    print(f"[INFO] Active set Z': {mapper.activated}")
    print(f"[INFO] Farthest point set Z'': {mapper.farthest_points}")
    print(f"[INFO] Initial c shape: {mapper.c.shape}\n{mapper.c}")

    # mapper 初期化のあと、初期の activated を保存
    transformed_points_initial = mapper.Phi @ mapper.c.T

    step_angles = [-45, -90, -125, -150, -200]  # Step2..Step6 の角度
    #step_angles = [-150]
    transformed_list = [transformed_points_initial]
    dst_list = [src_ctrl.copy()]

    # 追加: 各ステップごとの activated set を保存するリスト
    activated_list = [mapper.activated.copy() if hasattr(mapper, 'activated') else []]

    for i, ang in enumerate(step_angles, start=2):
        dst_i = rotate_control_point(src_ctrl, ctrl_index=0, angle_deg=ang, center=[10.0, 9.0])
        transformed_i = apply_mapper_step(mapper, dst_i)
        transformed_list.append(transformed_i)
        dst_list.append(dst_i)

        # --- ここでそのステップの activated set をコピーして保存 ---
        act = mapper.activated.copy() if hasattr(mapper, 'activated') else []
        activated_list.append(act)

        print(f"[INFO] Completed Step {i}: rotate {ang} deg, activated count = {len(act)}")

    # メッシュ接続情報を構築
    mesh_simplices = build_mesh_connectivity(sample_points)

    # ===== 並列プロセスで各ステップのウィンドウだけを開く（activated_list を各プロセスに渡す）=====
    procs = []
    for idx, (transformed_points, dstctrl, activated_indices) in enumerate(zip(transformed_list, dst_list, activated_list), start=1):
        if idx == 1:
            title = "Step 1: Identity"
        else:
            angle = step_angles[idx-2] if idx-2 < len(step_angles) else None
            title = f"Step {idx}: rotate {angle}°" if angle is not None else f"Step {idx}"

        p = mp.Process(
            target=plot_step_window,
            args=(sample_points, src_ctrl, dstctrl, transformed_points, title,
                  mapper.farthest_points, mesh_simplices, activated_indices),
            daemon=False
        )
        p.start()
        procs.append(p)

    # 各ウィンドウ（子プロセス）が閉じられるのを待つ
    for p in procs:
        p.join()

    # 最後のステップの最大変位（デバッグ表示）
    last = transformed_list[-1]
    print("for debug: max displacement step 5 =", float(np.max(np.hypot(
        last[:,0]-sample_points[:,0],
        last[:,1]-sample_points[:,1]
    ))))
