# 実装計画書: `deform_algo.py` を論文に完全準拠させる

**対象論文:** Poranne & Lipman, "Provably Good Planar Mappings", ACM Trans. Graph. 33(4), 2014  
**対象ファイル:** `v2/bevy_image_deform/scripts/deform_algo.py`  
**参照実装:** `v1/refactored2.py`（部分的に論文準拠）  
**作成日:** 2026-02-16

---

## 目次

1. [現状の評価サマリ](#1-現状の評価サマリ)
2. [修正項目一覧](#2-修正項目一覧)
3. [修正項目の詳細](#3-修正項目の詳細)
4. [修正の優先順位と依存関係](#4-修正の優先順位と依存関係)
5. [テスト計画](#5-テスト計画)

---

## 1. 現状の評価サマリ

### ✅ 正しく実装されている部分

| 項目 | 論文参照 | 状態 |
|------|----------|------|
| 基底関数の構成 $\Phi = [\phi_1, \ldots, \phi_n, 1, x, y]$ | Eq. 3 | ✅ 正しい |
| Gaussian RBF $\phi(r) = \exp(-r^2/\epsilon^2)$ と勾配 | Table 1 | ✅ 正しい |
| $J_S f, J_A f$ の定義 | Eq. 19 | ✅ 正しい |
| 特異値の表現 $\Sigma = \|J_S f\| + \|J_A f\|$, $\sigma = \|\|J_S f\| - \|J_A f\|\|$ | Eq. 20 | ✅ 正しい |
| 上界制約 $\|J_S f\| + \|J_A f\| \le K$ | Eq. 21, 23 | ✅ 正しい |
| 下界制約の凸化 $J_S f \cdot d_i - \|J_A f\| \ge 1/K$ | Eq. 25, 26 | ✅ 正しい |
| フレーム更新 $d_i = J_S f / \|J_S f\|$ | Eq. 27 | ✅ 正しい |
| 恒等写像での初期化 (RBF重み=0, アフィン=identity) | Section 5 | ✅ 正しい |
| フレームの初期値 $d_i = (1, 0)$ | Section 5 | ✅ 正しい |
| Strategy 1, 2, 3 の基本構造 | Section 4 | ✅ 正しい |
| バイハーモニック正則化行列 $H = \Phi^T L^T L \Phi$ | Eq. 31 | ✅ 正しい |

### ❌ 問題のある部分（本計画書で修正する項目）

| # | 項目 | 論文参照 | 重要度 |
|---|------|----------|--------|
| P1 | `compute_mapping` のループ順序が論文と逆 | Algorithm 1 | **Critical** |
| P2 | Farthest Point Sampling ($Z''$) が未実装 | Algorithm 1 | **Critical** |
| P3 | Active Set の除去ロジックが未実装 | Algorithm 1 | **High** |
| P4 | Local Maxima フィルタリングが未実装 | Algorithm 1 | **High** |
| P5 | $K_{\text{high}}, K_{\text{low}}$ 閾値が未使用 | Section 5 | **High** |
| P6 | 位置制約が L2² (論文は L1 = ノルムの和) | Eq. 29, 30 | **Medium** |
| P7 | $\omega$ に $2\|\|\|c\|\|\|$ が欠落 | Eq. 8, 9 | **Medium** |
| P8 | 正則化がRBF部分のみに不適切に限定 | Eq. 31 | **Medium** |
| P9 | Conformal distortion 制約が未実装 | Eq. 12, 28 | **Low** |
| P10 | ARAP エネルギーが未実装 | Eq. 32, 33 | **Low** |

---

## 2. 修正項目一覧

```
P1  [Critical]  compute_mapping のループ順序を論文 Algorithm 1 に合わせる
P2  [Critical]  Farthest Point Sampling (Z'') を実装する
P3  [High]      Active Set の除去ロジックを実装する
P4  [High]      Local Maxima フィルタリングを実装する
P5  [High]      K_high, K_low 閾値を導入する
P6  [Medium]    位置制約エネルギーを L1 ノルム (SOCP) に変更する
P7  [Medium]    omega に 2|||c||| を含めるか、近似であることを明記する
P8  [Medium]    正則化の適用範囲を修正する
P9  [Low]       Conformal distortion 制約をオプションとして実装する
P10 [Low]       ARAP エネルギーをオプションとして実装する
```

---

## 3. 修正項目の詳細

---

### P1: `compute_mapping` のループ順序を論文 Algorithm 1 に合わせる

**現状の問題:**

現在のループ:
```
while iteration < max_iterations:
    1. _optimize_step(target_handles)     ← 最適化が先
    2. _update_frames()
    3. _update_active_set()               ← Active Set 更新が後
    if no new violations: break
```

**論文 Algorithm 1 の正しい順序:**
```
Initialization:
    Evaluate D(z) for z ∈ Z
    Find Z_max (local maxima of D(z))
    Insert z ∈ Z_max with D(z) > K_high into Z'
    Remove z ∈ Z' with D(z) < K_low from Z'

Optimization:
    Solve SOCP to find c

Postprocessing:
    Compute f using c and F
    Update d_i using Eq. 27
```

つまり、**Active Set 更新 → 最適化 → フレーム更新** の順序。

**修正内容:**

```python
def compute_mapping(self, target_handles: np.ndarray):
    max_iterations = 10
    iteration = 0

    while iteration < max_iterations:
        # 1. フレーム更新 (現在の c に基づく)
        self._update_frames()

        # 2. 歪み評価 → Active Set 更新 (Algorithm 1: Initialization 部分)
        new_violations = self._update_active_set()

        # 3. 最適化 (Algorithm 1: Optimization 部分)
        self._optimize_step(target_handles)

        # 4. 収束判定: 新規違反がなければ終了
        if new_violations == 0 and iteration > 0:
            break

        iteration += 1

    # 最終的なフレーム更新 (Algorithm 1: Postprocessing)
    self._update_frames()
```

**注意:** 初回イテレーション (`iteration == 0`) では、たとえ `new_violations == 0` でも
少なくとも1回は最適化を実行する必要がある（ハンドル位置が変わっているため）。

---

### P2: Farthest Point Sampling ($Z''$) を実装する

**論文 Algorithm 1:**
> Initialize set $Z''$ with farthest point samples.

**論文 Section 5:**
> To further stabilize the process against fast movement of the handles by the user, we may keep a small subset of equally spread collocation points always active.

$Z''$ は常時アクティブな安定化点であり、最適化ステップでは $Z' \cup Z''$ 上の制約を使用する。

**v1 実装 (`refactored2.py` L98-154):** FPS は実装済み。`farthest_point_sampling(k)` メソッドが存在する。

**修正内容:**

1. `ProvablyGoodPlanarMapping` クラスに `_farthest_point_sampling(k)` メソッドを追加
2. `_initialize_solver()` で `self.permanent_indices` (= $Z''$) を生成
3. `self.activated_indices` の初期値に $Z''$ を含める
4. `_update_active_set()` で $Z''$ の点は除去しないように保護
5. `_optimize_step()` で $Z' \cup Z''$ を制約対象とする

```python
def _farthest_point_sampling(self, k: int) -> List[int]:
    """FPS on collocation grid to select Z'' (permanently active points)."""
    points = self.collocation_grid
    m = points.shape[0]
    if k >= m:
        return list(range(m))
    if k <= 0:
        return []

    indices = np.empty(k, dtype=int)
    indices[0] = int(np.argmin(points[:, 0]))

    chosen = points[indices[0]]
    min_d2 = np.sum((points - chosen) ** 2, axis=1)

    for i in range(1, k):
        next_idx = int(np.argmax(min_d2))
        indices[i] = next_idx
        d2 = np.sum((points - points[next_idx]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)

    return indices.tolist()
```

**パラメータ:** `fps_k` を `SolverConfig` に追加（デフォルト: 20）。
論文 Figure 6 では常時アクティブな点が可視化されており、グリッド全体の 0.5〜2% 程度。

---

### P3: Active Set の除去ロジックを実装する

**論文 Section 5:**
> If any collocation point has distortion lower than $K_{\text{low}}$ where $K_{\text{low}} \in [1, K_{\text{high}}]$, then that point is removed from the active-set.

**現状:** `_update_active_set()` 内のコメント:
```python
# Note: We do not remove points in this strict version (monotonically increasing active set)
```

除去が行われないため、イテレーション毎に制約数が単調増加し、パフォーマンスが悪化する。

**修正内容:**

`_update_active_set()` に以下の除去ロジックを追加:

```python
# --- 除去: K_low 未満の点を Z' から除去 (Z'' は保護) ---
K_vals, _ = self._compute_distortion_on_grid()
permanent_set = set(self.permanent_indices)
remaining = []
for idx in self.activated_indices:
    if idx in permanent_set:
        remaining.append(idx)  # Z'' は除去しない
    elif K_vals[idx] >= self.K_low:
        remaining.append(idx)
    # else: 歪みが十分低いので除去
self.activated_indices = remaining
```

---

### P4: Local Maxima フィルタリングを実装する

**論文 Algorithm 1:**
> Find the set $Z_{\max}$ of local maxima of $D(z)$.
> foreach $z \in Z_{\max}$ such that $D(z) > K_{\text{high}}$ do insert $z$ to $Z'$.

**現状:** 歪み違反のある**全ての点**をアクティブセットに追加している。

**修正内容:**

グリッド上で歪み $D(z)$ の局所最大値を検出するメソッドを追加:

```python
def _find_local_maxima(self, K_vals: np.ndarray) -> np.ndarray:
    """
    グリッド上で K_vals の局所最大値（4近傍）を検出する。
    Returns: 局所最大値のインデックス配列
    """
    nx, ny = self.grid_shape
    K_grid = K_vals.reshape(ny, nx)  # meshgrid は y が先
    local_max_mask = np.zeros_like(K_grid, dtype=bool)

    for iy in range(ny):
        for ix in range(nx):
            val = K_grid[iy, ix]
            is_max = True
            for (dy, dx) in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny2, nx2 = iy + dy, ix + dx
                if 0 <= ny2 < ny and 0 <= nx2 < nx:
                    if K_grid[ny2, nx2] >= val:
                        is_max = False
                        break
            if is_max:
                local_max_mask[iy, ix] = True

    return np.where(local_max_mask.ravel())[0]
```

`_update_active_set()` 内で、`violators` をこの局所最大値でフィルタリングする:

```python
# 歪みの局所最大値を検出
local_maxima = self._find_local_maxima(K_vals)

# K_high を超える局所最大値のみを追加
for idx in local_maxima:
    if K_vals[idx] > self.K_high and idx not in current_set:
        current_set.add(idx)
        new_added_count += 1
```

---

### P5: $K_{\text{high}}, K_{\text{low}}$ 閾値を導入する

**論文 Section 5:**
> In our implementation we used the default values $K_{\text{high}} = 0.1 + 0.9K$ and $K_{\text{low}} = 0.5 + 0.5K$.

**v1 実装 (`refactored2.py` L88-89):**
```python
self.K_high = 0.5 + 0.9 * self.K  # 注: v1 は 0.5 だが論文は 0.1
self.K_low  = 0.5 + 0.5 * self.K
```

**現状 v2:** 閾値は一切使われず、`self.K_upper + 1e-4` でハードコードされた判定。

**修正内容:**

`_initialize_solver()` で閾値を計算:

```python
self.K_high = 0.1 + 0.9 * self.K_upper   # 論文デフォルト
self.K_low  = 0.5 + 0.5 * self.K_upper   # 論文デフォルト
```

`_update_active_set()` で:
- **追加判定:** `K_vals[idx] > self.K_high` (従来の `self.K_upper + 1e-4` を置き換え)
- **除去判定:** `K_vals[idx] < self.K_low` (P3 で追加)

**効果:**
- $K_{\text{high}} < K$ であるため、歪みが $K$ に達する**前に**予防的に制約が追加される
- これにより「急に制約が発動して変形がジャンプする」現象を防ぐ（論文 Section 5 の意図）

---

### P6: 位置制約エネルギーを L1 ノルム (SOCP) に変更する

**論文 Eq. 29-30:**

$$E_{\text{pos}}(f) = \sum_l \| f(p_l) - q_l \|$$

これは各点のユークリッドノルムの**和** (L1 of L2 norms)。SOCP に自然な形式:

$$\min \sum_l r_l \quad \text{s.t.} \quad \| \sum_i c_i f_i(p_l) - q_l \| \le r_l$$

**現状:**
```python
position_loss = cp.sum_squares(diff)  # ← L2² (二次)
```

**v1 実装 (`refactored2.py` L430-436):** 論文どおり L1 で実装済み。

**修正内容:**

```python
# L1 ノルム (Eq. 30): SOCP 形式
r_vars = []
constraints_pos = []
for l in range(n_handles):
    diff_l = c_var @ Phi_src[l] - target_handles[l]
    r_l = cp.Variable(nonneg=True)
    constraints_pos.append(cp.norm(diff_l, 2) <= r_l)
    r_vars.append(r_l)

position_loss = cp.sum(r_vars)
constraints.extend(constraints_pos)
```

---

### P7: $\omega$ に $2\|\|\|c\|\|\|$ を含めるか、近似であることを明記する

**論文 Eq. 9:**

$$\omega = 2\,\|\|\|c\|\|\|\,\omega_{\nabla\mathcal{F}}$$

ここで $\|\|\|c\|\|\| = \max_{\ell \in \{1,2\}} \sum_{i=1}^n |c_i^\ell|$（行列最大ノルム）。

**現状:** `compute_omega(h)` は $\omega_{\nabla\mathcal{F}}(h) = h/s^2$ のみを返す。
Strategy の `resolve_constraints()` でもこの値をそのまま使用。

**問題点:**
- $\|\|\|c\|\|\|$ は最適化の**結果**であり、事前に知ることはできない
- 論文 Section 6 でも、インタラクション中は固定グリッド ($200^2$) を使い、事後的に Strategy 2 で検証している
- したがって、インタラクション中の $\omega$ 計算で $\|\|\|c\|\|\|$ を省略するのは実用上妥当

**修正内容:**

1. `compute_omega(h)` を `compute_omega_grad_basis(h)` にリネーム（これは $\omega_{\nabla\mathcal{F}}$ である）
2. 新たに `compute_omega_full(h, c)` メソッドを追加:

```python
def compute_omega_full(self, h: float, c: np.ndarray) -> float:
    """
    論文 Eq. 9: omega = 2 * |||c||| * omega_grad_F(h)
    c: (2, N_basis) 係数行列
    """
    omega_grad = self.compute_omega_grad_basis(h)
    # |||c||| = max over rows of sum of absolute values (matrix max-norm)
    c_norm = np.max(np.sum(np.abs(c), axis=1))
    return 2.0 * c_norm * omega_grad
```

3. Strategy の `resolve_constraints()` では引き続き `compute_omega_grad_basis()` を使用（近似）
4. 事後検証用に `verify_bounds(c, h)` メソッドを追加し、$\omega_{\text{full}}$ を使って厳密な保証を確認可能にする
5. コード内のドキュメントで、インタラクション中の $\omega$ は $\omega_{\nabla\mathcal{F}}$ の近似であることを明記

---

### P8: 正則化の適用範囲を修正する

**論文 Eq. 31:**

$$E_{\text{bh}}(f) = \iint_\Omega \|H_u(x)\|_F^2 + \|H_v(x)\|_F^2 \, dA$$

これは $f$ 全体（RBF + アフィン項）のヘッセアンに基づく。

**現状:**
```python
N = self.config.source_handles.shape[0]
H_rbf = self.H_reg[:N, :N]  # ← RBF部分のみ切り出し

c_rbf_u = c_var[0, :N]
c_rbf_v = c_var[1, :N]

quad = cp.quad_form(c_rbf_u, H_rbf) + cp.quad_form(c_rbf_v, H_rbf)
```

**理論的背景:**
- アフィン関数 $1, x, y$ のヘッセアンは厳密にゼロ
- したがって、$H_{\text{reg}}$ の最後3行3列はゼロになるはず
- しかし、グラフラプラシアンベースの離散化では、アフィン列も非ゼロになりうる

**修正内容:**

全基底に対して正則化を適用する（理論的に正しいアプローチ）:

```python
# H_reg は (N_basis, N_basis) = (N+3, N+3)
# 全体に適用（アフィン項のヘッセアンは理論上ゼロなので影響は小さい）
quad = cp.quad_form(c_var[0, :], self.H_reg) + cp.quad_form(c_var[1, :], self.H_reg)
reg_term = self.config.lambda_biharmonic * quad
```

あるいは、理論的により正確には、$H_{\text{reg}}$ 計算時にアフィン列を明示的にゼロにする:

```python
# _precompute_basis_on_grid() 内:
# アフィン項はヘッセアンがゼロなので、正則化行列の対応部分をゼロにする
N = self.config.source_handles.shape[0]
self.H_reg[N:, :] = 0.0
self.H_reg[:, N:] = 0.0
```

前者のアプローチを採用する。結果的にアフィン項の寄与は小さいが、
切り出しによる次元不整合リスクを排除できる。

---

### P9: Conformal distortion 制約をオプションとして実装する

**論文 Eq. 12, 28:**

Conformal 制約:
$$\|J_A f(x_i)\| \le \frac{K-1}{K+1} J_S f(x_i) \cdot d_i \tag{28a}$$
$$\|J_A f(x_i)\| \le J_S f(x_i) \cdot d_i - \delta \tag{28b}$$

**修正内容:**

1. `SolverConfig` に `distortion_type: str = 'isometric'` フィールドを追加 (`'isometric'` or `'conformal'`)
2. `SolverConfig` に `delta_conformal: float = 0.1` フィールドを追加 (Eq. 12 の $\delta$)
3. `_optimize_step()` 内で `distortion_type` に応じて制約を切り替え:

```python
if self.config.distortion_type == 'isometric':
    # Eq. 21, 26: 現在の実装
    constraints.append(cp.norm(fz_vec, 2) + cp.norm(fzb_vec, 2) <= self.K_upper)
    constraints.append(dot_prod - cp.norm(fzb_vec, 2) >= self.Sigma_lower)

elif self.config.distortion_type == 'conformal':
    # Eq. 28a, 28b
    K = self.K_upper
    delta = self.config.delta_conformal
    constraints.append(
        cp.norm(fzb_vec, 2) <= ((K - 1) / (K + 1)) * dot_prod
    )
    constraints.append(
        cp.norm(fzb_vec, 2) <= dot_prod - delta
    )
```

4. Strategy の `resolve_constraints()` も conformal 用に拡張 (Eq. 13, 15, 17)

**優先度:** Low — Isometric のみで実用上十分な場合が多い。

---

### P10: ARAP エネルギーをオプションとして実装する

**論文 Eq. 32-33:**

フレーム $d_s$ を使った二次形式:
$$E_{\text{arap}}(f) = \sum_{s=1}^{n_s} \left( \|J_A f(x)\|_F^2 + \|J_S f(x) - d_s\|_F^2 \right)$$

**修正内容:**

1. `SolverConfig` に `lambda_arap: float = 0.0` フィールドを追加
2. `_optimize_step()` 内で ARAP 項を構築:

```python
if self.config.lambda_arap > 0:
    # Eq. 33: ARAP using frames
    # サンプル点は collocation grid の一部（または別途指定）
    arap_indices = self.permanent_indices  # Z'' を流用するか、別途均等サンプル
    arap_term = 0
    for idx in arap_indices:
        G_k = self.GradPhi[idx]
        d_k = self.di[idx]
        
        grad_u_k = c_var[0] @ G_k
        grad_v_k = c_var[1] @ G_k
        
        fz_vec = 0.5 * cp.hstack([grad_u_k[0] + grad_v_k[1], grad_v_k[0] - grad_u_k[1]])
        fzb_vec = 0.5 * cp.hstack([grad_u_k[0] - grad_v_k[1], grad_v_k[0] + grad_u_k[1]])
        
        arap_term += cp.sum_squares(fzb_vec) + cp.sum_squares(fz_vec - d_k)
    
    reg_term += self.config.lambda_arap * arap_term
```

**優先度:** Low — バイハーモニックだけでも十分な平滑化が得られる。

---

## 4. 修正の優先順位と依存関係

```
Phase 1: Algorithm 1 の骨格を修正 (Critical)
  ├─ P5: K_high, K_low を導入 (他の修正の前提)
  ├─ P2: FPS (Z'') を実装
  ├─ P4: Local Maxima フィルタリングを実装
  ├─ P3: Active Set 除去ロジックを実装 (P5 に依存)
  └─ P1: compute_mapping のループ順序を修正 (P2〜P5 に依存)

Phase 2: 最適化問題の精度改善 (Medium)
  ├─ P6: 位置制約を L1 に変更
  ├─ P8: 正則化の適用範囲を修正
  └─ P7: omega に |||c||| を含める（ドキュメント明記 + 検証メソッド追加）

Phase 3: 機能拡張 (Low)
  ├─ P9: Conformal distortion 制約
  └─ P10: ARAP エネルギー
```

### 依存関係図

```
P5 (K_high/K_low)
 ├──> P3 (Active Set 除去)
 └──> P4 (Local Maxima)
       └──> P1 (ループ順序) <── P2 (FPS/Z'')

P6 (L1 位置制約) ── 独立
P7 (omega/|||c|||) ── 独立
P8 (正則化範囲)   ── 独立

P9 (Conformal)  ── 独立 (P1 完了後が望ましい)
P10 (ARAP)      ── 独立 (P1 完了後が望ましい)
```

---

## 5. テスト計画

### 5.1 単体テスト

| テスト | 検証内容 |
|--------|----------|
| `test_identity_mapping` | 恒等写像で歪み $K=1$、全制約が自明に満たされることを確認 |
| `test_fps_coverage` | FPS で選択された点がグリッド全体に均等に分布することを確認 |
| `test_local_maxima` | 既知の歪みパターンで局所最大値検出が正しいことを確認 |
| `test_active_set_removal` | $K_{\text{low}}$ 未満の点が正しく除去されることを確認 |
| `test_active_set_zpp_protection` | $Z''$ の点が除去されないことを確認 |
| `test_omega_full` | $\omega = 2\|\|\|c\|\|\|\,\omega_{\nabla\mathcal{F}}$ の計算が正しいことを確認 |

### 5.2 統合テスト

| テスト | 検証内容 |
|--------|----------|
| `test_bar_bending` | v1 の棒曲げテスト (Figure 5 相当) で $K$ 制約が全点で満たされることを確認 |
| `test_rotation_sequence` | v1 の回転シーケンスを再現し、Active Set の挙動が論文に準拠することを確認 |
| `test_loop_order` | ループ順序変更前後で、同一入力に対して結果が改善されることを確認 |
| `test_l1_vs_l2` | L1 と L2² の位置制約で、ハンドル追従性と歪みのトレードオフを比較 |

### 5.3 回帰テスト

| テスト | 検証内容 |
|--------|----------|
| `test_bevy_bridge_unchanged` | `BevyBridge` の公開APIが変更されていないことを確認 |
| `test_solver_convergence` | 修正後も max_iterations 内に収束することを確認 |

---

## 補足: 修正しない項目とその理由

| 項目 | 理由 |
|------|------|
| B-Spline / TPS 基底関数の実装 | 論文では3種の基底を実験しているが、Gaussian のみで実用上十分。フレームワークは拡張可能な設計になっている |
| Shape-aware Gaussian (内部距離) | 論文 Section 6 で言及されているが、$\omega_{\nabla\mathcal{F}}$ の導出が未完了 (future work) |
| GPU による歪み評価の並列化 | 論文 Section 7 で言及されているが、Python 実装では不要 |
| 非凸ドメインの内部距離 | 論文 Section 4 で言及されているが、現在のユースケースでは不要 |
