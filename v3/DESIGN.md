# v3 設計書 — Provably Good Planar Mappings

> 本設計はPoranne & Lipman (2014) "Provably Good Planar Mappings" に**完全準拠**する。
> 論文に記述されていない機能・ヒューリスティクスは一切含まない。
> 論文参照は `ProvablyGoodPlanarMappings.md` に対する式番号・セクション番号で行う。

---

## 0. 設計原則

1. **pgpm-core**: 論文のアルゴリズムのみを含む純粋なライブラリクレート。Bevy非依存。テスト可能。
2. **bevy-pgpm**: Bevyを使ったインタラクティブアプリケーション。pgpm-coreを呼び出す。
3. **Python排除**: SOCPソルバーにRust製の`clarabel`を使い、IPCを完全に排除する。

---

## 1. pgpm-core クレート設計

### 1.1 依存クレート

| クレート | 用途 |
|----------|------|
| `clarabel` | SOCP求解 (内点法) — 論文はMosekを使用 (Section 6) |
| `nalgebra` | 線形代数 (行列・ベクトル演算) |

他の依存は追加しない。特にBevy、画像処理、UIライブラリへの依存は禁止。

### 1.2 モジュール構成

```
pgpm-core/src/
├── lib.rs              # 公開API
├── types.rs            # 共通型定義
├── basis/
│   ├── mod.rs          # BasisFunction trait
│   ├── gaussian.rs     # Gaussian RBF (Table 1)
│   ├── bspline.rs      # Cubic B-Spline (Table 1)
│   └── tps.rs          # Thin-Plate Spline (Table 1)
├── distortion.rs       # 歪み計算
├── active_set.rs       # Active set 管理
├── solver.rs           # SOCP問題構築・求解
├── strategy.rs         # Strategy 1/2/3
└── algorithm.rs        # Algorithm 1 統合
```

---

### 1.3 types.rs — 共通型定義

```rust
use nalgebra::{DMatrix, DVector, Vector2, Matrix2};

/// 論文 Eq. 3: 係数行列 c ∈ R^{2×n}
/// c = [c_1, c_2, ..., c_n], c_i = (c¹_i, c²_i)^T
pub type CoefficientMatrix = DMatrix<f64>;

/// 論文 Section 3: 歪みの種類
pub enum DistortionType {
    /// D_iso(x) = max{Σ(x), 1/σ(x)}
    Isometric,
    /// D_conf(x) = Σ(x) / σ(x)
    Conformal { delta: f64 },
}

/// 論文 Eq. 5: ドメイン Ω のバウンディングボックス
pub struct DomainBounds {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

/// 論文 Algorithm 1 の入力パラメータ
pub struct AlgorithmParams {
    /// 歪みの種類と上界 K (Eq. 4: D(z_j) ≤ K)
    pub distortion_type: DistortionType,
    pub k_bound: f64,

    /// 正則化の重み λ (Eq. 1, 18)
    pub lambda_reg: f64,

    /// 正則化の種類と混合比
    pub regularization: RegularizationType,
}

/// 論文 Section 5.4 の正則化エネルギー
pub enum RegularizationType {
    /// E_bh のみ (Eq. 31)
    Biharmonic,
    /// E_arap のみ (Eq. 33)
    Arap,
    /// E_pos + λ_bh * E_bh + λ_arap * E_arap
    /// 論文 Section 6: Figure 5 は E_pos + 10^{-2} E_arap, Figure 8 は E_pos + 10^{-1} E_bh
    Mixed { lambda_bh: f64, lambda_arap: f64 },
}

/// Algorithm 1 の内部状態
pub struct AlgorithmState {
    /// 係数行列 c ∈ R^{2×n} (Eq. 3)
    pub coefficients: CoefficientMatrix,

    /// コロケーション点 Z = {z_j} (Eq. 4)
    pub collocation_points: Vec<Vector2<f64>>,

    /// Active set Z' ⊂ Z (Algorithm 1)
    pub active_set: Vec<usize>,

    /// 安定化用の等間隔サンプル Z'' (Algorithm 1: "farthest point samples")
    pub stable_set: Vec<usize>,

    /// フレームベクトル d_i (Eq. 27), 各コロケーション点に対応
    pub frames: Vec<Vector2<f64>>,

    /// K_high, K_low (Section 5 "Activation of constraints")
    pub k_high: f64,
    pub k_low: f64,

    /// 事前計算キャッシュ
    pub precomputed: Option<PrecomputedData>,
}

/// Algorithm 1 "if first step then" で事前計算されるデータ
pub struct PrecomputedData {
    /// f_i(z) for all z ∈ Z, shape: (m, n)
    pub phi: DMatrix<f64>,

    /// ∇f_i(z) for all z ∈ Z, shape: (m, n, 2) → flatten as needed
    pub grad_phi_x: DMatrix<f64>,  // ∂f_i/∂x at each z, shape: (m, n)
    pub grad_phi_y: DMatrix<f64>,  // ∂f_i/∂y at each z, shape: (m, n)

    /// Biharmonic 二次形式行列 (Eq. 31 を数値積分して得る)
    /// ∫∫ ||H_u||²_F + ||H_v||²_F dA → c に関する二次形式
    pub biharmonic_matrix: Option<DMatrix<f64>>,
}
```

---

### 1.4 basis/mod.rs — BasisFunction trait

```rust
/// 論文 Table 1 に対応する基底関数の抽象化
///
/// 論文は B-Spline, TPS, Gaussian の3種を扱う。
/// 各基底関数は以下を提供する必要がある:
/// - 値の評価 f_i(x)
/// - 勾配の評価 ∇f_i(x)
/// - ヘッシアンの評価 H_{f_i}(x) (Eq. 31 の biharmonic エネルギー用)
/// - 勾配のモジュラス ω_{∇F}(t) (Table 1, Eq. 9 で使用)
pub trait BasisFunction: Send + Sync {
    /// 基底関数の数 n を返す
    fn count(&self) -> usize;

    /// f_i(x) を全基底について評価
    /// 戻り値: Vec<f64> of length n
    fn evaluate(&self, x: Vector2<f64>) -> DVector<f64>;

    /// ∇f_i(x) を全基底について評価
    /// 戻り値: (∂f_i/∂x, ∂f_i/∂y) の列、各 DVector<f64> of length n
    fn gradient(&self, x: Vector2<f64>) -> (DVector<f64>, DVector<f64>);

    /// H_{f_i}(x) を全基底について評価 (Eq. 31 用)
    /// 戻り値: (∂²f_i/∂x², ∂²f_i/∂x∂y, ∂²f_i/∂y²) 各 DVector<f64> of length n
    fn hessian(&self, x: Vector2<f64>) -> (DVector<f64>, DVector<f64>, DVector<f64>);

    /// Table 1: 勾配のモジュラス ω_{∇F}(t)
    /// Eq. 9: ω = 2 |||c||| ω_{∇F} で使用される
    fn gradient_modulus(&self, t: f64) -> f64;

    /// 恒等写像 f(x) = x に対応する係数 c ∈ R^{2×n} を返す
    /// (初期状態で J_f = I となる c)
    fn identity_coefficients(&self) -> CoefficientMatrix;
}
```

**論文根拠**: Table 1 に3種の基底関数とそれぞれの $\omega_{\nabla\mathcal{F}}(t)$ が列挙されている。

#### 1.4.1 gaussian.rs

```rust
/// Table 1: Gaussian RBF
/// f_i(x) = exp(-|x - x_i|² / (2s²))
/// ω_{∇F}(t) = t / s²
///
/// 実装では RBF に加えてアフィン項 {1, x, y} を追加する。
/// これは論文の "bases mentioned above (and others)" (Section 3) に含まれる
/// 一般的な慣行であり、恒等写像を表現可能にするために必要。
/// アフィン項の ∇f は定数なので ω_{∇f_i} = 0 であり、
/// Eq. 8 の ω_{∇F} の計算には影響しない。
pub struct GaussianBasis {
    /// RBF中心 {x_i}
    centers: Vec<Vector2<f64>>,
    /// スケールパラメータ s (Table 1)
    s: f64,
}
```

**基底の構成**: $n = |\text{centers}| + 3$ 個の基底関数:
- $f_1, \dots, f_{|\text{centers}|}$: Gaussian RBF
- $f_{n-2}(x) = 1$ (定数)
- $f_{n-1}(x) = x$ (x座標)
- $f_{n}(x) = y$ (y座標)

**恒等写像係数**: $c^1_{n-1} = 1, c^2_{n} = 1$, 他はすべて0。

#### 1.4.2 bspline.rs

```rust
/// Table 1: Cubic B-Spline (テンソル積)
/// f_i(x) = B³_Δ(x - x_i) · B³_Δ(y - y_i)
/// ω_{∇F}(t) = (4 / 3Δ²) · t
pub struct BSplineBasis {
    /// グリッド間隔 Δ
    delta: f64,
    /// ノット位置 {x_i}
    knots: Vec<Vector2<f64>>,
}
```

#### 1.4.3 tps.rs

```rust
/// Table 1: Thin-Plate Spline
/// f_i(x) = (1/2) |x - x_i|² ln(|x - x_i|²)
/// ω_{∇F}(t) = t(5.8 + 5|ln t|)  (|x-y| ≤ (1.25e)^{-1} ≈ 0.29 の範囲で)
///
/// 注: Appendix A に記述されているように、TPSの勾配モジュラスは局所的にのみ有効。
/// 実用上、充填距離 h はこの範囲より小さいため問題にならない (Table 1 注記)。
pub struct TpsBasis {
    centers: Vec<Vector2<f64>>,
}
```

---

### 1.5 distortion.rs — 歪み計算

論文のセクション対応:
- Section 3 "Distortion": $\Sigma(x), \sigma(x)$ の定義
- Eq. 19: $J_S f, J_A f$ の定義
- Eq. 20: 特異値の表現

```rust
/// Eq. 19: J_S f(x) と J_A f(x) を計算
/// J_S f(x) = (∇u(x) + I∇v(x)) / 2
/// J_A f(x) = (∇u(x) - I∇v(x)) / 2
/// ここで I は π/2 反時計回り回転行列 [[0,-1],[1,0]]
pub fn compute_j_s_j_a(
    grad_u: Vector2<f64>,  // ∇u(x) = (∂u/∂x, ∂u/∂y)
    grad_v: Vector2<f64>,  // ∇v(x) = (∂v/∂x, ∂v/∂y)
) -> (Vector2<f64>, Vector2<f64>)

/// Eq. 20: 特異値を J_S, J_A から計算
/// Σ(x) = ||J_S f(x)|| + ||J_A f(x)||
/// σ(x) = | ||J_S f(x)|| - ||J_A f(x)|| |
pub fn singular_values(
    j_s: Vector2<f64>,
    j_a: Vector2<f64>,
) -> (f64, f64)  // (Σ, σ)

/// Section 3 "Distortion": 歪み値を計算
pub fn isometric_distortion(sigma_max: f64, sigma_min: f64) -> f64
    // D_iso = max{Σ, 1/σ}

pub fn conformal_distortion(sigma_max: f64, sigma_min: f64) -> f64
    // D_conf = Σ / σ

/// 全コロケーション点の歪みを一括計算
/// coefficients (Eq. 3), precomputed grad_phi を使って効率的に計算
pub fn evaluate_distortion_all(
    coefficients: &CoefficientMatrix,
    precomputed: &PrecomputedData,
    distortion_type: &DistortionType,
) -> Vec<f64>

/// 全コロケーション点の J_S f を一括計算 (フレーム更新 Eq. 27 に必要)
pub fn evaluate_j_s_all(
    coefficients: &CoefficientMatrix,
    precomputed: &PrecomputedData,
) -> Vec<Vector2<f64>>
```

---

### 1.6 active_set.rs — Active Set 管理

論文のセクション対応:
- Algorithm 1 の Initialization ブロック
- Section 5 "Activation of constraints"

```rust
/// Algorithm 1: Active set の更新
///
/// 論文の記述に忠実に実装する:
/// 1. 全コロケーション点の歪み D(z) を評価
/// 2. Z_max = グリッド上の D(z) の局所最大値の集合を求める
/// 3. z ∈ Z_max かつ D(z) > K_high → Z' に追加
/// 4. z ∈ Z' かつ D(z) < K_low → Z' から削除
///
/// 【重要】論文に書かれていないルール (fold-over予防、σチェック等) は追加しない。
pub fn update_active_set(
    state: &mut AlgorithmState,
    distortions: &[f64],
    grid_width: usize,   // グリッドの列数 (局所最大値の8近傍検索に必要)
    grid_height: usize,  // グリッドの行数
)

/// Algorithm 1 "Initialize set Z'' with farthest point samples"
///
/// 論文 Section 5: "we may keep a small subset of equally spread
/// collocation points always active"
///
/// FPS (Farthest Point Sampling) で等間隔な点を選択する。
/// k の値について論文は具体的な数を規定していないが、
/// Section 6 Figure 6 のキャプションで "some of the points remain
/// activated throughout to stabilize the process" と記述されている。
pub fn initialize_stable_set(
    collocation_points: &[Vector2<f64>],
    k: usize,  // サンプル数
) -> Vec<usize>

/// グリッド上の局所最大値を検出 (8近傍比較)
///
/// 論文 Section 5: "the local maxima of the distortion are found"
/// "the collocation points are sampled on a dense rectangular grid"
fn find_local_maxima(
    distortions: &[f64],
    grid_width: usize,
    grid_height: usize,
) -> Vec<usize>
```

**設計の根拠**: 論文は「局所最大値で $K_{\text{high}}$ を超えるもの**だけ**を追加」と明記。
局所最大値でない点は $K$ を超えていても追加しない。
論文の主張: "only a small number of isolated points will be activated at each iteration"。
これが成り立つのは局所最大値フィルタがあるからである。

---

### 1.7 solver.rs — SOCP 問題構築・求解

論文のセクション対応:
- Eq. 18: 最適化問題の全体定式化
- Eq. 23a-c, 26: Isometric 制約
- Eq. 28a-b: Conformal 制約
- Eq. 30: 位置拘束のSOCP形式
- Eq. 31: Biharmonic エネルギー
- Eq. 33: ARAP エネルギー

```rust
/// SOCP問題を構築して解く
///
/// 論文 Eq. 18 の完全な実装:
/// min_c  E_pos(f) + λ E_reg(f)
/// s.t.   D(f; z) ≤ K,  ∀z ∈ Z' ∪ Z''
///        f = Σ c_i f_i
pub fn solve_socp(
    /// ハンドルのソース位置 {p_l} (Eq. 29)
    source_handles: &[Vector2<f64>],
    /// ハンドルのターゲット位置 {q_l} (Eq. 29)
    target_handles: &[Vector2<f64>],
    /// 基底関数
    basis: &dyn BasisFunction,
    /// アルゴリズム状態 (active set, frames, precomputed data)
    state: &AlgorithmState,
    /// パラメータ (K, λ, 正則化種類)
    params: &AlgorithmParams,
) -> Result<CoefficientMatrix, SolverError>
```

#### SOCP の構造 (clarabel に渡す形式)

**目的関数** (Eq. 18, 30, 31, 33):

$$\min \sum_l r_l + \lambda \left( \lambda_{\text{bh}} \cdot E_{\text{bh}}(c) + \lambda_{\text{arap}} \cdot E_{\text{arap}}(c) \right)$$

- $\sum r_l$: 線形項 (Eq. 30 の補助変数)
- $E_{\text{bh}}(c)$: 二次形式 $c^T H_{\text{bh}} c$ (Eq. 31 を数値積分)
- $E_{\text{arap}}(c)$: 二次形式 $\sum_s (\|J_A f(r_s)\|^2 + \|J_S f(r_s) - d_s\|^2)$ (Eq. 33)

**制約** (各 active collocation point $z_i \in Z' \cup Z''$ について):

Isometric (Eq. 23, 26):
- $\|J_S f(z_i)\| \le t_i$ — SOC制約 (Eq. 23a)
- $\|J_A f(z_i)\| \le s_i$ — SOC制約 (Eq. 23b)
- $t_i + s_i \le K$ — 線形制約 (Eq. 23c)
- $J_S f(z_i) \cdot d_i - s_i \ge 1/K$ — 線形制約 (Eq. 26)

Conformal (Eq. 28):
- $\|J_A f(z_i)\| \le \frac{K-1}{K+1} J_S f(z_i) \cdot d_i$ — SOC制約 (Eq. 28a)
- $\|J_A f(z_i)\| \le J_S f(z_i) \cdot d_i - \delta$ — SOC制約 (Eq. 28b)

位置拘束 (Eq. 30):
- $\|\sum_i c_i f_i(p_l) - q_l\| \le r_l$ — SOC制約

**$J_S f, J_A f$ の線形性** (Section 5 "Collocation points"):

$J_f(x) = \sum_{i=1}^n c_i \nabla f_i(x)$ はcに対して線形。
したがって $J_S f(z)$ と $J_A f(z)$ も c に対して線形であり、
上記の制約はすべてSOCPの標準形に落ちる。

具体的に、grad_phi の事前計算値を使って:

$$\nabla u(z) = \sum_i c^1_i \nabla f_i(z), \quad \nabla v(z) = \sum_i c^2_i \nabla f_i(z)$$

$$J_S f(z) = \frac{1}{2}\begin{pmatrix} \nabla u_x - \nabla v_y \\ \nabla u_y + \nabla v_x \end{pmatrix}, \quad J_A f(z) = \frac{1}{2}\begin{pmatrix} \nabla u_x + \nabla v_y \\ \nabla u_y - \nabla v_x \end{pmatrix}$$

(ここで $I\nabla v = (-\partial_y v, \partial_x v)^T$)

---

### 1.8 strategy.rs — Strategy 1/2/3

論文のセクション対応:
- Eq. 9: $\omega = 2 |||c||| \cdot \omega_{\nabla\mathcal{F}}$
- Eq. 11, 14, 16: Isometric の Strategy 1/2/3
- Eq. 13, 15, 17: Conformal の Strategy 1/2/3

```rust
/// Eq. 8: 係数のマトリクス最大ノルム |||c|||
/// |||c||| = max_{ℓ∈{1,2}} Σ_{i=1}^{n} |c^ℓ_i|
///
/// 注: アフィン項の ω_{∇f_i} = 0 なので、Eq. 8 の和に寄与しない。
/// しかし |||c||| の定義自体はアフィン項を含む全係数の和。
/// ただし恒等写像のアフィン係数は {0,0,...,1,...,0} なので
/// |c^ℓ_i| の和に含まれる。
///
/// 【設計判断】
/// v2 ではアフィン項を除外していたが、厳密には |||c||| は全係数で計算すべき。
/// ただし Eq. 8 の不等式の導出を追うと、各 |c^ℓ_i| に ω_{∇f_i}(t) が
/// 掛かる形なので、ω_{∇f_i}=0 の項は ω の計算に影響しない。
/// 一方 |||c||| は Eq. 8 の最後の不等号で上界として使われるだけなので、
/// アフィン項を含めた方がより保守的（安全側）な上界を与える。
/// 論文の定義に従い、全係数で計算する。
pub fn coefficient_max_norm(coefficients: &CoefficientMatrix) -> f64

/// Eq. 9: ω(t) = 2 |||c||| ω_{∇F}(t)
pub fn omega(
    t: f64,
    coeff_norm: f64,            // |||c|||
    basis: &dyn BasisFunction,  // ω_{∇F}(t) を提供
) -> f64

/// Strategy 1 (Eq. 11): Z と K から K_max を計算
pub fn strategy1_isometric(
    k: f64,
    fill_distance: f64,
    coeff_norm: f64,
    basis: &dyn BasisFunction,
) -> Result<f64, StrategyError>
// K_max = max{ K + ω(h), 1/(1/K - ω(h)) }
// 条件: 1/K > ω(h)

/// Strategy 2 (Eq. 14): K と K_max から必要な h を計算
pub fn strategy2_isometric(
    k: f64,
    k_max: f64,
    coeff_norm: f64,
    basis: &dyn BasisFunction,
) -> Result<f64, StrategyError>
// h ≤ ω^{-1}(min{K_max - K, 1/K - 1/K_max})
// Gaussian の場合 ω(t) = 2|||c||| t/s² は線形なので ω^{-1}(y) = y·s²/(2|||c|||)

/// Strategy 3 (Eq. 16): Z と K_max から K を計算
pub fn strategy3_isometric(
    k_max: f64,
    fill_distance: f64,
    coeff_norm: f64,
    basis: &dyn BasisFunction,
) -> Result<f64, StrategyError>
// K ≤ min{K_max - ω(h), 1/(1/K_max + ω(h))}

/// Conformal 版の Strategy 1/2/3 (Eq. 13, 15, 17) も同様に実装
pub fn strategy1_conformal(/* ... */) -> Result<f64, StrategyError>
pub fn strategy2_conformal(/* ... */) -> Result<f64, StrategyError>
pub fn strategy3_conformal(/* ... */) -> Result<f64, StrategyError>

/// コロケーション点グリッドの生成
/// 論文 Section 4 末尾: "consider all the points from a surrounding
/// uniform grid that fall inside the domain"
pub fn generate_collocation_grid(
    bounds: &DomainBounds,
    h: f64,  // グリッド間隔 = 充填距離
) -> (Vec<Vector2<f64>>, usize, usize)  // (点列, grid_width, grid_height)

/// 充填距離 h(Z,Ω) の計算 (Eq. 5)
/// 矩形グリッドの場合: h = √2/2 · grid_spacing (対角半分)
pub fn fill_distance_of_grid(grid_spacing: f64) -> f64
```

**|||c||| の計算に関する設計判断の詳細**:

論文 Eq. 8 の導出:
$$\|\nabla u(x) - \nabla u(y)\| \le \sum_{i=1}^n |c^1_i| \omega_{\nabla f_i}(\|x-y\|) \le |||c||| \cdot \omega_{\nabla\mathcal{F}}(\|x-y\|)$$

最後の不等号は $\omega_{\nabla\mathcal{F}}(t) \ge \omega_{\nabla f_i}(t)$ for all $i$ を使っている。
アフィン項 $f_i(x) = 1, x, y$ の $\omega_{\nabla f_i} = 0$ なので、
中辺の $|c^1_i| \omega_{\nabla f_i}$ はゼロ。
しかし $|||c|||$ は $\max_\ell \sum_i |c^\ell_i|$ であり、アフィン項を含む。
これは上界を緩くするだけで**安全側**。
論文の定義通り全係数で計算し、v2 のようにアフィン項を除外する「改良」はしない。

---

### 1.9 algorithm.rs — Algorithm 1 統合

論文のセクション対応: Algorithm 1 (Section 5 末尾)

```rust
/// Algorithm 1 の完全な実装
pub struct Algorithm {
    basis: Box<dyn BasisFunction>,
    params: AlgorithmParams,
    state: AlgorithmState,
    source_handles: Vec<Vector2<f64>>,  // {p_l} (不変)
    grid_width: usize,
    grid_height: usize,
    is_first_step: bool,
}

impl Algorithm {
    /// コンストラクタ
    pub fn new(
        basis: Box<dyn BasisFunction>,
        params: AlgorithmParams,
        domain_bounds: &DomainBounds,
        source_handles: Vec<Vector2<f64>>,
        grid_resolution: usize,  // 論文 Section 6: 200
        fps_k: usize,            // Z'' のサンプル数
    ) -> Self

    /// Algorithm 1 の1ステップを実行
    ///
    /// 論文の疑似コードに完全に対応:
    /// 1. [初回のみ] Precompute, d_i=(1,0), Z'=∅, Z''=FPS
    /// 2. D(z) を全 z∈Z で評価
    /// 3. Z_max (局所最大値) を求める
    /// 4. Z_max ∩ {D(z) > K_high} を Z' に追加
    /// 5. Z' ∩ {D(z) < K_low} を Z' から削除
    /// 6. SOCP を解いて c を更新
    /// 7. d_i を更新 (Eq. 27)
    pub fn step(&mut self, target_handles: &[Vector2<f64>]) -> Result<(), AlgorithmError>

    /// 写像の評価 f(x) = Σ c_i f_i(x) (Eq. 3)
    pub fn evaluate(&self, x: Vector2<f64>) -> Vector2<f64>

    /// 現在の係数を取得 (レンダリング側に渡す)
    pub fn coefficients(&self) -> &CoefficientMatrix

    /// Strategy 2: 係数確定後に理論的な充填距離を計算し、
    /// 必要に応じてグリッドを細分化
    ///
    /// 論文 Section 6: "after being satisfied with the results
    /// switched to higher grid resolutions using Strategy 2
    /// to guarantee the bounds on the distortion"
    pub fn verify_and_refine(&mut self, k_max: f64) -> VerificationResult
}

pub enum VerificationResult {
    /// 現在のグリッドで K_max が保証される
    Verified { actual_k_max: f64 },
    /// グリッドを細分化した
    Refined { old_h: f64, new_h: f64, new_grid_size: usize },
    /// 保証不可能 (1/K ≤ ω(h))
    CannotGuarantee { reason: String },
}
```

#### step() の内部フロー (Algorithm 1 に完全対応)

```
step(target_handles):
    // === Initialization (if first step) ===
    if is_first_step:
        precompute phi(z), grad_phi(z) for all z in Z          // Algorithm 1
        set d_i = (1,0) for all i                               // Algorithm 1
        active_set = []                                          // Algorithm 1: "Initialize empty active set Z'"
        stable_set = farthest_point_sampling(Z, fps_k)           // Algorithm 1: "Initialize set Z'' with farthest point samples"
        coefficients = basis.identity_coefficients()
        is_first_step = false

    // === Active Set Update ===
    distortions = evaluate_distortion_all(coefficients, precomputed)  // "Evaluate D(z) for z ∈ Z"
    z_max = find_local_maxima(distortions, grid_width, grid_height)   // "Find the set Z_max of local maxima of D(z)"
    for z in z_max:
        if distortions[z] > k_high:
            active_set.insert(z)                                      // "foreach z ∈ Z_max such that D(z) > K_high do insert z to Z'"
    for z in active_set.clone():
        if distortions[z] < k_low:
            active_set.remove(z)                                      // "foreach z ∈ A such that D(z) < K_low do remove z from Z'"

    // === Optimization ===
    // "Solve the problem in (18) using the SOCP formulation"
    // "Use the constraints ... on the collocation points in Z' ∪ Z''"
    coefficients = solve_socp(source_handles, target_handles, basis, state, params)?

    // === Postprocessing ===
    // "Update d_i using eq. (27)"
    j_s_values = evaluate_j_s_all(coefficients, precomputed)
    for i in active_set ∪ stable_set:
        norm = j_s_values[i].norm()
        if norm > ε:
            frames[i] = j_s_values[i] / norm                        // Eq. 27
```

---

## 2. bevy-pgpm クレート設計

### 2.1 依存

| クレート | 用途 |
|----------|------|
| `bevy` | ゲームエンジン (レンダリング・入力・UI) |
| `pgpm-core` | アルゴリズム |

### 2.2 アーキテクチャ

```
bevy-pgpm/src/
├── main.rs             # Bevy app setup, プラグイン登録
├── state.rs            # App states, resources
├── input.rs            # マウス・キーボード入力 (ハンドル操作)
├── rendering/
│   ├── mod.rs
│   ├── mesh.rs         # メッシュ生成 (Delaunay of collocation points)
│   ├── material.rs     # カスタムマテリアル
│   └── deform.wgsl     # 頂点シェーダ (forward mapping)
├── image.rs            # 画像読み込み・輪郭抽出
└── ui.rs               # テキスト表示・パラメータ調整パネル
```

### 2.3 アプリケーション状態

```rust
// state.rs

/// アプリ全体の状態遷移
#[derive(States, Default, Clone, Eq, PartialEq, Hash, Debug)]
pub enum AppState {
    #[default]
    Setup,       // 画像読み込み・ハンドル配置
    Deforming,   // インタラクティブ変形
    Verifying,   // Strategy 2 による検証 (Phase 3 で有効化)
}

/// pgpm-core の Algorithm をラップするリソース
#[derive(Resource)]
pub struct DeformationState {
    pub algorithm: Algorithm,
    pub source_handles: Vec<Vector2<f64>>,
    pub target_handles: Vec<Vector2<f64>>,
}

/// UI表示用の情報リソース
#[derive(Resource, Default)]
pub struct DeformationInfo {
    pub max_distortion: f64,
    pub active_set_size: usize,
    pub stable_set_size: usize,
    pub step_count: usize,
    pub k_bound: f64,
    pub verification_status: VerificationStatus,
}

#[derive(Default)]
pub enum VerificationStatus {
    #[default]
    NotVerified,
    Verified { k_max: f64 },
    CannotGuarantee,
}
```

### 2.4 入力システム

```rust
// input.rs

/// ハンドルの選択・ドラッグ操作
///
/// 操作フロー:
/// 1. Setup 状態: クリックでハンドル追加
/// 2. Deforming 状態: ハンドルをドラッグ → target_handles 更新
/// 3. ドラッグ中の毎フレーム: algorithm.step() を呼び出し

fn handle_mouse_input(
    mouse: Res<ButtonInput<MouseButton>>,
    window: Query<&Window>,
    camera: Query<(&Camera, &GlobalTransform)>,
    mut state: ResMut<DeformationState>,
    mut info: ResMut<DeformationInfo>,
) {
    // マウス座標をワールド座標に変換
    // ハンドルの選択・ドラッグ・リリースを処理
}
```

### 2.5 pgpm-core との統合

```rust
/// ドラッグ中に毎フレーム呼ばれるシステム
fn update_deformation(
    mut state: ResMut<DeformationState>,
    mut info: ResMut<DeformationInfo>,
) {
    // Algorithm 1 の 1 ステップを実行
    match state.algorithm.step(&state.target_handles) {
        Ok(step_info) => {
            info.max_distortion = step_info.max_distortion;
            info.active_set_size = step_info.active_set_size;
            info.stable_set_size = step_info.stable_set_size;
            info.step_count += 1;
        }
        Err(e) => warn!("SOCP solve failed: {e:?}"),
    }
}

/// ドラッグ終了時に呼ばれるシステム (Phase 3 で Strategy 2 検証を追加)
fn on_drag_end(
    mut state: ResMut<DeformationState>,
    mut info: ResMut<DeformationInfo>,
) {
    // Phase 3: Strategy 2 の検証
    // let result = state.algorithm.verify_and_refine(k_max);
    // 現時点では未検証のまま
    info.verification_status = VerificationStatus::NotVerified;
}
```

### 2.6 GPU レンダリング

v2 の設計を踏襲するが改善:

- **Storage Buffer** (Uniform ではなく) で RBF パラメータを渡す → 基底関数数の制限を撤廃
- 頂点シェーダで forward mapping $f(x)$ を評価、UV は元座標を保持 (v2 と同じ方式)
- メッシュは Delaunay 三角形分割 (collocation points を頂点とする)

### 2.7 UI パネル

テキストオーバーレイで以下を常時表示:

| 表示項目 | 情報源 |
|----------|--------|
| max D(z) | `StepInfo::max_distortion` |
| K bound | `AlgorithmParams::k_bound` |
| Active set size | `StepInfo::active_set_size` |
| Step count | フレームカウンタ |
| Verification | `VerificationStatus` (Phase 3) |

パラメータ調整 (キーボード):

| キー | 操作 |
|------|------|
| `K` / `Shift+K` | K bound 増減 |
| `L` / `Shift+L` | λ (正則化重み) 増減 |
| `R` | リセット (恒等写像に戻る) |
| `Space` | Setup ↔ Deforming 切り替え |
| `V` | Strategy 2 検証実行 (Phase 3) |

---

## 3. 論文 Section 6 実験パラメータの再現

設計の正しさを検証するため、論文の実験設定を再現テストとして実装する。

| 実験 | 基底 | 基底数 | エネルギー | K | グリッド |
|------|------|--------|-----------|---|---------|
| Figure 5 (bar) | B-Spline 6×6 | 36 | $E_{\text{pos}} + 10^{-2} E_{\text{arap}}$ | 2,3,4 | 200² (interactive), 3000² (verification) |
| Figure 8 (square/TPS) | TPS | 25 (5×5 grid) | $E_{\text{pos}} + 10^{-1} E_{\text{bh}}$ | 3,5 | 200² (interactive), 6000² (verification) |
| Figure 9 (bird) | Gaussian | — | ARAP | — | 200² (interactive), 6000² (verification) |
| Figure 10 (disk) | Gaussian | — | — | $D_{\text{iso}}=5$ | 2000² (proof), 4000² (eval) |
| Figure 11 (shape-aware) | Shape-aware Gaussian | 40 | — | $K_{\text{iso}} \le 3$ | — |

---

## 4. 実装順序

### Phase 1: pgpm-core 最小構成 ✅
1. `types.rs`: 型定義
2. `basis/gaussian.rs`: Gaussian RBF (最も単純、v1/v2 で実績あり)
3. `distortion.rs`: 歪み計算 (Eq. 19-20)
4. `active_set.rs`: Active set 管理
5. `solver.rs`: clarabel を使った SOCP 構築・求解
6. `algorithm.rs`: Algorithm 1 統合
7. **テスト**: 恒等写像 → 単純な変形で distortion bound 検証

### Phase 2: bevy-pgpm (Gaussian + Isometric で動くUI)

Phase 1 の pgpm-core 最小構成（Gaussian 基底 + Isometric 歪み + Algorithm 1）で
インタラクティブなアプリケーションを先に構築する。
Phase 3 の未実装機能はスタブ/無効化で対応する。

8. Bevy app 基盤: `main.rs`, `state.rs` (AppState, DeformationState)
9. 入力システム: `input.rs` (ハンドル配置・ドラッグ)
10. GPU レンダリング: `rendering/` (mesh, material, deform.wgsl)
11. 画像読み込み: `image.rs` (テクスチャロード・輪郭からドメイン定義)
12. UI: `ui.rs` (情報表示パネル・パラメータ調整)
13. **テスト**: ハンドル操作 → リアルタイム変形の動作確認

**Phase 2 時点でのスタブ/制限事項:**

| Phase 3 機能 | Phase 2 での対応 |
|---|---|
| Strategy 1/2/3 (`strategy.rs`) | `verify_and_refine()` は `CannotGuarantee` を返す。UIでは「未検証」表示 |
| B-Spline (`bspline.rs`) | 未実装。基底選択UIでは Gaussian のみ有効 |
| TPS (`tps.rs`) | 未実装。基底選択UIでは Gaussian のみ有効 |
| Conformal 歪み | Isometric モードのみ使用可能。Conformal はグレーアウト |

### Phase 3: pgpm-core 完成 + bevy-pgpm 拡張
14. `strategy.rs`: Strategy 1/2/3
15. `basis/bspline.rs`: B-Spline
16. `basis/tps.rs`: TPS
17. Conformal 歪みのフル対応 (δ を Strategy に基づいて計算)
18. bevy-pgpm: Strategy 2 検証UI、基底切替、Conformal モード
19. **テスト**: 論文 Section 6 の実験再現

---

## 5. v1/v2 からの非継承事項

以下は v1/v2 に存在したが、**論文に根拠がないため v3 では実装しない**:

| 項目 | v1/v2 での実装 | 不採用理由 |
|------|----------------|-----------|
| fold-over予防的追加 | v2: `is_foldover`, `is_near_foldover` による一括追加 | 論文に記述なし。active set + 歪み制約で十分 |
| active set サイズ制限 | v2: `max_active = 5000` で打ち切り | 論文に記述なし。局所最大値フィルタで自然に少数に収まる |
| グラフラプラシアン正則化 | v1: Delaunay → L = D-A → H_reg = Φ^T L^T L Φ | 論文はヘッシアンベースのbiharmonic (Eq. 31)。メッシュ不要 |
| stable set の5点固定 | v2: グリッドの4隅+中心のみ | 論文は "farthest point samples" (FPS) で等間隔 |
| ARAP独自フレーム | v2: `_arap_frames` を歪みフレームとは別に管理 | 論文 Eq. 33: "d_s is the frame at r_s"。同一のフレーム配列を共有 |
| 複素数表現での歪み計算 | v1: $f_z, f_{\bar{z}}$ ベース | 論文は $J_S f, J_A f$ (Eq. 19-20) を使用。等価だが論文に合わせる |
| solve_frame での複数回呼び出し | v2: `for i in range(3): update_mapping()` | 論文 Algorithm 1 は1ステップ1回のSOCP。複数回呼ぶ記述なし |
| conformal δ の固定値 | v2: `self._delta_conf = 0.1` | 論文 Eq. 13: δ > ω(h) が必要。Strategy に応じて決定すべき |
