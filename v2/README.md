# プログラム概要

[Poranne & Lipman (2014) Provably Good Planar Mappings](https://doi.org/10.1145/2601097.2601123) をRustで実装しました。

![デモ](screenshots/demo.gif)
<!-- TODO: screenshots/ にデモ GIF を配置 -->

Provably Good Planar Mapingsは画像変形のアルゴリズムで、特に非凸な空間の変形に焦点を当てています。非凸領域上に離散的に存在する計算点で最適化を実行するだけで、非凸領域全体で画像の裏返りが発生しないことを数学的に保証しています。

このようなアルゴリズムを使用し、背景透過画像(PNG)を変形させて新たに保存するツールとしてまとめました。
# 実行方法

[Nix](https://nixos.org/download/) をインストールし、Flakesを有効化してください。

Flakes の有効化:
```bash
# ~/.config/nix/nix.conf に追記
mkdir -p ~/.config/nix
echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf
```

nix runでプログラムが起動します。
```bash
nix run
```

# 操作方法
## 論文概要

### 解く問題

2D ドメイン Ω 上の写像 f: Ω → R² を、ユーザが指定したハンドル制約 (p_l → q_l) のもとで構築します。このとき、写像の歪み（形状のつぶれや引き伸ばしの度合い）に上界 K を課すことで、写像の品質を保証します。

「画像の一部をドラッグして変形する」操作を、局所単射性（写像が局所的に裏返らないこと）と歪み上界を数学的に保証しつつ実現します。

### アルゴリズムの構造

写像は放射基底関数 (RBF) の線形結合で表現されます（論文 Eq. 3）。

```
f(x) = Σ c_i φ_i(x)
```

この係数 c を、歪み制約付き二次錐計画 (SOCP) として最適化します（論文 Eq. 18）。歪みの計算にはヤコビアンの共形・反共形分解（Eq. 19-20）を用い、制約は「歪みが大きい点」だけに選択的に課すアクティブセット戦略で効率化しています（Algorithm 1, Section 5）。

主要な概念の対応関係:

| 論文の概念 | 本実装での対応 |
|-----------|--------------|
| 写像の表現 (Eq. 3) | `PgpmAlgorithm::evaluate_mapping_at()` |
| 歪み尺度 (Section 3) | `DistortionPolicy` trait |
| SOCP 最適化 (Eq. 18) | `solver::solve_socp()` |
| アクティブセット (Algorithm 1) | `active_set::update_active_set()` |
| 基底関数 (Table 1) | `BasisFunction` trait |
| Strategy 2 精緻化 | `PgpmAlgorithm::refine_strategy2()` |

## アーキテクチャ

```
pgpm-core (論文アルゴリズム)          bevy-pgpm (インタラクティブUI)
┌─────────────────────────┐         ┌──────────────────────┐
│ BasisFunction           │         │ Bevy App             │
│ DistortionPolicy        │◄────────│   input / rendering  │
│ Domain                  │ Bridge  │   state / ui         │
│ PgpmAlgorithm ──────┐   │         │                      │
│ MappingBridge ◄─────┘   │         │                      │
└─────────────────────────┘         └──────────────────────┘
     ▲ 依存方向: 内側にのみ向かう
```

- **pgpm-core**: 論文アルゴリズムの純粋な実装。Bevy・UI・画像処理への依存は一切ありません。
- **bevy-pgpm**: Bevy によるインタラクティブ UI。`pgpm-core` の公開 API (`MappingBridge` trait) にのみ依存します。

`bevy-pgpm` から `pgpm-core` への依存はファクトリ関数 + `MappingBridge` trait のみで、具象型 `Algorithm<D>` は公開していません。Clean Architecture の「依存は内側にのみ向ける」原則に従っています。

## trait 設計と論文の対応

本実装の核心は、論文の数学的構造を Rust の trait で直接表現している点です。

### BasisFunction — 基底関数の抽象化 (Table 1)

論文 Table 1 は Gaussian, B-Spline, TPS の 3 種の基底関数を定義し、それぞれに値・勾配・勾配モジュラスの式を与えています。これを trait として抽象化しました。

```rust
// crates/pgpm-core/src/basis/mod.rs

pub trait BasisFunction: Send + Sync {
    fn count(&self) -> usize;
    fn evaluate(&self, x: Vector2<f64>) -> DVector<f64>;
    fn gradient(&self, x: Vector2<f64>) -> (DVector<f64>, DVector<f64>);
    fn hessian(&self, x: Vector2<f64>) -> (DVector<f64>, DVector<f64>, DVector<f64>);
    fn gradient_modulus(&self, t: f64) -> f64;          // ω_{∇F}(t)
    fn gradient_modulus_inverse(&self, v: f64) -> f64;   // ω_{∇F}⁻¹(v)
    fn identity_coefficients(&self) -> CoefficientMatrix;
}
```

各メソッドが論文の要素に 1 対 1 で対応します。

- `evaluate` / `gradient` / `hessian`: φ_i(x), ∇φ_i(x), H_{φ_i}(x)
- `gradient_modulus`: Strategy 2 の充填距離計算に使う連続の度合い ω_{∇F}(t) (Eq. 9)
- `identity_coefficients`: f(x) = x となる初期係数

実装: `GaussianBasis` (ユークリッド距離), `ShapeAwareGaussianBasis` (測地距離)

### DistortionPolicy — 歪み計算の抽象化 (Section 3)

論文は等長 (isometric) と等角 (conformal) の 2 種の歪み尺度を定義し、それぞれ異なる SOCP 制約式を導出しています。この「歪みの種類によって変わる部分」を trait として切り出しました。

```rust
// crates/pgpm-core/src/policy/mod.rs

pub trait DistortionPolicy: Send + Sync {
    fn distortion_value(&self, sigma_max: f64, sigma_min: f64) -> f64;
    fn extra_vars_per_active(&self) -> usize;
    fn append_constraints(&self, ...);
    fn required_h(&self, k: f64, k_max: f64, c_norm: f64, basis: &dyn BasisFunction)
        -> Option<f64>;
    fn compute_k_max(&self, k: f64, omega_h: f64) -> Option<f64>;
}
```

- `distortion_value`: D_iso = max{Σ, 1/σ} (等長) or D_conf = Σ/σ (等角) の切り替え
- `extra_vars_per_active`: 等長は t_i, s_i の 2 変数 (Eq. 23)、等角は 0 (Eq. 28)
- `append_constraints`: SOCP 制約行の構築を policy に委譲
- `required_h` / `compute_k_max`: Strategy 方程式が歪み尺度によって異なる (Eq. 11/13, 14/15)

実装: `IsometricPolicy`, `ConformalPolicy`

### Domain — ドメインの抽象化 (Section 4)

アルゴリズムが必要とするのは「点がドメイン Ω の内部にあるか」の判定のみです。ドメインの表現方法（ポリゴン、アルファチャンネル、SDF 等）はアルゴリズムの関心外なので、最小のインターフェースで切り離しました。

```rust
// crates/pgpm-core/src/model/domain.rs

pub trait Domain: Send + Sync {
    fn contains(&self, pt: &Vector2<f64>) -> bool;
}
```

実装: `PolygonDomain` (レイキャスティング法)

### PgpmAlgorithm — Algorithm 1 の骨格 (Section 5)

ここが設計上最も重要な trait です。論文の Algorithm 1 全体（初期化、SOCP 最適化、アクティブセット管理、フレーム更新、Strategy 2 精緻化）をデフォルトメソッドとして実装しています。

```rust
// crates/pgpm-core/src/mapping/pgpm_algorithm.rs

pub trait PgpmAlgorithm: Send + Sync {
    // === 必須メソッド（3つのみ）===
    fn parts(&self) -> (MappingContext<'_>, &AlgorithmState);
    fn parts_mut(&mut self) -> (MappingContext<'_>, &mut AlgorithmState);
    fn set_params(&mut self, params: MappingParams);

    // === デフォルトメソッド: Algorithm 1 の完全な実装 ===
    fn step(&mut self, target_handles: &[Vector2<f64>])
        -> Result<StepInfo, AlgorithmError> { ... }
    fn refine_strategy2(&mut self, ...)
        -> Result<Strategy2Result, AlgorithmError> { ... }
    fn evaluate_mapping_at(&self, points: &[Vector2<f64>])
        -> Vec<Vector2<f64>> { ... }
    // ... 他のデフォルトメソッド
}
```

具象型はデータの格納方法と借用分離アクセサだけを提供すれば、Algorithm 1 のロジック全体を得られます。

`step()` の処理の流れ（Algorithm 1, Section 5 に対応）:

```
step(target_handles):
    1. [初回のみ] φ_i(z), ∇φ_i(z) を事前計算
    2. 全 z ∈ Z で歪み D(z) を評価       ... Eq. 19-20
    3. D の局所最大を見つける
    4. D > K_high の局所最大を Z' に追加   ... Section 5
    5. D < K_low の点を Z' から削除
    6. SOCP を求解 → 係数 c を更新        ... Eq. 18
    7. フレーム d_i を更新                 ... Eq. 27
```

### MappingBridge — ファサード

`PgpmAlgorithm` はアルゴリズム内部で使う詳細なメソッド（`coefficients`, `j_s_j_a_at` 等）も含みます。UI 側にはこれらを見せたくないので、公開用のサブセットとして `MappingBridge` を定義し、blanket impl で自動的に橋渡ししています。

```rust
// crates/pgpm-core/src/mapping/bridge.rs

pub trait MappingBridge: Send + Sync {
    fn step(&mut self, target_handles: &[Vector2<f64>])
        -> Result<StepInfo, AlgorithmError>;
    fn evaluate_mapping_at(&self, points: &[Vector2<f64>])
        -> Vec<Vector2<f64>>;
    fn update_params(&mut self, params: MappingParams);
    fn refine_strategy2(&mut self, ...)
        -> Result<Strategy2Result, AlgorithmError>;
    // + クエリメソッド群
}

// PgpmAlgorithm を実装する全ての型に MappingBridge を自動実装
impl<T: PgpmAlgorithm + ?Sized> MappingBridge for T {
    fn step(&mut self, target_handles: &[Vector2<f64>])
        -> Result<StepInfo, AlgorithmError>
    {
        PgpmAlgorithm::step(self, target_handles)
    }
    // ... 全メソッドを PgpmAlgorithm に委譲
}
```

ファクトリ関数は `Box<dyn MappingBridge>` を返すため、UI 側は具象型 `Algorithm<IsometricPolicy>` / `Algorithm<ConformalPolicy>` を知る必要がありません。

```rust
// crates/pgpm-core/src/lib.rs

pub fn create_isometric_mapping(...) -> Box<dyn MappingBridge> {
    Box::new(Algorithm::new(basis, params, IsometricPolicy, ...))
}
```

## 実装上の工夫

### 借用分離パターン (`parts()` / `parts_mut()`)

`PgpmAlgorithm` のデフォルトメソッド（特に `step()`）は、不変の設定情報（基底関数、パラメータ等）を参照しながら可変の状態（係数、アクティブセット等）を更新する必要があります。Rust の借用チェッカーは `&self` と `&mut self` の同時使用を禁止するため、`self` を不変コンテキスト `MappingContext` と可変状態 `AlgorithmState` に分離して返すアクセサを設けました。

```rust
// crates/pgpm-core/src/model/types.rs

pub struct MappingContext<'a> {
    pub basis: &'a dyn BasisFunction,
    pub policy: &'a dyn DistortionPolicy,
    pub params: &'a MappingParams,
    pub source_handles: &'a [Vector2<f64>],
    pub domain_bounds: &'a DomainBounds,
    pub domain: Option<&'a dyn Domain>,
    pub solver_config: &'a SolverConfig,
}
```

```rust
// crates/pgpm-core/src/mapping/pgpm_algorithm.rs — step() 内での使用例

// 不変コンテキストと可変状態に分離して同時にアクセス
let (ctx, state) = self.parts_mut();
active_set::update_active_set(state, &distortions);  // state のみ変更
let k = ctx.params.k_bound;                           // ctx は読み取りのみ
```

これにより、具象型 `Algorithm<D>` の `PgpmAlgorithm` 実装はアクセサ 3 つのみとなり、500 行超のアルゴリズムロジック全体が trait のデフォルトメソッドに配置されています。

```rust
// crates/pgpm-core/src/algorithm/runner.rs

impl<D: DistortionPolicy> PgpmAlgorithm for Algorithm<D> {
    fn parts(&self) -> (MappingContext<'_>, &AlgorithmState) {
        (MappingContext { basis: self.basis.as_ref(), ... }, &self.state)
    }
    fn parts_mut(&mut self) -> (MappingContext<'_>, &mut AlgorithmState) {
        (MappingContext { basis: self.basis.as_ref(), ... }, &mut self.state)
    }
    fn set_params(&mut self, params: MappingParams) {
        self.params = params;
    }
    // step(), refine_strategy2(), evaluate_mapping_at() 等は全てデフォルト実装を使用
}
```

### blanket impl による型消去

`impl<T: PgpmAlgorithm + ?Sized> MappingBridge for T` により、`PgpmAlgorithm` を実装する任意の型が自動的に `MappingBridge` も実装します。ファクトリ関数が `Box<dyn MappingBridge>` を返すことで、呼び出し側（bevy-pgpm）は歪みポリシーの型パラメータ `D` を意識せずに済みます。

この構造は Clean Architecture における依存関係逆転の原則と対応しています。内側（pgpm-core）が trait を定義し、外側（bevy-pgpm）はその trait にのみ依存します。

### 論文忠実性の方針

独自のヒューリスティクスや「改良」を一切追加しない方針で実装しています。

- fold-over 予防のための独自チェックは入れない（論文はアクティブセット + 歪み制約で十分としている）
- near-fold-over 点の予防的追加はしない（論文は局所最大フィルタのみ）
- アクティブセットサイズの人為的制限はしない
- メッシュ依存の処理はしない（本手法はメッシュレス）

実装の根拠は全てコード中のコメントで論文の式番号・セクション番号を付記しています。

## bevy-pgpm 概要

Bevy ベースのインタラクティブ UI です。画像を読み込み、ハンドルを配置してドラッグすることで変形を可視化します。

- 画像の輪郭を自動抽出して `PolygonDomain` を構築
- ハンドルのドラッグに応じて `MappingBridge::step()` を呼び出し
- CPU レンダリングパス: `evaluate_mapping_at()` でメッシュ頂点を変形
- UI パネルで K, λ, 正則化タイプを調整可能

pgpm-core の `MappingBridge` trait のみに依存しており、アルゴリズムの内部実装には触れていません。

## 注目ファイル一覧

pgpm-core の設計を理解するために、以下のファイルを順に読むことを推奨します。

| 順序 | ファイル | 内容 |
|------|---------|------|
| 1 | `crates/pgpm-core/src/basis/mod.rs` | `BasisFunction` trait 定義 |
| 2 | `crates/pgpm-core/src/policy/mod.rs` | `DistortionPolicy` trait 定義と 2 つの実装 |
| 3 | `crates/pgpm-core/src/model/domain.rs` | `Domain` trait 定義と `PolygonDomain` |
| 4 | `crates/pgpm-core/src/model/types.rs` | `MappingContext`, `AlgorithmState` 等の型定義 |
| 5 | **`crates/pgpm-core/src/mapping/pgpm_algorithm.rs`** | **`PgpmAlgorithm` trait — Algorithm 1 全体のデフォルト実装** |
| 6 | `crates/pgpm-core/src/mapping/bridge.rs` | `MappingBridge` trait と blanket impl |
| 7 | `crates/pgpm-core/src/algorithm/runner.rs` | `Algorithm<D>` 具象型 — 借用分離アクセサのみ提供 |
| 8 | `crates/pgpm-core/src/lib.rs` | ファクトリ関数 — `Box<dyn MappingBridge>` を返す公開 API |
| 9 | `crates/pgpm-core/src/numerics/solver.rs` | SOCP 構築・求解 (Eq. 18, 23, 26, 28, 30, 31, 33) |
| 10 | `crates/pgpm-core/src/basis/gaussian.rs` | `GaussianBasis` — Table 1 の Gaussian RBF 実装 |

特に **5 番 (`pgpm_algorithm.rs`)** と **7 番 (`runner.rs`)** の対比が、借用分離パターンと trait デフォルトメソッドによるアルゴリズム実装の要点です。


## 論文参照

- R. Poranne and Y. Lipman, "Provably Good Planar Mappings," *ACM Transactions on Graphics (SIGGRAPH)*, 2014.
  - DOI: [10.1145/2601097.2601123](https://doi.org/10.1145/2601097.2601123)
