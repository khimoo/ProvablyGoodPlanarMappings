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
| 写像の表現 (Eq. 3) | `Algorithm::evaluate_mapping_at()` |
| 歪み尺度 (Section 3) | `DistortionPolicy` trait |
| SOCP 最適化 (Eq. 18) | `solver::solve_socp()` |
| アクティブセット (Algorithm 1) | `active_set::update_active_set()` |
| 基底関数 (Table 1) | `BasisFunction` trait |
| Strategy 2 精緻化 | `strategy::refine_to_target()` |

## アーキテクチャ

```
pgpm-core (論文アルゴリズム)          bevy-pgpm (インタラクティブUI)
┌─────────────────────────┐         ┌──────────────────────┐
│ BasisFunction (trait)    │         │ Bevy App             │
│ DistortionPolicy (trait) │◄────────│   input / rendering  │
│ Domain (trait)           │  pub API│   state / ui         │
│ Algorithm (struct)       │         │                      │
│ strategy (自由関数)       │         │                      │
└─────────────────────────┘         └──────────────────────┘
     ▲ 依存方向: 内側にのみ向かう
```

- **pgpm-core**: 論文アルゴリズムの純粋な実装。Bevy・UI・画像処理への依存は一切ありません。
- **bevy-pgpm**: Bevy によるインタラクティブ UI。`pgpm-core` の公開 API (`Algorithm` 構造体) にのみ依存します。

## 設計の工夫 — C++/Python の抽象クラスから Rust の合成設計へ

### C++ で実装する場合の典型的な設計

この論文を C++ や Python で実装する場合、典型的にはオブジェクト指向の継承を使って設計します。

```cpp
// C++: 抽象クラスによる設計
class BasisFunction {
public:
    virtual ~BasisFunction() = default;
    virtual VectorXd evaluate(Vector2d x) const = 0;
    virtual pair<VectorXd, VectorXd> gradient(Vector2d x) const = 0;
    virtual double gradient_modulus(double t) const = 0;
    // ...
};

class DistortionPolicy {
public:
    virtual ~DistortionPolicy() = default;
    virtual double distortion_value(double sigma_max, double sigma_min) const = 0;
    virtual void append_constraints(SOCPBuilder& builder, ...) const = 0;
    // ...
};

// Algorithm は抽象クラスを仮想関数で呼び出す
class Algorithm {
    unique_ptr<BasisFunction> basis_;
    unique_ptr<DistortionPolicy> policy_;
public:
    StepInfo step(const vector<Vector2d>& targets);
    // ...
};
```

抽象クラスを使い、`BasisFunction` を `GaussianBasis` / `BSplineBasis` / `TpsBasis` に、`DistortionPolicy` を `IsometricPolicy` / `ConformalPolicy` にそれぞれ差し替える設計です。`Algorithm` クラスは仮想関数テーブル経由でこれらを呼び出します。

さらに C++ では、Algorithm 1 の共通ロジックを基底クラスに置き、テンプレートメソッドパターンで差し替え点を定義する設計がよく使われます。

### Rust での設計: trait による合成

Rust には C++ の継承がありません。代わりに **trait** (振る舞いの契約) と **合成** (フィールドとして部品を持つ) を使います。

**trait は「振る舞いの差し替え点」にのみ使う。** 論文で複数の実装が存在し、入出力の型は共通だが振る舞いが異なる概念を trait にしました。

```rust
// 基底関数 — Table 1 で Gaussian, B-Spline, TPS が列挙されている
pub trait BasisFunction: Send + Sync {
    fn evaluate(&self, x: Vector2<f64>) -> DVector<f64>;
    fn gradient(&self, x: Vector2<f64>) -> (DVector<f64>, DVector<f64>);
    fn gradient_modulus(&self, t: f64) -> f64;
    // ...
}

// 歪み種別 — Section 3 で Isometric と Conformal が定義されている
pub trait DistortionPolicy: Send + Sync {
    fn distortion_value(&self, sigma_max: f64, sigma_min: f64) -> f64;
    fn append_constraints(&self, ...);
    // ...
}

// ドメイン — Section 4 で非凸ドメインの扱いが議論されている
pub trait Domain: Send + Sync {
    fn contains(&self, pt: &Vector2<f64>) -> bool;
}
```

**Algorithm は普通の構造体。** 論文の Algorithm 1 は1つしかないので、trait にする必要がありません。C++ の仮想関数テーブルに相当するのは trait object (`Box<dyn BasisFunction>`) であり、`Algorithm` はこれらを**フィールドとして保持する合成**で実現します。

```rust
pub struct Algorithm {
    basis: Box<dyn BasisFunction>,        // 差し替え可能な基底関数
    policy: Box<dyn DistortionPolicy>,    // 差し替え可能な歪み種別
    domain: Option<Box<dyn Domain>>,      // 差し替え可能なドメイン
    state: AlgorithmState,                // 内部状態 (係数, アクティブセット等)
    params: MappingParams,                // パラメータ (K, λ 等)
    config: SolverConfig,
}

impl Algorithm {
    pub fn step(&mut self, targets: &[Vector2<f64>]) -> Result<StepInfo, AlgorithmError> {
        // self.basis, self.policy に直接アクセスできる
        // C++ のような仮想関数テーブル経由の呼び出しと同等
    }
}
```

### C++ との比較: なぜこの設計になるか

| 設計上の判断 | C++ | Rust (本実装) |
|------------|-----|--------------|
| 差し替え可能な部品 | 抽象クラス + 仮想関数 | trait + trait object (`Box<dyn T>`) |
| アルゴリズム本体 | 具象クラス (or テンプレートメソッドの基底クラス) | 普通の構造体 + `impl` ブロック |
| 部品の所有 | `unique_ptr<Base>` | `Box<dyn Trait>` |
| 部品の合成 | コンストラクタで注入 | `new()` でフィールドに格納 |
| 型消去 | 基底クラスのポインタ | `Box<dyn Trait>` |

C++ で「Algorithm を基底クラスにしてテンプレートメソッドパターンで `step()` を定義し、サブクラスが `parts()` だけオーバーライドする」という設計は Rust ではアンチパターンになります。Rust の trait のデフォルトメソッドから `self` のフィールドにアクセスできないため、`parts()` / `parts_mut()` のような借用分離ハックが必要になってしまいます。

代わりに、Rust では `Algorithm` を普通の構造体にして `self.basis`, `self.state` に直接アクセスする方が自然です。「実装が1つしかないものを trait にしない」というのが Rust の原則です。

### Strategy の分離 — Algorithm 1 の「使い方」は自由関数で

論文 Section 4 の Strategy 1/2/3 は Algorithm 1 の内部部品ではなく、Algorithm 1 を「どう使うか」の上位ワークフローです。

- **Strategy 1**: Algorithm 1 を1回実行後、K_max を計算（検証のみ, Eq. 11/13）
- **Strategy 2**: 必要な h を計算 → グリッド再構築 → Algorithm 1 を収束まで反復（Eq. 14/15）
- **Strategy 3**: Strategy 1 + 2 の組み合わせ（Eq. 16/17）

C++ ではこれらを `Algorithm` のメソッドにするか、Strategy パターンのクラスにするかの選択肢がありますが、Rust では `&mut Algorithm` を受け取る自由関数にします。

```rust
pub mod strategy {
    /// Strategy 1: 現在の状態から K_max を計算（Eq. 11/13）
    pub fn verify_distortion_bound(alg: &Algorithm) -> VerificationResult;

    /// Strategy 2: 必要なグリッド密度で Algorithm 1 を収束まで実行（Eq. 14/15）
    pub fn refine_to_target(alg: &mut Algorithm, ...) -> Result<RefinementResult>;

    /// Strategy 3: 検証 → 不十分なら精緻化（Eq. 16/17）
    pub fn verify_and_refine(alg: &mut Algorithm, ...) -> Result<RefinementResult>;
}
```

この設計により:
- `Algorithm` の `impl` は Algorithm 1 のコアロジック（`step`, `evaluate_mapping_at` 等）に集中し、凝集度が高い
- Strategy は Algorithm 1 を外から操作する上位ワークフローとして、抽象レベルが明確に分離される
- Strategy 3 が Strategy 1 + 2 の合成であることが自然に表現できる

### 正則化は enum で切り替え

Biharmonic (Eq. 31) と ARAP (Eq. 32-33) は trait にせず enum で扱っています。実装が2つで将来増える想定がなく、両方を混合して使うケース（`Mixed`）があるため、enum + match の方がシンプルです。

```rust
pub enum RegularizationType {
    Biharmonic,              // Eq. 31
    Arap,                    // Eq. 33
    Mixed { lambda_bh: f64, lambda_arap: f64 },
}
```

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
- ハンドルのドラッグに応じて `Algorithm::step()` を呼び出し
- CPU レンダリングパス: `evaluate_mapping_at()` でメッシュ頂点を変形
- UI パネルで K, λ, 正則化タイプを調整可能

pgpm-core の `Algorithm` 構造体の pub API のみに依存しており、アルゴリズムの内部実装には触れていません。

## 注目ファイル一覧

pgpm-core の設計を理解するために、以下のファイルを順に読むことを推奨します。

| 順序 | ファイル | 内容 |
|------|---------|------|
| 1 | `crates/pgpm-core/src/basis/mod.rs` | `BasisFunction` trait 定義 |
| 2 | `crates/pgpm-core/src/policy/mod.rs` | `DistortionPolicy` trait 定義と 2 つの実装 |
| 3 | `crates/pgpm-core/src/model/domain.rs` | `Domain` trait 定義と `PolygonDomain` |
| 4 | `crates/pgpm-core/src/model/types.rs` | `AlgorithmState`, `MappingParams` 等の型定義 |
| 5 | **`crates/pgpm-core/src/algorithm/mod.rs`** | **`Algorithm` 構造体 — Algorithm 1 の実装** |
| 6 | `crates/pgpm-core/src/algorithm/strategy.rs` | Strategy 1/2/3 (自由関数) |
| 7 | `crates/pgpm-core/src/algorithm/active_set.rs` | Active set管理 |
| 8 | `crates/pgpm-core/src/lib.rs` | ファクトリ関数 — 公開 API |
| 9 | `crates/pgpm-core/src/numerics/solver.rs` | SOCP 構築・求解 (Eq. 18, 23, 26, 28, 30, 31, 33) |
| 10 | `crates/pgpm-core/src/basis/gaussian.rs` | `GaussianBasis` — Table 1 の Gaussian RBF 実装 |

特に **5 番 (`algorithm/mod.rs`)** が設計の要点です。`Algorithm` が `BasisFunction`, `DistortionPolicy`, `Domain` の trait object をフィールドとして保持し、Algorithm 1 のロジックを `impl` ブロックに直接実装しています。

## 論文参照

- R. Poranne and Y. Lipman, "Provably Good Planar Mappings," *ACM Transactions on Graphics (SIGGRAPH)*, 2014.
  - DOI: [10.1145/2601097.2601123](https://doi.org/10.1145/2601097.2601123)
