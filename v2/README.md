# プログラム概要

[Poranne & Lipman (2014) Provably Good Planar Mappings](https://doi.org/10.1145/2601097.2601123) をRustで実装しました。
このプログラムは、Porraneらの論文のアルゴリズムを実装することで、背景透過png画像をマウスで変形させ新たにpng画像として保存することができるツールとして開発しました。

![デモ](screenshots/demo.gif)
<!-- TODO: screenshots/ にデモ GIF を配置 -->

Provably Good Planar Mapingsは画像変形のアルゴリズムで、特に非凸な空間の変形に焦点を当てています。非凸領域上に離散的に存在する計算点で最適化を実行するだけで、非凸領域全体で画像の裏返りが発生しないことを数学的に保証しています。

## 実行方法

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

## 操作方法

## 操作の流れ

1. **画像読み込み**: 右パネルの「Load Image」で背景透過PNG等を読み込む(デフォルトの画像でも大丈夫)
2. **ハンドル配置**: 画像上を左クリックしてハンドル（制御点）を配置する
3. **変形開始**: 「Start Deforming」を押すと変形モードに切り替わる
4. **ドラッグ変形**: ハンドルを左ドラッグして画像を変形する
5. **保存**: 「Save Image」で変形後の画像をPNGとして保存する

## パネル操作

各パラメータの意味は後述の論文概要を参照してください。

| 操作                      | 説明                                              |
| ----------------------- | ----------------------------------------------- |
| **K ±**                 | 歪み上界 K を調整（デフォルト 3.0、±0.5刻み）                    |
| **λ ÷10 / ×10**         | 正則化係数を対数スケールで調整                                 |
| **Reg. Mode**           | 正則化タイプを切替（ARAP / Biharmonic / Mixed）            |
| **Basis Function**      | 基底関数を選択（Setup時のみ。Gaussian / ShapeAwareGaussian） |
| **Refine (Strategy 2)** | 後述の論文概要セクションを参照してください                           |
| **Reset**               | ハンドルを全消去して Setup モードに戻る                         |

# 実装で工夫した点

実装の際工夫した点は、論文で提示される抽象的に定義されたアルゴリズムや、その定義に必要な諸概念の定義をなるべくそのままコーディングしたことです。traitと構造体を適切に使いわけることで、論文が唯一無二の情報源になることを目指しました。

## 論文のアルゴリズムと実装の方針

<!-- TODO: Algorithm 1 のスクショを配置 -->
<!-- ![Algorithm 1](screenshots/paper_algorithm1.png) -->
<!-- *Figure from Poranne & Lipman, "Provably Good Planar Mappings," ACM Trans. Graph. (SIGGRAPH), 2014. (c) ACM.* -->

制御点をマウスによって操作するとき、毎stepで画像のAlgorithm1が実行されます。
また、Algorithm1を実行するためには、条件を満たした基底関数や歪み尺度をプログラマが与える必要があります。
そのため、基底関数や歪み尺度などをtraitによりモデル化した上で、Algorithm1の構造体はそれらの具象実装を注入する設計が最も論文に沿った実装だと判断しました。

```rust
// 基底関数 (Table 1) — Gaussian, ShapeAwareGaussian 等が実装
trait BasisFunction {
    fn evaluate(&self, x: Vector2) -> DVector;              // Eq. 3: f_i(x)
    fn gradient(&self, x: Vector2) -> (DVector, DVector);   // ∇f_i(x), ヤコビアン計算用 (Eq. 19-20)
    fn gradient_modulus(&self, t: f64) -> f64;              // Table 1: ω_{∇F}(t), 歪み上界の理論的保証 (Eq. 9)
    fn identity_coefficients(&self) -> CoefficientMatrix;   // f(x) = x となる係数, Algorithm 1 の初期解
    ...
}
```

```rust
// 歪みポリシー (Section 3 "Distortion") — IsometricPolicy, ConformalPolicy が実装
trait DistortionPolicy {
    fn distortion_value(&self, sigma_max: f64, sigma_min: f64) -> f64;  // 歪み定義: iso=max{Σ,1/σ}, conf=Σ/σ
    fn append_constraints(&self, ...);  // SOCP歪み制約: iso (Eq. 23, 26) vs conf (Eq. 28)
    ...
}
```

```rust
// Algorithm 1 (Section 5) — trait の具象実装を注入される
struct Algorithm {
    basis: Box<dyn BasisFunction>,       // 差し替え可能な基底関数
    policy: Box<dyn DistortionPolicy>,   // 差し替え可能な歪みポリシー
    domain: Option<Box<dyn Domain>>,     // 差し替え可能なドメイン (Section 4)
    state: AlgorithmState,               // 内部状態 (係数 c, アクティブ集合 Z', フレーム d_i)
    params: MappingParams,               // パラメータ (K, λ, 正則化タイプ)
    ...
}
impl Algorithm {...}
```

また、Algorithm1を実行するメソッドではinitialize, optimize, postprocessメソッドを実行することで、論文のAlgorithm1の表記方法と対応を取りました。手続き的凝集になってしまいますが、論文に沿うという実装方針を優先した結果です。

``` Rust
impl Algorithm {
    // Algorithm 1 の1ステップを実行
    fn step(&mut self, target_handles: &[Vector2]) -> Result<StepInfo, AlgorithmError> {
        let init = self.initialize(target_handles)?;        // D(z) 評価, Z' 更新
        let coefficients = self.optimize(target_handles)?;  // SOCP 求解 (Eq. 18)
        self.postprocess(coefficients);                     // 係数適用, フレーム d_i 更新 (Eq. 27)
        Ok(StepInfo { max_distortion: init.max_distortion, ... })
    }
}
```


# 論文概要

### 写像の構成 (Eq. 3)

画像の変形は写像 f: Ω → ℝ²（Ω ⊂ ℝ²）として表現されます。この写像は基底関数 f_i: Ω → ℝ の線形結合で構成されます（Table 1）。

```
f(x) = Σ c_i f_i(x)    (Eq. 3)
```

位置拘束エネルギー（Eq. 29-30）と正則化エネルギー（Eq. 31-33）を目的関数とし、歪みの上界制約を課した最適化問題を解くというのが主な方針なのですが、数式を工夫することで最適化問題がSOCPに帰着され（Eq. 18）、リアルタイムでインタラクティブな画像変形が可能になります。

### 最適化問題の構造 (Eq. 18, 29-33)

Algorithm 1 は各ステップで以下の最適化問題を解きます。

```
min_c  E_pos(f) + λ E_reg(f)
s.t.   D(f; z) ≤ K   ∀z ∈ Z'
       f = Σ c_i f_i
```

- **E_pos** (Eq. 29-30): 位置拘束エネルギー。制御点 p_l での写像値 f(p_l) が目標位置 q_l に近いことを要求する
- **E_reg**: 正則化エネルギー。変形の滑らかさを制御する
  - **Biharmonic** (Eq. 31): ∬_Ω ‖H_u‖² + ‖H_v‖² dA を最小化。高い滑らかさ（C² 的）を促進する
  - **ARAP** (Eq. 32-33): Σ ‖J_f(r_s) - Q(r_s)‖² を最小化。局所的に剛体に近い変形を得る
- **λ**: 正則化の重み。大きいほど滑らかさを優先し、小さいほど位置拘束を優先する
- **K**: 歪みの上界。K = 1 が最も厳しく（無歪み）、大きくするほど歪みを許容する
- **D**: 歪み関数（前節参照）

ヤコビアンの特異値分解とフレーム d_i の導入により、この問題は SOCP（二次錐計画問題）に帰着され（Eq. 18）、凸最適化ソルバーで係数 c を求解できます。

本プログラムでは K（`MappingParams::k_bound`）、λ（`MappingParams::lambda_reg`）、正則化タイプ（`RegularizationType`）を UI パネルから調整できます。

### 歪み上界の理論的保証と Strategy (Section 4, Eq. 9-15)

Algorithm 1 は離散的な計算点 z ∈ Z' 上でのみ D(z) ≤ K を制約します。しかし局所単射性の保証には、計算点だけでなく**全域** x ∈ Ω で σ(x) > 0（det J_f > 0）が成り立つ必要があります。

論文はこのギャップを、基底関数の勾配モジュラス ω_{∇F}(t)（Table 1）と計算点の充填距離 h（Eq. 5: ドメイン内の任意の点から最近の計算点までの最大距離）を用いて埋めます。特異値関数 Σ(x), σ(x) が ω-連続であることから（Lemma 2, Eq. 9）、離散点での制約が全域での歪み上界 K_max に変換されます（Eq. 11/13）。

論文 Section 4 "Controlling the distortion of f" では、この理論に基づき歪みを制御する3つの戦略が提示されています（"one can control the distortion of the map f in one of three strategies"の直後の箇条書きの部分）。以下ではこれらを Strategy 1, 2, 3 と呼びます。

本プログラムでは Strategy 2 のみ実装しており（`crates/pgpm-core/src/algorithm/strategy.rs`）、論文の実験（Section 6）と同じ運用を採用しています。インタラクション中は固定解像度（200² グリッド）で Algorithm 1 を実行し、変形結果に満足した後に UI パネルの「Refine (Strategy 2)」を手動で実行することで、全域での歪み上界の理論的保証を得ます。Strategy 1, 3 は未実装です。
#### Strategy 1: 検証 (Eq. 11/13)

Algorithm 1 を実行した後、現在の充填距離 h と勾配モジュラス ω から、全域での歪み上界 K_max を事後的に計算します。Isometric の場合 K_max = max{K + ω(h), 1/(1/K − ω(h))}（Eq. 11）で、K_max が有限であれば全域で σ(x) > 0 が保証されます。計算点の追加は行わず、現在の結果がどの程度の保証を持つかを確認するだけです。

#### Strategy 2: 精緻化 (Eq. 14/15)

目標の K_max を達成するために必要な充填距離 h を逆算し（Eq. 14/15）、その h を満たすように計算点グリッドの解像度を上げてから Algorithm 1 を再実行します。計算コストは増加しますが、より厳しい歪み上界を全域で保証できます。

#### Strategy 3: 反復 (Strategy 1 + 2)

Strategy 1 で検証し、不十分であれば Strategy 2 で精緻化するという手順を交互に繰り返します。

# 設計概要
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
```

### pgpm-core
バックエンドで、論文のAlgorithm1はほぼ全てこちら側で実装されています。

`algorithm/mod.rs` ファイルで `Algorithm` 構造体が実装されており、`step()` メソッドが Section 5 の Algorithm 1 と対応しています。

### bevy-pgpm
Bevyで作成したフロントエンドです。

主なファイルは以下となります。
`lifecycle.rs`ファイルで画像読み込みと毎フレームの `Algorithm::step()` 呼び出しを行なっています。

# 参考資料

- R. Poranne and Y. Lipman, "Provably Good Planar Mappings," *ACM Transactions on Graphics (SIGGRAPH)*, 2014.
  - DOI: [10.1145/2601097.2601123](https://doi.org/10.1145/2601097.2601123)
