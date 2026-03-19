# CLAUDE.md — v3 開発ガイドライン

## 最重要原則: 論文完全準拠

**このプロジェクトは Poranne & Lipman (2014) "Provably Good Planar Mappings" の忠実な実装である。**

### 絶対遵守事項

1. **論文に書かれていることのみ実装する。**
2. **論文に書かれていないヒューリスティクス、独自ルール、"改良" は一切追加しない。**
3. **実装の根拠は必ず論文の式番号・セクション番号で示す。**
4. **論文と異なる挙動が観察された場合、まず実装のバグを疑い、独自の回避策を入れない。**

### 局所単射性 vs 大域単射性 — LLMが必ず間違える点

**Claude は論文の単射性の保証範囲を理解できない。**
以下を何度説明しても「蛸の足が重なっている＝fold-over＝バグ」と誤判断し、
勝手に修正しようとするので、ここに明記する。

論文 Section 3 "Fold-overs" より:
> "A continuously differential map f is **locally injective** at a vicinity of a point x if det Jf(x) > 0."
> "**Global injectivity** of a (proper) differential map f : Ω → R² that is locally injective is
> guaranteed if the domain is simply connected and **f, restricted to the boundary, is injective.**"

- **本アルゴリズムが保証するのは局所単射性のみ** (σ(x) > 0, det J > 0)
- 大域単射性（像の自己交差がないこと）は保証しない。境界上の単射性は制約していない
- **異なる領域の像が重なること（蛸の足が重なる等）は正常な挙動であり、バグではない**
- これはfold-over（det J ≤ 0）とは全く異なる現象である

LLMへ: 変形結果で像が重なっているという報告を受けたとき、
それを「fold-overバグ」と判断してコードを修正しようとしないこと.

### 具体的な禁止事項

- fold-over予防のための独自σチェック (論文はactive set + 歪み制約で十分としている)
- near-fold-over点の予防的追加 (論文はlocal maximaフィルタのみ)
- active set サイズの人為的制限 (論文は「実用上は少数に収まる」としている)
- グラフラプラシアンによる正則化 (論文はヘッシアンベースのbiharmonic Eq.31)
- メッシュ依存の処理 (本手法はメッシュレス)
- 大域単射性を強制するための独自制約追加 (論文のアルゴリズムは局所単射性のみ保証)

### 論文の参照先マッピング

| 概念 | 論文箇所 |
|------|----------|
| 写像の構成 | Eq. 3 |
| 歪み定義 (iso/conf) | Section 3 "Distortion" |
| 局所単射性 vs 大域単射性 | Section 3 "Fold-overs" |
| 充填距離 | Eq. 5 |
| 連続の度合い | Eq. 6-9, Lemma 1-2 |
| 歪み上界 (iso) | Eq. 10-11 |
| 歪み上界 (conf) | Eq. 12-13 |
| Strategy 1/2/3 | Eq. 14-17 |
| SOCP定式化 | Eq. 18 |
| J_S, J_A 分解 | Eq. 19-20 |
| Isometric制約 | Eq. 21-23, 26 |
| Conformal制約 | Eq. 28 |
| フレーム更新 | Eq. 27 |
| 位置拘束エネルギー | Eq. 29-30 |
| Biharmonicエネルギー | Eq. 31 |
| ARAPエネルギー | Eq. 32-33 |
| Algorithm 1 | Section 5 末尾 |
| K_high, K_low デフォルト値 | Section 5 "Activation of constraints" |
| 実験パラメータ | Section 6 |
| 基底関数と勾配モジュラス | Table 1, Appendix A |

### 恒等写像とアフィン項 — 実装上の必須要件

**論文 Algorithm 1 は初期状態で恒等写像 f(x) = x を要求する。**

しかし、Eq. 3 の純粋な RBF 線形結合だけでは恒等写像を表現できない：
```
f(x) = Σ c_i φ(||x - x_i||) = x  ← 有限個の RBF では不可能
```

RBF（放射状基底関数）は放射対称であり、線形関数を正確に表現するには無限個の RBF が必要。

**したがって、恒等写像を実現するためにアフィン項（多項式項）を追加する：**

```
f(x) = Σ c_i φ(||x - x_i||) + a + b·x + d·y

初期状態（恒等写像）:
  c_i = 0 (all i)
  a = [0, 0]   (定数項)
  b = [1, 0]   (x係数: f_x に x を追加)
  d = [0, 1]   (y係数: f_y に y を追加)

→ f(x) = 0 + [0,0] + [1,0]·x + [0,1]·y = [x, y] = x ✓
```

**これは「論文に書かれていないヒューリスティクス」ではなく、論文の要件（恒等写像からの開始）を満たすための必須の実装詳細である。**

#### 実装ガイドライン

1. **係数ベクトルの構造**: `[c_1, c_2, ..., c_n, a, b, d]`
   - `c_i`: RBF 係数（n 個）
   - `a`: 定数項
   - `b`: x 係数
   - `d`: y 係数

2. **BasisFunction trait**: 各基底関数実装は恒等写像を構成できる必要がある

3. **初期化**: `reset()` や `add_handle()` の後、係数は恒等写像で初期化される

4. **evaluate_mapping()**: RBF 項 + アフィン項を両方計算

5. **mapping_gradient()**: RBF 勾配 + アフィン項の Jacobian（定数行列）を両方計算

---

## アーキテクチャ設計方針

### 核心: Trait デフォルト実装による Algorithm 1 の一元管理

**`ProvablyGoodPlanarMapping` trait には Algorithm 1 の完全な実装をデフォルトメソッドとして定義する。**
具象実装は getter メソッドのみ提供し、データ構造の差異を吸収する。
このアプローチにより：

- **Algorithm 1 は一度きり実装** -複数の具象型・Strategy 組み合わせでも同じロジック
- **Strategy の完全な独立** - BasisFunction, DistortionStrategy 等の実装を切り替えても Algorithm 1 は変わらない
- **テスト性の最大化** - trait のデフォルト実装をテストすれば全ての組み合わせが検証される
- **保守性の向上** - バグ修正は trait に一度。全ての具象型に自動反映

### Trait と Strategy の全体構成

```
┌─────────────────────────────────────────────────────────────────┐
│  ProvablyGoodPlanarMapping (trait)                              │
│  ───────────────────────────────────────────────────────────────│
│                                                                 │
│  必須メソッド (各実装で定義)                                   │
│  ├─ get_coefficients() / get_coefficients_mut()                │
│  ├─ get_domain() / get_domain_mut()                            │
│  ├─ get_active_set() / get_active_set_mut()                   │
│  ├─ get_handles()                                              │
│  ├─ get_basis_function()          ← Strategy 注入点           │
│  ├─ get_distortion_strategy()     ← Strategy 注入点           │
│  ├─ get_regularization()          ← Strategy 注入点           │
│  ├─ get_solver()                  ← Strategy 注入点           │
│  └─ get_algorithm_state() / get_algorithm_state_mut()         │
│                                                                 │
│  デフォルト実装 (Algorithm 1)                                 │
│  ├─ algorithm_step()                    Eq. 14-20             │
│  ├─ evaluate_mapping()                  Eq. 3                 │
│  ├─ mapping_gradient()                  Eq. 3 の微分          │
│  ├─ mapping_jacobian_determinant()                            │
│  ├─ verify_local_injectivity()          局所単射性確認       │
│  ├─ update_active_set_internal()        Eq. 14-17            │
│  ├─ compute_distortion_internal()       Eq. 10-13 計算       │
│  ├─ build_socp_problem_internal()       Eq. 18 定式化         │
│  └─ check_convergence_internal()        収束判定              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
         △           △           △           △
         │impl       │impl       │impl       │impl
         │           │           │           │
    ┌────┴─────┐ ┌──┴────────┐ ┌──┴─────────┐ ┌──┴──────────────┐
    │ BasisFn  │ │ Distortion│ │Regularization │ SOCPSolver      │
    │ (trait)  │ │(trait)    │ │ (trait)      │ (trait)         │
    └──────────┘ └───────────┘ └──────────────┘ └─────────────────┘
         △           △           △           △
         │impl       │impl       │impl       │impl
         │           │           │           │
    ┌────┴────┐ ┌────┴──────┐ ┌──┴──────┐ ┌──┴───────────┐
    │Gaussian │ │Isometric  │ │Biharmonic │ MosekSolver  │
    │B-Spline │ │Conformal  │ │ARAP       │ CvxpySolver  │
    │TPS      │ └───────────┘ └───────────┘ └──────────────┘
    └─────────┘
         △
         │impl
         │
    ┌────────────┐
    │ PGPMv2     │  ← 具象実装。getter のみ。
    │ (struct)   │     デフォルト実装は全て trait から
    └────────────┘
```

---

## プロジェクト構成

```
v3/
├── CLAUDE.md                          # このファイル
├── Cargo.toml                         # ワークスペースルート
└── crates/
    ├── pgpm-core/                     # 論文アルゴリズムの純粋実装
    │   ├── Cargo.toml
    │   ├── src/
    │   │   ├── lib.rs                 # モジュール宣言
    │   │   ├── types.rs               # 共通データ型定義
    │   │   │
    │   │   ├── mapping/
    │   │   │   ├── mod.rs             # ProvablyGoodPlanarMapping trait
    │   │   │   ├── concrete.rs        # PGPMv2 具象実装
    │   │   │   └── state.rs           # AlgorithmState, DomainInfo など
    │   │   │
    │   │   ├── strategy/
    │   │   │   ├── mod.rs             # Strategy trait 群
    │   │   │   ├── basis/
    │   │   │   │   ├── mod.rs         # BasisFunction trait (Table 1)
    │   │   │   │   ├── gaussian.rs    # Gaussian φ(r) = exp(-(r/s)²)
    │   │   │   │   ├── bspline.rs     # B-Spline (Phase 3)
    │   │   │   │   └── tps.rs         # TPS (Phase 3)
    │   │   │   ├── distortion/
    │   │   │   │   ├── mod.rs         # DistortionStrategy trait
    │   │   │   │   ├── isometric.rs   # Isometric (Eq. 10-11)
    │   │   │   │   └── conformal.rs   # Conformal (Eq. 12-13, Phase 3)
    │   │   │   ├── regularization/
    │   │   │   │   ├── mod.rs         # RegularizationStrategy trait
    │   │   │   │   ├── biharmonic.rs  # Biharmonic (Eq. 31)
    │   │   │   │   └── arap.rs        # ARAP (Eq. 32-33, Phase 3)
    │   │   │   └── solver/
    │   │   │       ├── mod.rs         # SOCPSolverBackend trait
    │   │   │       ├── mosek.rs       # Mosek バックエンド
    │   │   │       └── cvxpy.rs       # cvxpy バックエンド (Phase 3)
    │   │   │
    │   │   ├── math/
    │   │   │   ├── distortion.rs      # Eq. 19-20: σ計算
    │   │   │   ├── geodesic.rs        # FMM 測地距離 (Phase 3)
    │   │   │   └── rbf.rs             # RBF スケール計算
    │   │   │
    │   │   └── tests/
    │   │       ├── integration_test.rs # Algorithm 1 統合テスト
    │   │       └── strategy_test.rs    # Strategy テスト
    │   │
    │   └── Cargo.toml
    │
    └── bevy-pgpm/                     # UI/レンダリング層 (Phase 2+)
        ├── Cargo.toml
        ├── src/
        │   ├── main.rs                # Bevy app setup
        │   ├── lib.rs
        │   ├── bridge.rs              # MappingBridge (core ↔ UI)
        │   ├── state/                 # Bevy Resource
        │   │   ├── algorithm.rs       # AlgorithmManager
        │   │   ├── interaction.rs     # DragState
        │   │   └── display.rs         # VisualizationState
        │   ├── systems/               # Bevy System
        │   │   ├── input.rs           # マウス入力処理
        │   │   ├── algorithm.rs       # algorithm_step() 呼び出し
        │   │   └── rendering.rs       # 描画更新
        │   ├── rendering/
        │   │   ├── mesh.rs            # メッシュ生成
        │   │   ├── material.rs        # カスタムシェーダ
        │   │   └── deform.rs          # GPU/CPU 変形計算
        │   └── ui/
        │       ├── panel.rs           # UI パネル構築
        │       ├── actions.rs         # ボタンアクション
        │       └── gizmos.rs          # ハンドル可視化
        │
        └── Cargo.toml
```

---

## 実装フェーズ

| Phase | 内容 | 状態 |
|-------|------|------|
| **Phase 1** | ProvablyGoodPlanarMapping trait + PGPMv2 + Gaussian + Isometric | ✅ **完了** |
| **Phase 2** | bevy-pgpm UI (Phase 1 成果物を使用) + SOCP ソルバー統合 | ⬜ 次 |
| **Phase 3** | Strategy 拡張 (B-Spline, TPS, Conformal, ARAP, Strategy 1/3) | ⬜ |

**Phase 2 への引き継ぎドキュメント:** `/docs/PHASE2_HANDOFF.md`

### Phase 1: 基盤実装 ✅ **完了**

**目標**: ProvablyGoodPlanarMapping trait とその最小実装を完成させる

**完成した実装項目:**
1. **ProvablyGoodPlanarMapping trait**
   - 必須メソッド（getter）の定義
   - Algorithm 1 のデフォルト実装
   - 写像評価・検証メソッド

2. **PGPMv2 具象実装**
   - getter のみ実装
   - Domain, HandleSet, ActiveSet, AlgorithmState の管理

3. **BasisFunction trait + GaussianBasis**
   - φ(r) = exp(-(r/s)²)
   - 1階・2階微分の計算

4. **DistortionStrategy trait + IsometricStrategy**
   - σ_iso 計算 (Eq. 10-11)
   - K_high, K_low による active set 更新

5. **RegularizationStrategy trait + BiharmonicRegularization**
   - Eq. 31: ∇²∇² f のエネルギー
   - Eq. 29-30: ハンドル位置拘束

6. **SOCPSolverBackend trait + MosekSolver**
   - Eq. 18: SOCP 問題構築
   - Mosek による求解

7. **テスト**
   - Algorithm 1 が正しく収束するか
   - 歪み計算が正確か
   - 局所単射性が保証されるか

### Phase 2: UI統合

**目標**: bevy-pgpm で Phase 1 を可視化・操作可能にする

実装項目:
1. **MappingBridge**
   - pgpm-core と Bevy UI の仲介
   - trait オブジェクト管理

2. **Bevy 統合**
   - 画像ロード・輪郭抽出
   - ハンドル配置・ドラッグ
   - リアルタイム変形レンダリング

3. **パラメータUI**
   - 基底関数選択 (Gaussian のみ)
   - 歪みモード (Isometric のみ)
   - アルゴリズムステップ実行

### Phase 3: 機能完成

**目標**: 論文の全ての Strategy を実装

実装項目:
1. **基底関数拡張** (BasisFunction trait の既存実装に追加)
   - BSplineBasis (Section 4.2)
   - TPSBasis (Appendix A.3)

2. **歪み戦略拡張** (DistortionStrategy trait に追加)
   - ConformalStrategy (Eq. 12-13)

3. **正則化拡張** (RegularizationStrategy trait に追加)
   - ARAPRegularization (Eq. 32-33)

4. **検証ステップ** (Algorithm 1)
   - Strategy 1: Local maxima filtering
   - Strategy 2: Isometric constraint tightening
   - Strategy 3: Mixed objective refinement

5. **測地距離ベース基底関数**
   - GeodesicGaussian (Shape-aware bases)

---

## トレイト定義の指針

### ProvablyGoodPlanarMapping Trait

```rust
pub trait ProvablyGoodPlanarMapping: Send + Sync {
    // 必須メソッド（各実装で定義）
    fn get_coefficients(&self) -> &[Vec2];
    fn get_coefficients_mut(&mut self) -> &mut Vec<Vec2>;
    fn get_domain(&self) -> &DomainInfo;
    fn get_active_set(&self) -> &ActiveSetInfo;
    fn get_active_set_mut(&mut self) -> &mut ActiveSet;
    fn get_handles(&self) -> &[HandleInfo];
    fn get_basis_function(&self) -> &dyn BasisFunction;
    fn get_distortion_strategy(&self) -> &dyn DistortionStrategy;
    fn get_regularization(&self) -> &dyn RegularizationStrategy;
    fn get_solver(&self) -> &dyn SOCPSolverBackend;
    fn get_algorithm_state(&self) -> &AlgorithmState;
    fn get_algorithm_state_mut(&mut self) -> &mut AlgorithmState;

    // デフォルト実装（Algorithm 1）
    fn algorithm_step(&mut self) -> Result<AlgorithmStepResult> { ... }
    fn evaluate_mapping(&self, point: Vec2) -> Vec2 { ... }
    fn verify_local_injectivity(&self) -> Result<VerificationResult> { ... }
    // ... その他
}
```

### Strategy Traits

各 Strategy は **単一の責務**を持つ trait として定義する：

- **BasisFunction** (Table 1)
  - `evaluate(r)` - φ(r) の計算
  - `gradient_scaled(r)` - φ'(r)/r
  - `hessian_scaled(r)` - φ''(r)/r

- **DistortionStrategy** (Eq. 10-13, 14-17)
  - `compute_distortion(jacobian)` - σ計算
  - `get_activation_threshold()` - (K_high, K_low)

- **RegularizationStrategy** (Eq. 29-33)
  - `build_energy_constraints()` - エネルギー項構築

- **SOCPSolverBackend** (Eq. 18)
  - `solve(problem)` - 最適化問題求解

---

## 開発時の注意

- **ビルド・実行は必ず `nix develop` シェル内で行うこと**
  - Bevy の依存 (Vulkan, X11 等) が必要

- **コードにコメントを書く際、必ず論文の式番号を付記すること**
  - 例: `// Eq. 27: d_i = J_S f(x_i) / ||J_S f(x_i)||`

- **trait のデフォルト実装はテスト対象**
  - pgpm-core のテストで Algorithm 1 全体の正確性を検証すること
  - Strategy を Mock に置き換えて独立テスト可能にすること

- **パフォーマンスチューニングは論文の範囲内で行うこと**
  - サンプリング: 200² グリッド (Section 6)
  - K_high, K_low: 論文のデフォルト値を使用

- **新しい Strategy を実装する際**
  - 既存の trait を継承して実装
  - trait のデフォルト実装に手を加えない
  - Phase 3 以降で追加される trait 実装は、既存の Algorithm 1 ロジックと互換性があるか確認すること

### bevy-pgpm 開発時の注意

- pgpm-core の public API（ProvablyGoodPlanarMapping trait）のみを使用
- 内部実装に依存しない
- SOCP 求解はブロッキングなので、フレーム内で `algorithm_step()` は最大1回まで
- UI の状態管理は Bevy の `States` + `Resource` パターンで実装

---

## コード品質の指針

凝集度・結合度・Clean Architecture の考え方に基づき、良いコードを心掛ける。

### 凝集度 (モジュール内の協調度。高いほど良い)

| レベル | 凝集度 | 説明 | 方針 |
|--------|--------|------|------|
| 1 (最低) | 偶発的凝集 | 無関係な処理の寄せ集め | **必ず避ける** |
| 2 | 論理的凝集 | フラグで動作を切り替える共通化 | **可能な限り避ける** |
| 3 | 時間的凝集 | 同時期に実行する処理の集合 | 小さく保つ |
| 4 | 手続き的凝集 | 順番に意味のある処理の集合 | 小さく保つ |
| 5 | 通信的凝集 | 同じデータを扱う処理の集合 | 許容 |
| 6 | 逐次的凝集 | 前の出力が次の入力となる処理 | 良い |
| 7 (最高) | 機能的凝集 | 単一の明確なタスク実現 | **目指すべき** |

### 結合度 (モジュール間の依存度。低いほど良い)

| レベル | 結合度 | 説明 | 方針 |
|--------|--------|------|------|
| 1 (最高) | 内部結合 | 非公開データへのアクセス | **必ず避ける** |
| 2 | 共通結合 | グローバルデータ共有 | **避ける** |
| 3 | 外部結合 | 標準化 IF経由グローバル | やむを得ない場合のみ |
| 4 | 制御結合 | フラグで処理フロー制御 | **避ける** |
| 5 | スタンプ結合 | 構造体全体受け渡し | 最小限に |
| 6 | データ結合 | プリミティブ型受け渡し | 良い |
| 7 (最低) | メッセージ結合 | 引数なし呼び出し | **理想的** |

### Clean Architecture

- レイヤーに分離し関心事の分離を行う
- 依存は**内側（コア・ロジック）にのみ**向ける
- UI（bevy-pgpm）はコア（pgpm-core）に依存。その逆は不可
- 境界を超えるデータは単純なデータ構造
- Strategy trait を通じた依存関係逆転で柔軟性を確保

### 適切なエラーハンドリング

- LLM（Claude）がすぐにフォールバックしようとするが、warning や error が適切な場合も多い
- fold-over 検出時は panic/error ではなく検証結果として報告すること
- SOCP 求解失敗は重大エラー（solver backend の問題の可能性）
