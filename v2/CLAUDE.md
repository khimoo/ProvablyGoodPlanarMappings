# CLAUDE.md — v2 開発ガイドライン

## 最重要原則: 論文完全準拠

**このプロジェクトは Poranne & Lipman (2014) "Provably Good Planar Mappings" の忠実な実装である。**

### 絶対遵守事項

1. **論文に書かれていることのみ実装する。**
2. **論文に書かれていないヒューリスティクス、独自ルール、"改良" は一切追加しない。**
3. **実装の根拠は必ず論文の式番号・セクション番号で示す。**
4. **論文と異なる挙動が観察された場合、まず実装のバグを疑い、独自の回避策を入れない。**

### 局所単射性 vs 大域単射性 — LLMが必ず間違える点

**Claude (opus-4-6) は論文の単射性の保証範囲を理解できない。**
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

### 論文の参照先マッピング (ProvablyGoodPlanarMappings.md)

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

## プロジェクト構成

```
v2/
├── CLAUDE.md                    # このファイル
├── DESIGN.md                    # 設計書
├── Cargo.toml                   # ワークスペースルート
├── crates/
│   ├── pgpm-core/               # 論文アルゴリズムの純粋実装
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── types.rs
│   │   │   ├── basis/           # 基底関数 (Table 1)
│   │   │   │   ├── mod.rs
│   │   │   │   ├── gaussian.rs  # ✅ Phase 1: ユークリッド距離Gaussian
│   │   │   │   ├── shape_aware_gaussian.rs # ✅ Phase 3: 測地距離Gaussian
│   │   │   │   ├── bspline.rs   # ⬜ Phase 3
│   │   │   │   └── tps.rs       # ⬜ Phase 3
│   │   │   ├── geodesic.rs      # ✅ FMM測地距離計算 (Section "Shape aware bases")
│   │   │   ├── distortion.rs    # ✅ 歪み計算 (Eq. 19-20)
│   │   │   ├── active_set.rs    # ✅ Active set管理 (Algorithm 1)
│   │   │   ├── solver.rs        # ✅ SOCP構築・求解 (Eq. 18, 23, 26, 28, 30)
│   │   │   ├── strategy.rs      # ✅ Strategy 2 (Eq. 11, 14); ⬜ Phase 3: Strategy 1/3 (Eq. 15-17)
│   │   │   └── algorithm.rs     # ✅ Algorithm 1 統合
│   │   └── tests/
│   │       └── integration_verify.rs
│   └── bevy-pgpm/               # Phase 2: Bevy統合 (レンダリング・UI)
│       ├── Cargo.toml
│       ├── src/
│       │   ├── main.rs          # Bevy app setup, カメラ
│       │   ├── lib.rs           # モジュール宣言
│       │   ├── lifecycle.rs     # 画像ロード・アルゴリズムステップ
│       │   ├── input.rs         # マウス入力 (ハンドル配置・ドラッグ)
│       │   ├── domain/          # ドメインユーティリティ (Bevy非依存部分)
│       │   │   ├── coords.rs    # 座標変換 (pixel <-> world) 一元管理
│       │   │   ├── rbf.rs       # RBFスケール計算
│       │   │   └── image_loader.rs # 画像読み込み・輪郭抽出
│       │   ├── state/           # Bevy Resource定義 (責務分割)
│       │   │   ├── algorithm.rs # AlgorithmState (アルゴリズム・ハンドル)
│       │   │   ├── interaction.rs # DragState (ドラッグUI状態)
│       │   │   ├── params.rs    # AlgoParams, BasisType, RegMode
│       │   │   ├── image_info.rs # ImageInfo, ImagePathConfig
│       │   │   └── display_info.rs # DeformationInfo (表示用)
│       │   ├── rendering/
│       │   │   ├── mesh.rs      # メッシュ生成
│       │   │   ├── material.rs  # カスタムマテリアル + identity()
│       │   │   ├── gpu_deform.rs # GPU変形パス (シェーダuniform更新)
│       │   │   └── cpu_deform.rs # CPU変形パス (頂点位置更新)
│       │   └── ui/              # UI (責務分割)
│       │       ├── markers.rs   # マーカーコンポーネント定義
│       │       ├── panel.rs     # パネル構築
│       │       ├── actions.rs   # ボタンアクション
│       │       ├── display.rs   # テキスト更新・ボタン視覚フィードバック
│       │       └── gizmos.rs    # ハンドル描画
│       └── assets/
└── assets/
```

## 実装フェーズ

| Phase | 内容 | 状態 |
|-------|------|------|
| **Phase 1** | pgpm-core 最小構成 (Gaussian + Isometric + Algorithm 1) | ✅ 完了 |
| **Phase 2** | bevy-pgpm UI構築 (Gaussian + Isometric のみで動作) | ⬜ 次 |
| **Phase 3** | pgpm-core 完成 (Strategy, B-Spline, TPS) + bevy-pgpm 拡張 | ⬜ |

**Phase 2 の方針**: pgpm-core の Phase 1 成果物のみで動作するUIを構築する。
Phase 3 の未実装機能はスタブ/無効化で対応し、UIレベルで制限を明示する。
- Strategy 1/2/3 → `verify_and_refine()` は `CannotGuarantee` を返す → UIで「未検証」表示
- B-Spline / TPS → 基底選択UIでは Gaussian のみ有効
- Conformal → Isometric モードのみ使用可能

## コード品質の指針

凝集度・結合度・Clean Architectureの考え方に基づき、良いコードを心掛ける。

### 凝集度 (モジュール内の協調度。高いほど良い)

| レベル | 凝集度 | 説明 | 方針 |
|--------|--------|------|------|
| 1 (最低) | 偶発的凝集 | 無関係な処理の寄せ集め。「とりあえず動く」状態 | **必ず避ける** |
| 2 | 論理的凝集 | フラグで動作を切り替える共通化 | **可能な限り避ける** |
| 3 | 時間的凝集 | 同時期に実行する処理の集合（初期化等）。順序入替可 | できるだけ小さく保つ |
| 4 | 手続き的凝集 | 順番に意味のある処理の集合 | できるだけ小さく保つ |
| 5 | 通信的凝集 | 同じデータを扱う処理の集合。順序は不問 | 許容 |
| 6 | 逐次的凝集 | 前の出力が次の入力となる処理の集合 | 良い |
| 7 (最高) | 機能的凝集 | 単一の明確なタスクを実現し、これ以上分解できない | **目指すべき状態** |

- 凝集度はモジュール内の**一番低いレベル**で評価する
- 関数分割のしすぎは認知負荷を上げる。意味のわかる単位で区切ること

### 結合度 (モジュール間の依存度。低いほど良い)

| レベル | 結合度 | 説明 | 方針 |
|--------|--------|------|------|
| 1 (最高) | 内部結合 | 非公開データへの直接アクセス（リフレクション等） | **必ず避ける** |
| 2 | 共通結合 | 複数モジュールが同じグローバルデータにアクセス | **避ける** |
| 3 | 外部結合 | 標準化インターフェース経由でのグローバルアクセス | やむを得ない場合のみ |
| 4 | 制御結合 | フラグで他モジュールの処理フローを制御 | **避ける** |
| 5 | スタンプ結合 | 構造体/クラス全体の受け渡し（不要データを含みうる） | 必要最小限に |
| 6 | データ結合 | プリミティブ型など必要最小限のデータの受け渡し | 良い |
| 7 (最低) | メッセージ結合 | 引数なしのやりとり（タイミングのみ伝達） | **理想的** |

- 凝集度が高くなれば結合度は低くなる傾向がある

### Clean Architecture

- レイヤーに分離し**関心事の分離**を行う
- 依存は**内側（ビジネスロジック）にのみ**向ける。外側（UI、DB、フレームワーク）に依存しない
- 境界を超えるデータは単純なデータ構造を使い、内側が外側の知識を持たないようにする
- 内側から外側への情報伝達にはストリームや依存関係逆転（インターフェース）を使う

### 適切にエラーハンドリングすること.
AIはすぐにフォールバックしようとするが, warningやerrorが良いタイミングもよね

## 開発時の注意

- **ビルド・実行は必ず `nix develop` シェル内で行うこと**
  - `nix develop` で開発シェルに入った後に `cargo build`, `cargo run` 等を実行する
  - もしくは `nix run` でそのまま実行可能
  - Nix 外でビルドすると Bevy の依存ライブラリ (Vulkan, X11, Wayland 等) が見つからない
- コードにコメントを書く際、必ず論文の式番号を付記すること
  - 例: `// Eq. 27: d_i = J_S f(x_i) / ||J_S f(x_i)||`
- テストケースは論文 Section 6 の実験設定を再現すること
- パフォーマンスチューニングは論文の範囲内で行うこと（例: 200²グリッド）

### bevy-pgpm 開発時の注意 (Phase 2)

- pgpm-core の公開APIのみを使用し、内部実装に依存しないこと
- Phase 3 未実装の機能を呼び出す箇所には `// Phase 3: ...` コメントを残すこと
- SOCP求解はブロッキングなので、フレーム内で `step()` を1回のみ呼ぶこと
  - 将来の非同期化が必要になったら Phase 3 以降で検討
- GPU レンダリング (頂点シェーダでの写像評価) は v2 の `deform.wgsl` を参考にすること
- UI の状態管理は Bevy の `States` + `Resource` パターンで行うこと
