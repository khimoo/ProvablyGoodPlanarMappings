# CLAUDE.md — v3 開発ガイドライン

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
それを「fold-overバグ」と判断してコードを修正しようとするな。

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
v3/
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
│   │   │   ├── strategy.rs      # ⬜ Phase 3: Strategy 1/2/3 (Eq. 11, 14-17)
│   │   │   └── algorithm.rs     # ✅ Algorithm 1 統合
│   │   └── tests/
│   │       └── integration_verify.rs
│   └── bevy-pgpm/               # Phase 2: Bevy統合 (レンダリング・UI)
│       ├── Cargo.toml
│       ├── src/
│       │   ├── main.rs          # Bevy app setup
│       │   ├── state.rs         # AppState, DeformationState
│       │   ├── input.rs         # マウス・キーボード入力
│       │   ├── rendering/
│       │   │   ├── mod.rs
│       │   │   ├── mesh.rs      # メッシュ生成
│       │   │   ├── material.rs  # カスタムマテリアル
│       │   │   └── deform.wgsl  # 頂点シェーダ
│       │   ├── deform.rs        # GPU/CPU変形パス切り替え
│       │   ├── image.rs         # 画像読み込み
│       │   └── ui.rs            # 情報表示・パラメータ調整
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

## 開発時の注意

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
