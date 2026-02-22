# コロケーションポイント設定分析 (Option B)

## 現在のフロー (Strategy 2)

### 1. initialize_domain() 呼び出し時
```python
# bevy_bridge.py line 103-110
self.solver = BetterFitwithGaussian(
    domain_bounds=domain_bounds,
    s_param=epsilon,
    K_solver=3.5,
    K_max=5.0
)
# guarantee_strategy は渡されないため Strategy2 がデフォルト
```

### 2. finalize_setup() 呼び出し時
```python
# bevy_bridge.py line 155-175
self.solver.initialize_mapping(src_handles)

# Strategy2 の場合:
# - get_initial_h() が呼ばれる
# - h = max_span / 200  (interactive_resolution=200)
# - 例: 800x800 画像 → h = 800/200 = 4.0
# - コロケーションポイント数: (800/4)^2 = 40,000 点

# コンター内部のポイントのみにフィルタリング
if self.contour is not None:
    self.solver.collocation_points = filter_points_inside_contour(...)
    self.solver._update_hessian_term()
```

### 3. ドラッグ中 (update_drag)
```python
# Strategy2 では粗いグリッド (h=4.0) で高速に計算
# 40,000 点のコロケーションポイント上で最適化
```

### 4. ドラッグ終了時 (end_drag)
```python
# Strategy2 の場合:
# - get_strict_h_after_drag() が呼ばれる
# - 実際の係数ノルムから必要な h を逆算
# - 例: 計算結果 h_strict = 1.5 < current_h (4.0)
# - グリッドを細かくして再計算
# - 新しいコロケーションポイント: (800/1.5)^2 ≈ 285,000 点
```

---

## Option B での変更: Strategy 1 の場合

### 1. initialize_domain() 呼び出し時 (変更あり)
```python
# bevy_bridge.py (修正後)
def initialize_domain(self, image_width, image_height, epsilon, strategy_type="strategy1"):
    domain_bounds = (0.0, image_width, 0.0, image_height)
    
    # Strategy を明示的に作成
    if strategy_type == "strategy1":
        strategy = Strategy1(
            domain_bounds,
            max_expected_c_norm=100.0  # ← ワーストケース想定値
        )
    else:
        strategy = Strategy2(domain_bounds, interactive_resolution=200)
    
    self.solver = BetterFitwithGaussian(
        domain_bounds=domain_bounds,
        guarantee_strategy=strategy,  # ← 明示的に渡す
        s_param=epsilon,
        K_solver=3.5,
        K_max=5.0
    )
```

### 2. finalize_setup() 呼び出し時 (変更なし)
```python
# bevy_bridge.py (変更なし)
self.solver.initialize_mapping(src_handles)

# Strategy1 の場合:
# - get_initial_h() が呼ばれる
# - h = margin / (2 * max_expected_c_norm * C)
# - 例: margin=0.5, max_expected_c_norm=100, C=1/s^2
#   - s=40 の場合: C = 1/1600 = 0.000625
#   - h = 0.5 / (2 * 100 * 0.000625) = 0.5 / 0.125 = 4.0
#   - または s=10 の場合: C = 1/100 = 0.01
#   - h = 0.5 / (2 * 100 * 0.01) = 0.5 / 2.0 = 0.25
# - コロケーションポイント数: (800/0.25)^2 = 10,240,000 点 (s=10の場合)

# コンター内部のポイントのみにフィルタリング (変更なし)
if self.contour is not None:
    self.solver.collocation_points = filter_points_inside_contour(...)
    self.solver._update_hessian_term()
```

### 3. ドラッグ中 (update_drag) (変更なし)
```python
# Strategy1 では細かいグリッド (h=0.25 など) で計算
# 10,240,000 点のコロケーションポイント上で最適化
# → 計算が重い (Strategy2 の 256 倍)
```

### 4. ドラッグ終了時 (end_drag) (変更あり)
```python
# Strategy1 の場合:
# - get_strict_h_after_drag() が呼ばれる
# - 常に float('inf') を返す
# - グリッド細分化は発生しない
# - コロケーションポイント数は変わらない
```

---

## コロケーションポイント数の比較

### 例: 800x800 画像, s=40, K_solver=3.5, K_max=5.0

#### Strategy 2 (楽観的)
```
初期化時:
  - h = 800 / 200 = 4.0
  - グリッド点数: (800/4)^2 = 40,000 点
  - コンター内: 約 30,000 点 (画像の75%)

ドラッグ中:
  - 30,000 点で高速に計算

ドラッグ終了時:
  - 実際の係数ノルムから h を逆算
  - 例: h_strict = 2.0 < 4.0 → 細分化
  - 新しいグリッド点数: (800/2.0)^2 = 160,000 点
  - コンター内: 約 120,000 点
  - 再最適化 (5 イテレーション)
```

#### Strategy 1 (悲観的)
```
初期化時:
  - margin = min(5.0 - 3.5, 1/3.5 - 1/5.0) = min(1.5, 0.057) = 0.057
  - C = 1 / s^2 = 1 / 1600 = 0.000625
  - h = 0.057 / (2 * 100 * 0.000625) = 0.057 / 0.125 = 0.456
  - グリッド点数: (800/0.456)^2 ≈ 3,080,000 点
  - コンター内: 約 2,310,000 点

ドラッグ中:
  - 2,310,000 点で計算 (Strategy2 の 77 倍)
  - 計算時間が大幅に増加

ドラッグ終了時:
  - get_strict_h_after_drag() → float('inf')
  - グリッド細分化なし
  - コロケーションポイント数は変わらない
```

---

## パラメータの影響分析

### max_expected_c_norm の影響

```python
# Strategy1 の h 計算式:
# h = margin / (2 * max_expected_c_norm * C)

# max_expected_c_norm が大きい → h が小さい → グリッドが細かい
# max_expected_c_norm が小さい → h が大きい → グリッドが粗い

例: margin=0.057, C=0.000625
  - max_expected_c_norm=50  → h = 0.057/(2*50*0.000625) = 0.912
  - max_expected_c_norm=100 → h = 0.057/(2*100*0.000625) = 0.456
  - max_expected_c_norm=200 → h = 0.057/(2*200*0.000625) = 0.228
```

### s_param (ガウス基底の幅) の影響

```python
# C = 1 / s^2 なので、s が大きい → C が小さい → h が大きい

例: margin=0.057, max_expected_c_norm=100
  - s=10  → C=0.01    → h = 0.057/(2*100*0.01) = 0.0285
  - s=40  → C=0.000625 → h = 0.057/(2*100*0.000625) = 0.456
  - s=100 → C=0.0001  → h = 0.057/(2*100*0.0001) = 2.85
```

---

## Option B での推奨設定

### シナリオ 1: 高速ドラッグ重視 (Strategy 2 継続)
```python
# bevy_bridge.py
def initialize_domain(self, image_width, image_height, epsilon):
    domain_bounds = (0.0, image_width, 0.0, image_height)
    
    strategy = Strategy2(
        domain_bounds,
        interactive_resolution=200  # 粗いグリッド
    )
    
    self.solver = BetterFitwithGaussian(
        domain_bounds=domain_bounds,
        guarantee_strategy=strategy,
        s_param=epsilon,
        K_solver=3.5,
        K_max=5.0
    )
```

**特性:**
- 初期グリッド: 粗い (h ≈ 4.0)
- ドラッグ中: 高速
- ドラッグ終了時: グリッド細分化の可能性あり

---

### シナリオ 2: 安全性重視 (Strategy 1)
```python
# bevy_bridge.py
def initialize_domain(self, image_width, image_height, epsilon):
    domain_bounds = (0.0, image_width, 0.0, image_height)
    
    strategy = Strategy1(
        domain_bounds,
        max_expected_c_norm=100.0  # ワーストケース想定
    )
    
    self.solver = BetterFitwithGaussian(
        domain_bounds=domain_bounds,
        guarantee_strategy=strategy,
        s_param=epsilon,
        K_solver=3.5,
        K_max=5.0
    )
```

**特性:**
- 初期グリッド: 細かい (h ≈ 0.456)
- ドラッグ中: 遅い (計算量が多い)
- ドラッグ終了時: グリッド細分化なし (安全性保証)

---

### シナリオ 3: バランス型 (Strategy 1 + 調整)
```python
# bevy_bridge.py
def initialize_domain(self, image_width, image_height, epsilon):
    domain_bounds = (0.0, image_width, 0.0, image_height)
    
    # max_expected_c_norm を小さくして h を大きくする
    strategy = Strategy1(
        domain_bounds,
        max_expected_c_norm=50.0  # より楽観的な想定
    )
    
    self.solver = BetterFitwithGaussian(
        domain_bounds=domain_bounds,
        guarantee_strategy=strategy,
        s_param=epsilon,
        K_solver=3.5,
        K_max=5.0
    )
```

**特性:**
- 初期グリッド: 中程度 (h ≈ 0.912)
- ドラッグ中: 中程度の速度
- ドラッグ終了時: グリッド細分化なし

---

## コロケーションポイント数の計算式

### Strategy 2
```
h_interactive = max_span / interactive_resolution
grid_points = (width/h)^2 + (height/h)^2  (近似)
```

### Strategy 1
```
margin = min(K_max - K_solver, 1/K_solver - 1/K_max)
C = 1 / s^2
h_strict = margin / (2 * max_expected_c_norm * C)
grid_points = (width/h)^2 + (height/h)^2  (近似)
```

---

## コンター内部フィルタリングの影響

### 現在の実装
```python
# bevy_bridge.py line 165-170
if self.contour is not None:
    original_count = len(self.solver.collocation_points)
    self.solver.collocation_points = filter_points_inside_contour(...)
    filtered_count = len(self.solver.collocation_points)
    self.solver._update_hessian_term()
```

**重要な点:**
1. **フィルタリングは Strategy に依存しない**
   - Strategy 1 でも Strategy 2 でも同じ処理
   - コンター内部のポイントのみを使用

2. **H_term の再計算が必要**
   - コロケーションポイントが変わったため
   - ヘッシアン行列を再計算

3. **ドラッグ終了時の再フィルタリング (Strategy 2 のみ)**
   ```python
   # bevy_bridge.py line 265-270
   if was_refined:
       if self.contour is not None:
           self.solver.collocation_points = filter_points_inside_contour(...)
           self.solver._update_hessian_term()
   ```
   - Strategy 2 でグリッド細分化が発生した場合のみ
   - Strategy 1 では発生しない

---

## Option B での実装時の注意点

### 1. コロケーションポイント数の大幅な増加
```
Strategy 2: 30,000 点
Strategy 1: 2,310,000 点 (77倍)
```
- メモリ使用量が増加
- 計算時間が大幅に増加
- ドラッグ中のレスポンスが低下

### 2. max_expected_c_norm の選択
```python
# 小さすぎる → グリッドが細かすぎて遅い
# 大きすぎる → 安全性保証が弱い

推奨値: 50.0 ~ 150.0
```

### 3. s_param (epsilon) との相互作用
```python
# s が小さい → C が大きい → h が小さい → グリッドが細かい
# s が大きい → C が小さい → h が大きい → グリッドが粗い

現在の設定: epsilon = 40.0
```

### 4. ドラッグ終了時の動作の違い
```
Strategy 2:
  - グリッド細分化の可能性あり
  - 再最適化が発生する可能性あり
  - end_drag() の処理時間が不確定

Strategy 1:
  - グリッド細分化なし
  - 再最適化なし
  - end_drag() の処理時間が確定的
```

---

## 推奨: Option B での実装方針

### 1. デフォルトは Strategy 2 を継続
```python
# 既存ユーザーへの影響を最小化
def initialize_domain(self, image_width, image_height, epsilon, 
                     strategy_type="strategy2"):  # ← デフォルト
```

### 2. Strategy 1 は明示的に選択
```python
# Rust 側から明示的に指定
PyCommand::InitializeDomain {
    width: 800.0,
    height: 800.0,
    epsilon: 40.0,
    strategy: "strategy1".to_string(),
}
```

### 3. max_expected_c_norm は環境に応じて調整
```python
# 画像サイズに応じた調整
if image_width * image_height > 1000000:  # 1000x1000 以上
    max_norm = 150.0  # より粗いグリッド
else:
    max_norm = 100.0  # 標準
```

### 4. パフォーマンス監視
```python
# ドラッグ中の計算時間を記録
import time
start = time.time()
self.solver.update_drag(target_handles, num_iterations=2)
elapsed = time.time() - start
print(f"update_drag took {elapsed:.3f}s")
```

---

## まとめ

### コロケーションポイント設定の流れ (Option B)

```
initialize_domain(strategy_type)
  ↓
Strategy1 or Strategy2 を作成
  ↓
BetterFitwithGaussian に渡す
  ↓
finalize_setup()
  ↓
initialize_mapping() → get_initial_h() → generate_grid()
  ↓
コロケーションポイント生成
  ↓
コンター内部フィルタリング (オプション)
  ↓
H_term 再計算
  ↓
ドラッグ開始
```

### Strategy 1 の場合の特徴
- **初期化時**: 細かいグリッド生成 (計算量多い)
- **ドラッグ中**: 多数のコロケーションポイント上で計算 (遅い)
- **ドラッグ終了時**: グリッド細分化なし (安全性保証)
- **全体**: 計算量が多いが、安全性が高い
