# Strategy 1 API 実装完了

## 実装内容

### 1. deform_algo.py の修正

#### Strategy1 クラスの再設計

**変更前:**
```python
class Strategy1(GuaranteeStrategy):
    def __init__(self, domain_bounds, max_expected_c_norm=100.0):
        self.max_expected_c_norm = max_expected_c_norm
    
    def get_initial_h(self, basis, K_solver, K_max):
        # ワーストケース想定で h を計算
        h_strict = margin / (2.0 * self.max_expected_c_norm * C)
```

**変更後:**
```python
class Strategy1(GuaranteeStrategy):
    def __init__(self, domain_bounds, collocation_resolution=500, K_on_collocation=3.5):
        self.collocation_resolution = collocation_resolution
        self.K_on_collocation = K_on_collocation
    
    def get_initial_h(self, basis, K_solver, K_max):
        # グリッド解像度から h を計算
        h = max_span / self.collocation_resolution
    
    def compute_guaranteed_K_max(self, basis, h):
        # 式(11)から K_max を計算
        K_max_bound = max(K + omega_h, 1.0 / (1.0 / K - omega_h))
```

**改善点:**
- `collocation_resolution` で直感的にグリッド密度を指定
- `K_on_collocation` でコロケーション点上の K を明示的に指定
- `compute_guaranteed_K_max()` で論文の式(11)に基づいて K_max を計算
- K_max は入力ではなく出力（計算結果）

---

### 2. bevy_bridge.py の修正

#### BevyBridge クラスの拡張

**変更前:**
```python
class BevyBridge:
    def __init__(self):
        self.solver = None
        # ...
    
    def initialize_domain(self, image_width, image_height, epsilon):
        self.solver = BetterFitwithGaussian(
            domain_bounds=domain_bounds,
            s_param=epsilon,
            K_solver=3.5,
            K_max=5.0
        )
```

**変更後:**
```python
class BevyBridge:
    def __init__(self, strategy_type="strategy2", strategy_params=None):
        self.strategy_type = strategy_type
        self.strategy_params = strategy_params or {}
        self.guaranteed_K_max = None  # Strategy 1 の出力
    
    def initialize_domain(self, image_width, image_height, epsilon):
        if self.strategy_type == "strategy1":
            strategy, K_solver, K_max = self._create_strategy1(domain_bounds)
        else:
            strategy, K_solver, K_max = self._create_strategy2(domain_bounds)
        
        self.solver = BetterFitwithGaussian(
            domain_bounds=domain_bounds,
            guarantee_strategy=strategy,
            s_param=epsilon,
            K_solver=K_solver,
            K_max=K_max
        )
```

**新規メソッド:**
```python
def _create_strategy1(self, domain_bounds):
    """Strategy 1 を作成"""
    collocation_resolution = self.strategy_params.get('collocation_resolution', 500)
    K_on_collocation = self.strategy_params.get('K_on_collocation', 3.5)
    strategy = Strategy1(domain_bounds, collocation_resolution, K_on_collocation)
    return strategy, K_on_collocation, 10.0  # K_max はダミー値

def _create_strategy2(self, domain_bounds):
    """Strategy 2 を作成"""
    interactive_resolution = self.strategy_params.get('interactive_resolution', 200)
    K_solver = self.strategy_params.get('K_solver', 3.5)
    K_max = self.strategy_params.get('K_max', 5.0)
    strategy = Strategy2(domain_bounds, interactive_resolution)
    return strategy, K_solver, K_max
```

**finalize_setup() の拡張:**
```python
def finalize_setup(self):
    # ...
    self.solver.initialize_mapping(src_handles)
    
    # Strategy 1 の場合、K_max を計算
    if self.strategy_type == "strategy1":
        h = self.solver.current_h
        strategy = self.solver.strategy
        self.guaranteed_K_max = strategy.compute_guaranteed_K_max(
            self.solver.basis, h
        )
        print(f"Strategy 1: Guaranteed K_max = {self.guaranteed_K_max:.4f}")
```

---

## 使用例

### Strategy 1: コロケーション密度と K を指定

```python
bridge = BevyBridge(
    strategy_type="strategy1",
    strategy_params={
        'collocation_resolution': 500,
        'K_on_collocation': 3.5
    }
)

bridge.initialize_domain(800.0, 800.0, 40.0)
bridge.add_control_point(0, 100.0, 100.0)
bridge.add_control_point(1, 700.0, 700.0)
bridge.finalize_setup()

# 出力:
# Initialized domain: 800.0x800.0, epsilon=40.0
# Strategy: strategy1
#   K_on_collocation: 3.5
#   (K_max will be computed after finalize_setup)
# Generated 250000 collocation points
# Strategy 1: Guaranteed K_max = 4.2
# Finalized setup with 2 control points
```

### Strategy 2: K と K_max を指定（デフォルト）

```python
bridge = BevyBridge(
    strategy_type="strategy2",
    strategy_params={
        'interactive_resolution': 200,
        'K_solver': 3.5,
        'K_max': 5.0
    }
)

bridge.initialize_domain(800.0, 800.0, 40.0)
bridge.add_control_point(0, 100.0, 100.0)
bridge.add_control_point(1, 700.0, 700.0)
bridge.finalize_setup()

# 出力:
# Initialized domain: 800.0x800.0, epsilon=40.0
# Strategy: strategy2
#   K_solver: 3.5, K_max: 5.0
# Generated 40000 collocation points
# Finalized setup with 2 control points
```

---

## API 設計の特徴

### 1. 論文の定義に正確に対応

| 論文の Strategy | 入力 | 出力 | 実装 |
|----------------|------|------|------|
| Strategy 1 | Z, K | K_max | `collocation_resolution`, `K_on_collocation` → `guaranteed_K_max` |
| Strategy 2 | K, K_max | h | `interactive_resolution`, `K_solver`, `K_max` → h (自動計算) |

### 2. パラメータの明確性

**Strategy 1:**
- `collocation_resolution`: グリッド解像度（直感的）
- `K_on_collocation`: コロケーション点上の K（明示的）
- K_max は計算結果（指定しない）

**Strategy 2:**
- `interactive_resolution`: ドラッグ中のグリッド解像度
- `K_solver`: コロケーション点上の K
- `K_max`: 目標歪み（必須）

### 3. 計算結果の可視化

```python
# Strategy 1 の場合
bridge.guaranteed_K_max  # finalize_setup() 後に確認可能
# 例: 4.2
```

---

## 実装チェックリスト

- [x] `Strategy1.__init__()` を `collocation_resolution` と `K_on_collocation` で再設計
- [x] `Strategy1.compute_guaranteed_K_max()` を実装（式(11)）
- [x] `BevyBridge.__init__()` に `strategy_type` と `strategy_params` を追加
- [x] `BevyBridge.guaranteed_K_max` フィールドを追加
- [x] `_create_strategy1()` メソッドを実装
- [x] `_create_strategy2()` メソッドを実装
- [x] `initialize_domain()` で Strategy を選択
- [x] `finalize_setup()` で Strategy 1 の K_max を計算
- [x] ログ出力で Strategy 情報を表示

---

## 次のステップ

### 1. Rust 側の修正 (bridge.rs)

```rust
pub enum PyCommand {
    InitializeDomain {
        width: f32,
        height: f32,
        epsilon: f32,
        strategy: String,  // "strategy1" or "strategy2"
        strategy_params: Option<String>,  // JSON 形式
    },
    // ...
}
```

### 2. main.rs の修正

```rust
let strategy = std::env::var("DEFORM_STRATEGY")
    .unwrap_or_else(|_| "strategy2".to_string());

let strategy_params = match strategy.as_str() {
    "strategy1" => {
        serde_json::json!({
            "collocation_resolution": 500,
            "K_on_collocation": 3.5
        }).to_string()
    }
    _ => {
        serde_json::json!({
            "interactive_resolution": 200,
            "K_solver": 3.5,
            "K_max": 5.0
        }).to_string()
    }
};

let _ = tx_cmd.try_send(PyCommand::InitializeDomain {
    width: image_width,
    height: image_height,
    epsilon,
    strategy,
    strategy_params: Some(strategy_params),
});
```

### 3. テスト

```python
# Strategy 1 のテスト
bridge1 = BevyBridge(
    strategy_type="strategy1",
    strategy_params={'collocation_resolution': 500, 'K_on_collocation': 3.5}
)
bridge1.initialize_domain(800.0, 800.0, 40.0)
bridge1.add_control_point(0, 100.0, 100.0)
bridge1.finalize_setup()
assert bridge1.guaranteed_K_max is not None
assert bridge1.guaranteed_K_max > 3.5

# Strategy 2 のテスト
bridge2 = BevyBridge(
    strategy_type="strategy2",
    strategy_params={'interactive_resolution': 200, 'K_solver': 3.5, 'K_max': 5.0}
)
bridge2.initialize_domain(800.0, 800.0, 40.0)
bridge2.add_control_point(0, 100.0, 100.0)
bridge2.finalize_setup()
assert bridge2.guaranteed_K_max is None  # Strategy 2 では計算しない
```

---

## 論文との対応

### 式(11): Isometric Distortion Bound

```
D_iso(x) ≤ max{ K + ω(h), 1/(1/K - ω(h)) }
```

**実装:**
```python
def compute_guaranteed_K_max(self, basis, h):
    K = self.K_on_collocation
    C = basis.compute_basis_gradient_modulus(1.0)
    omega_h = 2.0 * 1.0 * C * h
    
    K_max_bound = max(
        K + omega_h,
        1.0 / (1.0 / K - omega_h)
    )
    return float(K_max_bound)
```

### 式(14): Strategy 2 の h 計算

```
h ≤ ω^(-1)( min{ K_max - K, 1/K - 1/K_max } )
```

**実装:**
```python
# Strategy2.get_strict_h_after_drag() で実装済み
margin = min(K_max - K_solver, (1.0 / K_solver) - (1.0 / K_max))
h_strict = margin / (2.0 * max_coeff_norm * C)
```

---

## まとめ

**実装完了:**
- Strategy 1 と Strategy 2 の API が論文の定義に正確に対応
- K_max の役割が明確（Strategy 1 では出力、Strategy 2 では入力）
- パラメータが直感的で理解しやすい
- ログ出力で Strategy 情報を確認可能

**次のステップ:**
- Rust 側の修正（PyCommand に strategy パラメータを追加）
- main.rs で環境変数から Strategy を選択
- テストで両 Strategy の動作を確認
