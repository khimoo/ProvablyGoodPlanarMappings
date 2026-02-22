# 現在のアプリケーション Strategy 状態

## 現在使用している Strategy

**Strategy 2 (デフォルト)**

---

## 理由

### 1. bevy_bridge.py の初期化

```python
class BevyBridge:
    def __init__(self, strategy_type: str = "strategy2",
                 strategy_params: Optional[Dict] = None):
        self.strategy_type = strategy_type
        self.strategy_params = strategy_params or {}
```

**デフォルト:** `strategy_type = "strategy2"`

### 2. bridge.rs の初期化

```rust
let bridge_instance = bridge_class.call0().expect("Failed to init BevyBridge");
```

**重要:** `call0()` は引数なしで呼び出す
- つまり `BevyBridge()` が実行される
- デフォルトパラメータが使用される
- `strategy_type = "strategy2"` が適用される

### 3. main.rs での初期化

```rust
let _ = tx_cmd.try_send(PyCommand::InitializeDomain {
    width: image_width,
    height: image_height,
    epsilon,
});
```

**重要:** `PyCommand::InitializeDomain` に strategy パラメータがない
- Strategy を指定する仕組みが実装されていない
- Python 側は `initialize_domain(width, height, epsilon)` を呼び出す
- Strategy 2 のデフォルト設定が使用される

---

## 現在の Strategy 2 の設定

```python
# bevy_bridge.py の _create_strategy2() から
interactive_resolution = 200  # デフォルト
K_solver = 3.5                # デフォルト
K_max = 5.0                   # デフォルト

strategy = Strategy2(
    domain_bounds=domain_bounds,
    interactive_resolution=200
)
```

**コロケーションポイント数:**
```
h = max_span / 200
例: 800x800 画像 → h = 4.0 → グリッド点数 = (800/4)^2 = 40,000 点
```

**動作:**
- ドラッグ中: 粗いグリッド (40,000 点) で高速に計算
- ドラッグ終了時: 必要に応じてグリッドを細分化

---

## Strategy 1 に切り替えるには

### 方法 1: bridge.rs を修正（推奨）

```rust
// bridge.rs の python_thread_loop() 内
let bridge: PyObject = Python::with_gil(|py| {
    let sys = py.import("sys").unwrap();
    let current_dir = env::current_dir().unwrap();
    let script_dir = current_dir.join("scripts");
    if let Ok(path) = sys.getattr("path") {
        if let Ok(path_list) = path.downcast::<PyList>() {
            let _ = path_list.insert(0, script_dir);
        }
    }
    let module = PyModule::import(py, "bevy_bridge").expect("Failed to import bevy_bridge");
    let bridge_class = module.getattr("BevyBridge").expect("No BevyBridge class");
    
    // Strategy 1 を使用する場合
    let strategy_params = pyo3::types::PyDict::new_bound(py);
    strategy_params.set_item("collocation_resolution", 500).unwrap();
    strategy_params.set_item("K_on_collocation", 3.5).unwrap();
    
    let bridge_instance = bridge_class.call(
        ("strategy1", strategy_params),
        None
    ).expect("Failed to init BevyBridge");
    bridge_instance.into()
});
```

### 方法 2: 環境変数を使用

```rust
// main.rs の setup() 関数内
let strategy = std::env::var("DEFORM_STRATEGY")
    .unwrap_or_else(|_| "strategy2".to_string());

// PyCommand に strategy パラメータを追加
let _ = tx_cmd.try_send(PyCommand::InitializeDomain {
    width: image_width,
    height: image_height,
    epsilon,
    strategy: strategy.clone(),
});
```

### 方法 3: コンパイル時定数

```rust
// main.rs の先頭
const DEFORM_STRATEGY: &str = "strategy1";  // または "strategy2"

// setup() 内
let _ = tx_cmd.try_send(PyCommand::InitializeDomain {
    width: image_width,
    height: image_height,
    epsilon,
    strategy: DEFORM_STRATEGY.to_string(),
});
```

---

## 現在の実装の問題点

### 1. Strategy を指定できない

```rust
// 現在: Strategy パラメータがない
PyCommand::InitializeDomain { width, height, epsilon }

// 必要: Strategy パラメータを追加
PyCommand::InitializeDomain { width, height, epsilon, strategy, strategy_params }
```

### 2. Python 側で Strategy を選択できない

```python
# 現在: bridge_class.call0() で引数なし
bridge_instance = bridge_class.call0()

# 必要: Strategy パラメータを渡す
bridge_instance = bridge_class.call(("strategy1", params), None)
```

### 3. ドキュメントが古い

```rust
// main.rs のコメント
// - onMouseUp: Python verifies distortion bounds (end_drag + Strategy 2)
// ↑ Strategy 2 に固定されている
```

---

## 推奨される次のステップ

### Phase 1: PyCommand を拡張

```rust
// src/python/commands.rs
pub enum PyCommand {
    InitializeDomain {
        width: f32,
        height: f32,
        epsilon: f32,
        strategy: String,           // ← 追加
        strategy_params: String,    // ← 追加 (JSON)
    },
    // ...
}
```

### Phase 2: bridge.rs を修正

```rust
// src/python/bridge.rs
PyCommand::InitializeDomain {
    width,
    height,
    epsilon,
    strategy,
    strategy_params,
} => {
    // strategy_params を Python 辞書に変換
    // BevyBridge(strategy, params) で初期化
}
```

### Phase 3: main.rs を修正

```rust
// src/main.rs
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
    strategy_params,
});
```

### Phase 4: テスト

```bash
# Strategy 2 (デフォルト)
cargo run

# Strategy 1
DEFORM_STRATEGY=strategy1 cargo run
```

---

## 現在の Strategy 2 の特性

| 項目 | 値 |
|------|-----|
| グリッド解像度 | 200x200 |
| グリッド間隔 (h) | 4.0 (800x800 画像の場合) |
| コロケーションポイント数 | 40,000 点 |
| K_solver | 3.5 |
| K_max | 5.0 |
| ドラッグ中の速度 | 高速 |
| ドラッグ終了時の動作 | グリッド細分化の可能性あり |

---

## まとめ

**現在:** Strategy 2 (デフォルト)
- ドラッグ中は粗いグリッド (40,000 点) で高速に計算
- ドラッグ終了時に必要に応じてグリッドを細分化

**Strategy 1 に切り替えるには:**
- PyCommand に strategy パラメータを追加
- bridge.rs で Strategy を選択
- main.rs で環境変数から Strategy を指定

**推奨:** Phase 1-4 の修正を実装して、両 Strategy を選択可能にする
