# Strategy 1 への切り替え完了

## 実装内容

### 1. PyCommand の拡張 (commands.rs)

**変更前:**
```rust
pub enum PyCommand {
    InitializeDomain { width: f32, height: f32, epsilon: f32 },
    // ...
}
```

**変更後:**
```rust
pub enum PyCommand {
    InitializeDomain {
        width: f32,
        height: f32,
        epsilon: f32,
        strategy: String,           // "strategy1" or "strategy2"
        strategy_params: String,    // JSON format
    },
    // ...
}
```

### 2. bridge.rs の修正

**追加内容:**
- `serde_json` をインポート
- `InitializeDomain` コマンドで strategy パラメータを処理
- JSON パラメータを Python 辞書に変換
- BevyBridge を再初期化

**重要な処理:**
```rust
// strategy_params を Python 辞書に変換
let params_dict = Python::with_gil(|py| {
    if let Ok(params_json) = serde_json::from_str::<serde_json::Value>(&strategy_params) {
        let dict = pyo3::types::PyDict::new_bound(py);
        for (key, value) in params_json.as_object().unwrap() {
            // JSON値をPython値に変換
            let py_value = match value {
                serde_json::Value::Number(n) => { /* ... */ },
                serde_json::Value::String(s) => s.into_py(py),
                _ => continue,
            };
            let _ = dict.set_item(key, py_value);
        }
        dict.into()
    } else {
        pyo3::types::PyDict::new_bound(py).into()
    }
});

// BevyBridge を再初期化
bridge_bound.call_method1("__init__", (strategy.clone(), params_dict))?;
```

### 3. main.rs の修正

**変更前:**
```rust
let _ = tx_cmd.try_send(PyCommand::InitializeDomain {
    width: image_width,
    height: image_height,
    epsilon,
});
```

**変更後:**
```rust
let strategy = "strategy1".to_string();
let strategy_params = serde_json::json!({
    "collocation_resolution": 500,
    "K_on_collocation": 3.5
}).to_string();

let _ = tx_cmd.try_send(PyCommand::InitializeDomain {
    width: image_width,
    height: image_height,
    epsilon,
    strategy,
    strategy_params,
});
```

**追加:**
- `serde_json` をインポート
- コメントを更新（Strategy 2 → Strategy 1）

---

## 現在の設定

### Strategy 1 パラメータ

```json
{
    "collocation_resolution": 500,
    "K_on_collocation": 3.5
}
```

**意味:**
- `collocation_resolution`: グリッド解像度 500x500
- `K_on_collocation`: コロケーション点上の歪み上限 3.5

**コロケーションポイント数:**
```
h = max_span / 500
例: 800x800 画像 → h = 1.6 → グリッド点数 = (800/1.6)^2 ≈ 250,000 点
```

### 動作の違い

| 項目 | Strategy 2 | Strategy 1 |
|------|-----------|-----------|
| グリッド解像度 | 200x200 | 500x500 |
| グリッド間隔 (h) | 4.0 | 1.6 |
| コロケーションポイント数 | 40,000 点 | 250,000 点 |
| ドラッグ中の速度 | 高速 | 遅い |
| ドラッグ終了時 | グリッド細分化あり | グリッド細分化なし |
| K_max | 計算結果 | 計算結果 |

---

## 実行方法

### ビルド

```bash
cargo build --release
```

### 実行

```bash
cargo run --release
```

### ログ出力

実行時に以下のようなログが出力されます：

```
Rust->Py: InitializeDomain 800x800, eps=40, strategy=strategy1
Strategy: strategy1
  K_on_collocation: 3.5
  (K_max will be computed after finalize_setup)
Generated 250000 collocation points
Strategy 1: Guaranteed K_max = 4.2
```

---

## Strategy 2 に戻すには

main.rs の以下の部分を変更：

```rust
// Strategy 2 を使用する場合
let strategy = "strategy2".to_string();
let strategy_params = serde_json::json!({
    "interactive_resolution": 200,
    "K_solver": 3.5,
    "K_max": 5.0
}).to_string();
```

---

## 修正ファイル一覧

1. **src/python/commands.rs**
   - PyCommand::InitializeDomain に strategy と strategy_params を追加

2. **src/python/bridge.rs**
   - serde_json をインポート
   - InitializeDomain コマンドで strategy パラメータを処理
   - JSON を Python 辞書に変換
   - BevyBridge を再初期化

3. **src/main.rs**
   - serde_json をインポート
   - Strategy 1 パラメータを設定
   - InitializeDomain コマンドに strategy と strategy_params を追加
   - コメントを更新

---

## 技術的な詳細

### JSON から Python 辞書への変換

```rust
// serde_json::Value を Python オブジェクトに変換
let py_value = match value {
    serde_json::Value::Number(n) => {
        if let Some(i) = n.as_i64() {
            i.into_py(py)
        } else if let Some(f) = n.as_f64() {
            f.into_py(py)
        } else {
            continue;
        }
    }
    serde_json::Value::String(s) => s.into_py(py),
    _ => continue,
};
```

### BevyBridge の再初期化

```rust
// Python 側の __init__ メソッドを呼び出して再初期化
bridge_bound.call_method1("__init__", (strategy.clone(), params_dict))?;
```

---

## パフォーマンスへの影響

### 初期化時

- **Strategy 2**: 高速（40,000 点のグリッド生成）
- **Strategy 1**: 遅い（250,000 点のグリッド生成）

### ドラッグ中

- **Strategy 2**: 高速（40,000 点で計算）
- **Strategy 1**: 遅い（250,000 点で計算）

### ドラッグ終了時

- **Strategy 2**: グリッド細分化の可能性あり（追加計算）
- **Strategy 1**: グリッド細分化なし（追加計算なし）

---

## 今後の改善案

### 1. 環境変数で Strategy を選択

```rust
let strategy = std::env::var("DEFORM_STRATEGY")
    .unwrap_or_else(|_| "strategy1".to_string());
```

実行時：
```bash
DEFORM_STRATEGY=strategy2 cargo run
```

### 2. コンパイル時定数で Strategy を選択

```rust
const DEFORM_STRATEGY: &str = "strategy1";
```

### 3. UI で Strategy を選択

ユーザーが実行時に Strategy を選択できるようにする

---

## まとめ

**Strategy 1 への切り替え完了:**
- PyCommand に strategy パラメータを追加
- bridge.rs で JSON パラメータを処理
- main.rs で Strategy 1 を使用するように設定
- コロケーションポイント数が 40,000 → 250,000 に増加
- ドラッグ中の計算が遅くなるが、安全性が向上
- ドラッグ終了時のグリッド細分化がなくなる
