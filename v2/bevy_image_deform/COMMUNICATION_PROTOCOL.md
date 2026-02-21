# Bevy Image Deformer - 通信プロトコル仕様書

## 概要

Bevy（Rust）フロントエンドと Python バックエンド間の通信は、**非同期メッセージパッシング**で実現されています。

- **送信**: `tokio::sync::mpsc::Sender<PyCommand>` (Rust → Python)
- **受信**: `crossbeam_channel::Receiver<PyResult>` (Python → Rust)

---

## コマンド定義（Rust → Python）

### 1. InitializeDomain

**目的**: 変形領域を初期化

```rust
PyCommand::InitializeDomain {
    width: f32,      // 画像幅（ピクセル）
    height: f32,     // 画像高さ（ピクセル）
    epsilon: f32,    // ガウス RBF の幅パラメータ
}
```

**Python 処理**:
```python
def initialize_domain(self, image_width: float, image_height: float, epsilon: float) -> None:
    self.image_width = image_width
    self.image_height = image_height
    
    domain_bounds = (0.0, image_width, 0.0, image_height)
    self.solver = BetterFitwithGaussian(
        domain_bounds=domain_bounds,
        s_param=epsilon,
        K_solver=3.0,
        K_max=5.0
    )
```

**戻り値**: `PyResult::DomainInitialized`

**タイミング**: アプリケーション起動時（1回のみ）

---

### 2. SetContour

**目的**: 変形領域の境界を設定（オプション）

```rust
PyCommand::SetContour {
    contour: Vec<(f32, f32)>,  // [(x1, y1), (x2, y2), ...]
}
```

**Python 処理**:
```python
def set_contour(self, contour_points: List[Tuple[float, float]]) -> None:
    self.contour = np.array(contour_points, dtype=np.float64)
    # 後で finalize_setup() で使用
```

**戻り値**: なし（非同期）

**タイミング**: `InitializeDomain` の直後

**用途**:
- 画像の輪郭を抽出して設定
- 変形を輪郭内に制限
- 背景領域への変形を防止

---

### 3. AddControlPoint

**目的**: 制御点を追加（セットアップフェーズ）

```rust
PyCommand::AddControlPoint {
    index: usize,    // 制御点の識別番号（0, 1, 2, ...）
    x: f32,          // 画像座標系での X 座標
    y: f32,          // 画像座標系での Y 座標
}
```

**Python 処理**:
```python
def add_control_point(self, control_index: int, x: float, y: float) -> None:
    if self.is_setup_finalized:
        raise RuntimeError("Cannot add control points after finalize_setup")
    
    self.control_points.append((control_index, x, y))
```

**戻り値**: なし（非同期）

**タイミング**: セットアップフェーズ中（複数回）

**制約**:
- `finalize_setup()` 前のみ有効
- 輪郭が設定されている場合、輪郭内のみ有効
- 既存の制御点から 20 ピクセル以上離れている必要がある

---

### 4. FinalizeSetup

**目的**: セットアップを確定し、ソルバーを初期化

```rust
PyCommand::FinalizeSetup
```

**Python 処理**:
```python
def finalize_setup(self) -> None:
    if self.is_setup_finalized:
        return
    
    # 制御点から RBF 中心を設定
    src_handles = np.array(
        [[x, y] for (_, x, y) in self.control_points],
        dtype=np.float64
    )
    
    # ソルバーを初期化
    self.solver.initialize_mapping(src_handles)
    
    # 輪郭が設定されている場合、選点をフィルタリング
    if self.contour is not None:
        self.solver.collocation_points = filter_points_inside_contour(
            self.solver.collocation_points,
            self.contour
        )
        self.solver._update_hessian_term()
    
    self.is_setup_finalized = True
```

**戻り値**: `PyResult::SetupFinalized`

**タイミング**: ユーザーが Enter キーを押したとき（1回のみ）

**重要な計算**:
- RBF 基底関数の中心を制御点に配置
- 初期グリッド（選点）を生成（解像度: 200×200）
- 事前計算行列 `B_mat`, `H_term` を計算

---

### 5. StartDrag

**目的**: ドラッグ操作を開始（マウスボタン押下）

```rust
PyCommand::StartDrag
```

**Python 処理**:
```python
def start_drag_operation(self) -> None:
    if not self.is_setup_finalized:
        raise RuntimeError("Setup not finalized")
    
    src_handles = np.array(
        [[x, y] for (_, x, y) in self.control_points],
        dtype=np.float64
    )
    
    self.solver.start_drag(src_handles)
    # [事前計算用の準備]
```

**戻り値**: なし（非同期）

**タイミング**: ユーザーがマウスボタンを押したとき

**目的**:
- ドラッグ中の高速応答のため、重い計算をここで実施
- Active Set の初期化

---

### 6. UpdatePoint

**目的**: 制御点の位置を更新（ドラッグ中、毎フレーム）

```rust
PyCommand::UpdatePoint {
    control_index: usize,  // 更新する制御点のインデックス
    x: f32,                // 新しい X 座標
    y: f32,                // 新しい Y 座標
}
```

**Python 処理**:
```python
def update_control_point(self, control_index: int, x: float, y: float) -> None:
    if not self.is_setup_finalized:
        raise RuntimeError("Setup not finalized")
    
    # 位置を更新
    idx, _, _ = self.control_points[control_index]
    self.control_points[control_index] = (idx, x, y)

def solve_frame(self, inverse_grid_resolution: int = 64) -> Dict:
    # 目標ハンドル位置を取得
    target_handles = np.array(
        [[x, y] for (_, x, y) in self.control_points],
        dtype=np.float64
    )
    
    # 最適化を実行
    self.solver.update_drag(target_handles, num_iterations=2)
    
    # 逆マッピングを計算
    inverse_grid = self._compute_inverse_grid(inverse_grid_resolution)
    
    # 結果を返す
    return {
        'coefficients': self.solver.coefficients.tolist(),
        'centers': self.solver.basis.centers.tolist(),
        's_param': float(self.solver.basis.s),
        'n_rbf': len(self.control_points),
        'image_width': float(self.image_width),
        'image_height': float(self.image_height),
        'inverse_grid': inverse_grid.tolist(),
        'grid_width': inverse_grid.shape[1],
        'grid_height': inverse_grid.shape[0],
    }
```

**戻り値**: `PyResult::MappingParameters { ... }`

**タイミング**: ドラッグ中、毎フレーム（60 FPS 想定）

**処理内容**:
1. 制御点の位置を更新
2. Local-Global Solver で最適化（2 イテレーション）
3. 逆マッピングを計算
4. 結果を Rust に返す

---

### 7. EndDrag

**目的**: ドラッグ操作を終了（マウスボタン解放）

```rust
PyCommand::EndDrag
```

**Python 処理**:
```python
def end_drag_operation(self) -> bool:
    if not self.is_setup_finalized:
        raise RuntimeError("Setup not finalized")
    
    target_handles = np.array(
        [[x, y] for (_, x, y) in self.control_points],
        dtype=np.float64
    )
    
    # Strategy 2 検証
    was_refined = self.solver.end_drag(target_handles)
    
    if was_refined:
        # グリッドが細かくなった場合、新しいグリッドで再最適化
        if self.contour is not None:
            self.solver.collocation_points = filter_points_inside_contour(
                self.solver.collocation_points,
                self.contour
            )
            self.solver._update_hessian_term()
    
    return was_refined
```

**戻り値**: `PyResult::MappingParameters { ... }`

**タイミング**: ユーザーがマウスボタンを離したとき

**重要な処理**:
- **Strategy 2 検証**: 論文の定理に基づき、歪みの上界を数学的に保証
- **適応的グリッド細分化**: 必要に応じてグリッドを自動的に細かくする

---

### 8. Reset

**目的**: 全ての状態をリセット

```rust
PyCommand::Reset
```

**Python 処理**:
```python
def reset_mesh(self) -> None:
    self.control_points.clear()
    self.is_setup_finalized = False
    self.solver = None
    self.contour = None
```

**戻り値**: なし（非同期）

**タイミング**: ユーザーが R キーを押したとき

---

## 結果定義（Python → Rust）

### 1. DomainInitialized

```rust
PyResult::DomainInitialized
```

**送信タイミング**: `InitializeDomain` コマンド処理完了後

**Rust 処理**:
```rust
PyResult::DomainInitialized => {
    println!("Domain initialized");
}
```

---

### 2. SetupFinalized

```rust
PyResult::SetupFinalized
```

**送信タイミング**: `FinalizeSetup` コマンド処理完了後

**Rust 処理**:
```rust
PyResult::SetupFinalized => {
    println!("Setup finalized, switching to Deform mode");
    if *state.get() == AppMode::Finalizing {
        next_state.set(AppMode::Deform);
    }
}
```

---

### 3. MappingParameters

```rust
PyResult::MappingParameters {
    coefficients: Vec<Vec<f32>>,      // (2, N_basis)
    centers: Vec<Vec<f32>>,           // (N_ctrl, 2)
    s_param: f32,                     // ガウス基底の幅
    n_rbf: usize,                     // RBF 基底関数の数
    image_width: f32,                 // 画像幅
    image_height: f32,                // 画像高さ
    inverse_grid: Vec<Vec<Vec<f32>>>, // (H, W, 2)
    grid_width: usize,                // グリッド幅
    grid_height: usize,               // グリッド高さ
}
```

**送信タイミング**: `UpdatePoint` または `EndDrag` コマンド処理完了後

**データ詳細**:

#### coefficients (2 × N_basis)

変形マッピング f(x) = c · Φ(x) の係数行列

```
c = [
    [c0_0, c0_1, ..., c0_{N-1}, c0_N, c0_{N+1}, c0_{N+2}],  // x 成分
    [c1_0, c1_1, ..., c1_{N-1}, c1_N, c1_{N+1}, c1_{N+2}],  // y 成分
]

ここで:
- c_i_0, ..., c_i_{N-1}: RBF 係数（N = 制御点数）
- c_i_N: 定数項
- c_i_{N+1}: x の線形係数
- c_i_{N+2}: y の線形係数
```

#### centers (N_ctrl × 2)

RBF 基底関数の中心位置（制御点と同じ）

```
centers = [
    [x1, y1],
    [x2, y2],
    ...
    [xN, yN],
]
```

#### s_param

ガウス基底関数の幅パラメータ σ

```
φ_i(x) = exp(-||x - c_i||^2 / (2σ^2))
```

#### inverse_grid (H × W × 2)

逆マッピング f^{-1}(y)

```
inverse_grid[y][x] = [src_x, src_y]

意味: 出力画像のピクセル (x, y) に対して、
      元画像のどこから色をサンプルするか
```

**Rust 処理**:
```rust
PyResult::MappingParameters { ... } => {
    println!("Received mapping parameters: {} RBFs, inverse grid {}x{}",
             n_rbf, grid_width, grid_height);
    
    mapping_params.coefficients = coefficients;
    mapping_params.centers = centers;
    mapping_params.s_param = s_param;
    mapping_params.n_rbf = n_rbf;
    mapping_params.image_width = image_width;
    mapping_params.image_height = image_height;
    mapping_params.inverse_grid = inverse_grid;
    mapping_params.grid_width = grid_width;
    mapping_params.grid_height = grid_height;
    mapping_params.is_valid = true;
}
```

---

## 逆マッピング計算アルゴリズム

### 目的

出力画像の各ピクセル (x, y) に対して、元画像のどこから色をサンプルするかを計算

### 方法

Newton-Raphson 法で f(src) = (x, y) を満たす src を求める

```python
def _compute_inverse_grid(self, resolution: int) -> np.ndarray:
    # 出力グリッドを作成
    y_coords = np.linspace(0, self.image_height, resolution)
    x_coords = np.linspace(0, self.image_width, resolution)
    Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    # 目標位置（出力空間）
    target_positions = np.stack([X.ravel(), Y.ravel()], axis=1)
    
    # 初期推定値（恒等写像）
    source_positions = target_positions.copy()
    
    # Newton-Raphson イテレーション
    for iteration in range(max_iterations):
        # 1. 順方向マッピングを評価
        mapped = self.solver.evaluate_map(source_positions)
        
        # 2. 残差を計算
        residual = mapped - target_positions
        
        # 3. 収束判定
        if np.max(np.abs(residual)) < tolerance:
            break
        
        # 4. ヤコビアンを計算
        jacobians = self.solver.evaluate_jacobian(source_positions)
        
        # 5. ニュートン更新
        # J * delta = -residual
        # delta = -J^{-1} * residual
        inv_jac = np.linalg.inv(jacobians)
        delta = -np.einsum('nij,nj->ni', inv_jac, residual)
        
        # 6. ダンピング付きで更新
        source_positions += damping * delta
    
    return source_positions.reshape(resolution, resolution, 2)
```

### パラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `max_iterations` | 10 | 最大イテレーション数 |
| `tolerance` | 1e-3 | 収束判定の閾値 |
| `damping` | 0.8 | ステップサイズの減衰係数 |

---

## メッセージフロー図

### セットアップフェーズ

```
Rust                          Python
  │                             │
  ├─ InitializeDomain ────────→ │
  │                             ├─ initialize_domain()
  │                             │
  ├─ SetContour ──────────────→ │
  │                             ├─ set_contour()
  │                             │
  ├─ AddControlPoint ─────────→ │
  │                             ├─ add_control_point()
  │                             │
  ├─ AddControlPoint ─────────→ │
  │                             ├─ add_control_point()
  │                             │
  ├─ ... (複数回) ────────────→ │
  │                             │
  ├─ FinalizeSetup ───────────→ │
  │                             ├─ finalize_setup()
  │                             ├─ initialize_mapping()
  │                             ├─ filter_collocation_points()
  │                             │
  │ ← SetupFinalized ──────────┤
  │                             │
```

### ドラッグフェーズ

```
Rust                          Python
  │                             │
  ├─ StartDrag ───────────────→ │
  │                             ├─ start_drag_operation()
  │                             │
  ├─ UpdatePoint ─────────────→ │
  │                             ├─ update_control_point()
  │                             ├─ solve_frame()
  │                             ├─ _compute_inverse_grid()
  │                             │
  │ ← MappingParameters ───────┤
  │                             │
  ├─ UpdatePoint ─────────────→ │
  │                             ├─ update_control_point()
  │                             ├─ solve_frame()
  │                             │
  │ ← MappingParameters ───────┤
  │                             │
  ├─ ... (毎フレーム) ────────→ │
  │                             │
  ├─ EndDrag ─────────────────→ │
  │                             ├─ end_drag_operation()
  │                             ├─ Strategy 2 検証
  │                             ├─ グリッド細分化（必要に応じて）
  │                             │
  │ ← MappingParameters ───────┤
  │                             │
```

---

## エラーハンドリング

### Python 側

```python
try:
    # 処理
except RuntimeError as e:
    print(f"Py Error: {e}")
    # Rust には結果を返さない
except ValueError as e:
    print(f"Py Error: {e}")
    # Rust には結果を返さない
```

### Rust 側

```rust
// コマンド送信
if let Err(e) = channels.tx_command.try_send(cmd) {
    eprintln!("Failed to send command: {}", e);
}

// 結果受信
while let Ok(res) = channels.rx_result.try_recv() {
    match res {
        PyResult::MappingParameters { ... } => { /* 処理 */ }
        _ => {}
    }
}
```

---

## パフォーマンス特性

### 計算時間（目安）

| 処理 | 時間 | 備考 |
|-----|------|------|
| `InitializeDomain` | < 1 ms | 初期化のみ |
| `FinalizeSetup` | 10-100 ms | グリッド生成、行列計算 |
| `UpdatePoint` (ドラッグ中) | 5-20 ms | 2 イテレーション、逆マッピング計算 |
| `EndDrag` | 50-500 ms | Strategy 2 検証、グリッド細分化（必要に応じて） |

### メモリ使用量

| データ | サイズ | 備考 |
|-------|-------|------|
| `coefficients` | 2 × (N + 3) × 4 bytes | N = 制御点数 |
| `centers` | N × 2 × 8 bytes | |
| `inverse_grid` | W × H × 2 × 4 bytes | W, H = グリッド解像度 |
| `collocation_points` | M × 2 × 8 bytes | M = 選点数（通常 40,000） |

---

## 拡張性

### 新しいコマンドの追加

1. `src/python/commands.rs` に `PyCommand` バリアントを追加
2. `src/python/bridge.rs` の `python_thread_loop()` に処理を追加
3. `scripts/bevy_bridge.py` に対応するメソッドを実装

### 新しい結果型の追加

1. `src/python/commands.rs` に `PyResult` バリアントを追加
2. `src/main.rs` の `receive_python_results()` に処理を追加

---

## まとめ

| 項目 | 説明 |
|-----|------|
| **通信方式** | 非同期メッセージパッシング |
| **送信チャネル** | tokio::sync::mpsc |
| **受信チャネル** | crossbeam_channel |
| **コマンド数** | 8 種類 |
| **結果型** | 3 種類 |
| **座標系** | 画像座標系（左上原点） |
| **主要データ** | 係数、中心、逆マッピング |
