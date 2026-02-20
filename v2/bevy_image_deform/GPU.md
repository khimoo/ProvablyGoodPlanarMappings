# Rust/Wgpu GPU実装計画書
## 逆写像計算のGPU化

**目的**: Python側のNewton-Raphson反復計算をWgpuコンピュートシェーダーに移行し、GPU並列処理で高速化

---

## 1. 現状分析

### 現在のボトルネック
- **Python側**: `_compute_inverse_grid()` で64x64=4096点のNewton-Raphson反復
  - 各点の計算は完全に独立（並列化可能）
  - 毎反復でヤコビアン計算・2x2行列式・逆行列計算が発生
  - CPU逐次実行のため、複雑な変形では数秒かかる可能性

### 現在のデータフロー
```
Python (CPU)
  ↓ (Newton-Raphson反復)
  ↓ (4096点 × 10反復)
  ↓ (逆写像グリッド生成)
Rust (CPU)
  ↓ (テクスチャ作成)
  ↓ (GPU転送)
GPU (シェーダー)
  ↓ (テクスチャサンプリング)
  ↓ (画像レンダリング)
```

### 改善後のデータフロー
```
Python (CPU)
  ↓ (マッピング係数のみ計算)
Rust (CPU)
  ↓ (係数をGPUに転送)
GPU (コンピュートシェーダー)
  ↓ (Newton-Raphson反復 × 4096点 並列)
  ↓ (逆写像グリッド生成)
  ↓ (フラグメントシェーダー)
  ↓ (画像レンダリング)
```

---

## 2. 実装アーキテクチャ

### 2.1 新規コンポーネント

#### A. コンピュートシェーダー (`inverse_mapping.wgsl`)
**責務**: 逆写像計算（Newton-Raphson反復）

```wgsl
// 入力バッファ
@group(0) @binding(0) var<storage, read> coefficients: array<vec4<f32>>;  // RBF係数
@group(0) @binding(1) var<storage, read> centers: array<vec2<f32>>;       // RBF中心
@group(0) @binding(2) var<uniform> params: InverseMappingParams;           // s, K_solver等

// 出力バッファ
@group(0) @binding(3) var<storage, read_write> inverse_grid: array<vec2<f32>>;

// 計算ロジック
@compute @workgroup_size(16, 16)
fn compute_inverse_mapping(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // 各スレッドが1ピクセルの逆写像を計算
    // Newton-Raphson反復をローカルで実行
}
```

**計算内容**:
- 各スレッド = 1グリッド点の逆写像計算
- ローカルメモリで Newton-Raphson 反復（10回程度）
- 最終的な源座標をバッファに書き込み

#### B. Rust側の計算エンジン (`gpu_inverse_mapping.rs`)
**責務**: GPU計算の管理・調整

```rust
pub struct GpuInverseMappingEngine {
    compute_pipeline: ComputePipeline,
    bind_group: BindGroup,
    coefficients_buffer: Buffer,
    centers_buffer: Buffer,
    params_buffer: Buffer,
    inverse_grid_buffer: Buffer,
    grid_width: u32,
    grid_height: u32,
}

impl GpuInverseMappingEngine {
    pub fn new(device: &Device, queue: &Queue, ...) -> Self { ... }
    
    pub fn update_coefficients(&mut self, coefficients: &[[f32; 3]]) { ... }
    pub fn update_centers(&mut self, centers: &[[f32; 2]]) { ... }
    pub fn compute(&mut self, queue: &Queue) { ... }
    pub fn read_inverse_grid(&self, device: &Device, queue: &Queue) -> Vec<[f32; 2]> { ... }
}
```

#### C. Bevy統合レイヤー (`gpu_compute_system.rs`)
**責務**: Bevy ECS との統合

```rust
pub fn gpu_compute_inverse_mapping(
    mut gpu_engine: ResMut<GpuInverseMappingEngine>,
    mapping_params: Res<MappingParameters>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if !mapping_params.is_valid {
        return;
    }
    
    // 係数をGPUに転送
    gpu_engine.update_coefficients(&mapping_params.coefficients);
    gpu_engine.update_centers(&mapping_params.centers);
    
    // GPU計算実行
    gpu_engine.compute(&render_queue);
    
    // 結果を取得してマテリアルに反映
    let inverse_grid = gpu_engine.read_inverse_grid(...);
    // → 既存の render_deformed_image() に統合
}
```

---

## 3. 実装ステップ

### Phase 1: 基盤構築（1-2日）

**1.1 コンピュートシェーダー骨組み**
- `assets/shaders/inverse_mapping.wgsl` 作成
- 基本的なバッファレイアウト定義
- ワークグループサイズ決定（16x16推奨）

**1.2 Rust側の基本構造**
- `src/rendering/gpu_compute.rs` 作成
- `GpuInverseMappingEngine` の基本実装
- パイプライン・バインドグループ作成

**1.3 テスト用ダミー実装**
- 恒等写像（逆写像 = 入力座標）で動作確認
- GPU↔CPU間のデータ転送が正常か検証

---

### Phase 2: Newton-Raphson実装（2-3日）

**2.1 シェーダー側の計算ロジック**
- RBF基底関数の評価 (`evaluate_phi`)
- ヤコビアン計算 (`compute_jacobian`)
- 2x2行列式・逆行列 (`invert_2x2`)
- Newton-Raphson反復ループ

**2.2 数値安定性対策**
- 特異ヤコビアン対策（行列式チェック）
- 反復収束判定（許容誤差）
- ダンピング係数の適用

**2.3 パラメータ転送**
- RBF係数 (2 × N_rbf)
- RBF中心 (N_rbf × 2)
- ガウス幅 s
- 反復パラメータ（最大反復数、許容誤差）

---

### Phase 3: 最適化・統合（1-2日）

**3.1 パフォーマンス最適化**
- ローカルメモリ使用量の最小化
- バンク競合回避
- ワークグループサイズ調整（プロファイリング）

**3.2 Bevy ECS統合**
- `gpu_compute_system` を Update ステージに追加
- Python結果受信 → GPU計算 → レンダリング のパイプライン構築
- 既存の `render_deformed_image()` との統合

**3.3 フォールバック機構**
- GPU計算失敗時は Python 側にフォールバック
- デバイス非対応時の処理

---

### Phase 4: 検証・デバッグ（1-2日）

**4.1 正確性検証**
- Python版との結果比較（許容誤差内か）
- 複数の変形パターンでテスト

**4.2 パフォーマンス測定**
- Python版との実行時間比較
- GPU使用率・メモリ使用量の監視

**4.3 エッジケース対応**
- 大規模グリッド（1024x1024）での動作
- 複雑な変形（高歪み）での収束性

---

## 4. 技術詳細

### 4.1 コンピュートシェーダー設計

#### バッファレイアウト
```rust
// Rust側
#[repr(C)]
pub struct InverseMappingParams {
    pub grid_width: u32,
    pub grid_height: u32,
    pub n_rbf: u32,
    pub s_param: f32,
    pub max_iterations: u32,
    pub tolerance: f32,
    pub damping: f32,
    pub _padding: u32,
}

// WGSL側
struct InverseMappingParams {
    grid_width: u32,
    grid_height: u32,
    n_rbf: u32,
    s_param: f32,
    max_iterations: u32,
    tolerance: f32,
    damping: f32,
}
```

#### ワークグループ構成
```wgsl
@compute @workgroup_size(16, 16, 1)
fn compute_inverse_mapping(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let grid_x = global_id.x;
    let grid_y = global_id.y;
    
    if (grid_x >= params.grid_width || grid_y >= params.grid_height) {
        return;
    }
    
    // 出力座標 (grid_x, grid_y) に対応する源座標を計算
    let output_pos = vec2<f32>(f32(grid_x), f32(grid_y));
    
    // Newton-Raphson反復
    var source_pos = output_pos;  // 初期値: 恒等写像
    
    for (var iter = 0u; iter < params.max_iterations; iter++) {
        // f(source_pos) を計算
        let mapped = evaluate_forward_map(source_pos);
        
        // 残差
        let residual = mapped - output_pos;
        
        // 収束判定
        if (length(residual) < params.tolerance) {
            break;
        }
        
        // ヤコビアン計算
        let jacobian = compute_jacobian(source_pos);
        
        // 逆行列 (2x2)
        let inv_jac = invert_2x2(jacobian);
        
        // Newton-Raphson更新
        let delta = -inv_jac * residual;
        source_pos += params.damping * delta;
    }
    
    // 結果を書き込み
    let idx = grid_y * params.grid_width + grid_x;
    inverse_grid[idx] = source_pos;
}
```

### 4.2 RBF基底関数の実装

```wgsl
fn evaluate_phi(x: vec2<f32>) -> vec4<f32> {
    // ガウス基底 + アフィン項
    // 戻り値: [φ_1(x), φ_2(x), ..., φ_N(x), 1, x, y]
    
    var result = vec4<f32>(0.0);
    
    // RBF項（最初のN個）
    for (var i = 0u; i < params.n_rbf; i++) {
        let center = centers[i];
        let diff = x - center;
        let r2 = dot(diff, diff);
        let phi_i = exp(-r2 / (2.0 * params.s_param * params.s_param));
        // result[i] = phi_i;  // 実際にはより大きい配列が必要
    }
    
    // アフィン項
    // result[n_rbf] = 1.0;
    // result[n_rbf+1] = x.x;
    // result[n_rbf+2] = x.y;
    
    return result;
}

fn compute_jacobian(x: vec2<f32>) -> mat2x2<f32> {
    // J_f(x) = c * ∇Φ(x)
    // ∇Φ(x) は (N+3) × 2 行列
    
    var jac = mat2x2<f32>(0.0);
    
    // RBF項の勾配
    for (var i = 0u; i < params.n_rbf; i++) {
        let center = centers[i];
        let diff = x - center;
        let r2 = dot(diff, diff);
        let phi_i = exp(-r2 / (2.0 * params.s_param * params.s_param));
        let grad_phi_i = -(1.0 / (params.s_param * params.s_param)) * diff * phi_i;
        
        // jac += c[i] * grad_phi_i;  // 係数を掛ける
    }
    
    // アフィン項の勾配（定数）
    // jac[0][0] += c[n_rbf+1];  // ∂(x)/∂x = 1
    // jac[1][1] += c[n_rbf+2];  // ∂(y)/∂y = 1
    
    return jac;
}

fn invert_2x2(m: mat2x2<f32>) -> mat2x2<f32> {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let det_safe = select(det, 1e-8, abs(det) < 1e-8);
    
    return mat2x2<f32>(
        vec2<f32>(m[1][1] / det_safe, -m[0][1] / det_safe),
        vec2<f32>(-m[1][0] / det_safe, m[0][0] / det_safe)
    );
}
```

### 4.3 Rust側の統合

```rust
// src/rendering/gpu_compute.rs

use bevy::render::{
    render_resource::*,
    renderer::{RenderContext, RenderDevice, RenderQueue},
};

pub struct GpuInverseMappingEngine {
    compute_pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    bind_group: Option<BindGroup>,
    
    coefficients_buffer: Buffer,
    centers_buffer: Buffer,
    params_buffer: Buffer,
    inverse_grid_buffer: Buffer,
    
    grid_width: u32,
    grid_height: u32,
}

impl GpuInverseMappingEngine {
    pub fn new(
        device: &RenderDevice,
        grid_width: u32,
        grid_height: u32,
    ) -> Self {
        // パイプライン作成
        let shader = device.load_shader("shaders/inverse_mapping.wgsl");
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("inverse_mapping_pipeline"),
            layout: vec![bind_group_layout.clone()],
            push_constant_ranges: vec![],
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: "compute_inverse_mapping".into(),
        });
        
        // バッファ作成
        let coefficients_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("coefficients_buffer"),
            contents: &[],
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        
        // ... 他のバッファも同様
        
        Self {
            compute_pipeline: pipeline,
            bind_group_layout,
            bind_group: None,
            coefficients_buffer,
            centers_buffer,
            params_buffer,
            inverse_grid_buffer,
            grid_width,
            grid_height,
        }
    }
    
    pub fn update_coefficients(&mut self, device: &RenderDevice, queue: &RenderQueue, coefficients: &[f32]) {
        queue.write_buffer(&self.coefficients_buffer, 0, bytemuck::cast_slice(coefficients));
    }
    
    pub fn compute(&self, render_context: &mut RenderContext) {
        let mut pass = render_context.begin_compute_pass(&ComputePassDescriptor {
            label: Some("inverse_mapping_pass"),
        });
        
        pass.set_pipeline(&self.compute_pipeline);
        pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        
        // ワークグループ数を計算
        let workgroup_x = (self.grid_width + 15) / 16;
        let workgroup_y = (self.grid_height + 15) / 16;
        
        pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
    }
}
```

---

## 5. 既存コードとの統合ポイント

### 5.1 Python側の変更（最小限）
- `_compute_inverse_grid()` は廃止
- 代わりに `compute_strict_h()` の結果をRust側に送信
- マッピング係数のみ計算・転送

### 5.2 Rust側の変更
```rust
// src/main.rs
.add_systems(Update, (
    handle_input,
    receive_python_results,
    gpu_compute_inverse_mapping,  // ← 新規追加
    render_deformed_image,
))
```

### 5.3 既存シェーダーの変更（なし）
- `deform.wgsl` は変更不要
- 逆写像グリッドの生成方法が変わるだけ

---

## 6. パフォーマンス予測

### 現在（Python CPU）
- 64x64グリッド: ~100-500ms（変形複雑度による）
- ボトルネック: Newton-Raphson反復の逐次実行

### 改善後（GPU）
- 64x64グリッド: ~5-20ms（GPU転送含む）
- 256x256グリッド: ~20-80ms（スケーラビリティ向上）
- 1024x1024グリッド: ~100-300ms（フル解像度対応可能）

**期待される高速化**: 10-50倍

---

## 7. リスク・対策

| リスク | 対策 |
|--------|------|
| GPU非対応環境 | フォールバック機構（Python側に戻す） |
| 数値精度問題 | 倍精度浮動小数点の検討、許容誤差調整 |
| メモリ不足 | ストリーミング処理（グリッドを分割） |
| シェーダーコンパイル失敗 | 段階的な実装、テスト用ダミー版から開始 |

---

## 8. 成功基準

- ✅ GPU計算結果がPython版と許容誤差内で一致
- ✅ 64x64グリッドで10倍以上の高速化
- ✅ 256x256グリッドでも安定動作
- ✅ フォールバック機構が正常に動作
- ✅ 既存のレンダリングパイプラインに統合完了

---

## 9. スケジュール

| Phase | 期間 | 成果物 |
|-------|------|--------|
| 1 | 1-2日 | 基本シェーダー・Rust基盤 |
| 2 | 2-3日 | Newton-Raphson実装 |
| 3 | 1-2日 | 最適化・統合 |
| 4 | 1-2日 | 検証・デバッグ |
| **合計** | **5-9日** | **本番対応版** |


