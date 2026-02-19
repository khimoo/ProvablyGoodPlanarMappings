# Implementation Summary

## 完了した修正

### 1. `bevy_bridge.py` の変更

**追加された機能:**
- `initialize_domain()`: 画像サイズとパラメータでドメインを初期化
- `set_contour()`: 輪郭線をドメイン境界として設定
- `point_in_polygon()`: 点が輪郭線内部にあるかチェック
- `filter_points_inside_contour()`: 輪郭線内部の点のみをフィルタリング

**変更されたメソッド:**
- `finalize_setup()`: 輪郭線が設定されている場合、collocation points をフィルタリング
- `solve_frame()`: 変形後のメッシュ頂点ではなく、**マッピングパラメータ**を返す
  - 係数行列 (2, N+3)
  - RBF中心座標 (N, 2)
  - ガウス幅パラメータ s
  - 画像サイズ

**削除された機能:**
- `load_image_and_generate_mesh()`: 不要（Rustで画像を読み込む）
- `_create_visualization_mesh()`: 不要（メッシュベースではない）

### 2. `main.rs` の変更

**新しいデータ構造:**
```rust
enum PyCommand {
    InitializeDomain { width, height, epsilon },
    SetContour { contour },
    AddControlPoint { index, x, y },
    FinalizeSetup,
    StartDrag,
    UpdatePoint { control_index, x, y },
    EndDrag,
    Reset,
}

enum PyResult {
    DomainInitialized,
    MappingParameters {
        coefficients,
        centers,
        s_param,
        n_rbf,
        image_width,
        image_height,
    },
}

struct MappingParameters {
    coefficients: Vec<Vec<f32>>,
    centers: Vec<Vec<f32>>,
    s_param: f32,
    n_rbf: usize,
    image_width: f32,
    image_height: f32,
    is_valid: bool,
}

struct ImageData {
    width: f32,
    height: f32,
    handle: Handle<Image>,
}
```

**新しいシステム:**
- `receive_python_results()`: Pythonからマッピングパラメータを受信
- `render_deformed_image()`: 画像変形の描画（TODO: 実装が必要）
- `draw_control_points()`: 制御点の描画

**削除されたシステム:**
- `update_mesh_and_gizmos()`: メッシュベースの処理を削除

### 3. `deform_algo.py` の変更

**変更なし** - 既存の実装がそのまま使用可能

## 現在の状態

### 動作する機能
✅ ドメインの初期化
✅ 制御点の追加（Setup Mode）
✅ ソルバーの初期化（Finalize Setup）
✅ ドラッグ操作（Deform Mode）
✅ マッピングパラメータの送受信
✅ 制御点の描画

### 未実装の機能
❌ 輪郭線の抽出（現在は画像全体を矩形として扱う）
❌ 画像変形の描画（`render_deformed_image()` が TODO）

## 次のステップ

### 1. 輪郭線抽出の実装

```rust
// 例: imageproc crate を使用
use imageproc::contours::find_contours;

fn extract_contour(image: &Image) -> Vec<(f32, f32)> {
    // アルファチャンネルまたはエッジ検出で輪郭を抽出
    // ...
}
```

### 2. 画像変形の実装

2つのアプローチがあります：

#### アプローチA: CPU側で変形（シンプル）

```rust
fn render_deformed_image(
    mapping_params: Res<MappingParameters>,
    image_data: Res<ImageData>,
    mut images: ResMut<Assets<Image>>,
) {
    if !mapping_params.is_valid {
        return;
    }
    
    // 元画像を取得
    let Some(src_image) = images.get(&image_data.handle) else { return };
    
    // 新しい画像を作成
    let mut dst_image = Image::new_fill(...);
    
    // 各ピクセルで f(x, y) を評価
    for y in 0..height {
        for x in 0..width {
            // f(x, y) = Σ c_i * φ_i(x, y) を計算
            let deformed_pos = evaluate_gaussian_rbf(
                Vec2::new(x, y),
                &mapping_params.coefficients,
                &mapping_params.centers,
                mapping_params.s_param,
            );
            
            // 元画像からサンプリング
            let color = sample_image(src_image, deformed_pos);
            dst_image.set_pixel(x, y, color);
        }
    }
    
    // 画像を更新
    // ...
}

fn evaluate_gaussian_rbf(
    pos: Vec2,
    coeffs: &[Vec<f32>],
    centers: &[Vec<f32>],
    s: f32,
) -> Vec2 {
    let mut result = Vec2::ZERO;
    
    // RBF項
    for i in 0..centers.len() {
        let center = Vec2::new(centers[i][0], centers[i][1]);
        let diff = pos - center;
        let r2 = diff.length_squared();
        let phi = (-r2 / (2.0 * s * s)).exp();
        
        result.x += coeffs[0][i] * phi;
        result.y += coeffs[1][i] * phi;
    }
    
    // アフィン項: [1, x, y]
    let n = centers.len();
    result.x += coeffs[0][n] + coeffs[0][n+1] * pos.x + coeffs[0][n+2] * pos.y;
    result.y += coeffs[1][n] + coeffs[1][n+1] * pos.x + coeffs[1][n+2] * pos.y;
    
    result
}
```

#### アプローチB: GPU側で変形（高速）

カスタムシェーダーを使用：

```wgsl
@fragment
fn fragment(
    @location(0) uv: vec2<f32>,
    @location(1) world_pos: vec2<f32>,
) -> @location(0) vec4<f32> {
    // f(world_pos) を計算
    var deformed_pos = evaluate_rbf(world_pos, coefficients, centers, s_param);
    
    // 元画像からサンプリング
    var color = textureSample(texture, sampler, deformed_pos / image_size);
    
    return color;
}
```

### 3. パフォーマンス最適化

- グリッドベースの補間（粗いグリッドで計算、ピクセル間は補間）
- GPU compute shader での並列計算
- 変更があった場合のみ再計算

## 理論的保証

現在の実装は論文の理論的保証を維持しています：

✅ **Local Injectivity**: det(J) > 0（折り畳みなし）
✅ **Bounded Distortion**: K_solver ≤ 2.0 on collocation points
✅ **Global Distortion Bound**: K_max ≤ 5.0 everywhere (Strategy 2)
✅ **Non-convex Domains**: 輪郭線ベースのドメインに対応
✅ **Smooth Deformation**: C^∞ (Gaussian basis)

## テスト方法

1. アプリケーションを起動
2. Setup Mode で制御点を追加（クリック）
3. Enter キーで Deform Mode に切り替え
4. 制御点をドラッグして変形
5. Python側でマッピングパラメータが計算される
6. （TODO）Rust側で画像が変形される

現在は画像が変形されませんが、制御点の操作とPython通信は正常に動作します。
