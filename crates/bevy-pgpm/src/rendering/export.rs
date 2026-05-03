//! CPUソフトウェアラスタライズによる変形画像エクスポート。
//!
//! Bevyのレンダリングパイプラインはオフスクリーンレンダリング時にアルファを
//! 保持しないため（bevyengine/bevy#18229）、メッシュの頂点・UV・インデックスと
//! 元テクスチャから直接ラスタライズして透過PNG出力を行う。

use image::{Rgba, RgbaImage};

/// 変形メッシュをCPUでラスタライズし、透過背景のRGBA画像を生成する。
///
/// # 引数
/// - `positions`: 変形後の頂点位置（ワールド座標: 中央原点、Y上向き）
/// - `uvs`: テクスチャ座標 [0,1]²
/// - `indices`: 三角形インデックスリスト
/// - `src_data`: 元テクスチャのRGBA8ピクセルデータ
/// - `src_width`, `src_height`: 元テクスチャの寸法
pub fn rasterize_deformed_image(
    positions: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
    src_data: &[u8],
    src_width: u32,
    src_height: u32,
) -> RgbaImage {
    // バウンディングボックス計算
    let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
    let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
    for &[x, y, _] in positions {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    let padding = 2.0;
    min_x -= padding;
    min_y -= padding;
    max_x += padding;
    max_y += padding;

    let out_w = (max_x - min_x).ceil() as u32;
    let out_h = (max_y - min_y).ceil() as u32;

    // 出力画像（全ピクセル透過で初期化）
    let mut output = RgbaImage::new(out_w, out_h);

    // 三角形ごとにラスタライズ
    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);

        // ワールド座標 → 出力ピクセル座標
        // ワールド: 中央原点、Y上向き
        // ピクセル: 左上原点、Y下向き
        let p0 = (positions[i0][0] - min_x, max_y - positions[i0][1]);
        let p1 = (positions[i1][0] - min_x, max_y - positions[i1][1]);
        let p2 = (positions[i2][0] - min_x, max_y - positions[i2][1]);

        // 退化三角形をスキップ
        let area = edge_fn(p0, p1, p2);
        if area.abs() < 1e-6 {
            continue;
        }

        // 三角形のバウンディングボックス（出力画像内にクランプ）
        let tri_min_x = p0.0.min(p1.0).min(p2.0).floor().max(0.0) as u32;
        let tri_min_y = p0.1.min(p1.1).min(p2.1).floor().max(0.0) as u32;
        let tri_max_x = (p0.0.max(p1.0).max(p2.0).ceil() as u32).min(out_w.saturating_sub(1));
        let tri_max_y = (p0.1.max(p1.1).max(p2.1).ceil() as u32).min(out_h.saturating_sub(1));

        for py in tri_min_y..=tri_max_y {
            for px in tri_min_x..=tri_max_x {
                let p = (px as f32 + 0.5, py as f32 + 0.5);

                let w0 = edge_fn(p1, p2, p);
                let w1 = edge_fn(p2, p0, p);
                let w2 = edge_fn(p0, p1, p);

                // 全て同符号なら三角形内部（CW/CCW両方対応）
                let inside = (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0)
                    || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0);

                if !inside {
                    continue;
                }

                let sum = w0 + w1 + w2;
                if sum.abs() < 1e-10 {
                    continue;
                }
                let (b0, b1, b2) = (w0 / sum, w1 / sum, w2 / sum);

                // UV 補間
                let u = b0 * uvs[i0][0] + b1 * uvs[i1][0] + b2 * uvs[i2][0];
                let v = b0 * uvs[i0][1] + b1 * uvs[i1][1] + b2 * uvs[i2][1];

                let color = sample_bilinear(src_data, src_width, src_height, u, v);
                output.put_pixel(px, py, color);
            }
        }
    }

    output
}

/// 辺関数（2倍の符号付き面積）。
fn edge_fn(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> f32 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

/// バイリニア補間によるテクスチャサンプリング。
fn sample_bilinear(data: &[u8], w: u32, h: u32, u: f32, v: f32) -> Rgba<u8> {
    let fx = u * w as f32 - 0.5;
    let fy = v * h as f32 - 0.5;

    let x0 = fx.floor().max(0.0) as u32;
    let y0 = fy.floor().max(0.0) as u32;
    let x1 = (x0 + 1).min(w.saturating_sub(1));
    let y1 = (y0 + 1).min(h.saturating_sub(1));

    let dx = (fx - x0 as f32).clamp(0.0, 1.0);
    let dy = (fy - y0 as f32).clamp(0.0, 1.0);

    let get = |x: u32, y: u32| -> [f32; 4] {
        let idx = ((y * w + x) * 4) as usize;
        if idx + 3 < data.len() {
            [
                data[idx] as f32,
                data[idx + 1] as f32,
                data[idx + 2] as f32,
                data[idx + 3] as f32,
            ]
        } else {
            [0.0; 4]
        }
    };

    let c00 = get(x0, y0);
    let c10 = get(x1, y0);
    let c01 = get(x0, y1);
    let c11 = get(x1, y1);

    let mut result = [0u8; 4];
    for i in 0..4 {
        let val = c00[i] * (1.0 - dx) * (1.0 - dy)
            + c10[i] * dx * (1.0 - dy)
            + c01[i] * (1.0 - dx) * dy
            + c11[i] * dx * dy;
        result[i] = val.round().clamp(0.0, 255.0) as u8;
    }

    Rgba(result)
}
