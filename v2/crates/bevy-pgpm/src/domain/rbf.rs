//! RBF スケールパラメータの計算。

use nalgebra::Vector2;

/// 適切な RBF スケールパラメータ s を計算。
/// 可能であれば平均最近傍距離の ~0.8 倍を使用し、
/// そうでなければドメインサイズの一部にフォールバック。
pub fn compute_rbf_scale(centers: &[Vector2<f64>], width: f64, height: f64) -> f64 {
    if centers.len() < 2 {
        return (width.max(height)) / 4.0;
    }

    let mut total = 0.0;
    for (i, p) in centers.iter().enumerate() {
        let mut min_d = f64::INFINITY;
        for (j, q) in centers.iter().enumerate() {
            if i != j {
                let d = (p - q).norm();
                if d < min_d {
                    min_d = d;
                }
            }
        }
        total += min_d;
    }
    let avg_nn = total / centers.len() as f64;
    avg_nn * 0.8
}
