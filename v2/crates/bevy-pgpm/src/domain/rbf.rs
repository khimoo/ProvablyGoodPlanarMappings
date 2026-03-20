//! RBF scale parameter computation.

use nalgebra::Vector2;

/// Compute a reasonable RBF scale parameter s.
/// Uses ~0.8 of the average nearest-neighbor distance if available,
/// otherwise falls back to a fraction of the domain size.
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
