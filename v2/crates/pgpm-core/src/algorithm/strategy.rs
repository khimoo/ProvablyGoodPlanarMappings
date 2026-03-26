//! Strategy 2: 必要な充填距離の計算と再最適化。
//!
//! 論文参照: Section 4 (Eq. 14-15), Strategy 2。
//!
//! Strategy 2 は歪み上界 K_max がドメイン全体で成立することを保証するために
//! 必要な充填距離 h を計算し、より密なグリッド上で再最適化して
//! この保証を強制する。
//!
//! 主要な式:
//! - Eq. 8: ω(h) = 2 |||c||| · ω_{∇F}(h)
//! - Eq. 9: 連続の度合いの合成
//! - Eq. 11: 等長歪みの K_max(K, ω(h))
//! - Eq. 14: h ≤ ω⁻¹(min{K_max - K, 1/K - 1/K_max})（Strategy 2、等長）

use crate::basis::BasisFunction;
use crate::model::types::{CoefficientMatrix, DomainBounds};

/// Strategy 2 再最適化の結果。
pub struct Strategy2Result {
    /// 必要な充填距離 h（Eq. 14）
    pub required_h: f64,
    /// h を達成するのに必要なグリッド解像度
    pub required_resolution: usize,
    /// 現在の充填距離（細分化前）
    pub current_h: f64,
    /// 達成された K_max 上界（Eq. 11）
    pub k_max_achieved: f64,
    /// |||c|||（Eq. 8）
    pub c_norm: f64,
    /// 再最適化中に実行された Algorithm 1 ステップ数
    pub refinement_steps: usize,
}

/// |||c||| = max{Σ|c¹ᵢ|, Σ|c²ᵢ|} を計算（Eq. 8）。
///
/// 係数行列 c の形状は (2, n):
/// - 行 0: c¹ 係数（u 成分）
/// - 行 1: c² 係数（v 成分）
pub fn compute_c_norm(coefficients: &CoefficientMatrix) -> f64 {
    let n = coefficients.ncols();
    let mut sum_row0 = 0.0;
    let mut sum_row1 = 0.0;
    for j in 0..n {
        sum_row0 += coefficients[(0, j)].abs();
        sum_row1 += coefficients[(1, j)].abs();
    }
    sum_row0.max(sum_row1)
}

/// ドメイン上の一様グリッドの充填距離 h を計算（Eq. 5）。
///
/// 矩形ドメイン上で1辺あたり `resolution` 点の一様正方グリッドの場合、
/// 充填距離はグリッドセルの半対角線長:
///   h = √(dx² + dy²) / 2
/// ここで dx = (x_max - x_min) / (resolution - 1)。
pub fn fill_distance(bounds: &DomainBounds, resolution: usize) -> f64 {
    assert!(resolution >= 2, "Resolution must be at least 2");
    let dx = (bounds.x_max - bounds.x_min) / (resolution as f64 - 1.0);
    let dy = (bounds.y_max - bounds.y_min) / (resolution as f64 - 1.0);
    (dx * dx + dy * dy).sqrt() / 2.0
}

/// Strategy 2（Eq. 14）: 等長歪みに必要な充填距離 h を計算。
///
/// h ≤ ω⁻¹(min{K_max - K, 1/K - 1/K_max})
///
/// ここで ω(h) = 2 |||c||| · ω_{∇F}(h)（Eq. 9）
/// かつ ω⁻¹(v) = ω_{∇F}⁻¹(v / (2 |||c|||))
///
/// K_max ≤ K（前提条件違反）または c_norm ≈ 0 の場合は `None` を返す。
pub fn required_h_isometric(
    k: f64,
    k_max: f64,
    c_norm: f64,
    basis: &dyn BasisFunction,
) -> Option<f64> {
    if k_max <= k {
        return None;
    }
    if c_norm < 1e-15 {
        // 零係数 → 恒等写像的、任意の h で可
        return Some(f64::INFINITY);
    }

    // Eq. 14: min{K_max - K, 1/K - 1/K_max}
    let margin1 = k_max - k;
    let margin2 = 1.0 / k - 1.0 / k_max;
    if margin2 <= 0.0 {
        // 1/K ≤ 1/K_max は K ≥ K_max を意味、k_max > k なので発生しないはず
        return None;
    }
    let v = margin1.min(margin2);

    // ω⁻¹(v) = ω_{∇F}⁻¹(v / (2 |||c|||))
    let inner = v / (2.0 * c_norm);
    let h = basis.gradient_modulus_inverse(inner);

    Some(h)
}

/// 充填距離 ≤ h を達成するために必要なグリッド解像度を計算。
///
/// 一様正方グリッドの場合: h = √(dx² + dy²) / 2
/// 正方セル（dx = dy = d）では: h = d/√2
/// よって d = h·√2、resolution = max(W, H) / d + 1
///
/// 異なる W と H を持つ矩形ドメインの場合:
///   N ≥ max(W/dx, H/dy) + 1、ここで √(dx² + dy²)/2 ≤ h が必要
///   正方グリッド（両軸で同じ N）を使用:
///   dx = W/(N-1), dy = H/(N-1)
///   h_actual = √((W/(N-1))² + (H/(N-1))²) / 2 = √(W² + H²) / (2(N-1))
///   N ≥ √(W² + H²) / (2h) + 1
pub fn resolution_for_h(bounds: &DomainBounds, h: f64) -> usize {
    assert!(h > 0.0, "Fill distance h must be positive");
    let w = bounds.x_max - bounds.x_min;
    let h_domain = bounds.y_max - bounds.y_min;
    let diag = (w * w + h_domain * h_domain).sqrt();
    let n = (diag / (2.0 * h)).ceil() as usize + 1;
    // 最小解像度は 2
    n.max(2)
}

/// Strategy 1（Eq. 11）: 最適化上界 K と連続の度合い ω(h) から
/// 歪み上界 K_max を計算。
///
/// K_max = max{K + ω(h), 1/(1/K - ω(h))}   ただし 1/K > ω(h)
///
/// ω(h) ≥ 1/K（単射性が保証できない）の場合は `None` を返す。
pub fn compute_k_max_isometric(k: f64, omega_h: f64) -> Option<f64> {
    // Eq. 11: σ(x) > 0 の保証には 1/K > ω(h) が必要
    if omega_h >= 1.0 / k {
        return None;
    }

    let k_max1 = k + omega_h;
    let k_max2 = 1.0 / (1.0 / k - omega_h);
    Some(k_max1.max(k_max2))
}

/// ω(h) = 2 |||c||| · ω_{∇F}(h) を計算（Eq. 9）。
pub fn omega(h: f64, c_norm: f64, basis: &dyn BasisFunction) -> f64 {
    2.0 * c_norm * basis.gradient_modulus(h)
}

/// 再最適化中の Algorithm 1 ステップの最大数。
pub const MAX_REFINEMENT_STEPS: usize = 200;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::gaussian::GaussianBasis;
    use nalgebra::{DMatrix, Vector2};

    fn make_basis() -> GaussianBasis {
        let centers = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(0.0, 1.0),
            Vector2::new(1.0, 1.0),
        ];
        GaussianBasis::new(centers, 0.5)
    }

    fn make_bounds() -> DomainBounds {
        DomainBounds {
            x_min: 0.0,
            x_max: 1.0,
            y_min: 0.0,
            y_max: 1.0,
        }
    }

    #[test]
    fn test_compute_c_norm_identity() {
        let basis = make_basis();
        let c = basis.identity_coefficients();
        // 恒等: c¹ = [0,0,0,0, 0, 1, 0], c² = [0,0,0,0, 0, 0, 1]
        // Σ|c¹| = 1, Σ|c²| = 1
        let norm = compute_c_norm(&c);
        assert!((norm - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_compute_c_norm_scaled() {
        let mut c = DMatrix::zeros(2, 3);
        c[(0, 0)] = 2.0;
        c[(0, 1)] = -3.0;
        c[(1, 2)] = 4.0;
        // 行 0: |2| + |-3| = 5
        // 行 1: |4| = 4
        let norm = compute_c_norm(&c);
        assert!((norm - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_fill_distance() {
        let bounds = make_bounds();
        // [0,1]² 上に 11 点: dx = dy = 0.1
        // h = √(0.01 + 0.01) / 2 = √0.02 / 2 ≈ 0.0707
        let h = fill_distance(&bounds, 11);
        let expected = (0.02_f64).sqrt() / 2.0;
        assert!((h - expected).abs() < 1e-10);
    }

    #[test]
    fn test_resolution_for_h_roundtrip() {
        let bounds = make_bounds();
        let target_h = 0.05;
        let n = resolution_for_h(&bounds, target_h);
        let actual_h = fill_distance(&bounds, n);
        assert!(actual_h <= target_h, "actual_h={} > target_h={}", actual_h, target_h);
    }

    #[test]
    fn test_required_h_isometric_basic() {
        let basis = make_basis();
        // 小さい c_norm では、必要な h は大きくなるべき
        let h = required_h_isometric(2.0, 3.0, 0.1, &basis);
        assert!(h.is_some());
        assert!(h.unwrap() > 0.0);
    }

    #[test]
    fn test_required_h_isometric_precondition() {
        let basis = make_basis();
        // K_max ≤ K → None
        assert!(required_h_isometric(3.0, 3.0, 1.0, &basis).is_none());
        assert!(required_h_isometric(3.0, 2.0, 1.0, &basis).is_none());
    }

    #[test]
    fn test_compute_k_max_isometric() {
        // K=2, ω(h)=0.1
        // K_max = max{2 + 0.1, 1/(1/2 - 0.1)} = max{2.1, 1/0.4} = max{2.1, 2.5} = 2.5
        let k_max = compute_k_max_isometric(2.0, 0.1);
        assert!(k_max.is_some());
        assert!((k_max.unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_compute_k_max_isometric_too_large_omega() {
        // ω(h) ≥ 1/K → 単射性を保証できない
        assert!(compute_k_max_isometric(2.0, 0.5).is_none());
        assert!(compute_k_max_isometric(2.0, 0.6).is_none());
    }

    #[test]
    fn test_omega() {
        let basis = make_basis();
        // ω(h) = 2 * c_norm * h / s²
        let h = 0.1;
        let c_norm = 1.5;
        let expected = 2.0 * c_norm * h / (0.5 * 0.5);
        let actual = omega(h, c_norm, &basis);
        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn test_gradient_modulus_inverse_roundtrip() {
        let basis = make_basis();
        let t = 0.3;
        let v = basis.gradient_modulus(t);
        let t_recovered = basis.gradient_modulus_inverse(v);
        assert!((t - t_recovered).abs() < 1e-12);
    }
}
