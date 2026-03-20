//! Strategy 2: required fill distance computation and re-optimization.
//!
//! Paper reference: Section 4 (Eq. 14-15), Strategy 2.
//!
//! Strategy 2 computes the fill distance h required to guarantee that
//! the distortion bound K_max holds everywhere in the domain, then
//! re-optimizes on a denser grid to enforce this guarantee.
//!
//! Key equations:
//! - Eq. 8: ω(h) = 2 |||c||| · ω_{∇F}(h)
//! - Eq. 9: continuity modulus composition
//! - Eq. 11: K_max(K, ω(h)) for isometric distortion
//! - Eq. 14: h ≤ ω⁻¹(min{K_max - K, 1/K - 1/K_max}) (Strategy 2, isometric)

use crate::basis::BasisFunction;
use crate::model::types::{CoefficientMatrix, DomainBounds};

/// Result of a Strategy 2 re-optimization.
pub struct Strategy2Result {
    /// Required fill distance h (Eq. 14)
    pub required_h: f64,
    /// Required grid resolution to achieve h
    pub required_resolution: usize,
    /// Current fill distance (before refinement)
    pub current_h: f64,
    /// Achieved K_max upper bound (Eq. 11)
    pub k_max_achieved: f64,
    /// |||c||| (Eq. 8)
    pub c_norm: f64,
    /// Number of Algorithm 1 steps executed during re-optimization
    pub refinement_steps: usize,
}

/// Compute |||c||| = max{Σ|c¹ᵢ|, Σ|c²ᵢ|} (Eq. 8).
///
/// The coefficient matrix c has shape (2, n):
/// - Row 0: c¹ coefficients (u component)
/// - Row 1: c² coefficients (v component)
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

/// Compute the fill distance h of a uniform grid over the domain (Eq. 5).
///
/// For a uniform square grid with `resolution` points per side on a
/// rectangular domain, the fill distance is the half-diagonal of a grid cell:
///   h = √(dx² + dy²) / 2
/// where dx = (x_max - x_min) / (resolution - 1).
pub fn fill_distance(bounds: &DomainBounds, resolution: usize) -> f64 {
    assert!(resolution >= 2, "Resolution must be at least 2");
    let dx = (bounds.x_max - bounds.x_min) / (resolution as f64 - 1.0);
    let dy = (bounds.y_max - bounds.y_min) / (resolution as f64 - 1.0);
    (dx * dx + dy * dy).sqrt() / 2.0
}

/// Strategy 2 (Eq. 14): compute the required fill distance h for isometric distortion.
///
/// h ≤ ω⁻¹(min{K_max - K, 1/K - 1/K_max})
///
/// where ω(h) = 2 |||c||| · ω_{∇F}(h)  (Eq. 9)
/// and ω⁻¹(v) = ω_{∇F}⁻¹(v / (2 |||c|||))
///
/// Returns `None` if K_max ≤ K (precondition violated) or if c_norm ≈ 0.
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
        // Zero coefficients → identity-like mapping, any h works
        return Some(f64::INFINITY);
    }

    // Eq. 14: min{K_max - K, 1/K - 1/K_max}
    let margin1 = k_max - k;
    let margin2 = 1.0 / k - 1.0 / k_max;
    if margin2 <= 0.0 {
        // 1/K ≤ 1/K_max means K ≥ K_max, shouldn't happen given k_max > k
        return None;
    }
    let v = margin1.min(margin2);

    // ω⁻¹(v) = ω_{∇F}⁻¹(v / (2 |||c|||))
    let inner = v / (2.0 * c_norm);
    let h = basis.gradient_modulus_inverse(inner);

    Some(h)
}

/// Compute the grid resolution needed to achieve fill distance ≤ h.
///
/// For a uniform square grid: h = √(dx² + dy²) / 2
/// With square cells (dx = dy = d): h = d/√2
/// So d = h·√2, and resolution = max(W, H) / d + 1
///
/// For a rectangular domain with different W and H:
///   N ≥ max(W/dx, H/dy) + 1, where we need √(dx² + dy²)/2 ≤ h
///   Using a square grid (same N for both axes):
///   dx = W/(N-1), dy = H/(N-1)
///   h_actual = √((W/(N-1))² + (H/(N-1))²) / 2 = √(W² + H²) / (2(N-1))
///   N ≥ √(W² + H²) / (2h) + 1
pub fn resolution_for_h(bounds: &DomainBounds, h: f64) -> usize {
    assert!(h > 0.0, "Fill distance h must be positive");
    let w = bounds.x_max - bounds.x_min;
    let h_domain = bounds.y_max - bounds.y_min;
    let diag = (w * w + h_domain * h_domain).sqrt();
    let n = (diag / (2.0 * h)).ceil() as usize + 1;
    // Minimum resolution of 2
    n.max(2)
}

/// Strategy 1 (Eq. 11): compute the distortion upper bound K_max given
/// the optimization bound K and the continuity modulus ω(h).
///
/// K_max = max{K + ω(h), 1/(1/K - ω(h))}   if 1/K > ω(h)
///
/// Returns `None` if ω(h) ≥ 1/K (injectivity cannot be guaranteed).
pub fn compute_k_max_isometric(k: f64, omega_h: f64) -> Option<f64> {
    // Eq. 11: requires 1/K > ω(h) for σ(x) > 0 guarantee
    if omega_h >= 1.0 / k {
        return None;
    }

    let k_max1 = k + omega_h;
    let k_max2 = 1.0 / (1.0 / k - omega_h);
    Some(k_max1.max(k_max2))
}

/// Compute ω(h) = 2 |||c||| · ω_{∇F}(h) (Eq. 9).
pub fn omega(h: f64, c_norm: f64, basis: &dyn BasisFunction) -> f64 {
    2.0 * c_norm * basis.gradient_modulus(h)
}

/// Maximum number of Algorithm 1 steps during re-optimization.
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
        // Identity: c¹ = [0,0,0,0, 0, 1, 0], c² = [0,0,0,0, 0, 0, 1]
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
        // Row 0: |2| + |-3| = 5
        // Row 1: |4| = 4
        let norm = compute_c_norm(&c);
        assert!((norm - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_fill_distance() {
        let bounds = make_bounds();
        // 11 points on [0,1]²: dx = dy = 0.1
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
        // With small c_norm, the required h should be large
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
        // ω(h) ≥ 1/K → cannot guarantee injectivity
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
