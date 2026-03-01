//! Distortion computation.
//!
//! Paper Section 3 "Distortion":
//! - Eq. 19: J_S f, J_A f decomposition of the Jacobian
//! - Eq. 20: Singular values Σ, σ from J_S, J_A
//! - D_iso = max{Σ, 1/σ}, D_conf = Σ/σ

use crate::types::{CoefficientMatrix, DistortionType, PrecomputedData};
use nalgebra::Vector2;

/// Eq. 19: Compute J_S f(x) and J_A f(x) from gradients of u and v.
///
/// J_S f(x) = (∇u(x) + I∇v(x)) / 2
/// J_A f(x) = (∇u(x) - I∇v(x)) / 2
///
/// where I is the π/2 clockwise rotation: I·(a,b) = (b, -a)
/// so I∇v = (∂v/∂y, -∂v/∂x)
///
/// Note: The paper states "counter-clockwise" but the formulas in Eq. 19
/// require I to act as the complex-conjugate rotation (CW) so that
/// J_S captures the conformal (similarity) part and J_A the anti-conformal part.
/// Verification: identity f(x,y)=(x,y) is conformal → J_A=0, J_S=(1,0).
pub fn compute_j_s_j_a(
    grad_u: Vector2<f64>, // ∇u(x) = (∂u/∂x, ∂u/∂y)
    grad_v: Vector2<f64>, // ∇v(x) = (∂v/∂x, ∂v/∂y)
) -> (Vector2<f64>, Vector2<f64>) {
    // I∇v = (∂v/∂y, -∂v/∂x)  [CW π/2 rotation]
    let i_grad_v = Vector2::new(grad_v.y, -grad_v.x);

    let j_s = (grad_u + i_grad_v) / 2.0;
    let j_a = (grad_u - i_grad_v) / 2.0;

    (j_s, j_a)
}

/// Eq. 20: Compute singular values from J_S and J_A.
///
/// Σ(x) = ||J_S f(x)|| + ||J_A f(x)||
/// σ(x) = | ||J_S f(x)|| - ||J_A f(x)|| |
pub fn singular_values(j_s: Vector2<f64>, j_a: Vector2<f64>) -> (f64, f64) {
    let norm_j_s = j_s.norm();
    let norm_j_a = j_a.norm();

    let sigma_max = norm_j_s + norm_j_a;
    let sigma_min = (norm_j_s - norm_j_a).abs();

    (sigma_max, sigma_min)
}

/// Section 3: Isometric distortion D_iso = max{Σ, 1/σ}
pub fn isometric_distortion(sigma_max: f64, sigma_min: f64) -> f64 {
    if sigma_min < 1e-15 {
        return f64::INFINITY;
    }
    f64::max(sigma_max, 1.0 / sigma_min)
}

/// Section 3: Conformal distortion D_conf = Σ / σ
pub fn conformal_distortion(sigma_max: f64, sigma_min: f64) -> f64 {
    if sigma_min < 1e-15 {
        return f64::INFINITY;
    }
    sigma_max / sigma_min
}

/// Compute gradients ∇u(z) and ∇v(z) at collocation point index `idx`
/// using precomputed data and current coefficients.
///
/// ∇u(z) = Σ c¹_i ∇f_i(z),  ∇v(z) = Σ c²_i ∇f_i(z)
fn grad_uv_at(
    coefficients: &CoefficientMatrix,
    precomputed: &PrecomputedData,
    idx: usize,
) -> (Vector2<f64>, Vector2<f64>) {
    let n = coefficients.ncols();

    let mut grad_u = Vector2::new(0.0, 0.0);
    let mut grad_v = Vector2::new(0.0, 0.0);

    for i in 0..n {
        let c1 = coefficients[(0, i)];
        let c2 = coefficients[(1, i)];
        let gx = precomputed.grad_phi_x[(idx, i)];
        let gy = precomputed.grad_phi_y[(idx, i)];

        grad_u.x += c1 * gx;
        grad_u.y += c1 * gy;
        grad_v.x += c2 * gx;
        grad_v.y += c2 * gy;
    }

    (grad_u, grad_v)
}

/// Evaluate distortion at all collocation points.
///
/// Uses precomputed grad_phi values for efficiency.
/// Returns a Vec of distortion values, one per collocation point.
pub fn evaluate_distortion_all(
    coefficients: &CoefficientMatrix,
    precomputed: &PrecomputedData,
    distortion_type: &DistortionType,
) -> Vec<f64> {
    let m = precomputed.grad_phi_x.nrows(); // number of collocation points

    (0..m)
        .map(|idx| {
            let (grad_u, grad_v) = grad_uv_at(coefficients, precomputed, idx);
            let (j_s, j_a) = compute_j_s_j_a(grad_u, grad_v);
            let (sigma_max, sigma_min) = singular_values(j_s, j_a);

            match distortion_type {
                DistortionType::Isometric => isometric_distortion(sigma_max, sigma_min),
                DistortionType::Conformal { .. } => conformal_distortion(sigma_max, sigma_min),
            }
        })
        .collect()
}

/// Evaluate J_S f at all collocation points (needed for frame update, Eq. 27).
pub fn evaluate_j_s_all(
    coefficients: &CoefficientMatrix,
    precomputed: &PrecomputedData,
) -> Vec<Vector2<f64>> {
    let m = precomputed.grad_phi_x.nrows();

    (0..m)
        .map(|idx| {
            let (grad_u, grad_v) = grad_uv_at(coefficients, precomputed, idx);
            let (j_s, _) = compute_j_s_j_a(grad_u, grad_v);
            j_s
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_jacobian() {
        // For identity mapping: ∇u = (1,0), ∇v = (0,1)
        let grad_u = Vector2::new(1.0, 0.0);
        let grad_v = Vector2::new(0.0, 1.0);

        let (j_s, j_a) = compute_j_s_j_a(grad_u, grad_v);

        // I∇v = I·(0,1) = (1, 0)  [CW π/2: I(a,b) = (b,-a)]
        // J_S = ((1,0) + (1,0))/2 = (1, 0)
        // J_A = ((1,0) - (1,0))/2 = (0, 0)
        //
        // Identity is conformal → J_A = 0, J_S ≠ 0. ✓
        // Σ = ||J_S|| + ||J_A|| = 1 + 0 = 1
        // σ = |1 - 0| = 1
        // D_iso = max{1, 1/1} = 1 ✓
        assert!((j_s.x - 1.0).abs() < 1e-12);
        assert!(j_s.y.abs() < 1e-12);
        assert!(j_a.x.abs() < 1e-12);
        assert!(j_a.y.abs() < 1e-12);

        let (sigma_max, sigma_min) = singular_values(j_s, j_a);
        assert!((sigma_max - 1.0).abs() < 1e-12);
        assert!((sigma_min - 1.0).abs() < 1e-12);

        let d = isometric_distortion(sigma_max, sigma_min);
        assert!((d - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_scaling_jacobian() {
        // Uniform scaling by 2: ∇u = (2,0), ∇v = (0,2)
        let grad_u = Vector2::new(2.0, 0.0);
        let grad_v = Vector2::new(0.0, 2.0);

        let (j_s, j_a) = compute_j_s_j_a(grad_u, grad_v);
        let (sigma_max, sigma_min) = singular_values(j_s, j_a);

        // For uniform scaling by k: Σ = k, σ = k
        assert!((sigma_max - 2.0).abs() < 1e-12);
        assert!((sigma_min - 2.0).abs() < 1e-12);

        // D_iso = max{2, 1/2} = 2
        let d = isometric_distortion(sigma_max, sigma_min);
        assert!((d - 2.0).abs() < 1e-12);

        // D_conf = 2/2 = 1 (conformal for uniform scaling)
        let d_conf = conformal_distortion(sigma_max, sigma_min);
        assert!((d_conf - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_rotation_jacobian() {
        // 90° rotation: ∇u = (0,-1), ∇v = (1,0)
        // (u = -y, v = x)
        let grad_u = Vector2::new(0.0, -1.0);
        let grad_v = Vector2::new(1.0, 0.0);

        let (j_s, j_a) = compute_j_s_j_a(grad_u, grad_v);
        let (sigma_max, sigma_min) = singular_values(j_s, j_a);

        // Rotation is isometric: Σ = 1, σ = 1
        assert!((sigma_max - 1.0).abs() < 1e-12);
        assert!((sigma_min - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_anisotropic_scaling() {
        // Scale x by 3, y by 1: ∇u = (3,0), ∇v = (0,1)
        let grad_u = Vector2::new(3.0, 0.0);
        let grad_v = Vector2::new(0.0, 1.0);

        let (j_s, j_a) = compute_j_s_j_a(grad_u, grad_v);
        let (sigma_max, sigma_min) = singular_values(j_s, j_a);

        assert!((sigma_max - 3.0).abs() < 1e-12);
        assert!((sigma_min - 1.0).abs() < 1e-12);

        // D_iso = max{3, 1/1} = 3
        let d = isometric_distortion(sigma_max, sigma_min);
        assert!((d - 3.0).abs() < 1e-12);

        // D_conf = 3/1 = 3
        let d_conf = conformal_distortion(sigma_max, sigma_min);
        assert!((d_conf - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_conformal_mapping() {
        // Conformal mapping (holomorphic): uniform scaling + rotation
        // Scale by 2, rotate 45°: Jacobian = 2*[[cos45, -sin45], [sin45, cos45]]
        let c = std::f64::consts::FRAC_1_SQRT_2;
        let grad_u = Vector2::new(2.0 * c, -2.0 * c);
        let grad_v = Vector2::new(2.0 * c, 2.0 * c);

        let (j_s, j_a) = compute_j_s_j_a(grad_u, grad_v);
        let (sigma_max, sigma_min) = singular_values(j_s, j_a);

        // Conformal: Σ = σ = 2
        assert!((sigma_max - 2.0).abs() < 1e-10);
        assert!((sigma_min - 2.0).abs() < 1e-10);

        // D_conf = 1 for conformal
        let d_conf = conformal_distortion(sigma_max, sigma_min);
        assert!((d_conf - 1.0).abs() < 1e-10);
    }
}
