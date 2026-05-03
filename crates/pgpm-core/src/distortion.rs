//! 歪み計算。
//!
//! 論文 Section 3 "Distortion":
//! - Eq. 19: J_S f, J_A f によるヤコビアンの分解
//! - Eq. 20: J_S, J_A からの特異値 Σ, σ
//! - D_iso = max{Σ, 1/σ}, D_conf = Σ/σ

use crate::model::types::{CoefficientMatrix, PrecomputedData};
use crate::policy::DistortionPolicy;
use nalgebra::Vector2;

/// Eq. 19: ∇u と ∇v の勾配から J_S f(x) と J_A f(x) を計算する。
///
/// J_S f(x) = (∇u(x) + I∇v(x)) / 2
/// J_A f(x) = (∇u(x) - I∇v(x)) / 2
///
/// I は π/2 時計回り回転: I·(a,b) = (b, -a)
/// よって I∇v = (∂v/∂y, -∂v/∂x)
///
/// 注: 論文では「反時計回り」と記述されているが、Eq. 19 の式は
/// 複素共役回転（時計回り）として I が作用する必要がある。
/// J_S が等角（相似）部分、J_A が反等角部分を捉えるため。
/// 検証: 恒等写像 f(x,y)=(x,y) は等角 → J_A=0, J_S=(1,0)。
pub fn compute_j_s_j_a(
    grad_u: Vector2<f64>, // ∇u(x) = (∂u/∂x, ∂u/∂y)
    grad_v: Vector2<f64>, // ∇v(x) = (∂v/∂x, ∂v/∂y)
) -> (Vector2<f64>, Vector2<f64>) {
    // I∇v = (∂v/∂y, -∂v/∂x)  [時計回り π/2 回転]
    let i_grad_v = Vector2::new(grad_v.y, -grad_v.x);

    let j_s = (grad_u + i_grad_v) / 2.0;
    let j_a = (grad_u - i_grad_v) / 2.0;

    (j_s, j_a)
}

/// Eq. 20: J_S と J_A から特異値を計算する。
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

/// Section 3: 等長歪み D_iso = max{Σ, 1/σ}
pub fn isometric_distortion(sigma_max: f64, sigma_min: f64) -> f64 {
    if sigma_min < 1e-15 {
        return f64::INFINITY;
    }
    f64::max(sigma_max, 1.0 / sigma_min)
}

/// Section 3: 等角歪み D_conf = Σ / σ
pub fn conformal_distortion(sigma_max: f64, sigma_min: f64) -> f64 {
    if sigma_min < 1e-15 {
        return f64::INFINITY;
    }
    sigma_max / sigma_min
}

/// コロケーション点インデックス `idx` における勾配 ∇u(z) と ∇v(z) を
/// 事前計算データと現在の係数から計算する。
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

/// 全コロケーション点で歪みを評価する。
///
/// 効率化のため事前計算した grad_phi 値を使用する。
/// コロケーション点ごとに1つの歪み値からなる Vec を返す。
pub fn evaluate_distortion_all(
    coefficients: &CoefficientMatrix,
    precomputed: &PrecomputedData,
    policy: &dyn DistortionPolicy,
) -> Vec<f64> {
    let m = precomputed.grad_phi_x.nrows(); // コロケーション点の数

    (0..m)
        .map(|idx| {
            let (grad_u, grad_v) = grad_uv_at(coefficients, precomputed, idx);
            let (j_s, j_a) = compute_j_s_j_a(grad_u, grad_v);
            let (sigma_max, sigma_min) = singular_values(j_s, j_a);

            policy.distortion_value(sigma_max, sigma_min)
        })
        .collect()
}

/// 全コロケーション点で J_S f を評価する（フレーム更新 Eq. 27 に必要）。
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
        // 恒等写像の場合: ∇u = (1,0), ∇v = (0,1)
        let grad_u = Vector2::new(1.0, 0.0);
        let grad_v = Vector2::new(0.0, 1.0);

        let (j_s, j_a) = compute_j_s_j_a(grad_u, grad_v);

        // I∇v = I·(0,1) = (1, 0)  [時計回り π/2: I(a,b) = (b,-a)]
        // J_S = ((1,0) + (1,0))/2 = (1, 0)
        // J_A = ((1,0) - (1,0))/2 = (0, 0)
        //
        // 恒等写像は等角 → J_A = 0, J_S ≠ 0. ✓
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
        // 2倍一様スケーリング: ∇u = (2,0), ∇v = (0,2)
        let grad_u = Vector2::new(2.0, 0.0);
        let grad_v = Vector2::new(0.0, 2.0);

        let (j_s, j_a) = compute_j_s_j_a(grad_u, grad_v);
        let (sigma_max, sigma_min) = singular_values(j_s, j_a);

        // 一様スケーリング k 倍: Σ = k, σ = k
        assert!((sigma_max - 2.0).abs() < 1e-12);
        assert!((sigma_min - 2.0).abs() < 1e-12);

        // D_iso = max{2, 1/2} = 2
        let d = isometric_distortion(sigma_max, sigma_min);
        assert!((d - 2.0).abs() < 1e-12);

        // D_conf = 2/2 = 1 (一様スケーリングは等角)
        let d_conf = conformal_distortion(sigma_max, sigma_min);
        assert!((d_conf - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_rotation_jacobian() {
        // 90°回転: ∇u = (0,-1), ∇v = (1,0)
        // (u = -y, v = x)
        let grad_u = Vector2::new(0.0, -1.0);
        let grad_v = Vector2::new(1.0, 0.0);

        let (j_s, j_a) = compute_j_s_j_a(grad_u, grad_v);
        let (sigma_max, sigma_min) = singular_values(j_s, j_a);

        // 回転は等長: Σ = 1, σ = 1
        assert!((sigma_max - 1.0).abs() < 1e-12);
        assert!((sigma_min - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_anisotropic_scaling() {
        // xを3倍、yを1倍にスケーリング: ∇u = (3,0), ∇v = (0,1)
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
        // 等角写像（正則関数）: 一様スケーリング + 回転
        // 2倍スケーリング、45°回転: ヤコビアン = 2*[[cos45, -sin45], [sin45, cos45]]
        let c = std::f64::consts::FRAC_1_SQRT_2;
        let grad_u = Vector2::new(2.0 * c, -2.0 * c);
        let grad_v = Vector2::new(2.0 * c, 2.0 * c);

        let (j_s, j_a) = compute_j_s_j_a(grad_u, grad_v);
        let (sigma_max, sigma_min) = singular_values(j_s, j_a);

        // 等角: Σ = σ = 2
        assert!((sigma_max - 2.0).abs() < 1e-10);
        assert!((sigma_min - 2.0).abs() < 1e-10);

        // 等角では D_conf = 1
        let d_conf = conformal_distortion(sigma_max, sigma_min);
        assert!((d_conf - 1.0).abs() < 1e-10);
    }
}
