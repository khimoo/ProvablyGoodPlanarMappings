//! Gaussian RBF basis functions.
//!
//! Table 1: f_i(x) = exp(-|x - x_i|² / (2s²))
//!          ω_{∇F}(t) = t / s²
//!
//! We augment the RBF basis with affine terms {1, x, y} to enable
//! representation of the identity mapping. This is standard practice
//! consistent with "bases mentioned above (and others)" in Section 3.
//! Affine terms have ∇f = const, so ω_{∇f_i} = 0 and they do not
//! affect ω_{∇F} in Eq. 8.

use super::BasisFunction;
use crate::model::types::CoefficientMatrix;
use nalgebra::{DMatrix, DVector, Vector2};

/// Gaussian RBF basis with affine augmentation.
///
/// Basis structure (n = num_centers + 3):
/// - f_0 ... f_{num_centers-1}: Gaussian RBFs
/// - f_{n-3}: constant 1
/// - f_{n-2}: x coordinate
/// - f_{n-1}: y coordinate
pub struct GaussianBasis {
    /// RBF centers {x_i}
    centers: Vec<Vector2<f64>>,
    /// Scale parameter s (Table 1: f_i = exp(-|x-x_i|²/(2s²)))
    s: f64,
}

impl GaussianBasis {
    pub fn new(centers: Vec<Vector2<f64>>, s: f64) -> Self {
        assert!(s > 0.0, "Scale parameter s must be positive");
        Self { centers, s }
    }

    /// Number of RBF centers (excluding affine terms)
    pub fn num_centers(&self) -> usize {
        self.centers.len()
    }

    /// Scale parameter
    pub fn scale(&self) -> f64 {
        self.s
    }
}

impl BasisFunction for GaussianBasis {
    fn count(&self) -> usize {
        // RBF centers + 3 affine terms (1, x, y)
        self.centers.len() + 3
    }

    fn evaluate(&self, x: Vector2<f64>) -> DVector<f64> {
        let n = self.count();
        let mut result = DVector::zeros(n);
        let s2 = self.s * self.s;

        // Gaussian RBFs: f_i(x) = exp(-|x - x_i|² / (2s²))
        for (i, center) in self.centers.iter().enumerate() {
            let diff = x - center;
            let r2 = diff.dot(&diff);
            result[i] = (-r2 / (2.0 * s2)).exp();
        }

        // Affine terms
        let nc = self.centers.len();
        result[nc] = 1.0;     // constant
        result[nc + 1] = x.x; // x coordinate
        result[nc + 2] = x.y; // y coordinate

        result
    }

    fn gradient(&self, x: Vector2<f64>) -> (DVector<f64>, DVector<f64>) {
        let n = self.count();
        let mut grad_x = DVector::zeros(n);
        let mut grad_y = DVector::zeros(n);
        let s2 = self.s * self.s;

        // ∇f_i(x) = f_i(x) · (-(x - x_i) / s²)
        for (i, center) in self.centers.iter().enumerate() {
            let diff = x - center;
            let r2 = diff.dot(&diff);
            let phi = (-r2 / (2.0 * s2)).exp();
            grad_x[i] = phi * (-diff.x / s2);
            grad_y[i] = phi * (-diff.y / s2);
        }

        // Affine terms: ∇1 = (0,0), ∇x = (1,0), ∇y = (0,1)
        let nc = self.centers.len();
        // grad_x[nc] = 0, grad_y[nc] = 0  (constant term, already zero)
        grad_x[nc + 1] = 1.0; // ∂x/∂x
        // grad_y[nc + 1] = 0  (∂x/∂y, already zero)
        // grad_x[nc + 2] = 0  (∂y/∂x, already zero)
        grad_y[nc + 2] = 1.0; // ∂y/∂y

        (grad_x, grad_y)
    }

    fn hessian(&self, x: Vector2<f64>) -> (DVector<f64>, DVector<f64>, DVector<f64>) {
        let n = self.count();
        let mut hxx = DVector::zeros(n);
        let mut hxy = DVector::zeros(n);
        let mut hyy = DVector::zeros(n);
        let s2 = self.s * self.s;

        // H_{f_i}(x):
        // ∂²f_i/∂x² = f_i · (dx² - s²) / s⁴
        // ∂²f_i/∂x∂y = f_i · (dx·dy) / s⁴
        // ∂²f_i/∂y² = f_i · (dy² - s²) / s⁴
        for (i, center) in self.centers.iter().enumerate() {
            let diff = x - center;
            let r2 = diff.dot(&diff);
            let phi = (-r2 / (2.0 * s2)).exp();
            let s4 = s2 * s2;

            hxx[i] = phi * (diff.x * diff.x - s2) / s4;
            hxy[i] = phi * (diff.x * diff.y) / s4;
            hyy[i] = phi * (diff.y * diff.y - s2) / s4;
        }

        // Affine terms: all second derivatives are zero (already zero)

        (hxx, hxy, hyy)
    }

    fn gradient_modulus(&self, t: f64) -> f64 {
        // Table 1: ω_{∇F}(t) = t / s²
        t / (self.s * self.s)
    }

    fn gradient_modulus_inverse(&self, v: f64) -> f64 {
        // ω_{∇F}(t) = t / s² → ω_{∇F}⁻¹(v) = v · s²
        v * self.s * self.s
    }

    fn identity_coefficients(&self) -> CoefficientMatrix {
        // Identity mapping: f(x) = x
        // c¹_{n-2} = 1 (x-coefficient for u), c²_{n-1} = 1 (y-coefficient for v)
        // All other coefficients are 0.
        let n = self.count();
        let mut c = DMatrix::zeros(2, n);
        let nc = self.centers.len();
        c[(0, nc + 1)] = 1.0; // u = x
        c[(1, nc + 2)] = 1.0; // v = y
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_basis() -> GaussianBasis {
        let centers = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(0.0, 1.0),
            Vector2::new(1.0, 1.0),
        ];
        GaussianBasis::new(centers, 0.5)
    }

    #[test]
    fn test_count() {
        let basis = make_simple_basis();
        // 4 RBF centers + 3 affine = 7
        assert_eq!(basis.count(), 7);
    }

    #[test]
    fn test_evaluate_at_center() {
        let basis = make_simple_basis();
        let val = basis.evaluate(Vector2::new(0.0, 0.0));
        // f_0 at its own center should be 1.0
        assert!((val[0] - 1.0).abs() < 1e-12);
        // Affine terms at origin
        assert!((val[4] - 1.0).abs() < 1e-12); // constant
        assert!(val[5].abs() < 1e-12);          // x = 0
        assert!(val[6].abs() < 1e-12);          // y = 0
    }

    #[test]
    fn test_identity_mapping() {
        let basis = make_simple_basis();
        let c = basis.identity_coefficients();
        assert_eq!(c.nrows(), 2);
        assert_eq!(c.ncols(), 7);

        // Evaluate f(x) = Σ c_i f_i(x) at a test point
        let x = Vector2::new(0.3, 0.7);
        let phi = basis.evaluate(x);
        let u: f64 = c.row(0).dot(&phi.transpose());
        let v: f64 = c.row(1).dot(&phi.transpose());

        assert!((u - x.x).abs() < 1e-12, "u = {}, expected {}", u, x.x);
        assert!((v - x.y).abs() < 1e-12, "v = {}, expected {}", v, x.y);
    }

    #[test]
    fn test_gradient_at_center() {
        let basis = make_simple_basis();
        let (gx, gy) = basis.gradient(Vector2::new(0.0, 0.0));
        // ∂f_0/∂x at center (0,0): phi * (-0/s²) = 0
        assert!(gx[0].abs() < 1e-12);
        assert!(gy[0].abs() < 1e-12);
        // Affine x-term: ∂x/∂x = 1
        assert!((gx[5] - 1.0).abs() < 1e-12);
        // Affine y-term: ∂y/∂y = 1
        assert!((gy[6] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_gradient_numerical() {
        let basis = make_simple_basis();
        let x = Vector2::new(0.3, 0.7);
        let (gx, gy) = basis.gradient(x);
        let eps = 1e-7;

        let f0 = basis.evaluate(x);
        let fx = basis.evaluate(x + Vector2::new(eps, 0.0));
        let fy = basis.evaluate(x + Vector2::new(0.0, eps));

        for i in 0..basis.count() {
            let gx_num = (fx[i] - f0[i]) / eps;
            let gy_num = (fy[i] - f0[i]) / eps;
            assert!(
                (gx[i] - gx_num).abs() < 1e-5,
                "grad_x[{}]: analytic={}, numerical={}",
                i, gx[i], gx_num
            );
            assert!(
                (gy[i] - gy_num).abs() < 1e-5,
                "grad_y[{}]: analytic={}, numerical={}",
                i, gy[i], gy_num
            );
        }
    }

    #[test]
    fn test_hessian_numerical() {
        let basis = make_simple_basis();
        let x = Vector2::new(0.3, 0.7);
        let (hxx, hxy, hyy) = basis.hessian(x);
        let eps = 1e-5;

        let (gx0, gy0) = basis.gradient(x);
        let (gx_dx, _) = basis.gradient(x + Vector2::new(eps, 0.0));
        let (gx_dy, gy_dy) = basis.gradient(x + Vector2::new(0.0, eps));

        for i in 0..basis.count() {
            let hxx_num = (gx_dx[i] - gx0[i]) / eps;
            let hxy_num = (gx_dy[i] - gx0[i]) / eps;
            let hyy_num = (gy_dy[i] - gy0[i]) / eps;
            assert!(
                (hxx[i] - hxx_num).abs() < 1e-3,
                "hxx[{}]: analytic={}, numerical={}",
                i, hxx[i], hxx_num
            );
            assert!(
                (hxy[i] - hxy_num).abs() < 1e-3,
                "hxy[{}]: analytic={}, numerical={}",
                i, hxy[i], hxy_num
            );
            assert!(
                (hyy[i] - hyy_num).abs() < 1e-3,
                "hyy[{}]: analytic={}, numerical={}",
                i, hyy[i], hyy_num
            );
        }
    }

    #[test]
    fn test_gradient_modulus() {
        let basis = make_simple_basis();
        // ω_{∇F}(t) = t / s²
        let t = 0.1;
        let expected = t / (0.5 * 0.5);
        assert!((basis.gradient_modulus(t) - expected).abs() < 1e-12);
    }
}
