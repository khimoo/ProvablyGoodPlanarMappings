//! Basis function trait and implementations.
//!
//! Paper Table 1: three basis function types (Gaussian, B-Spline, TPS)
//! each providing value, gradient, Hessian, and gradient modulus.

pub mod gaussian;
pub mod shape_aware_gaussian;

use crate::types::CoefficientMatrix;
use nalgebra::{DVector, Vector2};

/// Abstraction over basis functions from Table 1.
///
/// Each implementation provides:
/// - Value evaluation f_i(x)
/// - Gradient evaluation ∇f_i(x)
/// - Hessian evaluation H_{f_i}(x) (for biharmonic energy, Eq. 31)
/// - Gradient modulus ω_{∇F}(t) (Table 1, used in Eq. 9)
pub trait BasisFunction: Send + Sync {
    /// Number of basis functions n
    fn count(&self) -> usize;

    /// Evaluate f_i(x) for all basis functions.
    /// Returns DVector<f64> of length n.
    fn evaluate(&self, x: Vector2<f64>) -> DVector<f64>;

    /// Evaluate ∇f_i(x) for all basis functions.
    /// Returns (∂f_i/∂x, ∂f_i/∂y), each DVector<f64> of length n.
    fn gradient(&self, x: Vector2<f64>) -> (DVector<f64>, DVector<f64>);

    /// Evaluate H_{f_i}(x) for all basis functions (Eq. 31).
    /// Returns (∂²f_i/∂x², ∂²f_i/∂x∂y, ∂²f_i/∂y²), each DVector<f64> of length n.
    fn hessian(&self, x: Vector2<f64>) -> (DVector<f64>, DVector<f64>, DVector<f64>);

    /// Table 1: gradient modulus ω_{∇F}(t).
    /// Used in Eq. 9: ω = 2 |||c||| ω_{∇F}
    fn gradient_modulus(&self, t: f64) -> f64;

    /// Inverse of gradient modulus: ω_{∇F}⁻¹(v) = t such that ω_{∇F}(t) = v.
    /// Used in Strategy 2 (Eq. 14) to compute the required fill distance h.
    fn gradient_modulus_inverse(&self, v: f64) -> f64;

    /// Identity mapping coefficients c ∈ R^{2×n} such that f(x) = x.
    /// (J_f = I everywhere)
    fn identity_coefficients(&self) -> CoefficientMatrix;
}
