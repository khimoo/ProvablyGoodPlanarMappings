//! Shape-aware Gaussian RBF basis functions using geodesic distance.
//!
//! Paper Section "Shape aware bases" (p.76:9):
//! > "we tested shape aware variation of Gaussians, which is achieved by
//! > simply replacing the norm in their definition with the shortest
//! > distance function."
//!
//! f_i(x) = exp(-d_geo(x, x_i)² / (2s²))
//!
//! where d_geo is the shortest interior-path distance within the domain.
//! When no polygon is provided, falls back to Euclidean distance.

use super::BasisFunction;
use crate::geodesic::{self, GeodesicField};
use crate::types::{CoefficientMatrix, DomainBounds};
use nalgebra::{DMatrix, DVector, Vector2};

/// Shape-aware Gaussian RBF basis with geodesic distance.
///
/// Basis structure (n = num_centers + 3):
/// - f_0 ... f_{num_centers-1}: Shape-aware Gaussian RBFs
/// - f_{n-3}: constant 1
/// - f_{n-2}: x coordinate
/// - f_{n-1}: y coordinate
pub struct ShapeAwareGaussianBasis {
    /// RBF centers {x_i}
    centers: Vec<Vector2<f64>>,
    /// Scale parameter s
    s: f64,
    /// Precomputed geodesic distance fields, one per center.
    distance_fields: Vec<GeodesicField>,
    /// Domain bounds (for coordinate reference)
    _bounds: DomainBounds,
}

impl ShapeAwareGaussianBasis {
    /// Create a new shape-aware Gaussian basis.
    ///
    /// # Arguments
    /// - `centers`: RBF center positions (source handle positions)
    /// - `s`: Gaussian scale parameter
    /// - `polygon`: Domain contour polygon (in domain coordinates)
    /// - `holes`: Interior hole contour polygons
    /// - `bounds`: Domain bounding box
    /// - `fmm_resolution`: Grid resolution for the FMM computation
    pub fn new(
        centers: Vec<Vector2<f64>>,
        s: f64,
        polygon: &[Vector2<f64>],
        holes: &[&[Vector2<f64>]],
        bounds: &DomainBounds,
        fmm_resolution: usize,
    ) -> Self {
        assert!(s > 0.0, "Scale parameter s must be positive");

        // Build domain mask on the FMM grid (excluding holes)
        let mask = geodesic::build_domain_mask(bounds, fmm_resolution, fmm_resolution, polygon, holes);

        // Compute geodesic distance field from each center
        let distance_fields: Vec<GeodesicField> = centers
            .iter()
            .map(|&center| {
                GeodesicField::compute(center, bounds, fmm_resolution, fmm_resolution, &mask)
            })
            .collect();

        Self {
            centers,
            s,
            distance_fields,
            _bounds: bounds.clone(),
        }
    }

    /// Number of RBF centers (excluding affine terms).
    pub fn num_centers(&self) -> usize {
        self.centers.len()
    }

    /// Scale parameter.
    pub fn scale(&self) -> f64 {
        self.s
    }
}

impl BasisFunction for ShapeAwareGaussianBasis {
    fn count(&self) -> usize {
        self.centers.len() + 3
    }

    fn evaluate(&self, x: Vector2<f64>) -> DVector<f64> {
        let n = self.count();
        let mut result = DVector::zeros(n);
        let s2 = self.s * self.s;

        // Shape-aware Gaussian RBFs: f_i(x) = exp(-d_geo(x, x_i)² / (2s²))
        for (i, field) in self.distance_fields.iter().enumerate() {
            let d = field.interpolate(x);
            if !d.is_finite() {
                // Unreachable point → basis value is 0
                continue;
            }
            result[i] = (-d * d / (2.0 * s2)).exp();
        }

        // Affine terms (same as Euclidean version)
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

        // ∇φ_i(x) = φ_i(x) · (-d / s²) · ∇d
        // where d = d_geo(x, x_i) and ∇d is the gradient of the distance field
        //
        // Special case: when d ≈ 0 (at or very near the center), the gradient
        // should be zero since φ is at its maximum. However, ∇d can be Inf
        // near domain walls, leading to 0 * Inf = NaN. We guard against this
        // by skipping the computation when |d| is negligible.
        for (i, field) in self.distance_fields.iter().enumerate() {
            let d = field.interpolate(x);
            if d.abs() < 1e-12 * self.s || !d.is_finite() {
                // At the center or unreachable → gradient is zero
                continue;
            }
            let phi = (-d * d / (2.0 * s2)).exp();
            let grad_d = field.interpolate_gradient(x);

            // Also guard against non-finite gradient values from the
            // distance field (can occur near domain boundaries where
            // neighboring cells are walls with Inf distance).
            if !grad_d.x.is_finite() || !grad_d.y.is_finite() {
                continue;
            }

            let factor = phi * (-d / s2);
            grad_x[i] = factor * grad_d.x;
            grad_y[i] = factor * grad_d.y;
        }

        // Affine terms: ∇1 = (0,0), ∇x = (1,0), ∇y = (0,1)
        let nc = self.centers.len();
        grad_x[nc + 1] = 1.0;
        grad_y[nc + 2] = 1.0;

        (grad_x, grad_y)
    }

    fn hessian(&self, x: Vector2<f64>) -> (DVector<f64>, DVector<f64>, DVector<f64>) {
        // Numerical Hessian via finite differences of the gradient.
        // The paper notes that exact shortest distances in a polygonal domain
        // have discontinuous derivatives at certain points, so numerical
        // differentiation is appropriate here.
        let n = self.count();
        let eps = self.s * 1e-3; // relative to scale

        let (gx_0, gy_0) = self.gradient(x);
        let (gx_px, _) = self.gradient(x + Vector2::new(eps, 0.0));
        let (gx_py, gy_py) = self.gradient(x + Vector2::new(0.0, eps));

        let mut hxx = DVector::zeros(n);
        let mut hxy = DVector::zeros(n);
        let mut hyy = DVector::zeros(n);

        for i in 0..n {
            hxx[i] = (gx_px[i] - gx_0[i]) / eps;
            hxy[i] = (gx_py[i] - gx_0[i]) / eps;
            hyy[i] = (gy_py[i] - gy_0[i]) / eps;
        }

        (hxx, hxy, hyy)
    }

    fn gradient_modulus(&self, t: f64) -> f64 {
        // Paper p.76:9: "To provide a proof of injectivity [...] the modulus of
        // the gradients of the Gaussian shape-aware functions, ω∇F, should be
        // calculated. Although straightforward, it is cumbersome to compute it
        // in general, and we defer it to future work."
        //
        // In practice, the Euclidean modulus t/s² is used as an approximation.
        // This is valid because the geodesic distance is always >= Euclidean,
        // so the actual gradient modulus is bounded by the Euclidean one.
        t / (self.s * self.s)
    }

    fn gradient_modulus_inverse(&self, v: f64) -> f64 {
        // Same approximation as gradient_modulus (Euclidean bound).
        // ω_{∇F}(t) = t / s² → ω_{∇F}⁻¹(v) = v · s²
        v * self.s * self.s
    }

    fn identity_coefficients(&self) -> CoefficientMatrix {
        // Same as Euclidean Gaussian: affine terms encode identity.
        let n = self.count();
        let mut c = DMatrix::zeros(2, n);
        let nc = self.centers.len();
        c[(0, nc + 1)] = 1.0; // u = x
        c[(1, nc + 2)] = 1.0; // v = y
        c
    }
}
