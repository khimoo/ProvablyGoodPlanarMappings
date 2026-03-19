//! Basis function implementations (Table 1)

use crate::strategy::BasisFunction;

/// Gaussian RBF basis function
/// φ(r) = exp(-(r/σ)²)
///
/// Table 1 reference: Euclidean distance Gaussian
pub struct GaussianBasis {
    /// Bandwidth σ - controls the radius of influence
    scale: f64,
}

impl GaussianBasis {
    /// Create Gaussian basis with given scale
    pub fn new(scale: f64) -> Self {
        Self { scale }
    }

    /// Default Gaussian with scale 1.0
    pub fn default() -> Self {
        Self { scale: 1.0 }
    }
}

impl BasisFunction for GaussianBasis {
    /// φ(r) = exp(-(r/σ)²)
    fn evaluate(&self, r: f64) -> f64 {
        let normalized = r / self.scale;
        (-normalized * normalized).exp()
    }

    /// φ'(r) / r = -2 exp(-(r/σ)²) / σ²
    ///
    /// Used to compute ∇f = Σ c_i * φ'(r) / r * (x - x_i)
    fn gradient_scaled(&self, r: f64) -> f64 {
        let normalized = r / self.scale;
        let s2 = self.scale * self.scale;
        -2.0 * (-normalized * normalized).exp() / s2
    }

    /// φ''(r) / r for second-order terms
    ///
    /// Used in regularization (biharmonic) computations
    fn hessian_scaled(&self, r: f64) -> f64 {
        let s2 = self.scale * self.scale;
        let s4 = s2 * s2;
        let normalized = r / self.scale;
        let exp_term = (-normalized * normalized).exp();

        // φ''(r) = d/dr[-2(r/σ²) exp(-(r/σ)²)]
        //        = -2/σ² exp(-(r/σ)²) + 4(r²/σ⁴) exp(-(r/σ)²)
        //        = [-2/σ² + 4r²/σ⁴] exp(-(r/σ)²)
        // φ''(r)/r = [-2/(σ²r) + 4r/σ⁴] exp(-(r/σ)²)

        // For computational stability, compute differently:
        // φ''(r)/r = exp(-(r/σ)²) * [-2/σ² * (1/r) + 4r/σ⁴]
        //          = exp(-(r/σ)²) / σ² * [-2/r + 4r²/σ²]
        //          = exp(-(r/σ)²) * [-2/(σ²r) + 4r/σ⁴]

        if r < 1e-12 {
            // Limit as r → 0
            -2.0 / s2
        } else {
            exp_term * (-2.0 / (s2 * r) + 4.0 * r / s4)
        }
    }

    fn name(&self) -> &'static str {
        "Gaussian"
    }
}

// Phase 3: Placeholder for B-Spline basis
// pub struct BSplineBasis { ... }

// Phase 3: Placeholder for TPS basis
// pub struct TPSBasis { ... }
