//! Strategy trait definitions
//!
//! These traits define the pluggable components of the algorithm:
//! - BasisFunction: φ(r) basis function (Table 1)
//! - DistortionStrategy: σ computation (Eq. 10-13)
//! - RegularizationStrategy: Energy regularization (Eq. 29-33)
//! - SOCPSolverBackend: SOCP problem solver (Eq. 18)

use crate::types::*;

/// Basis function trait (Table 1: Basis functions φ and moduli of gradients)
pub trait BasisFunction: Send + Sync {
    /// φ(r) - radial basis function value
    fn evaluate(&self, r: f64) -> f64;

    /// φ'(r) / r - used in gradient computation
    fn gradient_scaled(&self, r: f64) -> f64;

    /// φ''(r) / r - used in higher-order computations
    fn hessian_scaled(&self, r: f64) -> f64;

    /// Name of this basis function
    fn name(&self) -> &'static str;
}

/// Distortion strategy (Eq. 10-13: Distortion definition)
pub trait DistortionStrategy: Send + Sync {
    /// Compute σ from singular values of Jacobian
    /// Returns scalar distortion value
    ///
    /// For Isometric: σ_iso = max{σ_max, 1/σ_min} (Eq. 10)
    /// For Conformal: σ_conf = σ_max / σ_min (Eq. 11)
    fn compute_distortion(&self, jacobian: Mat2) -> f64;

    /// Eq. 14-17: K_high and K_low for active set management
    /// Returns (K_high, K_low) thresholds
    fn get_activation_threshold(&self) -> (f64, f64);

    /// Build SOCP constraints for this distortion type
    /// (Eq. 21-23 for Isometric, Eq. 28 for Conformal)
    fn build_constraints(
        &self,
        active_set: &ActiveSetInfo,
        basis_evals: &[f64],
        basis_grads: &[Vec2],
    ) -> Result<Vec<ConeConstraint>>;

    /// Name of this strategy
    fn name(&self) -> &'static str;
}

/// Regularization strategy (Eq. 29-33: Regularization energy)
pub trait RegularizationStrategy: Send + Sync {
    /// Build energy terms for regularization
    ///
    /// Biharmonic: E_bh = ∫∫ ||∇²f||² dΩ (Eq. 31)
    /// ARAP: E_arap = ∫∫ ||∇f - R||² dΩ (Eq. 32-33)
    fn build_energy_terms(
        &self,
        domain: &DomainInfo,
        basis_hessians: &[f64],
    ) -> Result<EnergyTerms>;

    /// Name of this regularization
    fn name(&self) -> &'static str;
}

/// SOCP solver backend (Eq. 18: SOCP formulation)
pub trait SOCPSolverBackend: Send + Sync {
    /// Solve the SOCP problem
    /// min c^T Q c  s.t.  ||A_i c + b_i|| ≤ d_i
    ///
    /// Returns coefficient vector (as Vec2 array)
    fn solve(&self, problem: &SOCPProblem) -> Result<Vec<Vec2>>;

    /// Name of this solver backend
    fn name(&self) -> &'static str;
}

// Module declarations for implementations
pub mod basis {
    pub use crate::basis::*;
}

pub mod distortion {
    pub use crate::distortion::*;
}

pub mod regularization {
    pub use crate::regularization::*;
}

pub mod solver {
    pub use crate::solver::*;
}
