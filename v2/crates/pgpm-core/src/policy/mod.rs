//! Distortion policy trait and concrete implementations.
//!
//! The `DistortionPolicy` trait encapsulates the distortion-type-specific
//! behaviour (isometric vs conformal) that varies in:
//! - distortion value computation (Section 3)
//! - SOCP constraint construction (Eq. 23/26 vs Eq. 28)
//! - Strategy 2 fill distance computation (Eq. 14 vs Eq. 15)
//!
//! This is `pub(crate)` — external consumers interact via `PlanarMapping`
//! and factory functions.

use crate::basis::BasisFunction;
use crate::distortion;
use crate::algorithm::strategy;
use crate::model::types::{AlgorithmState, PrecomputedData};
use crate::numerics::solver;

/// Distortion-type-specific behaviour for the SOCP formulation.
pub trait DistortionPolicy: Send + Sync {
    /// Compute the distortion value from singular values (Section 3).
    fn distortion_value(&self, sigma_max: f64, sigma_min: f64) -> f64;

    /// Number of extra decision variables per active point.
    /// Isometric: 2 (t_i, s_i per Eq. 23).  Conformal: 0.
    fn extra_vars_per_active(&self) -> usize;

    /// Append distortion constraints to the SOCP (Eq. 23/26 or 28).
    fn append_constraints(
        &self,
        state: &AlgorithmState,
        precomputed: &PrecomputedData,
        n_basis: usize,
        n_handles: usize,
        active_indices: &[usize],
        n_active: usize,
        k: f64,
        rows: &mut Vec<Vec<(usize, f64)>>,
        b: &mut Vec<f64>,
        cones: &mut Vec<clarabel::solver::SupportedConeT<f64>>,
    );

    /// Strategy 2: compute required fill distance h (Eq. 14 or 15).
    /// Returns `None` if the computation is not possible.
    fn required_h(&self, k: f64, k_max: f64, c_norm: f64, basis: &dyn BasisFunction)
        -> Option<f64>;

    /// Strategy 1: compute K_max from K and omega(h) (Eq. 11 or 13).
    /// Returns `None` if injectivity cannot be guaranteed.
    fn compute_k_max(&self, k: f64, omega_h: f64) -> Option<f64>;
}

/// Isometric distortion policy: D_iso(x) = max{Sigma(x), 1/sigma(x)}.
///
/// Constraints: Eq. 23a-c, 26.  Strategy 2: Eq. 14.
pub struct IsometricPolicy;

impl DistortionPolicy for IsometricPolicy {
    fn distortion_value(&self, sigma_max: f64, sigma_min: f64) -> f64 {
        distortion::isometric_distortion(sigma_max, sigma_min)
    }

    fn extra_vars_per_active(&self) -> usize {
        2 // t_i, s_i (Eq. 23)
    }

    fn append_constraints(
        &self,
        state: &AlgorithmState,
        precomputed: &PrecomputedData,
        n_basis: usize,
        n_handles: usize,
        active_indices: &[usize],
        n_active: usize,
        k: f64,
        rows: &mut Vec<Vec<(usize, f64)>>,
        b: &mut Vec<f64>,
        cones: &mut Vec<clarabel::solver::SupportedConeT<f64>>,
    ) {
        solver::append_isometric_constraints(
            state, precomputed, n_basis, n_handles,
            active_indices, n_active, k,
            rows, b, cones,
        );
    }

    fn required_h(&self, k: f64, k_max: f64, c_norm: f64, basis: &dyn BasisFunction)
        -> Option<f64>
    {
        strategy::required_h_isometric(k, k_max, c_norm, basis)
    }

    fn compute_k_max(&self, k: f64, omega_h: f64) -> Option<f64> {
        strategy::compute_k_max_isometric(k, omega_h)
    }
}

/// Conformal distortion policy: D_conf(x) = Sigma(x) / sigma(x).
///
/// Constraints: Eq. 28a-b.
/// Strategy functions: Phase 3 -- falls back to isometric versions.
pub struct ConformalPolicy {
    pub delta: f64,
}

impl DistortionPolicy for ConformalPolicy {
    fn distortion_value(&self, sigma_max: f64, sigma_min: f64) -> f64 {
        distortion::conformal_distortion(sigma_max, sigma_min)
    }

    fn extra_vars_per_active(&self) -> usize {
        0 // Conformal constraints (Eq. 28) need no extra variables
    }

    fn append_constraints(
        &self,
        state: &AlgorithmState,
        precomputed: &PrecomputedData,
        n_basis: usize,
        _n_handles: usize,
        active_indices: &[usize],
        _n_active: usize,
        k: f64,
        rows: &mut Vec<Vec<(usize, f64)>>,
        b: &mut Vec<f64>,
        cones: &mut Vec<clarabel::solver::SupportedConeT<f64>>,
    ) {
        solver::append_conformal_constraints(
            state, precomputed, n_basis,
            active_indices, k, self.delta,
            rows, b, cones,
        );
    }

    /// Phase 3: conformal strategy (Eq. 15) not yet implemented.
    /// Falls back to isometric required_h (Eq. 14).
    fn required_h(&self, k: f64, k_max: f64, c_norm: f64, basis: &dyn BasisFunction)
        -> Option<f64>
    {
        // Phase 3: implement Eq. 15 for conformal
        strategy::required_h_isometric(k, k_max, c_norm, basis)
    }

    /// Phase 3: conformal K_max (Eq. 13) not yet implemented.
    /// Falls back to isometric K_max (Eq. 11).
    fn compute_k_max(&self, k: f64, omega_h: f64) -> Option<f64> {
        // Phase 3: implement Eq. 13 for conformal
        strategy::compute_k_max_isometric(k, omega_h)
    }
}
