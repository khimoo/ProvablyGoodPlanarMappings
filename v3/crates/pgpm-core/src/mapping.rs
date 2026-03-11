//! Planar mapping traits.
//!
//! Two traits define the boundary between pgpm-core and its consumers:
//!
//! - [`PlanarMapping`]: Abstract definition of the full algorithm
//!   (Algorithm 1, Section 5). Includes both the mathematical evaluation
//!   (Eq. 3, Section 3) and the optimization procedure (SOCP, active set,
//!   Strategy 2). This is the "what is a provably good planar mapping"
//!   trait — analogous to v2's `ProvablyGoodPlanarMapping` abstract class.
//!
//! - [`MappingBridge`]: Subset of `PlanarMapping` exposed to frontend
//!   consumers (e.g. bevy-pgpm). Hides internal methods like `grad_uv_at`,
//!   `j_s_j_a_at`, `singular_values_at` that are used only by the algorithm
//!   internals. A blanket impl provides `MappingBridge` for any `PlanarMapping`.

use crate::basis::BasisFunction;
use crate::distortion;
use crate::strategy::Strategy2Result;
use crate::types::{AlgorithmError, CoefficientMatrix, MappingParams, StepInfo};
use nalgebra::Vector2;

/// Abstract definition of a provably good planar mapping (Algorithm 1, Section 5).
///
/// This trait captures the full algorithm: initialization, SOCP optimization,
/// active set management, frame updates, and Strategy 2 refinement.
/// Concrete implementations (e.g. `Algorithm<IsometricPolicy>`) inject
/// a basis function and distortion policy.
///
/// Default methods implement the mathematical properties of the
/// mapping f = Σ c_i φ_i (Eq. 3, Section 3) that are determined
/// solely by `coefficients()` and `basis()`.
///
/// Frontend consumers should depend on [`MappingBridge`] instead,
/// which exposes only the subset needed for UI interaction and rendering.
pub trait PlanarMapping: Send + Sync {
    // ─────────────────────────────────────────
    // Required methods (Algorithm-specific)
    // ─────────────────────────────────────────

    /// Algorithm 1: execute one step (Section 5).
    fn step(&mut self, target_handles: &[Vector2<f64>]) -> Result<StepInfo, AlgorithmError>;

    /// Get current coefficient matrix c (Eq. 3).
    fn coefficients(&self) -> &CoefficientMatrix;

    /// Get basis function reference (Table 1).
    fn basis(&self) -> &dyn BasisFunction;

    /// Update algorithm parameters at runtime (K, lambda, regularization).
    fn update_params(&mut self, params: MappingParams);

    /// Strategy 2 post-hoc refinement (Section 5 "Strategies").
    ///
    /// During interactive manipulation, Algorithm 1 runs on a fixed
    /// coarse grid for responsiveness. Once manipulation ends, this
    /// method refines the grid resolution so that the fill distance h
    /// satisfies Eq. 14 (isometric) or Eq. 15 (conformal), guaranteeing
    /// the distortion upper bound K_max everywhere in the domain.
    fn refine_strategy2(
        &mut self,
        k_max: f64,
        target_handles: &[Vector2<f64>],
    ) -> Result<Strategy2Result, AlgorithmError>;

    // ─────────────────────────────────────────
    // Default methods (Eq. 3 / Section 3)
    //
    // These are pure mathematical properties of the mapping f,
    // fully determined by coefficients and basis.
    // ─────────────────────────────────────────

    /// Evaluate the mapping f(x) = Σ c_i φ_i(x) (Eq. 3).
    fn evaluate(&self, x: Vector2<f64>) -> Vector2<f64> {
        let phi = self.basis().evaluate(x);
        let c = self.coefficients();
        let n = self.basis().count();
        let mut u = 0.0;
        let mut v = 0.0;
        for i in 0..n {
            u += c[(0, i)] * phi[i];
            v += c[(1, i)] * phi[i];
        }
        Vector2::new(u, v)
    }

    /// Compute the Jacobian gradients (∇u, ∇v) at point x (Eq. 3 differentiated).
    ///
    /// ∇u(x) = Σ c¹_i ∇φ_i(x),  ∇v(x) = Σ c²_i ∇φ_i(x)
    fn grad_uv_at(&self, x: Vector2<f64>) -> (Vector2<f64>, Vector2<f64>) {
        let (gx, gy) = self.basis().gradient(x);
        let c = self.coefficients();
        let n = self.basis().count();

        let mut grad_u = Vector2::new(0.0, 0.0);
        let mut grad_v = Vector2::new(0.0, 0.0);
        for i in 0..n {
            grad_u.x += c[(0, i)] * gx[i];
            grad_u.y += c[(0, i)] * gy[i];
            grad_v.x += c[(1, i)] * gx[i];
            grad_v.y += c[(1, i)] * gy[i];
        }
        (grad_u, grad_v)
    }

    /// Compute J_S f(x) and J_A f(x) at point x (Eq. 19-20).
    ///
    /// J_S f = (∇u + I∇v) / 2  (similarity part)
    /// J_A f = (∇u - I∇v) / 2  (anti-similarity part)
    fn j_s_j_a_at(&self, x: Vector2<f64>) -> (Vector2<f64>, Vector2<f64>) {
        let (grad_u, grad_v) = self.grad_uv_at(x);
        distortion::compute_j_s_j_a(grad_u, grad_v)
    }

    /// Compute singular values (Σ, σ) at point x (Eq. 20).
    ///
    /// Σ(x) = ||J_S f(x)|| + ||J_A f(x)||
    /// σ(x) = | ||J_S f(x)|| - ||J_A f(x)|| |
    fn singular_values_at(&self, x: Vector2<f64>) -> (f64, f64) {
        let (j_s, j_a) = self.j_s_j_a_at(x);
        distortion::singular_values(j_s, j_a)
    }
}

// ─────────────────────────────────────────────────────────────
// MappingBridge: frontend communication trait
// ─────────────────────────────────────────────────────────────

/// Frontend bridge: subset of [`PlanarMapping`] exposed to UI consumers.
///
/// `bevy-pgpm` depends only on this trait, not on `PlanarMapping` directly.
/// Internal methods (`grad_uv_at`, `j_s_j_a_at`, `singular_values_at`) are
/// used by the algorithm internals and are not part of this interface.
pub trait MappingBridge: Send + Sync {
    /// Algorithm 1: execute one step (Section 5).
    fn step(&mut self, target_handles: &[Vector2<f64>]) -> Result<StepInfo, AlgorithmError>;

    /// Get current coefficient matrix c (Eq. 3). Used by GPU rendering path.
    fn coefficients(&self) -> &CoefficientMatrix;

    /// Get basis function reference (Table 1). Used by GPU rendering path.
    fn basis(&self) -> &dyn BasisFunction;

    /// Evaluate the mapping f(x) (Eq. 3). Used by CPU rendering path.
    fn evaluate(&self, x: Vector2<f64>) -> Vector2<f64>;

    /// Update algorithm parameters at runtime (K, lambda, regularization).
    fn update_params(&mut self, params: MappingParams);

    /// Strategy 2 post-hoc refinement (Section 5 "Strategies").
    fn refine_strategy2(
        &mut self,
        k_max: f64,
        target_handles: &[Vector2<f64>],
    ) -> Result<Strategy2Result, AlgorithmError>;
}

/// Blanket impl: any `PlanarMapping` automatically satisfies `MappingBridge`.
impl<T: PlanarMapping + ?Sized> MappingBridge for T {
    fn step(&mut self, target_handles: &[Vector2<f64>]) -> Result<StepInfo, AlgorithmError> {
        PlanarMapping::step(self, target_handles)
    }

    fn coefficients(&self) -> &CoefficientMatrix {
        PlanarMapping::coefficients(self)
    }

    fn basis(&self) -> &dyn BasisFunction {
        PlanarMapping::basis(self)
    }

    fn evaluate(&self, x: Vector2<f64>) -> Vector2<f64> {
        PlanarMapping::evaluate(self, x)
    }

    fn update_params(&mut self, params: MappingParams) {
        PlanarMapping::update_params(self, params)
    }

    fn refine_strategy2(
        &mut self,
        k_max: f64,
        target_handles: &[Vector2<f64>],
    ) -> Result<Strategy2Result, AlgorithmError> {
        PlanarMapping::refine_strategy2(self, k_max, target_handles)
    }
}
