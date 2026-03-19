//! Frontend bridge trait.

use crate::algorithm::strategy;
use crate::basis::BasisFunction;
use crate::mapping::PlanarMapping;
use crate::model::types::{AlgorithmError, CoefficientMatrix, MappingParams, StepInfo};
use nalgebra::Vector2;

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
    ) -> Result<strategy::Strategy2Result, AlgorithmError>;
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
    ) -> Result<strategy::Strategy2Result, AlgorithmError> {
        PlanarMapping::refine_strategy2(self, k_max, target_handles)
    }
}
