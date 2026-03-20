//! Frontend bridge trait.

use crate::algorithm::strategy;
use crate::basis::BasisFunction;
use crate::mapping::PlanarMapping;
use crate::model::types::{AlgorithmError, CoefficientMatrix, DomainBounds, MappingParams, StepInfo};
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

    /// Update algorithm parameters at runtime (K, lambda, regularization).
    fn update_params(&mut self, params: MappingParams);

    /// Strategy 2 post-hoc refinement (Section 5 "Strategies").
    fn refine_strategy2(
        &mut self,
        k_max: f64,
        target_handles: &[Vector2<f64>],
    ) -> Result<strategy::Strategy2Result, AlgorithmError>;

    // ─────────────────────────────────────────
    // Query methods: read-only state inspection
    // ─────────────────────────────────────────

    /// Get current algorithm parameters (K, lambda, regularization).
    fn params(&self) -> MappingParams;

    /// Grid resolution (width, height) for collocation grid (Section 4).
    fn grid_resolution(&self) -> (usize, usize);

    /// Total number of collocation points |Z| (Section 4).
    fn num_collocation_points(&self) -> usize;

    /// Number of basis functions n (Table 1).
    fn num_basis_functions(&self) -> usize;

    /// Source handle positions {p_l} (Eq. 29).
    fn source_handles(&self) -> Vec<Vector2<f64>>;

    /// Bounding box of domain Omega (Eq. 5).
    fn domain_bounds(&self) -> DomainBounds;
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

    fn params(&self) -> MappingParams {
        PlanarMapping::params(self)
    }

    fn grid_resolution(&self) -> (usize, usize) {
        PlanarMapping::grid_resolution(self)
    }

    fn num_collocation_points(&self) -> usize {
        PlanarMapping::num_collocation_points(self)
    }

    fn num_basis_functions(&self) -> usize {
        PlanarMapping::num_basis_functions(self)
    }

    fn source_handles(&self) -> Vec<Vector2<f64>> {
        PlanarMapping::source_handles(self)
    }

    fn domain_bounds(&self) -> DomainBounds {
        PlanarMapping::domain_bounds_query(self)
    }
}
