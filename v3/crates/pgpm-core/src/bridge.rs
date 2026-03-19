//! API compatibility bridge for bevy-pgpm integration
//!
//! This module provides a compatibility layer between the Phase 1 trait-based API
//! (ProvablyGoodPlanarMapping) and the interface that bevy-pgpm expects.
//!
//! The bridge allows bevy-pgpm to work with pgpm-core without requiring a major rewrite
//! of the UI layer, while maintaining the clean separation of concerns in Phase 1.

use crate::types::*;
use crate::mapping::ProvablyGoodPlanarMapping;
use crate::concrete::PGPMv2;

/// Compatibility trait that matches bevy-pgpm's expected interface
///
/// This trait wraps the new ProvablyGoodPlanarMapping API to provide
/// the interface that existing bevy-pgpm code expects.
pub trait MappingBridge: Send + Sync {
    /// Run one algorithm step with given target positions for handles
    fn step(&mut self, targets: &[Vec2]) -> Result<StepInfo>;

    /// Refine mapping using Strategy 2 (Phase 3, not yet implemented)
    fn refine_strategy2(&mut self, k_max: f64, targets: &[Vec2]) -> Result<RefinementResult>;

    /// Update algorithm parameters (K_bound, lambda_reg, etc.)
    fn update_params(&mut self, params: MappingParams);

    /// Evaluate mapping at a single point: f(point)
    fn evaluate(&self, point: Vec2) -> Vec2;

    /// Get all active set centers
    fn get_centers(&self) -> Vec<Vec2>;

    /// Get all coefficient vectors
    fn get_coefficients(&self) -> Vec<Vec2>;

    /// Get basis function scale parameter (for GPU rendering)
    fn get_basis_scale(&self) -> f64;
}

/// Step information returned after algorithm_step()
#[derive(Clone, Debug)]
pub struct StepInfo {
    /// Maximum distortion value at any point
    pub max_distortion: f64,
    /// Number of points in active set
    pub active_set_size: usize,
    /// Number of active constraints
    pub stable_set_size: usize,
    /// Algorithm has converged
    pub converged: bool,
}

/// Algorithm parameters for runtime updates
#[derive(Clone, Debug)]
pub struct MappingParams {
    /// K_high threshold for active set management
    pub k_bound: f64,
    /// Regularization weight (lambda)
    pub lambda_reg: f64,
    /// Regularization type (Isometric/Conformal, energy function)
    pub regularization: RegularizationType,
}

/// Regularization type selection
#[derive(Clone, Debug, PartialEq)]
pub enum RegularizationType {
    /// Biharmonic energy (Eq. 31)
    Biharmonic,
    /// ARAP energy (Eq. 32-33, Phase 3)
    Arap,
    /// Mixed regularization
    Mixed { lambda_bh: f64, lambda_arap: f64 },
}

impl Default for MappingParams {
    fn default() -> Self {
        Self {
            k_bound: 1.1,
            lambda_reg: 0.01,
            regularization: RegularizationType::Biharmonic,
        }
    }
}

/// Result of Strategy 2 refinement (Phase 3)
#[derive(Clone, Debug)]
pub struct RefinementResult {
    /// Refinement was successful
    pub success: bool,
    /// New K_max value after refinement
    pub new_k_max: f64,
    /// Required fill distance (Eq. 14-15)
    pub required_h: f64,
    /// Required grid resolution for verification
    pub required_resolution: usize,
    /// Number of refinement steps executed
    pub refinement_steps: usize,
    /// Actual K_max achieved
    pub k_max_achieved: f64,
    /// Coefficient vector norm
    pub c_norm: f64,
}

/// Adapter that wraps PGPMv2 with MappingBridge interface
///
/// This struct implements MappingBridge by delegating to the underlying
/// PGPMv2 instance while handling target position updates.
pub struct PGPMv2Bridge {
    /// The underlying mapping (private)
    mapping: PGPMv2,
    /// Current target positions for handles
    current_targets: Vec<Vec2>,
}

impl PGPMv2Bridge {
    /// Create a new bridge wrapping the given PGPMv2 instance
    pub fn new(mapping: PGPMv2) -> Self {
        let n_handles = mapping.get_handles().len();
        Self {
            mapping,
            current_targets: vec![Vec2::zeros(); n_handles],
        }
    }
}

impl MappingBridge for PGPMv2Bridge {
    fn step(&mut self, targets: &[Vec2]) -> Result<StepInfo> {
        // Update handle targets if they've changed
        if targets != self.current_targets {
            for (id, &target) in targets.iter().enumerate() {
                self.mapping.update_handle(id, target)?;
            }
            self.current_targets = targets.to_vec();
        }

        // Run algorithm step
        let result = self.mapping.algorithm_step()?;

        // Convert AlgorithmStepResult to StepInfo
        let active_set_info = self.mapping.get_active_set();
        let active_count = active_set_info
            .is_active
            .iter()
            .filter(|&&is_active| is_active)
            .count();

        Ok(StepInfo {
            max_distortion: result.distortion_info.max_distortion,
            active_set_size: active_set_info.centers.len(),
            stable_set_size: active_count,
            converged: result.is_converged,
        })
    }

    fn refine_strategy2(
        &mut self,
        _k_max: f64,
        _targets: &[Vec2],
    ) -> Result<RefinementResult> {
        // Phase 3: Strategy 2 not yet implemented
        Err(AlgorithmError::SolverFailed(
            "Strategy 2 refinement not yet implemented (Phase 3)".to_string(),
        ))
    }

    fn update_params(&mut self, _params: MappingParams) {
        // Phase 1 API doesn't support runtime parameter updates
        // Would require mutable access to strategy objects
        log::warn!("update_params: Not supported in Phase 1 API - create new mapping with updated parameters");
    }

    fn evaluate(&self, point: Vec2) -> Vec2 {
        self.mapping.evaluate_mapping(point)
    }

    fn get_centers(&self) -> Vec<Vec2> {
        self.mapping
            .get_handles()
            .iter()
            .map(|h| h.position)
            .collect()
    }

    fn get_coefficients(&self) -> Vec<Vec2> {
        self.mapping.get_coefficients().to_vec()
    }

    fn get_basis_scale(&self) -> f64 {
        // Phase 1: Returns 1.0 (default Gaussian scale)
        // Phase 3: Will retrieve actual scale from basis function
        1.0
    }
}
