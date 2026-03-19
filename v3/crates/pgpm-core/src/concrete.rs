//! PGPMv2: Concrete implementation of ProvablyGoodPlanarMapping
//!
//! PGPMv2 implements only the getter methods required by the trait.
//! All algorithm logic is inherited from the trait's default implementations.

use crate::mapping::{ProvablyGoodPlanarMapping, ActiveSet};
use crate::types::*;
use crate::strategy::*;

/// PGPMv2: Complete planar mapping implementation
///
/// This struct manages all state (domain, handles, coefficients, active set)
/// and holds references to injected strategy implementations.
///
/// Algorithm 1 logic is entirely inherited from the ProvablyGoodPlanarMapping trait;
/// this struct only stores data and provides getter access.
pub struct PGPMv2 {
    // State
    domain: DomainInfo,
    handles: Vec<HandleInfo>,
    active_set: ActiveSet,
    coefficients: Vec<Vec2>,
    algorithm_state: AlgorithmState,

    // Injected strategies
    basis_fn: Box<dyn BasisFunction>,
    distortion: Box<dyn DistortionStrategy>,
    regularization: Box<dyn RegularizationStrategy>,
    solver: Box<dyn SOCPSolverBackend>,
}

impl PGPMv2 {
    /// Create new mapping with injected strategies
    pub fn new(
        domain: DomainInfo,
        basis_fn: Box<dyn BasisFunction>,
        distortion: Box<dyn DistortionStrategy>,
        regularization: Box<dyn RegularizationStrategy>,
        solver: Box<dyn SOCPSolverBackend>,
    ) -> Self {
        Self {
            domain,
            handles: vec![],
            active_set: ActiveSet::new(),
            coefficients: vec![],
            algorithm_state: AlgorithmState::default(),
            basis_fn,
            distortion,
            regularization,
            solver,
        }
    }

    /// Add a new handle (constraint point)
    pub fn add_handle(
        &mut self,
        id: HandleId,
        position: Vec2,
        target_value: Vec2,
    ) -> Result<()> {
        self.handles.push(HandleInfo {
            id,
            position,
            target_value,
            radius: 1.0, // Default radius
        });

        // Rebuild coefficients: RBF terms (zeros) + affine terms (identity)
        self.rebuild_coefficients_with_identity();

        Ok(())
    }

    /// Update an existing handle's target value
    pub fn update_handle(&mut self, id: HandleId, target_value: Vec2) -> Result<()> {
        for handle in self.handles.iter_mut() {
            if handle.id == id {
                handle.target_value = target_value;
                return Ok(());
            }
        }
        Err(AlgorithmError::InvalidHandle(format!(
            "Handle {} not found",
            id
        )))
    }

    /// Remove a handle
    pub fn remove_handle(&mut self, id: HandleId) -> Result<()> {
        if let Some(idx) = self.handles.iter().position(|h| h.id == id) {
            self.handles.remove(idx);
            // Rebuild coefficients with identity mapping
            self.rebuild_coefficients_with_identity();
            Ok(())
        } else {
            Err(AlgorithmError::InvalidHandle(format!(
                "Handle {} not found",
                id
            )))
        }
    }

    /// Reset algorithm state (clear coefficients, step counter, etc.)
    pub fn reset(&mut self) -> Result<()> {
        self.rebuild_coefficients_with_identity();
        self.algorithm_state = AlgorithmState::default();
        self.active_set = ActiveSet::new();
        Ok(())
    }

    /// Rebuild coefficient vector with identity mapping initialization
    ///
    /// Structure: [c_1, c_2, ..., c_n, a, b, d]
    /// - c_i: RBF coefficients (initialized to zero)
    /// - a: constant term [0, 0]
    /// - b: x coefficient [1, 0]  (for f_x(x,y) = x)
    /// - d: y coefficient [0, 1]  (for f_y(x,y) = y)
    ///
    /// This produces identity mapping: f(x) = x
    fn rebuild_coefficients_with_identity(&mut self) {
        let n_handles = self.handles.len();

        // RBF coefficients: all zeros
        let mut coeffs = vec![Vec2::zeros(); n_handles];

        // Affine terms for identity mapping: f(x) = 0 + [1,0]*x + [0,1]*y = x
        coeffs.push(Vec2::new(0.0, 0.0)); // a: constant term
        coeffs.push(Vec2::new(1.0, 0.0)); // b: x coefficient
        coeffs.push(Vec2::new(0.0, 1.0)); // d: y coefficient

        self.coefficients = coeffs;
    }

    /// Get current distortion info
    pub fn get_current_distortion(&self) -> f64 {
        self.algorithm_state.current_distortion
    }

    /// Get step counter
    pub fn step_count(&self) -> usize {
        self.algorithm_state.step_counter
    }

    /// Check if converged
    pub fn is_converged(&self) -> bool {
        self.algorithm_state.is_converged
    }
}

// ========== Implement ProvablyGoodPlanarMapping Trait ==========
// Only getters! All algorithm logic is in the trait.

impl ProvablyGoodPlanarMapping for PGPMv2 {
    fn get_coefficients(&self) -> &[Vec2] {
        &self.coefficients
    }

    fn get_coefficients_mut(&mut self) -> &mut Vec<Vec2> {
        &mut self.coefficients
    }

    fn get_domain(&self) -> &DomainInfo {
        &self.domain
    }

    fn get_active_set(&self) -> ActiveSetInfo {
        self.active_set.info()
    }

    fn get_active_set_mut(&mut self) -> &mut crate::mapping::ActiveSet {
        &mut self.active_set
    }

    fn get_handles(&self) -> &[HandleInfo] {
        &self.handles
    }

    fn get_basis_function(&self) -> &dyn BasisFunction {
        self.basis_fn.as_ref()
    }

    fn get_distortion_strategy(&self) -> &dyn DistortionStrategy {
        self.distortion.as_ref()
    }

    fn get_regularization(&self) -> &dyn RegularizationStrategy {
        self.regularization.as_ref()
    }

    fn get_solver(&self) -> &dyn SOCPSolverBackend {
        self.solver.as_ref()
    }

    fn get_algorithm_state(&self) -> &AlgorithmState {
        &self.algorithm_state
    }

    fn get_algorithm_state_mut(&mut self) -> &mut AlgorithmState {
        &mut self.algorithm_state
    }
}
