//! ProvablyGoodPlanarMapping trait and implementations
//!
//! Core trait that defines the complete Algorithm 1 with default implementations.
//! Concrete implementations need only provide getter methods.

use crate::types::*;
use crate::strategy::*;

/// ProvablyGoodPlanarMapping trait: Algorithm 1 (Section 5)
///
/// This trait defines the complete planar mapping algorithm. Concrete implementations
/// (like PGPMv2) need only provide getter methods; the full Algorithm 1 logic is
/// implemented in default methods.
///
/// **Required Methods (getters):**
/// - Data accessors (immutable and mutable)
/// - Strategy accessors (basis function, distortion, regularization, solver)
///
/// **Default Methods (Algorithm 1 + utilities):**
/// - `algorithm_step()`: One iteration of Algorithm 1 (Section 5, Algorithm 1, steps 1-7)
/// - `evaluate_mapping()`: Compute f(x) = Σ c_i φ(||x - x_i||) (Eq. 3)
/// - `mapping_gradient()`: Compute ∇f(x)
/// - `verify_local_injectivity()`: Check det J > 0 everywhere
/// - Internal helpers for active set update, distortion computation, etc.
pub trait ProvablyGoodPlanarMapping: Send + Sync {
    // ========== Required Methods (getters only) ==========

    /// Get coefficients c = [c_1, c_2, ..., c_n] (immutable)
    fn get_coefficients(&self) -> &[Vec2];

    /// Get coefficients c (mutable)
    fn get_coefficients_mut(&mut self) -> &mut Vec<Vec2>;

    /// Get domain information
    fn get_domain(&self) -> &DomainInfo;

    /// Get active set information (immutable)
    fn get_active_set(&self) -> ActiveSetInfo;

    /// Get active set (mutable)
    fn get_active_set_mut(&mut self) -> &mut ActiveSet;

    /// Get all handles (constraint points)
    fn get_handles(&self) -> &[HandleInfo];

    /// Get basis function
    fn get_basis_function(&self) -> &dyn BasisFunction;

    /// Get distortion strategy
    fn get_distortion_strategy(&self) -> &dyn DistortionStrategy;

    /// Get regularization strategy
    fn get_regularization(&self) -> &dyn RegularizationStrategy;

    /// Get SOCP solver backend
    fn get_solver(&self) -> &dyn SOCPSolverBackend;

    /// Get algorithm state (immutable)
    fn get_algorithm_state(&self) -> &AlgorithmState;

    /// Get algorithm state (mutable)
    fn get_algorithm_state_mut(&mut self) -> &mut AlgorithmState;

    // ========== Default Methods (Algorithm 1) ==========

    /// Eq. 3 + Affine: Evaluate mapping f(x) = Σ c_i φ(||x - x_i||) + a + b·x + d·y
    ///
    /// Evaluates the RBF mapping at a single point, including affine terms.
    /// Coefficient structure: [c_1, ..., c_n, a, b, d]
    /// - c_i: RBF coefficients
    /// - a: constant term
    /// - b: x coefficient
    /// - d: y coefficient
    ///
    /// Identity mapping is achieved when c_i = 0, a = [0,0], b = [1,0], d = [0,1]
    fn evaluate_mapping(&self, point: Vec2) -> Vec2 {
        let coeffs = self.get_coefficients();
        let n_handles = self.get_handles().len();
        let mut result = Vec2::zeros();

        // RBF part: Σ c_i φ(||x - x_i||)
        for (i, handle) in self.get_handles().iter().enumerate() {
            if i >= coeffs.len() {
                break;
            }
            let diff = point - handle.position;
            let r = diff.norm();

            let phi = self.get_basis_function().evaluate(r);
            result += coeffs[i] * phi;
        }

        // Affine part: a + b·x + d·y
        // coeffs[n_handles]     = a (constant term)
        // coeffs[n_handles + 1] = b (x coefficient)
        // coeffs[n_handles + 2] = d (y coefficient)
        if coeffs.len() >= n_handles + 3 {
            result += coeffs[n_handles];                    // + a
            result += coeffs[n_handles + 1] * point.x;     // + b·x
            result += coeffs[n_handles + 2] * point.y;     // + d·y
        }

        result
    }

    /// Compute gradient ∇f(x) of the mapping (including affine part)
    ///
    /// ∇f = ∇(RBF part) + ∇(affine part)
    /// where affine part = a + b·x + d·y has Jacobian [[b_x, d_x], [b_y, d_y]]
    fn mapping_gradient(&self, point: Vec2) -> Mat2 {
        let coeffs = self.get_coefficients();
        let n_handles = self.get_handles().len();
        let mut grad = Mat2::zeros();

        // RBF part gradient: Σ c_i * φ'(r) / r * (x - x_i)^T
        for (i, handle) in self.get_handles().iter().enumerate() {
            if i >= coeffs.len() {
                break;
            }
            let diff = point - handle.position;
            let r = diff.norm();

            if r < 1e-12 {
                continue;
            }

            let phi_prime_scaled = self.get_basis_function().gradient_scaled(r);
            let c_i = coeffs[i];

            // ∂f/∂x_j = Σ c_i * φ'(r) / r * (x_j - x_{i,j})
            grad += c_i * diff.transpose() * phi_prime_scaled;
        }

        // Affine part gradient: [[b_x, d_x], [b_y, d_y]]
        // f_affine = [a_x, a_y] + [b_x, b_y]·x + [d_x, d_y]·y
        // J_affine = [[∂f_x/∂x, ∂f_x/∂y], [∂f_y/∂x, ∂f_y/∂y]]
        //          = [[b_x, d_x], [b_y, d_y]]
        if coeffs.len() >= n_handles + 3 {
            let b = coeffs[n_handles + 1]; // x coefficient vector [b_x, b_y]
            let d = coeffs[n_handles + 2]; // y coefficient vector [d_x, d_y]

            grad[(0, 0)] += b.x; // ∂f_x/∂x
            grad[(0, 1)] += d.x; // ∂f_x/∂y
            grad[(1, 0)] += b.y; // ∂f_y/∂x
            grad[(1, 1)] += d.y; // ∂f_y/∂y
        }

        grad
    }

    /// Compute det J(x) - used for fold-over detection
    fn mapping_jacobian_determinant(&self, point: Vec2) -> f64 {
        self.mapping_gradient(point).determinant()
    }

    /// Algorithm 1 main step (Section 5, Algorithm 1)
    ///
    /// Performs one complete iteration:
    /// 1. Update active set (Eq. 14-17)
    /// 2. Build SOCP problem (Eq. 18)
    /// 3. Solve SOCP
    /// 4. Update coefficients
    /// 5. Check convergence
    fn algorithm_step(&mut self) -> Result<AlgorithmStepResult> {
        // Step 1: Update active set based on distortion
        self.update_active_set_internal()?;

        // Step 2: Build SOCP problem
        let problem = self.build_socp_problem_internal()?;

        // Step 3: Solve SOCP
        let new_coeffs = self.get_solver().solve(&problem)?;
        *self.get_coefficients_mut() = new_coeffs;

        // Step 4: Compute distortion
        let distortion_info = self.compute_distortion_internal();

        // Step 5: Update state
        let is_converged = self.check_convergence_internal();
        {
            let state = self.get_algorithm_state_mut();
            state.step_counter += 1;
            state.current_distortion = distortion_info.max_distortion;
            state.is_converged = is_converged;
        }

        Ok(AlgorithmStepResult {
            step_num: self.get_algorithm_state().step_counter,
            distortion_info,
            is_converged,
        })
    }

    /// Update active set based on K_high, K_low thresholds
    /// Eq. 14-17: Activation of constraints
    fn update_active_set_internal(&mut self) -> Result<()> {
        let (k_high, k_low) = self.get_distortion_strategy().get_activation_threshold();

        // Sample domain points
        let sample_points = self.sample_domain_points();

        for point in sample_points {
            let sigma = self.compute_sigma_at_internal(point)?;

            // Activate if σ > K_high
            if sigma > k_high {
                self.get_active_set_mut().activate(point)?;
            }
            // Deactivate if σ < K_low
            else if sigma < k_low {
                self.get_active_set_mut().deactivate(point)?;
            }
        }

        Ok(())
    }

    /// Compute distortion at all sample points
    /// Eq. 10-13: Distortion formulas
    fn compute_distortion_internal(&self) -> DistortionInfo {
        let mut max_distortion: f64 = 1.0;
        let mut local_maxima = vec![];
        let (k_high, _) = self.get_distortion_strategy().get_activation_threshold();

        let sample_points = self.sample_domain_points();

        for point in sample_points {
            let jacobian = self.mapping_gradient(point);
            let sigma = self.get_distortion_strategy().compute_distortion(jacobian);

            max_distortion = max_distortion.max(sigma);

            if sigma > k_high {
                local_maxima.push((point, sigma));
            }
        }

        DistortionInfo {
            max_distortion,
            local_maxima,
        }
    }

    /// Build SOCP problem (Eq. 18)
    /// min c^T Q c  s.t.  ||A_i c + b_i|| ≤ d_i
    fn build_socp_problem_internal(&self) -> Result<SOCPProblem> {
        // Precompute basis evaluations at active set centers
        let basis_evals = self.precompute_basis_values();
        let basis_grads = self.precompute_basis_gradients();
        let basis_hessians = self.precompute_basis_hessians();

        // Build constraints from distortion strategy
        let active_set = self.get_active_set();
        let constraints = self.get_distortion_strategy().build_constraints(
            &active_set,
            &basis_evals,
            &basis_grads,
        )?;

        // Build regularization energy
        let energy = self
            .get_regularization()
            .build_energy_terms(self.get_domain(), &basis_hessians)?;

        // Build handle constraints (Eq. 29-30)
        let handle_constraints = self.build_handle_constraints_internal();

        Ok(SOCPProblem {
            constraints,
            energy,
            handle_constraints,
        })
    }

    /// Verify local injectivity (Section 3 "Fold-overs")
    ///
    /// Checks that det J(x) > 0 at all sample points.
    /// Note: This guarantees local injectivity only, not global.
    fn verify_local_injectivity(&self) -> Result<VerificationResult> {
        let mut all_positive = true;
        let mut problematic_points = vec![];

        let sample_points = self.sample_domain_points();

        for point in sample_points {
            let det = self.mapping_jacobian_determinant(point);

            if det <= 0.0 {
                all_positive = false;
                problematic_points.push(point);
            }
        }

        if all_positive {
            Ok(VerificationResult::LocallyInjective)
        } else {
            Ok(VerificationResult::HasFoldOvers(problematic_points))
        }
    }

    // ========== Helper Methods ==========

    /// Compute σ(x) at a single point
    fn compute_sigma_at_internal(&self, point: Vec2) -> Result<f64> {
        let jacobian = self.mapping_gradient(point);
        Ok(self.get_distortion_strategy().compute_distortion(jacobian))
    }

    /// Check convergence condition
    fn check_convergence_internal(&self) -> bool {
        // Simple criterion: distortion is small enough
        let state = self.get_algorithm_state();
        state.current_distortion < 1.001 || state.step_counter > 50
    }

    /// Precompute φ(r) at active set centers
    fn precompute_basis_values(&self) -> Vec<f64> {
        self.get_active_set()
            .centers
            .iter()
            .enumerate()
            .filter(|(i, _)| self.get_active_set().is_active[*i])
            .map(|(_, center)| {
                self.get_handles()
                    .iter()
                    .map(|h| {
                        let r = (center - h.position).norm();
                        self.get_basis_function().evaluate(r)
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect()
    }

    /// Precompute φ'(r)/r at active set centers
    fn precompute_basis_gradients(&self) -> Vec<Vec2> {
        self.get_active_set()
            .centers
            .iter()
            .enumerate()
            .filter(|(i, _)| self.get_active_set().is_active[*i])
            .flat_map(|(_, center)| {
                self.get_handles()
                    .iter()
                    .map(|h| {
                        let diff = center - h.position;
                        let r = diff.norm();
                        if r < 1e-12 {
                            Vec2::zeros()
                        } else {
                            let phi_prime = self.get_basis_function().gradient_scaled(r);
                            diff * phi_prime
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Precompute φ''(r)/r at active set centers
    fn precompute_basis_hessians(&self) -> Vec<f64> {
        self.get_active_set()
            .centers
            .iter()
            .enumerate()
            .filter(|(i, _)| self.get_active_set().is_active[*i])
            .flat_map(|(_, center)| {
                self.get_handles()
                    .iter()
                    .map(|h| {
                        let r = (center - h.position).norm();
                        self.get_basis_function().hessian_scaled(r)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Build handle constraints (Eq. 29-30: Position constraint energy)
    fn build_handle_constraints_internal(&self) -> Vec<HandleConstraint> {
        self.get_handles()
            .iter()
            .map(|h| HandleConstraint {
                position: h.position,
                target: h.target_value,
                weight: 1000.0, // Strong constraint
            })
            .collect()
    }

    /// Sample domain points for evaluation (Section 6: 200×200 grid)
    fn sample_domain_points(&self) -> Vec<Vec2> {
        // Phase 1: Simple rectangular grid
        // In Phase 2+: Use proper domain boundaries
        let mut points = vec![];
        let grid_res = 50; // Lower res for Phase 1

        for i in 0..grid_res {
            for j in 0..grid_res {
                let u = (i as f64) / (grid_res as f64);
                let v = (j as f64) / (grid_res as f64);

                // Assume unit square for now
                let point = Vec2::new(u, v);

                if self.get_domain().boundary.contains(point) {
                    points.push(point);
                }
            }
        }

        points
    }
}

/// Mutable active set for managing constraint points
pub struct ActiveSet {
    centers: Vec<Vec2>,
    is_active: Vec<bool>,
    sigma_values: Vec<f64>,
}

impl ActiveSet {
    /// Create new empty active set
    pub fn new() -> Self {
        Self {
            centers: vec![],
            is_active: vec![],
            sigma_values: vec![],
        }
    }

    /// Activate a constraint center
    pub fn activate(&mut self, center: Vec2) -> Result<()> {
        if let Some(idx) = self.centers.iter().position(|c| (c - center).norm() < 1e-10) {
            self.is_active[idx] = true;
        } else {
            self.centers.push(center);
            self.is_active.push(true);
            self.sigma_values.push(1.0);
        }
        Ok(())
    }

    /// Deactivate a constraint center
    pub fn deactivate(&mut self, center: Vec2) -> Result<()> {
        if let Some(idx) = self.centers.iter().position(|c| (c - center).norm() < 1e-10) {
            self.is_active[idx] = false;
        }
        Ok(())
    }

    /// Get active set info snapshot
    pub fn info(&self) -> ActiveSetInfo {
        ActiveSetInfo {
            centers: self.centers.clone(),
            is_active: self.is_active.clone(),
            sigma_values: self.sigma_values.clone(),
        }
    }

    /// Update sigma value at index
    pub fn update_sigma(&mut self, idx: usize, sigma: f64) {
        if idx < self.sigma_values.len() {
            self.sigma_values[idx] = sigma;
        }
    }
}

impl Default for ActiveSet {
    fn default() -> Self {
        Self::new()
    }
}
