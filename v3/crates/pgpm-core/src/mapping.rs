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

use crate::active_set;
use crate::basis::BasisFunction;
use crate::distortion;
use crate::strategy;
use crate::types::{
    AlgorithmError, AlgorithmState, CoefficientMatrix, DomainBounds, MappingParams,
    PrecomputedData, RegularizationType, StepInfo,
};
use log::warn;
use nalgebra::{DMatrix, Vector2};

/// Abstract definition of a provably good planar mapping (Algorithm 1, Section 5).
///
/// This trait captures the full algorithm: initialization, SOCP optimization,
/// active set management, frame updates, and Strategy 2 refinement.
/// Concrete implementations (e.g. `Algorithm<IsometricPolicy>`) inject
/// a basis function and distortion policy.
///
/// **Design**:
/// - *Required methods*: state accessors + implementation-specific hooks
///   (basis/policy-dependent operations that need borrow splitting).
/// - *Default methods*: the algorithm skeleton (Algorithm 1, Section 5)
///   and mathematical properties of f = Σ c_i φ_i (Eq. 3, Section 3).
///
/// Frontend consumers should depend on [`MappingBridge`] instead,
/// which exposes only the subset needed for UI interaction and rendering.
pub trait PlanarMapping: Send + Sync {
    // ─────────────────────────────────────────
    // Required: state accessors
    // ─────────────────────────────────────────

    /// Get current algorithm parameters.
    fn params(&self) -> &MappingParams;

    /// Get algorithm state (immutable).
    fn state(&self) -> &AlgorithmState;

    /// Get algorithm state (mutable).
    fn state_mut(&mut self) -> &mut AlgorithmState;

    /// Get domain bounds (Eq. 5).
    fn domain_bounds(&self) -> &DomainBounds;

    /// Maximum grid resolution for Strategy 2 refinement.
    fn max_refinement_resolution(&self) -> usize;

    /// Get basis function reference (Table 1).
    fn basis(&self) -> &dyn BasisFunction;

    // ─────────────────────────────────────────
    // Required: implementation-specific hooks
    // ─────────────────────────────────────────

    /// Evaluate distortion at all collocation points (policy-dependent).
    fn evaluate_all_distortions(&self) -> Result<Vec<f64>, AlgorithmError>;

    /// Solve the SOCP problem (Eq. 18) and return new coefficients.
    fn solve_socp_step(
        &self,
        targets: &[Vector2<f64>],
    ) -> Result<CoefficientMatrix, AlgorithmError>;

    /// Evaluate J_S f at all collocation points (Eq. 27 frame update).
    fn evaluate_j_s_all_points(&self) -> Result<Vec<Vector2<f64>>, AlgorithmError>;

    /// Check if targets changed since the last step, and record the new targets.
    /// Returns `true` if targets differ from the previous step.
    fn targets_changed_and_record(&mut self, targets: &[Vector2<f64>]) -> bool;

    /// Ensure the biharmonic matrix is built in `state.precomputed`.
    /// Called when the regularization type requires it (Eq. 31).
    /// No-op if already present or if precomputed data is not yet initialized.
    fn ensure_biharmonic_matrix(&mut self);

    /// Store new algorithm parameters (K, lambda, regularization).
    fn set_params(&mut self, params: MappingParams);

    // ─────────────────────────────────────────
    // Required: Strategy 2 hooks (policy-dependent)
    // ─────────────────────────────────────────

    /// Strategy 2: compute required fill distance h (Eq. 14 or 15).
    /// Returns `None` if the computation is not possible.
    fn required_h_for_strategy2(
        &self,
        k: f64,
        k_max: f64,
        c_norm: f64,
    ) -> Option<f64>;

    /// Strategy 1: compute K_max from K and omega(h) (Eq. 11 or 13).
    /// Returns `None` if injectivity cannot be guaranteed.
    fn compute_k_max_from_omega(&self, k: f64, omega_h: f64) -> Option<f64>;

    /// Rebuild the collocation grid at a new resolution (for Strategy 2).
    /// Preserves current coefficients. Resets active set, frames, precomputed data.
    fn rebuild_grid(&mut self, new_resolution: usize);

    // ─────────────────────────────────────────
    // Default methods: Algorithm 1 skeleton
    // ─────────────────────────────────────────

    /// Get current coefficient matrix c (Eq. 3).
    fn coefficients(&self) -> &CoefficientMatrix {
        &self.state().coefficients
    }

    /// Precompute basis function values and gradients at all collocation points.
    /// Algorithm 1: "if first step then" block.
    ///
    /// No-op if `state.precomputed` is already `Some`.
    ///
    /// Procedural decomposition:
    /// 1. [`Self::compute_basis_matrices`] — evaluate φ_i(z), ∇φ_i(z)
    /// 2. Store as [`PrecomputedData`]
    /// 3. [`Self::ensure_biharmonic_matrix`] — build Eq. 31 matrix if needed
    fn ensure_precomputed(&mut self) {
        if self.state().precomputed.is_some() {
            return;
        }

        let (phi, grad_phi_x, grad_phi_y) = self.compute_basis_matrices();

        self.state_mut().precomputed = Some(PrecomputedData {
            phi,
            grad_phi_x,
            grad_phi_y,
            biharmonic_matrix: None,
        });

        // Build biharmonic matrix if needed by current regularization type
        let needs_bh = matches!(
            self.params().regularization,
            RegularizationType::Biharmonic | RegularizationType::Mixed { .. }
        );
        if needs_bh {
            self.ensure_biharmonic_matrix();
        }
    }

    /// Evaluate basis function values and gradients at all collocation points.
    ///
    /// Returns (phi, grad_phi_x, grad_phi_y) matrices,
    /// each of shape (num_collocation, num_basis).
    ///
    /// NaN/Inf values (from shape-aware bases near domain boundaries) are
    /// replaced with 0.0 and a warning is logged.
    fn compute_basis_matrices(&self) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
        let state = self.state();
        let basis = self.basis();
        let m = state.collocation_points.len();
        let n = basis.count();

        let mut phi = DMatrix::zeros(m, n);
        let mut grad_phi_x = DMatrix::zeros(m, n);
        let mut grad_phi_y = DMatrix::zeros(m, n);

        let mut nan_inf_count = 0usize;
        for (idx, pt) in state.collocation_points.iter().enumerate() {
            let val = basis.evaluate(*pt);
            let (gx, gy) = basis.gradient(*pt);

            for i in 0..n {
                // Guard against NaN/Inf from basis functions (can happen
                // with shape-aware bases near domain boundaries where
                // geodesic distances are infinite).
                if !val[i].is_finite() || !gx[i].is_finite() || !gy[i].is_finite() {
                    nan_inf_count += 1;
                }
                phi[(idx, i)] = if val[i].is_finite() { val[i] } else { 0.0 };
                grad_phi_x[(idx, i)] = if gx[i].is_finite() { gx[i] } else { 0.0 };
                grad_phi_y[(idx, i)] = if gy[i].is_finite() { gy[i] } else { 0.0 };
            }
        }
        if nan_inf_count > 0 {
            warn!(
                "Precompute: {} NaN/Inf basis values replaced with 0.0 \
                 (expected near domain boundaries with shape-aware bases)",
                nan_inf_count,
            );
        }

        (phi, grad_phi_x, grad_phi_y)
    }

    /// Algorithm 1: execute one step (Section 5).
    ///
    /// Pseudocode correspondence:
    /// 1. [first step only] Precompute phi(z), grad_phi(z)
    /// 2. Evaluate D(z) for all z in Z
    /// 3. Find Z_max (local maxima of D)
    /// 4. Add z in Z_max with D(z) > K_high to Z'
    /// 5. Remove z in Z' with D(z) < K_low from Z'
    /// 6. Solve SOCP (Eq. 18) -> update c
    /// 7. Update d_i (Eq. 27)
    fn step(&mut self, target_handles: &[Vector2<f64>]) -> Result<StepInfo, AlgorithmError> {
        // === Initialization (if first step) ===
        self.ensure_precomputed();

        // === Evaluate distortion at all collocation points ===
        let distortions = self.evaluate_all_distortions()?;

        // === Update active set (Algorithm 1 lines 5-8) ===
        let prev_active_set = self.state().active_set.clone();
        active_set::update_active_set(self.state_mut(), &distortions);

        let max_distortion = distortions
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let n_active = self.state().active_set.len();

        // Algorithm 1 convergence: distortion within bound, active set stable,
        // and targets unchanged since last step (so the SOCP output from the
        // previous step has been verified by this step's distortion evaluation).
        let targets_changed = self.targets_changed_and_record(target_handles);
        let converged = !targets_changed
            && max_distortion <= self.params().k_bound
            && self.state().active_set == prev_active_set;

        // === Solve SOCP (Eq. 18) ===
        let new_coefficients = self.solve_socp_step(target_handles)?;
        self.state_mut().coefficients = new_coefficients;

        // === Update frames (Eq. 27) ===
        // d_i = J_S f(z_i) / ||J_S f(z_i)||
        //
        // Frames are updated only for constraint points (Z' union Z''),
        // consistent with Algorithm 1's postprocessing scope.
        // Frames at non-constraint points remain at their last value
        // (initialized to (1,0)).  This does not affect fold-over
        // guarantees -- see Section 5 "Initialization of the frames".
        let j_s_values = self.evaluate_j_s_all_points()?;
        {
            let state = self.state_mut();
            let eps = 1e-10;
            // Collect indices to avoid borrow conflict on state
            let indices: Vec<usize> = state
                .active_set
                .iter()
                .chain(state.stable_set.iter())
                .copied()
                .collect();
            for idx in indices {
                let j_s = j_s_values[idx];
                let norm = j_s.norm();
                if norm > eps {
                    // Eq. 27: d_i = J_S f(x_i) / ||J_S f(x_i)||
                    state.frames[idx] = j_s / norm;
                }
            }
        }

        Ok(StepInfo {
            max_distortion,
            active_set_size: n_active,
            stable_set_size: self.state().stable_set.len(),
            converged,
        })
    }

    /// Update algorithm parameters at runtime (K, lambda, regularization).
    ///
    /// K_high and K_low are re-derived from the new K bound.
    /// The active set is NOT cleared -- stale points will be naturally
    /// removed at the next step when their distortion falls below K_low.
    fn update_params(&mut self, params: MappingParams) {
        // Section 5 "Activation of constraints":
        // K_high = 0.1 + 0.9*K, K_low = 0.5 + 0.5*K
        let k = params.k_bound;
        {
            let state = self.state_mut();
            state.k_high = 0.1 + 0.9 * k;
            state.k_low = 0.5 + 0.5 * k;
        }

        // If switching to a regularization that needs biharmonic matrix,
        // ensure it's computed.
        let needs_bh = matches!(
            params.regularization,
            RegularizationType::Biharmonic | RegularizationType::Mixed { .. }
        );
        let has_bh = self
            .state()
            .precomputed
            .as_ref()
            .map_or(false, |p| p.biharmonic_matrix.is_some());

        if needs_bh && !has_bh {
            self.ensure_biharmonic_matrix();
        }

        self.set_params(params);
    }

    /// Strategy 2 post-hoc refinement (Section 5 "Strategies").
    ///
    /// During interactive manipulation, Algorithm 1 runs on a fixed
    /// coarse grid for responsiveness. Once manipulation ends, this
    /// method refines the grid to guarantee K_max everywhere:
    ///
    /// 1. Compute |||c||| from current coefficients (Eq. 8)
    /// 2. Compute required fill distance h via policy (Eq. 14/15)
    /// 3. Determine grid resolution to achieve h
    /// 4. Rebuild collocation grid and precomputed data at higher resolution
    ///    (coefficients c are preserved as the initial guess)
    /// 5. Run Algorithm 1 steps until convergence or step limit
    ///
    /// Convergence: max_distortion <= K and active set unchanged from previous step.
    /// Step limit: `strategy::MAX_REFINEMENT_STEPS`.
    fn refine_strategy2(
        &mut self,
        k_max: f64,
        target_handles: &[Vector2<f64>],
    ) -> Result<strategy::Strategy2Result, AlgorithmError> {
        let k = self.params().k_bound;
        if k_max <= k {
            return Err(AlgorithmError::InvalidInput(format!(
                "Strategy 2 requires K_max ({}) > K ({})",
                k_max, k
            )));
        }

        // Step 1: compute |||c||| (Eq. 8)
        let c_norm = strategy::compute_c_norm(self.coefficients());

        // Step 2: compute required h (delegated to policy)
        let required_h = self
            .required_h_for_strategy2(k, k_max, c_norm)
            .ok_or_else(|| {
                AlgorithmError::InvalidInput(
                    "Strategy 2: cannot compute required h (K_max too close to K or c_norm issue)"
                        .into(),
                )
            })?;

        // Current fill distance
        let current_h = strategy::fill_distance(
            self.domain_bounds(),
            self.state().grid_width,
        );

        // Step 3: determine required resolution
        let new_resolution = strategy::resolution_for_h(self.domain_bounds(), required_h);
        // Check against configured maximum
        let max_res = self.max_refinement_resolution();
        if new_resolution > max_res {
            return Err(AlgorithmError::ResolutionExceeded {
                required: new_resolution,
                max: max_res,
            });
        }

        // Only rebuild if we need a denser grid
        if new_resolution > self.state().grid_width {
            // Step 4: rebuild collocation grid at higher resolution
            self.rebuild_grid(new_resolution);
        }

        // Step 5: run Algorithm 1 steps until convergence
        let mut refinement_steps = 0;

        for _ in 0..strategy::MAX_REFINEMENT_STEPS {
            let info = self.step(target_handles)?;
            refinement_steps += 1;

            if info.converged {
                break;
            }
        }

        // Compute achieved K_max (delegated to policy)
        let final_h = strategy::fill_distance(
            self.domain_bounds(),
            self.state().grid_width,
        );
        let final_omega = strategy::omega(final_h, c_norm, self.basis());
        let k_max_achieved = self.compute_k_max_from_omega(k, final_omega).unwrap_or_else(|| {
            warn!(
                "Strategy 2: cannot guarantee finite K_max \
                 (omega(h)={:.4} >= 1/K={:.4})",
                final_omega,
                1.0 / k,
            );
            f64::INFINITY
        });

        Ok(strategy::Strategy2Result {
            required_h,
            required_resolution: new_resolution,
            current_h,
            k_max_achieved,
            c_norm,
            refinement_steps,
        })
    }

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
