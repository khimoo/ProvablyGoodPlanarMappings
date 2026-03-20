//! Planar mapping trait.
//!
//! [`PlanarMapping`] is the abstract definition of the full algorithm
//! (Algorithm 1, Section 5). It includes both the mathematical evaluation
//! (Eq. 3, Section 3) and the optimization procedure (SOCP, active set,
//! Strategy 2).
//!
//! **Design**: The trait uses a `parts()`/`parts_mut()` pattern to split
//! borrows between immutable context ([`MappingContext`]) and mutable state
//! ([`AlgorithmState`]).  This allows *all* Algorithm 1 logic to live in
//! default methods while concrete types only provide the borrow-splitting
//! accessors and `set_params`.

use crate::algorithm::active_set;
use crate::algorithm::strategy;
use crate::basis::BasisFunction;
use crate::distortion;
use crate::model::types::{
    AlgorithmError, AlgorithmState, CoefficientMatrix, DomainBounds, MappingContext,
    MappingParams, PrecomputedData, RegularizationType, StepInfo,
};
use crate::numerics::solver;
use log::warn;
use nalgebra::{DMatrix, Vector2};

/// Abstract definition of a provably good planar mapping (Algorithm 1, Section 5).
///
/// This trait captures the full algorithm: initialization, SOCP optimization,
/// active set management, frame updates, and Strategy 2 refinement.
///
/// **Required methods** (3 only):
/// - [`parts`](PlanarMapping::parts) / [`parts_mut`](PlanarMapping::parts_mut) —
///   borrow-splitting accessors that return `(MappingContext, &AlgorithmState)`
///   or `(MappingContext, &mut AlgorithmState)`.
/// - [`set_params`](PlanarMapping::set_params) — update algorithm parameters.
///
/// **Default methods**: the complete Algorithm 1 skeleton and
/// mathematical evaluation (Eq. 3).
pub trait PlanarMapping: Send + Sync {
    // ─────────────────────────────────────────
    // Required: borrow-splitting accessors
    // ─────────────────────────────────────────

    /// Split self into immutable context and immutable state.
    fn parts(&self) -> (MappingContext<'_>, &AlgorithmState);

    /// Split self into immutable context and mutable state.
    fn parts_mut(&mut self) -> (MappingContext<'_>, &mut AlgorithmState);

    /// Store new algorithm parameters (K, lambda, regularization).
    /// This mutates the params field which lives outside AlgorithmState.
    fn set_params(&mut self, params: MappingParams);

    // ─────────────────────────────────────────
    // Default methods: Algorithm 1 skeleton
    // ─────────────────────────────────────────

    /// Get current coefficient matrix c (Eq. 3).
    fn coefficients(&self) -> &CoefficientMatrix {
        let (_, state) = self.parts();
        &state.coefficients
    }

    /// Get basis function reference (Table 1).
    fn basis(&self) -> &dyn BasisFunction {
        let (ctx, _) = self.parts();
        ctx.basis
    }

    /// Get current algorithm parameters.
    fn params(&self) -> MappingParams {
        let (ctx, _) = self.parts();
        ctx.params.clone()
    }

    /// Get algorithm state (immutable) for external inspection.
    fn state(&self) -> &AlgorithmState {
        let (_, state) = self.parts();
        state
    }

    /// Maximum grid resolution for Strategy 2 refinement.
    fn max_refinement_resolution(&self) -> usize {
        let (ctx, _) = self.parts();
        ctx.solver_config.max_refinement_resolution
    }

    /// Grid resolution (width, height) for collocation grid (Section 4).
    fn grid_resolution(&self) -> (usize, usize) {
        let (_, state) = self.parts();
        (state.grid_width, state.grid_height)
    }

    /// Number of collocation points |Z| (Section 4).
    fn num_collocation_points(&self) -> usize {
        let (_, state) = self.parts();
        state.collocation_points.len()
    }

    /// Number of basis functions n (Table 1).
    fn num_basis_functions(&self) -> usize {
        let (ctx, _) = self.parts();
        ctx.basis.count()
    }

    /// Source handle positions {p_l} (Eq. 29).
    fn source_handles(&self) -> Vec<Vector2<f64>> {
        let (ctx, _) = self.parts();
        ctx.source_handles.to_vec()
    }

    /// Bounding box of domain Omega (Eq. 5).
    fn domain_bounds_query(&self) -> DomainBounds {
        let (ctx, _) = self.parts();
        ctx.domain_bounds.clone()
    }

    /// Precompute basis function values and gradients at all collocation points.
    /// Algorithm 1: "if first step then" block.
    ///
    /// No-op if `state.precomputed` is already `Some`.
    ///
    /// Procedural decomposition:
    /// 1. [`Self::compute_basis_matrices`] — evaluate φ_i(z), ∇φ_i(z)
    /// 2. Store as [`PrecomputedData`]
    /// 3. Build Eq. 31 biharmonic matrix if needed
    fn ensure_precomputed(&mut self) {
        {
            let (_, state) = self.parts();
            if state.precomputed.is_some() {
                return;
            }
        }

        let (phi, grad_phi_x, grad_phi_y) = self.compute_basis_matrices();

        let (ctx, state) = self.parts_mut();
        state.precomputed = Some(PrecomputedData {
            phi,
            grad_phi_x,
            grad_phi_y,
            biharmonic_matrix: None,
        });

        // Build biharmonic matrix if needed by current regularization type
        let needs_bh = matches!(
            ctx.params.regularization,
            RegularizationType::Biharmonic | RegularizationType::Mixed { .. }
        );
        if needs_bh {
            ensure_biharmonic_matrix_inner(ctx.basis, ctx.domain_bounds, state);
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
        let (ctx, state) = self.parts();
        let m = state.collocation_points.len();
        let n = ctx.basis.count();

        let mut phi = DMatrix::zeros(m, n);
        let mut grad_phi_x = DMatrix::zeros(m, n);
        let mut grad_phi_y = DMatrix::zeros(m, n);

        let mut nan_inf_count = 0usize;
        for (idx, pt) in state.collocation_points.iter().enumerate() {
            let val = ctx.basis.evaluate(*pt);
            let (gx, gy) = ctx.basis.gradient(*pt);

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
        self.ensure_precomputed();

        // 2. Evaluate distortions (Eq. 19-20)
        let distortions = {
            let (ctx, state) = self.parts();
            let precomputed = state.precomputed.as_ref().ok_or_else(|| {
                AlgorithmError::InvalidInput("Precomputed data not available (bug)".into())
            })?;
            distortion::evaluate_distortion_all(
                &state.coefficients,
                precomputed,
                ctx.policy,
            )
        };

        // 3-5. Update active set
        let prev_active_set = {
            let (_, state) = self.parts();
            state.active_set.clone()
        };
        {
            let (_, state) = self.parts_mut();
            active_set::update_active_set(state, &distortions);
        }

        let max_distortion = distortions
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Check convergence
        let (n_active, converged) = {
            let (ctx, state) = self.parts_mut();
            let n_active = state.active_set.len();

            let changed = state.prev_target_handles.as_deref() != Some(target_handles);
            state.prev_target_handles = Some(target_handles.to_vec());

            let converged = !changed
                && max_distortion <= ctx.params.k_bound
                && state.active_set == prev_active_set;

            (n_active, converged)
        };

        // 6. Solve SOCP (Eq. 18)
        let new_coefficients = {
            let (ctx, state) = self.parts();
            state.precomputed.as_ref().ok_or_else(|| {
                AlgorithmError::InvalidInput("Precomputed data not available (bug)".into())
            })?;
            solver::solve_socp(
                ctx.source_handles,
                target_handles,
                ctx.basis,
                state,
                ctx.params,
                ctx.policy,
                ctx.solver_config,
            )?
        };

        // 7. Update coefficients and frames (Eq. 27)
        {
            let (_, state) = self.parts_mut();
            state.coefficients = new_coefficients;
        }

        // Evaluate J_S at all points for frame update (Eq. 27)
        let j_s_values = {
            let (_, state) = self.parts();
            let pre = state.precomputed.as_ref().ok_or_else(|| {
                AlgorithmError::InvalidInput("Precomputed data not available (bug)".into())
            })?;
            distortion::evaluate_j_s_all(&state.coefficients, pre)
        };

        let stable_set_size = {
            let (_, state) = self.parts_mut();
            let eps = 1e-10;
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
                    state.frames[idx] = j_s / norm;
                }
            }
            state.stable_set.len()
        };

        Ok(StepInfo {
            max_distortion,
            active_set_size: n_active,
            stable_set_size,
            converged,
        })
    }

    /// Update algorithm parameters at runtime (K, lambda, regularization).
    fn update_params(&mut self, params: MappingParams) {
        let k = params.k_bound;
        {
            let (_, state) = self.parts_mut();
            state.k_high = 0.1 + 0.9 * k;
            state.k_low = 0.5 + 0.5 * k;
        }

        let (needs_bh, has_bh) = {
            let (_, state) = self.parts();
            let needs_bh = matches!(
                params.regularization,
                RegularizationType::Biharmonic | RegularizationType::Mixed { .. }
            );
            let has_bh = state
                .precomputed
                .as_ref()
                .map_or(false, |p| p.biharmonic_matrix.is_some());
            (needs_bh, has_bh)
        };

        if needs_bh && !has_bh {
            let (ctx, state) = self.parts_mut();
            ensure_biharmonic_matrix_inner(ctx.basis, ctx.domain_bounds, state);
        }

        self.set_params(params);
    }

    /// Strategy 2 post-hoc refinement (Section 5 "Strategies").
    fn refine_strategy2(
        &mut self,
        k_max: f64,
        target_handles: &[Vector2<f64>],
    ) -> Result<strategy::Strategy2Result, AlgorithmError> {
        let (k, c_norm, required_h, current_h) = {
            let (ctx, state) = self.parts();
            let k = ctx.params.k_bound;
            if k_max <= k {
                return Err(AlgorithmError::InvalidInput(format!(
                    "Strategy 2 requires K_max ({}) > K ({})",
                    k_max, k
                )));
            }

            let c_norm = strategy::compute_c_norm(&state.coefficients);
            let required_h = ctx.policy
                .required_h(k, k_max, c_norm, ctx.basis)
                .ok_or_else(|| {
                    AlgorithmError::InvalidInput(
                        "Strategy 2: cannot compute required h (K_max too close to K or c_norm issue)"
                            .into(),
                    )
                })?;
            let current_h = strategy::fill_distance(ctx.domain_bounds, state.grid_width);
            (k, c_norm, required_h, current_h)
        };

        let new_resolution = {
            let (ctx, _state) = self.parts();
            let new_resolution = strategy::resolution_for_h(ctx.domain_bounds, required_h);
            let max_res = ctx.solver_config.max_refinement_resolution;
            if new_resolution > max_res {
                return Err(AlgorithmError::ResolutionExceeded {
                    required: new_resolution,
                    max: max_res,
                });
            }
            new_resolution
        };

        // Rebuild grid if needed
        let needs_rebuild = {
            let (_, state) = self.parts();
            new_resolution > state.grid_width
        };
        if needs_rebuild {
            self.rebuild_grid(new_resolution);
        }

        let mut refinement_steps = 0;
        for _ in 0..strategy::MAX_REFINEMENT_STEPS {
            let info = self.step(target_handles)?;
            refinement_steps += 1;
            if info.converged {
                break;
            }
        }

        let k_max_achieved = {
            let (ctx, state) = self.parts();
            let final_h = strategy::fill_distance(ctx.domain_bounds, state.grid_width);
            let final_omega = strategy::omega(final_h, c_norm, ctx.basis);
            ctx.policy.compute_k_max(k, final_omega).unwrap_or_else(|| {
                warn!(
                    "Strategy 2: cannot guarantee finite K_max \
                     (omega(h)={:.4} >= 1/K={:.4})",
                    final_omega,
                    1.0 / k,
                );
                f64::INFINITY
            })
        };

        Ok(strategy::Strategy2Result {
            required_h,
            required_resolution: new_resolution,
            current_h,
            k_max_achieved,
            c_norm,
            refinement_steps,
        })
    }

    /// Compute the Jacobian gradients (∇u, ∇v) at point x (Eq. 3 differentiated).
    fn grad_uv_at(&self, x: Vector2<f64>) -> (Vector2<f64>, Vector2<f64>) {
        let (ctx, state) = self.parts();
        let (gx, gy) = ctx.basis.gradient(x);
        let c = &state.coefficients;
        let n = ctx.basis.count();

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
    fn j_s_j_a_at(&self, x: Vector2<f64>) -> (Vector2<f64>, Vector2<f64>) {
        let (grad_u, grad_v) = self.grad_uv_at(x);
        distortion::compute_j_s_j_a(grad_u, grad_v)
    }

    /// Compute singular values (Σ, σ) at point x (Eq. 20).
    fn singular_values_at(&self, x: Vector2<f64>) -> (f64, f64) {
        let (j_s, j_a) = self.j_s_j_a_at(x);
        distortion::singular_values(j_s, j_a)
    }

    /// Rebuild the collocation grid at a new resolution (for Strategy 2).
    /// Preserves current coefficients. Resets active set, frames, precomputed data.
    fn rebuild_grid(&mut self, new_resolution: usize) {
        let (ctx, state) = self.parts_mut();

        let (new_points, new_gw, new_gh) =
            generate_collocation_grid(ctx.domain_bounds, new_resolution);
        let m = new_points.len();
        let new_mask = build_domain_mask(&new_points, ctx.domain);
        let new_frames = vec![Vector2::new(1.0, 0.0); m];

        let fps_k = state.stable_set.len().max(4);
        let new_stable_set =
            active_set::initialize_stable_set(&new_points, fps_k, &new_mask);

        let coefficients = state.coefficients.clone();

        *state = AlgorithmState {
            coefficients,
            collocation_points: new_points,
            active_set: Vec::new(),
            stable_set: new_stable_set,
            frames: new_frames,
            k_high: state.k_high,
            k_low: state.k_low,
            domain_mask: new_mask,
            precomputed: None,
            grid_width: new_gw,
            grid_height: new_gh,
            prev_target_handles: None,
        };
    }
}

// ─────────────────────────────────────────────
// Private helpers (used by default methods)
// ─────────────────────────────────────────────

/// Build biharmonic matrix (Eq. 31) into precomputed data.
/// Free function to avoid &mut self borrow conflicts in trait default methods.
fn ensure_biharmonic_matrix_inner(
    basis: &dyn BasisFunction,
    domain_bounds: &crate::model::types::DomainBounds,
    state: &mut AlgorithmState,
) {
    if let Some(ref mut precomputed) = state.precomputed {
        if precomputed.biharmonic_matrix.is_none() {
            let n = basis.count();
            precomputed.biharmonic_matrix = Some(solver::build_biharmonic_matrix(
                basis,
                &state.collocation_points,
                domain_bounds,
                n,
            ));
        }
    }
}

/// Generate a uniform collocation grid within the domain bounds.
///
/// Section 4: "consider all the points from a surrounding uniform grid
/// that fall inside the domain"
pub(crate) fn generate_collocation_grid(
    bounds: &crate::model::types::DomainBounds,
    resolution: usize,
) -> (Vec<Vector2<f64>>, usize, usize) {
    let dx = (bounds.x_max - bounds.x_min) / (resolution as f64 - 1.0);
    let dy = (bounds.y_max - bounds.y_min) / (resolution as f64 - 1.0);

    let mut points = Vec::with_capacity(resolution * resolution);

    let res_x = resolution;
    let res_y = resolution;

    for row in 0..res_y {
        for col in 0..res_x {
            let x = bounds.x_min + col as f64 * dx;
            let y = bounds.y_min + row as f64 * dy;
            points.push(Vector2::new(x, y));
        }
    }

    (points, res_x, res_y)
}

/// Build domain mask from collocation points and an optional domain.
///
/// Paper Section 4: "consider all the points from a surrounding uniform
/// grid that fall inside the domain"
pub(crate) fn build_domain_mask(
    points: &[Vector2<f64>],
    domain: Option<&dyn crate::model::domain::Domain>,
) -> Vec<bool> {
    match domain {
        Some(d) => points.iter().map(|pt| d.contains(pt)).collect(),
        None => vec![true; points.len()],
    }
}
