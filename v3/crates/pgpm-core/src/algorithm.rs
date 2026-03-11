//! Algorithm 1 integration.
//!
//! Paper reference: Algorithm 1 (Section 5)
//! This module implements the complete iterative algorithm for
//! provably good planar mappings.

use crate::active_set;
use crate::basis::BasisFunction;
use crate::distortion;
use crate::distortion_policy::DistortionPolicy;
use crate::domain::Domain;
use crate::mapping::PlanarMapping;
use crate::solver;
use crate::strategy;
use crate::types::*;
use log::warn;
use nalgebra::{DMatrix, Vector2};

/// Algorithm 1: complete implementation, parameterised by distortion policy.
pub struct Algorithm<D: DistortionPolicy> {
    /// Basis functions (Table 1)
    basis: Box<dyn BasisFunction>,
    /// Algorithm parameters (K, lambda, regularization type)
    params: MappingParams,
    /// Distortion policy (isometric or conformal)
    policy: D,
    /// Algorithm state (coefficients, active set, frames, etc.)
    state: AlgorithmState,
    /// Source handle positions {p_l} (Eq. 29) -- fixed
    source_handles: Vec<Vector2<f64>>,
    /// Grid dimensions for local maxima search
    grid_width: usize,
    grid_height: usize,
    /// Whether this is the first step (triggers initialization)
    is_first_step: bool,
    /// Domain bounds for biharmonic energy integration
    domain_bounds: DomainBounds,
    /// Domain Omega for "x in Omega" test (Paper Section 4).
    /// When `None`, all grid points are considered inside the domain.
    domain: Option<Box<dyn Domain>>,
    /// SOCP solver numerical tuning (not from the paper).
    solver_config: SolverConfig,
}

impl<D: DistortionPolicy> Algorithm<D> {
    /// Create a new Algorithm instance.
    ///
    /// # Arguments
    /// - `basis`: Basis function implementation (Gaussian, B-Spline, TPS)
    /// - `params`: Algorithm parameters (K, lambda, regularization)
    /// - `policy`: Distortion policy (isometric or conformal)
    /// - `domain_bounds`: Bounding box of domain Omega (Eq. 5)
    /// - `source_handles`: Fixed handle positions {p_l} (Eq. 29)
    /// - `grid_resolution`: Number of grid points per side (paper Section 6: 200)
    /// - `fps_k`: Number of farthest point samples for Z'' (stable set)
    /// - `domain`: Abstract domain Omega. When `Some`, only grid points where
    ///   `domain.contains(pt)` returns true are eligible for the active/stable
    ///   sets and ARAP regularisation. The rectangular grid structure is
    ///   preserved for local-maxima detection (Section 5). When `None`, all
    ///   grid points are considered inside the domain.
    pub fn new(
        basis: Box<dyn BasisFunction>,
        params: MappingParams,
        policy: D,
        domain_bounds: DomainBounds,
        source_handles: Vec<Vector2<f64>>,
        grid_resolution: usize,
        fps_k: usize,
        domain: Option<Box<dyn Domain>>,
        solver_config: SolverConfig,
    ) -> Self {
        // Generate collocation grid
        // Section 4: "consider all the points from a surrounding
        // uniform grid that fall inside the domain"
        let (collocation_points, grid_width, grid_height) =
            generate_collocation_grid(&domain_bounds, grid_resolution);

        let m = collocation_points.len();

        // Build domain mask: true for points inside Omega.
        // Paper Section 4: "consider all the points from a surrounding
        // uniform grid that fall inside the domain"
        let domain_mask = build_domain_mask(&collocation_points, domain.as_deref());

        // Section 5 "Activation of constraints":
        // K_high = 0.1 + 0.9*K, K_low = 0.5 + 0.5*K
        let k = params.k_bound;
        let k_high = 0.1 + 0.9 * k;
        let k_low = 0.5 + 0.5 * k;

        // Initialize frames to (1, 0) -- Algorithm 1: "Initialize d_i"
        let frames = vec![Vector2::new(1.0, 0.0); m];

        // Initialize with identity coefficients
        let coefficients = basis.identity_coefficients();

        // Initialize stable set with FPS (only among domain-interior points)
        let stable_set = active_set::initialize_stable_set(
            &collocation_points, fps_k, &domain_mask,
        );

        let state = AlgorithmState {
            coefficients,
            collocation_points,
            active_set: Vec::new(), // Algorithm 1: "Initialize empty active set Z'"
            stable_set,
            frames,
            k_high,
            k_low,
            domain_mask,
            precomputed: None,
        };

        Self {
            basis,
            params,
            policy,
            state,
            source_handles,
            grid_width,
            grid_height,
            is_first_step: true,
            domain_bounds,
            domain,
            solver_config,
        }
    }

    /// Algorithm 1: execute one step.
    ///
    /// Pseudocode correspondence:
    /// 1. [first step only] Precompute phi(z), grad_phi(z), set d_i=(1,0), Z'={}, Z''=FPS
    /// 2. Evaluate D(z) for all z in Z
    /// 3. Find Z_max (local maxima of D)
    /// 4. Add z in Z_max with D(z) > K_high to Z'
    /// 5. Remove z in Z' with D(z) < K_low from Z'
    /// 6. Solve SOCP (Eq. 18) -> update c
    /// 7. Update d_i (Eq. 27)
    fn step_impl(&mut self, target_handles: &[Vector2<f64>]) -> Result<StepInfo, AlgorithmError> {
        // === Initialization (if first step) ===
        if self.is_first_step {
            self.precompute();
            self.is_first_step = false;
        }

        // === Evaluate distortion at all collocation points ===
        let precomputed = self.state.precomputed.as_ref().ok_or_else(|| {
            AlgorithmError::InvalidInput("Precomputed data not available (bug)".into())
        })?;
        let distortions = distortion::evaluate_distortion_all(
            &self.state.coefficients,
            precomputed,
            &self.policy,
        );

        // === Update active set (Algorithm 1 lines 5-8) ===
        active_set::update_active_set(
            &mut self.state,
            &distortions,
            self.grid_width,
            self.grid_height,
        );

        let max_distortion = distortions
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let n_active = self.state.active_set.len();

        // === Solve SOCP (Eq. 18) ===
        let new_coefficients = solver::solve_socp(
            &self.source_handles,
            target_handles,
            self.basis.as_ref(),
            &self.state,
            &self.params,
            &self.policy,
            &self.solver_config,
        )?;

        self.state.coefficients = new_coefficients;

        // === Update frames (Eq. 27) ===
        // d_i = J_S f(z_i) / ||J_S f(z_i)||
        //
        // Frames are updated only for constraint points (Z' union Z''),
        // consistent with Algorithm 1's postprocessing scope.
        // Frames at non-constraint points remain at their last value
        // (initialized to (1,0)).  This does not affect fold-over
        // guarantees -- see Section 5 "Initialization of the frames".
        let precomputed = self.state.precomputed.as_ref().ok_or_else(|| {
            AlgorithmError::InvalidInput("Precomputed data not available (bug)".into())
        })?;
        let j_s_values = distortion::evaluate_j_s_all(&self.state.coefficients, precomputed);

        let eps = 1e-10;
        for &idx in self.state.active_set.iter().chain(self.state.stable_set.iter()) {
            let j_s = j_s_values[idx];
            let norm = j_s.norm();
            if norm > eps {
                // Eq. 27: d_i = J_S f(x_i) / ||J_S f(x_i)||
                self.state.frames[idx] = j_s / norm;
            }
        }

        Ok(StepInfo {
            max_distortion,
            active_set_size: n_active,
            stable_set_size: self.state.stable_set.len(),
        })
    }

    /// Precompute basis function values and gradients at all collocation points.
    /// Algorithm 1: "if first step then" block.
    fn precompute(&mut self) {
        let m = self.state.collocation_points.len();
        let n = self.basis.count();

        let mut phi = DMatrix::zeros(m, n);
        let mut grad_phi_x = DMatrix::zeros(m, n);
        let mut grad_phi_y = DMatrix::zeros(m, n);

        let mut nan_inf_count = 0usize;
        for (idx, pt) in self.state.collocation_points.iter().enumerate() {
            let val = self.basis.evaluate(*pt);
            let (gx, gy) = self.basis.gradient(*pt);

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

        // Build biharmonic matrix if needed
        let biharmonic_matrix = match &self.params.regularization {
            RegularizationType::Biharmonic | RegularizationType::Mixed { .. } => {
                Some(solver::build_biharmonic_matrix(
                    self.basis.as_ref(),
                    &self.state.collocation_points,
                    &self.domain_bounds,
                    n,
                ))
            }
            _ => None,
        };

        self.state.precomputed = Some(PrecomputedData {
            phi,
            grad_phi_x,
            grad_phi_y,
            biharmonic_matrix,
        });
    }

    /// Update algorithm parameters at runtime.
    ///
    /// This allows changing K, lambda, and regularization type while the
    /// deformation is running, without re-creating the Algorithm.
    /// K_high and K_low are re-derived from the new K bound.
    /// The active set is NOT cleared -- stale points will be naturally
    /// removed at the next step when their distortion falls below K_low.
    fn update_params_impl(&mut self, params: MappingParams) {
        let k = params.k_bound;
        self.state.k_high = 0.1 + 0.9 * k;
        self.state.k_low = 0.5 + 0.5 * k;

        // If switching to a regularization that needs biharmonic matrix,
        // ensure it's computed.
        let needs_bh = matches!(
            params.regularization,
            RegularizationType::Biharmonic | RegularizationType::Mixed { .. }
        );
        let has_bh = self.state.precomputed.as_ref()
            .map_or(false, |p| p.biharmonic_matrix.is_some());

        if needs_bh && !has_bh {
            if let Some(ref mut precomputed) = self.state.precomputed {
                let n = self.basis.count();
                precomputed.biharmonic_matrix = Some(
                    solver::build_biharmonic_matrix(
                        self.basis.as_ref(),
                        &self.state.collocation_points,
                        &self.domain_bounds,
                        n,
                    )
                );
            }
        }

        self.params = params;
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
    fn refine_strategy2_impl(
        &mut self,
        k_max: f64,
        target_handles: &[Vector2<f64>],
    ) -> Result<strategy::Strategy2Result, AlgorithmError> {
        let k = self.params.k_bound;
        if k_max <= k {
            return Err(AlgorithmError::InvalidInput(format!(
                "Strategy 2 requires K_max ({}) > K ({})",
                k_max, k
            )));
        }

        // Step 1: compute |||c||| (Eq. 8)
        let c_norm = strategy::compute_c_norm(&self.state.coefficients);

        // Step 2: compute required h (delegated to policy)
        let required_h = self.policy.required_h(
            k, k_max, c_norm, self.basis.as_ref(),
        ).ok_or_else(|| AlgorithmError::InvalidInput(
            "Strategy 2: cannot compute required h (K_max too close to K or c_norm issue)".into(),
        ))?;

        // Current fill distance
        let current_h = strategy::fill_distance(&self.domain_bounds, self.grid_width);

        // Step 3: determine required resolution
        let new_resolution = strategy::resolution_for_h(&self.domain_bounds, required_h);
        // Check against configured maximum
        if new_resolution > self.solver_config.max_refinement_resolution {
            return Err(AlgorithmError::ResolutionExceeded {
                required: new_resolution,
                max: self.solver_config.max_refinement_resolution,
            });
        }

        // Only rebuild if we need a denser grid
        if new_resolution > self.grid_width {
            // Step 4: rebuild collocation grid at higher resolution
            let (new_points, new_gw, new_gh) =
                generate_collocation_grid(&self.domain_bounds, new_resolution);

            let m = new_points.len();

            // Rebuild domain mask on the refined grid using the stored domain.
            let new_mask = build_domain_mask(&new_points, self.domain.as_deref());

            // Reinitialize frames to (1,0) for new points
            let new_frames = vec![Vector2::new(1.0, 0.0); m];

            // Reinitialize stable set with FPS on the new grid
            let fps_k = self.state.stable_set.len().max(4);
            let new_stable_set = active_set::initialize_stable_set(
                &new_points, fps_k, &new_mask,
            );

            // Preserve current coefficients as initial guess
            let coefficients = self.state.coefficients.clone();

            self.state = AlgorithmState {
                coefficients,
                collocation_points: new_points,
                active_set: Vec::new(),
                stable_set: new_stable_set,
                frames: new_frames,
                k_high: self.state.k_high,
                k_low: self.state.k_low,
                domain_mask: new_mask,
                precomputed: None,
            };

            self.grid_width = new_gw;
            self.grid_height = new_gh;
            self.is_first_step = true;
        }

        // Step 5: run Algorithm 1 steps until convergence
        let mut refinement_steps = 0;
        let mut prev_active_set: Vec<usize> = Vec::new();

        for _ in 0..strategy::MAX_REFINEMENT_STEPS {
            let info = self.step_impl(target_handles)?;
            refinement_steps += 1;

            // Check convergence: max_distortion <= K and active set stable
            let current_active = self.state.active_set.clone();
            if info.max_distortion <= k && current_active == prev_active_set {
                break;
            }
            prev_active_set = current_active;
        }

        // Compute achieved K_max (delegated to policy)
        let final_h = strategy::fill_distance(&self.domain_bounds, self.grid_width);
        let final_omega = strategy::omega(final_h, c_norm, self.basis.as_ref());
        let k_max_achieved = self.policy.compute_k_max(k, final_omega)
            .unwrap_or_else(|| {
                warn!(
                    "Strategy 2: cannot guarantee finite K_max \
                     (omega(h)={:.4} >= 1/K={:.4})",
                    final_omega, 1.0 / k,
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

    /// Get current active set size.
    pub fn active_set_size(&self) -> usize {
        self.state.active_set.len()
    }

    /// Get collocation points.
    pub fn collocation_points(&self) -> &[Vector2<f64>] {
        &self.state.collocation_points
    }

    /// Get algorithm state (for inspection/testing).
    pub fn state(&self) -> &AlgorithmState {
        &self.state
    }

    /// Get current algorithm parameters (for inspection).
    pub fn params(&self) -> &MappingParams {
        &self.params
    }

    /// Get domain bounds.
    pub fn domain_bounds(&self) -> &DomainBounds {
        &self.domain_bounds
    }

    /// Get grid resolution (points per side).
    pub fn grid_width(&self) -> usize {
        self.grid_width
    }
}

// ─────────────────────────────────────────────
// PlanarMapping trait implementation
// ─────────────────────────────────────────────

impl<D: DistortionPolicy> PlanarMapping for Algorithm<D> {
    fn step(&mut self, target_handles: &[Vector2<f64>]) -> Result<StepInfo, AlgorithmError> {
        self.step_impl(target_handles)
    }

    // evaluate() uses default implementation from PlanarMapping (Eq. 3)

    fn coefficients(&self) -> &CoefficientMatrix {
        &self.state.coefficients
    }

    fn basis(&self) -> &dyn BasisFunction {
        self.basis.as_ref()
    }

    fn update_params(&mut self, params: MappingParams) {
        self.update_params_impl(params);
    }

    fn refine_strategy2(
        &mut self,
        k_max: f64,
        target_handles: &[Vector2<f64>],
    ) -> Result<strategy::Strategy2Result, AlgorithmError> {
        self.refine_strategy2_impl(k_max, target_handles)
    }
}

/// Generate a uniform collocation grid within the domain bounds.
///
/// Section 4: "consider all the points from a surrounding uniform grid
/// that fall inside the domain"
fn generate_collocation_grid(
    bounds: &DomainBounds,
    resolution: usize,
) -> (Vec<Vector2<f64>>, usize, usize) {
    let dx = (bounds.x_max - bounds.x_min) / (resolution as f64 - 1.0);
    let dy = (bounds.y_max - bounds.y_min) / (resolution as f64 - 1.0);

    let mut points = Vec::with_capacity(resolution * resolution);

    // Use the same resolution for both axes (square grid)
    // For non-square domains, we could use different resolutions,
    // but the paper uses square grids (200^2, 3000^2, etc.)
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
fn build_domain_mask(points: &[Vector2<f64>], domain: Option<&dyn Domain>) -> Vec<bool> {
    match domain {
        Some(d) => points.iter().map(|pt| d.contains(pt)).collect(),
        None => vec![true; points.len()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::gaussian::GaussianBasis;
    use crate::distortion_policy::IsometricPolicy;

    fn make_test_algorithm() -> Algorithm<IsometricPolicy> {
        // Simple 4-center Gaussian basis on [0,1]^2
        let centers = vec![
            Vector2::new(0.25, 0.25),
            Vector2::new(0.75, 0.25),
            Vector2::new(0.25, 0.75),
            Vector2::new(0.75, 0.75),
        ];
        let basis = Box::new(GaussianBasis::new(centers, 0.3));

        let params = MappingParams {
            k_bound: 3.0,
            lambda_reg: 0.0, // No regularization for simplicity
            regularization: RegularizationType::Arap,
        };

        let domain = DomainBounds {
            x_min: 0.0,
            x_max: 1.0,
            y_min: 0.0,
            y_max: 1.0,
        };

        let handles = vec![
            Vector2::new(0.5, 0.5),
        ];

        // Small grid for testing (no domain constraint -> full rectangle)
        Algorithm::new(
            basis, params, IsometricPolicy, domain, handles,
            10, 4, None, SolverConfig::default(),
        )
    }

    #[test]
    fn test_identity_mapping() {
        let alg = make_test_algorithm();

        // With identity coefficients, f(x) should be approximately x
        let test_point = Vector2::new(0.5, 0.5);
        let result = alg.evaluate(test_point);

        assert!(
            (result - test_point).norm() < 1e-10,
            "Identity mapping: expected ({}, {}), got ({}, {})",
            test_point.x, test_point.y, result.x, result.y
        );
    }

    #[test]
    fn test_identity_distortion_is_one() {
        let mut alg = make_test_algorithm();
        alg.precompute();

        let precomputed = alg.state.precomputed.as_ref().unwrap();
        let distortions = distortion::evaluate_distortion_all(
            &alg.state.coefficients,
            precomputed,
            &alg.policy,
        );

        // Identity mapping should have D ~ 1 everywhere
        for (i, &d) in distortions.iter().enumerate() {
            assert!(
                (d - 1.0).abs() < 1e-6,
                "Distortion at point {} = {}, expected ~1.0",
                i, d
            );
        }
    }

    #[test]
    fn test_step_with_identity_target() {
        let mut alg = make_test_algorithm();

        // Target = source (identity deformation)
        let target = vec![Vector2::new(0.5, 0.5)];
        let info = alg.step_impl(&target).expect("Step should succeed");

        // After solving with identity target, distortion should be close to 1
        assert!(
            info.max_distortion < 2.0,
            "Max distortion = {}, expected < 2.0 for near-identity",
            info.max_distortion
        );
    }

    #[test]
    fn test_step_with_deformed_target() {
        let mut alg = make_test_algorithm();

        // Move the handle slightly
        let target = vec![Vector2::new(0.6, 0.5)];
        let info = alg.step_impl(&target).expect("Step should succeed");

        // The mapping should still be valid (not infinite distortion)
        assert!(
            info.max_distortion < 100.0,
            "Max distortion = {}, expected finite value",
            info.max_distortion
        );

        // The evaluated point at the handle should be close to target
        let mapped = alg.evaluate(Vector2::new(0.5, 0.5));
        assert!(
            (mapped - Vector2::new(0.6, 0.5)).norm() < 0.5,
            "Handle should approximately reach target: got ({}, {})",
            mapped.x, mapped.y
        );
    }

    #[test]
    fn test_collocation_grid() {
        let bounds = DomainBounds {
            x_min: 0.0,
            x_max: 1.0,
            y_min: 0.0,
            y_max: 1.0,
        };
        let (points, w, h) = generate_collocation_grid(&bounds, 5);
        assert_eq!(w, 5);
        assert_eq!(h, 5);
        assert_eq!(points.len(), 25);

        // Check corners
        assert!((points[0] - Vector2::new(0.0, 0.0)).norm() < 1e-12);
        assert!((points[4] - Vector2::new(1.0, 0.0)).norm() < 1e-12);
        assert!((points[24] - Vector2::new(1.0, 1.0)).norm() < 1e-12);
    }

    #[test]
    fn test_multiple_steps() {
        let mut alg = make_test_algorithm();
        let target = vec![Vector2::new(0.6, 0.5)];

        // Run multiple steps -- the distortion should converge
        for step in 0..5 {
            let info = alg.step_impl(&target).expect("Step should succeed");
            println!(
                "Step {}: max_dist={:.4}, active={}",
                step, info.max_distortion, info.active_set_size
            );
            // After the first step, distortion should be finite
            assert!(info.max_distortion.is_finite());
        }
    }

    #[test]
    fn test_domain_mask_with_polygon_domain() {
        use crate::domain::PolygonDomain;

        let centers = vec![
            Vector2::new(0.5, 0.5),
        ];
        let basis = Box::new(GaussianBasis::new(centers.clone(), 0.3));
        let params = MappingParams {
            k_bound: 3.0,
            lambda_reg: 0.0,
            regularization: RegularizationType::Arap,
        };
        let domain = DomainBounds {
            x_min: 0.0,
            x_max: 1.0,
            y_min: 0.0,
            y_max: 1.0,
        };

        // A diamond contour that excludes the corners of the unit square
        let contour = vec![
            Vector2::new(0.5, 0.0),
            Vector2::new(1.0, 0.5),
            Vector2::new(0.5, 1.0),
            Vector2::new(0.0, 0.5),
        ];

        let polygon_domain = PolygonDomain::new(contour, vec![]);
        let alg = Algorithm::new(
            basis, params, IsometricPolicy, domain, centers, 5, 2,
            Some(Box::new(polygon_domain)),
            SolverConfig::default(),
        );

        // Grid is 5x5 = 25 points, but some should be masked out
        let mask = &alg.state().domain_mask;
        assert_eq!(mask.len(), 25);

        // Corners of the grid (0,0), (1,0), (0,1), (1,1) should be outside
        assert!(!mask[0],  "(0,0) should be outside diamond");
        assert!(!mask[4],  "(1,0) should be outside diamond");
        assert!(!mask[20], "(0,1) should be outside diamond");
        assert!(!mask[24], "(1,1) should be outside diamond");

        // Center (0.5, 0.5) should be inside
        assert!(mask[12], "(0.5,0.5) should be inside diamond");

        // Stable set should only contain domain-interior points
        for &idx in &alg.state().stable_set {
            assert!(mask[idx], "Stable set point {} should be inside domain", idx);
        }
    }
}
