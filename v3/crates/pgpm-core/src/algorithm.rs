//! Algorithm 1 integration.
//!
//! Paper reference: Algorithm 1 (Section 5)
//! This module implements the complete iterative algorithm for
//! provably good planar mappings.

use crate::active_set;
use crate::basis::BasisFunction;
use crate::distortion;
use crate::solver;
use crate::types::*;
use nalgebra::{DMatrix, Vector2};

/// Algorithm 1: complete implementation.
pub struct Algorithm {
    /// Basis functions (Table 1)
    basis: Box<dyn BasisFunction>,
    /// Algorithm parameters (K, λ, regularization type)
    params: AlgorithmParams,
    /// Algorithm state (coefficients, active set, frames, etc.)
    state: AlgorithmState,
    /// Source handle positions {p_l} (Eq. 29) — fixed
    source_handles: Vec<Vector2<f64>>,
    /// Grid dimensions for local maxima search
    grid_width: usize,
    grid_height: usize,
    /// Whether this is the first step (triggers initialization)
    is_first_step: bool,
    /// Domain bounds for biharmonic energy integration
    domain_bounds: DomainBounds,
}

impl Algorithm {
    /// Create a new Algorithm instance.
    ///
    /// # Arguments
    /// - `basis`: Basis function implementation (Gaussian, B-Spline, TPS)
    /// - `params`: Algorithm parameters (K, λ, regularization)
    /// - `domain_bounds`: Bounding box of domain Ω (Eq. 5)
    /// - `source_handles`: Fixed handle positions {p_l} (Eq. 29)
    /// - `grid_resolution`: Number of grid points per side (paper Section 6: 200)
    /// - `fps_k`: Number of farthest point samples for Z'' (stable set)
    pub fn new(
        basis: Box<dyn BasisFunction>,
        params: AlgorithmParams,
        domain_bounds: DomainBounds,
        source_handles: Vec<Vector2<f64>>,
        grid_resolution: usize,
        fps_k: usize,
    ) -> Self {
        // Generate collocation grid
        // Section 4: "consider all the points from a surrounding
        // uniform grid that fall inside the domain"
        let (collocation_points, grid_width, grid_height) =
            generate_collocation_grid(&domain_bounds, grid_resolution);

        let m = collocation_points.len();
        let _n_basis = basis.count();

        // Section 5 "Activation of constraints":
        // K_high = 0.1 + 0.9*K, K_low = 0.5 + 0.5*K
        let k = params.k_bound;
        let k_high = 0.1 + 0.9 * k;
        let k_low = 0.5 + 0.5 * k;

        // Initialize frames to (1, 0) — Algorithm 1: "Initialize d_i"
        let frames = vec![Vector2::new(1.0, 0.0); m];

        // Initialize with identity coefficients
        let coefficients = basis.identity_coefficients();

        // Initialize stable set with FPS
        let stable_set = active_set::initialize_stable_set(&collocation_points, fps_k);

        let state = AlgorithmState {
            coefficients,
            collocation_points,
            active_set: Vec::new(), // Algorithm 1: "Initialize empty active set Z'"
            stable_set,
            frames,
            k_high,
            k_low,
            precomputed: None,
        };

        Self {
            basis,
            params,
            state,
            source_handles,
            grid_width,
            grid_height,
            is_first_step: true,
            domain_bounds,
        }
    }

    /// Algorithm 1: execute one step.
    ///
    /// Pseudocode correspondence:
    /// 1. [first step only] Precompute φ(z), ∇φ(z), set d_i=(1,0), Z'=∅, Z''=FPS
    /// 2. Evaluate D(z) for all z ∈ Z
    /// 3. Find Z_max (local maxima of D)
    /// 4. Add z ∈ Z_max with D(z) > K_high to Z'
    /// 5. Remove z ∈ Z' with D(z) < K_low from Z'
    /// 6. Solve SOCP (Eq. 18) → update c
    /// 7. Update d_i (Eq. 27)
    pub fn step(&mut self, target_handles: &[Vector2<f64>]) -> Result<StepInfo, AlgorithmError> {
        // === Initialization (if first step) ===
        if self.is_first_step {
            self.precompute();
            self.is_first_step = false;
        }

        // === Evaluate distortion at all collocation points ===
        let precomputed = self.state.precomputed.as_ref().unwrap();
        let distortions = distortion::evaluate_distortion_all(
            &self.state.coefficients,
            precomputed,
            &self.params.distortion_type,
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
        )?;

        self.state.coefficients = new_coefficients;

        // === Update frames (Eq. 27) ===
        // d_i = J_S f(z_i) / ||J_S f(z_i)||
        let precomputed = self.state.precomputed.as_ref().unwrap();
        let j_s_values = distortion::evaluate_j_s_all(&self.state.coefficients, precomputed);

        // Update frames for ALL collocation points.
        // Algorithm 1 specifies updating d_i for active/stable points,
        // but ARAP regularization (Eq. 33) evaluates at all sample points,
        // so all frames must be current for the ARAP linear term to be correct.
        let eps = 1e-10;
        for idx in 0..j_s_values.len() {
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

        for (idx, pt) in self.state.collocation_points.iter().enumerate() {
            let val = self.basis.evaluate(*pt);
            let (gx, gy) = self.basis.gradient(*pt);

            for i in 0..n {
                phi[(idx, i)] = val[i];
                grad_phi_x[(idx, i)] = gx[i];
                grad_phi_y[(idx, i)] = gy[i];
            }
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

    /// Evaluate the mapping f(x) = Σ c_i f_i(x) (Eq. 3).
    pub fn evaluate(&self, x: Vector2<f64>) -> Vector2<f64> {
        let phi = self.basis.evaluate(x);
        let n = self.basis.count();

        let mut u = 0.0;
        let mut v = 0.0;
        for i in 0..n {
            u += self.state.coefficients[(0, i)] * phi[i];
            v += self.state.coefficients[(1, i)] * phi[i];
        }

        Vector2::new(u, v)
    }

    /// Get current coefficients (for rendering).
    pub fn coefficients(&self) -> &CoefficientMatrix {
        &self.state.coefficients
    }

    /// Get current active set size.
    pub fn active_set_size(&self) -> usize {
        self.state.active_set.len()
    }

    /// Get collocation points.
    pub fn collocation_points(&self) -> &[Vector2<f64>] {
        &self.state.collocation_points
    }

    /// Get the basis function reference.
    pub fn basis(&self) -> &dyn BasisFunction {
        self.basis.as_ref()
    }

    /// Get algorithm state (for inspection/testing).
    pub fn state(&self) -> &AlgorithmState {
        &self.state
    }
}

/// Information returned from each algorithm step.
#[derive(Debug)]
pub struct StepInfo {
    /// Maximum distortion over all collocation points
    pub max_distortion: f64,
    /// Number of points in the active set Z'
    pub active_set_size: usize,
    /// Number of points in the stable set Z''
    pub stable_set_size: usize,
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
    // but the paper uses square grids (200², 3000², etc.)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::gaussian::GaussianBasis;

    fn make_test_algorithm() -> Algorithm {
        // Simple 4-center Gaussian basis on [0,1]²
        let centers = vec![
            Vector2::new(0.25, 0.25),
            Vector2::new(0.75, 0.25),
            Vector2::new(0.25, 0.75),
            Vector2::new(0.75, 0.75),
        ];
        let basis = Box::new(GaussianBasis::new(centers, 0.3));

        let params = AlgorithmParams {
            distortion_type: DistortionType::Isometric,
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

        // Small grid for testing
        Algorithm::new(basis, params, domain, handles, 10, 4)
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
            &alg.params.distortion_type,
        );

        // Identity mapping should have D ≈ 1 everywhere
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
        let info = alg.step(&target).expect("Step should succeed");

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
        let info = alg.step(&target).expect("Step should succeed");

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

        // Run multiple steps — the distortion should converge
        for step in 0..5 {
            let info = alg.step(&target).expect("Step should succeed");
            println!(
                "Step {}: max_dist={:.4}, active={}",
                step, info.max_distortion, info.active_set_size
            );
            // After the first step, distortion should be finite
            assert!(info.max_distortion.is_finite());
        }
    }
}
