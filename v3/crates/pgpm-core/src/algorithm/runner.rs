//! Algorithm 1 integration.
//!
//! Paper reference: Algorithm 1 (Section 5)
//! This module provides the concrete `Algorithm<D>` struct that holds
//! the data for a provably good planar mapping.  All algorithmic logic
//! lives in the [`PlanarMapping`] trait's default methods; this struct
//! only supplies the borrow-splitting accessors.

use crate::algorithm::active_set;
use crate::basis::BasisFunction;
use crate::mapping::planar_mapping::{build_domain_mask, generate_collocation_grid};
use crate::mapping::PlanarMapping;
use crate::model::domain::Domain;
use crate::model::types::*;
use crate::policy::DistortionPolicy;
use nalgebra::Vector2;

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
        let stable_set =
            active_set::initialize_stable_set(&collocation_points, fps_k, &domain_mask);

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
            grid_width,
            grid_height,
            prev_target_handles: None,
        };

        Self {
            basis,
            params,
            policy,
            state,
            source_handles,
            domain_bounds,
            domain,
            solver_config,
        }
    }

    // ─────────────────────────────────────────
    // Inherent convenience methods (not in trait)
    // ─────────────────────────────────────────

    /// Get current active set size.
    pub fn active_set_size(&self) -> usize {
        self.state.active_set.len()
    }

    /// Get collocation points.
    pub fn collocation_points(&self) -> &[Vector2<f64>] {
        &self.state.collocation_points
    }
}

// ─────────────────────────────────────────────
// PlanarMapping trait implementation
// ─────────────────────────────────────────────

impl<D: DistortionPolicy> PlanarMapping for Algorithm<D> {
    // All algorithmic methods (step, evaluate, refine_strategy2, etc.)
    // use the default implementations from PlanarMapping.
    // Only the three required accessors are provided here.

    fn parts(&self) -> (MappingContext<'_>, &AlgorithmState) {
        (
            MappingContext {
                basis: self.basis.as_ref(),
                policy: &self.policy,
                params: &self.params,
                source_handles: &self.source_handles,
                domain_bounds: &self.domain_bounds,
                domain: self.domain.as_deref(),
                solver_config: &self.solver_config,
            },
            &self.state,
        )
    }

    fn parts_mut(&mut self) -> (MappingContext<'_>, &mut AlgorithmState) {
        (
            MappingContext {
                basis: self.basis.as_ref(),
                policy: &self.policy,
                params: &self.params,
                source_handles: &self.source_handles,
                domain_bounds: &self.domain_bounds,
                domain: self.domain.as_deref(),
                solver_config: &self.solver_config,
            },
            &mut self.state,
        )
    }

    fn set_params(&mut self, params: MappingParams) {
        self.params = params;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::gaussian::GaussianBasis;
    use crate::policy::IsometricPolicy;

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

        let handles = vec![Vector2::new(0.5, 0.5)];

        // Small grid for testing (no domain constraint -> full rectangle)
        Algorithm::new(
            basis,
            params,
            IsometricPolicy,
            domain,
            handles,
            10,
            4,
            None,
            SolverConfig::default(),
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
            test_point.x,
            test_point.y,
            result.x,
            result.y
        );
    }

    #[test]
    fn test_identity_distortion_is_one() {
        use crate::distortion;

        let mut alg = make_test_algorithm();
        alg.ensure_precomputed();

        let (ctx, state) = alg.parts();
        let precomputed = state.precomputed.as_ref().unwrap();
        let distortions = distortion::evaluate_distortion_all(
            &state.coefficients,
            precomputed,
            ctx.policy,
        );

        // Identity mapping should have D ~ 1 everywhere
        for (i, &d) in distortions.iter().enumerate() {
            assert!(
                (d - 1.0).abs() < 1e-6,
                "Distortion at point {} = {}, expected ~1.0",
                i,
                d
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
            mapped.x,
            mapped.y
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
            let info = alg.step(&target).expect("Step should succeed");
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
        use crate::model::domain::PolygonDomain;

        let centers = vec![Vector2::new(0.5, 0.5)];
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
            basis,
            params,
            IsometricPolicy,
            domain,
            centers,
            5,
            2,
            Some(Box::new(polygon_domain)),
            SolverConfig::default(),
        );

        // Grid is 5x5 = 25 points, but some should be masked out
        let (_, state) = alg.parts();
        let mask = &state.domain_mask;
        assert_eq!(mask.len(), 25);

        // Corners of the grid (0,0), (1,0), (0,1), (1,1) should be outside
        assert!(!mask[0], "(0,0) should be outside diamond");
        assert!(!mask[4], "(1,0) should be outside diamond");
        assert!(!mask[20], "(0,1) should be outside diamond");
        assert!(!mask[24], "(1,1) should be outside diamond");

        // Center (0.5, 0.5) should be inside
        assert!(mask[12], "(0.5,0.5) should be inside diamond");

        // Stable set should only contain domain-interior points
        for &idx in &state.stable_set {
            assert!(mask[idx], "Stable set point {} should be inside domain", idx);
        }
    }
}
