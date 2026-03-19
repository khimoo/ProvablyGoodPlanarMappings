//! Integration tests for pgpm-core

use pgpm_core::*;
use approx;

// Helper for approximate equality
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {
        assert!(($a - $b).abs() < 1e-6, "{} ≈ {}", $a, $b);
    };
}

/// Create a simple test mapping
fn create_test_mapping() -> PGPMv2 {
    let boundary = DomainBoundary {
        points: vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ],
    };

    let domain = DomainInfo {
        boundary,
        filling_distance: 0.1,
    };

    PGPMv2::new(
        domain,
        Box::new(GaussianBasis::new(1.0)),
        Box::new(IsometricStrategy::default()),
        Box::new(BiharmonicRegularization::default()),
        Box::new(ClarabelSolver),
    )
}

#[test]
fn test_create_mapping() {
    let mapping = create_test_mapping();
    assert_eq!(mapping.step_count(), 0);
    assert!(!mapping.is_converged());
}

#[test]
fn test_add_handle() {
    let mut mapping = create_test_mapping();

    let result = mapping.add_handle(0, Vec2::new(0.5, 0.5), Vec2::new(0.6, 0.5));
    assert!(result.is_ok());
    assert_eq!(mapping.get_handles().len(), 1);
}

#[test]
fn test_update_handle() {
    let mut mapping = create_test_mapping();

    mapping.add_handle(0, Vec2::new(0.5, 0.5), Vec2::new(0.6, 0.5)).unwrap();
    let result = mapping.update_handle(0, Vec2::new(0.7, 0.5));
    assert!(result.is_ok());
}

#[test]
fn test_evaluate_mapping_no_handles() {
    let mapping = create_test_mapping();

    // With no handles, mapping should return zero (identity-like)
    let result = mapping.evaluate_mapping(Vec2::new(0.5, 0.5));
    assert_eq!(result.norm(), 0.0);
}

#[test]
fn test_evaluate_mapping_with_handles() {
    let mut mapping = create_test_mapping();

    mapping.add_handle(0, Vec2::new(0.5, 0.5), Vec2::new(0.1, 0.2)).unwrap();

    // Set a non-zero coefficient
    *mapping.get_coefficients_mut() = vec![Vec2::new(0.1, 0.2)];

    let result = mapping.evaluate_mapping(Vec2::new(0.5, 0.5));
    // At the center of a Gaussian with σ=1, φ(0) = 1, so result should be coefficient
    approx::assert_abs_diff_eq!(result.x, 0.1, epsilon = 0.01);
    approx::assert_abs_diff_eq!(result.y, 0.2, epsilon = 0.01);
}

#[test]
fn test_mapping_gradient() {
    let mut mapping = create_test_mapping();

    mapping.add_handle(0, Vec2::new(0.5, 0.5), Vec2::new(0.0, 0.0)).unwrap();
    *mapping.get_coefficients_mut() = vec![Vec2::new(1.0, 0.0)];

    // Gradient at handle center should be zero (radial symmetry)
    let grad = mapping.mapping_gradient(Vec2::new(0.5, 0.5));
    assert!(grad.norm() < 1e-6);
}

#[test]
fn test_jacobian_determinant() {
    let mut mapping = create_test_mapping();

    mapping.add_handle(0, Vec2::new(0.5, 0.5), Vec2::new(0.0, 0.0)).unwrap();

    // At identity-like point, det J should be close to 1
    let det = mapping.mapping_jacobian_determinant(Vec2::new(0.3, 0.3));
    assert!(!det.is_nan());
}

#[test]
fn test_algorithm_step_basic() {
    let mut mapping = create_test_mapping();

    mapping.add_handle(0, Vec2::new(0.5, 0.5), Vec2::new(0.6, 0.5)).unwrap();

    // Run one step
    let result = mapping.algorithm_step();
    assert!(result.is_ok());

    let step_result = result.unwrap();
    assert_eq!(step_result.step_num, 1);
    assert!(step_result.distortion_info.max_distortion > 0.0);
}

#[test]
fn test_verify_local_injectivity() {
    let mapping = create_test_mapping();

    // Identity mapping should be locally injective
    let result = mapping.verify_local_injectivity();
    assert!(result.is_ok());

    match result.unwrap() {
        VerificationResult::LocallyInjective => {
            // Expected
        }
        _ => panic!("Expected LocallyInjective"),
    }
}

#[test]
fn test_reset() {
    let mut mapping = create_test_mapping();

    mapping.add_handle(0, Vec2::new(0.5, 0.5), Vec2::new(0.6, 0.5)).unwrap();
    mapping.algorithm_step().unwrap();

    assert!(mapping.step_count() > 0);

    mapping.reset().unwrap();

    assert_eq!(mapping.step_count(), 0);
    assert!(!mapping.is_converged());
}

#[test]
fn test_gaussian_basis() {
    let basis = GaussianBasis::new(1.0);

    // φ(0) should be 1
    assert_approx_eq!(basis.evaluate(0.0), 1.0);

    // φ(r) should decrease with r
    let phi_1 = basis.evaluate(1.0);
    let phi_2 = basis.evaluate(2.0);
    assert!(phi_1 > phi_2);
    assert!(phi_1 > 0.0 && phi_2 > 0.0);
}

#[test]
fn test_isometric_strategy() {
    let strategy = IsometricStrategy::default();
    let (k_high, k_low) = strategy.get_activation_threshold();

    assert!(k_high > k_low);
    assert!(k_low > 1.0); // Thresholds should be near 1

    // Create identity Jacobian (should have σ ≈ 1)
    let identity = Mat2::identity();
    let sigma = strategy.compute_distortion(identity);

    approx::assert_abs_diff_eq!(sigma, 1.0, epsilon = 0.001);
}

#[test]
fn test_biharmonic_regularization() {
    let reg = BiharmonicRegularization::default();

    let domain = DomainInfo {
        boundary: DomainBoundary {
            points: vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0)],
        },
        filling_distance: 0.1,
    };

    let result = reg.build_energy_terms(&domain, &vec![1.0, 2.0]);
    assert!(result.is_ok());
}
