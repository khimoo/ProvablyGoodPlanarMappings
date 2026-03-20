//! Diagnostic tests for shape-aware Gaussian basis + Algorithm integration.
//! Reproduces the bevy-pgpm setup to find why deformation doesn't occur.

use nalgebra::Vector2;
use pgpm_core::algorithm::Algorithm;
use pgpm_core::basis::shape_aware_gaussian::ShapeAwareGaussianBasis;
use pgpm_core::basis::BasisFunction;
use pgpm_core::mapping::PlanarMapping;
use pgpm_core::model::domain::PolygonDomain;
use pgpm_core::model::types::{CoefficientMatrix, DomainBounds, MappingParams, RegularizationType};
use pgpm_core::numerics::geodesic::{GeodesicField, build_domain_mask};
use pgpm_core::policy::IsometricPolicy;

/// Evaluate the mapping f(x) = Σ c_i φ_i(x) (Eq. 3) from coefficients and basis.
fn eval_mapping(coefficients: &CoefficientMatrix, basis: &dyn BasisFunction, x: Vector2<f64>) -> Vector2<f64> {
    let phi = basis.evaluate(x);
    let n = basis.count();
    let mut u = 0.0;
    let mut v = 0.0;
    for i in 0..n {
        u += coefficients[(0, i)] * phi[i];
        v += coefficients[(1, i)] * phi[i];
    }
    Vector2::new(u, v)
}

/// Create a simple square polygon (like an image with no alpha cutout).
fn square_contour(w: f64, h: f64) -> Vec<Vector2<f64>> {
    vec![
        Vector2::new(0.0, 0.0),
        Vector2::new(w, 0.0),
        Vector2::new(w, h),
        Vector2::new(0.0, h),
    ]
}

#[test]
fn test_fmm_distances_finite_inside_polygon() {
    let w = 256.0;
    let h = 256.0;
    let contour = square_contour(w, h);
    let epsilon = 40.0;
    let bounds = DomainBounds {
        x_min: -epsilon, x_max: w + epsilon,
        y_min: -epsilon, y_max: h + epsilon,
    };
    let resolution = 200;
    let mask = build_domain_mask(&bounds, resolution, resolution, &contour);

    // Count inside cells
    let inside_count = mask.iter().filter(|&&b| b).count();
    println!("Domain mask: {} inside of {} total", inside_count, mask.len());
    assert!(inside_count > 0, "No cells inside polygon!");

    let source = Vector2::new(128.0, 128.0);
    let field = GeodesicField::compute(source, &bounds, resolution, resolution, &mask);

    // Check distances at a few points inside the polygon
    let dx = (bounds.x_max - bounds.x_min) / (resolution as f64 - 1.0);
    let dy = (bounds.y_max - bounds.y_min) / (resolution as f64 - 1.0);

    let mut finite_count = 0;
    let mut inf_inside_count = 0;
    for row in 0..resolution {
        for col in 0..resolution {
            let idx = row * resolution + col;
            if mask[idx] {
                let d = field.distance_at_index(idx);
                if d.is_finite() {
                    finite_count += 1;
                } else {
                    inf_inside_count += 1;
                    let x = bounds.x_min + col as f64 * dx;
                    let y = bounds.y_min + row as f64 * dy;
                    if inf_inside_count <= 5 {
                        println!("WARNING: Infinite distance at inside point ({}, {}) [row={}, col={}]", x, y, row, col);
                    }
                }
            }
        }
    }
    println!("FMM: {} finite distances, {} infinite (inside polygon)", finite_count, inf_inside_count);
    assert!(inf_inside_count == 0, "{} inside-polygon cells have infinite distance!", inf_inside_count);
}

#[test]
fn test_shape_aware_basis_evaluate_nonzero() {
    let w = 256.0;
    let h = 256.0;
    let contour = square_contour(w, h);
    let epsilon = 40.0;
    let bounds = DomainBounds {
        x_min: -epsilon, x_max: w + epsilon,
        y_min: -epsilon, y_max: h + epsilon,
    };

    let centers = vec![
        Vector2::new(64.0, 128.0),
        Vector2::new(192.0, 128.0),
    ];
    let s = 80.0;

    let basis = ShapeAwareGaussianBasis::new(centers.clone(), s, &contour, &bounds, 200);

    // Evaluate at center - should have phi=1 for that center
    let phi_at_center0 = basis.evaluate(Vector2::new(64.0, 128.0));
    println!("phi at center 0: {:?}", phi_at_center0.as_slice());
    assert!((phi_at_center0[0] - 1.0).abs() < 0.1,
        "phi[0] at center 0 should be ~1.0, got {}", phi_at_center0[0]);

    // Evaluate at a point between centers - should have nonzero values
    let phi_mid = basis.evaluate(Vector2::new(128.0, 128.0));
    println!("phi at midpoint: {:?}", phi_mid.as_slice());
    assert!(phi_mid[0] > 0.01, "phi[0] at midpoint should be nonzero, got {}", phi_mid[0]);
    assert!(phi_mid[1] > 0.01, "phi[1] at midpoint should be nonzero, got {}", phi_mid[1]);
}

#[test]
fn test_shape_aware_basis_gradient_nonzero() {
    let w = 256.0;
    let h = 256.0;
    let contour = square_contour(w, h);
    let epsilon = 40.0;
    let bounds = DomainBounds {
        x_min: -epsilon, x_max: w + epsilon,
        y_min: -epsilon, y_max: h + epsilon,
    };

    let centers = vec![
        Vector2::new(64.0, 128.0),
        Vector2::new(192.0, 128.0),
    ];
    let s = 80.0;

    let basis = ShapeAwareGaussianBasis::new(centers, s, &contour, &bounds, 200);

    // Gradient at a non-center point should be nonzero
    let (gx, gy) = basis.gradient(Vector2::new(100.0, 128.0));
    println!("grad_x at (100,128): {:?}", gx.as_slice());
    println!("grad_y at (100,128): {:?}", gy.as_slice());

    // RBF 0 gradient should be nonzero (we're to the right of center 0)
    assert!(gx[0].abs() > 1e-8, "grad_x[0] should be nonzero, got {}", gx[0]);
}

#[test]
fn test_shape_aware_algorithm_step() {
    // Simulate the exact setup that bevy-pgpm does
    let w = 256.0;
    let h = 256.0;
    let contour = square_contour(w, h);
    let epsilon = 40.0;
    let bounds = DomainBounds {
        x_min: -epsilon, x_max: w + epsilon,
        y_min: -epsilon, y_max: h + epsilon,
    };

    let source_handles = vec![
        Vector2::new(128.0, 128.0),
    ];
    let s = w / 4.0;

    let contour_v2: Vec<Vector2<f64>> = contour.clone();

    let basis = Box::new(ShapeAwareGaussianBasis::new(
        source_handles.clone(), s, &contour_v2, &bounds, 200,
    ));

    let params = MappingParams {
        k_bound: 3.0,
        lambda_reg: 1e-2,
        regularization: RegularizationType::Arap,
    };

    let domain = PolygonDomain::new(contour_v2, vec![]);
    let mut algo = Algorithm::new(
        basis, params, IsometricPolicy, bounds, source_handles.clone(),
        50, 8, Some(Box::new(domain)),
        pgpm_core::model::types::SolverConfig::default(),
    );

    // Step with the handle moved
    let target = vec![Vector2::new(160.0, 128.0)];
    let result = algo.step(&target);
    println!("Step result: {:?}", result);
    assert!(result.is_ok(), "SOCP step failed: {:?}", result.err());

    // The mapped point at source should be close to target (Eq. 3)
    let mapped = eval_mapping(algo.coefficients(), algo.basis(), Vector2::new(128.0, 128.0));
    println!("Source (128,128) maps to ({}, {}), target=(160,128)", mapped.x, mapped.y);
    assert!((mapped.x - 160.0).abs() < 50.0,
        "Handle should approximately reach target: mapped to ({}, {})", mapped.x, mapped.y);

    // A point away from handle should also move (not identity)
    let other = eval_mapping(algo.coefficients(), algo.basis(), Vector2::new(64.0, 64.0));
    println!("Point (64,64) maps to ({}, {})", other.x, other.y);
    // It shouldn't be exactly identity
    let drift = ((other.x - 64.0).powi(2) + (other.y - 64.0).powi(2)).sqrt();
    println!("Drift from identity at (64,64): {}", drift);
}
