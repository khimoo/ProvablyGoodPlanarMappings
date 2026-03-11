//! # pgpm-core
//!
//! Pure Rust implementation of "Provably Good Planar Mappings"
//! (Poranne & Lipman, 2014).
//!
//! This crate contains **only** the algorithm described in the paper.
//! No Bevy, no UI, no image processing dependencies.

pub mod types;
pub mod domain;
pub mod basis;
pub mod distortion;
pub mod active_set;
pub mod solver;
pub mod strategy;
pub mod mapping;
#[doc(hidden)]
pub mod distortion_policy;

// Algorithm module is #[doc(hidden)] pub so that integration tests
// (external crate scope) can access Algorithm<D> directly for testing,
// while the primary public API is the PlanarMapping trait + factories.
#[doc(hidden)]
pub mod algorithm;
pub mod geodesic;

pub use types::*;
pub use domain::{Domain, PolygonDomain};
pub use basis::BasisFunction;
pub use mapping::PlanarMapping;
pub use strategy::Strategy2Result;

// Re-export distortion_policy types under #[doc(hidden)] for integration tests.
#[doc(hidden)]
pub use distortion_policy::{IsometricPolicy, ConformalPolicy, DistortionPolicy};

// ─────────────────────────────────────────────
// Factory functions: primary public API
// ─────────────────────────────────────────────

use crate::algorithm::Algorithm;
use nalgebra::Vector2;

/// Create an isometric planar mapping (D_iso = max{Sigma, 1/sigma}).
///
/// This is the main entry point for consumers of pgpm-core.
/// Returns a `Box<dyn PlanarMapping>` that hides the concrete
/// `Algorithm<IsometricPolicy>` type.
pub fn create_isometric_mapping(
    basis: Box<dyn BasisFunction>,
    params: MappingParams,
    domain_bounds: DomainBounds,
    source_handles: Vec<Vector2<f64>>,
    grid_resolution: usize,
    fps_k: usize,
    domain: Option<Box<dyn Domain>>,
    solver_config: SolverConfig,
) -> Box<dyn PlanarMapping> {
    Box::new(Algorithm::new(
        basis,
        params,
        IsometricPolicy,
        domain_bounds,
        source_handles,
        grid_resolution,
        fps_k,
        domain,
        solver_config,
    ))
}

/// Create a conformal planar mapping (D_conf = Sigma / sigma).
///
/// `delta` must satisfy delta > omega(h) (Eq. 13).
pub fn create_conformal_mapping(
    basis: Box<dyn BasisFunction>,
    params: MappingParams,
    delta: f64,
    domain_bounds: DomainBounds,
    source_handles: Vec<Vector2<f64>>,
    grid_resolution: usize,
    fps_k: usize,
    domain: Option<Box<dyn Domain>>,
    solver_config: SolverConfig,
) -> Box<dyn PlanarMapping> {
    Box::new(Algorithm::new(
        basis,
        params,
        ConformalPolicy { delta },
        domain_bounds,
        source_handles,
        grid_resolution,
        fps_k,
        domain,
        solver_config,
    ))
}
