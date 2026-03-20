//! # pgpm-core
//!
//! Pure Rust implementation of "Provably Good Planar Mappings"
//! (Poranne & Lipman, 2014).
//!
//! This crate contains **only** the algorithm described in the paper.
//! No Bevy, no UI, no image processing dependencies.

pub mod model;
pub mod basis;
pub mod distortion;
pub mod numerics;
pub mod mapping;
#[doc(hidden)]
pub mod policy;

// Algorithm module is #[doc(hidden)] pub so that integration tests
// (external crate scope) can access Algorithm<D> directly for testing,
// while the primary public API is the MappingBridge trait + factories.
#[doc(hidden)]
pub mod algorithm;

// ─────────────────────────────────────────────
// Factory functions: primary public API
// ─────────────────────────────────────────────

use crate::algorithm::Algorithm;
use crate::basis::BasisFunction;
use crate::mapping::MappingBridge;
use crate::model::domain::Domain;
use crate::model::types::{DomainBounds, MappingParams, SolverConfig};
use crate::policy::{ConformalPolicy, IsometricPolicy};
use nalgebra::Vector2;

/// Create an isometric planar mapping (D_iso = max{Sigma, 1/sigma}).
///
/// This is the main entry point for consumers of pgpm-core.
/// Returns a `Box<dyn MappingBridge>` that hides the concrete
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
) -> Box<dyn MappingBridge> {
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
) -> Box<dyn MappingBridge> {
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
