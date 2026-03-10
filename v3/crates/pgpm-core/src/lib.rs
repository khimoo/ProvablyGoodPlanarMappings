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
pub mod algorithm;
pub mod geodesic;
pub mod strategy;

pub use types::*;
pub use domain::{Domain, PolygonDomain};
pub use basis::BasisFunction;
pub use algorithm::Algorithm;
pub use strategy::Strategy2Result;
