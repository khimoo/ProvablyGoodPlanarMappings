//! Display information resource, updated each algorithm step.

use bevy::prelude::*;

/// UI display information, updated each algorithm step.
///
/// Contains only per-step results from pgpm-core. Algorithm parameters
/// (K, lambda, regularization) are read from [`super::AlgoParams`] directly.
#[derive(Resource, Default)]
pub struct DeformationInfo {
    pub max_distortion: f64,
    pub active_set_size: usize,
    pub stable_set_size: usize,
    pub step_count: usize,
    /// Algorithm 1 convergence flag from pgpm-core.
    pub converged: bool,
    /// Strategy 2 result status message (None = not yet run)
    pub strategy2_status: Option<String>,
}
