//! Display information resource, updated each algorithm step.

use bevy::prelude::*;

/// UI display information, updated each step.
#[derive(Resource)]
pub struct DeformationInfo {
    pub max_distortion: f64,
    pub active_set_size: usize,
    pub stable_set_size: usize,
    pub step_count: usize,
    pub k_bound: f64,
    pub lambda_reg: f64,
    pub reg_mode_label: &'static str,
    /// Strategy 2 result status message (None = not yet run)
    pub strategy2_status: Option<String>,
}

impl Default for DeformationInfo {
    fn default() -> Self {
        Self {
            max_distortion: 0.0,
            active_set_size: 0,
            stable_set_size: 0,
            step_count: 0,
            k_bound: 3.0,
            lambda_reg: 1e-2,
            reg_mode_label: "ARAP",
            strategy2_status: None,
        }
    }
}
