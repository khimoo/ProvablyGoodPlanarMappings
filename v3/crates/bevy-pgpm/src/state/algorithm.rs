//! Core algorithm state: wraps pgpm-core's MappingBridge for Bevy resource use.

use bevy::prelude::*;
use nalgebra::Vector2;
use pgpm_core::MappingBridge;

/// Core algorithm state, used by the solver and rendering systems.
#[derive(Resource)]
pub struct AlgorithmState {
    /// The pgpm-core mapping instance. Created on finalization.
    pub algorithm: Option<Box<dyn MappingBridge>>,
    /// Source handle positions in domain (pixel) coordinates.
    pub source_handles: Vec<Vector2<f64>>,
    /// Current target handle positions in domain (pixel) coordinates.
    pub target_handles: Vec<Vector2<f64>>,
    /// Whether the algorithm needs to run a step (targets changed).
    pub needs_solve: bool,
}

impl Default for AlgorithmState {
    fn default() -> Self {
        Self {
            algorithm: None,
            source_handles: Vec::new(),
            target_handles: Vec::new(),
            needs_solve: false,
        }
    }
}

impl AlgorithmState {
    /// Reset all state to default (used on image reload and reset button).
    pub fn reset(&mut self) {
        self.source_handles.clear();
        self.target_handles.clear();
        self.algorithm = None;
        self.needs_solve = false;
    }

    /// Set the mapping instance, initializing targets to source positions.
    pub fn set_mapping(&mut self, algorithm: Box<dyn MappingBridge>) {
        self.target_handles = self.source_handles.clone();
        self.algorithm = Some(algorithm);
    }
}
