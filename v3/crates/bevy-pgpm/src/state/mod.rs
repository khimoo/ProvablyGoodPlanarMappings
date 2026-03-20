//! Application state and resources.
//!
//! Defines the AppState FSM, marker components, and re-exports all
//! sub-module types for convenient `use crate::state::*` access.

pub mod algorithm;
pub mod display_info;
pub mod image_info;
pub mod interaction;
pub mod params;

pub use algorithm::{AlgorithmState, OriginalVertexPositions};
pub use display_info::DeformationInfo;
pub use image_info::{ImageInfo, ImagePathConfig};
pub use interaction::DragState;
pub use params::{AlgoParams, BasisType, RegMode};

use bevy::prelude::*;

/// Application state machine.
/// Setup -> Deforming (-> Verifying in Phase 3)
#[derive(States, Default, Clone, Eq, PartialEq, Hash, Debug)]
pub enum AppState {
    #[default]
    Setup,
    Deforming,
    // Phase 3: Verifying,
}

/// Marker for the main camera.
#[derive(Component)]
pub struct MainCamera;

/// Marker for the deformed image entity.
#[derive(Component)]
pub struct DeformedImage;
