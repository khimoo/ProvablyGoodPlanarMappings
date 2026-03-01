//! Application state and resources.
//!
//! Defines the AppState FSM and resources that bridge pgpm-core with Bevy.

use bevy::prelude::*;
use nalgebra::Vector2;
use pgpm_core::{Algorithm, AlgorithmParams, DomainBounds, DistortionType, RegularizationType};

/// Application state machine.
/// Setup → Deforming (→ Verifying in Phase 3)
#[derive(States, Default, Clone, Eq, PartialEq, Hash, Debug)]
pub enum AppState {
    #[default]
    Setup,
    Deforming,
    // Phase 3: Verifying,
}

/// Wraps pgpm-core's Algorithm for Bevy resource use.
#[derive(Resource)]
pub struct DeformationState {
    /// The pgpm-core algorithm instance. Created on finalization.
    pub algorithm: Option<Algorithm>,
    /// Source handle positions in domain (pixel) coordinates.
    pub source_handles: Vec<Vector2<f64>>,
    /// Current target handle positions in domain (pixel) coordinates.
    pub target_handles: Vec<Vector2<f64>>,
    /// Whether a drag operation is currently in progress.
    pub dragging: bool,
    /// Index of the currently dragged handle, if any.
    pub dragging_index: Option<usize>,
    /// Whether the algorithm needs to run a step (coefficients changed).
    pub needs_solve: bool,
}

impl Default for DeformationState {
    fn default() -> Self {
        Self {
            algorithm: None,
            source_handles: Vec::new(),
            target_handles: Vec::new(),
            dragging: false,
            dragging_index: None,
            needs_solve: false,
        }
    }
}

impl DeformationState {
    /// Finalize the setup: create the Algorithm with current handles.
    pub fn finalize(&mut self, image_width: f64, image_height: f64, algo_params: &AlgoParams) {
        let epsilon = algo_params.epsilon;
        let domain = DomainBounds {
            x_min: -epsilon,
            x_max: image_width + epsilon,
            y_min: -epsilon,
            y_max: image_height + epsilon,
        };

        // Determine RBF scale s from the average distance between centers.
        let s = compute_rbf_scale(&self.source_handles, image_width, image_height);

        let basis = Box::new(pgpm_core::basis::gaussian::GaussianBasis::new(
            self.source_handles.clone(),
            s,
        ));

        let params = AlgorithmParams {
            distortion_type: DistortionType::Isometric,
            k_bound: algo_params.k_bound,
            lambda_reg: algo_params.lambda_reg,
            regularization: RegularizationType::Arap,
        };

        let algorithm = Algorithm::new(
            basis,
            params,
            domain,
            self.source_handles.clone(),
            algo_params.grid_resolution,
            algo_params.fps_k,
        );

        self.target_handles = self.source_handles.clone();
        self.algorithm = Some(algorithm);
    }
}

/// Compute a reasonable RBF scale parameter s.
/// Uses ~1/3 of the average nearest-neighbor distance if available,
/// otherwise falls back to a fraction of the domain size.
fn compute_rbf_scale(centers: &[Vector2<f64>], width: f64, height: f64) -> f64 {
    if centers.len() < 2 {
        return (width.max(height)) / 4.0;
    }

    // Average nearest-neighbor distance
    let mut total = 0.0;
    for (i, p) in centers.iter().enumerate() {
        let mut min_d = f64::INFINITY;
        for (j, q) in centers.iter().enumerate() {
            if i != j {
                let d = (p - q).norm();
                if d < min_d {
                    min_d = d;
                }
            }
        }
        total += min_d;
    }
    let avg_nn = total / centers.len() as f64;

    // s ≈ average_nn_distance * 0.8 — ensures decent overlap of RBFs
    avg_nn * 0.8
}

/// UI display information, updated each step.
#[derive(Resource, Default)]
pub struct DeformationInfo {
    pub max_distortion: f64,
    pub active_set_size: usize,
    pub stable_set_size: usize,
    pub step_count: usize,
    pub k_bound: f64,
    pub lambda_reg: f64,
    // Phase 3: pub verification_status: VerificationStatus,
}

/// Adjustable algorithm parameters.
#[derive(Resource)]
pub struct AlgoParams {
    pub k_bound: f64,
    pub lambda_reg: f64,
    pub grid_resolution: usize,
    pub fps_k: usize,
    pub epsilon: f64,
}

impl Default for AlgoParams {
    fn default() -> Self {
        Self {
            k_bound: 3.0,
            lambda_reg: 1e-2,
            grid_resolution: 50,
            fps_k: 8,
            epsilon: 40.0,
        }
    }
}

/// Marker for the main camera.
#[derive(Component)]
pub struct MainCamera;

/// Marker for the deformed image entity.
#[derive(Component)]
pub struct DeformedImage;

/// Marker for the UI text entity.
#[derive(Component)]
pub struct InfoText;

/// Data about the loaded image.
#[derive(Resource)]
pub struct ImageInfo {
    pub width: f32,
    pub height: f32,
    pub handle: Handle<Image>,
    pub contour: Vec<(f32, f32)>,
}
