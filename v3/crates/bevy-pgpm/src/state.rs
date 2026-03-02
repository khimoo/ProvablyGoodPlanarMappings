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
    ///
    /// `contour` is the image's alpha-channel contour in pixel coordinates.
    /// If non-empty, only collocation points inside this polygon will be
    /// eligible for distortion constraints and ARAP regularisation
    /// (Paper Section 4: "points from a surrounding uniform grid that
    /// fall inside the domain").
    /// Returns `true` if shape-aware basis was selected (caller should set
    /// the `UseShapeAwareBasis` resource accordingly).
    pub fn finalize(
        &mut self,
        image_width: f64,
        image_height: f64,
        algo_params: &AlgoParams,
        contour: &[(f32, f32)],
    ) -> bool {
        let epsilon = algo_params.epsilon;
        let domain = DomainBounds {
            x_min: -epsilon,
            x_max: image_width + epsilon,
            y_min: -epsilon,
            y_max: image_height + epsilon,
        };

        // Determine RBF scale s from the average distance between centers.
        let s = compute_rbf_scale(&self.source_handles, image_width, image_height);

        let is_shape_aware = algo_params.basis_type == BasisType::ShapeAwareGaussian;

        // Convert contour from (f32, f32) to Vector2<f64> for pgpm-core
        let contour_v2: Vec<nalgebra::Vector2<f64>> = contour
            .iter()
            .map(|&(x, y)| nalgebra::Vector2::new(x as f64, y as f64))
            .collect();

        let basis: Box<dyn pgpm_core::basis::BasisFunction> = match algo_params.basis_type {
            BasisType::Gaussian => {
                Box::new(pgpm_core::basis::gaussian::GaussianBasis::new(
                    self.source_handles.clone(),
                    s,
                ))
            }
            BasisType::ShapeAwareGaussian => {
                let fmm_resolution = 256; // Grid resolution for geodesic FMM
                Box::new(
                    pgpm_core::basis::shape_aware_gaussian::ShapeAwareGaussianBasis::new(
                        self.source_handles.clone(),
                        s,
                        &contour_v2,
                        &domain,
                        fmm_resolution,
                    ),
                )
            }
        };

        let params = AlgorithmParams {
            distortion_type: DistortionType::Isometric,
            k_bound: algo_params.k_bound,
            lambda_reg: algo_params.reg_mode.effective_lambda(algo_params.lambda_reg),
            regularization: algo_params.reg_mode.to_core(
                algo_params.lambda_arap,
                algo_params.lambda_bh,
            ),
        };

        let domain_contour = if contour_v2.is_empty() {
            None
        } else {
            Some(contour_v2.as_slice())
        };

        let algorithm = Algorithm::new(
            basis,
            params,
            domain,
            self.source_handles.clone(),
            algo_params.grid_resolution,
            algo_params.fps_k,
            domain_contour,
        );

        self.target_handles = self.source_handles.clone();
        self.algorithm = Some(algorithm);
        is_shape_aware
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
#[derive(Resource)]
pub struct DeformationInfo {
    pub max_distortion: f64,
    pub active_set_size: usize,
    pub stable_set_size: usize,
    pub step_count: usize,
    pub k_bound: f64,
    pub lambda_reg: f64,
    pub reg_mode_label: &'static str,
    // Phase 3: pub verification_status: VerificationStatus,
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
        }
    }
}

/// Which basis function type to use (Table 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisType {
    /// Standard Euclidean Gaussian — GPU path (vertex shader evaluates RBFs).
    Gaussian,
    /// Shape-aware Gaussian using geodesic distance — CPU path.
    ShapeAwareGaussian,
    // Phase 3: BSpline,
    // Phase 3: TPS,
}

impl BasisType {
    pub fn next(self) -> Self {
        match self {
            BasisType::Gaussian => BasisType::ShapeAwareGaussian,
            BasisType::ShapeAwareGaussian => BasisType::Gaussian,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            BasisType::Gaussian => "Gaussian",
            BasisType::ShapeAwareGaussian => "Shape-Aware",
        }
    }
}

impl std::fmt::Display for BasisType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Which regularization mode the user has selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegMode {
    Arap,
    Biharmonic,
    Mixed,
    None,
}

impl RegMode {
    pub fn next(self) -> Self {
        match self {
            RegMode::Arap => RegMode::Biharmonic,
            RegMode::Biharmonic => RegMode::Mixed,
            RegMode::Mixed => RegMode::None,
            RegMode::None => RegMode::Arap,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            RegMode::Arap => "ARAP",
            RegMode::Biharmonic => "Biharmonic",
            RegMode::Mixed => "Mixed",
            RegMode::None => "None",
        }
    }

    /// Convert to pgpm-core RegularizationType.
    pub fn to_core(self, lambda_arap: f64, lambda_bh: f64) -> RegularizationType {
        match self {
            RegMode::Arap => RegularizationType::Arap,
            RegMode::Biharmonic => RegularizationType::Biharmonic,
            RegMode::Mixed => RegularizationType::Mixed {
                lambda_bh,
                lambda_arap,
            },
            RegMode::None => RegularizationType::Arap, // λ=0 handles this
        }
    }

    /// Effective λ_reg: for None mode we force 0.
    pub fn effective_lambda(self, lambda_reg: f64) -> f64 {
        match self {
            RegMode::None => 0.0,
            _ => lambda_reg,
        }
    }
}

impl std::fmt::Display for RegMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Adjustable algorithm parameters.
#[derive(Resource)]
pub struct AlgoParams {
    pub k_bound: f64,
    pub lambda_reg: f64,
    pub grid_resolution: usize,
    pub fps_k: usize,
    pub epsilon: f64,
    /// Regularization mode (ARAP / Biharmonic / Mixed / None)
    pub reg_mode: RegMode,
    /// ARAP weight for Mixed mode
    pub lambda_arap: f64,
    /// Biharmonic weight for Mixed mode
    pub lambda_bh: f64,
    /// Basis function type (Table 1)
    pub basis_type: BasisType,
}

impl Default for AlgoParams {
    fn default() -> Self {
        Self {
            k_bound: 3.0,
            lambda_reg: 1e-2,
            grid_resolution: 50,
            fps_k: 8,
            epsilon: 40.0,
            reg_mode: RegMode::Arap,
            lambda_arap: 1.0,
            lambda_bh: 0.1,
            basis_type: BasisType::Gaussian,
        }
    }
}

/// Marker for the main camera.
#[derive(Component)]
pub struct MainCamera;

/// Marker for the deformed image entity.
#[derive(Component)]
pub struct DeformedImage;

/// Configuration for the image file path.
///
/// The `abs_path` is used for `image::open()` (contour extraction, dimension query).
/// The `bevy_path` is used for `AssetServer::load()` (GPU texture).
#[derive(Resource)]
pub struct ImagePathConfig {
    /// Absolute filesystem path to the image file.
    pub abs_path: String,
    /// Whether the image needs to be (re)loaded.
    pub needs_reload: bool,
}

impl Default for ImagePathConfig {
    fn default() -> Self {
        // Default: texture.png inside the crate's assets/ directory.
        // Resolved at runtime relative to CARGO_MANIFEST_DIR or cwd.
        Self {
            abs_path: String::new(),
            needs_reload: true,
        }
    }
}

impl ImagePathConfig {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            abs_path: path.into(),
            needs_reload: true,
        }
    }
}

/// Data about the loaded image.
#[derive(Resource)]
pub struct ImageInfo {
    pub width: f32,
    pub height: f32,
    pub handle: Handle<Image>,
    pub contour: Vec<(f32, f32)>,
}
