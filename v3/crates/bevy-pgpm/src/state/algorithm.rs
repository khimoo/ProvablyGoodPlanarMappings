//! Core algorithm state: wraps pgpm-core's Algorithm for Bevy resource use.

use bevy::prelude::*;
use nalgebra::Vector2;
use pgpm_core::{Algorithm, AlgorithmParams, DomainBounds, DistortionType, PolygonDomain};

use crate::domain::rbf::compute_rbf_scale;
use crate::state::params::{AlgoParams, BasisType};

/// Core algorithm state, used by the solver and rendering systems.
#[derive(Resource)]
pub struct AlgorithmState {
    /// The pgpm-core algorithm instance. Created on finalization.
    pub algorithm: Option<Algorithm>,
    /// Source handle positions in domain (pixel) coordinates.
    pub source_handles: Vec<Vector2<f64>>,
    /// Current target handle positions in domain (pixel) coordinates.
    pub target_handles: Vec<Vector2<f64>>,
    /// Whether the algorithm needs to run a step (coefficients changed).
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

    /// Finalize the setup: create the Algorithm with current handles.
    ///
    /// `contour` is the image's alpha-channel outer contour in pixel coordinates.
    /// `holes` are interior hole contours in pixel coordinates.
    /// Returns `true` if shape-aware basis was selected.
    pub fn finalize(
        &mut self,
        image_width: f64,
        image_height: f64,
        algo_params: &AlgoParams,
        contour: &[(f32, f32)],
        holes: &[Vec<(f32, f32)>],
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

        // Convert contours from (f32, f32) to Vector2<f64> for pgpm-core
        let contour_v2: Vec<Vector2<f64>> = contour
            .iter()
            .map(|&(x, y)| Vector2::new(x as f64, y as f64))
            .collect();

        let holes_v2: Vec<Vec<Vector2<f64>>> = holes
            .iter()
            .map(|hole| {
                hole.iter()
                    .map(|&(x, y)| Vector2::new(x as f64, y as f64))
                    .collect()
            })
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
                let holes_refs: Vec<&[Vector2<f64>]> =
                    holes_v2.iter().map(|h| h.as_slice()).collect();
                Box::new(
                    pgpm_core::basis::shape_aware_gaussian::ShapeAwareGaussianBasis::new(
                        self.source_handles.clone(),
                        s,
                        &contour_v2,
                        &holes_refs,
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

        // Build abstract domain Ω for the algorithm (Domain trait).
        // The algorithm only needs "x ∈ Ω" — polygon details stay here.
        let algo_domain: Option<Box<dyn pgpm_core::Domain>> = if contour_v2.is_empty() {
            None
        } else {
            Some(Box::new(PolygonDomain::new(contour_v2, holes_v2)))
        };

        let algorithm = Algorithm::new(
            basis,
            params,
            domain,
            self.source_handles.clone(),
            algo_params.grid_resolution,
            algo_params.fps_k,
            algo_domain,
        );

        self.target_handles = self.source_handles.clone();
        self.algorithm = Some(algorithm);
        is_shape_aware
    }
}
