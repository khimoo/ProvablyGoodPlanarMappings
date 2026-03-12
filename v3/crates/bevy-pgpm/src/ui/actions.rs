//! Button action systems: handle user interactions on UI buttons.

use bevy::prelude::*;
use bevy::sprite::MeshMaterial2d;

use nalgebra::Vector2;
use pgpm_core::{DomainBounds, MappingParams, PolygonDomain};

use crate::domain::rbf::compute_rbf_scale;
use crate::rendering::{DeformMaterial, DeformUniform};
use crate::state::{
    AlgoParams, AlgorithmState, AppState, BasisType, DeformationInfo, DeformedImage, DragState,
    ImageInfo, ImagePathConfig,
};
use crate::ui::markers::*;

/// System: toggle between Setup and Deforming mode.
pub fn on_toggle_mode(
    query: Query<&Interaction, (Changed<Interaction>, With<ToggleModeButton>)>,
    state: Res<State<AppState>>,
    mut next_state: ResMut<NextState<AppState>>,
    mut algo_state: ResMut<AlgorithmState>,
    mut drag_state: ResMut<DragState>,
    mut deform_info: ResMut<DeformationInfo>,
    algo_params: Res<AlgoParams>,
    image_info: Option<Res<ImageInfo>>,
) {
    for interaction in &query {
        if *interaction != Interaction::Pressed { continue; }
        match state.get() {
            AppState::Setup => {
                let Some(ref image_info) = image_info else { return };
                if algo_state.source_handles.is_empty() {
                    info!("No handles added yet!");
                    return;
                }
                let mapping = build_mapping(
                    &algo_state.source_handles,
                    image_info.width as f64,
                    image_info.height as f64,
                    &algo_params,
                    &image_info.contour,
                    &image_info.holes,
                );
                algo_state.set_mapping(mapping);
                deform_info.k_bound = algo_params.k_bound;
                deform_info.lambda_reg = algo_params.lambda_reg;
                deform_info.reg_mode_label = algo_params.reg_mode.label();
                next_state.set(AppState::Deforming);
            }
            AppState::Deforming => {
                drag_state.end();
                next_state.set(AppState::Setup);
            }
        }
    }
}

/// System: reset all state and return to Setup.
pub fn on_reset(
    query: Query<&Interaction, (Changed<Interaction>, With<ResetButton>)>,
    mut algo_state: ResMut<AlgorithmState>,
    mut drag_state: ResMut<DragState>,
    mut deform_info: ResMut<DeformationInfo>,
    mut next_state: ResMut<NextState<AppState>>,
    image_info: Option<Res<ImageInfo>>,
    mut materials: ResMut<Assets<DeformMaterial>>,
    mat_query: Query<&MeshMaterial2d<DeformMaterial>, With<DeformedImage>>,
) {
    for interaction in &query {
        if *interaction != Interaction::Pressed { continue; }

        algo_state.reset();
        drag_state.end();
        *deform_info = DeformationInfo::default();
        next_state.set(AppState::Setup);

        // Reset shader uniform to identity mapping
        if let (Some(ref image_info), Ok(mat_handle)) = (image_info.as_ref(), mat_query.get_single()) {
            if let Some(material) = materials.get_mut(&mat_handle.0) {
                material.params = DeformUniform::identity(image_info.width, image_info.height);
            }
        }
    }
}

/// System: adjust K bound via +/- buttons.
pub fn on_k_bound(
    q_minus: Query<&Interaction, (Changed<Interaction>, With<KMinusButton>)>,
    q_plus: Query<&Interaction, (Changed<Interaction>, With<KPlusButton>)>,
    mut params: ResMut<AlgoParams>,
    mut info: ResMut<DeformationInfo>,
    mut algo_state: ResMut<AlgorithmState>,
) {
    let mut changed = false;
    for interaction in &q_minus {
        if *interaction == Interaction::Pressed {
            params.k_bound = (params.k_bound - 0.5).max(1.1);
            changed = true;
        }
    }
    for interaction in &q_plus {
        if *interaction == Interaction::Pressed {
            params.k_bound += 0.5;
            changed = true;
        }
    }
    if changed {
        enforce_k_max_invariant(&mut params);
        info.k_bound = params.k_bound;
        push_params(&params, &mut algo_state);
    }
}

/// System: adjust lambda via /10 and x10 buttons.
pub fn on_lambda(
    q_down: Query<&Interaction, (Changed<Interaction>, With<LambdaDownButton>)>,
    q_up: Query<&Interaction, (Changed<Interaction>, With<LambdaUpButton>)>,
    mut params: ResMut<AlgoParams>,
    mut info: ResMut<DeformationInfo>,
    mut algo_state: ResMut<AlgorithmState>,
) {
    let mut changed = false;
    for interaction in &q_down {
        if *interaction == Interaction::Pressed {
            params.lambda_reg /= 10.0;
            changed = true;
        }
    }
    for interaction in &q_up {
        if *interaction == Interaction::Pressed {
            params.lambda_reg *= 10.0;
            changed = true;
        }
    }
    if changed {
        info.lambda_reg = params.lambda_reg;
        push_params(&params, &mut algo_state);
    }
}

/// System: cycle regularization mode.
pub fn on_reg_mode(
    query: Query<&Interaction, (Changed<Interaction>, With<RegModeButton>)>,
    mut params: ResMut<AlgoParams>,
    mut info: ResMut<DeformationInfo>,
    mut algo_state: ResMut<AlgorithmState>,
) {
    for interaction in &query {
        if *interaction != Interaction::Pressed { continue; }
        params.reg_mode = params.reg_mode.next();
        info.reg_mode_label = params.reg_mode.label();
        push_params(&params, &mut algo_state);
    }
}

/// System: cycle basis function type (only effective in Setup mode).
pub fn on_basis_type(
    query: Query<&Interaction, (Changed<Interaction>, With<BasisTypeButton>)>,
    mut params: ResMut<AlgoParams>,
    state: Res<State<AppState>>,
) {
    for interaction in &query {
        if *interaction != Interaction::Pressed { continue; }
        if *state.get() != AppState::Setup {
            info!("Basis type can only be changed in Setup mode");
            continue;
        }
        params.basis_type = params.basis_type.next();
        info!("Basis type: {}", params.basis_type.label());
    }
}

/// System: handle the "Load Image" button click (opens native file dialog).
pub fn on_image_path(
    query: Query<&Interaction, (Changed<Interaction>, With<ImageLoadButton>)>,
    mut path_config: ResMut<ImagePathConfig>,
    mut path_text: Query<&mut Text, With<ImagePathText>>,
    state: Res<State<AppState>>,
) {
    for interaction in &query {
        if *interaction != Interaction::Pressed { continue; }

        // Only allow image loading in Setup mode
        if *state.get() != AppState::Setup {
            info!("Switch to Setup mode before loading a new image");
            continue;
        }

        // Open native file dialog
        let dialog = rfd::FileDialog::new()
            .add_filter("PNG Images", &["png"])
            .add_filter("All Images", &["png", "jpg", "jpeg", "bmp", "tga"])
            .set_title("Select Image");

        if let Some(path) = dialog.pick_file() {
            let abs_path = path.to_string_lossy().into_owned();
            info!("Selected image: {}", abs_path);

            // Update the display text (show just filename)
            let display_name = path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| abs_path.clone());
            for mut text in &mut path_text {
                **text = display_name.clone();
            }

            // Trigger reload
            path_config.abs_path = abs_path;
            path_config.needs_reload = true;
        }
    }
}

/// System: adjust K_max parameter via +/- buttons.
pub fn on_k_max(
    q_minus: Query<&Interaction, (Changed<Interaction>, With<KMaxMinusButton>)>,
    q_plus: Query<&Interaction, (Changed<Interaction>, With<KMaxPlusButton>)>,
    mut params: ResMut<AlgoParams>,
) {
    let mut changed = false;
    for interaction in &q_minus {
        if *interaction == Interaction::Pressed {
            params.k_max -= 0.5;
            changed = true;
        }
    }
    for interaction in &q_plus {
        if *interaction == Interaction::Pressed {
            params.k_max += 0.5;
            changed = true;
        }
    }
    if changed {
        enforce_k_max_invariant(&mut params);
        info!("K_max = {:.1}", params.k_max);
    }
}

/// System: handle the "Refine (Strategy 2)" button click.
pub fn on_strategy2(
    query: Query<&Interaction, (Changed<Interaction>, With<Strategy2Button>)>,
    state: Res<State<AppState>>,
    mut algo_state: ResMut<AlgorithmState>,
    mut deform_info: ResMut<DeformationInfo>,
    params: Res<AlgoParams>,
) {
    for interaction in &query {
        if *interaction != Interaction::Pressed { continue; }

        if *state.get() != AppState::Deforming {
            info!("Strategy 2 is only available in Deforming mode");
            continue;
        }

        if algo_state.algorithm.is_none() {
            info!("No algorithm instance");
            continue;
        };

        let targets = algo_state.target_handles.clone();
        let k_max = params.k_max;

        info!(
            "Strategy 2: K={:.2}, K_max={:.2}, running refinement...",
            params.k_bound, k_max
        );

        let algo = algo_state.algorithm.as_mut().unwrap();
        match algo.refine_strategy2(k_max, &targets) {
            Ok(result) => {
                let msg = format!(
                    "h_req={:.4}, res={}, steps={}\nK_max={:.3}, |||c|||={:.3}",
                    result.required_h,
                    result.required_resolution,
                    result.refinement_steps,
                    result.k_max_achieved,
                    result.c_norm,
                );
                info!("Strategy 2 complete: {}", msg);
                deform_info.strategy2_status = Some(msg);
                algo_state.needs_solve = false;
            }
            Err(e) => {
                let msg = format!("Error: {}", e);
                warn!("Strategy 2 failed: {}", msg);
                deform_info.strategy2_status = Some(msg);
            }
        }
    }
}

// Private helpers

/// Enforce Strategy 2 invariant: K_max > K (Eq. 14 precondition).
fn enforce_k_max_invariant(params: &mut AlgoParams) {
    let min_k_max = params.k_bound + 0.1;
    if params.k_max < min_k_max {
        params.k_max = min_k_max;
    }
}

/// Push updated parameters to the algorithm instance and flag for re-solve.
fn push_params(params: &AlgoParams, algo_state: &mut AlgorithmState) {
    if let Some(ref mut algo) = algo_state.algorithm {
        let core_params = MappingParams {
            k_bound: params.k_bound,
            lambda_reg: params.reg_mode.effective_lambda(params.lambda_reg),
            regularization: params.reg_mode.to_core(params.lambda_arap, params.lambda_bh),
        };
        algo.update_params(core_params);
        algo_state.needs_solve = true;
    }
}

/// Construct a pgpm-core mapping from UI parameters and handle positions.
///
/// This is the bridge between bevy-pgpm's UI types (`AlgoParams`, `BasisType`)
/// and pgpm-core's algorithm types (`MappingParams`, `BasisFunction`, `Domain`).
fn build_mapping(
    source_handles: &[Vector2<f64>],
    image_width: f64,
    image_height: f64,
    algo_params: &AlgoParams,
    contour: &[(f32, f32)],
    holes: &[Vec<(f32, f32)>],
) -> Box<dyn pgpm_core::MappingBridge> {
    let epsilon = algo_params.epsilon;
    let domain = DomainBounds {
        x_min: -epsilon,
        x_max: image_width + epsilon,
        y_min: -epsilon,
        y_max: image_height + epsilon,
    };

    let s = compute_rbf_scale(source_handles, image_width, image_height);

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
                source_handles.to_vec(),
                s,
            ))
        }
        BasisType::ShapeAwareGaussian => {
            let fmm_resolution = 256;
            Box::new(
                pgpm_core::basis::shape_aware_gaussian::ShapeAwareGaussianBasis::new(
                    source_handles.to_vec(),
                    s,
                    &contour_v2,
                    &domain,
                    fmm_resolution,
                ),
            )
        }
    };

    let params = MappingParams {
        k_bound: algo_params.k_bound,
        lambda_reg: algo_params.reg_mode.effective_lambda(algo_params.lambda_reg),
        regularization: algo_params.reg_mode.to_core(
            algo_params.lambda_arap,
            algo_params.lambda_bh,
        ),
    };

    let algo_domain: Option<Box<dyn pgpm_core::Domain>> = if contour_v2.is_empty() {
        None
    } else {
        Some(Box::new(PolygonDomain::new(contour_v2, holes_v2)))
    };

    pgpm_core::create_isometric_mapping(
        basis,
        params,
        domain,
        source_handles.to_vec(),
        algo_params.grid_resolution,
        algo_params.fps_k,
        algo_domain,
        pgpm_core::SolverConfig::default(),
    )
}
