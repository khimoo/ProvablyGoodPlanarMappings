//! Input handling: mouse clicks and keyboard shortcuts.
//!
//! Setup mode: click to add handles.
//! Deforming mode: drag handles, keyboard to adjust parameters.

use bevy::prelude::*;

use crate::state::{
    AlgoParams, AppState, DeformationInfo, DeformationState, ImageInfo, MainCamera,
};

/// Threshold (in world units) for clicking near a handle.
const HANDLE_CLICK_RADIUS: f32 = 15.0;
/// Minimum distance between new handles (in domain pixel coords).
const MIN_HANDLE_DISTANCE: f32 = 20.0;

/// System: process mouse and keyboard input.
pub fn handle_input(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    mut deform_state: ResMut<DeformationState>,
    mut deform_info: ResMut<DeformationInfo>,
    mut algo_params: ResMut<AlgoParams>,
    state: Res<State<AppState>>,
    mut next_state: ResMut<NextState<AppState>>,
    image_info: Option<Res<ImageInfo>>,
) {
    let Some(image_info) = image_info else { return };
    let Ok(window) = windows.get_single() else { return };
    let Ok((camera, cam_transform)) = camera_q.get_single() else { return };

    // Global: R to reset
    if keys.just_pressed(KeyCode::KeyR) {
        deform_state.source_handles.clear();
        deform_state.target_handles.clear();
        deform_state.algorithm = None;
        deform_state.dragging = false;
        deform_state.dragging_index = None;
        deform_state.needs_solve = false;
        deform_info.step_count = 0;
        deform_info.max_distortion = 0.0;
        deform_info.active_set_size = 0;
        deform_info.stable_set_size = 0;
        next_state.set(AppState::Setup);
        return;
    }

    // Parameter adjustment (available in both modes)
    handle_parameter_keys(&keys, &mut algo_params, &mut deform_info);

    // Space: toggle Setup ↔ Deforming
    if keys.just_pressed(KeyCode::Space) {
        match state.get() {
            AppState::Setup => {
                if deform_state.source_handles.is_empty() {
                    info!("No handles added yet!");
                    return;
                }
                // Finalize: create Algorithm
                deform_state.finalize(
                    image_info.width as f64,
                    image_info.height as f64,
                    &algo_params,
                );
                deform_info.k_bound = algo_params.k_bound;
                deform_info.lambda_reg = algo_params.lambda_reg;
                next_state.set(AppState::Deforming);
                info!("Switched to Deforming mode with {} handles", deform_state.source_handles.len());
            }
            AppState::Deforming => {
                deform_state.dragging = false;
                deform_state.dragging_index = None;
                next_state.set(AppState::Setup);
                info!("Switched back to Setup mode");
            }
        }
        return;
    }

    // Convert cursor to world coordinates
    let Some(cursor_pos) = window.cursor_position() else { return };
    let Ok(ray) = camera.viewport_to_world(cam_transform, cursor_pos) else { return };
    let world_pos = ray.origin.truncate();

    // Convert world → domain (pixel) coordinates
    let img_w = image_info.width;
    let img_h = image_info.height;
    let domain_x = world_pos.x + img_w / 2.0;
    let domain_y = img_h / 2.0 - world_pos.y;

    // Bounds check
    if domain_x < 0.0 || domain_x > img_w || domain_y < 0.0 || domain_y > img_h {
        // If dragging and released outside, end drag
        if deform_state.dragging && buttons.just_released(MouseButton::Left) {
            end_drag(&mut deform_state);
        }
        return;
    }

    match state.get() {
        AppState::Setup => {
            handle_setup_input(
                &buttons,
                world_pos,
                domain_x,
                domain_y,
                &mut deform_state,
                &image_info,
            );
        }
        AppState::Deforming => {
            handle_deform_input(
                &buttons,
                world_pos,
                domain_x,
                domain_y,
                &mut deform_state,
            );
        }
    }
}

fn handle_parameter_keys(
    keys: &Res<ButtonInput<KeyCode>>,
    params: &mut ResMut<AlgoParams>,
    info: &mut ResMut<DeformationInfo>,
) {
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);

    // K / Shift+K: K bound ±0.5
    if keys.just_pressed(KeyCode::KeyK) {
        if shift {
            params.k_bound = (params.k_bound - 0.5).max(1.1);
        } else {
            params.k_bound += 0.5;
        }
        info.k_bound = params.k_bound;
        info!("K bound = {:.1}", params.k_bound);
    }

    // L / Shift+L: lambda ×10 / ÷10
    if keys.just_pressed(KeyCode::KeyL) {
        if shift {
            params.lambda_reg /= 10.0;
        } else {
            params.lambda_reg *= 10.0;
        }
        info.lambda_reg = params.lambda_reg;
        info!("Lambda = {:.1e}", params.lambda_reg);
    }
}

fn handle_setup_input(
    buttons: &Res<ButtonInput<MouseButton>>,
    _world_pos: Vec2,
    domain_x: f32,
    domain_y: f32,
    deform_state: &mut ResMut<DeformationState>,
    _image_info: &ImageInfo,
) {
    if buttons.just_pressed(MouseButton::Left) {
        // Check minimum distance to existing handles
        let new_pt = nalgebra::Vector2::new(domain_x as f64, domain_y as f64);
        let too_close = deform_state.source_handles.iter().any(|p| {
            (p - new_pt).norm() < MIN_HANDLE_DISTANCE as f64
        });

        if !too_close {
            deform_state.source_handles.push(new_pt);
            deform_state.target_handles.push(new_pt);
            info!(
                "Added handle {} at ({:.1}, {:.1})",
                deform_state.source_handles.len() - 1,
                domain_x,
                domain_y
            );
        }
    }
}

fn handle_deform_input(
    buttons: &Res<ButtonInput<MouseButton>>,
    _world_pos: Vec2,
    domain_x: f32,
    domain_y: f32,
    deform_state: &mut ResMut<DeformationState>,
) {
    if buttons.just_pressed(MouseButton::Left) {
        // Find closest handle (using domain pixel coordinates)
        let threshold = HANDLE_CLICK_RADIUS;
        let mut best_idx = None;
        let mut best_dist = f32::INFINITY;

        for (i, tgt) in deform_state.target_handles.iter().enumerate() {
            let dx = domain_x as f64 - tgt.x;
            let dy = domain_y as f64 - tgt.y;
            let dist = ((dx * dx + dy * dy) as f32).sqrt();
            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(i);
            }
        }

        if best_dist < threshold && best_idx.is_some() {
            deform_state.dragging = true;
            deform_state.dragging_index = best_idx;
        }
    }

    if buttons.pressed(MouseButton::Left) && deform_state.dragging {
        if let Some(idx) = deform_state.dragging_index {
            let new_pos = nalgebra::Vector2::new(domain_x as f64, domain_y as f64);
            deform_state.target_handles[idx] = new_pos;
            deform_state.needs_solve = true;
        }
    }

    if buttons.just_released(MouseButton::Left) && deform_state.dragging {
        end_drag(deform_state);
    }
}

fn end_drag(deform_state: &mut ResMut<DeformationState>) {
    deform_state.dragging = false;
    deform_state.dragging_index = None;
    // Phase 3: trigger Strategy 2 verification here
}

/// System: run one Algorithm step if needed (Deforming state only).
pub fn update_deformation(
    state: Res<State<AppState>>,
    mut deform_state: ResMut<DeformationState>,
    mut deform_info: ResMut<DeformationInfo>,
) {
    if *state.get() != AppState::Deforming {
        return;
    }

    if !deform_state.needs_solve {
        return;
    }

    let targets: Vec<nalgebra::Vector2<f64>> = deform_state.target_handles.clone();

    if let Some(ref mut algo) = deform_state.algorithm {
        match algo.step(&targets) {
            Ok(step_info) => {
                deform_info.max_distortion = step_info.max_distortion;
                deform_info.active_set_size = step_info.active_set_size;
                deform_info.stable_set_size = step_info.stable_set_size;
                deform_info.step_count += 1;
            }
            Err(e) => {
                warn!("SOCP solve failed: {:?}", e);
            }
        }
    }

    deform_state.needs_solve = false;
}
