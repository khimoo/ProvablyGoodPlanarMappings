//! Input handling: mouse interactions on the canvas.
//!
//! Setup mode: click to add handles.
//! Deforming mode: drag handles to move them.
//!
//! All parameter adjustments and mode switching are handled by the
//! GUI panel in `ui.rs`.

use bevy::prelude::*;

use crate::state::{
    AppState, DeformationInfo, DeformationState, ImageInfo, MainCamera,
};

/// Threshold (in world units) for clicking near a handle.
const HANDLE_CLICK_RADIUS: f32 = 15.0;
/// Minimum distance between new handles (in domain pixel coords).
const MIN_HANDLE_DISTANCE: f32 = 20.0;

/// System: process mouse input on the canvas (handle placement & dragging).
pub fn handle_input(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    mut deform_state: ResMut<DeformationState>,
    state: Res<State<AppState>>,
    image_info: Option<Res<ImageInfo>>,
    ui_interaction: Query<&Interaction, With<Button>>,
) {
    let Some(image_info) = image_info else { return };
    let Ok(window) = windows.get_single() else { return };
    let Ok((camera, cam_transform)) = camera_q.get_single() else { return };

    // If any UI button is being interacted with, don't process canvas clicks
    let ui_active = ui_interaction.iter().any(|i| *i != Interaction::None);
    if ui_active && !deform_state.dragging {
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
        if deform_state.dragging && buttons.just_released(MouseButton::Left) {
            end_drag(&mut deform_state);
        }
        return;
    }

    match state.get() {
        AppState::Setup => {
            handle_setup_input(&buttons, domain_x, domain_y, &mut deform_state);
        }
        AppState::Deforming => {
            handle_deform_input(&buttons, domain_x, domain_y, &mut deform_state);
        }
    }
}

fn handle_setup_input(
    buttons: &Res<ButtonInput<MouseButton>>,
    domain_x: f32,
    domain_y: f32,
    deform_state: &mut ResMut<DeformationState>,
) {
    if buttons.just_pressed(MouseButton::Left) {
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
    domain_x: f32,
    domain_y: f32,
    deform_state: &mut ResMut<DeformationState>,
) {
    if buttons.just_pressed(MouseButton::Left) {
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

    // Keep iterating while dragging OR while distortion exceeds the bound.
    // This ensures the algorithm converges even after drag release.
    let needs_more = deform_state.needs_solve
        || (deform_info.max_distortion > deform_info.k_bound
            && deform_info.step_count > 0);

    if !needs_more {
        return;
    }

    let targets: Vec<nalgebra::Vector2<f64>> = deform_state.target_handles.clone();

    // Run exactly one Algorithm 1 step per frame (SOCP is blocking).
    // Auto-continuation ensures this is called every frame until convergence.
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
