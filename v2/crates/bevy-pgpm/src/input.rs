//! Input handling: mouse interactions on the canvas.
//!
//! Setup mode: click to add handles.
//! Deforming mode: drag handles to move them.
//!
//! All parameter adjustments and mode switching are handled by the
//! GUI panel in `ui/`.

use bevy::prelude::*;
use log::info;

use crate::domain::coords::ImageCoords;
use crate::state::{AlgorithmState, AppState, DragState, ImageInfo, MainCamera};

/// Threshold (in world units) for clicking near a handle.
const HANDLE_CLICK_RADIUS: f32 = 15.0;
/// Minimum distance between new handles (in domain pixel coords).
const MIN_HANDLE_DISTANCE: f32 = 20.0;

/// System: process mouse input on the canvas (handle placement & dragging).
pub fn handle_input(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    mut algo_state: ResMut<AlgorithmState>,
    mut drag_state: ResMut<DragState>,
    state: Res<State<AppState>>,
    image_info: Option<Res<ImageInfo>>,
    ui_interaction: Query<&Interaction, With<Button>>,
) {
    let Some(image_info) = image_info else { return };
    let Ok(window) = windows.single() else { return };
    let Ok((camera, cam_transform)) = camera_q.single() else { return };

    // If any UI button is being interacted with, don't process canvas clicks
    let ui_active = ui_interaction.iter().any(|i| *i != Interaction::None);
    if ui_active && !drag_state.active {
        return;
    }

    // Convert cursor to world coordinates
    let Some(cursor_pos) = window.cursor_position() else { return };
    let Ok(ray) = camera.viewport_to_world(cam_transform, cursor_pos) else { return };
    let world_pos = ray.origin.truncate();

    // Convert world -> domain (pixel) coordinates
    let coords = ImageCoords::new(image_info.width, image_info.height);
    let (domain_x, domain_y) = coords.world_to_pixel(world_pos);

    // Bounds check
    if domain_x < 0.0 || domain_x > image_info.width || domain_y < 0.0 || domain_y > image_info.height {
        if drag_state.active && buttons.just_released(MouseButton::Left) {
            drag_state.end();
        }
        return;
    }

    match state.get() {
        AppState::Setup => {
            handle_setup_input(&buttons, domain_x, domain_y, &mut algo_state);
        }
        AppState::Deforming => {
            handle_deform_input(&buttons, domain_x, domain_y, &mut algo_state, &mut drag_state);
        }
    }
}

fn handle_setup_input(
    buttons: &Res<ButtonInput<MouseButton>>,
    domain_x: f32,
    domain_y: f32,
    algo_state: &mut ResMut<AlgorithmState>,
) {
    if buttons.just_pressed(MouseButton::Left) {
        let new_pt = nalgebra::Vector2::new(domain_x as f64, domain_y as f64);
        let too_close = algo_state.source_handles.iter().any(|p| {
            (p - new_pt).norm() < MIN_HANDLE_DISTANCE as f64
        });

        if !too_close {
            algo_state.source_handles.push(new_pt);
            algo_state.target_handles.push(new_pt);
            info!(
                "Added handle {} at ({:.1}, {:.1})",
                algo_state.source_handles.len() - 1,
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
    algo_state: &mut ResMut<AlgorithmState>,
    drag_state: &mut ResMut<DragState>,
) {
    if buttons.just_pressed(MouseButton::Left) {
        let threshold = HANDLE_CLICK_RADIUS;
        let mut best_idx = None;
        let mut best_dist = f32::INFINITY;

        for (i, tgt) in algo_state.target_handles.iter().enumerate() {
            let dx = domain_x as f64 - tgt.x;
            let dy = domain_y as f64 - tgt.y;
            let dist = ((dx * dx + dy * dy) as f32).sqrt();
            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(i);
            }
        }

        if best_dist < threshold {
            if let Some(idx) = best_idx {
                drag_state.start(idx);
            }
        }
    }

    if buttons.pressed(MouseButton::Left) && drag_state.active {
        if let Some(idx) = drag_state.handle_index {
            let new_pos = nalgebra::Vector2::new(domain_x as f64, domain_y as f64);
            algo_state.target_handles[idx] = new_pos;
            algo_state.needs_solve = true;
        }
    }

    if buttons.just_released(MouseButton::Left) && drag_state.active {
        drag_state.end();
    }
}
