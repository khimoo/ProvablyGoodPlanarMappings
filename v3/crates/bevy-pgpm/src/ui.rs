//! UI overlay: information text and control point gizmos.

use bevy::prelude::*;

use crate::state::{
    AppState, DeformationInfo, DeformationState, ImageInfo, InfoText,
};

/// System: update the info text overlay.
pub fn update_ui_text(
    state: Res<State<AppState>>,
    deform_info: Res<DeformationInfo>,
    deform_state: Res<DeformationState>,
    mut query: Query<(&mut Text, &mut TextColor), With<InfoText>>,
) {
    let Ok((mut text, mut color)) = query.get_single_mut() else {
        return;
    };

    match state.get() {
        AppState::Setup => {
            let n = deform_state.source_handles.len();
            **text = format!(
                "Mode: SETUP\n\
                 Handles: {}\n\
                 [Click] Add handle\n\
                 [Space] Start deforming\n\
                 [R] Reset",
                n
            );
            color.0 = Color::srgb(0.0, 1.0, 0.0);
        }
        AppState::Deforming => {
            **text = format!(
                "Mode: DEFORM\n\
                 max D = {:.2}\n\
                 K = {:.1}\n\
                 λ = {:.1e}\n\
                 Active: {} | Stable: {}\n\
                 Steps: {}\n\
                 [Drag] Move handles\n\
                 [K/Shift+K] K ±0.5\n\
                 [L/Shift+L] λ ×/÷10\n\
                 [Space] Back to setup\n\
                 [R] Reset",
                deform_info.max_distortion,
                deform_info.k_bound,
                deform_info.lambda_reg,
                deform_info.active_set_size,
                deform_info.stable_set_size,
                deform_info.step_count,
            );
            color.0 = Color::srgb(1.0, 0.5, 0.5);
        }
    }
}

/// System: draw handle gizmos.
pub fn draw_handles(
    deform_state: Res<DeformationState>,
    state: Res<State<AppState>>,
    image_info: Option<Res<ImageInfo>>,
    mut gizmos: Gizmos,
) {
    let Some(image_info) = image_info else { return };
    let img_w = image_info.width;
    let img_h = image_info.height;

    let (source_color, target_color) = match state.get() {
        AppState::Setup => (Color::srgb(1.0, 1.0, 0.0), Color::srgb(1.0, 1.0, 0.0)),
        AppState::Deforming => (
            Color::srgba(0.3, 0.3, 1.0, 0.4),
            Color::srgb(0.0, 1.0, 1.0),
        ),
    };

    // Draw source handles (small, dimmer in deform mode)
    for (_i, src) in deform_state.source_handles.iter().enumerate() {
        let wx = src.x as f32 - img_w / 2.0;
        let wy = img_h / 2.0 - src.y as f32;
        let pos = Vec2::new(wx, wy);

        gizmos.circle_2d(pos, 6.0, source_color);
    }

    // Draw target handles (only in deform mode)
    if *state.get() == AppState::Deforming {
        for (i, tgt) in deform_state.target_handles.iter().enumerate() {
            let wx = tgt.x as f32 - img_w / 2.0;
            let wy = img_h / 2.0 - tgt.y as f32;
            let pos = Vec2::new(wx, wy);

            let color = if deform_state.dragging_index == Some(i) {
                Color::srgb(1.0, 0.0, 0.0)
            } else {
                target_color
            };

            gizmos.circle_2d(pos, 8.0, color);

            // Draw line from source to target
            let src = &deform_state.source_handles[i];
            let src_wx = src.x as f32 - img_w / 2.0;
            let src_wy = img_h / 2.0 - src.y as f32;
            if (src_wx - wx).abs() > 1.0 || (src_wy - wy).abs() > 1.0 {
                gizmos.line_2d(
                    Vec2::new(src_wx, src_wy),
                    pos,
                    Color::srgba(1.0, 1.0, 1.0, 0.3),
                );
            }
        }
    }
}
