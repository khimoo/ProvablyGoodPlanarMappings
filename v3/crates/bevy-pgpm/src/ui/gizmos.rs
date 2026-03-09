//! Handle gizmo rendering: circles and lines for source/target handles.

use bevy::prelude::*;

use crate::domain::coords::ImageCoords;
use crate::state::{AlgorithmState, AppState, DragState, ImageInfo};

/// System: draw handle gizmos on the canvas.
pub fn draw_handles(
    algo_state: Res<AlgorithmState>,
    drag_state: Res<DragState>,
    state: Res<State<AppState>>,
    image_info: Option<Res<ImageInfo>>,
    mut gizmos: Gizmos,
) {
    let Some(image_info) = image_info else { return };
    let coords = ImageCoords::new(image_info.width, image_info.height);

    let (source_color, target_color) = match state.get() {
        AppState::Setup => (Color::srgb(1.0, 1.0, 0.0), Color::srgb(1.0, 1.0, 0.0)),
        AppState::Deforming => (
            Color::srgba(0.3, 0.3, 1.0, 0.4),
            Color::srgb(0.0, 1.0, 1.0),
        ),
    };

    for src in algo_state.source_handles.iter() {
        let w = coords.pixel_to_world(src.x as f32, src.y as f32);
        gizmos.circle_2d(w, 6.0, source_color);
    }

    if *state.get() == AppState::Deforming {
        for (i, tgt) in algo_state.target_handles.iter().enumerate() {
            let pos = coords.pixel_to_world(tgt.x as f32, tgt.y as f32);
            let color = if drag_state.handle_index == Some(i) {
                Color::srgb(1.0, 0.0, 0.0)
            } else {
                target_color
            };
            gizmos.circle_2d(pos, 8.0, color);

            let src = &algo_state.source_handles[i];
            let src_w = coords.pixel_to_world(src.x as f32, src.y as f32);
            if (src_w.x - pos.x).abs() > 1.0 || (src_w.y - pos.y).abs() > 1.0 {
                gizmos.line_2d(src_w, pos, Color::srgba(1.0, 1.0, 1.0, 0.3));
            }
        }
    }
}
