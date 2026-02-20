use bevy::prelude::*;

use crate::state::{AppMode, ControlPoints, ModeText};

pub fn update_ui_text(
    state: Res<State<AppMode>>,
    mut query: Query<(&mut Text, &mut TextColor), With<ModeText>>,
) {
    if let Ok((mut text, mut color)) = query.get_single_mut() {
        match state.get() {
            AppMode::Setup => {
                **text = "Mode: SETUP\n[Click] Add Point\n[Enter] Start Deform\n[R] Reset".to_string();
                color.0 = Color::srgb(0.0, 1.0, 0.0);
            },
            AppMode::Finalizing => {
                **text = "Mode: FINALIZING...\nPlease wait...".to_string();
                color.0 = Color::srgb(1.0, 1.0, 0.0);
            },
            AppMode::Deform => {
                **text = "Mode: DEFORM\n[Drag] Move Points\n[R] Reset".to_string();
                color.0 = Color::srgb(1.0, 0.5, 0.5);
            },
        }
    }
}

pub fn draw_control_points(
    control_points: Res<ControlPoints>,
    mut gizmos: Gizmos,
    state: Res<State<AppMode>>,
) {
    let point_color = match state.get() {
        AppMode::Setup => Color::srgb(1.0, 1.0, 0.0),
        AppMode::Finalizing => Color::srgb(0.5, 0.5, 0.5),
        AppMode::Deform => Color::srgb(0.0, 1.0, 1.0),
    };

    for (i, (_, pos)) in control_points.points.iter().enumerate() {
        let color = if Some(i) == control_points.dragging_index {
            Color::srgb(1.0, 0.0, 0.0)
        } else {
            point_color
        };
        gizmos.circle_2d(*pos, 8.0, color);
    }
}
