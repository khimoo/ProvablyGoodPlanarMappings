use bevy::prelude::*;
use geo::{Contains, Polygon, Coord};

use crate::state::{AppMode, ControlPoints};
use crate::image::ImageData;
use crate::python::{PyCommand, PythonChannels};

pub fn handle_input(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    channels: Res<PythonChannels>,
    mut control_points: ResMut<ControlPoints>,
    state: Res<State<AppMode>>,
    mut next_state: ResMut<NextState<AppMode>>,
    image_data: Res<ImageData>,
) {
    let Ok(window) = windows.get_single() else { return };
    let Ok((camera, cam_transform)) = camera_q.get_single() else { return };

    if keys.just_pressed(KeyCode::KeyR) {
        control_points.points.clear();
        control_points.dragging_index = None;
        let _ = channels.tx_command.try_send(PyCommand::Reset);
        next_state.set(AppMode::Setup);
        return;
    }

    if *state.get() == AppMode::Setup && keys.just_pressed(KeyCode::Enter) {
        if control_points.points.is_empty() {
            println!("No control points added!");
            return;
        }
        println!("Starting finalization...");
        let _ = channels.tx_command.try_send(PyCommand::FinalizeSetup);
        next_state.set(AppMode::Finalizing);
        return;
    }

    if *state.get() == AppMode::Finalizing {
        return;
    }

    let Some(cursor_pos) = window.cursor_position() else { return };
    let Ok(ray) = camera.viewport_to_world(cam_transform, cursor_pos) else { return };
    let world_pos = ray.origin.truncate();

    let img_w = image_data.width;
    let img_h = image_data.height;

    let py_x = world_pos.x + img_w / 2.0;
    let py_y = img_h / 2.0 - world_pos.y;

    if py_x < 0.0 || py_x > img_w || py_y < 0.0 || py_y > img_h {
        if control_points.dragging_index.is_some() && buttons.just_released(MouseButton::Left) {
            control_points.dragging_index = None;
            let _ = channels.tx_command.try_send(PyCommand::EndDrag);
        }
        return;
    }

    match state.get() {
        AppMode::Setup => {
            if buttons.just_pressed(MouseButton::Left) {
                if !image_data.contour.is_empty() {
                    let coords: Vec<Coord<f32>> = image_data.contour
                        .iter()
                        .map(|&(x, y)| Coord { x, y })
                        .collect();

                    let polygon = Polygon::new(
                        geo::LineString::from(coords),
                        vec![]
                    );

                    let point = Coord { x: py_x, y: py_y };
                    let is_inside = polygon.contains(&point);

                    println!(
                        "Click at ({:.1}, {:.1}) - Inside contour: {} (contour has {} points)",
                        py_x, py_y, is_inside, image_data.contour.len()
                    );

                    if !is_inside {
                        println!("Cannot add control point outside contour");
                        return;
                    }
                }

                let threshold = 20.0;
                let too_close = control_points
                    .points
                    .iter()
                    .any(|(_, pos)| pos.distance(world_pos) < threshold);

                if !too_close {
                    let control_idx = control_points.points.len();
                    control_points.points.push((control_idx, world_pos));

                    let _ = channels.tx_command.try_send(PyCommand::AddControlPoint {
                        index: control_idx,
                        x: py_x,
                        y: py_y,
                    });
                    println!(
                        "Added control point {} at ({:.1}, {:.1})",
                        control_idx, py_x, py_y
                    );
                }
            }
        }
        AppMode::Finalizing => {}
        AppMode::Deform => {
            if buttons.just_pressed(MouseButton::Left) {
                if let Some(idx) = control_points
                    .points
                    .iter()
                    .position(|(_, pos)| pos.distance(world_pos) < 20.0)
                {
                    control_points.dragging_index = Some(idx);
                    let _ = channels.tx_command.try_send(PyCommand::StartDrag);
                }
            }

            if buttons.pressed(MouseButton::Left) {
                if let Some(idx) = control_points.dragging_index {
                    if idx < control_points.points.len() {
                        control_points.points[idx].1 = world_pos;

                        let _ = channels.tx_command.try_send(PyCommand::UpdatePoint {
                            control_index: idx,
                            x: py_x,
                            y: py_y,
                        });
                    }
                }
            }

            if buttons.just_released(MouseButton::Left) {
                if control_points.dragging_index.is_some() {
                    let _ = channels.tx_command.try_send(PyCommand::EndDrag);
                    control_points.dragging_index = None;
                }
            }
        }
    }
}
