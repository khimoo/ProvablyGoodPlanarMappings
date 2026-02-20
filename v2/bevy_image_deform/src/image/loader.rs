use bevy::prelude::*;

#[derive(Resource)]
pub struct ImageData {
    pub width: f32,
    pub height: f32,
    pub handle: Handle<Image>,
    pub contour: Vec<(f32, f32)>,
}
