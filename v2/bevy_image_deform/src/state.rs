use bevy::prelude::*;

#[derive(States, Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
pub enum AppMode {
    #[default]
    Setup,
    Finalizing,
    Deform,
}

#[derive(Resource, Default)]
pub struct ControlPoints {
    pub points: Vec<(usize, Vec2)>,
    pub dragging_index: Option<usize>,
}

#[derive(Resource, Default)]
pub struct MappingParameters {
    pub coefficients: Vec<Vec<f32>>,
    pub centers: Vec<Vec<f32>>,
    pub s_param: f32,
    pub n_rbf: usize,
    pub image_width: f32,
    pub image_height: f32,
    pub inverse_grid: Vec<Vec<Vec<f32>>>,
    pub grid_width: usize,
    pub grid_height: usize,
    pub is_valid: bool,
}

#[derive(Component)]
pub struct DeformedImage;

#[derive(Component)]
pub struct ModeText;

#[derive(Component)]
pub struct MainCamera;
