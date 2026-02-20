use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderRef};
use bevy::sprite::Material2d;

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct DeformMaterial {
    #[texture(0)]
    #[sampler(1)]
    pub source_texture: Handle<Image>,

    #[texture(2)]
    #[sampler(3)]
    pub inverse_grid_texture: Handle<Image>,

    #[uniform(4)]
    pub grid_size: Vec2,
}

impl Material2d for DeformMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/deform.wgsl".into()
    }
}
