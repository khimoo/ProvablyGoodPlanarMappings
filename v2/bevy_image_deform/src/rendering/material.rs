use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderRef, ShaderType};
use bevy::sprite::Material2d;

// Maximum number of RBF basis functions
pub const MAX_RBF_COUNT: usize = 256;

#[derive(ShaderType, Debug, Clone, Copy)]
pub struct RBFCenter {
    pub pos: Vec2,
    pub _padding: Vec2,  // Alignment padding
}

#[derive(ShaderType, Debug, Clone, Copy)]
pub struct RBFCoeff {
    pub x: f32,
    pub y: f32,
    pub _padding: Vec2,  // Alignment padding to 16 bytes
}

#[derive(ShaderType, Debug, Clone)]
pub struct DeformUniform {
    pub image_width: f32,
    pub image_height: f32,
    pub s_param: f32,
    pub n_rbf: u32,
    pub centers: [RBFCenter; MAX_RBF_COUNT],
    pub coeffs: [RBFCoeff; MAX_RBF_COUNT],
}

impl Default for DeformUniform {
    fn default() -> Self {
        Self {
            image_width: 512.0,
            image_height: 512.0,
            s_param: 1.0,
            n_rbf: 0,
            centers: [RBFCenter {
                pos: Vec2::ZERO,
                _padding: Vec2::ZERO,
            }; MAX_RBF_COUNT],
            coeffs: [RBFCoeff {
                x: 0.0,
                y: 0.0,
                _padding: Vec2::ZERO,
            }; MAX_RBF_COUNT],
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct DeformMaterial {
    #[texture(0)]
    #[sampler(1)]
    pub source_texture: Handle<Image>,

    #[uniform(2)]
    pub params: DeformUniform,
}

impl Material2d for DeformMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/deform.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/deform.wgsl".into()
    }
}
