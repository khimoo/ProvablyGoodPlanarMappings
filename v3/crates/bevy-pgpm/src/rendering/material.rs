//! Custom material for deformed image rendering.
//!
//! The vertex shader evaluates f(x) = Σ c_i φ_i(x) per vertex,
//! producing the forward-mapped deformation. The fragment shader
//! simply samples the source texture at the original UV coordinates.

use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;
use bevy::sprite_render::{AlphaMode2d, Material2d};

/// Maximum number of RBF centers supported in the uniform buffer.
/// The shader uses a fixed-size array for GPU compatibility.
pub const MAX_RBF_COUNT: usize = 256;

#[derive(ShaderType, Debug, Clone, Copy)]
pub struct RBFCenter {
    pub pos: Vec2,
    pub _padding: Vec2,
}

#[derive(ShaderType, Debug, Clone, Copy)]
pub struct RBFCoeff {
    pub x: f32,
    pub y: f32,
    pub _padding: Vec2,
}

/// Uniform buffer sent to the deformation shader.
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

impl DeformUniform {
    /// Create an identity mapping uniform: f(x) = x (no deformation).
    /// coeffs[0] = const(0,0), coeffs[1] = x(1,0), coeffs[2] = y(0,1)
    pub fn identity(width: f32, height: f32) -> Self {
        let mut params = Self::default();
        params.image_width = width;
        params.image_height = height;
        params.n_rbf = 0;
        params.coeffs[0] = RBFCoeff { x: 0.0, y: 0.0, _padding: Vec2::ZERO };
        params.coeffs[1] = RBFCoeff { x: 1.0, y: 0.0, _padding: Vec2::ZERO };
        params.coeffs[2] = RBFCoeff { x: 0.0, y: 1.0, _padding: Vec2::ZERO };
        params
    }
}

/// The Bevy material that holds texture + deformation parameters.
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

    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }
}
