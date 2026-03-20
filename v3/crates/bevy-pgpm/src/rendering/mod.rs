//! Rendering module: mesh, material, shader, and deformation paths.

pub mod mesh;
pub mod material;
pub mod gpu_deform;

pub use mesh::create_contour_mesh;
pub use material::{DeformMaterial, DeformUniform, RBFCenter, RBFCoeff, MAX_RBF_COUNT};
pub use gpu_deform::update_deform_material;
