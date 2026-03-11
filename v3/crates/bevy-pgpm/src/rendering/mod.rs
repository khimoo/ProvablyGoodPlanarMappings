//! Rendering module: mesh, material, shader, and deformation paths.

pub mod mesh;
pub mod material;
pub mod gpu_deform;
pub mod cpu_deform;

pub use mesh::create_contour_mesh;
pub use material::{DeformMaterial, DeformUniform, RBFCenter, RBFCoeff, MAX_RBF_COUNT};
pub use gpu_deform::update_deform_material;
pub use cpu_deform::{cpu_update_mesh_positions, is_shape_aware_basis, OriginalVertexPositions};
