//! Rendering module: mesh, material, and shader for deformed image display.

pub mod mesh;
pub mod material;

pub use mesh::create_contour_mesh;
pub use material::{DeformMaterial, DeformUniform, RBFCenter, RBFCoeff, MAX_RBF_COUNT};
