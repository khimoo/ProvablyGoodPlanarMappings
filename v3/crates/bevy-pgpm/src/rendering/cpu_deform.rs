//! CPU deformation path: evaluate f(x) per vertex on CPU and update mesh.
//!
//! Used when the basis is shape-aware Gaussian (geodesic distance) and
//! the shader cannot evaluate basis values.

use bevy::prelude::*;
use bevy::sprite::MeshMaterial2d;

use crate::domain::coords::ImageCoords;
use crate::rendering::material::{DeformMaterial, DeformUniform};
use crate::state::{AlgorithmState, DeformedImage, ImageInfo};

/// Resource storing original pixel-coordinate positions of mesh vertices.
/// Needed for CPU displacement: we evaluate f(original_pixel_pos) each frame.
#[derive(Resource)]
pub struct OriginalVertexPositions {
    /// Pixel-space positions (x, y) for each vertex, matching mesh vertex order.
    pub positions: Vec<[f32; 2]>,
}

/// Run condition: true when the active basis cannot be evaluated on GPU.
pub fn needs_cpu_deform(params: Option<Res<crate::state::AlgoParams>>) -> bool {
    params.map_or(false, |p| !p.basis_type.supports_gpu_eval())
}

/// System: compute deformed positions on CPU and update mesh vertices.
///
/// The shader uniform is set to identity (n_rbf=0, affine=identity)
/// so the vertex shader is effectively a pass-through.
///
/// Run conditions: `DeformingSet` + `needs_cpu_deform`
pub fn cpu_update_mesh_positions(
    algo_state: Res<AlgorithmState>,
    image_info: Option<Res<ImageInfo>>,
    original_verts: Option<Res<OriginalVertexPositions>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<DeformMaterial>>,
    query: Query<(&Mesh2d, &MeshMaterial2d<DeformMaterial>), With<DeformedImage>>,
) {
    let Some(image_info) = image_info else { return };
    let Some(ref algo) = algo_state.algorithm else { return };
    let Some(original_verts) = original_verts else { return };

    let Ok((mesh_handle, material_handle)) = query.get_single() else { return };
    let Some(mesh) = meshes.get_mut(&mesh_handle.0) else { return };

    let coords = ImageCoords::new(image_info.width, image_info.height);

    // Evaluate f(x) at each original vertex position
    let positions: Vec<[f32; 3]> = original_verts.positions.iter().map(|&[px, py]| {
        let pixel_pos = nalgebra::Vector2::new(px as f64, py as f64);
        let deformed = algo.evaluate(pixel_pos);
        let w = coords.pixel_to_world(deformed.x as f32, deformed.y as f32);
        [w.x, w.y, 0.0]
    }).collect();

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);

    // Set shader to identity pass-through
    if let Some(material) = materials.get_mut(&material_handle.0) {
        material.params = DeformUniform::identity(image_info.width, image_info.height);
    }
}
