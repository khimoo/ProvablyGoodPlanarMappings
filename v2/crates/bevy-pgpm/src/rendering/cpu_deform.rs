//! CPU deformation path: evaluate mapping at mesh vertices via pgpm-core.
//!
//! Works correctly with any basis function type (Euclidean Gaussian,
//! Shape-Aware Gaussian, etc.) because all evaluation goes through
//! pgpm-core's `MappingBridge::evaluate_mapping_at()`.

use bevy::prelude::*;

use crate::domain::coords::ImageCoords;
use crate::state::{AlgorithmState, DeformedImage, ImageInfo, OriginalVertexPositions};

/// System: update mesh vertex positions based on the current mapping (Eq. 3).
///
/// Run condition: `DeformingSet` (only runs while deforming).
///
/// After each algorithm step, evaluates f(x) at the original pixel-space
/// vertex positions and writes the deformed world-space positions into
/// the mesh's `POSITION` attribute.
pub fn update_cpu_deform(
    algo_state: Res<AlgorithmState>,
    image_info: Option<Res<ImageInfo>>,
    original_positions: Option<Res<OriginalVertexPositions>>,
    mut meshes: ResMut<Assets<Mesh>>,
    query: Query<&Mesh2d, With<DeformedImage>>,
) {
    let Some(image_info) = image_info else { return };
    let Some(ref algo) = algo_state.algorithm else { return };
    let Some(ref original_positions) = original_positions else { return };
    let Ok(mesh_handle) = query.single() else { return };
    let Some(mesh) = meshes.get_mut(&mesh_handle.0) else { return };

    // Evaluate forward mapping f(x) at all original pixel positions (Eq. 3)
    let deformed_pixels = algo.evaluate_mapping_at(&original_positions.pixel_positions);

    // Convert deformed pixel coords -> world coords and update mesh
    let coords = ImageCoords::new(image_info.width, image_info.height);
    let new_positions: Vec<[f32; 3]> = deformed_pixels
        .iter()
        .map(|p| {
            let world = coords.pixel_to_world(p.x as f32, p.y as f32);
            [world.x, world.y, 0.0]
        })
        .collect();

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, new_positions);
}
